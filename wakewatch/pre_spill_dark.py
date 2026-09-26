"""
Pre-Spill AIS Gap & Trajectory Analysis -- "Scenario 2" dark-vessel evidence.

WHAT THIS DOES
---------------
Scenario 1 (`dark_vessel.py`) catches a vessel the satellite *saw* but AIS
didn't report. This module catches the opposite case: a vessel the satellite
never saw at all, because it may have already gone dark before the SAR
overpass. It searches AIS history *before* the drift-hindcasted spill origin
and time for vessels whose transmissions stopped (a "gap"), then asks three
separable questions about each gap:

  1. Is the gap actually near the estimated spill time? (`filter_pre_spill_gaps`)
  2. Could the vessel physically have reached the estimated origin during the
     gap, given its last known speed? (`check_physical_reachability`)
  3. Is its last-known heading/speed, dead-reckoned forward, and (when
     available) its next AIS position after the gap, geographically and
     temporally consistent with the estimated origin? (`estimate_gap_trajectory`,
     `evaluate_time_consistency`)

WHAT THIS DELIBERATELY DOES NOT DO
------------------------------------
- No new ML model. AIS gap detection is temporal arithmetic on existing AIS
  data; reachability and gap-trajectory are classical kinematics (speed x
  time, dead reckoning); the existing LSTM is used only as supplementary
  next-ping evidence where enough history exists, exactly as in
  `dark_vessel.py` -- never stretched across the full multi-hour gap.
- No claim of guilt. Every candidate here is a "Potential Pre-Spill AIS Gap
  Candidate" -- an investigative lead, never a confirmed dark vessel,
  responsible party, or evidence of intentional AIS shutdown.
- No change to `fusion.py` or the existing attribution score. This module's
  evidence score is entirely separate, exactly as `dark_vessel.py`'s is.
- No fabricated gaps. If the real AIS feed for a scenario has no significant
  gaps, this module reports zero candidates -- it does not manufacture one to
  make the feature look successful, and it never singles out a named vessel
  (e.g. Wakashio) by anything other than the same arithmetic applied to every
  vessel in the feed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import drift as drift_mod
from .inference.trajectory import haversine_km

KNOTS_TO_KM_H = 1.852
KNOTS_TO_MS = 0.514444

_LABEL_NUMERIC = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.25, "INCONSISTENT": 0.0, "UNKNOWN": 0.4}
_LABEL_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "INCONSISTENT": 0, "UNKNOWN": -1}


@dataclass
class PreSpillConfig:
    pre_spill_window_hours: float = 12.0     # how far before the estimated release time to search for gaps starting
    min_dark_gap_minutes: float = 30.0       # below this, treat as normal AIS reporting cadence, not a "gap"
    after_release_slack_hours: float = 1.0   # a gap starting slightly after the point-estimate release time is
                                              # still considered, to absorb release-time uncertainty
    time_consistency_slack_hours: float = 2.0
    max_reachability_speed_knots: float = 25.0   # generous cap used when last-known speed is near zero (e.g.
                                                  # anchored/drifting) so a stopped-at-last-ping vessel isn't
                                                  # automatically ruled unreachable
    min_speed_for_own_speed_knots: float = 1.0   # below this, last known speed is treated as unreliable for
                                                  # reachability and the generous cap above is used instead
    trajectory_high_km: float = 5.0
    trajectory_medium_km: float = 20.0


# --------------------------------------------------------------------------
# Step 1-2: chronological AIS gap detection
# --------------------------------------------------------------------------
@dataclass
class AisGap:
    mmsi: int
    vessel_name: Optional[str]
    gap_start: datetime                       # last AIS ping before the gap
    gap_end: Optional[datetime]                # next AIS ping after the gap, if any resumption was observed
    gap_duration_minutes: float
    last_lat: float
    last_lon: float
    last_speed_knots: float
    last_course_deg: float
    next_lat: Optional[float] = None
    next_lon: Optional[float] = None
    resumed: bool = True                      # False when the vessel never reappears in the feed at all

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi, "vessel_name": self.vessel_name,
            "gap_start": self.gap_start.isoformat(),
            "gap_end": self.gap_end.isoformat() if self.gap_end else None,
            "gap_duration_minutes": round(self.gap_duration_minutes, 1),
            "last_lat": self.last_lat, "last_lon": self.last_lon,
            "last_speed_knots": self.last_speed_knots, "last_course_deg": self.last_course_deg,
            "next_lat": self.next_lat, "next_lon": self.next_lon, "resumed": self.resumed,
        }


def calculate_gap_duration(t1: datetime, t2: datetime) -> float:
    """Minutes between two AIS timestamps. A thin, named wrapper so gap
    duration is computed the same way everywhere in this module."""
    return (t2 - t1).total_seconds() / 60.0


def find_ais_gaps(ais_df: pd.DataFrame, config: PreSpillConfig) -> List[AisGap]:
    """
    Chronological, per-vessel scan for AIS reporting gaps of at least
    `config.min_dark_gap_minutes`. Ordinary reporting cadence (AIS pings can
    legitimately be minutes apart depending on vessel speed/class) is not
    treated as a gap -- only intervals at or above the configured threshold.

    Also detects an open-ended gap when a vessel's last ping in the feed is
    itself more than the threshold before the feed's own last timestamp --
    i.e. every *other* vessel kept transmitting after this one went quiet,
    so it plausibly went dark and never came back within the observed
    window. `gap_end` is None for these; `resumed` is False.
    """
    if ais_df.empty:
        return []
    ts_all = pd.to_datetime(ais_df["timestamp"], utc=True)
    feed_end = ts_all.max().to_pydatetime()

    gaps: List[AisGap] = []
    for mmsi, group in ais_df.groupby("mmsi"):
        track = group.copy()
        track["_ts"] = pd.to_datetime(track["timestamp"], utc=True)
        track = track.sort_values("_ts").reset_index(drop=True)
        name = None
        if "vessel_name" in track.columns and len(track):
            v = track["vessel_name"].iloc[0]
            name = str(v) if pd.notna(v) else None

        for i in range(len(track) - 1):
            t1, t2 = track.loc[i, "_ts"].to_pydatetime(), track.loc[i + 1, "_ts"].to_pydatetime()
            dur = calculate_gap_duration(t1, t2)
            if dur >= config.min_dark_gap_minutes:
                gaps.append(AisGap(
                    mmsi=int(mmsi), vessel_name=name, gap_start=t1, gap_end=t2,
                    gap_duration_minutes=dur,
                    last_lat=float(track.loc[i, "latitude"]), last_lon=float(track.loc[i, "longitude"]),
                    last_speed_knots=float(track.loc[i].get("speed", 0.0) or 0.0),
                    last_course_deg=float(track.loc[i].get("course", 0.0) or 0.0),
                    next_lat=float(track.loc[i + 1, "latitude"]), next_lon=float(track.loc[i + 1, "longitude"]),
                    resumed=True,
                ))

        if len(track):
            last_t = track["_ts"].iloc[-1].to_pydatetime()
            trailing = calculate_gap_duration(last_t, feed_end)
            if trailing >= config.min_dark_gap_minutes:
                last = track.iloc[-1]
                gaps.append(AisGap(
                    mmsi=int(mmsi), vessel_name=name, gap_start=last_t, gap_end=None,
                    gap_duration_minutes=trailing,
                    last_lat=float(last["latitude"]), last_lon=float(last["longitude"]),
                    last_speed_knots=float(last.get("speed", 0.0) or 0.0),
                    last_course_deg=float(last.get("course", 0.0) or 0.0),
                    resumed=False,
                ))
    return gaps


# --------------------------------------------------------------------------
# Step 3: keep only gaps that are actually near the estimated spill
# --------------------------------------------------------------------------
def filter_pre_spill_gaps(gaps: List[AisGap], release_time: datetime, config: PreSpillConfig) -> List[AisGap]:
    """
    A gap is pre-spill-relevant when the vessel stopped transmitting inside
    the search window before the estimated release time, and did not resume
    well before that time (a small `after_release_slack_hours` allowance
    absorbs the fact that the release time itself is an estimate, not an
    exact instant).
    """
    if release_time.tzinfo is None:
        release_time = release_time.replace(tzinfo=timezone.utc)
    window_start = release_time - timedelta(hours=config.pre_spill_window_hours)
    latest_relevant_start = release_time + timedelta(hours=config.after_release_slack_hours)

    relevant = []
    for g in gaps:
        gs = g.gap_start if g.gap_start.tzinfo else g.gap_start.replace(tzinfo=timezone.utc)
        if window_start <= gs <= latest_relevant_start:
            relevant.append(g)
    return relevant


# --------------------------------------------------------------------------
# Step 4-5: distance to origin, physical reachability
# --------------------------------------------------------------------------
def calculate_origin_distance(lat: float, lon: float, origin_lat: float, origin_lon: float) -> float:
    """Great-circle (haversine) distance in km -- never a raw lat/lon subtraction."""
    return haversine_km(lat, lon, origin_lat, origin_lon)


@dataclass
class ReachabilityCheck:
    plausible: bool
    max_reachable_km: float
    distance_to_origin_km: float
    effective_speed_knots: float
    note: str


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _effective_release_time(release_time: datetime, release_time_uncertainty_hours: Optional[float]) -> datetime:
    """
    The most permissive plausible release time within the release-time
    uncertainty window, when one is known -- release_time + uncertainty.
    Widening in this direction (later, giving the vessel more time) means a
    real uncertainty gives the candidate the benefit of the doubt rather
    than being penalised for imprecision that isn't its fault, while a
    physically-inconsistent verdict still has to hold even at that most
    generous bound.

    Returns `release_time` unchanged when no uncertainty value is supplied
    -- this function never invents an uncertainty that wasn't given to it.
    """
    rt = _aware(release_time)
    if not release_time_uncertainty_hours:
        return rt
    return rt + timedelta(hours=release_time_uncertainty_hours)


def _travel_time_to_release_hours(
    gap: "AisGap", release_time: datetime, release_time_uncertainty_hours: Optional[float] = None,
) -> float:
    """
    Hours available to the vessel between its last known AIS ping and the
    estimated spill release time (or the latest plausible release time
    within a known uncertainty window) -- NOT the full AIS gap duration.

    A vessel whose AIS resumes long after the spill (e.g. last seen 09:40,
    next seen 11:45, spill estimated at 10:30) only had until 10:30 to reach
    the origin; what it did between 10:30 and 11:45 is irrelevant to whether
    it *could have caused* the spill. Using the full 2h05m gap here would
    overstate how far the vessel plausibly travelled before the release.

    Clamped at zero: a gap that starts at or after the (uncertainty-widened)
    release time gives the vessel no pre-release travel time at all.
    """
    effective_release = _effective_release_time(release_time, release_time_uncertainty_hours)
    hours = (effective_release - _aware(gap.gap_start)).total_seconds() / 3600.0
    return max(0.0, hours)


def check_physical_reachability(
    last_speed_knots: float, gap_hours: float, distance_to_origin_km: float, config: PreSpillConfig,
) -> ReachabilityCheck:
    """
    Could the vessel plausibly have covered the distance to the estimated
    origin in the time available, at something like its last known speed?

    IMPORTANT: `gap_hours` here must be the time available to reach the
    origin BY the estimated release time -- see `_travel_time_to_release_hours`
    -- not the full AIS gap duration (last AIS to next AIS). Time after the
    vessel could already have caused the spill is not relevant to whether it
    could physically have gotten there. `score_pre_spill_dark_candidate`
    (the only caller in this module) always passes the correct, release-time-
    bound value; this function itself stays generic ("given this many hours,
    is this distance reachable") so its own unit tests can exercise it
    directly with an explicit duration.

    A last-known speed near zero (anchored, drifting, or just a noisy AIS
    reading) is not treated as "this vessel cannot move" -- a vessel does
    not necessarily keep transmitting its speed right up to the moment it
    goes dark, and a stationary-looking last ping should not automatically
    clear a vessel that then travelled normally before the release. In that
    case a generous, explicitly configured maximum speed is used instead,
    and the note says so.
    """
    if last_speed_knots >= config.min_speed_for_own_speed_knots:
        effective_speed = last_speed_knots
        basis = "last known speed"
    else:
        effective_speed = config.max_reachability_speed_knots
        basis = f"generous cap ({config.max_reachability_speed_knots:.0f} kn) -- last known speed was near zero"

    max_reachable_km = effective_speed * KNOTS_TO_KM_H * gap_hours
    plausible = distance_to_origin_km <= max_reachable_km
    note = (f"At {effective_speed:.1f} kn ({basis}) over the {gap_hours:.2f} h available before the "
           f"estimated release time, maximum plausible travel is ~{max_reachable_km:.1f} km; "
           f"the estimated origin is {distance_to_origin_km:.1f} km away.")
    return ReachabilityCheck(plausible=plausible, max_reachable_km=round(max_reachable_km, 2),
                             distance_to_origin_km=round(distance_to_origin_km, 2),
                             effective_speed_knots=round(effective_speed, 1), note=note)


# --------------------------------------------------------------------------
# Step 6: trajectory consistency during the gap (classical dead reckoning),
# plus a cross-check against the next AIS position when one exists.
# --------------------------------------------------------------------------
@dataclass
class GapTrajectoryEstimate:
    label: str                            # HIGH | MEDIUM | LOW
    dead_reckoning_lat: float
    dead_reckoning_lon: float
    dead_reckoning_distance_km: float
    note: str


def estimate_gap_trajectory(
    gap: AisGap, origin_lat: float, origin_lon: float, config: PreSpillConfig,
    origin_uncertainty_km: float = 0.0,
    release_time: Optional[datetime] = None, release_time_uncertainty_hours: Optional[float] = None,
) -> GapTrajectoryEstimate:
    """
    Classical dead reckoning from the last known position/speed/course --
    explicitly NOT an LSTM rollout (the model was trained for a single ~60s
    step; presenting a multi-hour extrapolation as a model prediction would
    overstate what it actually supports). The estimated origin's own
    positional uncertainty (from the drift hindcast) widens the consistency
    bands, so a wide-uncertainty origin doesn't get penalised for
    imprecision that isn't the candidate's fault.

    When `release_time` is given, dead-reckons only for the time available
    before the estimated release (see `_travel_time_to_release_hours`) --
    the same correction as `check_physical_reachability`: where the vessel
    would be by the time the spill happened, not wherever it ends up by the
    unrelated moment its AIS happened to resume. Without `release_time`
    (e.g. calling this function directly to inspect the full-gap track),
    falls back to dead-reckoning the full gap duration.
    """
    if release_time is not None:
        gap_hours = _travel_time_to_release_hours(gap, release_time, release_time_uncertainty_hours)
        duration_s = gap_hours * 3600.0
        horizon_note = "before the estimated release time"
    else:
        gap_hours = gap.gap_duration_minutes / 60.0
        duration_s = gap.gap_duration_minutes * 60.0
        horizon_note = "over the full AIS gap"

    speed_ms = gap.last_speed_knots * KNOTS_TO_MS
    bearing_rad = np.radians(gap.last_course_deg)
    u_ms, v_ms = speed_ms * np.sin(bearing_rad), speed_ms * np.cos(bearing_rad)
    dr_lat, dr_lon = drift_mod._advance(gap.last_lat, gap.last_lon, u_ms, v_ms, duration_s)
    dist = haversine_km(dr_lat, dr_lon, origin_lat, origin_lon)

    high_band = config.trajectory_high_km + origin_uncertainty_km
    medium_band = config.trajectory_medium_km + origin_uncertainty_km
    if dist <= high_band:
        label = "HIGH"
    elif dist <= medium_band:
        label = "MEDIUM"
    else:
        label = "LOW"

    note = (f"Dead-reckoned from last known speed/course, {gap_hours:.2f} h {horizon_note}; landed {dist:.1f} km "
           f"from the estimated origin (uncertainty-widened bands: \u2264{high_band:.1f} km HIGH, "
           f"\u2264{medium_band:.1f} km MEDIUM).")
    return GapTrajectoryEstimate(label, float(dr_lat), float(dr_lon), round(dist, 2), note)


@dataclass
class NextPositionCheck:
    interpolated_lat: float
    interpolated_lon: float
    distance_to_origin_km: float
    label: str
    note: str


def check_next_position_consistency(
    gap: AisGap, release_time: datetime, origin_lat: float, origin_lon: float, config: PreSpillConfig,
    origin_uncertainty_km: float = 0.0,
) -> Optional[NextPositionCheck]:
    """
    When AIS resumes after the gap, use BOTH endpoints as evidence: linearly
    interpolate (in time) between the last-known and next-known positions at
    the estimated release time, and see how close that lands to the origin.

    This is explicitly a straight-line estimate between two real, observed
    points -- not an assumption that the vessel travelled in a straight line
    in reality, and it is labelled as such. Returns None when there is no
    next position to check against (the gap has not resumed in this feed).
    """
    if gap.gap_end is None or gap.next_lat is None or gap.next_lon is None:
        return None
    total_s = (gap.gap_end - gap.gap_start).total_seconds()
    if total_s <= 0:
        return None
    rt = release_time if release_time.tzinfo else release_time.replace(tzinfo=timezone.utc)
    frac = (rt - gap.gap_start).total_seconds() / total_s
    frac = min(max(frac, 0.0), 1.0)
    lat = gap.last_lat + frac * (gap.next_lat - gap.last_lat)
    lon = gap.last_lon + frac * (gap.next_lon - gap.last_lon)
    dist = haversine_km(lat, lon, origin_lat, origin_lon)

    high_band = config.trajectory_high_km + origin_uncertainty_km
    medium_band = config.trajectory_medium_km + origin_uncertainty_km
    label = "HIGH" if dist <= high_band else ("MEDIUM" if dist <= medium_band else "LOW")
    note = (f"Straight-line interpolation between the last AIS position and the next one after the gap, "
           f"evaluated at the estimated release time -- {dist:.1f} km from the estimated origin. This is an "
           f"estimate, not an assumption that the vessel actually travelled in a straight line.")
    return NextPositionCheck(float(lat), float(lon), round(dist, 2), label, note)


# --------------------------------------------------------------------------
# Step 7: spill-time consistency
# --------------------------------------------------------------------------
@dataclass
class TimeConsistency:
    label: str                # HIGH | MEDIUM | LOW | INCONSISTENT
    note: str


def evaluate_time_consistency(
    gap: AisGap, release_time: datetime, config: PreSpillConfig,
    release_time_uncertainty_hours: Optional[float] = None,
) -> TimeConsistency:
    """
    When `release_time_uncertainty_hours` is given (from the drift hindcast,
    never invented), the "overlap" check is evaluated against the full
    plausible release-time window [release_time - unc, release_time + unc],
    not a single point -- consistent with how the rest of this module treats
    a real, known uncertainty as widening what counts as consistent, never
    as a reason to be stricter.
    """
    rt = _aware(release_time)
    gs = _aware(gap.gap_start)
    ge = _aware(gap.gap_end) if gap.gap_end is not None else None
    unc = timedelta(hours=release_time_uncertainty_hours) if release_time_uncertainty_hours else timedelta(0)
    rt_earliest, rt_latest = rt - unc, rt + unc

    overlapping = gs <= rt_latest and (ge is None or rt_earliest <= ge)
    if overlapping:
        note = "The AIS gap overlaps the estimated release time."
        if unc:
            note += f" (\u00b1{release_time_uncertainty_hours:.1f} h release-time uncertainty applied.)"
        return TimeConsistency("HIGH", note)

    edge_hours = min(abs((rt_earliest - gs).total_seconds()), abs((rt_latest - gs).total_seconds())) / 3600.0
    if ge is not None:
        edge_hours = min(edge_hours,
                         abs((rt_earliest - ge).total_seconds()) / 3600.0,
                         abs((rt_latest - ge).total_seconds()) / 3600.0)

    if edge_hours <= config.time_consistency_slack_hours:
        label = "MEDIUM"
    elif edge_hours <= config.pre_spill_window_hours:
        label = "LOW"
    else:
        label = "INCONSISTENT"
    return TimeConsistency(label, f"Gap does not overlap the estimated release time; "
                                  f"nearest edge is {edge_hours:.1f} h away.")


# --------------------------------------------------------------------------
# Step 8: combine into one explainable evidence score, per candidate
# --------------------------------------------------------------------------
@dataclass
class PreSpillDarkCandidate:
    gap: AisGap
    distance_to_origin_km: float
    reachability: ReachabilityCheck
    gap_trajectory: GapTrajectoryEstimate
    time_consistency: TimeConsistency
    next_position: Optional[NextPositionCheck]
    status: str                          # "potential_pre_spill_gap" | "physically_inconsistent"
    evidence_label: str                  # HIGH | MEDIUM | LOW | INCONSISTENT
    evidence_components: Dict[str, float]
    note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gap": self.gap.as_dict(), "distance_to_origin_km": self.distance_to_origin_km,
            "reachability": self.reachability.__dict__, "gap_trajectory": self.gap_trajectory.__dict__,
            "time_consistency": self.time_consistency.__dict__,
            "next_position": self.next_position.__dict__ if self.next_position else None,
            "status": self.status, "evidence_label": self.evidence_label,
            "evidence_components": self.evidence_components, "note": self.note,
        }


_RANK_LABEL = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 0: "INCONSISTENT"}


def _combine_evidence_labels(labels: List[str]) -> str:
    """
    Combine several HIGH/MEDIUM/LOW labels into one overall label.

    A plain average of numeric scores lets two strong signals mask one weak
    one -- e.g. reachability=True and time_consistency=HIGH alone can average
    above the HIGH threshold even when the dead-reckoned trajectory lands
    hundreds of km from the estimated origin, which is exactly the case this
    feature most needs to get right. Instead, the overall label is capped at
    one tier above the *worst* individual label: strong corroborating
    signals can lift a weak one by at most one step, never paper over it
    entirely.
    """
    ranks = [_LABEL_RANK[l] for l in labels]
    avg_rank = sum(ranks) / len(ranks)
    worst_rank = min(ranks)
    capped = min(round(avg_rank), worst_rank + 1)
    capped = max(0, min(3, capped))
    return _RANK_LABEL[capped]


def score_pre_spill_dark_candidate(
    gap: AisGap, origin_lat: float, origin_lon: float, release_time: datetime, config: PreSpillConfig,
    origin_uncertainty_km: float = 0.0, release_time_uncertainty_hours: Optional[float] = None,
) -> PreSpillDarkCandidate:
    # FIX: physical reachability and gap-trajectory dead reckoning must both
    # use the time available BEFORE the estimated release, not the full AIS
    # gap (last AIS -> next AIS). A vessel that resumed transmitting long
    # after the spill only had until the release time to get there -- what
    # it did afterward is not evidence of what it could reach beforehand.
    travel_time_to_release_hours = _travel_time_to_release_hours(gap, release_time, release_time_uncertainty_hours)
    distance_km = calculate_origin_distance(gap.last_lat, gap.last_lon, origin_lat, origin_lon)
    reach = check_physical_reachability(gap.last_speed_knots, travel_time_to_release_hours, distance_km, config)
    traj = estimate_gap_trajectory(gap, origin_lat, origin_lon, config, origin_uncertainty_km,
                                   release_time=release_time,
                                   release_time_uncertainty_hours=release_time_uncertainty_hours)
    time_c = evaluate_time_consistency(gap, release_time, config, release_time_uncertainty_hours)
    next_pos = check_next_position_consistency(gap, release_time, origin_lat, origin_lon, config,
                                               origin_uncertainty_km)

    components = {
        "reachability": 1.0 if reach.plausible else 0.0,
        "trajectory_consistency": _LABEL_NUMERIC[traj.label],
        "time_consistency": _LABEL_NUMERIC[time_c.label],
    }
    if next_pos is not None:
        components["next_position_consistency"] = _LABEL_NUMERIC[next_pos.label]

    if not reach.plausible:
        status, evidence_label = "physically_inconsistent", "INCONSISTENT"
        note = ("Physically inconsistent \u2014 the estimated origin is farther than this vessel could "
               f"plausibly have travelled by the estimated release time. {reach.note}")
    else:
        status = "potential_pre_spill_gap"
        # The overall label is driven by the geospatial/temporal signals
        # (trajectory, time, and next-position when available) -- reachability
        # is a binary gate already handled above, not an averaged-in score.
        label_inputs = [traj.label, time_c.label] + ([next_pos.label] if next_pos else [])
        evidence_label = _combine_evidence_labels(label_inputs)
        note = ("Potential Pre-Spill AIS Gap Candidate. This is an investigative lead, not proof of "
               "AIS shutdown or responsibility. " + reach.note + " " + traj.note)

    return PreSpillDarkCandidate(
        gap=gap, distance_to_origin_km=round(distance_km, 2), reachability=reach, gap_trajectory=traj,
        time_consistency=time_c, next_position=next_pos, status=status, evidence_label=evidence_label,
        evidence_components=components, note=note,
    )


# --------------------------------------------------------------------------
# Top-level orchestrator
# --------------------------------------------------------------------------
@dataclass
class PreSpillDarkReport:
    candidates: List[PreSpillDarkCandidate]
    ais_coverage_available: bool
    vessels_with_insufficient_history: int
    config: PreSpillConfig
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        return {"candidates": [c.as_dict() for c in self.candidates],
               "ais_coverage_available": self.ais_coverage_available,
               "vessels_with_insufficient_history": self.vessels_with_insufficient_history,
               "summary": self.summary}


def analyze_pre_spill_dark_vessels(
    ais_df: Optional[pd.DataFrame], origin_lat: float, origin_lon: float, release_time: datetime,
    ais_coverage_available: bool, origin_uncertainty_km: float = 0.0,
    release_time_uncertainty_hours: Optional[float] = None,
    config: Optional[PreSpillConfig] = None,
) -> PreSpillDarkReport:
    """
    Search AIS history before the estimated spill for vessels with a
    plausibly relevant AIS gap, and evidence-score each one.

    `ais_coverage_available` must come from the same independently-checked
    coverage decision the rest of the project already uses (e.g.
    `investigation.is_within_demo_aoi`/`_window`) -- when False, this
    function does not attempt any analysis and says plainly that a
    dark-vessel conclusion cannot be made, rather than silently returning an
    empty (and easily misread as "none found") result.

    `release_time_uncertainty_hours`, when supplied by the caller from a
    real value the drift hindcast provides, widens the physical/temporal
    consistency checks to the most permissive bound within that window
    (see `_effective_release_time`). Left as None (never invented here) when
    no such value is available -- which is the case today, since the
    existing hindcast reports positional uncertainty but not an explicit
    release-time uncertainty; this parameter exists so that is used
    correctly the moment one is.
    """
    config = config or PreSpillConfig()
    if release_time.tzinfo is None:
        release_time = release_time.replace(tzinfo=timezone.utc)

    if not ais_coverage_available or ais_df is None or ais_df.empty:
        return PreSpillDarkReport(
            candidates=[], ais_coverage_available=False, vessels_with_insufficient_history=0, config=config,
            summary="AIS coverage unavailable for this region/time. A dark-vessel conclusion cannot be made.",
        )

    n_insufficient = int((ais_df.groupby("mmsi").size() < 2).sum())

    all_gaps = find_ais_gaps(ais_df, config)
    relevant_gaps = filter_pre_spill_gaps(all_gaps, release_time, config)

    candidates = [
        score_pre_spill_dark_candidate(g, origin_lat, origin_lon, release_time, config, origin_uncertainty_km,
                                       release_time_uncertainty_hours)
        for g in relevant_gaps
    ]
    candidates.sort(key=lambda c: _LABEL_RANK.get(c.evidence_label, -1), reverse=True)

    n_high = sum(1 for c in candidates if c.evidence_label == "HIGH")
    n_inconsistent = sum(1 for c in candidates if c.status == "physically_inconsistent")
    if not candidates:
        summary = "No AIS gaps relevant to the estimated spill window were found in the available AIS data."
    else:
        summary = (f"{len(candidates)} potential pre-spill AIS gap candidate(s) "
                  f"({n_high} high-evidence, {n_inconsistent} ruled physically inconsistent).")

    return PreSpillDarkReport(candidates=candidates, ais_coverage_available=True,
                              vessels_with_insufficient_history=n_insufficient, config=config, summary=summary)


# --------------------------------------------------------------------------
# Scenario 1 + Scenario 2 merge -- never double-count the same vessel.
# --------------------------------------------------------------------------
@dataclass
class CombinedDarkVesselCandidate:
    mmsi: Optional[int]
    vessel_name: Optional[str]
    sources: List[str]                          # ["scenario1_sar_ais_mismatch"], ["scenario2_pre_spill_gap"], or both
    scenario1: Optional[Any] = None              # dark_vessel.DarkVesselCandidate
    scenario2: Optional[PreSpillDarkCandidate] = None
    combined_evidence_label: str = "UNKNOWN"
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mmsi": self.mmsi, "vessel_name": self.vessel_name, "sources": self.sources,
            "scenario1": self.scenario1.as_dict() if self.scenario1 else None,
            "scenario2": self.scenario2.as_dict() if self.scenario2 else None,
            "combined_evidence_label": self.combined_evidence_label, "note": self.note,
        }


def merge_dark_vessel_evidence(scenario1_candidates: List[Any], scenario2_candidates: List[PreSpillDarkCandidate]
                               ) -> List[CombinedDarkVesselCandidate]:
    """
    Combine Scenario 1 (SAR-AIS mismatch) and Scenario 2 (pre-spill AIS gap)
    candidates by MMSI, so a vessel flagged by both mechanisms appears once,
    with both signals shown -- not as two separate rows, and not silently
    averaged into one unexplained number.

    A Scenario 1 candidate only carries an MMSI when it found a plausible
    last-known AIS track (`.last_known.mmsi`); one with no last-known lookup
    at all (no plausible AIS track nearby) has nothing to merge against and
    is kept as its own unlinked entry.
    """
    by_mmsi: Dict[int, CombinedDarkVesselCandidate] = {}
    unlinked: List[CombinedDarkVesselCandidate] = []

    for c1 in scenario1_candidates:
        mmsi = c1.last_known.mmsi if getattr(c1, "last_known", None) is not None else None
        if mmsi is None:
            unlinked.append(CombinedDarkVesselCandidate(
                mmsi=None, vessel_name=None, sources=["scenario1_sar_ais_mismatch"], scenario1=c1,
                combined_evidence_label=getattr(c1, "evidence_label", "UNKNOWN"),
                note="SAR-detected target with no matching AIS, and no plausible last-known AIS track nearby.",
            ))
            continue
        by_mmsi[mmsi] = CombinedDarkVesselCandidate(
            mmsi=mmsi, vessel_name=c1.last_known.vessel_name, sources=["scenario1_sar_ais_mismatch"],
            scenario1=c1, combined_evidence_label=getattr(c1, "evidence_label", "UNKNOWN"),
        )

    for c2 in scenario2_candidates:
        mmsi = c2.gap.mmsi
        if mmsi in by_mmsi:
            existing = by_mmsi[mmsi]
            existing.sources.append("scenario2_pre_spill_gap")
            existing.scenario2 = c2
            rank1 = _LABEL_RANK.get(existing.combined_evidence_label, -1)
            rank2 = _LABEL_RANK.get(c2.evidence_label, -1)
            existing.combined_evidence_label = existing.combined_evidence_label if rank1 >= rank2 else c2.evidence_label
            existing.note = (
                "Multi-source dark-vessel evidence: this vessel was flagged both by a direct SAR "
                "detection with no matching AIS (Scenario 1) and by a pre-spill AIS gap consistent "
                "with the estimated origin (Scenario 2). Shown as corroborating evidence, not a "
                "multiplied or hidden score -- both individual assessments remain visible."
            )
        else:
            by_mmsi[mmsi] = CombinedDarkVesselCandidate(
                mmsi=mmsi, vessel_name=c2.gap.vessel_name, sources=["scenario2_pre_spill_gap"],
                scenario2=c2, combined_evidence_label=c2.evidence_label,
            )

    return list(by_mmsi.values()) + unlinked

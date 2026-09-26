"""
Dark Vessel Detection -- SAR-AIS cross-verification.

WHAT THIS DOES
---------------
Cross-checks SAR-detected vessel candidates (`wakewatch.inference.sar_vessel`)
against AIS positions around the SAR acquisition time. A SAR detection with
no matching AIS position is flagged as a **Potential Dark Vessel** -- never
asserted as a confirmed one, since a mismatch can equally be a SAR detection
error, an AIS coverage gap, a timing/positioning difference, or a vessel too
small for reliable detection. See `DarkVesselCandidate.status` and `.note`
for exactly which of those this module could and couldn't rule out.

WHAT THIS DOES NOT DO
-----------------------
- No new model is trained. Vessel candidates come from `sar_vessel.py`
  (which itself reuses the existing SAR segmentation model's Ship class).
  "Trajectory consistency" reuses the existing trajectory LSTM for a single
  next-ping prediction only; the multi-hour gap projection is classical dead
  reckoning (speed + course + elapsed time), clearly labelled as such --
  stretching the LSTM across a gap far longer than its trained ~60s step
  would manufacture false precision, not real evidence.
- This module does NOT touch `fusion.py`, its weights, or the existing
  attribution score in any way. It produces its own, separately-labelled
  "dark-vessel evidence" (HIGH/MEDIUM/LOW/UNKNOWN) -- a qualitative label,
  not a probability, and never a claim of legal responsibility.
- AIS coverage is never assumed. Callers must pass `ais_coverage_available`
  explicitly (from the same coverage check the rest of the project already
  uses); when False, this module reports "AIS coverage unavailable" rather
  than guessing at a dark-vessel flag it has no basis for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import drift as drift_mod
from .inference import trajectory as traj_mod
from .inference.sar_vessel import VesselCandidate

KNOTS_TO_MS = 0.514444


@dataclass
class MatchConfig:
    radius_km: float = 5.0
    time_tolerance_minutes: float = 60.0
    max_gap_hours: float = 48.0          # beyond this, "last known position" is too stale to be useful
    max_plausible_km: float = 150.0      # beyond this, treat as an unrelated vessel, not a lead
    dead_reckoning_high_km: float = 5.0
    dead_reckoning_medium_km: float = 20.0


# --------------------------------------------------------------------------
# Step 1: SAR <-> AIS spatial/temporal matching
# --------------------------------------------------------------------------
@dataclass
class AisMatch:
    matched: bool
    mmsi: Optional[int] = None
    vessel_name: Optional[str] = None
    distance_km: Optional[float] = None
    time_diff_minutes: Optional[float] = None


def _haversine_km_vec(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized haversine -- `trajectory.haversine_km` forces a scalar
    return, so a local array-safe version is used for one-vs-many distance
    checks (matching one SAR detection against many AIS rows at once)."""
    R = 6371.0
    lat1r, lon1r, lat2r, lon2r = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def match_candidate_to_ais(
    lat: float, lon: float, sar_time: datetime, ais_df: pd.DataFrame, config: MatchConfig,
) -> AisMatch:
    """Is there an AIS position close enough, in space and time, to this SAR detection?"""
    if sar_time.tzinfo is None:
        sar_time = sar_time.replace(tzinfo=None)
    ts = pd.to_datetime(ais_df["timestamp"], utc=True).dt.tz_localize(None)
    dt_minutes = (ts - pd.Timestamp(sar_time.replace(tzinfo=None))).dt.total_seconds().abs() / 60.0
    window = ais_df[dt_minutes <= config.time_tolerance_minutes]
    if window.empty:
        return AisMatch(matched=False)

    dists = _haversine_km_vec(lat, lon, window["latitude"].values, window["longitude"].values)
    idx = int(np.argmin(dists))
    best_row = window.iloc[idx]
    best_dist = float(dists[idx])
    if best_dist > config.radius_km:
        return AisMatch(matched=False)

    return AisMatch(
        matched=True, mmsi=int(best_row["mmsi"]),
        vessel_name=str(best_row.get("vessel_name", "")) or None,
        distance_km=round(best_dist, 3),
        time_diff_minutes=round(float(dt_minutes.iloc[window.index.get_loc(best_row.name)]), 1),
    )


# --------------------------------------------------------------------------
# Step 2: last known AIS position, before the SAR observation
# --------------------------------------------------------------------------
@dataclass
class LastKnownPosition:
    mmsi: int
    vessel_name: Optional[str]
    latitude: float
    longitude: float
    timestamp: datetime
    speed_knots: float
    course_deg: float
    gap_hours: float
    distance_km: float


def find_last_known_position(
    lat: float, lon: float, sar_time: datetime, ais_df: pd.DataFrame, config: MatchConfig,
) -> Optional[LastKnownPosition]:
    """
    Across every vessel in the feed, which one's most recent ping *before*
    the SAR observation sits plausibly close to where the SAR detection was
    made? This is the candidate lead for "which vessel might this be" -- not
    an identification, a starting point for the trajectory check below.
    """
    if sar_time.tzinfo is None:
        sar_time = sar_time.replace(tzinfo=None)
    ts = pd.to_datetime(ais_df["timestamp"], utc=True).dt.tz_localize(None)
    before = ais_df[ts < pd.Timestamp(sar_time.replace(tzinfo=None))].copy()
    if before.empty:
        return None
    before["_ts"] = ts[ts < pd.Timestamp(sar_time.replace(tzinfo=None))]

    best: Optional[LastKnownPosition] = None
    for mmsi, group in before.groupby("mmsi"):
        last = group.sort_values("_ts").iloc[-1]
        gap_hours = (sar_time.replace(tzinfo=None) - last["_ts"]).total_seconds() / 3600.0
        if gap_hours > config.max_gap_hours or gap_hours < 0:
            continue
        dist = traj_mod.haversine_km(lat, lon, last["latitude"], last["longitude"])
        if dist > config.max_plausible_km:
            continue
        if best is None or dist < best.distance_km:
            best = LastKnownPosition(
                mmsi=int(mmsi), vessel_name=str(last.get("vessel_name", "")) or None,
                latitude=float(last["latitude"]), longitude=float(last["longitude"]),
                timestamp=last["_ts"].to_pydatetime(),
                speed_knots=float(last.get("speed", 0.0)), course_deg=float(last.get("course", 0.0)),
                gap_hours=round(gap_hours, 2), distance_km=round(dist, 2),
            )
    return best


# --------------------------------------------------------------------------
# Step 3: trajectory consistency -- classical dead reckoning + an optional
# single-step LSTM prediction shown only as supplementary reference.
# --------------------------------------------------------------------------
@dataclass
class TrajectoryConsistency:
    label: str                    # "High" | "Medium" | "Low" | "Unknown"
    dead_reckoning_lat: Optional[float]
    dead_reckoning_lon: Optional[float]
    dead_reckoning_distance_km: Optional[float]
    lstm_next_ping_lat: Optional[float] = None
    lstm_next_ping_lon: Optional[float] = None
    note: str = ""


def assess_trajectory_consistency(
    last_known: LastKnownPosition, candidate_lat: float, candidate_lon: float,
    config: MatchConfig, traj_model=None, history_window: Optional[pd.DataFrame] = None,
) -> TrajectoryConsistency:
    """
    Dead-reckon the vessel forward from its last known speed/course for the
    full AIS-gap duration, then compare to where the SAR detection actually
    is. This is classical physics, not a model claim.

    If an 8-ping trajectory history is available immediately before the gap,
    the existing LSTM's single next-ping prediction is also computed and
    returned -- for reference only. It is never extrapolated across the
    multi-hour gap itself; the model was trained for a ~60s step, and
    presenting a naive multi-step rollout across hours as if it were a
    calibrated prediction would overstate what the model actually supports.
    """
    if last_known.gap_hours <= 0 or last_known.gap_hours > config.max_gap_hours:
        return TrajectoryConsistency("Unknown", None, None, None,
                                     note="AIS gap out of a plausible range for dead reckoning.")

    speed_ms = last_known.speed_knots * KNOTS_TO_MS
    bearing_rad = np.radians(last_known.course_deg)
    u_ms = speed_ms * np.sin(bearing_rad)
    v_ms = speed_ms * np.cos(bearing_rad)
    dr_lat, dr_lon = drift_mod._advance(last_known.latitude, last_known.longitude,
                                        u_ms, v_ms, last_known.gap_hours * 3600.0)
    dr_dist = traj_mod.haversine_km(dr_lat, dr_lon, candidate_lat, candidate_lon)

    if dr_dist <= config.dead_reckoning_high_km:
        label = "High"
    elif dr_dist <= config.dead_reckoning_medium_km:
        label = "Medium"
    else:
        label = "Low"

    lstm_lat = lstm_lon = None
    note = (f"Dead-reckoned from last known speed/course over {last_known.gap_hours:.1f} h; "
           f"landed {dr_dist:.1f} km from the SAR detection.")
    if traj_model is not None and history_window is not None and len(history_window) >= 8:
        try:
            pred = traj_mod.predict_next_position(traj_model, history_window.tail(8), strict=False)
            if pred.assessment.usable:
                lstm_lat, lstm_lon = pred.predicted_lat, pred.predicted_lon
                note += (" LSTM next-ping prediction shown for reference only "
                        "(not extrapolated across the full gap).")
        except Exception:
            pass

    return TrajectoryConsistency(label, dr_lat, dr_lon, round(dr_dist, 2), lstm_lat, lstm_lon, note)


# --------------------------------------------------------------------------
# Step 4: evidence score -- separate from, and never mixed into, `fusion.py`
# --------------------------------------------------------------------------
_TRAJ_NUMERIC = {"High": 1.0, "Medium": 0.6, "Low": 0.25, "Unknown": 0.4}


def _detection_confidence(candidate: VesselCandidate) -> float:
    """A simple, explainable 0-1 confidence from the blob's own SAR properties.

    Not a learned score -- brighter, more solidly-shaped, more elongated
    (vessel-like) blobs score higher. Documented, not hidden, in evidence.
    """
    intensity_term = min(candidate.mean_intensity / 255.0, 1.0)
    shape_term = min(candidate.aspect_ratio / 4.0, 1.0)   # elongated -> more vessel-like, caps at AR=4
    size_term = min(candidate.area_px / 200.0, 1.0)
    return round(0.5 * intensity_term + 0.3 * shape_term + 0.2 * size_term, 3)


@dataclass
class DarkVesselCandidate:
    candidate: VesselCandidate
    status: str                          # "ais_confirmed" | "potential_dark_vessel" |
                                          # "ais_coverage_unavailable" | "unverified_low_confidence"
    ais_match: Optional[AisMatch]
    last_known: Optional[LastKnownPosition]
    trajectory: Optional[TrajectoryConsistency]
    evidence_label: str                  # "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN"
    evidence_components: Dict[str, float]
    note: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidate": self.candidate.as_dict(),
            "status": self.status,
            "ais_match": self.ais_match.__dict__ if self.ais_match else None,
            "last_known": {**self.last_known.__dict__,
                           "timestamp": self.last_known.timestamp.isoformat()} if self.last_known else None,
            "trajectory": self.trajectory.__dict__ if self.trajectory else None,
            "evidence_label": self.evidence_label,
            "evidence_components": self.evidence_components,
            "note": self.note,
        }


def _evidence_label(components: Dict[str, float]) -> str:
    if not components:
        return "UNKNOWN"
    score = sum(components.values()) / len(components)
    if score >= 0.66:
        return "HIGH"
    if score >= 0.4:
        return "MEDIUM"
    return "LOW"


@dataclass
class DarkVesselReport:
    candidates: List[DarkVesselCandidate]
    ais_coverage_available: bool
    config: MatchConfig
    summary: str

    def as_dict(self) -> Dict[str, Any]:
        return {"candidates": [c.as_dict() for c in self.candidates],
               "ais_coverage_available": self.ais_coverage_available, "summary": self.summary}


def run_dark_vessel_detection(
    sar_candidates: List[VesselCandidate],
    sar_time: datetime,
    ais_df: Optional[pd.DataFrame],
    ais_coverage_available: bool,
    config: Optional[MatchConfig] = None,
    traj_model=None,
) -> DarkVesselReport:
    """
    Cross-check every geolocated SAR vessel candidate against AIS.

    `ais_coverage_available` must reflect a real, independently-checked
    coverage decision (e.g. `investigation.is_within_demo_aoi`/`_window`) --
    this function trusts that flag rather than re-deriving it, so it can
    never accidentally treat "no AIS rows happened to be nearby" the same
    as "there is no AIS source for this region at all."
    """
    config = config or MatchConfig()
    results: List[DarkVesselCandidate] = []

    for c in sar_candidates:
        if c.latitude is None or c.longitude is None:
            results.append(DarkVesselCandidate(
                candidate=c, status="unverified_low_confidence", ais_match=None,
                last_known=None, trajectory=None, evidence_label="UNKNOWN", evidence_components={},
                note="No geographic reference available for this detection (image was not "
                    "georeferenced) -- AIS cross-verification cannot run for it.",
            ))
            continue

        if not ais_coverage_available or ais_df is None or ais_df.empty:
            results.append(DarkVesselCandidate(
                candidate=c, status="ais_coverage_unavailable", ais_match=None,
                last_known=None, trajectory=None, evidence_label="UNKNOWN", evidence_components={},
                note="Unable to verify AIS \u2014 coverage unavailable for this region/time. "
                    "This detection is NOT labelled a dark vessel.",
            ))
            continue

        match = match_candidate_to_ais(c.latitude, c.longitude, sar_time, ais_df, config)
        if match.matched:
            results.append(DarkVesselCandidate(
                candidate=c, status="ais_confirmed", ais_match=match, last_known=None,
                trajectory=None, evidence_label="UNKNOWN", evidence_components={},
                note=f"Matched AIS vessel {match.vessel_name or match.mmsi} "
                    f"{match.distance_km:.2f} km away, {match.time_diff_minutes:.0f} min apart.",
            ))
            continue

        last_known = find_last_known_position(c.latitude, c.longitude, sar_time, ais_df, config)
        trajectory = None
        if last_known is not None:
            history = None
            if "mmsi" in ais_df.columns:
                vessel_track = ais_df[ais_df["mmsi"] == last_known.mmsi].sort_values("timestamp")
                if len(vessel_track) >= 8:
                    history = vessel_track.tail(8)
            trajectory = assess_trajectory_consistency(
                last_known, c.latitude, c.longitude, config, traj_model=traj_model,
                history_window=history)

        components = {"detection_confidence": _detection_confidence(c)}
        if last_known is not None:
            components["recency"] = max(0.0, 1.0 - last_known.gap_hours / config.max_gap_hours)
        if trajectory is not None:
            components["trajectory_consistency"] = _TRAJ_NUMERIC.get(trajectory.label, 0.4)

        note = "Potential Dark Vessel \u2014 SAR detected target with no matching AIS."
        if last_known is None:
            note += (" No plausible last-known AIS track was found nearby (within "
                    f"{config.max_gap_hours:.0f} h / {config.max_plausible_km:.0f} km) to compare against.")
        note += (" This is a potential SAR-AIS mismatch and does not by itself prove "
                "AIS shutdown or illegal activity.")

        results.append(DarkVesselCandidate(
            candidate=c, status="potential_dark_vessel", ais_match=match,
            last_known=last_known, trajectory=trajectory,
            evidence_label=_evidence_label(components), evidence_components=components,
            note=note,
        ))

    n_dark = sum(1 for r in results if r.status == "potential_dark_vessel")
    n_confirmed = sum(1 for r in results if r.status == "ais_confirmed")
    n_unavailable = sum(1 for r in results if r.status == "ais_coverage_unavailable")
    if not ais_coverage_available:
        summary = (f"{len(results)} SAR vessel candidate(s) detected. AIS coverage unavailable "
                  "for this region/time -- none can be verified against AIS.")
    else:
        summary = (f"{len(results)} SAR vessel candidate(s): {n_confirmed} AIS-confirmed, "
                  f"{n_dark} potential dark vessel(s), {n_unavailable} unverifiable.")

    return DarkVesselReport(candidates=results, ais_coverage_available=ais_coverage_available,
                            config=config, summary=summary)

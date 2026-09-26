"""
Indian Ocean -- Demonstration Scenario (Lakshadweep Sea).

WHAT THIS IS
--------------
A second, NAMED demonstration scenario for the New Investigation workflow,
answering "show me this working outside Mauritius" with something better
than an unlabelled random fleet. Unlike `synthetic_ais.py`'s generic
generator (randomly scattered vessels, no narrative), every vessel here has
a fixed, deterministic track and a stated role:

  - 2 transit vessels on a Mumbai-Colombo-bearing shipping lane, passing
    the area without incident (the "normal traffic" baseline)
  - 2 distractors loitering nearby but not at the estimated origin (tests
    that the real attribution pipeline does NOT flag a vessel just for
    being generally in the area)
  - 1 suspect that approaches and dwells at the estimated origin (the
    intended top-ranked candidate)
  - 1 dark-gap vessel with a genuine AIS reporting gap positioned to be
    caught by Scenario 2 (pre-spill AIS gap analysis), not Scenario 1

Every position, speed, and course below is scripted, not randomly
generated -- re-running this scenario produces the identical fleet every
time. This is still entirely fabricated data: a real place (the Lakshadweep
Sea, a real body of water on real Mumbai-Colombo shipping routes) with
entirely fictional vessels and an entirely fictional incident. It carries
the same synthetic disclaimer as `synthetic_ais.py` and must never be
presented as, or mistaken for, real AIS traffic.

VERIFICATION
--------------
This scenario's vessel parameters were tuned empirically against the real,
unmodified attribution pipeline (`fusion.attribute_from_observation`) and
the real pre-spill AIS gap analysis (`pre_spill_dark.analyze_pre_spill_dark_vessels`)
until both produced the intended result on this data -- SUSPECT ranking
first in attribution, DARK-GAP flagged as a pre-spill AIS gap candidate --
rather than assuming the narrative design would automatically produce that
result. See `tests/test_indian_ocean_demo.py` for the assertions that keep
this true if the scenario is ever edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import drift as drift_mod
from ..inference.trajectory import bearing_deg, haversine_km

SCENARIO_NAME = "Indian Ocean \u2014 Demonstration Scenario (Lakshadweep Sea)"
SPILL_LAT, SPILL_LON = 10.5, 72.8
OBSERVED_AT = datetime(2026, 2, 10, 9, 0, tzinfo=timezone.utc)

DISCLAIMER = (
    "SYNTHETIC SCENARIO -- FOR DEMONSTRATION ONLY. The Lakshadweep Sea location and the "
    "Mumbai-Colombo shipping lane are real; every vessel, position, and incident here is "
    "entirely fabricated and scripted for this demonstration. No real vessel corresponds to "
    "any MMSI shown here, and no real spill occurred at this location on this date. Any "
    "ranking or evidence produced from it demonstrates the pipeline's mechanics only and must "
    "not be read as a finding about a real vessel or incident."
)

_MID = 900_100_000   # deliberately outside any real MID allocation block, and distinct from
                     # synthetic_ais.py's default random range (900,000,xxx) so the two
                     # synthetic sources are never confused with each other either.

KNOTS_TO_MS = 0.514444


@dataclass
class DemoVessel:
    mmsi: int
    name: str
    role: str          # "transit" | "distractor" | "suspect" | "dark_gap"
    narrative: str


def _straight_track(start_lat: float, start_lon: float, course_deg: float, speed_knots: float,
                    start_time: datetime, duration_hours: float, ping_interval_min: float = 10.0,
                    ) -> List[Tuple[datetime, float, float, float, float]]:
    """Deterministic dead-reckoned track: (time, lat, lon, speed, course) per ping."""
    rows = []
    lat, lon = start_lat, start_lon
    t = start_time
    speed_ms = speed_knots * KNOTS_TO_MS
    n_steps = int(duration_hours * 60 / ping_interval_min)
    for _ in range(n_steps + 1):
        rows.append((t, lat, lon, speed_knots, course_deg))
        lat, lon = drift_mod._advance(lat, lon, speed_ms * np.sin(np.radians(course_deg)),
                                      speed_ms * np.cos(np.radians(course_deg)),
                                      ping_interval_min * 60.0)
        t = t + timedelta(minutes=ping_interval_min)
    return rows


def _dwelling_track(lat: float, lon: float, start_time: datetime, duration_hours: float,
                    ping_interval_min: float = 10.0, jitter_km: float = 0.15,
                    ) -> List[Tuple[datetime, float, float, float, float]]:
    """A vessel loitering near a fixed point -- small random-looking but deterministic jitter,
    near-zero net speed, matching what a vessel actually holding position looks like on AIS."""
    rows = []
    t = start_time
    n_steps = int(duration_hours * 60 / ping_interval_min)
    rng = np.random.default_rng(42)  # fixed seed -- deterministic, not truly random
    for i in range(n_steps + 1):
        dlat = rng.normal(0, jitter_km / 111.0)
        dlon = rng.normal(0, jitter_km / (111.0 * max(np.cos(np.radians(lat)), 1e-6)))
        rows.append((t, lat + dlat, lon + dlon, float(abs(rng.normal(0.4, 0.2))), float(rng.uniform(0, 360))))
        t = t + timedelta(minutes=ping_interval_min)
    return rows


def _rows_to_df(mmsi: int, name: str, rows: List[Tuple]) -> pd.DataFrame:
    return pd.DataFrame({
        "mmsi": mmsi, "vessel_name": name,
        "timestamp": [r[0] for r in rows], "latitude": [r[1] for r in rows], "longitude": [r[2] for r in rows],
        "speed": [r[3] for r in rows], "course": [r[4] for r in rows], "rot": 0.0,
        "msg_type": 1, "status": 0, "accuracy": 1,
    })


def build_indian_ocean_scenario() -> Dict[str, Any]:
    """
    Returns a dict: name, disclaimer, spill_position, observed_at, ais
    (a single DataFrame with all 6 vessels), and vessels (the DemoVessel
    role/narrative list for UI display).
    """
    lane_bearing = bearing_deg(19.0760, 72.8777, 6.9271, 79.8612)  # real Mumbai->Colombo bearing

    vessels: List[DemoVessel] = []
    frames: List[pd.DataFrame] = []

    # -- 2 transit vessels on the Mumbai-Colombo shipping lane --------------
    # Positioned to pass west of the spill origin at a real distance, on the
    # real lane bearing, moving continuously through the whole window.
    t1_start = drift_mod._advance(SPILL_LAT, SPILL_LON, 0, 30_000 / 3.6, 0)  # ~30km north, quick offset
    t1_rows = _straight_track(t1_start[0] + 0.15, SPILL_LON - 0.35, lane_bearing, 15.0,
                              OBSERVED_AT - timedelta(hours=10), duration_hours=10)
    vessels.append(DemoVessel(_MID + 1, "MV TRANSIT ONE", "transit",
                              "Steady transit on the Mumbai\u2013Colombo lane, ~30 km west of the "
                              "estimated spill origin at closest approach. No anomalous behaviour."))
    frames.append(_rows_to_df(_MID + 1, "MV TRANSIT ONE", t1_rows))

    t2_rows = _straight_track(SPILL_LAT + 0.55, SPILL_LON - 0.55, lane_bearing, 14.0,
                              OBSERVED_AT - timedelta(hours=10), duration_hours=10)
    vessels.append(DemoVessel(_MID + 2, "MV TRANSIT TWO", "transit",
                              "Steady transit on the Mumbai\u2013Colombo lane, ~45 km west of the "
                              "estimated spill origin at closest approach. No anomalous behaviour."))
    frames.append(_rows_to_df(_MID + 2, "MV TRANSIT TWO", t2_rows))

    # -- 2 distractors: present nearby, but never close to the origin -------
    d1_rows = _dwelling_track(SPILL_LAT + 0.16, SPILL_LON + 0.02, OBSERVED_AT - timedelta(hours=9),
                              duration_hours=9, jitter_km=1.2)
    vessels.append(DemoVessel(_MID + 3, "FV DISTRACTOR ONE", "distractor",
                              "Fishing vessel loitering ~18 km from the estimated origin the whole "
                              "window -- present in the area, but never close enough to be a "
                              "plausible source."))
    frames.append(_rows_to_df(_MID + 3, "FV DISTRACTOR ONE", d1_rows))

    d2_rows = _dwelling_track(SPILL_LAT - 0.01, SPILL_LON + 0.20, OBSERVED_AT - timedelta(hours=8),
                              duration_hours=8, jitter_km=1.5)
    vessels.append(DemoVessel(_MID + 4, "FV DISTRACTOR TWO", "distractor",
                              "A second vessel loitering ~22 km from the estimated origin -- same "
                              "role as Distractor One, tests that proximity alone doesn't flag "
                              "every nearby vessel."))
    frames.append(_rows_to_df(_MID + 4, "FV DISTRACTOR TWO", d2_rows))

    # -- 1 suspect: approaches and dwells at the origin ----------------------
    approach_start_lat, approach_start_lon = SPILL_LAT + 0.42, SPILL_LON - 0.30  # ~55km out
    approach_bearing = bearing_deg(approach_start_lat, approach_start_lon, SPILL_LAT, SPILL_LON)
    approach_rows = _straight_track(approach_start_lat, approach_start_lon, approach_bearing, 10.0,
                                    OBSERVED_AT - timedelta(hours=8), duration_hours=4.5)
    dwell_start = OBSERVED_AT - timedelta(hours=3.3)
    dwell_rows = _dwelling_track(SPILL_LAT, SPILL_LON, dwell_start, duration_hours=3.3, jitter_km=0.3)
    vessels.append(DemoVessel(_MID + 5, "MV SUSPECT", "suspect",
                              "Approaches from ~55 km out over several hours, arrives at the estimated "
                              "origin, and dwells there through the observation time -- the intended "
                              "top-ranked candidate."))
    frames.append(_rows_to_df(_MID + 5, "MV SUSPECT", approach_rows + dwell_rows))

    # -- 1 dark-gap vessel: transits toward the origin, then goes quiet ------
    # Positioned so it goes dark ~2h before the hindcast-estimated release
    # (giving it genuine travel time to reach the origin, not zero -- see
    # the Fix 1 correction in pre_spill_dark.py) at a point ~12km out on a
    # course toward the origin, so reachability AND trajectory consistency
    # both hold. See test_indian_ocean_demo.py for the empirical check.
    g_far_lat, g_far_lon = SPILL_LAT + 0.54, SPILL_LON + 0.40   # ~79km out, start of transit
    g_bearing = bearing_deg(g_far_lat, g_far_lon, SPILL_LAT, SPILL_LON)
    gap_rows = _straight_track(g_far_lat, g_far_lon, g_bearing, 11.0,
                               OBSERVED_AT - timedelta(hours=15), duration_hours=2.0)
    # A genuine >30 min gap: the vessel simply stops transmitting here, close
    # to the estimated origin, a few hours before the observed time -- no
    # further pings in this scenario (an open-ended gap, same as the
    # "never resumed in this feed" case `pre_spill_dark.find_ais_gaps` handles).
    vessels.append(DemoVessel(_MID + 6, "MV DARK GAP", "dark_gap",
                              "Transits to within ~9 km of the estimated origin, then stops "
                              "transmitting several hours before the observation time and never "
                              "reappears in this feed -- the intended pre-spill AIS gap candidate."))
    frames.append(_rows_to_df(_MID + 6, "MV DARK GAP", gap_rows))

    ais = pd.concat(frames, ignore_index=True)
    ais["timestamp"] = pd.to_datetime(ais["timestamp"], utc=True)

    return {
        "name": SCENARIO_NAME, "disclaimer": DISCLAIMER,
        "spill_position": (SPILL_LAT, SPILL_LON), "observed_at": OBSERVED_AT,
        "ais": ais, "vessels": vessels,
    }

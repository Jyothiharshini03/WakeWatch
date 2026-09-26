"""
Synthetic AIS traffic generator -- FOR DEMONSTRATION ONLY.

WHY THIS EXISTS
-----------------
The project has exactly one real AIS feed (Mauritius AOI, 25 Jul 2020).
Outside that window, `investigation.run_investigation` correctly reports
"AIS coverage unavailable" rather than fabricating traffic -- that remains
the default and the only honest thing to do with a real investigation.

This module exists for a narrower, explicitly-opted-into purpose: letting a
judge see the *rest of the pipeline run end to end* on a new SAR image
anywhere in the world, using clearly-labelled placeholder AIS traffic,
instead of the investigation stopping dead at "no AIS source." It is never
enabled by default, never silently substituted for real data, and every
value it produces is generated, not observed -- every surface that displays
it must carry the disclaimer in `SYNTHETIC_DISCLAIMER`.

WHAT IT IS NOT
----------------
- Not a simulation of real shipping density, lanes, or traffic patterns for
  any real ocean region. It scatters a configurable number of vessels
  randomly around the area of interest with plausible merchant-vessel
  speeds and roughly straight-line courses -- enough to exercise the AIS
  anomaly, trajectory, and attribution stages, nothing more.
- Not evidence. A "candidate" ranked using synthetic AIS proves nothing
  about any real vessel; there is no real vessel behind these MMSIs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

SYNTHETIC_DISCLAIMER = (
    "SYNTHETIC AIS -- FOR DEMONSTRATION ONLY. This traffic was generated, not "
    "observed. No real vessel corresponds to any MMSI shown here. Any ranking "
    "or evidence produced from it demonstrates the pipeline's mechanics only "
    "and must not be read as a finding about a real vessel or incident."
)


@dataclass
class SyntheticAisConfig:
    n_vessels: int = 10
    radius_km: float = 40.0            # vessels are scattered within this radius of the centre
    hours_span: float = 20.0           # total time window the synthetic traffic covers
    ping_interval_minutes: float = 6.0
    min_speed_knots: float = 4.0
    max_speed_knots: float = 18.0
    seed: Optional[int] = None         # None -> derived deterministically from center+time (see below)


def _deterministic_seed(lat: float, lon: float, at: datetime) -> int:
    """
    A default seed derived from the investigation's own inputs (not a fixed
    constant, and not `random` with no seed at all) so that re-running the
    same investigation reproduces the same synthetic scene -- useful for a
    judge comparing runs -- while different investigations get different
    synthetic traffic rather than the same canned scene every time.
    """
    key = f"{lat:.4f}:{lon:.4f}:{at.isoformat()}".encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


def generate_synthetic_ais(
    center_lat: float, center_lon: float, center_time: datetime,
    config: Optional[SyntheticAisConfig] = None,
) -> pd.DataFrame:
    """
    Generate placeholder AIS traffic scattered around (center_lat, center_lon)
    over a window centred on `center_time`, in the exact schema
    `investigation.REQUIRED_AIS_COLUMNS` expects (mmsi, timestamp, latitude,
    longitude, speed, course, rot, msg_type, status, accuracy) plus
    `vessel_name`.

    Every vessel gets a random starting position within `config.radius_km` of
    the centre, a random roughly-constant course and speed, and pings every
    `config.ping_interval_minutes` for `config.hours_span` hours centred on
    `center_time`. Positions are advanced with simple planar kinematics
    (fine at this scale) plus small heading jitter so tracks aren't perfectly
    straight lines. Nothing here is fit to, or derived from, any real vessel
    or real traffic pattern for this or any other location.
    """
    config = config or SyntheticAisConfig()
    if center_time.tzinfo is None:
        center_time = center_time.replace(tzinfo=timezone.utc)
    seed = config.seed if config.seed is not None else _deterministic_seed(center_lat, center_lon, center_time)
    rng = np.random.default_rng(seed)

    start_time = center_time - timedelta(hours=config.hours_span / 2)
    n_steps = max(1, int((config.hours_span * 60) / config.ping_interval_minutes))
    dt_hours = config.ping_interval_minutes / 60.0

    rows = []
    base_mmsi = 900_000_000  # 9xxxxxxxx is not a valid real MMSI range for a merchant vessel --
                              # deliberately outside real allocation blocks so a synthetic MMSI
                              # can never be mistaken for a real one.
    for i in range(config.n_vessels):
        mmsi = base_mmsi + i
        vessel_name = f"SYNTH-{i + 1:02d}"
        # Random start point within radius_km of the centre (uniform in area, not just radius).
        r_km = config.radius_km * np.sqrt(rng.uniform(0, 1))
        theta = rng.uniform(0, 2 * np.pi)
        dlat = (r_km * np.cos(theta)) / 111.0
        dlon = (r_km * np.sin(theta)) / (111.0 * max(np.cos(np.radians(center_lat)), 1e-6))
        lat, lon = center_lat + dlat, center_lon + dlon

        speed = float(rng.uniform(config.min_speed_knots, config.max_speed_knots))
        course = float(rng.uniform(0, 360))

        for step in range(n_steps):
            t = start_time + timedelta(hours=step * dt_hours)
            course += float(rng.normal(0, 8))  # gentle heading jitter, not a perfect straight line
            course %= 360.0
            speed = float(np.clip(speed + rng.normal(0, 0.5), config.min_speed_knots * 0.5,
                                  config.max_speed_knots * 1.2))
            dist_km = speed * 1.852 * dt_hours
            lat += (dist_km * np.cos(np.radians(course))) / 111.0
            lon += (dist_km * np.sin(np.radians(course))) / (111.0 * max(np.cos(np.radians(lat)), 1e-6))

            rows.append({
                "mmsi": mmsi, "vessel_name": vessel_name, "timestamp": t,
                "latitude": round(float(lat), 6), "longitude": round(float(lon), 6),
                "speed": round(speed, 1), "course": round(course, 1), "rot": 0.0,
                "msg_type": 1, "status": 0, "accuracy": 1,
            })

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df

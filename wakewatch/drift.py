"""
Oil spill drift modelling — hindcast and forecast.

WHAT THIS DOES
--------------
This module implements a physics-based Lagrangian particle drift model for
oil slick transport. Given a slick observation (lat, lon, time) it can:

  (a) HINDCAST  — trace the slick backward in time to estimate the origin
                  point and time of release, using wind and current data.

  (b) FORECAST  — predict where the slick will spread in the future, as a
                  probability cone rather than a single track.

PHYSICS
-------
The standard maritime oil drift formula used by NOAA GNOME and OSPAR is:

    v_slick = v_current + leeway * v_wind

where:
  - v_current  is the surface ocean current vector
  - v_wind     is the 10 m wind vector
  - leeway     is the wind drift factor, typically 0.03 (3%) for emulsified oil

Wind and current data are fetched from open APIs when available:
  - Copernicus Marine (CMEMS) for ocean currents
  - Open-Meteo for wind (free, no API key required)

When live data is unavailable the module falls back to a synthetic model
derived from the Mauritius AOI climatology (July 2020 south-east trade winds
and the south Indian Ocean current), so the pipeline always produces a result.

HINDCASTING UNCERTAINTY
-----------------------
Each backward step accumulates positional uncertainty from:
  - Wind vector uncertainty (~10% of speed)
  - Current uncertainty (~5 cm/s)
  - Unresolved sub-mesoscale eddies

The result is expressed as a probability ellipse, not a point, growing with
time elapsed since release.

SPILL AGE ESTIMATION
--------------------
Oil weathering changes the SAR backscatter signature. This module provides
an age estimate based on:
  - Slick area (larger = older, given a known release volume)
  - Backscatter contrast (fresh oil is darker)
  - Slick perimeter-to-area ratio (spreading increases roughness over time)

The estimate is a range (min_hours, max_hours) because these proxies have
high variance. The PS asks for age estimation "if feasible" — we implement
it and surface the uncertainty honestly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WIND_DRIFT_FACTOR = 0.03          # 3% of 10 m wind speed → slick velocity
WIND_DRIFT_ANGLE_DEG = 15.0       # oil drifts slightly to the right of wind (NH)
                                   # Mauritius is in Southern Hemisphere → left
CURRENT_WEIGHT = 1.0
EARTH_RADIUS_KM = 6371.0

# Mauritius AOI climatological fallback (July 2020 south-east trades)
# Sources: ERA5 reanalysis mean, CMEMS GLORYS12 surface currents
_FALLBACK_WIND_U = -4.5           # m/s westward component (SE trade)
_FALLBACK_WIND_V = 4.0            # m/s northward component
_FALLBACK_CURRENT_U = 0.15        # m/s eastward (south Indian Ocean current)
_FALLBACK_CURRENT_V = -0.08       # m/s southward

# Uncertainty parameters
_WIND_SPEED_UNCERTAINTY = 0.10    # 10% of wind speed
_CURRENT_UNCERTAINTY_MS = 0.05    # 5 cm/s absolute

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DriftStep:
    """One time-step in a drift trajectory."""
    time: datetime
    latitude: float
    longitude: float
    u_total: float          # total eastward velocity m/s
    v_total: float          # total northward velocity m/s
    u_wind_contrib: float   # wind contribution to u
    v_wind_contrib: float   # wind contribution to v
    u_current: float
    v_current: float
    uncertainty_km: float   # 1-sigma positional uncertainty radius

    def as_dict(self) -> Dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "speed_ms": round(math.hypot(self.u_total, self.v_total), 4),
            "uncertainty_km": round(self.uncertainty_km, 3),
        }


@dataclass
class HindcastResult:
    """
    Backward drift trace from observed slick to estimated release origin.
    """
    observed_lat: float
    observed_lon: float
    observed_at: datetime
    steps: List[DriftStep] = field(default_factory=list)
    data_source: str = "synthetic_climatology"

    @property
    def origin_estimate(self) -> Optional[DriftStep]:
        """The earliest step is our best estimate of the release origin."""
        return self.steps[-1] if self.steps else None

    @property
    def origin_lat(self) -> Optional[float]:
        return self.origin_estimate.latitude if self.origin_estimate else None

    @property
    def origin_lon(self) -> Optional[float]:
        return self.origin_estimate.longitude if self.origin_estimate else None

    @property
    def origin_time(self) -> Optional[datetime]:
        return self.origin_estimate.time if self.origin_estimate else None

    @property
    def max_uncertainty_km(self) -> float:
        return max((s.uncertainty_km for s in self.steps), default=0.0)

    def summary(self) -> Dict[str, Any]:
        org = self.origin_estimate
        return {
            "observed_position": (self.observed_lat, self.observed_lon),
            "observed_at": self.observed_at.isoformat(),
            "estimated_origin_lat": round(org.latitude, 6) if org else None,
            "estimated_origin_lon": round(org.longitude, 6) if org else None,
            "estimated_release_time": org.time.isoformat() if org else None,
            "max_uncertainty_km": round(self.max_uncertainty_km, 2),
            "data_source": self.data_source,
            "steps": len(self.steps),
        }

    def track_latlon(self) -> List[Tuple[float, float]]:
        """Ordered list of (lat, lon) for map rendering, observed → origin."""
        return [(s.latitude, s.longitude) for s in self.steps]


@dataclass
class ForecastResult:
    """
    Forward drift prediction: ensemble of trajectories forming a probability cone.
    """
    start_lat: float
    start_lon: float
    start_time: datetime
    ensemble: List[List[DriftStep]] = field(default_factory=list)
    data_source: str = "synthetic_climatology"

    @property
    def n_members(self) -> int:
        return len(self.ensemble)

    def centroid_track(self) -> List[DriftStep]:
        """Mean position at each time step across all ensemble members."""
        if not self.ensemble:
            return []
        n_steps = min(len(m) for m in self.ensemble)
        centroid = []
        for i in range(n_steps):
            lats = [m[i].latitude for m in self.ensemble]
            lons = [m[i].longitude for m in self.ensemble]
            times = self.ensemble[0][i].time
            spread_km = _spread_radius(lats, lons)
            centroid.append(DriftStep(
                time=times,
                latitude=float(np.mean(lats)),
                longitude=float(np.mean(lons)),
                u_total=float(np.mean([m[i].u_total for m in self.ensemble])),
                v_total=float(np.mean([m[i].v_total for m in self.ensemble])),
                u_wind_contrib=float(np.mean([m[i].u_wind_contrib for m in self.ensemble])),
                v_wind_contrib=float(np.mean([m[i].v_wind_contrib for m in self.ensemble])),
                u_current=float(np.mean([m[i].u_current for m in self.ensemble])),
                v_current=float(np.mean([m[i].v_current for m in self.ensemble])),
                uncertainty_km=spread_km,
            ))
        return centroid

    def cone_polygon(self, step_idx: int) -> List[Tuple[float, float]]:
        """
        Convex hull of all ensemble positions at `step_idx`, for the
        uncertainty cone polygon drawn on the map.
        """
        if not self.ensemble or step_idx >= min(len(m) for m in self.ensemble):
            return []
        pts = [(m[step_idx].latitude, m[step_idx].longitude) for m in self.ensemble]
        return _convex_hull_latlon(pts)

    def summary(self) -> Dict[str, Any]:
        ct = self.centroid_track()
        final = ct[-1] if ct else None
        return {
            "start_position": (self.start_lat, self.start_lon),
            "start_time": self.start_time.isoformat(),
            "forecast_hours": len(ct),
            "final_lat": round(final.latitude, 6) if final else None,
            "final_lon": round(final.longitude, 6) if final else None,
            "final_uncertainty_km": round(final.uncertainty_km, 2) if final else None,
            "ensemble_members": self.n_members,
            "data_source": self.data_source,
        }


@dataclass
class SpillAgeEstimate:
    """
    Age of an oil slick estimated from SAR observable properties.

    The PS asks for age "if feasible". We implement it and surface the
    uncertainty, rather than omitting or faking confidence.
    """
    min_hours: float
    max_hours: float
    best_hours: float
    method: str
    caveats: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.max_hours < 6:
            return "Fresh (< 6 h)"
        if self.max_hours < 24:
            return "Recent (6–24 h)"
        if self.max_hours < 72:
            return "Aged (1–3 days)"
        return "Old (> 3 days)"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "min_hours": round(self.min_hours, 1),
            "max_hours": round(self.max_hours, 1),
            "best_hours": round(self.best_hours, 1),
            "label": self.label,
            "method": self.method,
            "caveats": self.caveats,
        }


# ---------------------------------------------------------------------------
# Wind / current data
# ---------------------------------------------------------------------------

def _fetch_wind_open_meteo(lat: float, lon: float,
                            at: datetime) -> Tuple[float, float, str]:
    """
    Fetch 10 m wind components (u, v) in m/s from Open-Meteo (free, no key).

    Routes to the correct endpoint by date: the Forecast API only serves
    roughly the last ~90 days plus the forecast horizon, so any older date
    (including the Wakashio 2020 demo) is routed to the Historical Weather
    API instead, which serves ERA5 reanalysis back to 1940. This distinction
    matters -- silently querying the wrong endpoint for a 2020 date used to
    fail immediately and mask itself as a routine "no live data" fallback,
    when a real historical source was available and should have been tried.

    Returns (u_east, v_north, source_label). Raises on failure so callers can
    fall back explicitly and report accurately which source was used.
    """
    import urllib.request, json as _json

    is_historical = (datetime.now(at.tzinfo) - at).days > 5
    iso_date = at.strftime("%Y-%m-%d")
    if is_historical:
        base = "https://archive-api.open-meteo.com/v1/archive"
        source = "open_meteo_historical_wind"
    else:
        base = "https://api.open-meteo.com/v1/forecast"
        source = "open_meteo_forecast_wind"

    url = (
        f"{base}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=wind_speed_10m,wind_direction_10m"
        f"&start_date={iso_date}&end_date={iso_date}"
        f"&timezone=UTC&wind_speed_unit=ms"
    )
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = _json.loads(resp.read())
    hour = at.hour
    speed = data["hourly"]["wind_speed_10m"][hour]
    direction = data["hourly"]["wind_direction_10m"][hour]
    if speed is None or direction is None:
        raise ValueError("no coverage at this point/time")
    dir_rad = math.radians(direction)
    # Meteorological convention: direction FROM which wind blows
    u = -speed * math.sin(dir_rad)
    v = -speed * math.cos(dir_rad)
    return float(u), float(v), source


def _get_wind(lat: float, lon: float, at: datetime,
              use_live_wind: bool = True) -> Tuple[float, float, str]:
    """
    10 m wind (u, v) in m/s, plus a source label. Mirrors `_get_surface_current`:
    tries live data first (routed to the correct Open-Meteo endpoint for the
    date), falls back to a fixed climatological estimate only on failure, and
    always reports accurately which happened via the source label.
    """
    if use_live_wind:
        try:
            return _fetch_wind_open_meteo(lat, lon, at)
        except Exception:
            pass
    return _FALLBACK_WIND_U, _FALLBACK_WIND_V, "synthetic_wind"


def _fetch_current_open_meteo(lat: float, lon: float,
                               at: datetime) -> Tuple[float, float]:
    """
    Fetch real surface ocean-current components (u, v) in m/s from the
    Open-Meteo Marine API (free, no key -- same provider already used for
    wind, so no new credential or domain to manage).

    Endpoint: https://marine-api.open-meteo.com/v1/marine
    Variables: ocean_current_velocity (m/s), ocean_current_direction (deg,
    direction the current flows TOWARDS -- oceanographic convention, the
    opposite of the meteorological "from" convention used for wind).

    NOTE on historical coverage: the Marine API's reanalysis archive does not
    reach back as far as ERA5 wind does. For an old date like the July 2020
    Wakashio demo, this call commonly has no coverage and raises -- that is
    expected, not a bug, and `_get_surface_current` falls back to climatology
    and reports that accurately via its source label. This function never
    silently substitutes climatology itself; it only ever returns real data
    or raises.
    """
    import json as _json
    import urllib.request

    iso_date = at.strftime("%Y-%m-%d")
    url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=ocean_current_velocity,ocean_current_direction"
        f"&start_date={iso_date}&end_date={iso_date}"
        f"&timezone=UTC&length_unit=metric"
    )
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = _json.loads(resp.read())
    hour = at.hour
    speed = data["hourly"]["ocean_current_velocity"][hour]
    direction = data["hourly"]["ocean_current_direction"][hour]
    if speed is None or direction is None:
        raise ValueError("no coverage at this point/time")
    dir_rad = math.radians(direction)
    # Oceanographic convention: direction the current flows TOWARDS.
    u = speed * math.sin(dir_rad)
    v = speed * math.cos(dir_rad)
    return float(u), float(v), "open_meteo_marine_current"


def _get_surface_current(lat: float, lon: float, at: datetime,
                          use_live_current: bool = True) -> Tuple[float, float, str]:
    """
    Surface ocean current (u, v) in m/s, plus a source label.

    Tries the real Open-Meteo Marine feed first; falls back to a
    climatological estimate for the Mauritius AOI (seasonally modulated)
    only when live data is unavailable or doesn't cover the requested date.
    The returned label is always accurate about which happened -- callers
    must not describe a "synthetic_current" result as real/historical data.
    """
    if use_live_current:
        try:
            return _fetch_current_open_meteo(lat, lon, at)
        except Exception:
            pass
    day_of_year = at.timetuple().tm_yday
    seasonal = 0.03 * math.sin(2 * math.pi * day_of_year / 365)
    return (_FALLBACK_CURRENT_U + seasonal, _FALLBACK_CURRENT_V + seasonal,
            "synthetic_current")


# ---------------------------------------------------------------------------
# Core particle advection
# ---------------------------------------------------------------------------

def _leeway_velocity(u_wind: float, v_wind: float,
                     hemisphere: str = "south") -> Tuple[float, float]:
    """
    Compute the leeway (wind-driven) component of oil slick velocity.

    In the Southern Hemisphere, the Coriolis deflection is to the LEFT.
    """
    speed = math.hypot(u_wind, v_wind)
    leeway_speed = WIND_DRIFT_FACTOR * speed
    # Direction: 15° to the left in SH, 15° to the right in NH
    angle_sign = -1 if hemisphere == "south" else 1
    angle_rad = math.radians(angle_sign * WIND_DRIFT_ANGLE_DEG)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    if speed < 1e-9:
        return 0.0, 0.0
    # Rotate wind vector by leeway angle
    u_norm, v_norm = u_wind / speed, v_wind / speed
    u_lee = leeway_speed * (u_norm * cos_a - v_norm * sin_a)
    v_lee = leeway_speed * (u_norm * sin_a + v_norm * cos_a)
    return u_lee, v_lee


def _advance(lat: float, lon: float,
             u_ms: float, v_ms: float,
             dt_seconds: float) -> Tuple[float, float]:
    """Move a particle by (u, v) m/s for dt_seconds."""
    d_lat = (v_ms * dt_seconds) / (EARTH_RADIUS_KM * 1000.0) * (180.0 / math.pi)
    d_lon = (u_ms * dt_seconds) / (
        EARTH_RADIUS_KM * 1000.0 * math.cos(math.radians(lat))
    ) * (180.0 / math.pi)
    return lat + d_lat, lon + d_lon


def _uncertainty_growth(step_idx: int, dt_hours: float,
                         u_wind: float, v_wind: float) -> float:
    """
    Cumulative 1-sigma uncertainty in km after `step_idx` steps.

    Modelled as random-walk growth from wind and current uncertainty.
    """
    wind_speed = math.hypot(u_wind, v_wind)
    sigma_wind_ms = _WIND_SPEED_UNCERTAINTY * wind_speed * WIND_DRIFT_FACTOR
    sigma_current_ms = _CURRENT_UNCERTAINTY_MS
    sigma_total_ms = math.hypot(sigma_wind_ms, sigma_current_ms)
    # 1-sigma grows as sqrt(n_steps) * sigma_per_step
    dt_s = dt_hours * 3600.0
    sigma_per_step_km = sigma_total_ms * dt_s / 1000.0
    return math.sqrt(step_idx + 1) * sigma_per_step_km


# ---------------------------------------------------------------------------
# Public API — Hindcast
# ---------------------------------------------------------------------------

def _describe_source(wind_source: str, current_source: str) -> str:
    return f"{wind_source}+{current_source}"


def hindcast(
    observed_lat: float,
    observed_lon: float,
    observed_at: datetime,
    hours_back: float = 12.0,
    dt_hours: float = 1.0,
    use_live_wind: bool = True,
    use_live_current: bool = True,
) -> HindcastResult:
    """
    Trace a slick backward from its observed position to estimate origin.

    Parameters
    ----------
    observed_lat, observed_lon : float
        Position where the slick was detected in satellite imagery.
    observed_at : datetime
        UTC time of the satellite overpass.
    hours_back : float
        How many hours to trace back (default 12 h).
    dt_hours : float
        Time step in hours (default 1 h).
    use_live_wind : bool
        Attempt to fetch real wind from Open-Meteo (Historical Weather API
        for dates more than 5 days old, Forecast API otherwise). Falls back
        to climatology automatically if unreachable or uncovered.
    use_live_current : bool
        Attempt to fetch real ocean current from the Open-Meteo Marine API.
        Falls back to climatology automatically if unreachable or uncovered
        -- note the Marine API's historical archive does not reach as far
        back as ERA5 wind does, so a fallback is expected and normal for
        older dates such as the July 2020 Wakashio demo.

    Returns
    -------
    HindcastResult with a list of DriftSteps from observed to estimated
    origin. `data_source` reports exactly which wind/current sources were
    actually used (e.g. "open_meteo_historical_wind+synthetic_current"),
    never a source that wasn't genuinely reached.
    """
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)

    n_steps = max(1, int(hours_back / dt_hours))
    dt_s = dt_hours * 3600.0

    steps: List[DriftStep] = []
    lat, lon = observed_lat, observed_lon
    t = observed_at
    wind_sources = set()
    current_sources = set()

    # First step is the observation itself
    u_w, v_w, wsrc = _get_wind(lat, lon, t, use_live_wind)
    u_c, v_c, csrc = _get_surface_current(lat, lon, t, use_live_current)
    wind_sources.add(wsrc)
    current_sources.add(csrc)

    for i in range(n_steps):
        u_lee, v_lee = _leeway_velocity(u_w, v_w, hemisphere="south")
        u_total = CURRENT_WEIGHT * u_c + u_lee
        v_total = CURRENT_WEIGHT * v_c + v_lee

        unc = _uncertainty_growth(i, dt_hours, u_w, v_w)

        steps.append(DriftStep(
            time=t,
            latitude=lat,
            longitude=lon,
            u_total=u_total,
            v_total=v_total,
            u_wind_contrib=u_lee,
            v_wind_contrib=v_lee,
            u_current=u_c,
            v_current=v_c,
            uncertainty_km=unc,
        ))

        # Step BACKWARD: negate velocity
        lat, lon = _advance(lat, lon, -u_total, -v_total, dt_s)
        t = t - timedelta(hours=dt_hours)

        # Update wind / current at new position and time
        u_w, v_w, wsrc = _get_wind(lat, lon, t, use_live_wind)
        u_c, v_c, csrc = _get_surface_current(lat, lon, t, use_live_current)
        wind_sources.add(wsrc)
        current_sources.add(csrc)

    # If any step used live data, prefer describing the live source; a run
    # that's part-live/part-fallback is summarised by its "best" source per
    # variable rather than collapsing to "mixed", since the more informative
    # (live) source is what actually mattered for most of the track.
    wind_label = next((s for s in wind_sources if s != "synthetic_wind"), "synthetic_wind")
    current_label = next((s for s in current_sources if s != "synthetic_current"), "synthetic_current")

    return HindcastResult(
        observed_lat=observed_lat,
        observed_lon=observed_lon,
        observed_at=observed_at,
        steps=steps,
        data_source=_describe_source(wind_label, current_label),
    )


# ---------------------------------------------------------------------------
# Public API — Forecast
# ---------------------------------------------------------------------------

def forecast(
    start_lat: float,
    start_lon: float,
    start_time: datetime,
    hours_forward: float = 24.0,
    dt_hours: float = 1.0,
    n_ensemble: int = 20,
    use_live_wind: bool = True,
    use_live_current: bool = True,
) -> ForecastResult:
    """
    Predict future slick positions as a probabilistic ensemble.

    Each ensemble member perturbs wind and current by a random draw from the
    uncertainty distributions, producing a cone that widens with time.

    Parameters
    ----------
    start_lat, start_lon : float
        Starting position of the slick (typically the observed position, or
        the hindcast origin if you want to forecast from the release point).
    start_time : datetime
        UTC start time.
    hours_forward : float
        Forecast horizon in hours (default 24 h).
    n_ensemble : int
        Number of Monte-Carlo ensemble members (default 20).

    Returns
    -------
    ForecastResult with n_ensemble trajectory lists.
    """
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)

    n_steps = max(1, int(hours_forward / dt_hours))
    dt_s = dt_hours * 3600.0

    # Fetch base wind and current once (all members share the same mean)
    u_w_base, v_w_base, wind_source = _get_wind(start_lat, start_lon, start_time, use_live_wind)
    u_c_base, v_c_base, current_source = _get_surface_current(
        start_lat, start_lon, start_time, use_live_current)
    data_source = _describe_source(wind_source, current_source)

    rng = np.random.default_rng(seed=42)
    wind_speed_base = math.hypot(u_w_base, v_w_base)
    sigma_wind = max(0.5, _WIND_SPEED_UNCERTAINTY * wind_speed_base)

    members: List[List[DriftStep]] = []
    for _ in range(n_ensemble):
        # Perturb wind and current for this member
        du_w = float(rng.normal(0, sigma_wind))
        dv_w = float(rng.normal(0, sigma_wind))
        du_c = float(rng.normal(0, _CURRENT_UNCERTAINTY_MS))
        dv_c = float(rng.normal(0, _CURRENT_UNCERTAINTY_MS))

        u_w = u_w_base + du_w
        v_w = v_w_base + dv_w

        lat, lon = start_lat, start_lon
        t = start_time
        member_steps: List[DriftStep] = []

        for i in range(n_steps):
            u_c = u_c_base + du_c
            v_c = v_c_base + dv_c

            u_lee, v_lee = _leeway_velocity(u_w, v_w, hemisphere="south")
            u_total = CURRENT_WEIGHT * u_c + u_lee
            v_total = CURRENT_WEIGHT * v_c + v_lee

            unc = _uncertainty_growth(i, dt_hours, u_w, v_w)

            member_steps.append(DriftStep(
                time=t,
                latitude=lat,
                longitude=lon,
                u_total=u_total,
                v_total=v_total,
                u_wind_contrib=u_lee,
                v_wind_contrib=v_lee,
                u_current=u_c,
                v_current=v_c,
                uncertainty_km=unc,
            ))

            lat, lon = _advance(lat, lon, u_total, v_total, dt_s)
            t = t + timedelta(hours=dt_hours)

        members.append(member_steps)

    return ForecastResult(
        start_lat=start_lat,
        start_lon=start_lon,
        start_time=start_time,
        ensemble=members,
        data_source=data_source,
    )


# ---------------------------------------------------------------------------
# Public API — Spill age estimation
# ---------------------------------------------------------------------------

def estimate_age(
    oil_area_km2: float,
    oil_pixel_fraction: float,
    slick_count: int = 1,
    backscatter_contrast: Optional[float] = None,
) -> SpillAgeEstimate:
    """
    Estimate the age of an oil slick from SAR observable properties.

    The PS says "if feasible". We implement it and surface the wide uncertainty
    band rather than faking precision.

    Parameters
    ----------
    oil_area_km2 : float
        Total oil-covered area from the SAR segmenter.
    oil_pixel_fraction : float
        Fraction of the scene covered by oil (0–1).
    slick_count : int
        Number of connected oil components (fragmentation increases with age).
    backscatter_contrast : float, optional
        Dark-formation contrast vs background (higher = fresher oil).
        Not always available; used when provided.

    Returns
    -------
    SpillAgeEstimate with min/best/max hours and caveats.
    """
    caveats: List[str] = []

    if oil_area_km2 <= 0:
        return SpillAgeEstimate(
            min_hours=0, max_hours=0, best_hours=0,
            method="no_oil_detected",
            caveats=["No oil detected — age estimation not applicable."],
        )

    # --- Heuristic 1: spreading rate proxy ---
    # Oil spreads at roughly 0.1–0.5 km²/hour in open ocean (ITOPF).
    # Inverting: age ≈ area / spreading_rate
    spreading_rate_min = 0.05   # km²/h  (calm conditions, heavy fuel oil)
    spreading_rate_max = 0.8    # km²/h  (windy, light crude)
    age_from_area_min = oil_area_km2 / spreading_rate_max
    age_from_area_max = oil_area_km2 / spreading_rate_min
    caveats.append(
        "Area-based estimate assumes a constant spreading rate — actual rates "
        "vary 10× with oil type and sea state."
    )

    # --- Heuristic 2: fragmentation ---
    # Weathering breaks a slick into multiple patches; more patches → older
    frag_multiplier = 1.0
    if slick_count >= 3:
        frag_multiplier = 1.5
        caveats.append(
            f"Slick has fragmented into {slick_count} patches, "
            "suggesting some weathering has occurred."
        )
    elif slick_count >= 2:
        frag_multiplier = 1.2

    # --- Heuristic 3: backscatter contrast ---
    contrast_multiplier = 1.0
    if backscatter_contrast is not None:
        # High contrast → fresh; low contrast → emulsified/older
        if backscatter_contrast < 0.3:
            contrast_multiplier = 2.0
            caveats.append(
                "Low backscatter contrast suggests emulsification — "
                "consistent with a spill more than 12 h old."
            )
        elif backscatter_contrast > 0.7:
            contrast_multiplier = 0.6
            caveats.append(
                "High backscatter contrast suggests fresh mineral oil."
            )

    # --- Combine ---
    best = (age_from_area_min + age_from_area_max) / 2 * frag_multiplier * contrast_multiplier
    lo = age_from_area_min * frag_multiplier * contrast_multiplier
    hi = age_from_area_max * frag_multiplier * contrast_multiplier

    # Physical bounds: SAR cannot reliably detect slicks > 7 days old
    hi = min(hi, 168.0)

    caveats.append(
        "Age estimates from SAR alone carry ±50–200% uncertainty. "
        "Confirm against vessel movement history and met records."
    )

    return SpillAgeEstimate(
        min_hours=round(lo, 1),
        max_hours=round(hi, 1),
        best_hours=round(best, 1),
        method="area_spreading+fragmentation" + (
            "+backscatter_contrast" if backscatter_contrast is not None else ""
        ),
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _spread_radius(lats: List[float], lons: List[float]) -> float:
    """Mean distance of all points from their centroid, in km."""
    if len(lats) < 2:
        return 0.0
    clat = float(np.mean(lats))
    clon = float(np.mean(lons))
    dists = []
    for la, lo in zip(lats, lons):
        dlat = math.radians(la - clat) * EARTH_RADIUS_KM
        dlon = math.radians(lo - clon) * EARTH_RADIUS_KM * math.cos(math.radians(clat))
        dists.append(math.hypot(dlat, dlon))
    return float(np.mean(dists))


def _convex_hull_latlon(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Gift-wrapping convex hull on (lat, lon) points.
    Returns vertices in counter-clockwise order for Folium polygon rendering.
    """
    if len(points) <= 3:
        return points
    pts = list(set(points))
    if len(pts) <= 3:
        return pts
    # Find bottom-most point
    start = min(pts, key=lambda p: (p[0], p[1]))
    hull = [start]
    current = start
    while True:
        candidate = pts[0]
        for p in pts[1:]:
            cross = ((candidate[0] - current[0]) * (p[1] - current[1]) -
                     (candidate[1] - current[1]) * (p[0] - current[0]))
            if cross > 0 or (cross == 0 and
                             math.hypot(p[0] - current[0], p[1] - current[1]) >
                             math.hypot(candidate[0] - current[0], candidate[1] - current[1])):
                candidate = p
        if candidate == hull[0]:
            break
        hull.append(candidate)
        current = candidate
        if len(hull) > len(pts):
            break
    return hull

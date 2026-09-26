"""
Trajectory model domain-shift diagnostic.

WHAT THIS IS
--------------
The trajectory LSTM (`trajectory.py`) is normalised specifically for the
Mauritius AOI it was trained on -- applying it anywhere else produces
numbers that look precise but are statistically meaningless, since the
model learned specific coordinate ranges, not a general notion of
"vessel movement." This project's existing policy is therefore to skip
trajectory scoring entirely outside that AOI (see `investigation.py`).

WHAT THIS ADDS
----------------
A genuine domain adaptation layer or full retraining for a second region
would require real AIS trajectory data for that region -- this project has
none, and fabricating a "retrained" model without real data to train on
would be worse than not having one. What IS possible without new data: a
quantified diagnostic of *how far* a candidate location is from the trained
domain, expressed in AOI-diagonal multiples rather than raw degrees (so it's
comparable across latitudes). This is a genuine, if partial, step toward
domain adaptation -- the kind of distance-from-training-distribution check
that would inform where a real adaptation or retraining effort should focus
first -- not a claim that the model becomes usable outside its AOI.

Every place that skips the trajectory model outside its AOI can use this to
say exactly how far outside, rather than a flat "not applicable."
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from .trajectory import haversine_km

_AOI_CENTER_LAT = (LAT_MIN + LAT_MAX) / 2.0
_AOI_CENTER_LON = (LON_MIN + LON_MAX) / 2.0
_AOI_DIAGONAL_KM = haversine_km(LAT_MIN, LON_MIN, LAT_MAX, LON_MAX)


@dataclass
class DomainShiftAssessment:
    within_trained_aoi: bool
    distance_from_aoi_km: float          # 0 if inside the trained AOI
    distance_in_aoi_diagonals: float     # distance expressed as a multiple of the AOI's own size
    risk_level: str                      # "none" | "moderate" | "high" | "severe"
    note: str


def assess_domain_shift(lat: float, lon: float) -> DomainShiftAssessment:
    """
    How far is (lat, lon) from the trajectory model's trained domain
    (the Mauritius AOI), expressed relative to that AOI's own size?

    Distance is measured to the AOI's nearest edge (0 if inside), then
    expressed as a multiple of the AOI's own diagonal extent -- a location
    twice the AOI's diagonal away is a coarser, more comparable notion of
    "how far outside the training distribution" than raw kilometres, which
    means something different near the poles than at the equator.
    """
    within = (LAT_MIN <= lat <= LAT_MAX) and (LON_MIN <= lon <= LON_MAX)
    if within:
        return DomainShiftAssessment(
            within_trained_aoi=True, distance_from_aoi_km=0.0, distance_in_aoi_diagonals=0.0,
            risk_level="none",
            note="Inside the trajectory model's trained AOI -- deviation scoring is meaningful here.",
        )

    clamped_lat = min(max(lat, LAT_MIN), LAT_MAX)
    clamped_lon = min(max(lon, LON_MIN), LON_MAX)
    dist_km = haversine_km(lat, lon, clamped_lat, clamped_lon)
    diagonals = dist_km / _AOI_DIAGONAL_KM if _AOI_DIAGONAL_KM > 0 else float("inf")

    if diagonals <= 2:
        risk, desc = "moderate", "a few AOI-widths"
    elif diagonals <= 10:
        risk, desc = "high", "well"
    else:
        risk, desc = "severe", "hundreds of AOI-widths, likely a different ocean basin,"

    note = (
        f"{dist_km:.0f} km outside the trajectory model's trained AOI "
        f"({diagonals:.1f}\u00d7 the AOI's own diagonal extent -- {desc} outside the training domain). "
        "Trajectory deviation scoring is skipped here: the model's coordinate normalisation was "
        "fit to the Mauritius AOI specifically, and running it this far outside that domain would "
        "produce numbers that look precise but aren't. A real fix requires real AIS trajectory data "
        "for this region to retrain or fine-tune on -- this project has none, so none is fabricated."
    )
    return DomainShiftAssessment(
        within_trained_aoi=False, distance_from_aoi_km=round(dist_km, 1),
        distance_in_aoi_diagonals=round(diagonals, 2), risk_level=risk, note=note,
    )

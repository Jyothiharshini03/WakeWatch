"""
SAR vessel detection -- classical, rule-based, no new ML model.

WHAT THIS DOES
---------------
Identifies discrete vessel-like objects in a Sentinel-1 SAR scene so they can
be cross-checked against AIS (see `wakewatch.dark_vessel`).

Two candidate sources, in order of preference:

  1. **The existing SAR segmentation model's Ship class** (`SHIP_CLASS = 3`
     in `wakewatch.inference.sar`). This is the primary source: the model
     is already trained to separate ship pixels from sea/oil/land/look-alike
     backscatter, which is a strictly better starting mask than a naive
     brightness threshold (which false-positives on sun-glint, oil sheens,
     and bright coastline). No retraining, no new model -- this reuses the
     existing checkpoint's output exactly as `sar_mod.segment()` already
     produces it.

  2. **A classical backscatter threshold** (Otsu, or a configurable
     percentile), used only when the model's ship mask is empty or
     unavailable -- e.g. vessel detection is wanted on its own without
     running the full 5-class segmentation. This is genuinely rule-based
     image processing: threshold -> connected components -> shape/size
     filtering, per the brief. It is documented as the weaker fallback
     because it cannot distinguish a ship from other bright clutter as
     well as the trained classifier can.

Either way, this module turns a binary candidate mask into discrete objects
via connected-component analysis (OpenCV, already a project dependency --
no new dependency introduced), computes shape/intensity properties for each,
and applies configurable filtering. It does not classify every bright pixel
as a vessel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Configuration -- every threshold here is a parameter, not a hardcoded magic
# number buried in the detection logic.
# --------------------------------------------------------------------------
@dataclass
class VesselDetectionConfig:
    min_area_px: int = 4
    max_area_px: int = 4000
    min_aspect_ratio: float = 1.0     # length/width; a vessel is elongated, not round
    max_aspect_ratio: float = 15.0    # beyond this it's more likely a wake/artifact line
    min_intensity: float = 0.0        # 0-255 grayscale, only used for the classical path
    classical_percentile: float = 99.0  # brightness percentile used as the Otsu-alternative cutoff
    max_candidates: int = 200          # hard cap so a noisy image can't blow up downstream matching


@dataclass
class VesselCandidate:
    """One discrete SAR-detected candidate object.

    `latitude`/`longitude` are None whenever the source image has no
    geographic reference -- this module never invents a position.
    """
    id: int
    centroid_xy: Tuple[float, float]        # pixel coords (col, row) in the source image
    bbox_xywh: Tuple[int, int, int, int]     # x, y, w, h in pixels
    area_px: int
    width_px: float
    length_px: float
    aspect_ratio: float
    angle_deg: float
    mean_intensity: float
    max_intensity: float
    source: str                             # "model_ship_class" | "classical_threshold"
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "centroid_xy": self.centroid_xy, "bbox_xywh": self.bbox_xywh,
            "area_px": self.area_px, "width_px": round(self.width_px, 1),
            "length_px": round(self.length_px, 1), "aspect_ratio": round(self.aspect_ratio, 2),
            "angle_deg": round(self.angle_deg, 1), "mean_intensity": round(self.mean_intensity, 1),
            "max_intensity": round(self.max_intensity, 1), "source": self.source,
            "latitude": self.latitude, "longitude": self.longitude,
        }


def _to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim == 2:
        return image_rgb
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def _candidates_from_mask(binary_mask: np.ndarray, gray: np.ndarray, source: str,
                          config: VesselDetectionConfig) -> List[VesselCandidate]:
    """Connected components -> shape/intensity properties -> filtering.

    This is the classical image-processing core the brief asks for,
    regardless of which binary mask fed it.
    """
    binary = (binary_mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: List[VesselCandidate] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 0:
            # Degenerate (1-2px) contours have zero polygon area even though
            # they occupy real pixels -- fall back to the pixel count so
            # single-pixel-wide bright dots aren't silently dropped from the
            # area filter before it even gets a chance to reject them.
            area = float(cv2.countNonZero(cv2.drawContours(
                np.zeros_like(binary), [cnt], -1, 1, thickness=cv2.FILLED)))
        if not (config.min_area_px <= area <= config.max_area_px):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if len(cnt) >= 3:
            rect = cv2.minAreaRect(cnt)
            (rw, rh), angle = rect[1], rect[2]
            length_px, width_px = max(rw, rh), max(min(rw, rh), 1e-6)
        else:
            length_px, width_px, angle = max(w, h), max(min(w, h), 1e-6), 0.0
        aspect = length_px / width_px
        if not (config.min_aspect_ratio <= aspect <= config.max_aspect_ratio):
            continue

        mask_i = np.zeros_like(binary)
        cv2.drawContours(mask_i, [cnt], -1, 1, thickness=cv2.FILLED)
        pixel_vals = gray[mask_i.astype(bool)]
        mean_i = float(pixel_vals.mean()) if pixel_vals.size else 0.0
        max_i = float(pixel_vals.max()) if pixel_vals.size else 0.0
        if source == "classical_threshold" and mean_i < config.min_intensity:
            continue

        M = cv2.moments(cnt)
        cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2.0
        cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2.0

        candidates.append(VesselCandidate(
            id=len(candidates), centroid_xy=(float(cx), float(cy)),
            bbox_xywh=(int(x), int(y), int(w), int(h)), area_px=int(area),
            width_px=float(width_px), length_px=float(length_px), aspect_ratio=float(aspect),
            angle_deg=float(angle), mean_intensity=mean_i, max_intensity=max_i, source=source,
        ))

    candidates.sort(key=lambda c: c.area_px, reverse=True)
    candidates = candidates[:config.max_candidates]
    for i, c in enumerate(candidates):
        c.id = i
    return candidates


def _classical_backscatter_mask(gray: np.ndarray, config: VesselDetectionConfig) -> np.ndarray:
    """Otsu threshold, falling back to a brightness percentile if Otsu
    collapses (e.g. a near-uniform scene with no real contrast)."""
    if float(gray.std()) < 1.0:
        # No real contrast to threshold against -- a blank/uniform image has
        # no vessel signal, not "everything is a vessel." Returning an empty
        # mask here (rather than letting a degenerate percentile cutoff match
        # the whole frame) is what keeps this genuinely rule-based rather
        # than a rubber-stamp.
        return np.zeros_like(gray, dtype=np.uint8)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if otsu.sum() > 0 and otsu.sum() < 0.5 * otsu.size * 255:
        return otsu
    cutoff = np.percentile(gray, config.classical_percentile)
    return (gray >= cutoff).astype(np.uint8) * 255


def detect_vessels(
    image_rgb: np.ndarray,
    ship_mask: Optional[np.ndarray] = None,
    config: Optional[VesselDetectionConfig] = None,
) -> List[VesselCandidate]:
    """
    Detect discrete vessel-like candidates in a SAR scene.

    `ship_mask`, when given, should be a binary mask of the *existing*
    segmentation model's Ship class (`sar_result.mask == sar_mod.SHIP_CLASS`)
    -- this is the preferred, more accurate path since that model was
    trained to separate ships from other bright backscatter. When omitted
    or empty, falls back to a classical Otsu/percentile brightness
    threshold on the grayscale image, which is genuinely rule-based but
    weaker at rejecting non-vessel clutter.
    """
    config = config or VesselDetectionConfig()
    gray = _to_grayscale(image_rgb)

    if ship_mask is not None:
        # An explicit mask -- even an all-zero one -- means the model already
        # made a judgement call about ship pixels; that is respected as-is
        # rather than second-guessed with the noisier classical fallback.
        # Falling through to classical here would mean "the model found no
        # ships" and "we never ran the model" produce different behaviour,
        # which is the wrong coupling -- only a genuinely absent mask should
        # trigger the fallback.
        return _candidates_from_mask(ship_mask, gray, "model_ship_class", config)

    classical_mask = _classical_backscatter_mask(gray, config)
    return _candidates_from_mask(classical_mask, gray, "classical_threshold", config)


# --------------------------------------------------------------------------
# Georeferencing -- pixel (col, row) -> (lat, lon).
# --------------------------------------------------------------------------
def geolocate_candidates(
    candidates: List[VesselCandidate],
    bounds: Dict[str, float],
    image_width: int,
    image_height: int,
) -> List[VesselCandidate]:
    """
    Fill in latitude/longitude for each candidate from the scene's
    geographic bounds (min_lat, max_lat, min_lon, max_lon), assuming a
    linear map from pixel position to geographic position across the
    scene.

    This is a locally-linear (equirectangular) approximation, not a full
    CRS reprojection -- accurate enough at the scale of a single Sentinel-1
    scene (tens of km) for AIS matching at km-scale tolerances, but not a
    substitute for a proper geotransform if sub-pixel accuracy is ever
    needed. Never called, and never invents a position, when `bounds` is
    unavailable -- see `sar_vessel.detect_and_geolocate`.
    """
    min_lat, max_lat = bounds["min_lat"], bounds["max_lat"]
    min_lon, max_lon = bounds["min_lon"], bounds["max_lon"]
    out = []
    for c in candidates:
        col, row = c.centroid_xy
        frac_x = col / max(image_width - 1, 1)
        frac_y = row / max(image_height - 1, 1)
        lon = min_lon + frac_x * (max_lon - min_lon)
        lat = max_lat - frac_y * (max_lat - min_lat)   # row 0 is the top (max_lat)
        c.latitude, c.longitude = float(lat), float(lon)
        out.append(c)
    return out


def detect_and_geolocate(
    image_rgb: np.ndarray,
    ship_mask: Optional[np.ndarray] = None,
    bounds: Optional[Dict[str, float]] = None,
    config: Optional[VesselDetectionConfig] = None,
) -> Tuple[List[VesselCandidate], bool]:
    """
    Convenience wrapper: detect candidates, then geolocate them if -- and
    only if -- real geographic bounds were supplied.

    Returns (candidates, geolocated). When `geolocated` is False, every
    candidate's latitude/longitude stays None; callers must surface that
    plainly (per the brief: never invent vessel coordinates for a plain,
    non-georeferenced image).
    """
    h, w = image_rgb.shape[:2]
    candidates = detect_vessels(image_rgb, ship_mask=ship_mask, config=config)
    if bounds is not None:
        candidates = geolocate_candidates(candidates, bounds, w, h)
        return candidates, True
    return candidates, False

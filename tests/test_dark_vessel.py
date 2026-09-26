"""
Tests for `wakewatch.inference.sar_vessel` and `wakewatch.dark_vessel`.

Covers exactly what the brief calls out: coordinate conversion, spatial
matching, temporal matching, AIS gap calculation, and dark-vessel
classification logic -- plus the honesty requirements that matter most:
never inventing a position, and never calling something a dark vessel when
AIS coverage itself is the unknown.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch import dark_vessel as dv  # noqa: E402
from wakewatch.inference import sar_vessel as sv  # noqa: E402


# --------------------------------------------------------------------------
# SAR vessel detection: connected components, shape filtering
# --------------------------------------------------------------------------
def _blank(h=100, w=100):
    return np.zeros((h, w), dtype=np.uint8)


def test_detect_vessels_finds_a_bright_blob():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:46, 40:50] = 255  # a small bright rectangle, vessel-shaped
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:46, 40:50] = 1
    candidates = sv.detect_vessels(img, ship_mask=mask)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.source == "model_ship_class"
    assert c.area_px > 0
    assert c.aspect_ratio >= 1.0


def test_detect_vessels_finds_nothing_on_empty_mask():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    candidates = sv.detect_vessels(img, ship_mask=mask)
    assert candidates == []


def test_detect_vessels_size_filter_rejects_tiny_and_huge_blobs():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[0, 0] = 1                       # 1px -- should be rejected by min_area
    mask[50:150, 50:150] = 1             # 100x100 -- should be rejected by max_area
    config = sv.VesselDetectionConfig(min_area_px=4, max_area_px=500)
    candidates = sv.detect_vessels(img, ship_mask=mask, config=config)
    assert candidates == []


def test_detect_vessels_classical_fallback_when_no_mask_given():
    img = np.full((60, 60, 3), 20, dtype=np.uint8)
    img[20:26, 20:30] = 250
    candidates = sv.detect_vessels(img, ship_mask=None)
    assert all(c.source == "classical_threshold" for c in candidates)


def test_detect_vessels_respects_max_candidates_cap():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    for i in range(20):
        mask[i * 4:i * 4 + 2, 0:2] = 1
    config = sv.VesselDetectionConfig(min_area_px=1, max_candidates=5)
    candidates = sv.detect_vessels(img, ship_mask=mask, config=config)
    assert len(candidates) <= 5


# --------------------------------------------------------------------------
# Coordinate conversion (pixel -> lat/lon)
# --------------------------------------------------------------------------
def test_geolocate_candidates_maps_corners_correctly():
    c_topleft = sv.VesselCandidate(id=0, centroid_xy=(0, 0), bbox_xywh=(0, 0, 1, 1),
                                   area_px=1, width_px=1, length_px=1, aspect_ratio=1,
                                   angle_deg=0, mean_intensity=0, max_intensity=0, source="test")
    c_bottomright = sv.VesselCandidate(id=1, centroid_xy=(99, 99), bbox_xywh=(0, 0, 1, 1),
                                       area_px=1, width_px=1, length_px=1, aspect_ratio=1,
                                       angle_deg=0, mean_intensity=0, max_intensity=0, source="test")
    bounds = {"min_lat": -20.5, "max_lat": -20.3, "min_lon": 57.7, "max_lon": 57.9}
    out = sv.geolocate_candidates([c_topleft, c_bottomright], bounds, image_width=100, image_height=100)
    # Row 0 (top) must map to max_lat; col 0 (left) must map to min_lon.
    assert out[0].latitude == pytest.approx(-20.3, abs=0.01)
    assert out[0].longitude == pytest.approx(57.7, abs=0.01)
    # Bottom-right maps to min_lat, max_lon.
    assert out[1].latitude == pytest.approx(-20.5, abs=0.01)
    assert out[1].longitude == pytest.approx(57.9, abs=0.01)


def test_detect_and_geolocate_never_invents_coordinates_without_bounds():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:26, 20:30] = 1
    candidates, geolocated = sv.detect_and_geolocate(img, ship_mask=mask, bounds=None)
    assert geolocated is False
    assert all(c.latitude is None and c.longitude is None for c in candidates)


# --------------------------------------------------------------------------
# Spatial + temporal AIS matching
# --------------------------------------------------------------------------
def _ais_df(rows):
    return pd.DataFrame(rows)


def test_match_candidate_to_ais_finds_a_close_match():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ais = _ais_df([
        {"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:05:00Z"),
         "latitude": 10.001, "longitude": 20.001, "vessel_name": "ALPHA"},
        {"mmsi": 2, "timestamp": pd.Timestamp("2024-01-01T12:05:00Z"),
         "latitude": 30.0, "longitude": 40.0, "vessel_name": "FAR_AWAY"},
    ])
    match = dv.match_candidate_to_ais(10.0, 20.0, sar_time, ais, dv.MatchConfig(radius_km=5.0, time_tolerance_minutes=30))
    assert match.matched is True
    assert match.mmsi == 1
    assert match.distance_km < 1.0


def test_match_candidate_to_ais_rejects_out_of_radius():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 11.0, "longitude": 21.0, "vessel_name": "TOO_FAR"}])
    match = dv.match_candidate_to_ais(10.0, 20.0, sar_time, ais, dv.MatchConfig(radius_km=5.0))
    assert match.matched is False


def test_match_candidate_to_ais_rejects_out_of_time_window():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T20:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "vessel_name": "TOO_LATE"}])
    match = dv.match_candidate_to_ais(10.0, 20.0, sar_time, ais, dv.MatchConfig(time_tolerance_minutes=30))
    assert match.matched is False


# --------------------------------------------------------------------------
# Last-known-position search + AIS gap calculation
# --------------------------------------------------------------------------
def test_find_last_known_position_picks_the_nearest_plausible_track():
    sar_time = datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc)
    ais = _ais_df([
        {"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
         "latitude": 10.0, "longitude": 20.0, "speed": 10.0, "course": 90.0, "vessel_name": "NEAR"},
        {"mmsi": 2, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
         "latitude": 50.0, "longitude": 60.0, "speed": 5.0, "course": 0.0, "vessel_name": "FAR"},
    ])
    last = dv.find_last_known_position(10.05, 20.05, sar_time, ais,
                                       dv.MatchConfig(max_plausible_km=150, max_gap_hours=48))
    assert last is not None
    assert last.mmsi == 1
    assert last.gap_hours == pytest.approx(3.0, abs=0.01)


def test_find_last_known_position_excludes_pings_after_sar_time():
    sar_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 1.0, "course": 0.0}])
    last = dv.find_last_known_position(10.0, 20.0, sar_time, ais, dv.MatchConfig())
    assert last is None


def test_find_last_known_position_excludes_implausibly_distant_tracks():
    sar_time = datetime(2024, 1, 1, 15, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 1.0, "course": 0.0}])
    last = dv.find_last_known_position(50.0, 60.0, sar_time, ais, dv.MatchConfig(max_plausible_km=10))
    assert last is None


def test_find_last_known_position_excludes_stale_gaps():
    sar_time = datetime(2024, 1, 5, 15, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 1.0, "course": 0.0}])
    last = dv.find_last_known_position(10.0, 20.0, sar_time, ais, dv.MatchConfig(max_gap_hours=24))
    assert last is None


# --------------------------------------------------------------------------
# Trajectory consistency (classical dead reckoning)
# --------------------------------------------------------------------------
def test_trajectory_consistency_high_when_dead_reckoning_lands_close():
    # Stationary vessel (speed=0): after any gap, dead-reckoned position ==
    # last known position, so it should land exactly where it started.
    last = dv.LastKnownPosition(mmsi=1, vessel_name="X", latitude=10.0, longitude=20.0,
                                timestamp=datetime(2024, 1, 1), speed_knots=0.0, course_deg=0.0,
                                gap_hours=2.0, distance_km=1.0)
    result = dv.assess_trajectory_consistency(last, 10.0, 20.0, dv.MatchConfig())
    assert result.label == "High"
    assert result.dead_reckoning_distance_km < 1.0


def test_trajectory_consistency_low_when_far_from_dead_reckoned_position():
    last = dv.LastKnownPosition(mmsi=1, vessel_name="X", latitude=10.0, longitude=20.0,
                                timestamp=datetime(2024, 1, 1), speed_knots=0.0, course_deg=0.0,
                                gap_hours=2.0, distance_km=1.0)
    # Actual SAR detection far from where a stationary vessel would still be.
    result = dv.assess_trajectory_consistency(last, 12.0, 22.0, dv.MatchConfig())
    assert result.label == "Low"


def test_trajectory_consistency_unknown_for_implausible_gap():
    last = dv.LastKnownPosition(mmsi=1, vessel_name="X", latitude=10.0, longitude=20.0,
                                timestamp=datetime(2024, 1, 1), speed_knots=5.0, course_deg=90.0,
                                gap_hours=999.0, distance_km=1.0)
    result = dv.assess_trajectory_consistency(last, 10.0, 20.0, dv.MatchConfig(max_gap_hours=48))
    assert result.label == "Unknown"


# --------------------------------------------------------------------------
# Dark-vessel classification -- the honesty requirements
# --------------------------------------------------------------------------
def _candidate(lat=None, lon=None):
    return sv.VesselCandidate(id=0, centroid_xy=(1, 1), bbox_xywh=(0, 0, 1, 1), area_px=10,
                              width_px=2, length_px=4, aspect_ratio=2.0, angle_deg=0,
                              mean_intensity=200, max_intensity=255, source="model_ship_class",
                              latitude=lat, longitude=lon)


def test_run_dark_vessel_detection_confirms_a_matching_vessel():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 1.0, "course": 0.0,
                    "vessel_name": "ALPHA"}])
    report = dv.run_dark_vessel_detection([_candidate(10.0, 20.0)], sar_time, ais,
                                          ais_coverage_available=True)
    assert report.candidates[0].status == "ais_confirmed"


def test_run_dark_vessel_detection_flags_unmatched_as_potential_dark_vessel():
    sar_time = datetime(2024, 1, 1, 18, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 0.0, "course": 0.0,
                    "vessel_name": "ALPHA"}])
    report = dv.run_dark_vessel_detection([_candidate(10.0, 20.0)], sar_time, ais,
                                          ais_coverage_available=True,
                                          config=dv.MatchConfig(time_tolerance_minutes=30))
    assert report.candidates[0].status == "potential_dark_vessel"
    assert report.candidates[0].evidence_label in ("HIGH", "MEDIUM", "LOW")
    assert "does not by itself prove" in report.candidates[0].note


def test_run_dark_vessel_detection_never_flags_dark_when_coverage_unavailable():
    """The single most important distinction in the whole feature: no AIS
    source at all must never be presented the same as a genuine mismatch."""
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    report = dv.run_dark_vessel_detection([_candidate(10.0, 20.0)], sar_time, None,
                                          ais_coverage_available=False)
    assert report.candidates[0].status == "ais_coverage_unavailable"
    assert report.candidates[0].status != "potential_dark_vessel"
    assert "coverage unavailable" in report.candidates[0].note.lower()


def test_run_dark_vessel_detection_never_geolocates_ungeoreferenced_candidates():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 0.0, "course": 0.0}])
    report = dv.run_dark_vessel_detection([_candidate(None, None)], sar_time, ais,
                                          ais_coverage_available=True)
    assert report.candidates[0].status == "unverified_low_confidence"


def test_run_dark_vessel_detection_summary_counts_are_consistent():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    ais = _ais_df([{"mmsi": 1, "timestamp": pd.Timestamp("2024-01-01T12:00:00Z"),
                    "latitude": 10.0, "longitude": 20.0, "speed": 0.0, "course": 0.0,
                    "vessel_name": "ALPHA"}])
    report = dv.run_dark_vessel_detection(
        [_candidate(10.0, 20.0), _candidate(80.0, 150.0)], sar_time, ais,
        ais_coverage_available=True)
    statuses = [c.status for c in report.candidates]
    assert "ais_confirmed" in statuses
    assert "potential_dark_vessel" in statuses
    assert str(len(report.candidates)) in report.summary


def test_run_dark_vessel_detection_no_candidates_is_handled_cleanly():
    sar_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)
    report = dv.run_dark_vessel_detection([], sar_time, None, ais_coverage_available=False)
    assert report.candidates == []

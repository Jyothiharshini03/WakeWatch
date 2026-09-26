"""
Tests for the drift hindcast, forecast and age estimation modules.

These tests cover:
  - Hindcast produces a track that moves backward from the observed position
  - Forecast produces an ensemble with growing uncertainty
  - Spill age estimation returns sensible bounds for known inputs
  - Edge cases: zero oil area, single-step hindcast
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from wakewatch import drift as dm

# Mauritius Wakashio grounding position
SPILL_LAT = -20.4442
SPILL_LON = 57.7433
OBS_TIME = datetime(2020, 7, 25, 15, 27, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Hindcast
# ---------------------------------------------------------------------------

class TestHindcast:
    def test_returns_hindcast_result(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=3.0, use_live_wind=False)
        assert isinstance(hc, dm.HindcastResult)

    def test_correct_number_of_steps(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=6.0, dt_hours=1.0, use_live_wind=False)
        assert len(hc.steps) == 6

    def test_first_step_is_observation(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=3.0, use_live_wind=False)
        first = hc.steps[0]
        assert abs(first.latitude - SPILL_LAT) < 1e-5
        assert abs(first.longitude - SPILL_LON) < 1e-5

    def test_track_moves_backward(self):
        """Origin should differ from the observed position."""
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=12.0, use_live_wind=False)
        org = hc.origin_estimate
        assert org is not None
        dist = math.hypot(org.latitude - SPILL_LAT, org.longitude - SPILL_LON)
        assert dist > 1e-4, "Origin should be displaced from observation"

    def test_origin_time_is_earlier(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=6.0, use_live_wind=False)
        assert hc.origin_time < OBS_TIME

    def test_uncertainty_grows(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=6.0, use_live_wind=False)
        uncertainties = [s.uncertainty_km for s in hc.steps]
        # Should be non-decreasing
        assert all(uncertainties[i] <= uncertainties[i+1] + 1e-9
                   for i in range(len(uncertainties)-1))

    def test_summary_keys(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=3.0, use_live_wind=False)
        s = hc.summary()
        for key in ["observed_position", "estimated_origin_lat",
                    "estimated_origin_lon", "estimated_release_time",
                    "max_uncertainty_km", "data_source"]:
            assert key in s

    def test_track_latlon_length(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=4.0, use_live_wind=False)
        track = hc.track_latlon()
        assert len(track) == len(hc.steps)

    def test_single_step(self):
        hc = dm.hindcast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_back=1.0, dt_hours=1.0, use_live_wind=False)
        assert len(hc.steps) == 1


# ---------------------------------------------------------------------------
# Forecast
# ---------------------------------------------------------------------------

class TestForecast:
    def test_returns_forecast_result(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=6, n_ensemble=5, use_live_wind=False)
        assert isinstance(fc, dm.ForecastResult)

    def test_ensemble_size(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=6, n_ensemble=10, use_live_wind=False)
        assert fc.n_members == 10

    def test_each_member_has_correct_steps(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=6, n_ensemble=5, use_live_wind=False)
        for member in fc.ensemble:
            assert len(member) == 6

    def test_centroid_track_length(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=8, n_ensemble=5, use_live_wind=False)
        centroid = fc.centroid_track()
        assert len(centroid) == 8

    def test_members_diverge(self):
        """Ensemble members should not all be identical."""
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=6, n_ensemble=10, use_live_wind=False)
        last_lats = [m[-1].latitude for m in fc.ensemble]
        assert max(last_lats) - min(last_lats) > 1e-6

    def test_uncertainty_grows_in_centroid(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=12, n_ensemble=20, use_live_wind=False)
        centroid = fc.centroid_track()
        # Spread at end should be larger than spread at beginning
        assert centroid[-1].uncertainty_km >= centroid[0].uncertainty_km

    def test_cone_polygon_returns_points(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=6, n_ensemble=10, use_live_wind=False)
        cone = fc.cone_polygon(3)
        assert len(cone) >= 3

    def test_summary_keys(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=6, n_ensemble=5, use_live_wind=False)
        s = fc.summary()
        for key in ["start_position", "forecast_hours", "final_lat",
                    "final_lon", "ensemble_members"]:
            assert key in s

    def test_start_position_correct(self):
        fc = dm.forecast(SPILL_LAT, SPILL_LON, OBS_TIME,
                         hours_forward=3, n_ensemble=5, use_live_wind=False)
        assert abs(fc.start_lat - SPILL_LAT) < 1e-9
        assert abs(fc.start_lon - SPILL_LON) < 1e-9


# ---------------------------------------------------------------------------
# Spill age estimation
# ---------------------------------------------------------------------------

class TestSpillAge:
    def test_no_oil_returns_zero(self):
        age = dm.estimate_age(oil_area_km2=0.0, oil_pixel_fraction=0.0)
        assert age.min_hours == 0
        assert age.max_hours == 0
        assert age.method == "no_oil_detected"

    def test_small_fresh_spill(self):
        age = dm.estimate_age(oil_area_km2=0.1, oil_pixel_fraction=0.01, slick_count=1)
        assert age.min_hours >= 0
        assert age.max_hours > age.min_hours
        assert age.best_hours >= age.min_hours

    def test_large_spill_is_older(self):
        age_small = dm.estimate_age(oil_area_km2=0.5, oil_pixel_fraction=0.01)
        age_large = dm.estimate_age(oil_area_km2=5.0, oil_pixel_fraction=0.05)
        assert age_large.best_hours > age_small.best_hours

    def test_fragmentation_increases_age(self):
        age_1 = dm.estimate_age(oil_area_km2=2.0, oil_pixel_fraction=0.02, slick_count=1)
        age_3 = dm.estimate_age(oil_area_km2=2.0, oil_pixel_fraction=0.02, slick_count=3)
        assert age_3.best_hours > age_1.best_hours

    def test_high_contrast_reduces_age(self):
        age_low = dm.estimate_age(oil_area_km2=2.0, oil_pixel_fraction=0.02,
                                   backscatter_contrast=0.1)
        age_high = dm.estimate_age(oil_area_km2=2.0, oil_pixel_fraction=0.02,
                                    backscatter_contrast=0.9)
        assert age_high.best_hours < age_low.best_hours

    def test_label_fresh(self):
        age = dm.estimate_age(oil_area_km2=0.05, oil_pixel_fraction=0.001)
        assert "Fresh" in age.label or "Recent" in age.label

    def test_max_hours_capped(self):
        # Very large spill shouldn't exceed 7 days (168 h)
        age = dm.estimate_age(oil_area_km2=1000.0, oil_pixel_fraction=0.5)
        assert age.max_hours <= 168.0

    def test_caveats_present(self):
        age = dm.estimate_age(oil_area_km2=2.0, oil_pixel_fraction=0.02)
        assert len(age.caveats) > 0

    def test_as_dict_keys(self):
        age = dm.estimate_age(oil_area_km2=2.0, oil_pixel_fraction=0.02)
        d = age.as_dict()
        for key in ["min_hours", "max_hours", "best_hours", "label", "method", "caveats"]:
            assert key in d

    def test_wakashio_spill(self):
        """Wakashio spill: ~4.28 km², 3 slick components."""
        age = dm.estimate_age(oil_area_km2=4.28, oil_pixel_fraction=0.03, slick_count=3)
        # Should be 'Recent' or 'Aged' — not 'Fresh'
        assert "Fresh" not in age.label
        assert age.best_hours > 6


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

class TestGeometry:
    def test_advance_moves_north(self):
        lat, lon = dm._advance(-20.0, 57.0, u_ms=0.0, v_ms=1.0, dt_seconds=3600)
        assert lat > -20.0
        assert abs(lon - 57.0) < 1e-6

    def test_advance_moves_east(self):
        lat, lon = dm._advance(-20.0, 57.0, u_ms=1.0, v_ms=0.0, dt_seconds=3600)
        assert lon > 57.0
        assert abs(lat - (-20.0)) < 1e-6

    def test_leeway_nonzero_for_nonzero_wind(self):
        u, v = dm._leeway_velocity(5.0, 0.0)
        assert math.hypot(u, v) > 0

    def test_leeway_zero_for_zero_wind(self):
        u, v = dm._leeway_velocity(0.0, 0.0)
        assert u == 0.0 and v == 0.0

    def test_spread_radius_zero_for_single_point(self):
        r = dm._spread_radius([-20.0], [57.0])
        assert r == 0.0

    def test_convex_hull_returns_subset(self):
        pts = [(-20.0, 57.0), (-20.1, 57.1), (-20.05, 57.05),
               (-19.9, 57.2), (-20.2, 56.9)]
        hull = dm._convex_hull_latlon(pts)
        assert len(hull) >= 3
        assert len(hull) <= len(pts)

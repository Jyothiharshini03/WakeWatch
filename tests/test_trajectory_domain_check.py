"""Tests for `wakewatch.inference.trajectory_domain_check`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch.inference import trajectory_domain_check as tdc  # noqa: E402
from wakewatch.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX  # noqa: E402


def test_inside_aoi_reports_zero_shift():
    mid_lat, mid_lon = (LAT_MIN + LAT_MAX) / 2, (LON_MIN + LON_MAX) / 2
    r = tdc.assess_domain_shift(mid_lat, mid_lon)
    assert r.within_trained_aoi is True
    assert r.distance_from_aoi_km == 0.0
    assert r.risk_level == "none"


def test_far_away_location_is_severe():
    r = tdc.assess_domain_shift(19.0760, 72.8777)  # Mumbai
    assert r.within_trained_aoi is False
    assert r.risk_level == "severe"
    assert r.distance_from_aoi_km > 1000


def test_just_outside_edge_is_moderate():
    r = tdc.assess_domain_shift(LAT_MIN - 0.05, (LON_MIN + LON_MAX) / 2)
    assert r.within_trained_aoi is False
    assert r.risk_level in ("moderate", "high")  # small offset -> not severe
    assert r.distance_from_aoi_km < 50


def test_note_grammar_is_coherent_not_garbled():
    """Regression guard for the garbled-sentence bug caught during review."""
    r = tdc.assess_domain_shift(19.0760, 72.8777)
    assert " the training domain)" in r.note
    assert "the training domain the training domain" not in r.note


def test_note_never_claims_the_model_is_valid_outside_aoi():
    for lat, lon in [(19.0, 72.0), (0.0, 0.0), (-89.0, 179.0)]:
        r = tdc.assess_domain_shift(lat, lon)
        if not r.within_trained_aoi:
            assert "skipped" in r.note.lower()
            assert "fabricated" in r.note.lower() or "none is fabricated" in r.note.lower()


def test_distance_in_diagonals_scales_with_raw_distance():
    near = tdc.assess_domain_shift(LAT_MIN - 0.1, LON_MIN)
    far = tdc.assess_domain_shift(0.0, 0.0)
    assert far.distance_in_aoi_diagonals > near.distance_in_aoi_diagonals

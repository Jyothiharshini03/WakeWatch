"""Tests for `wakewatch.drift_validation`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch import drift_validation as dv  # noqa: E402


def test_run_validation_produces_a_result():
    result = dv.run_validation(hours_forward=6.0)
    assert result.data_source
    assert isinstance(result.direction_matches_published_range, bool)
    assert result.predicted_distance_km >= 0


def test_validation_never_hides_a_mismatch():
    """The whole point of this module is honesty when the model disagrees
    with reality -- a mismatch must be reported as False, not silently
    coerced toward True."""
    result = dv.run_validation(hours_forward=6.0)
    lo, hi = dv.REAL_DRIFT_BEARING_RANGE_DEG
    expected = lo <= result.predicted_bearing_deg <= hi
    assert result.direction_matches_published_range == expected


def test_synthetic_fallback_note_present_when_offline():
    result = dv.run_validation(hours_forward=6.0)
    if result.data_source == "synthetic_wind+synthetic_current":
        assert "SYNTHETIC" in result.note or "synthetic" in result.note.lower()
        assert "re-run" in result.note.lower()


def test_report_contains_citations_and_real_dates():
    result = dv.run_validation(hours_forward=6.0)
    report = dv.render_report(result)
    assert "sciencedirect.com" in report or "springer.com" in report
    assert "2020-08-06" in report
    assert "25 Jul" in report  # explicitly distinguishes leak date from grounding date


def test_wreck_coordinates_match_the_projects_own_scenario():
    """Sanity check: the validation's wreck position should be the same real
    location the project's own demo scenario uses, not a different point."""
    from wakewatch.scenario.real_ais import build_scenario
    sc = build_scenario()
    spill_lat, spill_lon = sc["spill_position"]
    assert abs(dv.WRECK_LAT - spill_lat) < 0.01
    assert abs(dv.WRECK_LON - spill_lon) < 0.01


def test_real_leak_date_is_after_grounding_date():
    """The oil leak (6 Aug) genuinely postdates the grounding (25 Jul) --
    guards against ever conflating the two in this module."""
    from wakewatch.scenario.real_ais import build_scenario
    sc = build_scenario()
    assert dv.REAL_LEAK_START > sc["grounding_utc"]

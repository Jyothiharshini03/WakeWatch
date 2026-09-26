"""
Tests for `wakewatch.pre_spill_dark` -- Scenario 2 (pre-spill AIS gap
analysis) and the Scenario 1 + Scenario 2 merge logic.

Synthetic data is used here, in unit tests, exactly where the brief allows
it. `tests/test_investigation.py` and the real-data checks documented in
`CHANGES.md` cover the real, unmodified Wakashio AIS feed.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch import pre_spill_dark as psd  # noqa: E402
from wakewatch import dark_vessel as dv  # noqa: E402
from wakewatch.inference import sar_vessel as sv  # noqa: E402


def _row(mmsi, ts, lat, lon, speed=10.0, course=90.0, name="TESTVESSEL"):
    return {"mmsi": mmsi, "timestamp": pd.Timestamp(ts), "latitude": lat, "longitude": lon,
           "speed": speed, "course": course, "vessel_name": name}


def _df(rows):
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 1. AIS gap detection
# --------------------------------------------------------------------------
def test_find_ais_gaps_detects_a_significant_gap():
    rows = [
        _row(1, "2024-01-01T09:10:00Z", 10.0, 20.0),
        _row(1, "2024-01-01T09:20:00Z", 10.01, 20.01),
        _row(1, "2024-01-01T09:40:00Z", 10.02, 20.02),
        _row(1, "2024-01-01T10:45:00Z", 10.03, 20.03),   # 65 min gap
    ]
    gaps = psd.find_ais_gaps(_df(rows), psd.PreSpillConfig(min_dark_gap_minutes=30))
    assert len(gaps) == 1
    assert gaps[0].gap_duration_minutes == pytest.approx(65.0, abs=0.1)
    assert gaps[0].resumed is True


# --------------------------------------------------------------------------
# 2. Small gaps ignored
# --------------------------------------------------------------------------
def test_find_ais_gaps_ignores_normal_reporting_cadence():
    rows = [_row(1, f"2024-01-01T09:{m:02d}:00Z", 10.0, 20.0) for m in range(0, 30, 5)]
    gaps = psd.find_ais_gaps(_df(rows), psd.PreSpillConfig(min_dark_gap_minutes=30))
    assert gaps == []


# --------------------------------------------------------------------------
# 3. Pre-spill gap filtering
# --------------------------------------------------------------------------
def test_filter_pre_spill_gaps_keeps_gaps_before_release():
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 45, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc), gap_duration_minutes=135,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=10, last_course_deg=90)
    relevant = psd.filter_pre_spill_gaps([gap], release_time, psd.PreSpillConfig(pre_spill_window_hours=12))
    assert relevant == [gap]


# --------------------------------------------------------------------------
# 4. Post-spill irrelevant gaps ignored
# --------------------------------------------------------------------------
def test_filter_pre_spill_gaps_excludes_gap_long_after_release():
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 18, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 19, 0, tzinfo=timezone.utc), gap_duration_minutes=60,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=10, last_course_deg=90)
    relevant = psd.filter_pre_spill_gaps([gap], release_time,
                                         psd.PreSpillConfig(pre_spill_window_hours=12, after_release_slack_hours=1))
    assert relevant == []


def test_filter_pre_spill_gaps_excludes_gap_too_far_before_window():
    release_time = datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc)
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 2, 0, tzinfo=timezone.utc), gap_duration_minutes=60,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=10, last_course_deg=90)
    relevant = psd.filter_pre_spill_gaps([gap], release_time, psd.PreSpillConfig(pre_spill_window_hours=12))
    assert relevant == []


# --------------------------------------------------------------------------
# 5. Gap duration calculation
# --------------------------------------------------------------------------
def test_calculate_gap_duration():
    t1 = datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc)
    t2 = datetime(2024, 1, 1, 10, 45, tzinfo=timezone.utc)
    assert psd.calculate_gap_duration(t1, t2) == pytest.approx(65.0)


# --------------------------------------------------------------------------
# 6. Haversine/geodesic distance
# --------------------------------------------------------------------------
def test_calculate_origin_distance_uses_great_circle_not_naive_subtraction():
    # 1 degree of longitude at the equator is ~111 km; a naive subtraction
    # of raw degrees would give "1.0", not a distance in km.
    d = psd.calculate_origin_distance(0.0, 0.0, 0.0, 1.0)
    assert 105 < d < 115


# --------------------------------------------------------------------------
# 7. Physical reachability
# --------------------------------------------------------------------------
def test_reachability_plausible_within_range():
    check = psd.check_physical_reachability(last_speed_knots=12.0, gap_hours=1.5, distance_to_origin_km=10.0,
                                            config=psd.PreSpillConfig())
    assert check.plausible is True
    assert check.max_reachable_km == pytest.approx(12.0 * 1.852 * 1.5, rel=0.01)


# --------------------------------------------------------------------------
# 8. Impossible travel distance
# --------------------------------------------------------------------------
def test_reachability_implausible_when_too_far():
    check = psd.check_physical_reachability(last_speed_knots=10.0, gap_hours=1.0, distance_to_origin_km=100.0,
                                            config=psd.PreSpillConfig())
    assert check.plausible is False


def test_reachability_uses_generous_cap_when_last_speed_near_zero():
    check = psd.check_physical_reachability(last_speed_knots=0.1, gap_hours=1.0, distance_to_origin_km=20.0,
                                            config=psd.PreSpillConfig(max_reachability_speed_knots=25.0))
    assert check.effective_speed_knots == 25.0
    assert check.plausible is True


# --------------------------------------------------------------------------
# 9. Time consistency
# --------------------------------------------------------------------------
def test_time_consistency_high_when_overlapping():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 45, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc), gap_duration_minutes=135,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=10, last_course_deg=90)
    result = psd.evaluate_time_consistency(gap, datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc), psd.PreSpillConfig())
    assert result.label == "HIGH"


def test_time_consistency_inconsistent_when_far_from_gap():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 45, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc), gap_duration_minutes=15,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=10, last_course_deg=90)
    result = psd.evaluate_time_consistency(gap, datetime(2024, 1, 2, 10, 30, tzinfo=timezone.utc),
                                           psd.PreSpillConfig(pre_spill_window_hours=12))
    assert result.label == "INCONSISTENT"


# --------------------------------------------------------------------------
# 10. Trajectory consistency
# --------------------------------------------------------------------------
def test_trajectory_consistency_high_for_stationary_vessel_near_origin():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc), gap_duration_minutes=60,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=0.0, last_course_deg=0.0)
    result = psd.estimate_gap_trajectory(gap, 10.0, 20.0, psd.PreSpillConfig())
    assert result.label == "HIGH"


def test_trajectory_consistency_low_when_far_from_origin():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc), gap_duration_minutes=60,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=0.0, last_course_deg=0.0)
    result = psd.estimate_gap_trajectory(gap, 12.0, 22.0, psd.PreSpillConfig())
    assert result.label == "LOW"


def test_trajectory_consistency_bands_widen_with_origin_uncertainty():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc), gap_duration_minutes=60,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=0.0, last_course_deg=0.0)
    tight = psd.estimate_gap_trajectory(gap, 10.05, 20.0, psd.PreSpillConfig(), origin_uncertainty_km=0.0)
    wide = psd.estimate_gap_trajectory(gap, 10.05, 20.0, psd.PreSpillConfig(), origin_uncertainty_km=50.0)
    assert psd._LABEL_RANK[wide.label] >= psd._LABEL_RANK[tight.label]


# --------------------------------------------------------------------------
# 11. Last AIS position extraction
# --------------------------------------------------------------------------
def test_gap_carries_last_known_position():
    rows = [_row(1, "2024-01-01T09:40:00Z", 10.0, 20.0, speed=12.0, course=45.0),
           _row(1, "2024-01-01T10:45:00Z", 10.5, 20.5)]
    gaps = psd.find_ais_gaps(_df(rows), psd.PreSpillConfig(min_dark_gap_minutes=30))
    assert gaps[0].last_lat == 10.0 and gaps[0].last_lon == 20.0
    assert gaps[0].last_speed_knots == 12.0 and gaps[0].last_course_deg == 45.0


# --------------------------------------------------------------------------
# 12. Next AIS position extraction
# --------------------------------------------------------------------------
def test_gap_carries_next_known_position_when_resumed():
    rows = [_row(1, "2024-01-01T09:40:00Z", 10.0, 20.0),
           _row(1, "2024-01-01T10:45:00Z", 10.5, 20.5)]
    gaps = psd.find_ais_gaps(_df(rows), psd.PreSpillConfig(min_dark_gap_minutes=30))
    assert gaps[0].next_lat == 10.5 and gaps[0].next_lon == 20.5
    assert gaps[0].resumed is True


def test_gap_has_no_next_position_when_vessel_never_resumes():
    rows = [_row(1, "2024-01-01T09:00:00Z", 10.0, 20.0),
           _row(2, "2024-01-01T12:00:00Z", 50.0, 60.0)]   # another vessel keeps the feed alive later
    gaps = psd.find_ais_gaps(_df(rows), psd.PreSpillConfig(min_dark_gap_minutes=30))
    vessel1_gap = [g for g in gaps if g.mmsi == 1][0]
    assert vessel1_gap.resumed is False
    assert vessel1_gap.gap_end is None
    assert vessel1_gap.next_lat is None


def test_next_position_consistency_uses_both_endpoints():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 0, tzinfo=timezone.utc), gap_duration_minutes=120,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=5.0, last_course_deg=90.0,
                     next_lat=10.0, next_lon=20.2)
    check = psd.check_next_position_consistency(
        gap, datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc), 10.0, 20.1, psd.PreSpillConfig())
    assert check is not None
    assert check.label in ("HIGH", "MEDIUM", "LOW")


def test_next_position_consistency_none_without_resumption():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 0, tzinfo=timezone.utc),
                     gap_end=None, gap_duration_minutes=120, last_lat=10.0, last_lon=20.0,
                     last_speed_knots=5.0, last_course_deg=90.0, resumed=False)
    check = psd.check_next_position_consistency(
        gap, datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc), 10.0, 20.1, psd.PreSpillConfig())
    assert check is None


# --------------------------------------------------------------------------
# 13. No AIS coverage handling
# --------------------------------------------------------------------------
def test_analyze_pre_spill_dark_vessels_no_coverage_never_flags_dark():
    report = psd.analyze_pre_spill_dark_vessels(
        None, 10.0, 20.0, datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc), ais_coverage_available=False)
    assert report.candidates == []
    assert report.ais_coverage_available is False
    assert "unavailable" in report.summary.lower()


# --------------------------------------------------------------------------
# 14. Insufficient history handling
# --------------------------------------------------------------------------
def test_analyze_pre_spill_dark_vessels_counts_insufficient_history():
    rows = [_row(1, "2024-01-01T09:00:00Z", 10.0, 20.0)]   # single ping, no gap computable
    report = psd.analyze_pre_spill_dark_vessels(
        _df(rows), 10.0, 20.0, datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc), ais_coverage_available=True)
    assert report.vessels_with_insufficient_history == 1
    assert report.candidates == []


# --------------------------------------------------------------------------
# 15. Candidate evidence classification
# --------------------------------------------------------------------------
def test_score_candidate_classifies_a_strong_match_as_high():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 45, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 15, tzinfo=timezone.utc), gap_duration_minutes=90,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=0.5, last_course_deg=0.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    candidate = psd.score_pre_spill_dark_candidate(gap, 10.0, 20.0, release_time, psd.PreSpillConfig())
    assert candidate.status == "potential_pre_spill_gap"
    assert candidate.evidence_label == "HIGH"
    assert "investigative lead" in candidate.note
    assert "AIS shutdown" in candidate.note


def test_score_candidate_never_says_confirmed_or_responsible():
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 9, 45, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 15, tzinfo=timezone.utc), gap_duration_minutes=90,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=0.5, last_course_deg=0.0)
    candidate = psd.score_pre_spill_dark_candidate(gap, 10.0, 20.0,
                                                   datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc),
                                                   psd.PreSpillConfig())
    text = candidate.note.lower()
    assert "confirmed dark vessel" not in text
    assert "responsible vessel" not in text
    assert "illegal" not in text


def test_score_candidate_weak_trajectory_cannot_be_masked_to_high():
    """The regression this test guards: strong reachability + strong timing
    alone must not average into HIGH when the actual dead-reckoned
    trajectory lands far from the origin."""
    gap = psd.AisGap(mmsi=1, vessel_name="X", gap_start=datetime(2024, 1, 1, 8, 0, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc), gap_duration_minutes=240,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=0.0)  # heads due north
    # Origin is due EAST, not north -- dead reckoning will miss it by a lot,
    # even though reachability (fast + long gap) and timing (overlaps) are strong.
    release_time = datetime(2024, 1, 1, 10, 0, tzinfo=timezone.utc)
    candidate = psd.score_pre_spill_dark_candidate(gap, 10.0, 22.0, release_time, psd.PreSpillConfig())
    assert candidate.gap_trajectory.label == "LOW"
    assert candidate.evidence_label != "HIGH"


# --------------------------------------------------------------------------
# 16. Scenario 1 + Scenario 2 duplicate candidate merging
# --------------------------------------------------------------------------
def _s1_candidate_with_mmsi(mmsi, vessel_name="SHARED", evidence="HIGH"):
    last_known = dv.LastKnownPosition(mmsi=mmsi, vessel_name=vessel_name, latitude=10.0, longitude=20.0,
                                      timestamp=datetime(2024, 1, 1, 9, 0), speed_knots=5.0, course_deg=90.0,
                                      gap_hours=1.0, distance_km=2.0)
    candidate = sv.VesselCandidate(id=0, centroid_xy=(1, 1), bbox_xywh=(0, 0, 1, 1), area_px=10, width_px=2,
                                   length_px=4, aspect_ratio=2.0, angle_deg=0, mean_intensity=200,
                                   max_intensity=255, source="model_ship_class", latitude=10.0, longitude=20.0)
    return dv.DarkVesselCandidate(candidate=candidate, status="potential_dark_vessel", ais_match=dv.AisMatch(False),
                                  last_known=last_known, trajectory=None, evidence_label=evidence,
                                  evidence_components={}, note="scenario1 note")


def _s2_candidate_with_mmsi(mmsi, vessel_name="SHARED", evidence="MEDIUM"):
    gap = psd.AisGap(mmsi=mmsi, vessel_name=vessel_name, gap_start=datetime(2024, 1, 1, 9, 0),
                     gap_end=datetime(2024, 1, 1, 11, 0), gap_duration_minutes=120,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=5.0, last_course_deg=90.0)
    return psd.PreSpillDarkCandidate(
        gap=gap, distance_to_origin_km=2.0,
        reachability=psd.ReachabilityCheck(True, 10.0, 2.0, 5.0, "note"),
        gap_trajectory=psd.GapTrajectoryEstimate("MEDIUM", 10.0, 20.0, 8.0, "note"),
        time_consistency=psd.TimeConsistency("HIGH", "note"), next_position=None,
        status="potential_pre_spill_gap", evidence_label=evidence, evidence_components={}, note="scenario2 note")


def test_merge_combines_same_mmsi_into_one_candidate_not_two():
    s1 = [_s1_candidate_with_mmsi(999)]
    s2 = [_s2_candidate_with_mmsi(999)]
    merged = psd.merge_dark_vessel_evidence(s1, s2)
    assert len(merged) == 1
    combined = merged[0]
    assert combined.mmsi == 999
    assert set(combined.sources) == {"scenario1_sar_ais_mismatch", "scenario2_pre_spill_gap"}
    assert "Multi-source" in combined.note


def test_merge_keeps_different_mmsis_separate():
    s1 = [_s1_candidate_with_mmsi(111)]
    s2 = [_s2_candidate_with_mmsi(222)]
    merged = psd.merge_dark_vessel_evidence(s1, s2)
    assert len(merged) == 2
    mmsis = {m.mmsi for m in merged}
    assert mmsis == {111, 222}


def test_merge_scenario1_only_candidate_with_no_last_known_stays_unlinked():
    candidate = sv.VesselCandidate(id=0, centroid_xy=(1, 1), bbox_xywh=(0, 0, 1, 1), area_px=10, width_px=2,
                                   length_px=4, aspect_ratio=2.0, angle_deg=0, mean_intensity=200,
                                   max_intensity=255, source="model_ship_class", latitude=10.0, longitude=20.0)
    s1_no_last_known = dv.DarkVesselCandidate(candidate=candidate, status="potential_dark_vessel",
                                              ais_match=dv.AisMatch(False), last_known=None, trajectory=None,
                                              evidence_label="MEDIUM", evidence_components={}, note="x")
    merged = psd.merge_dark_vessel_evidence([s1_no_last_known], [])
    assert len(merged) == 1
    assert merged[0].mmsi is None


def test_merge_combined_label_takes_the_stronger_of_the_two():
    s1 = [_s1_candidate_with_mmsi(999, evidence="LOW")]
    s2 = [_s2_candidate_with_mmsi(999, evidence="HIGH")]
    merged = psd.merge_dark_vessel_evidence(s1, s2)
    assert merged[0].combined_evidence_label == "HIGH"


def test_merge_never_produces_more_candidates_than_the_union_of_mmsis():
    s1 = [_s1_candidate_with_mmsi(1), _s1_candidate_with_mmsi(2)]
    s2 = [_s2_candidate_with_mmsi(2), _s2_candidate_with_mmsi(3)]
    merged = psd.merge_dark_vessel_evidence(s1, s2)
    mmsis = [m.mmsi for m in merged]
    assert sorted(mmsis) == [1, 2, 3]
    assert len(mmsis) == len(set(mmsis))


# ==========================================================================
# CORRECTION: physical reachability must use release time, not full gap
# duration. Tests 1-6 below are the specific cases requested for this fix.
# ==========================================================================

# --------------------------------------------------------------------------
# TEST 1: reachability uses time-to-release (50 min), not the full gap (2h05m)
# --------------------------------------------------------------------------
def test_fix1_reachability_uses_time_to_release_not_full_gap():
    """Last AIS 09:40, release 10:30, next AIS 11:45. The vessel only had
    until 10:30 (50 min) to reach the origin -- not until 11:45 (2h05m)."""
    gap = psd.AisGap(mmsi=1, vessel_name="X",
                     gap_start=datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 45, tzinfo=timezone.utc),
                     gap_duration_minutes=125,  # 2h05m -- must NOT be the value used for reachability
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=90.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)

    travel_time_hours = psd._travel_time_to_release_hours(gap, release_time)
    assert travel_time_hours == pytest.approx(50 / 60, abs=0.01)
    assert travel_time_hours != pytest.approx(125 / 60, abs=0.01)

    # And the full candidate scoring pipeline must use that same 50-minute
    # figure for its reachability check, not the 2h05m gap duration.
    candidate = psd.score_pre_spill_dark_candidate(gap, 10.0, 20.0, release_time, psd.PreSpillConfig())
    expected_max_km = 12.0 * psd.KNOTS_TO_KM_H * (50 / 60)
    assert candidate.reachability.max_reachable_km == pytest.approx(expected_max_km, rel=0.02)
    # Sanity check: this is meaningfully less than what the full 2h05m gap would allow.
    full_gap_km = 12.0 * psd.KNOTS_TO_KM_H * (125 / 60)
    assert candidate.reachability.max_reachable_km < full_gap_km


# --------------------------------------------------------------------------
# TEST 2: vessel CAN reach origin before release -> physically plausible
# --------------------------------------------------------------------------
def test_fix1_physically_plausible_when_reachable_before_release():
    gap = psd.AisGap(mmsi=1, vessel_name="X",
                     gap_start=datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 45, tzinfo=timezone.utc), gap_duration_minutes=125,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=90.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    # 50 minutes at 12 kn ~= 18.5 km max reach; origin well within that.
    candidate = psd.score_pre_spill_dark_candidate(gap, 10.0, 20.05, release_time, psd.PreSpillConfig())
    assert candidate.reachability.plausible is True
    assert candidate.status == "potential_pre_spill_gap"


# --------------------------------------------------------------------------
# TEST 3: vessel CANNOT reach origin before release -> physically inconsistent
# --------------------------------------------------------------------------
def test_fix1_physically_inconsistent_when_unreachable_before_release():
    gap = psd.AisGap(mmsi=1, vessel_name="X",
                     gap_start=datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 45, tzinfo=timezone.utc), gap_duration_minutes=125,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=90.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    # Origin is 100 km away, but only 50 min (~18.5 km max) is available before release.
    far_lat, far_lon = 10.0, 21.0   # ~110 km east at this latitude
    candidate = psd.score_pre_spill_dark_candidate(gap, far_lat, far_lon, release_time, psd.PreSpillConfig())
    assert candidate.reachability.plausible is False
    assert candidate.status == "physically_inconsistent"
    assert candidate.evidence_label == "INCONSISTENT"


# --------------------------------------------------------------------------
# TEST 4: release-time uncertainty is respected when available
# --------------------------------------------------------------------------
def test_fix1_release_time_uncertainty_widens_reachability():
    """A vessel unreachable by the point-estimate release time may become
    plausible once a real (not invented) uncertainty window is applied."""
    gap = psd.AisGap(mmsi=1, vessel_name="X",
                     gap_start=datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 45, tzinfo=timezone.utc), gap_duration_minutes=125,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=90.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    origin_lat, origin_lon = 10.0, 20.08   # ~8.9 km east -- unreachable in 50 min (~18.5km ok actually)
    # Use a distance that's unreachable at 50 min but reachable at 50+60=110 min.
    origin_lat, origin_lon = 10.0, 20.2    # ~22.2 km east

    no_uncertainty = psd.score_pre_spill_dark_candidate(gap, origin_lat, origin_lon, release_time,
                                                        psd.PreSpillConfig())
    assert no_uncertainty.reachability.plausible is False

    with_uncertainty = psd.score_pre_spill_dark_candidate(
        gap, origin_lat, origin_lon, release_time, psd.PreSpillConfig(),
        release_time_uncertainty_hours=1.0)
    assert with_uncertainty.reachability.plausible is True
    assert with_uncertainty.reachability.max_reachable_km > no_uncertainty.reachability.max_reachable_km


def test_fix1_uncertainty_not_invented_when_not_supplied():
    """No uncertainty value given -> behaves exactly as the point estimate,
    never silently assumes one."""
    gap = psd.AisGap(mmsi=1, vessel_name="X",
                     gap_start=datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 45, tzinfo=timezone.utc), gap_duration_minutes=125,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=90.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    t1 = psd._travel_time_to_release_hours(gap, release_time, release_time_uncertainty_hours=None)
    t2 = psd._travel_time_to_release_hours(gap, release_time, release_time_uncertainty_hours=0.0)
    assert t1 == pytest.approx(50 / 60, abs=0.01)
    assert t2 == pytest.approx(50 / 60, abs=0.01)


# --------------------------------------------------------------------------
# TEST 5: Scenario 1 + Scenario 2 duplicate vessels merged (corrected pipeline)
# --------------------------------------------------------------------------
def test_fix1_merge_still_works_with_corrected_scoring():
    """The merge logic is untouched by this fix, but re-verified end to end
    using the corrected score_pre_spill_dark_candidate output."""
    gap = psd.AisGap(mmsi=777, vessel_name="SHARED",
                     gap_start=datetime(2024, 1, 1, 9, 40, tzinfo=timezone.utc),
                     gap_end=datetime(2024, 1, 1, 11, 45, tzinfo=timezone.utc), gap_duration_minutes=125,
                     last_lat=10.0, last_lon=20.0, last_speed_knots=12.0, last_course_deg=90.0)
    release_time = datetime(2024, 1, 1, 10, 30, tzinfo=timezone.utc)
    s2_candidate = psd.score_pre_spill_dark_candidate(gap, 10.0, 20.05, release_time, psd.PreSpillConfig())
    s1_candidate = _s1_candidate_with_mmsi(777, vessel_name="SHARED")

    merged = psd.merge_dark_vessel_evidence([s1_candidate], [s2_candidate])
    assert len(merged) == 1
    assert merged[0].mmsi == 777
    assert len(merged[0].sources) == 2
    assert "Multi-source" in merged[0].note


# --------------------------------------------------------------------------
# TEST 6: existing LSTM behaviour (Scenario 1) is unchanged by this fix
# --------------------------------------------------------------------------
def test_fix1_scenario1_lstm_behaviour_unchanged():
    """This correction touches only wakewatch/pre_spill_dark.py. Scenario 1's
    dead-reckoning + optional-LSTM-reference logic in dark_vessel.py must be
    byte-for-byte behaviourally identical: dead reckoning remains the primary
    full-gap estimate, and any LSTM prediction is still reference-only."""
    last = dv.LastKnownPosition(mmsi=1, vessel_name="X", latitude=10.0, longitude=20.0,
                                timestamp=datetime(2024, 1, 1), speed_knots=0.0, course_deg=0.0,
                                gap_hours=2.0, distance_km=1.0)
    result = dv.assess_trajectory_consistency(last, 10.0, 20.0, dv.MatchConfig())
    assert result.label == "High"
    assert result.lstm_next_ping_lat is None    # no history_window given -> no LSTM call, as before
    assert result.lstm_next_ping_lon is None
    assert "Dead-reckoned" in result.note

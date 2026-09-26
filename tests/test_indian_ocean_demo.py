"""
Tests for `wakewatch.scenario.indian_ocean_demo`.

These codify the empirical tuning done against the real, unmodified
attribution and pre-spill-gap pipelines -- if this scenario is ever edited
(vessel positions, speeds, timing), these tests catch whether the intended
narrative (SUSPECT ranks first, DARK GAP is flagged as a pre-spill
candidate) still actually holds, rather than trusting the narrative
description alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch import drift as drift_mod  # noqa: E402
from wakewatch import fusion  # noqa: E402
from wakewatch import pre_spill_dark as psd  # noqa: E402
from wakewatch.inference import ais as ais_mod  # noqa: E402
from wakewatch.registry import get_registry  # noqa: E402
from wakewatch.scenario import indian_ocean_demo as iod  # noqa: E402


@pytest.fixture(scope="module")
def scenario():
    return iod.build_indian_ocean_scenario()


@pytest.fixture(scope="module")
def scored_ais(scenario):
    model, scaler = get_registry()["ais_anomaly"].get()
    return ais_mod.score_frame(model, scaler, scenario["ais"])


# --------------------------------------------------------------------------
# Basic structure
# --------------------------------------------------------------------------
def test_scenario_has_six_named_vessels(scenario):
    assert len(scenario["vessels"]) == 6
    roles = [v.role for v in scenario["vessels"]]
    assert roles.count("transit") == 2
    assert roles.count("distractor") == 2
    assert roles.count("suspect") == 1
    assert roles.count("dark_gap") == 1


def test_all_vessels_have_a_narrative(scenario):
    for v in scenario["vessels"]:
        assert v.narrative and len(v.narrative) > 10


def test_mmsi_range_is_distinct_from_generic_synthetic_generator(scenario):
    """This scenario's MMSIs must not collide with synthetic_ais.py's
    default random range, so the two synthetic sources can never be
    confused with each other."""
    from wakewatch.synthetic_ais import SyntheticAisConfig
    generic_base = 900_000_000
    for v in scenario["vessels"]:
        assert not (generic_base <= v.mmsi < generic_base + 1000)
        assert v.mmsi >= 900_000_000  # still outside any real MID allocation


def test_ais_dataframe_matches_required_schema(scenario):
    from wakewatch.investigation import REQUIRED_AIS_COLUMNS
    missing = [c for c in REQUIRED_AIS_COLUMNS if c not in scenario["ais"].columns]
    assert missing == []


def test_disclaimer_is_present_and_unambiguous(scenario):
    text = scenario["disclaimer"].lower()
    assert "synthetic" in text
    assert "fabricated" in text or "not" in text


def test_scenario_is_deterministic_not_random():
    s1 = iod.build_indian_ocean_scenario()
    s2 = iod.build_indian_ocean_scenario()
    pd_testing_ok = s1["ais"]["latitude"].equals(s2["ais"]["latitude"])
    assert pd_testing_ok


def test_location_is_the_named_lakshadweep_sea_point(scenario):
    assert scenario["spill_position"] == (10.5, 72.8)


# --------------------------------------------------------------------------
# The empirically-verified narrative -- the real point of this scenario
# --------------------------------------------------------------------------
def test_suspect_ranks_first_in_real_attribution(scenario, scored_ais):
    """The core promise: running the real, unmodified attribution pipeline
    against this designed data must rank MV SUSPECT first, by a clear
    margin, with every other vessel scoring at or near zero."""
    spill_lat, spill_lon = scenario["spill_position"]
    out = fusion.attribute_from_observation(scored_ais, spill_lat, spill_lon, scenario["observed_at"],
                                            hours_back=12)
    results = out["results"]
    assert results, "attribution must find at least one candidate"
    top = results[0]
    assert top.vessel_name == "MV SUSPECT"
    runner_up_score = results[1].total_score if len(results) > 1 else 0.0
    assert top.total_score - runner_up_score > 0.3, "must be a clear, non-marginal margin"


def test_transit_and_distractor_vessels_score_at_or_near_zero(scenario, scored_ais):
    """Negative-case correctness: normal traffic and nearby-but-not-close
    vessels must not be flagged just for being in the area."""
    spill_lat, spill_lon = scenario["spill_position"]
    out = fusion.attribute_from_observation(scored_ais, spill_lat, spill_lon, scenario["observed_at"],
                                            hours_back=12)
    by_name = {r.vessel_name: r.total_score for r in out["results"]}
    for name in ["MV TRANSIT ONE", "MV TRANSIT TWO", "FV DISTRACTOR ONE", "FV DISTRACTOR TWO"]:
        assert by_name.get(name, 0.0) < 0.1


def test_dark_gap_vessel_flagged_by_scenario_2(scenario):
    """The other core promise: the real pre-spill AIS gap pipeline, run
    against the real hindcast origin computed from this scenario, must
    flag MV DARK GAP as a genuine candidate -- not ruled physically
    inconsistent, and not missed entirely."""
    spill_lat, spill_lon = scenario["spill_position"]
    hc = drift_mod.hindcast(spill_lat, spill_lon, scenario["observed_at"], hours_back=12)
    org = hc.origin_estimate
    report = psd.analyze_pre_spill_dark_vessels(
        scenario["ais"], org.latitude, org.longitude, org.time,
        ais_coverage_available=True, origin_uncertainty_km=hc.max_uncertainty_km,
    )
    dark_gap_candidates = [c for c in report.candidates if c.gap.vessel_name == "MV DARK GAP"]
    assert dark_gap_candidates, "MV DARK GAP must appear as a candidate"
    c = dark_gap_candidates[0]
    assert c.status == "potential_pre_spill_gap"
    assert c.reachability.plausible is True


def test_dark_gap_vessel_is_not_in_scenario_2_for_other_vessels(scenario):
    """Only MV DARK GAP should have a relevant pre-spill AIS gap -- the
    transit/distractor/suspect vessels all keep transmitting normally
    throughout the window and must not spuriously appear here."""
    spill_lat, spill_lon = scenario["spill_position"]
    hc = drift_mod.hindcast(spill_lat, spill_lon, scenario["observed_at"], hours_back=12)
    org = hc.origin_estimate
    report = psd.analyze_pre_spill_dark_vessels(
        scenario["ais"], org.latitude, org.longitude, org.time,
        ais_coverage_available=True, origin_uncertainty_km=hc.max_uncertainty_km,
    )
    flagged_names = {c.gap.vessel_name for c in report.candidates}
    assert flagged_names.issubset({"MV DARK GAP"})


def test_dark_gap_vessel_genuinely_never_resumes_in_this_feed(scenario):
    """Confirms the gap is real (an actual reporting stop), not an artefact
    -- MV DARK GAP's last ping must be well before the scenario's observed
    time, with no further pings after it."""
    track = scenario["ais"][scenario["ais"]["mmsi"] == 900_100_006].sort_values("timestamp")
    assert track["timestamp"].max() < scenario["observed_at"] - __import__("datetime").timedelta(hours=1)

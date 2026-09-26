"""Tests for `wakewatch.attribution_ablation`."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch import attribution_ablation as abl  # noqa: E402


def test_schemes_all_sum_to_one():
    for name, weights in abl.SCHEMES.items():
        assert abs(sum(weights.values()) - 1.0) < 1e-6, f"{name} weights don't sum to 1.0"


def test_shipped_default_matches_fusion_weights():
    from wakewatch.fusion import WEIGHTS
    assert abl.SCHEMES["shipped_default"] == WEIGHTS


def test_run_ablation_on_synthetic_components_ranks_correctly():
    components = [
        {"vessel_name": "STRONG", "components": {"proximity": 1.0, "anomaly": 1.0, "dwell": 1.0, "deviation": 1.0}},
        {"vessel_name": "WEAK", "components": {"proximity": 0.1, "anomaly": 0.1, "dwell": 0.1, "deviation": 0.1}},
    ]
    results = abl.run_ablation(components)
    for r in results:
        assert r.ranked_names[0] == "STRONG"
        assert r.margin > 0


def test_scheme_result_top_is_wakashio_property():
    r = abl.SchemeResult(scheme="x", weights={}, ranked_names=["WAKASHIO", "OTHER"], ranked_scores=[0.8, 0.2])
    assert r.top_is_wakashio is True
    assert abs(r.margin - 0.6) < 1e-9


def test_scheme_result_handles_empty_candidates():
    r = abl.SchemeResult(scheme="x", weights={}, ranked_names=[], ranked_scores=[])
    assert r.top_is_wakashio is False


def test_real_ablation_against_the_real_wakashio_incident():
    """End-to-end against real data: the shipped default and at least most
    alternative schemes should correctly separate WAKASHIO from clean
    traffic on the one real, labelled incident this project has."""
    components = abl._get_real_components()
    assert len(components) > 1  # the wider real search must surface more than just WAKASHIO
    results = abl.run_ablation(components)
    shipped = next(r for r in results if r.scheme == "shipped_default")
    assert shipped.top_is_wakashio is True
    assert shipped.margin > 0


def test_render_report_contains_key_honesty_language():
    components = [{"vessel_name": "WAKASHIO", "components": {"proximity": 1.0, "anomaly": 1.0,
                                                             "dwell": 1.0, "deviation": 0.0}}]
    results = abl.run_ablation(components)
    report = abl.render_report(results, components)
    assert "cannot statistically prove" in report
    assert "WAKASHIO" in report

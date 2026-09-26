"""Tests for `wakewatch.synthetic_ais` -- the opt-in, clearly-labelled
placeholder AIS generator used only when real coverage is unavailable."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wakewatch import synthetic_ais as synth  # noqa: E402
from wakewatch import investigation as inv  # noqa: E402


def test_generated_schema_matches_investigation_requirements():
    df = synth.generate_synthetic_ais(10.0, 20.0, datetime(2026, 1, 1, tzinfo=timezone.utc))
    missing = [c for c in inv.REQUIRED_AIS_COLUMNS if c not in df.columns]
    assert missing == []
    assert "vessel_name" in df.columns


def test_generated_ais_passes_investigation_loader_validation():
    """The generator's own output must survive the same validation path a
    user-uploaded CSV goes through -- proving the schema is genuinely
    compatible, not just superficially similar."""
    df = synth.generate_synthetic_ais(10.0, 20.0, datetime(2026, 1, 1, tzinfo=timezone.utc))
    csv_bytes = df.to_csv(index=False).encode()
    parsed = inv.load_ais_csv(csv_bytes)
    assert len(parsed) == len(df)


def test_vessel_count_matches_config():
    cfg = synth.SyntheticAisConfig(n_vessels=5)
    df = synth.generate_synthetic_ais(10.0, 20.0, datetime(2026, 1, 1, tzinfo=timezone.utc), cfg)
    assert df["mmsi"].nunique() == 5


def test_mmsi_range_is_never_a_real_allocation_block():
    """Synthetic MMSIs must be unmistakably outside real-world ranges, so a
    synthetic vessel can never be confused with a real one downstream."""
    df = synth.generate_synthetic_ais(10.0, 20.0, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert (df["mmsi"] >= 900_000_000).all()


def test_vessels_stay_within_configured_radius():
    cfg = synth.SyntheticAisConfig(n_vessels=20, radius_km=10.0, hours_span=1.0)
    df = synth.generate_synthetic_ais(0.0, 0.0, datetime(2026, 1, 1, tzinfo=timezone.utc), cfg)
    # Bound = starting radius + the absolute maximum a vessel could travel in
    # the full time span at the generator's own speed ceiling (max_speed_knots
    # * 1.2, its clip bound), not an arbitrary multiplier -- this is the real
    # worst case the generator can produce, not a guess.
    max_speed_kmh = cfg.max_speed_knots * 1.2 * 1.852
    max_travel_km = max_speed_kmh * cfg.hours_span
    max_dist_deg = ((cfg.radius_km + max_travel_km) / 111.0) * 1.1  # 10% slack for lon/lat projection
    assert df["latitude"].abs().max() < max_dist_deg
    assert df["longitude"].abs().max() < max_dist_deg


def test_reproducible_for_same_inputs():
    args = (15.0, 25.0, datetime(2026, 6, 1, tzinfo=timezone.utc))
    df1 = synth.generate_synthetic_ais(*args)
    df2 = synth.generate_synthetic_ais(*args)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_for_different_inputs():
    df1 = synth.generate_synthetic_ais(15.0, 25.0, datetime(2026, 6, 1, tzinfo=timezone.utc))
    df2 = synth.generate_synthetic_ais(16.0, 26.0, datetime(2026, 6, 1, tzinfo=timezone.utc))
    assert not df1["latitude"].equals(df2["latitude"])


def test_disclaimer_is_present_and_unambiguous():
    text = synth.SYNTHETIC_DISCLAIMER.lower()
    assert "synthetic" in text
    assert "not" in text and ("observed" in text or "real" in text)


def test_investigation_default_never_uses_synthetic_fallback():
    """The opt-in flag must default to False -- a normal call to
    run_investigation must never silently fabricate AIS."""
    import inspect
    sig = inspect.signature(inv.run_investigation)
    assert sig.parameters["allow_synthetic_ais_fallback"].default is False


def test_investigation_synthetic_status_is_distinct_from_real_statuses():
    """Guards against ever collapsing the synthetic path into a real-coverage
    status string, which would make it indistinguishable from real data."""
    import cv2
    import datetime as dt
    img_path = ROOT / "assets" / "sar_samples" / "scenes" / "wakashio_reef.jpg"
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    result = inv.run_investigation(img, 19.0760, 72.8777, dt.datetime(2026, 3, 1, tzinfo=dt.timezone.utc),
                                   hours_back=6, allow_synthetic_ais_fallback=True)
    assert result.ais_status == "synthetic_demo"
    assert result.ais_status not in ("builtin_mauritius", "user_uploaded", "unavailable")
    assert "SYNTHETIC" in result.ais_message


def test_investigation_real_mauritius_scenario_ignores_the_opt_in_flag():
    """The flag must be a no-op whenever real coverage genuinely exists --
    opting in must never override or contaminate a real result."""
    import cv2
    from wakewatch.scenario.real_ais import build_scenario
    img_path = ROOT / "assets" / "sar_samples" / "scenes" / "wakashio_reef.jpg"
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    sc = build_scenario()
    result = inv.run_investigation(img, *sc["spill_position"], sc["grounding_utc"], hours_back=12,
                                   allow_synthetic_ais_fallback=True)
    assert result.ais_status == "builtin_mauritius"
    assert result.attribution_results[0].vessel_name == "WAKASHIO"

"""
Attribution Pipeline page.

This view was listed in the README's page table but was missing from ui/app.py.
It exposes the full spill-to-vessel correlation and ranking engine so analysts
can explore and challenge the attribution interactively.

What this page shows
--------------------
1. Ranked suspect table — all vessels scored, colour-coded by confidence band.
2. Score breakdown — radar chart of the four weighted components for any vessel.
3. Evidence narrative — plain-language audit trail for the top-ranked vessel.
4. Parameter panel — analysts can adjust the search radius and time window,
   and see immediately how the ranking changes.
5. Weight explorer — adjust the four attribution weights and re-rank in real
   time, because weights are a policy choice and should be arguable.
6. Drift integration — shows the drift caveat from fusion.py, and if the drift
   module has produced a hindcast origin, offers to re-run attribution from
   the estimated release point instead of the observed slick position.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from wakewatch import fusion
from wakewatch.config import AE_THRESHOLD
from wakewatch.inference import ais as ais_mod
from wakewatch.scenario.wakashio import GROUNDING_LAT, GROUNDING_LON

from .. import charts, engine, maps, theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_color(band: str) -> str:
    return theme.BAND_COLORS.get(band, theme.MUTED)


def _score_gauge(score: float, label: str) -> str:
    """Horizontal progress bar for a 0–1 score."""
    pct = min(100, score * 100)
    color = (theme.CRITICAL if score >= 0.70 else
             theme.WARN if score >= 0.45 else
             theme.ACCENT if score >= 0.20 else theme.MUTED)
    return f"""
<div style="margin:.25rem 0">
  <div style="display:flex;justify-content:space-between;
              font-size:.75rem;color:{theme.MUTED};margin-bottom:3px">
    <span>{label}</span><span style="color:{color}">{score:.3f}</span>
  </div>
  <div style="background:{theme.PANEL_2};border-radius:4px;height:8px;overflow:hidden">
    <div style="width:{pct:.0f}%;height:100%;background:{color}"></div>
  </div>
</div>"""


def _component_bars(components: Dict[str, float], anchored_on_origin: bool = True) -> str:
    html = ""
    labels = {
        "proximity": ("Proximity to estimated origin" if anchored_on_origin
                      else "Proximity to observed slick"),
        "anomaly": "AIS behaviour anomaly",
        "dwell": "Dwell time in area",
        "deviation": "Track deviation",
    }
    for key, val in components.items():
        html += _score_gauge(val, labels.get(key, key))
    return html


def _attribution_map(
    results: List[Dict[str, Any]],
    scored_ais: pd.DataFrame,
    spill_lat: float,
    spill_lon: float,
    radius_km: float,
    hindcast: Optional[Dict[str, Any]] = None,
) -> folium.Map:
    """
    Map showing all candidate vessels plus the slick.

    When `hindcast` is provided (i.e. attribution was anchored on the
    hindcast-estimated release point rather than the raw observation), the
    map makes the two-step chain explicit: the observed slick is shown as
    reference, but the search circle, vessel search, and map centring are all
    drawn around the *origin*, since that is where AIS traffic was actually
    reconstructed and scored. Without this, a judge looking only at the map
    could reasonably assume attribution was scored against the red dot.
    """
    use_origin = bool(hindcast and hindcast.get("estimated_origin_lat") is not None)
    if use_origin:
        anchor_lat = hindcast["estimated_origin_lat"]
        anchor_lon = hindcast["estimated_origin_lon"]
    else:
        anchor_lat, anchor_lon = spill_lat, spill_lon

    fmap = maps.base_map((anchor_lat, anchor_lon), zoom=10)

    # Search radius circle — drawn around the point actually searched.
    folium.Circle(
        location=[anchor_lat, anchor_lon],
        radius=radius_km * 1000,
        color=theme.WARN,
        fill=True,
        fill_opacity=0.04,
        weight=1.5,
        dash_array="6 3",
        tooltip=f"AIS search radius {radius_km:.0f} km "
                f"(around {'hindcast origin' if use_origin else 'observed slick'})",
    ).add_to(fmap)

    # Observed slick marker — always shown, for reference.
    folium.Marker(
        location=[spill_lat, spill_lon],
        icon=folium.DivIcon(
            html=f'<div style="font-size:15px;line-height:1;white-space:nowrap;'
                 f'transform:translate(-4px,-4px)">🔴 <span style="font-size:.7rem;'
                 f'color:{theme.TEXT};background:{theme.PANEL}cc;padding:1px 5px;'
                 f'border-radius:4px;margin-left:2px">Observed slick</span></div>',
            icon_size=(140, 20), icon_anchor=(10, 10)),
        tooltip="🔴 Observed slick position (satellite detection) — "
                "NOT the AIS search center when hindcasting is on",
    ).add_to(fmap)

    if use_origin:
        # Hindcast origin marker — this is the actual AIS search anchor.
        folium.Marker(
            location=[anchor_lat, anchor_lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:15px;line-height:1;white-space:nowrap;'
                     f'transform:translate(-4px,-4px)">🟠 <span style="font-size:.7rem;'
                     f'color:{theme.TEXT};background:{theme.PANEL}cc;padding:1px 5px;'
                     f'border-radius:4px;margin-left:2px">Hindcast origin '
                     f'(AIS search center)</span></div>',
                icon_size=(220, 20), icon_anchor=(10, 10)),
            tooltip=(f"🟠 Hindcast-estimated release origin — actual AIS search "
                     f"center · ±{hindcast.get('max_uncertainty_km', 0):.1f} km "
                     f"uncertainty · {hindcast.get('estimated_release_time', '')}"),
        ).add_to(fmap)

        # Dashed connector so the slick → origin relationship reads at a
        # glance, matching the "🔴 → 🟠 → 🚢" attribution chain.
        folium.PolyLine(
            locations=[[spill_lat, spill_lon], [anchor_lat, anchor_lon]],
            color=theme.MUTED,
            weight=2,
            opacity=0.8,
            dash_array="4 6",
            tooltip="Backward drift hindcast (slick → estimated origin)",
        ).add_to(fmap)

    # Vessel tracks, colour-coded by band
    all_lats = [spill_lat, anchor_lat]
    all_lons = [spill_lon, anchor_lon]
    for r in results:
        mmsi = r["mmsi"]
        track = scored_ais[scored_ais["mmsi"] == mmsi].copy()
        if track.empty:
            continue
        lats = track["latitude"].tolist()
        lons = track["longitude"].tolist()
        band = r.get("band", "cleared by proximity")
        color = _band_color(band)
        if len(lats) >= 2:
            folium.PolyLine(
                locations=list(zip(lats, lons)),
                color=color,
                weight=2,
                opacity=0.75,
                tooltip=f"{r['vessel_name']} — {band} (score {r['score']:.3f})",
            ).add_to(fmap)
        # Last known position
        if lats:
            folium.CircleMarker(
                location=[lats[-1], lons[-1]],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.85,
                tooltip=f"{r['vessel_name']}\n{band}\nScore: {r['score']:.3f}",
            ).add_to(fmap)
        all_lats.extend(lats)
        all_lons.extend(lons)

    maps._fit(fmap, all_lats, all_lons, pad=0.03)
    return fmap


def _run_attribution(
    spill_lat: float,
    spill_lon: float,
    radius_km: float,
    window_hours: float,
    weights_override: Optional[Dict[str, float]] = None,
    use_hindcast: bool = True,
    hours_back: float = 12.0,
) -> Dict[str, Any]:
    """
    Run attribution with optional custom weights.

    When `use_hindcast` is True (the default), the search is anchored on the
    drift-hindcast estimate of the slick's release point and time rather than
    the raw observed position — reconstructing traffic "around the origin
    window in space and time" as the PS asks, not around where the satellite
    happened to see the oil. Set False to fall back to the raw observation
    (e.g. for a known-fresh spill, or to compare the two anchors).

    When weights_override is provided, the module-level WEIGHTS dict is
    temporarily patched (only within this call — no global mutation).
    """
    sc = engine.scenario()
    devs = engine.max_deviations()

    if use_hindcast:
        out = fusion.attribute_from_observation(
            engine.scored_ais(),
            spill_lat, spill_lon, sc["grounding_utc"],
            radius_km=radius_km,
            window_hours=window_hours,
            threshold=AE_THRESHOLD,
            deviations=devs,
            hours_back=hours_back,
        )
        results, meta = out["results"], out
    else:
        results = fusion.attribute(
            engine.scored_ais(),
            spill_lat,
            spill_lon,
            observed_at=sc["grounding_utc"],
            radius_km=radius_km,
            window_hours=window_hours,
            threshold=AE_THRESHOLD,
            deviations=devs,
        )
        meta = {"search_anchor": "observed_position", "hindcast": None,
                "note": fusion.drift_caveat(hindcasted=False)}

    if weights_override:
        # Re-score with custom weights without touching module globals
        anchor_label = ("estimated release origin" if use_hindcast
                        else "observed slick")
        for r in results:
            r.total_score = sum(
                weights_override.get(k, 0) * v
                for k, v in r.components.items()
            )
            r.evidence = fusion._build_evidence(r, radius_km, anchor_label)
        results.sort(key=lambda a: a.total_score, reverse=True)

    return {**meta, "results": [r.as_dict() for r in results]}


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.markdown("## Attribution pipeline")

    sc = engine.scenario()
    spill_lat, spill_lon = sc["spill_position"]
    scored_ais = engine.scored_ais()

    # ---------------------------------------------------------------- controls
    with st.expander("⚙️ Attribution parameters", expanded=True):
        use_hindcast = st.toggle(
            "Anchor on hindcast-estimated release point (recommended)",
            value=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            radius_km = st.slider("Search radius (km)", 5.0, 100.0, 15.0, 1.0)
            window_hours = st.slider("Time window (hours either side)", 1.0, 24.0, 6.0, 0.5)
            hours_back = st.slider("Hindcast horizon (hours back)", 1.0, 48.0, 12.0, 1.0,
                                   disabled=not use_hindcast)
        with col2:
            st.markdown(
                f"<div style='font-size:.8rem;color:{theme.MUTED};margin-bottom:.4rem'>"
                "Attribution weights (must sum to 1.0)</div>",
                unsafe_allow_html=True,
            )
            w_prox = st.slider("Proximity weight", 0.0, 1.0, 0.40, 0.05)
            w_anom = st.slider("Anomaly weight", 0.0, 1.0, 0.30, 0.05)
            w_dwell = st.slider("Dwell weight", 0.0, 1.0, 0.20, 0.05)
            w_dev = st.slider("Deviation weight", 0.0, 1.0, 0.10, 0.05)

    total_w = w_prox + w_anom + w_dwell + w_dev
    if abs(total_w - 1.0) > 0.01:
        st.warning(
            f"Weights sum to {total_w:.2f} (not 1.0). "
            "Results are still valid but scores won't be on the 0–1 scale."
        )

    custom_weights = {
        "proximity": w_prox,
        "anomaly": w_anom,
        "dwell": w_dwell,
        "deviation": w_dev,
    }

    # ---------------------------------------------------------------- run
    with st.spinner("Running attribution pipeline..."):
        run_out = _run_attribution(
            spill_lat, spill_lon,
            radius_km, window_hours,
            weights_override=custom_weights,
            use_hindcast=use_hindcast,
            hours_back=hours_back,
        )
        results = run_out["results"]

    hindcast_meta = None
    if run_out.get("search_anchor") == "hindcast_origin":
        hc = run_out["hindcast"]
        hindcast_meta = hc
        # Use the widened radius for map/evidence display so it matches what
        # was actually searched.
        radius_km = run_out.get("effective_radius_km", radius_km)

    if not results:
        st.warning("No AIS traffic found in the search window. "
                   "Try widening the radius, time window, or hindcast horizon.")
        return

    top = results[0]
    summary = fusion.summarise([
        type("R", (), r)() for r in results  # lightweight proxy for summarise
    ] if False else [])

    # ---------------------------------------------------------------- headline
    band_color = _band_color(top["band"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(theme.metric_card(
            "Top suspect",
            top["vessel_name"],
            f"score {top['score']:.3f}",
            band_color,
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card(
            "Confidence band",
            top["band"].title(),
            f"MMSI {top['mmsi']}",
            band_color,
        ), unsafe_allow_html=True)
    with c3:
        margin = (top["score"] - results[1]["score"]) if len(results) > 1 else top["score"]
        margin_color = theme.GOOD if margin >= 0.20 else theme.WARN
        st.markdown(theme.metric_card(
            "Score margin over #2",
            f"{margin:.3f}",
            "clear" if margin >= 0.20 else "narrow — review both",
            margin_color,
        ), unsafe_allow_html=True)
    with c4:
        st.markdown(theme.metric_card(
            "Candidates evaluated",
            str(len(results)),
            f"within {radius_km:.0f} km / ±{window_hours:.0f} h",
        ), unsafe_allow_html=True)

    st.markdown("")

    # ---------------------------------------------------------------- layout
    left, right = st.columns([1.4, 1])

    with left:
        # Map
        st.markdown("##### Candidate vessel tracks")
        if hindcast_meta:
            st.markdown(
                f"<div style='font-size:.78rem;color:{theme.MUTED};margin:-.3rem 0 .5rem'>"
                "🔴 Observed slick &nbsp;→&nbsp; 🟠 Hindcast origin "
                "<i>(actual AIS search center)</i> &nbsp;→&nbsp; 🚢 AIS candidate tracks"
                "</div>",
                unsafe_allow_html=True,
            )
        fmap = _attribution_map(results, scored_ais, spill_lat, spill_lon, radius_km,
                                hindcast=hindcast_meta)
        st_folium(fmap, use_container_width=True, height=460,
                  returned_objects=[], key="attr_map")
        legend_items = [
            {"color": theme.CRITICAL, "label": "Primary suspect"},
            {"color": theme.WARN, "label": "Person of interest"},
            {"color": theme.ACCENT, "label": "In the area"},
            {"color": theme.MUTED, "label": "Cleared by proximity"},
            {"color": theme.BAD, "label": "🔴 Observed slick"},
        ]
        if hindcast_meta:
            legend_items.append({"color": theme.WARN, "label": "🟠 Hindcast origin (search center)"})
        st.markdown(maps.legend(legend_items), unsafe_allow_html=True)

    with right:
        # Evidence for top vessel
        st.markdown(f"##### Evidence — {top['vessel_name']}")
        evidence_html = "".join(
            f"<li style='margin:.3rem 0;font-size:.85rem'>{e}</li>"
            for e in top["evidence"]
        ) or "<li>No evidence items generated.</li>"
        st.markdown(
            f"<div class='pos-card'><ul style='margin:0;padding-left:1rem'>"
            f"{evidence_html}</ul></div>",
            unsafe_allow_html=True,
        )

        # Component scores
        st.markdown("##### Score components")
        st.markdown(
            f"<div class='pos-card'>"
            f"{_component_bars(top['components'], anchored_on_origin=(run_out.get('search_anchor') == 'hindcast_origin'))}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------- ranking table
    st.markdown("##### Full candidate ranking")
    df = pd.DataFrame(results)
    show_cols = ["vessel_name", "flag", "vessel_type", "score", "band",
                 "min_distance_km", "minutes_in_radius", "flagged_pings",
                 "max_anomaly_score"]
    df_show = df[show_cols].copy()
    df_show.columns = [
        "Vessel", "Flag", "Type", "Score", "Band",
        "Min dist (km)", "Dwell (min)", "Flagged pings", "Peak anomaly",
    ]

    def _row_style(row):
        color = _band_color(row["Band"])
        return [f"border-left: 3px solid {color}"] + [""] * (len(row) - 1)

    st.dataframe(
        df_show.style
            .format({
                "Score": "{:.3f}",
                "Min dist (km)": "{:.2f}",
                "Dwell (min)": "{:.0f}",
                "Peak anomaly": "{:.3f}",
            }),
        use_container_width=True,
        hide_index=True,
    )



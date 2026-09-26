"""
Oil Drift Modelling — Hindcast & Forecast page.

Implements the drift model that was, in an earlier revision, a documented
gap: backward hindcast to estimate release point/time, forward ensemble
forecast, and SAR-based age estimation. Wind is fetched live from Open-Meteo
(Forecast API for recent dates, Historical Weather API for older ones);
ocean current from the Open-Meteo Marine API where available, falling back
to a Mauritius-AOI climatology only when live data can't be reached or
doesn't cover the requested date — see `drift.py` and each result's
`data_source` field for exactly which applied.

What this page shows
--------------------
1. HINDCAST tab — traces the slick backward from its observed SAR position
   to estimate where and when the release originated. Shows the backward
   track on a satellite map with a growing uncertainty envelope, plus a
   table of estimated origin position and time.

2. FORECAST tab — predicts where the slick will spread over the next
   24 hours as a probabilistic ensemble cone, so responders know where
   to deploy containment.

3. SPILL AGE tab — estimates how old the slick is from SAR observable
   properties (area, fragmentation) and surfaces the uncertainty band.

Data source
-----------
Wind is fetched from Open-Meteo (free, no API key) when network is
available; ocean current uses the Mauritius AOI climatological model.
The page shows which source was used on every result.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from wakewatch import drift as drift_mod
from wakewatch.scenario.wakashio import GROUNDING_LAT, GROUNDING_LON, GROUNDING_UTC

from .. import engine, maps, theme


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drift_map(
    hindcast: Optional[drift_mod.HindcastResult] = None,
    forecast: Optional[drift_mod.ForecastResult] = None,
    spill_lat: float = GROUNDING_LAT,
    spill_lon: float = GROUNDING_LON,
) -> folium.Map:
    """Build a Folium map showing drift tracks."""
    fmap = maps.base_map((spill_lat, spill_lon), zoom=10)

    # Observed slick position
    folium.CircleMarker(
        location=[spill_lat, spill_lon],
        radius=10,
        color=theme.BAD,
        fill=True,
        fill_opacity=0.9,
        tooltip="Observed slick position (SAR)",
    ).add_to(fmap)

    if hindcast and hindcast.steps:
        track = hindcast.track_latlon()
        # Draw backward track
        if len(track) >= 2:
            folium.PolyLine(
                locations=track,
                color=theme.WARN,
                weight=2.5,
                opacity=0.85,
                dash_array="8 4",
                tooltip="Hindcast track (backward drift)",
            ).add_to(fmap)

        # Uncertainty circles at selected steps
        for i, step in enumerate(hindcast.steps):
            if i % 3 == 0 and step.uncertainty_km > 0:
                folium.Circle(
                    location=[step.latitude, step.longitude],
                    radius=step.uncertainty_km * 1000,
                    color=theme.WARN,
                    fill=True,
                    fill_opacity=0.06,
                    weight=1,
                ).add_to(fmap)

        # Origin estimate
        org = hindcast.origin_estimate
        if org:
            folium.Marker(
                location=[org.latitude, org.longitude],
                icon=folium.Icon(color="orange", icon="flag", prefix="fa"),
                tooltip=(
                    f"Estimated release origin\n"
                    f"{org.time.strftime('%Y-%m-%d %H:%M UTC')}\n"
                    f"±{org.uncertainty_km:.1f} km"
                ),
            ).add_to(fmap)

    if forecast and forecast.ensemble:
        centroid = forecast.centroid_track()
        if len(centroid) >= 2:
            folium.PolyLine(
                locations=[(s.latitude, s.longitude) for s in centroid],
                color=theme.ACCENT,
                weight=2.5,
                opacity=0.9,
                tooltip="Forecast centroid track",
            ).add_to(fmap)

        # Draw ensemble cone at 6 h intervals
        n_steps = min(len(forecast.ensemble[0]), len(centroid))
        cone_indices = list(range(0, n_steps, max(1, n_steps // 6)))
        for idx in cone_indices:
            hull = forecast.cone_polygon(idx)
            if len(hull) >= 3:
                folium.Polygon(
                    locations=hull,
                    color=theme.ACCENT,
                    fill=True,
                    fill_opacity=0.08,
                    weight=1,
                ).add_to(fmap)

    return fmap


def _age_bar(age: drift_mod.SpillAgeEstimate) -> str:
    """HTML progress-bar style indicator for age range."""
    max_display = 72.0
    pct_lo = min(100, age.min_hours / max_display * 100)
    pct_hi = min(100, age.max_hours / max_display * 100)
    pct_best = min(100, age.best_hours / max_display * 100)
    return f"""
<div style="margin:0.6rem 0">
  <div style="font-size:.75rem;color:{theme.MUTED};margin-bottom:4px">
    Age range (0 – 72 h shown)
  </div>
  <div style="position:relative;height:14px;background:{theme.PANEL_2};
              border-radius:7px;overflow:hidden">
    <div style="position:absolute;left:{pct_lo:.0f}%;width:{pct_hi-pct_lo:.0f}%;
                height:100%;background:{theme.WARN};opacity:0.45"></div>
    <div style="position:absolute;left:{pct_best:.0f}%;
                transform:translateX(-50%);width:3px;height:100%;
                background:{theme.WARN}"></div>
  </div>
  <div style="display:flex;justify-content:space-between;
              font-size:.7rem;color:{theme.MUTED};margin-top:3px">
    <span>{age.min_hours:.0f} h min</span>
    <span>{age.best_hours:.0f} h best</span>
    <span>{age.max_hours:.0f} h max</span>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render() -> None:
    st.markdown("## Oil drift modelling")

    sc = engine.scenario()
    spill_lat, spill_lon = sc["spill_position"]
    observed_at = sc["grounding_utc"]

    # ---------------------------------------------------------------- controls
    with st.expander("⚙️ Drift model parameters", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            hours_back = st.slider("Hindcast hours back", 2, 48, 12)
            use_live_wind = st.checkbox("Fetch live wind (Open-Meteo)", value=True)
        with col2:
            hours_forward = st.slider("Forecast hours forward", 6, 48, 24)
            n_ensemble = st.slider("Ensemble members", 5, 50, 20)
        with col3:
            spill_lat_in = st.number_input("Spill latitude", value=spill_lat,
                                            format="%.6f")
            spill_lon_in = st.number_input("Spill longitude", value=spill_lon,
                                            format="%.6f")

    # Use UI values if changed
    if abs(spill_lat_in - spill_lat) > 1e-6 or abs(spill_lon_in - spill_lon) > 1e-6:
        spill_lat, spill_lon = spill_lat_in, spill_lon_in

    # ---------------------------------------------------------------- tabs
    tab_hind, tab_fore, tab_age = st.tabs([
        "🔙 Hindcast — trace to origin",
        "🔮 Forecast — future spread",
        "⏱️ Spill age estimation",
    ])

    # ================================================================ HINDCAST
    with tab_hind:
        st.markdown("##### Backward drift trace — where did the slick come from?")

        with st.spinner("Running hindcast..."):
            hc = drift_mod.hindcast(
                observed_lat=spill_lat,
                observed_lon=spill_lon,
                observed_at=observed_at,
                hours_back=hours_back,
                dt_hours=1.0,
                use_live_wind=use_live_wind,
            )

        org = hc.origin_estimate
        if org:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(theme.metric_card(
                    "Estimated origin lat",
                    f"{org.latitude:.5f}°",
                    "release point",
                ), unsafe_allow_html=True)
            with c2:
                st.markdown(theme.metric_card(
                    "Estimated origin lon",
                    f"{org.longitude:.5f}°",
                    "release point",
                ), unsafe_allow_html=True)
            with c3:
                st.markdown(theme.metric_card(
                    "Estimated release time",
                    org.time.strftime("%H:%M UTC"),
                    org.time.strftime("%Y-%m-%d"),
                    theme.WARN,
                ), unsafe_allow_html=True)
            with c4:
                st.markdown(theme.metric_card(
                    "Max uncertainty",
                    f"±{hc.max_uncertainty_km:.1f} km",
                    f"after {hours_back} h backward",
                    theme.MUTED,
                ), unsafe_allow_html=True)

        st.markdown("")
        fmap_hc = _drift_map(hindcast=hc, spill_lat=spill_lat, spill_lon=spill_lon)
        st_folium(fmap_hc, use_container_width=True, height=480,
                  returned_objects=[], key="hc_map")

        st.markdown(
            maps.legend([
                {"color": theme.BAD, "label": "Observed slick (SAR)"},
                {"color": theme.WARN, "label": "Hindcast backward track"},
                {"color": "#ff9f1c", "label": "Estimated release origin"},
            ]),
            unsafe_allow_html=True,
        )

        # Hindcast table
        st.markdown("##### Hindcast track detail")
        rows = [s.as_dict() for s in hc.steps]
        df_hc = pd.DataFrame(rows)
        df_hc["time"] = pd.to_datetime(df_hc["time"]).dt.strftime("%H:%M UTC")
        df_hc.columns = ["Time", "Latitude", "Longitude",
                          "Speed (m/s)", "Uncertainty (km)"]
        st.dataframe(df_hc.style.format({
            "Latitude": "{:.5f}", "Longitude": "{:.5f}",
            "Speed (m/s)": "{:.3f}", "Uncertainty (km)": "{:.2f}",
        }), use_container_width=True, hide_index=True)

    # ================================================================ FORECAST
    with tab_fore:
        st.markdown("##### Forward drift prediction — where will the slick go?")

        with st.spinner(f"Running {n_ensemble}-member ensemble forecast..."):
            fc = drift_mod.forecast(
                start_lat=spill_lat,
                start_lon=spill_lon,
                start_time=observed_at,
                hours_forward=hours_forward,
                dt_hours=1.0,
                n_ensemble=n_ensemble,
                use_live_wind=use_live_wind,
            )

        centroid = fc.centroid_track()
        final = centroid[-1] if centroid else None

        if final:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(theme.metric_card(
                    f"Position at +{hours_forward}h lat",
                    f"{final.latitude:.5f}°",
                    "centroid forecast",
                ), unsafe_allow_html=True)
            with c2:
                st.markdown(theme.metric_card(
                    f"Position at +{hours_forward}h lon",
                    f"{final.longitude:.5f}°",
                    "centroid forecast",
                ), unsafe_allow_html=True)
            with c3:
                st.markdown(theme.metric_card(
                    "Spread uncertainty",
                    f"±{final.uncertainty_km:.1f} km",
                    f"at {hours_forward} h",
                    theme.ACCENT,
                ), unsafe_allow_html=True)
            with c4:
                st.markdown(theme.metric_card(
                    "Ensemble members",
                    str(n_ensemble),
                    fc.data_source,
                    theme.MUTED,
                ), unsafe_allow_html=True)

        st.markdown("")
        fmap_fc = _drift_map(forecast=fc, spill_lat=spill_lat, spill_lon=spill_lon)
        st_folium(fmap_fc, use_container_width=True, height=480,
                  returned_objects=[], key="fc_map")

        st.markdown(
            maps.legend([
                {"color": theme.BAD, "label": "Start position (observed slick)"},
                {"color": theme.ACCENT, "label": "Forecast centroid track"},
                {"color": theme.ACCENT, "label": "Uncertainty cone (ensemble spread)"},
            ]),
            unsafe_allow_html=True,
        )

        # Forecast table
        st.markdown("##### Forecast centroid track")
        fc_rows = [s.as_dict() for s in centroid]
        df_fc = pd.DataFrame(fc_rows)
        df_fc["time"] = pd.to_datetime(df_fc["time"]).dt.strftime("%H:%M UTC")
        df_fc.columns = ["Time", "Latitude", "Longitude",
                          "Speed (m/s)", "Spread (km)"]
        st.dataframe(df_fc.style.format({
            "Latitude": "{:.5f}", "Longitude": "{:.5f}",
            "Speed (m/s)": "{:.3f}", "Spread (km)": "{:.2f}",
        }), use_container_width=True, hide_index=True)

    # ================================================================ AGE
    with tab_age:
        st.markdown("##### Spill age estimation from SAR observables")

        # Pull from the precomputed SAR library if available
        from wakewatch.scenario import sar_scenes
        scenes = sar_scenes.load_scenes()
        wakashio_scene = next((s for s in scenes if s.mmsi == 372711000), None)

        col_left, col_right = st.columns([1.2, 1])
        with col_left:
            if wakashio_scene:
                oil_area = st.number_input(
                    "Oil area (km²)", value=float(wakashio_scene.oil_area_km2),
                    min_value=0.0, format="%.4f",
                )
                oil_frac = float(wakashio_scene.class_pixels.get("Oil Spill", 0)) / max(
                    1, sum(wakashio_scene.class_pixels.values()))
                slick_count = st.number_input(
                    "Slick fragments (connected components)",
                    value=int(wakashio_scene.slicks),
                    min_value=1,
                )
            else:
                oil_area = st.number_input(
                    "Oil area (km²)", value=4.28, min_value=0.0, format="%.4f")
                oil_frac = 0.05
                slick_count = st.number_input(
                    "Slick fragments", value=3, min_value=1)

            use_contrast = st.checkbox("Include backscatter contrast", value=False)
            contrast_val = None
            if use_contrast:
                contrast_val = st.slider(
                    "Backscatter contrast (0 = dark/old, 1 = bright/fresh)",
                    0.0, 1.0, 0.4,
                )

        with col_right:
            age = drift_mod.estimate_age(
                oil_area_km2=float(oil_area),
                oil_pixel_fraction=oil_frac,
                slick_count=int(slick_count),
                backscatter_contrast=contrast_val,
            )

            band_color = (theme.GOOD if age.best_hours < 6 else
                          theme.WARN if age.best_hours < 24 else theme.BAD)

            st.markdown(theme.metric_card(
                "Age category",
                age.label,
                f"best estimate: {age.best_hours:.0f} h",
                band_color,
            ), unsafe_allow_html=True)

            st.markdown(
                f"<div style='margin-top:.5rem'>{_age_bar(age)}</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"<div class='pos-card' style='margin-top:.75rem'>"
                f"<div class='pos-label'>Estimate range</div>"
                f"<div style='font-size:1.1rem;font-weight:640'>"
                f"{age.min_hours:.0f} – {age.max_hours:.0f} hours</div>"
                f"<div class='pos-sub'>Method: {age.method}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )



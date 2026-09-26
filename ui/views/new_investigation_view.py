"""
New Investigation page.

ADDITIVE ONLY. This page does not modify any existing page, model, weight,
scoring formula, or drift equation. It runs the exact same pipeline the
Wakashio demo pages already use (`wakewatch.investigation.run_investigation`,
which itself only calls the existing `sar`, `drift`, `ais`, `trajectory`, and
`fusion` modules) against a SAR image and location/time a judge supplies,
instead of the hardcoded demo scenario.

Two input modes, per the brief:
  A. A georeferenced Sentinel-1 GeoTIFF -- metadata is extracted automatically
     where possible (`investigation.extract_geotiff_metadata`); only genuinely
     missing fields are asked for.
  B. A plain PNG/JPG -- no geographic metadata can exist in the file, so
     location and time are always asked for directly.

AIS coverage is handled honestly: if the supplied location/time falls inside
the one region this project has real AIS for (the Mauritius AOI, matching the
existing demo), that feed is reused automatically. Outside it, the page says
so plainly and offers an optional AIS CSV upload rather than pretending
global coverage exists.
"""
from __future__ import annotations

from datetime import datetime, time as dtime, timezone
from typing import Optional

import pandas as pd
import streamlit as st

from streamlit_folium import st_folium

from wakewatch import investigation as inv
from wakewatch import synthetic_ais as synthetic_ais_mod
from wakewatch.inference import sar as sar_mod

from .. import charts, theme
from .attribution_view import _attribution_map, _component_bars


# --------------------------------------------------------------------------
# Session-state keys (namespaced so this page can't collide with any other)
# --------------------------------------------------------------------------
K_IMAGE = "newinv_image_rgb"
K_META = "newinv_metadata"
K_RESULT = "newinv_result"
K_AIS_DF = "newinv_ais_override"
K_AIS_ERR = "newinv_ais_error"
K_BOUNDS = "newinv_geo_bounds"
K_OBSERVED_AT = "newinv_observed_at"
K_DARK_VESSEL_SELECTED = "newinv_dark_vessel_selected"


def render() -> None:
    quick_start = st.radio(
        "Quick start",
        ["Custom upload", "Mauritius Demonstration (real incident)",
        "Indian Ocean Demonstration (designed scenario)"],
        horizontal=True, key="newinv_quick_start",
    )

    if quick_start != "Custom upload":
        image_rgb, lat, lon, obs_date, obs_time, meta_ok = _load_demo_scenario(quick_start)
        st.session_state[K_IMAGE] = image_rgb
        st.image(image_rgb, caption=f"{quick_start} \u2014 SAR scene", use_container_width=True)
    else:
        mode = st.radio(
            "Input type",
            ["Sentinel-1 GeoTIFF / Georeferenced Product", "SAR Image (PNG/JPG)"],
            horizontal=True,
        )
        is_geotiff_mode = mode.startswith("Sentinel-1")

        upload = st.file_uploader(
            "Upload SAR Image",
            type=["tif", "tiff"] if is_geotiff_mode else ["jpg", "jpeg", "png"],
        )

        if upload is None:
            st.session_state.pop(K_IMAGE, None)
            st.session_state.pop(K_META, None)
            st.session_state.pop(K_RESULT, None)
            st.session_state.pop(K_BOUNDS, None)
            return

        file_bytes = upload.getvalue()

        # Decode the image once per upload (cheap image, no need to cache).
        try:
            image_rgb = inv.read_uploaded_sar_image(file_bytes)
        except inv.InvestigationError as exc:
            st.error(f"Could not read this file: {exc}")
            return
        except Exception as exc:  # unexpected decode failure -- never silently continue
            st.error(f"Unexpected error reading this file: {exc}")
            return

        st.session_state[K_IMAGE] = image_rgb
        st.image(image_rgb, caption="Uploaded SAR scene", use_container_width=True)

        # ------------------------------------------------------------ metadata
        st.markdown("#### Metadata")
        lat, lon, obs_date, obs_time, meta_ok = _metadata_section(
            file_bytes, is_geotiff_mode)

    # ---------------------------------------------------------------- advanced params
    with st.expander("Advanced parameters (optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            resolution = st.slider("Ground resolution (m/pixel)", 1.0, 40.0, 10.0, 1.0)
        with c2:
            hours_back = st.slider("Hindcast horizon (hours back)", 1.0, 48.0, 12.0, 1.0)
        with c3:
            radius_km = st.slider("AIS search radius (km)", 5.0, 100.0, 15.0, 1.0)
        window_hours = st.slider("AIS time window (hours either side)", 1.0, 24.0, 6.0, 0.5)

        st.markdown("**AIS data (optional)**")
        ais_upload = st.file_uploader("Upload AIS CSV (optional)", type=["csv"], key="newinv_ais_upload")
        if ais_upload is not None:
            try:
                st.session_state[K_AIS_DF] = inv.load_ais_csv(ais_upload.getvalue())
                st.session_state.pop(K_AIS_ERR, None)
                st.success(f"Parsed {len(st.session_state[K_AIS_DF]):,} AIS rows.")
            except inv.InvestigationError as exc:
                st.session_state.pop(K_AIS_DF, None)
                st.session_state[K_AIS_ERR] = str(exc)

        st.markdown("**Synthetic AIS demo fallback (optional, off by default)**")
        allow_synthetic = st.checkbox(
            "Generate synthetic AIS traffic if real coverage is unavailable here",
            value=False, key="newinv_allow_synthetic_ais",
        )
        if allow_synthetic:
            st.warning(
                "\u26a0\ufe0f " + synthetic_ais_mod.SYNTHETIC_DISCLAIMER,
            )
        if st.session_state.get(K_AIS_ERR):
            st.error(st.session_state[K_AIS_ERR])

    # ---------------------------------------------------------------- run
    st.markdown("---")
    run_clicked = st.button("\U0001F50D RUN COMPLETE ANALYSIS", type="primary",
                            disabled=not meta_ok, use_container_width=True)

    if run_clicked:
        observed_at = datetime.combine(obs_date, obs_time, tzinfo=timezone.utc)
        st.session_state[K_OBSERVED_AT] = observed_at
        with st.spinner("Running the complete pipeline -- SAR \u2192 hindcast \u2192 AIS \u2192 attribution..."):
            try:
                result = inv.run_investigation(
                    image_rgb, lat, lon, observed_at,
                    pixel_resolution_m=resolution,
                    ais_override_df=st.session_state.get(K_AIS_DF),
                    allow_synthetic_ais_fallback=st.session_state.get("newinv_allow_synthetic_ais", False),
                    hours_back=hours_back, radius_km=radius_km, window_hours=window_hours,
                )
                st.session_state[K_RESULT] = result
            except Exception as exc:
                st.session_state.pop(K_RESULT, None)
                st.error(
                    "The analysis could not complete. This is most often a model-"
                    f"loading or data problem, not your input: {exc}"
                )
                return

    if K_RESULT in st.session_state:
        _render_results(st.session_state[K_RESULT], lat, lon, st.session_state.get(K_OBSERVED_AT))


# --------------------------------------------------------------------------
# Metadata section
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Demo scenarios -- Mauritius (real incident) and Indian Ocean (designed,
# synthetic). Both bypass the file uploader and metadata entry, loading
# fixed, known-good values instead.
# --------------------------------------------------------------------------
def _load_demo_scenario(quick_start: str):
    """Returns (image_rgb, lat, lon, obs_date, obs_time, meta_ok) for a
    selected demo scenario. The image is the same bundled Wakashio Sentinel-1
    sample in both cases -- for Mauritius that is genuinely correct imagery;
    for the Indian Ocean scenario it is explicitly disclosed as borrowed,
    illustrative imagery, not a real scene of the Lakshadweep Sea."""
    from wakewatch.config import PROJECT_ROOT
    from wakewatch.inference import sar as sar_mod
    sample_path = PROJECT_ROOT / "assets" / "sar_samples" / "scenes" / "wakashio_reef.jpg"
    image_rgb = sar_mod.read_image(str(sample_path))

    if quick_start.startswith("Mauritius"):
        from wakewatch.scenario.real_ais import build_scenario
        sc = build_scenario()
        lat, lon = sc["spill_position"]
        at = sc["grounding_utc"]
        st.session_state.pop(K_AIS_DF, None)  # real coverage resolves automatically -- no override needed
    else:
        from wakewatch.scenario.indian_ocean_demo import build_indian_ocean_scenario
        sc = build_indian_ocean_scenario()
        lat, lon = sc["spill_position"]
        at = sc["observed_at"]
        st.session_state[K_AIS_DF] = sc["ais"]
        theme.banner("\u26a0\ufe0f " + sc["disclaimer"], "warn")

    st.session_state[K_BOUNDS] = None
    return image_rgb, lat, lon, at.date(), at.time(), True


def _metadata_section(file_bytes: bytes, is_geotiff_mode: bool):
    """Returns (lat, lon, date, time, all_required_present)."""
    auto = inv.ExtractedMetadata()
    if is_geotiff_mode:
        auto = inv.extract_geotiff_metadata(file_bytes)
        st.session_state[K_BOUNDS] = auto.bounds
        if auto.is_complete:
            st.success(
                f"Metadata auto-detected from the file ({auto.extraction_backend})."
            )
        else:
            st.warning(
                "Metadata incomplete \u2014 please enter the missing information. "
                f"Could not auto-detect: {', '.join(auto.missing)}."
            )
    else:
        st.session_state[K_BOUNDS] = None

    c1, c2 = st.columns(2)
    with c1:
        if auto.latitude is not None:
            st.caption("Latitude \u2014 auto-detected")
            lat = st.number_input("Latitude", value=float(auto.latitude),
                                  min_value=-90.0, max_value=90.0, format="%.5f",
                                  key="newinv_lat")
        else:
            st.caption("Latitude \u2014 manual entry required")
            lat = st.number_input("Latitude", value=0.0, min_value=-90.0, max_value=90.0,
                                  format="%.5f", key="newinv_lat")
    with c2:
        if auto.longitude is not None:
            st.caption("Longitude \u2014 auto-detected")
            lon = st.number_input("Longitude", value=float(auto.longitude),
                                  min_value=-180.0, max_value=180.0, format="%.5f",
                                  key="newinv_lon")
        else:
            st.caption("Longitude \u2014 manual entry required")
            lon = st.number_input("Longitude", value=0.0, min_value=-180.0, max_value=180.0,
                                  format="%.5f", key="newinv_lon")

    c3, c4 = st.columns(2)
    with c3:
        default_date = None
        if auto.acquisition_date:
            try:
                default_date = datetime.strptime(auto.acquisition_date, "%Y-%m-%d").date()
            except ValueError:
                default_date = None
        st.caption("Acquisition Date \u2014 " + ("auto-detected" if default_date else "manual entry required"))
        obs_date = st.date_input("Acquisition Date", value=default_date, key="newinv_date")
    with c4:
        default_time = None
        if auto.acquisition_time_utc:
            try:
                h, m = auto.acquisition_time_utc.split(":")
                default_time = dtime(int(h), int(m))
            except ValueError:
                default_time = None
        st.caption("Acquisition Time (UTC) \u2014 " + ("auto-detected" if default_time else "manual entry required"))
        obs_time = st.time_input("Acquisition Time (UTC)", value=default_time, key="newinv_time")

    all_present = lat is not None and lon is not None and obs_date is not None and obs_time is not None
    if lat == 0.0 and lon == 0.0 and auto.latitude is None:
        all_present = False  # untouched default, not a real (0,0) location
    return lat, lon, obs_date, obs_time, all_present


# --------------------------------------------------------------------------
# Results (sections 1-11 from the brief)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# PDF investigation report export
# --------------------------------------------------------------------------
def _render_pdf_download_button(result: inv.InvestigationResult, spill_lat: float, spill_lon: float,
                                observed_at: Optional[datetime]) -> None:
    if observed_at is None:
        return
    import io as _io
    from wakewatch import report_export
    try:
        buf = _io.BytesIO()
        report_export.generate_investigation_pdf(result, buf, spill_lat, spill_lon, observed_at)
        buf.seek(0)
        st.download_button(
            "\U0001F4C4 Download Investigation Report (PDF)", data=buf,
            file_name=f"investigation_report_{observed_at.strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf", key="newinv_pdf_download",
        )
    except Exception as exc:
        st.caption(f"PDF report unavailable: {exc}")


def _render_results(result: inv.InvestigationResult, spill_lat: float, spill_lon: float,
                    observed_at: Optional[datetime] = None) -> None:
    st.markdown("## Investigation Results")
    _render_pdf_download_button(result, spill_lat, spill_lon, observed_at)

    for w in result.warnings:
        theme.banner(w, "warn")

    # 1-2: uploaded image + detected mask
    st.markdown("#### 1\u20132. SAR Scene &amp; Detected Oil-Spill Mask")
    mask_rgb = sar_mod.mask_to_rgb(result.sar.mask)
    overlay = sar_mod.overlay_on_image(st.session_state[K_IMAGE], result.sar.mask, alpha=0.45)
    t1, t2 = st.tabs(["Overlay", "Mask only"])
    with t1:
        st.image(overlay, use_container_width=True)
    with t2:
        st.image(mask_rgb, use_container_width=True)

    # 3: spill characteristics
    st.markdown("#### 3. Spill Characteristics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(theme.metric_card("Oil area", f"~{result.sar.oil_area_km2:.2f} km\u00b2",
                                      "approximate"), unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card("Distinct slicks", str(len(result.sar.slicks))),
                   unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card("Estimated age", result.age.label,
                                      f"{result.age.min_hours:.0f}\u2013{result.age.max_hours:.0f} h"),
                   unsafe_allow_html=True)
    with c4:
        st.markdown(theme.metric_card("Look-alike area", f"~{result.sar.lookalike_area_km2:.2f} km\u00b2"),
                   unsafe_allow_html=True)

    # 4-6: drift / hindcast / estimated origin & time
    st.markdown("#### 4\u20136. Drift Hindcast, Estimated Origin &amp; Release Time")
    org = result.hindcast.origin_estimate
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(theme.metric_card("Estimated origin",
                                      f"{org.latitude:.4f}, {org.longitude:.4f}" if org else "n/a"),
                   unsafe_allow_html=True)
    with c2:
        st.markdown(theme.metric_card("Estimated release time",
                                      org.time.strftime("%Y-%m-%d %H:%M UTC") if org else "n/a"),
                   unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card("Uncertainty", f"\u00b1{result.hindcast.max_uncertainty_km:.1f} km",
                                      f"source: {result.hindcast.data_source}"),
                   unsafe_allow_html=True)
    track_df = pd.DataFrame([s.as_dict() for s in result.hindcast.steps])
    if not track_df.empty:
        st.dataframe(track_df[["time", "latitude", "longitude", "uncertainty_km"]],
                    use_container_width=True, hide_index=True)

    # 7: AIS vessels considered
    st.markdown("#### 7. AIS Data")
    tone = {"builtin_mauritius": "good", "user_uploaded": "good",
           "unavailable": "warn", "synthetic_demo": "warn"}.get(result.ais_status, "warn")
    theme.banner(result.ais_message, tone)

    if result.scored_ais is None:
        _render_dark_vessel_analysis(result, observed_at)
        return

    n_vessels = result.scored_ais["mmsi"].nunique()
    n_pings = len(result.scored_ais)
    st.markdown(theme.metric_card("Vessels considered", str(n_vessels), f"{n_pings:,} AIS pings"),
               unsafe_allow_html=True)

    # 8: vessel trajectories
    st.markdown("#### 8. Vessel Trajectories")
    if result.deviations_used:
        st.caption("Trajectory-deviation scoring (LSTM) was run -- this location/time "
                  "is inside the trajectory model's trained AOI.")
    else:
        st.caption(result.trajectory_note or
                  "Trajectory-deviation scoring was not run for this investigation.")

    # 9: vessel behaviour / anomaly results
    st.markdown("#### 9. Vessel Behaviour / Anomaly Results")
    rollup = result.scored_ais.groupby("mmsi").agg(
        pings=("anomaly_score", "size"),
        flagged=("is_anomaly", "sum"),
        max_score=("anomaly_score", "max"),
    ).reset_index().sort_values("max_score", ascending=False)
    if "vessel_name" in result.scored_ais.columns:
        names = result.scored_ais.groupby("mmsi")["vessel_name"].first()
        rollup["vessel_name"] = rollup["mmsi"].map(names)
    st.dataframe(rollup, use_container_width=True, hide_index=True)

    # 10-11: ranked vessels + evidence
    st.markdown("#### 10\u201311. Ranked Vessels &amp; Evidence")
    if not result.attribution_results:
        st.info("No AIS traffic fell within the search radius/time window around "
               "the estimated origin -- no suspect vessels to rank.")
        _render_dark_vessel_analysis(result, observed_at)
        return

    results_dicts = [r.as_dict() for r in result.attribution_results]
    meta = result.attribution_meta or {}

    fmap = _attribution_map(results_dicts, result.scored_ais, spill_lat, spill_lon,
                            meta.get("effective_radius_km", 15.0), hindcast=meta.get("hindcast"))
    st_folium(fmap, use_container_width=True, height=420, returned_objects=[],
             key="newinv_attr_map")

    st.altair_chart(charts.attribution_bars(results_dicts), use_container_width=True)

    top = results_dicts[0]
    st.markdown(f"##### Top suspect \u2014 {top['vessel_name']} (score {top['score']:.3f})")
    ev_html = "".join(f"<li style='margin:.3rem 0;font-size:.85rem'>{e}</li>" for e in top["evidence"])
    st.markdown(f"<div class='pos-card'><ul style='margin:0;padding-left:1rem'>{ev_html}</ul></div>",
               unsafe_allow_html=True)
    st.markdown(_component_bars(top["components"], anchored_on_origin=(meta.get("search_anchor") == "hindcast_origin")),
               unsafe_allow_html=True)

    with st.expander("All ranked vessels"):
        table = pd.DataFrame([
            {"Vessel": r["vessel_name"], "MMSI": r["mmsi"], "Score": round(r["score"], 3),
             "Min distance (km)": round(r["min_distance_km"], 2)}
            for r in results_dicts
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)

    _render_dark_vessel_analysis(result, observed_at)


# --------------------------------------------------------------------------
# Dark Vessel Detection (SAR-AIS cross-verification) -- additive section.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Combined action: run both Scenario 1 and Scenario 2 from one button.
# Neither scenario's own section or button is removed -- this sits above
# them as a convenience that runs both in sequence, reusing the exact same
# stage functions the individual buttons call.
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# EO (Sentinel-2) analysis -- a real, second detection modality alongside
# SAR. Genuinely optional: EO needs a cloud-free daytime pass over the same
# area/time as the SAR overpass, which won't always exist.
# --------------------------------------------------------------------------
def _render_dark_vessel_analysis(result: inv.InvestigationResult, observed_at: Optional[datetime]) -> None:
    st.markdown("---")
    st.markdown("## Dark Vessel Analysis")

    if st.button("\U0001F311 RUN DARK VESSEL ANALYSIS", type="primary", key="newinv_run_combined_dark_vessel",
                use_container_width=True):
        if observed_at is None:
            st.error("Run the complete analysis above first.")
        else:
            with st.spinner("Running Scenario 1 (SAR-AIS cross-verification) and "
                            "Scenario 2 (pre-spill AIS gap analysis)..."):
                try:
                    inv.run_dark_vessel_stage(result, st.session_state[K_IMAGE], observed_at,
                                             bounds=st.session_state.get(K_BOUNDS))
                except Exception as exc:
                    st.error(f"Scenario 1 (SAR-AIS cross-verification) could not complete: {exc}")
                try:
                    inv.run_pre_spill_dark_stage(result)
                except Exception as exc:
                    st.error(f"Scenario 2 (pre-spill AIS gap analysis) could not complete: {exc}")

    if result.dark_vessel_report is not None or result.pre_spill_dark_report is not None:
        n_s1 = len(result.dark_vessel_report.candidates) if result.dark_vessel_report else 0
        n_s2 = len(result.pre_spill_dark_report.candidates) if result.pre_spill_dark_report else 0
        st.markdown(
            f"**Scenario 1 \u2014 SAR-AIS Cross Verification:** {n_s1} potential SAR-AIS mismatch(es)  \n"
            f"**Scenario 2 \u2014 Pre-Spill AIS Gap Analysis:** {n_s2} potential AIS-gap candidate(s)"
        )

    _render_dark_vessel_section(result, observed_at)


def _render_dark_vessel_section(result: inv.InvestigationResult, observed_at: Optional[datetime]) -> None:
    st.markdown("---")
    st.markdown("### Dark Vessel Detection")
    st.caption(
        "Sentinel-1 SAR uses radar to detect physical vessel targets. These detections "
        "are cross-checked against AIS positions acquired around the same time. A "
        "SAR-detected vessel without a corresponding AIS position is flagged as a "
        "**Potential Dark Vessel**. AIS coverage, timing, and detection uncertainty are "
        "considered before flagging."
    )
    st.caption(
        "\u26a0\ufe0f An AIS mismatch does not prove that a vessel intentionally disabled AIS."
    )

    if st.button("\U0001F6F0\ufe0f Run SAR Vessel Detection", key="newinv_run_dark_vessel"):
        if observed_at is None:
            st.error("Run the complete analysis above first \u2014 SAR vessel detection needs "
                     "the acquisition time from that run.")
            return
        with st.spinner("Detecting SAR vessel candidates and cross-checking against AIS..."):
            try:
                inv.run_dark_vessel_stage(
                    result, st.session_state[K_IMAGE], observed_at,
                    bounds=st.session_state.get(K_BOUNDS),
                )
            except Exception as exc:
                st.error(f"SAR vessel detection could not complete: {exc}")
                return

    if result.dark_vessel_report is None:
        st.info("Not run yet for this investigation.")
        _render_pre_spill_section(result)
        return

    report = result.dark_vessel_report
    st.markdown(theme.metric_card("Detected SAR vessels", str(len(result.sar_vessel_candidates or []))),
               unsafe_allow_html=True)
    st.caption(report.summary)

    if not result.sar_vessel_geolocated:
        theme.banner(
            "This image is not georeferenced, so SAR vessel candidates could not be "
            "placed on the map or matched against AIS by position \u2014 pixel-level "
            "detections are listed below without coordinates.", "warn",
        )

    if not report.candidates:
        st.info("No SAR vessel candidates passed the detection filters for this scene.")
        _render_pre_spill_section(result)
        return

    # ---- map ----
    if result.sar_vessel_geolocated:
        st.markdown(
            "**Legend:** \U0001F535 AIS-matched &nbsp;&nbsp; \U0001F7E0 Potential Dark Vessel "
            "&nbsp;&nbsp; \u26AA Unverified / low-confidence"
        )
        fmap = _dark_vessel_map(report)
        if fmap is not None:
            st_folium(fmap, use_container_width=True, height=380, returned_objects=[],
                     key="newinv_dark_vessel_map")

    # ---- candidate list + details panel ----
    labels = []
    for i, c in enumerate(report.candidates):
        icon = {"ais_confirmed": "\U0001F535", "potential_dark_vessel": "\U0001F7E0"}.get(c.status, "\u26AA")
        labels.append(f"{icon} Candidate {i + 1} \u2014 {c.status.replace('_', ' ')}")

    choice = st.selectbox("Select a candidate for details", labels, key=K_DARK_VESSEL_SELECTED)
    idx = labels.index(choice)
    _render_dark_vessel_details(report.candidates[idx])

    _render_pre_spill_section(result)


def _dark_vessel_map(report):
    from .. import maps
    geolocated = [c for c in report.candidates if c.candidate.latitude is not None]
    if not geolocated:
        return None
    center = (geolocated[0].candidate.latitude, geolocated[0].candidate.longitude)
    fmap = maps.base_map(center, zoom=12)
    color_for = {"ais_confirmed": "blue", "potential_dark_vessel": "orange"}
    import folium
    for c in geolocated:
        color = color_for.get(c.status, "lightgray")
        popup = f"{c.status.replace('_', ' ').title()}<br>Evidence: {c.evidence_label}"
        folium.CircleMarker(
            location=(c.candidate.latitude, c.candidate.longitude),
            radius=8, color=color, fill=True, fill_opacity=0.85, popup=popup,
        ).add_to(fmap)
        if c.last_known is not None:
            folium.PolyLine(
                [(c.last_known.latitude, c.last_known.longitude),
                 (c.candidate.latitude, c.candidate.longitude)],
                color="gray", dash_array="5,5", weight=2,
            ).add_to(fmap)
            folium.CircleMarker(
                location=(c.last_known.latitude, c.last_known.longitude),
                radius=5, color="gray", fill=True, fill_opacity=0.6,
                popup="Last known AIS position",
            ).add_to(fmap)
    return fmap


def _render_dark_vessel_details(c) -> None:
    st.markdown("#### Candidate Details")
    tone = {"ais_confirmed": "good", "potential_dark_vessel": "warn",
           "ais_coverage_unavailable": "info", "unverified_low_confidence": "info"}.get(c.status, "info")
    theme.banner(c.note, tone)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(theme.metric_card("SAR detection", "\u2713"), unsafe_allow_html=True)
    with c2:
        ais_symbol = "\u2713" if c.status == "ais_confirmed" else "\u2717"
        st.markdown(theme.metric_card("AIS match", ais_symbol), unsafe_allow_html=True)
    with c3:
        st.markdown(theme.metric_card("Dark-vessel evidence", c.evidence_label), unsafe_allow_html=True)

    loc = (f"{c.candidate.latitude:.4f}, {c.candidate.longitude:.4f}"
          if c.candidate.latitude is not None else "unavailable (not georeferenced)")
    st.markdown(f"**Estimated location:** {loc}")

    if c.ais_match and c.ais_match.matched:
        st.markdown(f"**Matched AIS vessel:** {c.ais_match.vessel_name or c.ais_match.mmsi} "
                   f"\u2014 {c.ais_match.distance_km:.2f} km away, "
                   f"{c.ais_match.time_diff_minutes:.0f} min apart")

    if c.last_known is not None:
        lk = c.last_known
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"**Last AIS position:** {lk.latitude:.4f}, {lk.longitude:.4f}")
            st.markdown(f"**Last AIS time:** {lk.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
        with cc2:
            st.markdown(f"**AIS gap:** {lk.gap_hours:.1f} hours")
            st.markdown(f"**Distance from last AIS position:** {lk.distance_km:.1f} km")

    if c.trajectory is not None:
        st.markdown(f"**Trajectory consistency:** {c.trajectory.label}")
        st.caption(c.trajectory.note)

    if c.evidence_components:
        with st.expander("Evidence components"):
            for k, v in c.evidence_components.items():
                st.caption(f"\u2022 {k.replace('_', ' ')}: {v:.2f}")


# --------------------------------------------------------------------------
# Pre-Spill AIS Gap Analysis ("Scenario 2") -- additive section.
# --------------------------------------------------------------------------
def _render_pre_spill_section(result: inv.InvestigationResult) -> None:
    st.markdown("---")
    st.markdown("### \U0001F311 Pre-Spill Dark Vessel Analysis")

    with st.expander("Pre-spill analysis parameters (optional)"):
        c1, c2 = st.columns(2)
        with c1:
            pre_spill_window_hours = st.slider("Search window before release (hours)", 1.0, 48.0, 12.0, 1.0,
                                               key="newinv_pre_spill_window")
        with c2:
            min_dark_gap_minutes = st.slider("Minimum gap to count as \"dark\" (minutes)", 5.0, 180.0, 30.0, 5.0,
                                             key="newinv_min_dark_gap")

    if st.button("\U0001F311 Run Pre-Spill AIS Gap Analysis", key="newinv_run_pre_spill"):
        if result.hindcast is None or result.hindcast.origin_estimate is None:
            st.error("No hindcast origin estimate is available to search around.")
            return
        with st.spinner("Searching AIS history before the estimated spill for AIS gaps..."):
            try:
                from wakewatch.pre_spill_dark import PreSpillConfig
                config = PreSpillConfig(pre_spill_window_hours=pre_spill_window_hours,
                                       min_dark_gap_minutes=min_dark_gap_minutes)
                inv.run_pre_spill_dark_stage(result, pre_spill_config=config)
            except Exception as exc:
                st.error(f"Pre-spill AIS gap analysis could not complete: {exc}")
                return

    if result.pre_spill_dark_report is None:
        st.info("Not run yet for this investigation.")
        return

    report = result.pre_spill_dark_report
    tone = "good" if report.ais_coverage_available else "warn"
    theme.banner(report.summary, tone)
    if report.vessels_with_insufficient_history:
        st.caption(f"Insufficient AIS history to evaluate {report.vessels_with_insufficient_history} "
                  "vessel(s) (fewer than 2 pings).")

    if report.candidates:
        st.markdown(f"**Candidates found:** {len(report.candidates)}")
        table = pd.DataFrame([{
            "MMSI": c.gap.mmsi,
            "Vessel": c.gap.vessel_name or "\u2014",
            "Last AIS": c.gap.gap_start.strftime("%Y-%m-%d %H:%M UTC"),
            "AIS Gap": f"{int(c.gap.gap_duration_minutes // 60)}h {int(c.gap.gap_duration_minutes % 60)}m",
            "Distance to Origin (km)": c.distance_to_origin_km,
            "Time Consistency": c.time_consistency.label,
            "Trajectory Consistency": c.gap_trajectory.label,
            "Physical Reachability": "YES" if c.reachability.plausible else "NO",
            "Evidence Level": c.evidence_label,
        } for c in report.candidates])
        st.dataframe(table, use_container_width=True, hide_index=True)

        with st.expander("Candidate details"):
            idx = st.selectbox("Select a candidate", range(len(report.candidates)),
                              format_func=lambda i: f"MMSI {report.candidates[i].gap.mmsi} "
                                                   f"({report.candidates[i].evidence_label})",
                              key="newinv_pre_spill_selected")
            c = report.candidates[idx]
            st.markdown(f"**{c.note}**")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                st.markdown(theme.metric_card("AIS gap", f"{c.gap.gap_duration_minutes:.0f} min"),
                           unsafe_allow_html=True)
            with cc2:
                st.markdown(theme.metric_card("Distance to origin", f"{c.distance_to_origin_km:.1f} km"),
                           unsafe_allow_html=True)
            with cc3:
                st.markdown(theme.metric_card("Evidence", c.evidence_label), unsafe_allow_html=True)
            st.caption(c.reachability.note)
            st.caption(c.gap_trajectory.note)
            if c.next_position:
                st.caption(c.next_position.note)

    _render_combined_dark_vessel_summary(result)


def _render_combined_dark_vessel_summary(result: inv.InvestigationResult) -> None:
    """Scenario 1 + Scenario 2, merged by MMSI -- distinguishing every case
    from the brief (AIS-confirmed, Scenario 1 only, Scenario 2 only,
    multi-source, coverage unavailable) rather than merging them into one
    undifferentiated list."""
    if not result.combined_dark_vessel_candidates:
        return
    st.markdown("---")
    st.markdown("#### Dark Vessel Analysis \u2014 Combined")
    n_s1 = len(result.dark_vessel_report.candidates) if result.dark_vessel_report else 0
    n_s2 = len(result.pre_spill_dark_report.candidates) if result.pre_spill_dark_report else 0
    st.caption(f"Scenario 1 (SAR-AIS mismatch): {n_s1} candidate(s). "
              f"Scenario 2 (pre-spill AIS gap): {n_s2} candidate(s).")

    rows = []
    for c in result.combined_dark_vessel_candidates:
        multi = len(c.sources) > 1
        rows.append({
            "MMSI": str(c.mmsi) if c.mmsi is not None else "\u2014",
            "Vessel": c.vessel_name or "\u2014",
            "Source": "Multi-source" if multi else (
                "Scenario 1 (SAR-AIS)" if c.sources == ["scenario1_sar_ais_mismatch"] else "Scenario 2 (AIS gap)"),
            "Evidence": c.combined_evidence_label,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    for c in result.combined_dark_vessel_candidates:
        if len(c.sources) > 1:
            theme.banner(f"MMSI {c.mmsi} ({c.vessel_name or 'unknown name'}): {c.note}", "warn")

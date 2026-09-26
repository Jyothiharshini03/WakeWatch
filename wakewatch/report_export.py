"""
Investigation report export (PDF).

Produces a structured, printable summary of a completed
`investigation.InvestigationResult` -- SAR detection, spill characterisation,
drift hindcast, AIS status, vessel attribution ranking, and dark-vessel
findings when available -- for an analyst to save, print, or attach to a
case file. This does not change what the pipeline computes; it only
formats an already-computed `InvestigationResult` for hand-off outside the
interactive console, which the existing Streamlit/API layers don't do.

Uses reportlab (pure Python, no external binary dependency such as
wkhtmltopdf or a Chromium install) so this works the same way on any machine
that can already run the rest of the project.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

from . import investigation as inv_mod

_STYLES = getSampleStyleSheet()
_TITLE = ParagraphStyle("InvTitle", parent=_STYLES["Title"], fontSize=18, spaceAfter=4)
_H2 = ParagraphStyle("InvH2", parent=_STYLES["Heading2"], spaceBefore=14, spaceAfter=6,
                     textColor=colors.HexColor("#1a3d5c"))
_BODY = ParagraphStyle("InvBody", parent=_STYLES["BodyText"], spaceAfter=4, leading=14)
_CAVEAT = ParagraphStyle("InvCaveat", parent=_STYLES["BodyText"], fontSize=8.5, leading=11,
                         textColor=colors.HexColor("#555555"))
_TABLE_HEADER_BG = colors.HexColor("#1a3d5c")


def _table(data, col_widths=None) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _TABLE_HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def generate_investigation_pdf(
    result: "inv_mod.InvestigationResult",
    output_path: str,
    spill_lat: float,
    spill_lon: float,
    observed_at: datetime,
    title: str = "WakeWatch Investigation Report",
) -> str:
    """
    Render `result` to a PDF at `output_path`. Returns `output_path`.

    Every section is optional and omitted cleanly if that stage wasn't run
    (e.g. no Dark Vessel Detection section if `run_dark_vessel_stage` was
    never called) -- the report reflects exactly what was actually computed
    for this investigation, not a fixed template with blanks.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    story = []

    story.append(Paragraph(title, _TITLE))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(
        f"Generated {generated} &nbsp;|&nbsp; Investigation location: {spill_lat:.4f}, {spill_lon:.4f} "
        f"&nbsp;|&nbsp; Observed at: {observed_at.strftime('%Y-%m-%d %H:%M UTC')}", _BODY))

    # ---- SAR detection & characterisation ----
    story.append(Paragraph("SAR Detection &amp; Spill Characterisation", _H2))
    sar = result.sar
    story.append(_table([
        ["Property", "Value"],
        ["Oil area (approx.)", f"{sar.oil_area_km2:.2f} sq km"],
        ["Look-alike area (approx.)", f"{sar.lookalike_area_km2:.2f} sq km"],
        ["Distinct slicks detected", str(len(sar.slicks))],
        ["Estimated age", f"{result.age.label} ({result.age.min_hours:.0f}\u2013{result.age.max_hours:.0f} h)"],
        ["Age-estimate method", result.age.method],
    ], col_widths=[65 * mm, 95 * mm]))
    if result.age.caveats:
        story.append(Spacer(1, 3))
        for c in result.age.caveats:
            story.append(Paragraph(f"\u2022 {c}", _CAVEAT))

    # ---- Drift hindcast ----
    story.append(Paragraph("Drift Hindcast &amp; Estimated Origin", _H2))
    hc = result.hindcast
    org = hc.origin_estimate
    hc_rows = [["Property", "Value"]]
    if org is not None:
        hc_rows += [
            ["Estimated release position", f"{org.latitude:.4f}, {org.longitude:.4f}"],
            ["Estimated release time", org.time.strftime("%Y-%m-%d %H:%M UTC")],
            ["Positional uncertainty", f"\u00b1{hc.max_uncertainty_km:.1f} km"],
        ]
    hc_rows.append(["Wind/current data source", hc.data_source])
    story.append(_table(hc_rows, col_widths=[65 * mm, 95 * mm]))
    story.append(Paragraph(
        "Wind/current \"synthetic\" means the live feed was unreachable or didn't cover this "
        "date and a labelled climatology fallback was used -- never presented as real-time or "
        "historical observation.", _CAVEAT))

    # ---- AIS status ----
    story.append(Paragraph("AIS Data", _H2))
    story.append(Paragraph(result.ais_message, _BODY))

    # ---- Attribution ranking ----
    if result.attribution_results:
        story.append(Paragraph("Vessel Attribution Ranking", _H2))
        meta = result.attribution_meta or {}
        anchor = meta.get("search_anchor", "observed_position")
        anchor_label = "hindcast-estimated origin" if anchor == "hindcast_origin" else "observed slick position"
        story.append(Paragraph(f"Search anchored on: {anchor_label}.", _BODY))
        rows = [["Rank", "Vessel", "MMSI", "Score", "Min. distance (km)"]]
        for i, r in enumerate(result.attribution_results, 1):
            d = r.as_dict()
            rows.append([str(i), d["vessel_name"], str(d["mmsi"]), f"{d['score']:.3f}",
                        f"{d['min_distance_km']:.2f}"])
        story.append(_table(rows, col_widths=[14 * mm, 46 * mm, 30 * mm, 22 * mm, 38 * mm]))

        top = result.attribution_results[0].as_dict()
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Top candidate evidence \u2014 {top['vessel_name']}:</b>", _BODY))
        for e in top["evidence"]:
            story.append(Paragraph(f"\u2022 {e}", _BODY))

    # ---- Scenario 1: SAR-AIS mismatch ----
    if result.dark_vessel_report is not None:
        story.append(Paragraph("Dark Vessel Detection \u2014 Scenario 1 (SAR-AIS Cross-Verification)", _H2))
        story.append(Paragraph(result.dark_vessel_report.summary, _BODY))
        dv_candidates = [c for c in result.dark_vessel_report.candidates if c.status == "potential_dark_vessel"]
        if dv_candidates:
            rows = [["SAR candidate", "Status", "Evidence"]]
            for c in dv_candidates:
                rows.append([str(c.candidate.id), "Potential Dark Vessel", c.evidence_label])
            story.append(_table(rows, col_widths=[40 * mm, 70 * mm, 40 * mm]))
        story.append(Paragraph(
            "A SAR-AIS mismatch does not by itself prove AIS shutdown or illegal activity.", _CAVEAT))

    # ---- Scenario 2: pre-spill AIS gaps ----
    if result.pre_spill_dark_report is not None:
        story.append(Paragraph("Dark Vessel Detection \u2014 Scenario 2 (Pre-Spill AIS Gap Analysis)", _H2))
        story.append(Paragraph(result.pre_spill_dark_report.summary, _BODY))
        gap_candidates = [c for c in result.pre_spill_dark_report.candidates
                         if c.status == "potential_pre_spill_gap"]
        if gap_candidates:
            rows = [["MMSI", "AIS Gap", "Distance to Origin", "Trajectory", "Time", "Evidence"]]
            for c in gap_candidates:
                rows.append([str(c.gap.mmsi), f"{c.gap.gap_duration_minutes:.0f} min",
                            f"{c.distance_to_origin_km:.1f} km", c.gap_trajectory.label,
                            c.time_consistency.label, c.evidence_label])
            story.append(_table(rows, col_widths=[24 * mm, 24 * mm, 30 * mm, 24 * mm, 24 * mm, 24 * mm]))
        story.append(Paragraph(
            "This is an investigative lead, not proof of AIS shutdown or responsibility.", _CAVEAT))

    # ---- Combined summary ----
    if result.combined_dark_vessel_candidates:
        multi = [c for c in result.combined_dark_vessel_candidates if len(c.sources) > 1]
        if multi:
            story.append(Paragraph("Multi-Source Dark Vessel Evidence", _H2))
            for c in multi:
                story.append(Paragraph(f"MMSI {c.mmsi} ({c.vessel_name or 'unknown name'}): {c.note}", _BODY))

    # ---- Standing caveats ----
    story.append(Paragraph("Standing Limitations", _H2))
    for line in [
        "Attribution weights (proximity/anomaly/dwell/deviation) are a documented policy choice, "
        "not a statistically validated optimum -- see docs/attribution_weight_ablation.md.",
        "The trajectory model is normalised for the Mauritius AOI and is not applied outside it.",
        "The AIS anomaly detector has 0.41 recall: it nominates vessels for review and cannot clear one.",
        "This report is a decision-support artefact, not a legal or scientific finding of responsibility.",
    ]:
        story.append(Paragraph(f"\u2022 {line}", _CAVEAT))

    doc.build(story)
    return output_path

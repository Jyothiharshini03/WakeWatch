"""
WakeWatch inference service.

An optional deployment tier. The Streamlit console runs the models in-process
and needs nothing here, which keeps a demo to a single command. This service
exists for the case that in-process loading stops being viable:

  * several analysts sharing one GPU, rather than one 110 MB SegFormer per seat
  * the UI restarting without paying the model load again
  * anything other than Streamlit -- a dashboard, a batch job, a duty console --
    consuming the same models through one contract

Both paths import the same `wakewatch.inference` modules and the same registry,
so there is exactly one implementation of each model's behaviour and no way for
the API and the UI to drift apart.

Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from wakewatch import __version__, drift, fusion  # noqa: E402
from wakewatch.config import AE_THRESHOLD, FEATURE_ORDER, SEQ_LEN  # noqa: E402
from wakewatch.inference import ais as ais_mod  # noqa: E402
from wakewatch.inference import sar as sar_mod  # noqa: E402
from wakewatch.inference import trajectory as traj_mod  # noqa: E402
from wakewatch.registry import get_registry  # noqa: E402
from wakewatch.scenario.real_ais import build_scenario  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Deliberately no eager warmup: a service that only ever answers AIS calls
    # should not pay for the SAR encoder. POST /admin/warmup to force it.
    yield
    for handle in get_registry().handles.values():
        handle.unload()


app = FastAPI(
    title="WakeWatch Inference API",
    version=__version__,
    description=(
        "Oil-spill segmentation, AIS anomaly detection, vessel trajectory "
        "prediction, wind/current drift hindcast and forecast, and the "
        "correlation layer that ranks candidate vessels against a spill's "
        "estimated release point and time."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------
class AisPing(BaseModel):
    speed: float = Field(..., description="Speed over ground, knots")
    course: float = Field(..., description="Course over ground, degrees")
    rot: float = Field(..., description="Rate of turn")
    msg_type: int = 1
    status: int = 0
    accuracy: int = 1
    course_diff: float = 0.0
    rot_diff: float = 0.0
    speed_diff: float = 0.0
    lat_diff: float = 0.0
    long_diff: float = 0.0


class TrackPoint(BaseModel):
    latitude: float
    longitude: float
    speed: float
    course: float
    rot: float
    timestamp: Optional[datetime] = None


class TrajectoryRequest(BaseModel):
    history: List[TrackPoint] = Field(..., description=f"Exactly {SEQ_LEN} pings, oldest first")
    actual_next: Optional[TrackPoint] = None


class AttributionRequest(BaseModel):
    spill_lat: float
    spill_lon: float
    spill_observed_at: Optional[datetime] = Field(
        None, description="UTC time of the satellite overpass. Required for "
                          "hindcast-anchored attribution (use_hindcast=True).")
    radius_km: float = 15.0
    window_hours: float = 6.0
    use_hindcast: bool = Field(
        True, description="Anchor the search on the hindcast-estimated release "
                          "point/time instead of the raw observed position. "
                          "Requires spill_observed_at.")
    hours_back: float = Field(12.0, description="Hindcast horizon in hours, if use_hindcast.")
    pings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Raw AIS rows. Omit to use the built-in Mauritius AOI feed.")


class DriftRequest(BaseModel):
    lat: float
    lon: float
    at: datetime = Field(..., description="UTC time of the observation.")
    hours: float = Field(12.0, description="Hours to trace back (hindcast) or forward (forecast).")
    use_live_wind: bool = True
    use_live_current: bool = True


class SpillAgeRequest(BaseModel):
    oil_area_km2: float
    oil_pixel_fraction: float
    slick_count: int = 1
    backscatter_contrast: Optional[float] = None


# --------------------------------------------------------------------------
# Health and administration
# --------------------------------------------------------------------------
@app.get("/health", tags=["system"])
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": __version__, **get_registry().status()}


@app.post("/admin/warmup", tags=["system"])
def warmup(keys: Optional[List[str]] = Body(None)) -> Dict[str, Any]:
    """Preload models so the first real request does not absorb the cost."""
    try:
        timings = get_registry().warmup(*(keys or ()))
    except Exception as exc:
        raise HTTPException(500, f"Warmup failed: {exc}") from exc
    return {"loaded": timings}


@app.get("/models", tags=["system"])
def models() -> Dict[str, Any]:
    return {
        "registry": get_registry().status(),
        "ais_anomaly": ais_mod.model_card(),
        "trajectory": traj_mod.model_card(),
        "limitations": [
            "Hindcast/forecast use real wind and ocean-current data from the "
            "Open-Meteo Forecast and Marine APIs when reachable, falling back "
            "to a Mauritius-AOI climatology otherwise; check each result's "
            "data_source field to see which applied.",
            "Hindcast origin is a probability estimate with a growing "
            "uncertainty radius, not an exact point -- attribution widens its "
            "search radius by that uncertainty but can still miss the true "
            "origin, especially far from shore or long after release.",
            "The trajectory model is valid only inside the Mauritius AOI it was "
            "normalised for.",
            "The anomaly detector has 0.41 recall: it nominates vessels for "
            "review and cannot clear one.",
            "Spill area figures are approximate -- the 512x512 output is resampled to the source resolution before measuring, but the resample carries error.",
        ],
    }


# --------------------------------------------------------------------------
# SAR
# --------------------------------------------------------------------------
@app.post("/sar/segment", tags=["sar"])
async def segment(
    file: UploadFile = File(..., description="SAR scene, JPG or PNG"),
    pixel_resolution_m: float = Query(10.0, gt=0, le=100),
) -> Dict[str, Any]:
    raw = await file.read()
    try:
        image = sar_mod.read_image(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    handle = get_registry()["sar"]
    result = handle.run(lambda m: sar_mod.segment(m, image, pixel_resolution_m))
    return {
        "filename": file.filename,
        "source_shape": list(image.shape),
        **result.summary(),
        "slicks": result.slicks,
    }


@app.get("/sar/legend", tags=["sar"])
def legend() -> List[Dict[str, Any]]:
    return sar_mod.legend()


# --------------------------------------------------------------------------
# AIS anomaly
# --------------------------------------------------------------------------
@app.post("/ais/score", tags=["ais"])
def score_ping(ping: AisPing) -> Dict[str, Any]:
    """Score one fully-specified ping. All 11 features must be supplied."""
    model, scaler = get_registry()["ais_anomaly"].get()
    result = ais_mod.score_row(model, scaler, ping.model_dump())
    return {**result.as_dict(), "guidance": ais_mod.model_card()["reading"]}


@app.post("/ais/score-track", tags=["ais"])
def score_track(pings: List[Dict[str, Any]] = Body(...)) -> Dict[str, Any]:
    """
    Score a whole track. Raw AIS rows are accepted -- the five `_diff` features
    are derived per vessel, so callers need not precompute them.
    """
    if not pings:
        raise HTTPException(400, "No pings supplied.")
    df = pd.DataFrame(pings)
    model, scaler = get_registry()["ais_anomaly"].get()
    try:
        scored = ais_mod.score_frame(model, scaler, df)
    except KeyError as exc:
        raise HTTPException(400, f"Missing column: {exc}") from exc

    return {
        "pings": len(scored),
        "flagged": int(scored["is_anomaly"].sum()),
        "threshold": AE_THRESHOLD,
        "vessels": ais_mod.vessel_rollup(scored).to_dict(orient="records"),
        "detail": scored[["mmsi", "anomaly_score", "is_anomaly", "severity",
                          "top_feature"]].to_dict(orient="records"),
    }


@app.get("/ais/features", tags=["ais"])
def features() -> Dict[str, Any]:
    return {
        "order": FEATURE_ORDER,
        "note": ("Order is positional, not by name. The scaler is mandatory: the "
                 "model was trained on standardised features and returns nonsense "
                 "on raw input."),
    }


# --------------------------------------------------------------------------
# Trajectory
# --------------------------------------------------------------------------
@app.post("/trajectory/predict", tags=["trajectory"])
def predict(request: TrajectoryRequest) -> Dict[str, Any]:
    if len(request.history) != SEQ_LEN:
        raise HTTPException(400, f"history must contain exactly {SEQ_LEN} pings, "
                                 f"got {len(request.history)}.")
    history = pd.DataFrame([p.model_dump() for p in request.history])
    assessment = traj_mod.assess_inputs(history)
    if not assessment.usable:
        raise HTTPException(422, {"error": "Input outside the model's operating envelope.",
                                  "detail": assessment.as_dict()})

    actual = request.actual_next.model_dump() if request.actual_next else None
    model = get_registry()["trajectory"].get()
    result = traj_mod.predict_next_position(model, history, actual_next=actual, strict=False)
    return result.as_dict()


# --------------------------------------------------------------------------
# Drift -- hindcast / forecast / age
# --------------------------------------------------------------------------
@app.post("/drift/hindcast", tags=["drift"])
def drift_hindcast(request: DriftRequest) -> Dict[str, Any]:
    """Trace a slick backward to estimate its release point and time."""
    hc = drift.hindcast(
        request.lat, request.lon, request.at,
        hours_back=request.hours,
        use_live_wind=request.use_live_wind,
        use_live_current=request.use_live_current,
    )
    return {
        **hc.summary(),
        "track": [s.as_dict() for s in hc.steps],
    }


@app.post("/drift/forecast", tags=["drift"])
def drift_forecast(request: DriftRequest, n_ensemble: int = Query(20, ge=1, le=200)) -> Dict[str, Any]:
    """Predict where a slick will spread forward as a probabilistic ensemble."""
    fc = drift.forecast(
        request.lat, request.lon, request.at,
        hours_forward=request.hours,
        n_ensemble=n_ensemble,
        use_live_wind=request.use_live_wind,
        use_live_current=request.use_live_current,
    )
    return fc.summary()


@app.post("/drift/age", tags=["drift"])
def drift_age(request: SpillAgeRequest) -> Dict[str, Any]:
    """Heuristic slick-age estimate from SAR-observable properties."""
    est = drift.estimate_age(
        request.oil_area_km2, request.oil_pixel_fraction,
        slick_count=request.slick_count,
        backscatter_contrast=request.backscatter_contrast,
    )
    return est.as_dict()


# --------------------------------------------------------------------------
# Attribution
# --------------------------------------------------------------------------
@app.post("/attribution/rank", tags=["attribution"])
def rank(request: AttributionRequest) -> Dict[str, Any]:
    model, scaler = get_registry()["ais_anomaly"].get()

    if request.pings:
        scored = ais_mod.score_frame(model, scaler, pd.DataFrame(request.pings))
        observed_at = request.spill_observed_at
    else:
        scenario = build_scenario()
        scored = ais_mod.score_frame(model, scaler, scenario["ais"])
        observed_at = request.spill_observed_at or scenario["grounding_utc"]

    if request.use_hindcast and observed_at is not None:
        out = fusion.attribute_from_observation(
            scored, request.spill_lat, request.spill_lon, observed_at,
            radius_km=request.radius_km,
            window_hours=request.window_hours,
            hours_back=request.hours_back,
        )
        results = out["results"]
        return {
            "summary": fusion.summarise(results),
            "candidates": [r.as_dict() for r in results],
            "weights": fusion.WEIGHTS,
            "search_anchor": out["search_anchor"],
            "hindcast": out["hindcast"],
            "caveat": out["note"] + " " + fusion.drift_caveat(hindcasted=True),
        }

    results = fusion.attribute(
        scored, request.spill_lat, request.spill_lon,
        observed_at=observed_at,
        radius_km=request.radius_km,
        window_hours=request.window_hours,
    )
    return {
        "summary": fusion.summarise(results),
        "candidates": [r.as_dict() for r in results],
        "weights": fusion.WEIGHTS,
        "search_anchor": "observed_position",
        "caveat": fusion.drift_caveat(hindcasted=False),
    }


@app.get("/scenario", tags=["scenario"])
def scenario(limit: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    """The real Mauritius AOI AIS feed, for clients without their own."""
    sc = build_scenario()
    incident = dict(sc["incident"])
    if incident.get("grounding_utc") is not None:
        incident["grounding_utc"] = incident["grounding_utc"].isoformat()
    incident.pop("observed", None)          # carries a raw datetime
    return {
        "incident": incident,
        "provenance": sc["provenance"],
        "spill_position": sc["spill_position"],
        "vessels": sc["vessels"].astype(str).to_dict(orient="records"),
        "ais_sample": sc["ais"].head(limit).to_dict(orient="records"),
        "total_pings": len(sc["ais"]),
    }

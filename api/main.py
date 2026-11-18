"""PathSnap evaluator API.

The model/decoder integration is deliberately injected through
PATHSNAP_PIPELINE_FACTORY; no synthetic predictions or guessed model APIs.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DATA_ROOT = Path(os.getenv("PATHSNAP_DATA_ROOT", "data"))
PARTITION = os.getenv("PATHSNAP_UI_PARTITION", "student_test")
PIPELINE_FACTORY = os.getenv("PATHSNAP_PIPELINE_FACTORY", "")
ALLOWED_PARTITIONS = {"student_test", "evaluator_test"}

app = FastAPI(title="PathSnap Evaluator API", version="0.1.0")
origins = [x.strip() for x in os.getenv("PATHSNAP_CORS_ORIGINS", "http://localhost:8501").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    trip_id: str = Field(pattern=r"^\d{8}$")

class PipelineUnavailable(RuntimeError):
    pass

def _partition_root() -> Path:
    if PARTITION not in ALLOWED_PARTITIONS:
        raise HTTPException(status_code=500, detail="PATHSNAP_UI_PARTITION must be student_test or evaluator_test")
    return DATA_ROOT / PARTITION

def _trip_dir(trip_id: str) -> Path:
    trip_dir = (_partition_root() / trip_id).resolve()
    root = _partition_root().resolve()
    if root not in trip_dir.parents or not trip_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} was not found in {PARTITION}")
    return trip_dir

def _load_pipeline() -> Any:
    if not PIPELINE_FACTORY:
        raise PipelineUnavailable(
            "Model/decoder adapter is not configured. Set PATHSNAP_PIPELINE_FACTORY "
            "to 'package.module:create_pipeline' after coordinating the real teammate interfaces."
        )
    try:
        module_name, factory_name = PIPELINE_FACTORY.split(":", 1)
        factory = getattr(importlib.import_module(module_name), factory_name)
        return factory()
    except Exception as exc:
        raise PipelineUnavailable(f"Could not initialize prediction pipeline: {exc}") from exc

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "partition": PARTITION, "pipeline": "configured" if PIPELINE_FACTORY else "not_configured"}

@app.get("/trips")
def list_trips() -> dict[str, Any]:
    root = _partition_root()
    if not root.is_dir():
        raise HTTPException(status_code=503, detail=f"Dataset partition directory is missing: {root}")
    trips = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return {"partition": PARTITION, "trip_ids": trips}

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    trip_dir = _trip_dir(request.trip_id)
    required = [f"{request.trip_id}.track", f"{request.trip_id}.nodes", f"{request.trip_id}.arcs"]
    missing = [name for name in required if not (trip_dir / name).is_file()]
    if missing:
        raise HTTPException(status_code=422, detail=f"Trip data is incomplete: {', '.join(missing)}")
    try:
        pipeline = _load_pipeline()
        result = pipeline.predict_trip(trip_id=request.trip_id, trip_dir=trip_dir)
    except PipelineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {exc}") from exc
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Pipeline must return a dictionary")
    # Never serialize reference labels, raw reference routes, or server-side truth.
    forbidden = {"ground_truth", "reference_route", "true_route", "labels", "y_true"}
    safe_result = {k: v for k, v in result.items() if k not in forbidden}
    return {"trip_id": request.trip_id, "partition": PARTITION, **safe_result}

""""PathSnap evaluator API.

The classifier receives only GPS/network inputs. Reference routes are kept in
this server process and are passed only to the separate evaluation hook.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ALLOWED_PARTITIONS = {"student_test", "evaluator_test"}
FORBIDDEN_FIELDS = {
    "ground_truth",
    "reference_route",
    "true_route",
    "labels",
    "y_true",
    "truth",
    "target",
}

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
    trip_id: str = Field(min_length=8, max_length=8, pattern=r"^\d{8}$")

class PipelineUnavailable(RuntimeError):
    pass

def _partition_root() -> Path:
    partition = os.getenv("PATHSNAP_UI_PARTITION", "student_test")
    if partition not in ALLOWED_PARTITIONS:
        raise HTTPException(status_code=500, detail="PATHSNAP_UI_PARTITION must be student_test or evaluator_test")
    return Path(os.getenv("PATHSNAP_DATA_ROOT", "data")) / partition


def _partition_name() -> str:
    return os.getenv("PATHSNAP_UI_PARTITION", "student_test")


def _pipeline_factory_name() -> str:
    return os.getenv("PATHSNAP_PIPELINE_FACTORY", "").strip()

def _trip_files(trip_id: str) -> dict[str, Path]:
    root = _partition_root().resolve()
    trip_dir = (root / trip_id).resolve()
    if root not in trip_dir.parents or not trip_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} was not found in {_partition_name()}")
    files = {
        "track": trip_dir / f"{trip_id}.track",
        "nodes": trip_dir / f"{trip_id}.nodes",
        "arcs": trip_dir / f"{trip_id}.arcs",
        "reference_route": trip_dir / f"{trip_id}.route",
    }
    missing = [str(path.name) for key, path in files.items() if key != "reference_route" and not path.is_file()]
    if missing:
        raise HTTPException(status_code=422, detail=f"Trip data is incomplete: {', '.join(missing)}")
    return files

def _load_pipeline() -> Any:
    pipeline_factory = _pipeline_factory_name()
    if not pipeline_factory:
        raise PipelineUnavailable(
            "Model/decoder adapter is not configured. Set PATHSNAP_PIPELINE_FACTORY "
            "to 'package.module:create_pipeline' after coordinating the real teammate interfaces."
        )
    try:
        module_name, factory_name = pipeline_factory.split(":", 1)
        if not module_name or not factory_name:
            raise ValueError("factory must use package.module:create_pipeline syntax")
        factory = getattr(importlib.import_module(module_name), factory_name)
        pipeline = factory()
        if not hasattr(pipeline, "predict_trip") or not callable(pipeline.predict_trip):
            raise TypeError("factory must return an object with callable predict_trip")
        return pipeline
    except Exception as exc:
        raise PipelineUnavailable(f"Could not initialize prediction pipeline: {exc}") from exc

@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "partition": _partition_name(),
        "pipeline": "configured" if _pipeline_factory_name() else "not_configured",
    }

@app.get("/trips")
def list_trips() -> dict[str, Any]:
    root = _partition_root()
    if not root.is_dir():
        raise HTTPException(status_code=503, detail=f"Dataset partition directory is missing: {root}")
    trips = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    return {"partition": _partition_name(), "trip_ids": trips}


def _without_reference_labels(value: Any) -> Any:
    """Remove truth-bearing fields recursively before data leaves the server."""
    if isinstance(value, Mapping):
        return {
            key: _without_reference_labels(item)
            for key, item in value.items()
            if str(key).lower() not in FORBIDDEN_FIELDS
        }
    if isinstance(value, list):
        return [_without_reference_labels(item) for item in value]
    if isinstance(value, tuple):
        return [_without_reference_labels(item) for item in value]
    return value

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    files = _trip_files(request.trip_id)
    try:
        pipeline = _load_pipeline()
        # IMPORTANT: the predictor gets only .track/.nodes/.arcs, never .route.
        prediction = pipeline.predict_trip(
            trip_id=request.trip_id,
            track_path=files["track"],
            nodes_path=files["nodes"],
            arcs_path=files["arcs"],
        )
        if not isinstance(prediction, dict):
            raise TypeError("predict_trip must return a dictionary")
        # Reference route is supplied only to the server-side evaluation hook.
        metrics = {}
        if files["reference_route"].is_file() and hasattr(pipeline, "evaluate_prediction"):
            evaluation_input = dict(prediction)
            evaluation_input["trip_id"] = request.trip_id
            metrics = pipeline.evaluate_prediction(
                prediction=evaluation_input,
                reference_route_path=files["reference_route"],
            )
        safe_prediction = _without_reference_labels(prediction)
        safe_metrics = _without_reference_labels(metrics) if isinstance(metrics, dict) else {}
        return {"trip_id": request.trip_id, "partition": _partition_name(), **safe_prediction, "metrics": safe_metrics}
    except PipelineUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction pipeline failed: {exc}") from exc

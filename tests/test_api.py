from pathlib import Path

import pytest
from fastapi import HTTPException

from api import main


def _make_trip(root: Path, trip_id: str = "00000000") -> Path:
    trip_dir = root / "student_test" / trip_id
    trip_dir.mkdir(parents=True)
    for suffix in ("track", "nodes", "arcs", "route"):
        (trip_dir / f"{trip_id}.{suffix}").write_text("0 0 0\n", encoding="utf-8")
    return trip_dir


def test_prediction_never_passes_reference_route_to_pipeline(monkeypatch, tmp_path):
    trip_dir = _make_trip(tmp_path)
    calls = {}

    class FakePipeline:
        def predict_trip(self, **kwargs):
            calls["predict"] = kwargs
            return {
                "predicted_route": [1, 2],
                "nested": {"ground_truth": [9], "value": "safe"},
            }

        def evaluate_prediction(self, **kwargs):
            calls["evaluate"] = kwargs
            return {"macro_f1": 0.8, "labels": ["secret"]}

    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PATHSNAP_UI_PARTITION", "student_test")
    monkeypatch.setattr(main, "_load_pipeline", lambda: FakePipeline())

    response = main.predict(main.PredictionRequest(trip_id="00000000"))

    assert set(calls["predict"]) == {"trip_id", "track_path", "nodes_path", "arcs_path"}
    assert calls["evaluate"]["reference_route_path"] == trip_dir / "00000000.route"
    assert response["nested"] == {"value": "safe"}
    assert response["metrics"] == {"macro_f1": 0.8}


def test_missing_pipeline_is_explicit(monkeypatch, tmp_path):
    _make_trip(tmp_path)
    monkeypatch.setenv("PATHSNAP_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("PATHSNAP_PIPELINE_FACTORY", raising=False)

    with pytest.raises(HTTPException) as error:
        main.predict(main.PredictionRequest(trip_id="00000000"))

    assert error.value.status_code == 503
    assert "not configured" in error.value.detail


def test_invalid_partition_is_rejected(monkeypatch):
    monkeypatch.setenv("PATHSNAP_UI_PARTITION", "evaluator_test/../student_test")

    with pytest.raises(HTTPException) as error:
        main.list_trips()

    assert error.value.status_code == 500

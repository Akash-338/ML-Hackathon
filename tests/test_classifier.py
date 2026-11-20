from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sys

from src.classifier.predict import ClassifierScorer
from scripts import train_classifier


def _candidate_rows(tmp_path: Path, partition: str) -> Path:
    path = tmp_path / f"{partition}.jsonl"
    rows = []
    for trip_number in range(12):
        trip_id = f"{trip_number:08d}"
        for observation in range(4):
            for rank in range(2):
                label = 1 if rank == 0 else 0
                if partition == "student_test":
                    label = None
                rows.append(
                    {
                        "schema_version": "pathsnap.candidates.v1",
                        "trip_id": trip_id,
                        "partition": partition,
                        "observation_index": observation,
                        "candidate_arc_id": rank,
                        "candidate_rank": rank,
                        "timestamp": observation,
                        "features": {
                            "gps_to_arc_distance_m": float(rank + 1),
                            "gps_heading_difference_deg": float(rank * 5),
                            "arc_length_m": 20.0,
                            "distance_from_arc_start_m": float(observation),
                            "fraction_along_arc": float(observation / 4),
                            "time_delta_s": 1.0,
                            "gps_speed_mps": 10.0,
                        },
                        "label": label,
                        "label_method": "test" if partition == "train" else None,
                    }
                )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def test_reader_rejects_evaluator_test(tmp_path: Path):
    with pytest.raises(ValueError, match="Evaluator-test"):
        train_classifier.read_jsonl(tmp_path / "evaluator_test.jsonl", "train")


def test_top1_accuracy_counts_wrong_and_missing_candidates():
    frame = pd.DataFrame({"trip_id": ["a"] * 5, "observation_index": [0, 0, 1, 1, 2],
                          "label": [1, 0, 1, 0, 0]})
    metrics = train_classifier.observation_metrics(frame, np.array([0.9, 0.1, 0.2, 0.8, 0.5]))
    assert metrics["top1_accuracy"] == pytest.approx(1 / 3)
    assert metrics["candidate_miss_rate"] == pytest.approx(1 / 3)


def test_group_cross_validation_and_five_models(tmp_path: Path):
    train_path = _candidate_rows(tmp_path, "train")
    test_path = _candidate_rows(tmp_path, "student_test")
    train = train_classifier.read_jsonl(train_path, "train")
    test = train_classifier.read_jsonl(test_path, "student_test")
    assert train["trip_id"].nunique() == 12
    assert set(train["label"].dropna().unique()) == {0, 1}
    assert test["label"].isna().all()
    assert len(train_classifier.make_models()) == 5
    for name, model in train_classifier.make_models().items():
        record = train_classifier.cross_validate(name, model, train, folds=3)
        assert record["attempt_status"] if "attempt_status" in record else record["macro_f1"] >= 0


def test_classifier_scorer(tmp_path: Path):
    train_path = _candidate_rows(tmp_path, "train")
    train = train_classifier.read_jsonl(train_path, "train")
    X = train_classifier.feature_frame(train)
    y = train["label"].astype(int)
    model = train_classifier.make_models()["logistic_regression"]
    model.fit(X, y)
    import joblib
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    joblib.dump(model, artifacts / "model.joblib")
    (artifacts / "metadata.json").write_text(json.dumps({"selected_model": "logistic_regression"}))
    scorer = ClassifierScorer(artifacts)
    probabilities = scorer.predict_frame(train.head(10))
    assert probabilities.shape == (10,)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_entrypoint_smoke_run(tmp_path: Path, monkeypatch):
    train_path = _candidate_rows(tmp_path, "train")
    test_path = _candidate_rows(tmp_path, "student_test")
    artifacts = tmp_path / "artifacts"
    results = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_classifier.py",
            "--train-jsonl",
            str(train_path),
            "--student-test-jsonl",
            str(test_path),
            "--artifacts-dir",
            str(artifacts),
            "--results-dir",
            str(results),
            "--cv-folds",
            "3",
            "--disable-bayesian",
        ],
    )
    assert train_classifier.main() == 0
    assert (artifacts / "model.joblib").exists()
    assert (artifacts / "metadata.json").exists()
    assert (results / "model_comparison.csv").exists()
    assert (results / "student_test_predictions.jsonl").exists()
    summary = json.loads((results / "student_test_summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_coverage_evaluated"] is False
    assert summary["candidate_miss_rate"] is None

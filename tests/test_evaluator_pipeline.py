import json

import pytest

from src.evaluator_pipeline import _candidate_rows


def test_prediction_top1_preserves_missing_observation(monkeypatch, tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from src import evaluator_pipeline
    from src.hmm.network import RoadNetwork
    from src.hmm.viterbi import HMMParams

    network = RoadNetwork(np.array([[0., 0.], [1., 0.]]), np.array([[0, 1]]))
    monkeypatch.setattr(evaluator_pipeline, "_load_inputs", lambda *args: (
        np.array([0., 1., 2.]), np.zeros((3, 2)), np.zeros((3, 2)),
        np.array([[0., 0.], [1., 0.]]), network))
    monkeypatch.setattr(evaluator_pipeline, "_candidate_rows", lambda *args: [
        {"observation_index": 0, "candidate_arc_id": 0},
        {"observation_index": 2, "candidate_arc_id": 0},
    ])
    pipeline = evaluator_pipeline.SavedArtifactPipeline.__new__(evaluator_pipeline.SavedArtifactPipeline)
    pipeline.scorer = SimpleNamespace(predict_frame=lambda frame: np.array([0.9, 0.8]))
    pipeline.candidates_path = tmp_path / "candidates.jsonl"
    pipeline.params = HMMParams()
    pipeline._trip_inputs = {}
    result = pipeline.predict_trip(trip_id="00000000", track_path=tmp_path, nodes_path=tmp_path, arcs_path=tmp_path)
    assert result["classifier_top1_route"] == [0, -1, 0]


def test_candidate_rows_reject_evaluator_partition(tmp_path):
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "pathsnap.candidates.v1",
                "partition": "evaluator_test",
                "trip_id": "00000000",
                "observation_index": 0,
                "candidate_arc_id": 1,
                "features": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluator_test"):
        _candidate_rows(path, "00000000")


def test_candidate_rows_only_selects_student_trip(tmp_path):
    path = tmp_path / "candidates.jsonl"
    rows = [
        {
            "schema_version": "pathsnap.candidates.v1",
            "partition": "student_test",
            "trip_id": "00000000",
            "observation_index": 0,
            "candidate_arc_id": 1,
            "features": {"arc_length_m": 2},
        },
        {
            "schema_version": "pathsnap.candidates.v1",
            "partition": "student_test",
            "trip_id": "00000001",
            "observation_index": 0,
            "candidate_arc_id": 2,
            "features": {"arc_length_m": 3},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = _candidate_rows(path, "00000000")

    assert result[0]["candidate_arc_id"] == 1
    assert result[0]["arc_length_m"] == 2

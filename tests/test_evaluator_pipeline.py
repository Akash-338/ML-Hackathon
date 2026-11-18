import json

import pytest

from src.evaluator_pipeline import _candidate_rows


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

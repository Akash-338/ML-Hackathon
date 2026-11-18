"""Candidate and Classifier Score Adapters for Pipeline Integration.

This module provides adapter functions to bridge teammates' interfaces to the
decoder contract:
- Akash K (Data pipeline & candidate generation)
- Akash S (Classifier & candidate ranking scores)
- Amar (FastAPI / UI backend)

Decoder Contract Expectation:
- candidates: List[np.ndarray], length T, each item containing trip-local arc_ids for step t.
- scores: List[np.ndarray], length T, each item containing probabilities aligned 1:1 with candidates[t].
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def adapt_dataframe_to_decoder(
    df: Any,
    T: int,
    step_col: str = "step",
    arc_col: str = "arc_id",
    score_col: str = "score",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert a tabular DataFrame (e.g. from Akash K/S) to the decoder contract.

    Assumes df has columns for time step index, candidate arc ID, and probability/score.
    If an observation has no candidate rows, it produces empty arrays for that step.
    """
    candidates: List[List[int]] = [[] for _ in range(T)]
    scores: List[List[float]] = [[] for _ in range(T)]

    # Handle pandas DataFrame or dict of columns
    if hasattr(df, "itertuples"):
        for row in df.itertuples():
            t = getattr(row, step_col)
            if 0 <= t < T:
                candidates[t].append(int(getattr(row, arc_col)))
                scores[t].append(float(getattr(row, score_col)))
    else:
        # Fallback dictionary of lists
        steps = df[step_col]
        arcs = df[arc_col]
        scs = df[score_col]
        for t, arc, s in zip(steps, arcs, scs):
            if 0 <= t < T:
                candidates[t].append(int(arc))
                scores[t].append(float(s))

    out_candidates = [np.array(c, dtype=np.int64) for c in candidates]
    out_scores = [np.array(s, dtype=np.float64) for s in scores]
    return out_candidates, out_scores


def adapt_dict_to_decoder(
    data: Dict[int, Union[Dict[str, Any], Sequence[Tuple[int, float]]]],
    T: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert a dictionary mapping step_idx -> candidates/scores to decoder contract."""
    candidates: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    for t in range(T):
        if t not in data:
            candidates.append(np.empty(0, dtype=np.int64))
            scores.append(np.empty(0, dtype=np.float64))
            continue

        item = data[t]
        if isinstance(item, dict):
            c = np.asarray(item.get("candidates", item.get("arc_ids", [])), dtype=np.int64)
            s = np.asarray(item.get("scores", item.get("probs", [])), dtype=np.float64)
        elif isinstance(item, (list, tuple)):
            # list of (arc_id, score) pairs
            if len(item) == 0:
                c = np.empty(0, dtype=np.int64)
                s = np.empty(0, dtype=np.float64)
            else:
                c = np.array([x[0] for x in item], dtype=np.int64)
                s = np.array([x[1] for x in item], dtype=np.float64)
        else:
            c = np.empty(0, dtype=np.int64)
            s = np.empty(0, dtype=np.float64)

        candidates.append(c)
        scores.append(s)

    return candidates, scores


def adapt_matrices_to_decoder(
    candidate_matrix: np.ndarray,
    score_matrix: np.ndarray,
    pad_value: int = -1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert 2D dense arrays of shape (T, K) with padding to decoder contract."""
    candidate_matrix = np.asarray(candidate_matrix)
    score_matrix = np.asarray(score_matrix)
    if candidate_matrix.ndim != 2 or score_matrix.shape != candidate_matrix.shape:
        raise ValueError(
            "candidate_matrix and score_matrix must be 2D arrays with the same shape"
        )
    T = len(candidate_matrix)
    candidates: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    for t in range(T):
        valid_mask = candidate_matrix[t] != pad_value
        c = candidate_matrix[t][valid_mask].astype(np.int64)
        s = score_matrix[t][valid_mask].astype(np.float64)
        candidates.append(c)
        scores.append(s)

    return candidates, scores


def load_candidate_jsonl_scores(
    candidate_jsonl: Union[str, Path],
    partition: str,
    artifact_dir: Union[str, Path],
    trip_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]:
    """Load the upstream Issue #2 classifier output contract for decoder evaluation.

    Reads `pathsnap.candidates.v1` JSONL, scores each candidate row with the saved
    model artifact, and groups probabilities by trip and observation. Training and
    student-test partitions only; evaluator-test paths and labels are rejected.
    """
    import json

    import joblib
    import pandas as pd

    candidate_path = Path(candidate_jsonl)
    path_text = str(candidate_path).replace("\\", "/").lower()
    if partition not in {"train", "student_test"} or "evaluator_test" in path_text:
        raise ValueError("Only train and student_test candidate partitions are permitted")

    artifact_path = Path(artifact_dir)
    model = joblib.load(artifact_path / "model.joblib")

    allowed_trip_ids = set(trip_ids) if trip_ids is not None else None
    rows: List[Dict[str, Any]] = []
    with candidate_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "pathsnap.candidates.v1":
                raise ValueError("Unsupported candidate JSONL schema")
            if row.get("partition") != partition:
                continue
            trip_id = str(row["trip_id"])
            if allowed_trip_ids is not None and trip_id not in allowed_trip_ids:
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                continue
            flat = dict(row)
            flat.update(features)
            flat["candidate_rank"] = row.get("candidate_rank")
            rows.append(flat)

    if not rows:
        raise ValueError(f"No candidate rows found for partition {partition}")

    frame = pd.DataFrame(rows)
    feature_names = (
        "gps_to_arc_distance_m",
        "gps_heading_difference_deg",
        "arc_length_m",
        "distance_from_arc_start_m",
        "fraction_along_arc",
        "time_delta_s",
        "gps_speed_mps",
        "candidate_rank",
    )
    for name in feature_names:
        if name not in frame:
            frame[name] = np.nan
    feature_frame = frame[list(feature_names)].apply(pd.to_numeric, errors="coerce")
    probabilities = model.predict_proba(feature_frame)
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
    frame["probability"] = probabilities[:, positive_index]

    grouped: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for trip_id in sorted(frame["trip_id"].astype(str).unique()):
        trip_frame = frame[frame["trip_id"].astype(str) == trip_id]
        max_observation = int(pd.to_numeric(trip_frame["observation_index"], errors="coerce").max())
        candidates: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(max_observation + 1)]
        scores: List[np.ndarray] = [np.empty(0, dtype=np.float64) for _ in range(max_observation + 1)]
        for observation_index, observation_frame in trip_frame.groupby("observation_index", sort=True):
            observation_frame = observation_frame.sort_values("candidate_rank")
            candidates[int(observation_index)] = observation_frame["candidate_arc_id"].to_numpy(dtype=np.int64)
            scores[int(observation_index)] = observation_frame["probability"].to_numpy(dtype=np.float64)
        grouped[trip_id] = (candidates, scores)

    return grouped

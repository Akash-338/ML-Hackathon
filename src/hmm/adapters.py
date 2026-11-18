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

"""DEVELOPMENT STAND-IN ONLY: Candidate Arc Generator and Classifier Simulator.

WARNING: This module is a temporary development stand-in used for tuning,
evaluation, and testing until upstream teammate deliverables are ready:
- Akash K: Upstream Candidate Generation Pipeline
- Akash S: Trained Candidate Classifier & Ranking Scores

The HMM Viterbi decoder (src/hmm/viterbi.py) is completely decoupled from this
module and only consumes candidate arc IDs and probabilities via the standard
decoder contract.

Mechanism:
1. Builds a spatial STRtree index over all road arcs in the trip's network.
2. For each GPS observation, queries candidate road arcs within a metric search radius.
3. Computes exact perpendicular point-to-segment distances (meters).
4. Selects the top K nearest arcs and converts distances to simulated classifier
   probabilities via Softmax(-distance / sigma).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from .network import RoadNetwork


def generate_dev_candidates(
    gps_xy: np.ndarray,
    network: RoadNetwork,
    k: int = 15,
    search_radius: float = 80.0,
    sigma: float = 12.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate K nearest candidate arcs and simulated classifier probabilities.

    Args:
        gps_xy: Shape (T, 2) array of GPS coordinates in projected meters.
        network: RoadNetwork instance for the trip.
        k: Maximum number of nearest candidate arcs per GPS point (default 15).
        search_radius: Maximum distance in meters to search for candidates (default 80m).
        sigma: Distance scaling parameter for softmax scores (default 12m).

    Returns:
        Tuple of (candidates, scores):
        - candidates: List of length T, each item is an np.ndarray of candidate arc_ids.
        - scores: List of length T, each item is an np.ndarray of probabilities summing to 1.0.
    """
    T = len(gps_xy)
    if T == 0:
        return [], []
    if network.num_arcs == 0:
        return ([np.empty(0, dtype=np.int64) for _ in range(T)],
                [np.empty(0, dtype=np.float64) for _ in range(T)])

    candidates: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    from_pts = network.nodes_xy[network.arcs[:, 0]]
    to_pts = network.nodes_xy[network.arcs[:, 1]]
    segment = to_pts - from_pts
    lengths_sq = np.sum(segment * segment, axis=1)

    for t, point in enumerate(gps_xy):
        fractions = np.zeros(network.num_arcs, dtype=np.float64)
        nonzero = lengths_sq > 0
        fractions[nonzero] = np.sum(
            (point - from_pts[nonzero]) * segment[nonzero],
            axis=1,
        ) / lengths_sq[nonzero]
        fractions = np.clip(fractions, 0.0, 1.0)
        projected = from_pts + fractions[:, np.newaxis] * segment
        dists = np.linalg.norm(point - projected, axis=1)
        within_radius = dists <= search_radius
        unique_arcs = np.flatnonzero(within_radius)
        if len(unique_arcs) == 0:
            unique_arcs = np.array([int(np.argmin(dists))], dtype=np.int64)
        if len(unique_arcs) > k:
            top_k_indices = np.argsort(dists[unique_arcs])[:k]
            unique_arcs = unique_arcs[top_k_indices]
            dists = dists[unique_arcs]
        else:
            dists = dists[unique_arcs]

        logits = -dists / max(sigma, 1e-3)
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

        candidates.append(unique_arcs.astype(np.int64))
        scores.append(probs.astype(np.float64))

    return candidates, scores

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
import shapely

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
    if T == 0 or network.num_arcs == 0:
        return [], []

    # 1. Build shapely LineStrings for all arcs
    from_pts = network.nodes_xy[network.arcs[:, 0]]
    to_pts = network.nodes_xy[network.arcs[:, 1]]
    line_coords = np.stack([from_pts, to_pts], axis=1)
    arc_lines = shapely.linestrings(line_coords)

    # 2. Build spatial STRtree index
    tree = shapely.STRtree(arc_lines)

    # 3. Create shapely Points for GPS observations
    gps_pts = shapely.points(gps_xy)

    # 4. Range query using buffers
    buffered_pts = shapely.buffer(gps_pts, search_radius)
    pt_indices, arc_indices = tree.query(buffered_pts)

    # Group candidate arcs by point index
    grouped_arcs: List[List[int]] = [[] for _ in range(T)]
    for p_idx, a_idx in zip(pt_indices, arc_indices):
        grouped_arcs[p_idx].append(int(a_idx))

    candidates: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    for t in range(T):
        arc_list = grouped_arcs[t]

        # If no arcs found within search_radius, fall back to single nearest arc
        if len(arc_list) == 0:
            nearest_match = tree.query_nearest(gps_pts[t])
            # nearest_match can be [pt_idx, geom_idx] or just [geom_idx]
            if isinstance(nearest_match, np.ndarray) and nearest_match.ndim == 2:
                arc_list = [int(nearest_match[1, 0])]
            elif isinstance(nearest_match, np.ndarray) and nearest_match.ndim == 1:
                arc_list = [int(nearest_match[0])]
            else:
                arc_list = [int(nearest_match)]

        # Unique arc IDs
        unique_arcs = np.unique(arc_list)

        # Compute exact point-to-segment distances
        candidate_geoms = arc_lines[unique_arcs]
        dists = shapely.distance(gps_pts[t], candidate_geoms)

        # Sort by distance and take top K
        if len(unique_arcs) > k:
            top_k_indices = np.argsort(dists)[:k]
            unique_arcs = unique_arcs[top_k_indices]
            dists = dists[top_k_indices]

        # Convert distances to softmax probabilities: P ~ exp(-dist / sigma)
        # Shift logits for numerical stability
        logits = -dists / max(sigma, 1e-3)
        logits_shifted = logits - np.max(logits)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits)

        candidates.append(unique_arcs.astype(np.int64))
        scores.append(probs.astype(np.float64))

    return candidates, scores

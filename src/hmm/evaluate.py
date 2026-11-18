"""Evaluation Metrics and Alignment for Route Reconstruction.

This module defines evaluation functions to benchmark Classifier Top-1 vs HMM Viterbi:
1. Ground Truth Alignment (align_gps_to_route):
   Maps each GPS point to a reference arc from the ground-truth .route sequence via
   spatial nearest-arc projection.
   Limitation: Point-to-segment projection can occasionally select a neighboring arc
   on the route at overlapping loops, overpasses, or dense intersections.
2. Observation-level Top-1 Accuracy:
   Fraction of GPS points where predicted arc == reference arc.
3. Candidate Miss Rate:
   Fraction of GPS points where the true reference arc is NOT present in the candidate set.
   (Theoretical error ceiling for any candidate-based decoder).
4. Route Continuity:
   Fraction of consecutive distinct predicted arcs that are physically connected in
   the directed road network (arc A connects to arc B if A.to_node == B.from_node).
5. Runtime:
   Decoding execution time in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shapely

from .network import RoadNetwork


@dataclass
class TripMetrics:
    """Evaluation metrics for a single trip."""
    trip_id: str
    num_points: int
    top1_accuracy: Optional[float]
    hmm_accuracy: Optional[float]
    delta_accuracy: Optional[float]
    top1_continuity: float
    hmm_continuity: float
    delta_continuity: float
    candidate_miss_rate: Optional[float]
    top1_runtime_s: float
    hmm_runtime_s: float
    status: str  # "helped", "hurt", or "neutral"


def align_gps_to_route(
    gps_xy: np.ndarray,
    route_arc_ids: np.ndarray,
    network: RoadNetwork,
) -> np.ndarray:
    """Align each GPS observation to its reference ground-truth arc from .route.

    Method:
    Constructs LineStrings for the ordered arcs in route_arc_ids and queries
    the nearest route arc for each GPS observation via STRtree spatial index.

    Limitation:
    Because .route contains only traversed arc IDs without per-second timestamps,
    spatial projection provides the most objective reference label. However, at
    tight switchbacks, overpasses, or intersections, a GPS observation may project
    closer to an adjacent segment of the route than the exact one traversed at
    that second.
    """
    T = len(gps_xy)
    if T == 0 or len(route_arc_ids) == 0:
        return np.full(T, -1, dtype=np.int64)

    valid_route_mask = (
        (route_arc_ids >= 0)
        & (route_arc_ids < network.num_arcs)
    )
    valid_route_ids = route_arc_ids[valid_route_mask]
    if len(valid_route_ids) == 0:
        return np.full(T, -1, dtype=np.int64)

    # Build lines for route arcs
    from_pts = network.nodes_xy[network.arcs[valid_route_ids, 0]]
    to_pts = network.nodes_xy[network.arcs[valid_route_ids, 1]]
    line_coords = np.stack([from_pts, to_pts], axis=1)
    route_lines = shapely.linestrings(line_coords)

    tree = shapely.STRtree(route_lines)
    gps_pts = shapely.points(gps_xy)

    nearest_indices = tree.query_nearest(gps_pts, all_matches=False)
    # Handle shapely query_nearest return format: shape (2, T) or (T,)
    nearest_indices = np.asarray(nearest_indices)
    if nearest_indices.ndim == 2:
        r_indices = nearest_indices[-1]
    else:
        r_indices = nearest_indices

    r_indices = np.asarray(r_indices, dtype=np.int64).reshape(-1)
    if len(r_indices) != T:
        return np.full(T, -1, dtype=np.int64)
    ref_arcs = valid_route_ids[r_indices]
    return ref_arcs.astype(np.int64)


def compute_route_continuity(arc_ids: np.ndarray, network: RoadNetwork) -> float:
    """Compute route continuity: fraction of consecutive distinct arcs that are connected.

    Formula:
    1. Collapse consecutive identical arcs (e.g. [1, 1, 2, 2, 3] -> [1, 2, 3]).
    2. Exclude invalid -1 predictions.
    3. Count transitions (u -> v) where network.are_connected(u, v) is True.
    4. Continuity = connected_transitions / total_transitions.
       (Returns 1.0 if total_transitions == 0).
    """
    valid = arc_ids[arc_ids >= 0]
    if len(valid) <= 1:
        return 1.0

    # Collapse consecutive identical arcs
    distinct = [int(valid[0])]
    for a in valid[1:]:
        if a != distinct[-1]:
            distinct.append(int(a))

    num_transitions = len(distinct) - 1
    if num_transitions <= 0:
        return 1.0

    connected = 0
    for i in range(num_transitions):
        u, v = distinct[i], distinct[i + 1]
        if network.are_connected(u, v):
            connected += 1

    return float(connected / num_transitions)


def evaluate_trip(
    trip_id: str,
    pred_arc_ids: np.ndarray,
    top1_arc_ids: np.ndarray,
    ref_arc_ids: Optional[np.ndarray],
    candidates: List[np.ndarray],
    network: RoadNetwork,
    top1_runtime_s: float,
    hmm_runtime_s: float,
) -> TripMetrics:
    """Evaluate classifier top-1 vs HMM Viterbi on a single trip."""
    T = len(pred_arc_ids)

    # Route continuity
    top1_cont = compute_route_continuity(top1_arc_ids, network)
    hmm_cont = compute_route_continuity(pred_arc_ids, network)
    delta_cont = hmm_cont - top1_cont

    top1_acc: Optional[float] = None
    hmm_acc: Optional[float] = None
    delta_acc: Optional[float] = None
    miss_rate: Optional[float] = None

    if ref_arc_ids is not None and len(ref_arc_ids) == T:
        valid_mask = ref_arc_ids >= 0
        if np.any(valid_mask):
            top1_acc = float(np.mean(top1_arc_ids[valid_mask] == ref_arc_ids[valid_mask]))
            hmm_acc = float(np.mean(pred_arc_ids[valid_mask] == ref_arc_ids[valid_mask]))
            delta_acc = hmm_acc - top1_acc

            # Candidate miss rate: fraction of observations where reference arc is NOT in candidates
            misses = 0
            for t in range(T):
                if ref_arc_ids[t] >= 0:
                    if len(candidates[t]) == 0 or (ref_arc_ids[t] not in candidates[t]):
                        misses += 1
            miss_rate = float(misses / T)

    # Status classification: helped / hurt / neutral
    if delta_acc is not None:
        if delta_acc > 0.001:
            status = "helped"
        elif delta_acc < -0.001:
            status = "hurt"
        elif delta_cont > 0.001:
            status = "helped"
        elif delta_cont < -0.001:
            status = "hurt"
        else:
            status = "neutral"
    else:
        if delta_cont > 0.001:
            status = "helped"
        elif delta_cont < -0.001:
            status = "hurt"
        else:
            status = "neutral"

    return TripMetrics(
        trip_id=trip_id,
        num_points=T,
        top1_accuracy=top1_acc,
        hmm_accuracy=hmm_acc,
        delta_accuracy=delta_acc,
        top1_continuity=top1_cont,
        hmm_continuity=hmm_cont,
        delta_continuity=delta_cont,
        candidate_miss_rate=miss_rate,
        top1_runtime_s=top1_runtime_s,
        hmm_runtime_s=hmm_runtime_s,
        status=status,
    )

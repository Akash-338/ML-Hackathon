"""Log-space Vectorized Viterbi Decoder for GPS Route Reconstruction.

This module implements the core Hidden Markov Model (HMM) Viterbi algorithm
to decode the most likely sequence of road network arcs from noisy GPS points
and candidate classification scores.

Key Viva Concepts:
1. Hidden States: The road arc being traversed at observation t: arc = candidates[t][k].
2. Observations: The GPS coordinates (x, y) at timestamp t.
3. Emissions: Log-likelihood from the candidate classifier probability:
     log P(obs_t | arc) = emission_weight * log(classifier_score + eps)
4. Transitions: Physical plausibility based on road network distance vs GPS displacement:
     - Same arc: delta = 0 (highest probability).
     - Directly connected or reachable within max_hops: delta = |network_dist - gps_disp|.
       log P(arc_cur | arc_prev) = -delta / beta.
     - Unreachable: soft penalty (-invalid_transition_penalty), avoiding hard rejection.
5. Viterbi Recurrence (log space):
     V[t, j] = max_i (V[t-1, i] + log_trans(i -> j)) + log_emit(j)
     backpointer[t, j] = argmax_i (V[t-1, i] + log_trans(i -> j))
6. Segment Handling: Restarts cleanly on time gaps (> gap_threshold_s) or empty candidates.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .network import RoadNetwork


@dataclass
class HMMParams:
    """Hyperparameters for HMM map-matching."""
    beta: float = 10.0                         # Scale factor for transition distance penalty (meters)
    emission_weight: float = 1.0               # Weight multiplying emission log-probabilities
    invalid_transition_penalty: float = 50.0   # Soft log-space penalty for disconnected/unreachable transitions
    max_hops: int = 5                          # Maximum network graph hops for shortest path search
    max_meters: float = 2000.0                 # Distance cutoff for shortest path search (meters)
    gap_threshold_s: float = 15.0              # Time gap in seconds to trigger segment restart
    eps: float = 1e-6                          # Small constant to prevent log(0) and division by zero

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "HMMParams":
        """Load parameters from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save parameters to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class DecodeResult:
    """Result of Viterbi route decoding for a trip."""
    pred_arc_ids: np.ndarray      # Shape (T,), decoded arc_ids (-1 where impossible)
    source: List[str]             # Shape (T,), "viterbi" | "fallback_top1" | "no_candidates"
    segment_breaks: List[int]     # Step indices where a new independent segment began
    log_prob: float               # Sum of log-probabilities of decoded paths across segments


def decode_trip(
    timestamps: Sequence[float] | np.ndarray,
    gps_xy: np.ndarray,
    candidates: Sequence[Sequence[int] | np.ndarray],
    scores: Sequence[Sequence[float] | np.ndarray],
    network: RoadNetwork,
    params: Optional[HMMParams] = None,
) -> DecodeResult:
    """Decode the most likely sequence of road arcs for a GPS trajectory.

    Args:
        timestamps: (T,) ordered timestamps in seconds.
        gps_xy: (T, 2) projected metric coordinates in meters.
        candidates: List of length T. Each item is an array/list of candidate arc_ids for step t.
        scores: List of length T. Each item is an array/list of classifier probabilities aligned
                1:1 with candidates[t].
        network: RoadNetwork instance providing graph connectivity and distances.
        params: HMMParams instance. If None, default hyperparameters are used.

    Returns:
        DecodeResult with predicted arc IDs, source tags, segment breaks, and log probability.
    """
    if params is None:
        params = HMMParams()

    timestamps = np.asarray(timestamps, dtype=np.float64)
    gps_xy = np.asarray(gps_xy, dtype=np.float64)
    T = len(timestamps)

    if timestamps.ndim != 1:
        raise ValueError("timestamps must be a one-dimensional sequence")
    if gps_xy.shape != (T, 2):
        raise ValueError(f"gps_xy must have shape ({T}, 2), got {gps_xy.shape}")
    if len(candidates) != T:
        raise ValueError(f"candidates must have length {T}, got {len(candidates)}")
    if len(scores) != T:
        raise ValueError(f"scores must have length {T}, got {len(scores)}")
    if not np.all(np.isfinite(gps_xy)):
        raise ValueError("gps_xy contains NaN or infinite coordinates")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("timestamps contains NaN or infinite values")
    if T > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be ordered monotonically")
    if params.beta <= 0:
        raise ValueError("HMMParams.beta must be positive")
    if params.emission_weight < 0:
        raise ValueError("HMMParams.emission_weight must be non-negative")
    if params.invalid_transition_penalty < 0:
        raise ValueError("HMMParams.invalid_transition_penalty must be non-negative")
    if params.gap_threshold_s < 0:
        raise ValueError("HMMParams.gap_threshold_s must be non-negative")
    if params.eps <= 0:
        raise ValueError("HMMParams.eps must be positive")

    pred_arc_ids = np.full(T, -1, dtype=np.int64)
    sources: List[str] = ["no_candidates"] * T
    segment_breaks: List[int] = []
    total_log_prob = 0.0

    if T == 0:
        return DecodeResult(pred_arc_ids, sources, segment_breaks, 0.0)

    # Pre-clean candidates and scores (handle NaNs, zero scores, convert to numpy)
    clean_candidates: List[np.ndarray] = []
    clean_scores: List[np.ndarray] = []
    for t in range(T):
        raw_candidates = candidates[t]
        raw_scores = scores[t]
        c_t = np.asarray(raw_candidates, dtype=np.int64).reshape(-1) if len(raw_candidates) > 0 else np.empty(0, dtype=np.int64)
        s_t = np.asarray(raw_scores, dtype=np.float64).reshape(-1) if len(raw_scores) > 0 else np.empty(0, dtype=np.float64)

        if len(c_t) != len(s_t):
            raise ValueError(
                f"candidates and scores must align at observation {t}: "
                f"got {len(c_t)} candidates and {len(s_t)} scores"
            )

        if len(c_t) > 0:
            # Clean NaNs or negative numbers in scores
            s_t = np.nan_to_num(s_t, nan=0.0, posinf=1.0, neginf=0.0)
            s_t = np.clip(s_t, 0.0, 1.0)
            valid_mask = (c_t >= 0) & (c_t < network.num_arcs)
            clean_candidates.append(c_t[valid_mask])
            s_t = s_t[valid_mask]
            if len(s_t) > 0:
                s_t = s_t / max(float(np.sum(s_t)), params.eps)
            clean_scores.append(s_t)
        else:
            clean_candidates.append(np.empty(0, dtype=np.int64))
            clean_scores.append(np.empty(0, dtype=np.float64))

    # Helper function to decode an isolated contiguous valid segment [seg_start, seg_end]
    def _decode_segment(seg_start: int, seg_end: int) -> float:
        seg_len = seg_end - seg_start + 1
        if seg_len <= 0:
            return 0.0

        # Handle T=1 single observation
        if seg_len == 1:
            t = seg_start
            best_idx = int(np.argmax(clean_scores[t]))
            pred_arc_ids[t] = clean_candidates[t][best_idx]
            sources[t] = "viterbi"
            emit_lp = params.emission_weight * np.log(clean_scores[t][best_idx] + params.eps)
            return float(emit_lp)

        # Multi-step segment: run vectorized log-space Viterbi
        # V[i]: best log probability ending at candidate i at current step
        # backpointers: list of 1D arrays recording argmax predecessor indices
        backpointers: List[Optional[np.ndarray]] = [None] * seg_len

        # Step 0 of segment: Initialize with pure emission probabilities
        c_0 = clean_candidates[seg_start]
        s_0 = clean_scores[seg_start]
        V_prev = params.emission_weight * np.log(s_0 + params.eps)

        for step_idx in range(1, seg_len):
            t = seg_start + step_idx
            t_prev = t - 1

            c_prev = clean_candidates[t_prev]
            c_curr = clean_candidates[t]
            s_curr = clean_scores[t]

            K_prev = len(c_prev)
            K_curr = len(c_curr)

            # Compute GPS displacement between consecutive observations
            gps_disp = float(np.linalg.norm(gps_xy[t] - gps_xy[t_prev]))

            # Compute log emissions for current step: Shape (K_curr,)
            log_emissions = params.emission_weight * np.log(s_curr + params.eps)

            # Build transition log-probability matrix: Shape (K_prev, K_curr)
            # T_mat[i, j] = log P(c_prev[i] -> c_curr[j])
            T_mat = np.empty((K_prev, K_curr), dtype=np.float64)

            for i in range(K_prev):
                arc_i = int(c_prev[i])
                for j in range(K_curr):
                    arc_j = int(c_curr[j])

                    if arc_i == arc_j:
                        # Same arc: delta = 0, highest transition likelihood
                        T_mat[i, j] = 0.0
                    else:
                        net_dist = network.shortest_path_distance(
                            arc_i,
                            arc_j,
                            max_hops=params.max_hops,
                            max_meters=params.max_meters,
                        )
                        if net_dist is not None:
                            # Reachable within road network: penalize discrepancy with GPS displacement
                            delta = abs(net_dist - gps_disp)
                            T_mat[i, j] = -delta / params.beta
                        else:
                            # Disconnected or unreachable within max_hops: apply soft penalty
                            T_mat[i, j] = -params.invalid_transition_penalty

            # Vectorized Viterbi update:
            # incoming_probs shape (K_prev, K_curr) = V_prev[:, None] + T_mat
            incoming_probs = V_prev[:, np.newaxis] + T_mat

            # Check if all incoming probabilities are invalid (-inf or NaN)
            if np.all(np.isnan(incoming_probs) | np.isneginf(incoming_probs)):
                # Fallback: restart step from raw emissions
                best_preds = np.zeros(K_curr, dtype=np.int64)
                V_curr = log_emissions
                backpointers[step_idx] = None
                sources[t] = "fallback_top1"
            else:
                best_preds = np.argmax(incoming_probs, axis=0)  # Shape (K_curr,)
                best_vals = np.max(incoming_probs, axis=0)      # Shape (K_curr,)
                V_curr = best_vals + log_emissions
                backpointers[step_idx] = best_preds

            V_prev = V_curr

        # Backtracking from terminal step of segment
        terminal_best_idx = int(np.argmax(V_prev))
        seg_log_prob = float(V_prev[terminal_best_idx])

        curr_k = terminal_best_idx
        for step_idx in range(seg_len - 1, -1, -1):
            t = seg_start + step_idx
            pred_arc_ids[t] = clean_candidates[t][curr_k]
            if sources[t] != "fallback_top1":
                sources[t] = "viterbi"

            bp = backpointers[step_idx]
            if bp is not None:
                curr_k = int(bp[curr_k])
            elif step_idx > 0:
                # If there was a restart within segment, pick argmax score
                curr_k = int(np.argmax(clean_scores[t - 1]))

        return seg_log_prob

    # Segment identification: break trajectory on empty candidates or time gaps
    seg_start: Optional[int] = None

    for t in range(T):
        # Edge case: No candidates at observation t
        if len(clean_candidates[t]) == 0:
            if seg_start is not None:
                # Close current segment
                total_log_prob += _decode_segment(seg_start, t - 1)
                seg_start = None
            pred_arc_ids[t] = -1
            sources[t] = "no_candidates"
            continue

        # Check for time gap with previous observation
        if seg_start is not None and t > 0:
            dt = timestamps[t] - timestamps[t - 1]
            if dt > params.gap_threshold_s:
                # Time gap exceeded: close segment and start new one
                total_log_prob += _decode_segment(seg_start, t - 1)
                seg_start = t
                segment_breaks.append(t)
                continue

        # Start a new segment if none active
        if seg_start is None:
            seg_start = t
            if t > 0:
                segment_breaks.append(t)

    # Close trailing segment if any
    if seg_start is not None:
        total_log_prob += _decode_segment(seg_start, T - 1)

    return DecodeResult(
        pred_arc_ids=pred_arc_ids,
        source=sources,
        segment_breaks=segment_breaks,
        log_prob=total_log_prob,
    )

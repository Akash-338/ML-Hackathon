"""Comprehensive PyTest suite for Viterbi Route Reconstruction Decoder.

Tests all required edge cases and behaviors on small, hand-crafted road networks:
1. Clean connected path vs noisy classifier distractor
2. Disconnected candidates with soft penalty fallback
3. Empty candidate set mid-trip (-1 output and segment restart)
4. Repeated arc / loop handling
5. Large time gap triggering segment break
6. T=1 single observation
7. Output shapes, order, and arc-id correspondence
8. NaN and zero score robustness
"""

import numpy as np
import pytest
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import DecodeResult, HMMParams, decode_trip


@pytest.fixture
def linear_network():
    """A simple 4-node, 3-arc linear road: 0->1 (arc 0), 1->2 (arc 1), 2->3 (arc 2).

    Also includes a parallel disconnected distractor: 4->5 (arc 3), 5->6 (arc 4).
    Nodes are placed 20 meters apart.
    """
    nodes = np.array([
        [0.0, 0.0],    # Node 0
        [20.0, 0.0],   # Node 1
        [40.0, 0.0],   # Node 2
        [60.0, 0.0],   # Node 3
        [0.0, 50.0],   # Node 4 (distractor)
        [20.0, 50.0],  # Node 5
        [40.0, 50.0],  # Node 6
    ])
    arcs = np.array([
        [0, 1],  # Arc 0 (length 20m)
        [1, 2],  # Arc 1 (length 20m)
        [2, 3],  # Arc 2 (length 20m)
        [4, 5],  # Arc 3 (length 20m)
        [5, 6],  # Arc 4 (length 20m)
    ])
    return RoadNetwork(nodes, arcs)


def test_clean_connected_path(linear_network):
    """Test that HMM overcomes noisy classifier distractor to recover a connected path."""
    timestamps = np.array([0.0, 1.0, 2.0])
    gps_xy = np.array([
        [10.0, 0.0],
        [30.0, 0.0],
        [50.0, 0.0],
    ])

    # At step 1, distractor arc 3 has slightly higher classifier score (0.6 vs 0.4)
    # A classifier-only argmax would pick [0, 3, 2] (disconnected jump to y=50 and back)
    # HMM should pick the fully connected path [0, 1, 2]
    candidates = [
        np.array([0, 3]),
        np.array([1, 3]),
        np.array([2, 4]),
    ]
    scores = [
        np.array([0.9, 0.1]),
        np.array([0.4, 0.6]),  # Noisy classifier prefers arc 3
        np.array([0.9, 0.1]),
    ]

    params = HMMParams(beta=10.0, emission_weight=1.0, invalid_transition_penalty=40.0)
    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network, params)

    assert isinstance(result, DecodeResult)
    np.testing.assert_array_equal(result.pred_arc_ids, [0, 1, 2])
    assert all(s == "viterbi" for s in result.source)
    assert np.isfinite(result.log_prob)


def test_disconnected_candidates(linear_network):
    """Test that unreachable candidates receive soft penalty rather than throwing errors."""
    timestamps = np.array([0.0, 1.0])
    gps_xy = np.array([[10.0, 0.0], [30.0, 50.0]])

    # Candidate at t=0 is arc 0; candidate at t=1 is arc 4 (disconnected from arc 0)
    candidates = [
        np.array([0]),
        np.array([4]),
    ]
    scores = [
        np.array([0.8]),
        np.array([0.8]),
    ]

    params = HMMParams(invalid_transition_penalty=30.0)
    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network, params)

    np.testing.assert_array_equal(result.pred_arc_ids, [0, 4])
    assert np.isfinite(result.log_prob)


def test_empty_candidate_set_mid_trip(linear_network):
    """Empty candidates at step t should yield -1, tag 'no_candidates', and restart Viterbi."""
    timestamps = np.array([0.0, 1.0, 2.0])
    gps_xy = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])

    candidates = [
        np.array([0]),
        np.array([]),  # Empty candidate set at step 1
        np.array([1]),
    ]
    scores = [
        np.array([0.9]),
        np.array([]),
        np.array([0.9]),
    ]

    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network)

    assert result.pred_arc_ids[0] == 0
    assert result.pred_arc_ids[1] == -1
    assert result.pred_arc_ids[2] == 1

    assert result.source[0] == "viterbi"
    assert result.source[1] == "no_candidates"
    assert result.source[2] == "viterbi"


def test_repeated_arc_and_loop():
    """Test repeated arcs (vehicle staying on same arc) and loops (a -> b -> a)."""
    # 2 nodes with bidirectional arcs: 0->1 (arc 0) and 1->0 (arc 1)
    nodes = np.array([[0.0, 0.0], [20.0, 0.0]])
    arcs = np.array([[0, 1], [1, 0]])
    net = RoadNetwork(nodes, arcs)

    # 1. Staying on same arc for 3 pings
    timestamps = np.array([0.0, 1.0, 2.0])
    gps_xy = np.array([[5.0, 0.0], [10.0, 0.0], [15.0, 0.0]])
    candidates = [np.array([0]), np.array([0]), np.array([0])]
    scores = [np.array([1.0]), np.array([1.0]), np.array([1.0])]

    result = decode_trip(timestamps, gps_xy, candidates, scores, net)
    np.testing.assert_array_equal(result.pred_arc_ids, [0, 0, 0])

    # 2. Loop: 0 -> 1 -> 0
    # At t=0 vehicle is at node 0, at t=1 vehicle reached node 1, at t=2 returned to node 0
    gps_xy_loop = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 0.0]])
    candidates_loop = [np.array([0, 1]), np.array([0, 1]), np.array([0, 1])]
    scores_loop = [np.array([0.8, 0.2]), np.array([0.2, 0.8]), np.array([0.8, 0.2])]
    result_loop = decode_trip(timestamps, gps_xy_loop, candidates_loop, scores_loop, net)
    np.testing.assert_array_equal(result_loop.pred_arc_ids, [0, 1, 0])


def test_large_time_gap(linear_network):
    """Time gap exceeding gap_threshold_s should trigger segment break."""
    timestamps = np.array([0.0, 1.0, 50.0, 51.0])  # 49s gap between index 1 and 2
    gps_xy = np.array([[10.0, 0.0], [15.0, 0.0], [35.0, 0.0], [45.0, 0.0]])

    candidates = [
        np.array([0]),
        np.array([0]),
        np.array([2]),
        np.array([2]),
    ]
    scores = [
        np.array([0.9]),
        np.array([0.9]),
        np.array([0.9]),
        np.array([0.9]),
    ]

    params = HMMParams(gap_threshold_s=15.0)
    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network, params)

    assert 2 in result.segment_breaks
    np.testing.assert_array_equal(result.pred_arc_ids, [0, 0, 2, 2])


def test_T_equals_1(linear_network):
    """Test trip with a single observation."""
    timestamps = np.array([0.0])
    gps_xy = np.array([[10.0, 0.0]])
    candidates = [np.array([0, 1])]
    scores = [np.array([0.3, 0.7])]

    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network)

    assert len(result.pred_arc_ids) == 1
    assert result.pred_arc_ids[0] == 1
    assert result.source[0] == "viterbi"
    assert np.isfinite(result.log_prob)


def test_order_and_arc_id_correspondence(linear_network):
    """Test output shapes, types, and alignment with input candidates."""
    T = 5
    timestamps = np.arange(T, dtype=float)
    gps_xy = np.zeros((T, 2))
    candidates = [np.array([0, 1, 2]) for _ in range(T)]
    scores = [np.array([0.2, 0.5, 0.3]) for _ in range(T)]

    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network)

    assert result.pred_arc_ids.shape == (T,)
    assert len(result.source) == T
    for t in range(T):
        assert result.pred_arc_ids[t] in candidates[t]


def test_nan_and_zero_scores(linear_network):
    """Test that zero scores and NaNs do not cause numerical exceptions."""
    timestamps = np.array([0.0, 1.0])
    gps_xy = np.array([[10.0, 0.0], [30.0, 0.0]])

    candidates = [
        np.array([0, 1]),
        np.array([1, 2]),
    ]
    # NaNs and zeros in scores
    scores = [
        np.array([np.nan, 0.0]),
        np.array([0.0, np.nan]),
    ]

    result = decode_trip(timestamps, gps_xy, candidates, scores, linear_network)

    assert len(result.pred_arc_ids) == 2
    assert not np.isnan(result.log_prob)
    assert not np.isinf(result.log_prob)


def test_mismatched_candidates_and_scores_raise(linear_network):
    timestamps = np.array([0.0])
    gps_xy = np.array([[10.0, 0.0]])

    with pytest.raises(ValueError, match="must align"):
        decode_trip(
            timestamps,
            gps_xy,
            [np.array([0, 1])],
            [np.array([0.5])],
            linear_network,
        )


def test_invalid_arc_ids_are_explicitly_empty(linear_network):
    timestamps = np.array([0.0, 1.0])
    gps_xy = np.array([[10.0, 0.0], [30.0, 0.0]])

    result = decode_trip(
        timestamps,
        gps_xy,
        [np.array([-1, 999]), np.array([1])],
        [np.array([0.5, 0.5]), np.array([1.0])],
        linear_network,
    )

    assert result.pred_arc_ids[0] == -1
    assert result.source[0] == "no_candidates"
    assert result.pred_arc_ids[1] == 1

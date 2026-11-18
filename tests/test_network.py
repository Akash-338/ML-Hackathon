"""Unit tests for RoadNetwork class."""

import numpy as np
import pytest
from src.hmm.network import RoadNetwork


def test_network_basic_connectivity():
    # 4 nodes in a straight line: 0 -> 1 -> 2 -> 3
    # Coordinates 10 meters apart: (0,0), (10,0), (20,0), (30,0)
    nodes = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
    ])
    # 3 arcs: arc 0: 0->1, arc 1: 1->2, arc 2: 2->3
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
    ])

    net = RoadNetwork(nodes, arcs)

    assert net.num_nodes == 4
    assert net.num_arcs == 3
    np.testing.assert_allclose(net.arc_lengths, [10.0, 10.0, 10.0])

    # Direct connectivity
    assert net.are_connected(0, 1) is True
    assert net.are_connected(1, 2) is True
    assert net.are_connected(0, 2) is False  # Not directly connected
    assert net.are_connected(2, 0) is False  # Directed: backwards is False

    # Shortest path distances
    assert net.shortest_path_distance(0, 0) == 0.0
    assert net.shortest_path_distance(0, 1) == 10.0  # length of arc 1
    # From arc 0 (0->1) to arc 2 (2->3): hops = 2 (arc 1 is intermediate edge 1->2)
    # Total distance = 10 (edge 1->2) + 10 (arc 2) = 20.0
    assert net.shortest_path_distance(0, 2, max_hops=3) == 20.0
    # Backward should be unreachable
    assert net.shortest_path_distance(2, 0) is None


def test_network_bounds_and_cache():
    nodes = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [20.0, 0.0],
        [30.0, 0.0],
    ])
    arcs = np.array([
        [0, 1],
        [1, 2],
        [2, 3],
    ])
    net = RoadNetwork(nodes, arcs)

    # If max_hops=1, cannot reach arc 2 from arc 0 (needs 2 hops)
    assert net.shortest_path_distance(0, 2, max_hops=1) is None

    # Clear cache and test max_meters limit
    net.clear_cache()
    assert net.shortest_path_distance(0, 2, max_hops=5, max_meters=15.0) is None
    net.clear_cache()
    assert net.shortest_path_distance(0, 2, max_hops=5, max_meters=25.0) == 20.0

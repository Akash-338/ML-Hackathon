"""Road Network representation for trip-local directed graphs.

This module models the directed road network of a single trip:
- Nodes: 2D coordinates in projected meters (UTM).
- Arcs: Directed road segments [from_node, to_node].
- Arc lengths: Euclidean length in meters.
- Arc adjacency and connectivity: arc A connects to arc B if A.to_node == B.from_node.
- Bounded shortest-path distance between arcs: computes road network travel distance
  with bounds on max_hops and max_meters, accelerated by an LRU / dictionary cache.
"""

from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

import numpy as np


class RoadNetwork:
    """Trip-local directed road network graph."""

    def __init__(
        self,
        nodes_xy: np.ndarray,
        arcs: np.ndarray,
        nodes_lonlat: Optional[np.ndarray] = None,
    ):
        """Initialize road network.

        Args:
            nodes_xy: Shape (N, 2) float array of node coordinates in projected meters.
            arcs: Shape (M, 2) int array of directed edges [from_node, to_node].
            nodes_lonlat: Optional shape (N, 2) float array of WGS84 [lon, lat].
        """
        self.nodes_xy = np.ascontiguousarray(nodes_xy, dtype=np.float64)
        self.arcs = np.ascontiguousarray(arcs, dtype=np.int64)
        self.nodes_lonlat = nodes_lonlat

        self.num_nodes = len(self.nodes_xy)
        self.num_arcs = len(self.arcs)

        # Precompute arc lengths in meters
        from_coords = self.nodes_xy[self.arcs[:, 0]]
        to_coords = self.nodes_xy[self.arcs[:, 1]]
        diff = to_coords - from_coords
        self.arc_lengths = np.sqrt(np.sum(diff ** 2, axis=1))

        # Build adjacency list: node -> list of (neighbor_node, arc_id, arc_length)
        self._adj: List[List[Tuple[int, int, float]]] = [[] for _ in range(self.num_nodes)]
        for arc_id in range(self.num_arcs):
            u = int(self.arcs[arc_id, 0])
            v = int(self.arcs[arc_id, 1])
            length = float(self.arc_lengths[arc_id])
            self._adj[u].append((v, arc_id, length))

        # Cache for shortest-path distances between pairs of arcs: (a, b) -> distance_or_None
        self._dist_cache: Dict[Tuple[int, int, int, float], Optional[float]] = {}

    def are_connected(self, arc_prev: int, arc_cur: int) -> bool:
        """Return True if arc_prev connects directly to arc_cur (1 hop).

        In a directed graph, arc A connects directly to arc B if and only if
        the head of A is the tail of B: A.to_node == B.from_node.
        """
        if arc_prev < 0 or arc_prev >= self.num_arcs:
            return False
        if arc_cur < 0 or arc_cur >= self.num_arcs:
            return False
        return bool(self.arcs[arc_prev, 1] == self.arcs[arc_cur, 0])

    def get_arc_geometry(self, arc_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (start_xy, end_xy) coordinates of an arc in meters."""
        u = self.arcs[arc_id, 0]
        v = self.arcs[arc_id, 1]
        return self.nodes_xy[u], self.nodes_xy[v]

    def shortest_path_distance(
        self,
        arc_prev: int,
        arc_cur: int,
        max_hops: int = 5,
        max_meters: float = 2000.0,
    ) -> Optional[float]:
        """Compute the bounded shortest-path network distance from arc_prev to arc_cur.

        Semantics:
        - If arc_prev == arc_cur: distance is 0.0 (vehicle remains on same arc).
        - If arc_prev connects directly to arc_cur (1 hop): distance is length(arc_cur).
        - If multi-hop: searches for shortest path from node A.to_node to B.from_node
          up to (max_hops - 1) intermediate edges and max_meters cutoff.
          Total distance = path_length(A.to_node -> B.from_node) + length(arc_cur).
        - If unreachable within bounds: returns None.

        Results are cached in self._dist_cache for instant repeated lookups.
        """
        cache_key = (arc_prev, arc_cur, int(max_hops), float(max_meters))
        if cache_key in self._dist_cache:
            return self._dist_cache[cache_key]

        # Case 1: Exact same arc
        if arc_prev == arc_cur:
            self._dist_cache[cache_key] = 0.0
            return 0.0

        target_from_node = int(self.arcs[arc_cur, 0])
        start_to_node = int(self.arcs[arc_prev, 1])
        cur_arc_len = float(self.arc_lengths[arc_cur])

        # Case 2: Directly connected (1 hop: arc_prev.to_node == arc_cur.from_node)
        if start_to_node == target_from_node:
            self._dist_cache[cache_key] = cur_arc_len
            return cur_arc_len

        # If max_hops <= 1 and not directly connected, cannot reach
        if max_hops <= 1:
            self._dist_cache[cache_key] = None
            return None

        # Case 3: Multi-hop bounded Dijkstra from start_to_node to target_from_node
        # Queue entries: (accumulated_dist, hops_used, current_node)
        pq: List[Tuple[float, int, int]] = [(0.0, 0, start_to_node)]
        best_dist: Dict[int, float] = {start_to_node: 0.0}
        found_dist: Optional[float] = None

        max_intermediate_hops = max_hops - 1
        intermediate_cutoff = max_meters - cur_arc_len

        while pq:
            d, hops, u = heapq.heappop(pq)

            if u == target_from_node:
                found_dist = d + cur_arc_len
                break

            if d > best_dist.get(u, float("inf")):
                continue

            if hops >= max_intermediate_hops or d >= intermediate_cutoff:
                continue

            for v, next_arc_id, edge_len in self._adj[u]:
                new_d = d + edge_len
                if new_d > intermediate_cutoff:
                    continue
                if new_d < best_dist.get(v, float("inf")):
                    best_dist[v] = new_d
                    heapq.heappush(pq, (new_d, hops + 1, v))

        self._dist_cache[cache_key] = found_dist
        return found_dist

    def clear_cache(self) -> None:
        """Clear distance lookup cache."""
        self._dist_cache.clear()

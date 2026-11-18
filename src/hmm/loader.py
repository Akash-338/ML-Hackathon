"""Trip Data Loader for Road Network and GPS Tracks.

This module provides a unified loader for PathSnap / Ufa / Germany GPS trips:
- Reads .track, .nodes, .arcs, and optional .route files.
- Verifies coordinate order: Column 0 is Longitude (X), Column 1 is Latitude (Y).
- Projects geographic coordinates (WGS84 lon/lat) to a metric UTM projection
  using pyproj so all distance calculations are performed in meters rather than
  Euclidean degrees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from pyproj import Transformer


@dataclass
class TripData:
    """Holds all raw and projected data for a single trip."""
    trip_id: str
    timestamps: np.ndarray      # Shape (T,), seconds (float64)
    gps_xy: np.ndarray          # Shape (T, 2), projected coordinates in meters [x, y]
    gps_lonlat: np.ndarray      # Shape (T, 2), raw WGS84 [lon, lat]
    nodes_xy: np.ndarray        # Shape (N, 2), projected coordinates in meters [x, y]
    nodes_lonlat: np.ndarray    # Shape (N, 2), raw WGS84 [lon, lat]
    arcs: np.ndarray            # Shape (M, 2), directed edges [from_node, to_node] (int64)
    route: Optional[np.ndarray] # Shape (L,), reference ground-truth traversed arc_ids (or None)
    epsg: int                   # UTM EPSG projection code used for metric distances


def get_utm_epsg(lon: float, lat: float) -> int:
    """Calculate the UTM EPSG code for a given WGS84 longitude and latitude.

    Standard UTM formula:
    - Zone number = floor((lon + 180) / 6) + 1 (zones 1 to 60)
    - Northern hemisphere: EPSG 32600 + zone
    - Southern hemisphere: EPSG 32700 + zone
    """
    zone = int((lon + 180.0) / 6.0) + 1
    zone = max(1, min(60, zone))
    return (32600 + zone) if lat >= 0.0 else (32700 + zone)


def find_trip_dir(trip_id: str, search_roots: Optional[list[str | Path]] = None) -> Path:
    """Locate the directory containing files for a given trip_id."""
    if search_roots is None:
        search_roots = [
            Path("data/train"),
            Path("data/student_test"),
            Path("ML-Hackathon/data/train"),
            Path("ML-Hackathon/data/student_test"),
            Path("data"),
            Path("ML-Hackathon/data"),
        ]

    for root in search_roots:
        root_path = Path(root)
        candidate = root_path / trip_id
        if candidate.is_dir() and (candidate / f"{trip_id}.track").exists():
            return candidate
        # Also check if files are directly inside root_path
        if (root_path / f"{trip_id}.track").exists():
            return root_path

    raise FileNotFoundError(f"Could not find data files for trip '{trip_id}' in {search_roots}")


def load_trip_data(
    trip_id: str,
    trip_dir: Optional[str | Path] = None,
) -> TripData:
    """Load and project a trip's track, nodes, arcs, and optional route.

    Coordinate Order Verification:
    - In .track: 3 whitespace-separated columns [lon, lat, timestamp_s].
    - In .nodes: 2 whitespace-separated columns [lon, lat]. Row index is node_id.
    - In .arcs: 2 whitespace-separated integer columns [from_node, to_node]. Row index is arc_id.
    - In .route: 1 integer column with ordered sequence of ground-truth arc_ids.

    Projection:
    - Automatically determines the local UTM zone from median coordinates.
    - Uses pyproj.Transformer to convert (lon, lat) -> (x, y) in metric meters.
    """
    if trip_dir is None:
        base_path = find_trip_dir(trip_id)
    else:
        base_path = Path(trip_dir)
        if (base_path / trip_id).is_dir():
            base_path = base_path / trip_id

    track_file = base_path / f"{trip_id}.track"
    nodes_file = base_path / f"{trip_id}.nodes"
    arcs_file = base_path / f"{trip_id}.arcs"
    route_file = base_path / f"{trip_id}.route"

    if not track_file.exists():
        raise FileNotFoundError(f"Missing track file: {track_file}")
    if not nodes_file.exists():
        raise FileNotFoundError(f"Missing nodes file: {nodes_file}")
    if not arcs_file.exists():
        raise FileNotFoundError(f"Missing arcs file: {arcs_file}")

    # Load raw files
    track_raw = np.loadtxt(track_file, dtype=np.float64)
    nodes_raw = np.loadtxt(nodes_file, dtype=np.float64)
    arcs_raw = np.loadtxt(arcs_file, dtype=np.int64)

    # Ensure 2D arrays even if 1 row
    if track_raw.ndim == 1:
        track_raw = track_raw.reshape(1, -1)
    if nodes_raw.ndim == 1:
        nodes_raw = nodes_raw.reshape(1, -1)
    if arcs_raw.ndim == 1:
        arcs_raw = arcs_raw.reshape(1, -1)

    route_raw: Optional[np.ndarray] = None
    if route_file.exists():
        try:
            route_data = np.loadtxt(route_file, dtype=np.int64)
            if route_data.ndim == 0:
                route_raw = np.array([int(route_data)], dtype=np.int64)
            else:
                route_raw = route_data
        except Exception:
            route_raw = None

    # Column separation
    # .track: lon = col 0, lat = col 1, timestamp = col 2
    gps_lonlat = track_raw[:, 0:2]
    timestamps = track_raw[:, 2]

    # .nodes: lon = col 0, lat = col 1
    nodes_lonlat = nodes_raw[:, 0:2]

    # Compute median center to find appropriate UTM EPSG
    median_lon = float(np.median(nodes_lonlat[:, 0]))
    median_lat = float(np.median(nodes_lonlat[:, 1]))
    epsg = get_utm_epsg(median_lon, median_lat)

    # Project coordinates: (lon, lat) -> (x, y) in meters
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    nodes_x, nodes_y = transformer.transform(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
    nodes_xy = np.column_stack([nodes_x, nodes_y])

    gps_x, gps_y = transformer.transform(gps_lonlat[:, 0], gps_lonlat[:, 1])
    gps_xy = np.column_stack([gps_x, gps_y])

    return TripData(
        trip_id=trip_id,
        timestamps=timestamps,
        gps_xy=gps_xy,
        gps_lonlat=gps_lonlat,
        nodes_xy=nodes_xy,
        nodes_lonlat=nodes_lonlat,
        arcs=arcs_raw,
        route=route_raw,
        epsg=epsg,
    )

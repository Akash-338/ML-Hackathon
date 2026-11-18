"""Interactive Folium Visualizer for Route Reconstruction.

Visualizes a trip with togglable layers:
1. Raw GPS Observations (blue markers with pings and timestamps)
2. Road Network (light gray network graph lines)
3. Reference Route (green ground-truth traversed line from .route)
4. Classifier-Only Top-1 (orange dashed line showing noisy jumps)
5. HMM Viterbi Decoded Route (purple line showing smooth continuous trajectory)

Includes:
- Folium LayerControl for interactive layer toggling
- Floating Legend explaining all colors
- Saved to results/hmm/maps/<trip_id>.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import folium
import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm.dev_candidates import generate_dev_candidates
from src.hmm.adapters import load_candidate_jsonl_scores
from src.hmm.loader import load_trip_data
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import HMMParams, decode_trip


def create_trip_map(
    trip_id: str,
    data_dir: Optional[str] = None,
    output_dir: str = "results/hmm/maps",
    params_path: Optional[str] = "configs/hmm_params.json",
    candidate_jsonl: Optional[str] = None,
    classifier_artifacts: Optional[str] = None,
    prediction_npz: Optional[str] = None,
) -> Path:
    """Generate and save an interactive Folium map for a single trip."""
    # 1. Load trip data
    trip = load_trip_data(trip_id, data_dir)
    network = RoadNetwork(trip.nodes_xy, trip.arcs, trip.nodes_lonlat)

    # 2. Generate candidates and predictions
    params = HMMParams.from_json(params_path) if params_path and Path(params_path).exists() else HMMParams()
    prediction_loaded = bool(prediction_npz and Path(prediction_npz).exists())
    if prediction_npz and Path(prediction_npz).exists():
        prediction = np.load(prediction_npz)
        top1_arcs = prediction["top1_arc_ids"].astype(np.int64)
        hmm_arcs = prediction["hmm_arc_ids"].astype(np.int64)
        candidates = [
            np.array([int(a)], dtype=np.int64) if int(a) >= 0 else np.empty(0, dtype=np.int64)
            for a in top1_arcs
        ]
        scores = [
            np.array([1.0], dtype=np.float64) if int(a) >= 0 else np.empty(0, dtype=np.float64)
            for a in top1_arcs
        ]
    elif candidate_jsonl and classifier_artifacts:
        partition = "student_test" if data_dir and "student_test" in str(data_dir).lower() else "train"
        scored = load_candidate_jsonl_scores(
            candidate_jsonl,
            partition,
            classifier_artifacts,
            trip_ids=[trip_id],
        )
        candidates, scores = scored[trip_id]
        if len(candidates) < len(trip.gps_xy):
            candidates = candidates + [np.empty(0, dtype=np.int64)] * (len(trip.gps_xy) - len(candidates))
            scores = scores + [np.empty(0, dtype=np.float64)] * (len(trip.gps_xy) - len(scores))
    else:
        candidates, scores = generate_dev_candidates(trip.gps_xy, network, k=15)

    if not prediction_loaded:
        # Classifier Top-1
        top1_arcs = np.array([
            candidates[t][int(np.argmax(scores[t]))] if len(candidates[t]) > 0 else -1
            for t in range(len(candidates))
        ], dtype=np.int64)

        # HMM Viterbi
        hmm_res = decode_trip(trip.timestamps, trip.gps_xy, candidates, scores, network, params)
        hmm_arcs = hmm_res.pred_arc_ids

    # 3. Initialize Folium Map centered on median GPS (lat, lon)
    # Recall: gps_lonlat is [lon, lat], so folium center is [lat, lon]
    center_lat = float(np.median(trip.gps_lonlat[:, 1]))
    center_lon = float(np.median(trip.gps_lonlat[:, 0]))

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # Layer 1: Road Network (Local candidate arcs to avoid rendering 100k lines)
    # Collect all unique candidate arcs and route arcs to render local road network
    relevant_arcs = set()
    for c in candidates:
        relevant_arcs.update(c.tolist())
    if trip.route is not None:
        relevant_arcs.update(trip.route.tolist())
    relevant_arcs.update(top1_arcs[top1_arcs >= 0].tolist())
    relevant_arcs.update(hmm_arcs[hmm_arcs >= 0].tolist())

    net_layer = folium.FeatureGroup(name="Road Network (Graph)", show=True)
    for arc_id in relevant_arcs:
        u = network.arcs[arc_id, 0]
        v = network.arcs[arc_id, 1]
        p_from = [trip.nodes_lonlat[u, 1], trip.nodes_lonlat[u, 0]]
        p_to = [trip.nodes_lonlat[v, 1], trip.nodes_lonlat[v, 0]]
        folium.PolyLine(
            locations=[p_from, p_to],
            color="#888888",
            weight=2,
            opacity=0.5,
            tooltip=f"Arc {arc_id}",
        ).add_to(net_layer)
    net_layer.add_to(fmap)

    # Layer 2: Reference Route (Ground Truth)
    if trip.route is not None and len(trip.route) > 0:
        route_layer = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        route_coords = []
        for arc_id in trip.route:
            u = network.arcs[arc_id, 0]
            v = network.arcs[arc_id, 1]
            route_coords.append([trip.nodes_lonlat[u, 1], trip.nodes_lonlat[u, 0]])
            route_coords.append([trip.nodes_lonlat[v, 1], trip.nodes_lonlat[v, 0]])
        folium.PolyLine(
            locations=route_coords,
            color="#2ca02c",  # Green
            weight=5,
            opacity=0.75,
            tooltip=f"Trip {trip_id} Reference Route",
        ).add_to(route_layer)
        route_layer.add_to(fmap)

    # Layer 3: Classifier-Only Prediction (Top-1)
    top1_layer = folium.FeatureGroup(name="Classifier Top-1 (Argmax)", show=False)
    top1_coords = []
    for arc_id in top1_arcs:
        if arc_id >= 0:
            u = network.arcs[arc_id, 0]
            v = network.arcs[arc_id, 1]
            top1_coords.append([trip.nodes_lonlat[u, 1], trip.nodes_lonlat[u, 0]])
            top1_coords.append([trip.nodes_lonlat[v, 1], trip.nodes_lonlat[v, 0]])
    if top1_coords:
        folium.PolyLine(
            locations=top1_coords,
            color="#ff7f0e",  # Orange
            weight=4,
            opacity=0.7,
            dash_array="5, 5",
            tooltip="Classifier Top-1",
        ).add_to(top1_layer)
    top1_layer.add_to(fmap)

    # Layer 4: HMM Viterbi Decoded Route
    hmm_layer = folium.FeatureGroup(name="HMM Viterbi Route", show=True)
    hmm_coords = []
    for arc_id in hmm_arcs:
        if arc_id >= 0:
            u = network.arcs[arc_id, 0]
            v = network.arcs[arc_id, 1]
            hmm_coords.append([trip.nodes_lonlat[u, 1], trip.nodes_lonlat[u, 0]])
            hmm_coords.append([trip.nodes_lonlat[v, 1], trip.nodes_lonlat[v, 0]])
    if hmm_coords:
        folium.PolyLine(
            locations=hmm_coords,
            color="#9467bd",  # Purple
            weight=4,
            opacity=0.9,
            tooltip="HMM Decoded Path",
        ).add_to(hmm_layer)
    hmm_layer.add_to(fmap)

    # Layer 5: Raw GPS Observations (blue markers)
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations", show=True)
    # Downsample points slightly if > 1500 to keep HTML lightweight
    stride = 1 if len(trip.gps_lonlat) <= 1500 else (len(trip.gps_lonlat) // 1000)
    for idx in range(0, len(trip.gps_lonlat), stride):
        lat = float(trip.gps_lonlat[idx, 1])
        lon = float(trip.gps_lonlat[idx, 0])
        ts = float(trip.timestamps[idx])
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#1f77b4",  # Blue
            fill=True,
            fill_color="#1f77b4",
            fill_opacity=0.8,
            tooltip=f"GPS #{idx} | t={ts:.1f}s",
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    # Floating Legend HTML
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 260px; height: 180px; 
                background-color: white; z-index:9999; font-size:13px;
                border:2px solid #555; padding: 10px; border-radius: 8px;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b>Trip {trip_id} Route Reconstruction</b><br>
        <span style="color:#1f77b4;">&#9679;</span> Raw GPS Observations<br>
        <span style="color:#888888;">&#9644;</span> Road Network Graph<br>
        <span style="color:#2ca02c; font-weight:bold;">&#9644;&#9644;</span> Reference Route (.route)<br>
        <span style="color:#ff7f0e; font-weight:bold;">&#9644; - -</span> Classifier Top-1 (Argmax)<br>
        <span style="color:#9467bd; font-weight:bold;">&#9644;&#9644;</span> HMM Viterbi Reconstructed<br>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))

    # Add LayerControl
    folium.LayerControl(collapsed=False).add_to(fmap)

    # Save HTML
    out_path = Path(output_dir) / f"{trip_id}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate interactive Folium map for a trip.")
    parser.add_argument("trip_id", type=str, help="Trip ID to visualize (e.g. 00000001 or 00000000)")
    parser.add_argument("--data-dir", type=str, default=None, help="Optional trip directory")
    parser.add_argument("--output-dir", type=str, default="results/hmm/maps", help="Output directory")
    parser.add_argument("--params", type=str, default="configs/hmm_params.json", help="Path to hmm_params.json")
    parser.add_argument("--candidate-jsonl", type=str, default=None, help="Permitted candidate JSONL")
    parser.add_argument("--classifier-artifacts", type=str, default=None, help="Classifier model artifact directory")
    parser.add_argument("--prediction-npz", type=str, default=None, help="Saved real comparison predictions")
    args = parser.parse_args()

    out_file = create_trip_map(
        trip_id=args.trip_id,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        params_path=args.params,
        candidate_jsonl=args.candidate_jsonl,
        classifier_artifacts=args.classifier_artifacts,
        prediction_npz=args.prediction_npz,
    )
    print(f"Map successfully generated at: {out_file}")


if __name__ == "__main__":
    main()

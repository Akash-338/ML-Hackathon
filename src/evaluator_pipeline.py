"""Saved classifier + HMM adapter used by the evaluator API.

The adapter reads only the trip's track/nodes/arcs during prediction. Candidate
rows and model artifacts are supplied through environment variables; reference
routes are opened only by ``evaluate_prediction``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer

from src.hmm.network import RoadNetwork


class SavedArtifactPipeline:
    """Run the saved candidate classifier followed by Anand's HMM decoder."""

    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        candidates_path: str | Path | None = None,
        hmm_params_path: str | Path | None = None,
    ):
        from src.classifier import ClassifierScorer
        from src.hmm.viterbi import HMMParams

        self.scorer = ClassifierScorer(
            artifact_dir or os.environ.get("PATHSNAP_CLASSIFIER_ARTIFACT_DIR", "artifacts/classifier")
        )
        self.candidates_path = Path(
            candidates_path or os.environ.get("PATHSNAP_CANDIDATES_JSONL", "artifacts/data-pipeline/candidates.jsonl")
        )
        params_path = hmm_params_path or os.environ.get(
            "PATHSNAP_HMM_PARAMS_PATH", "configs/hmm_params.json"
        )
        self.params = HMMParams.from_json(params_path)
        self._trip_inputs: dict[str, tuple[np.ndarray, np.ndarray, RoadNetwork]] = {}

    def predict_trip(
        self,
        *,
        trip_id: str,
        track_path: Path,
        nodes_path: Path,
        arcs_path: Path,
    ) -> dict[str, Any]:
        timestamps, gps_xy, gps_lonlat, nodes_lonlat, network = _load_inputs(
            track_path, nodes_path, arcs_path
        )
        import pandas as pd

        from src.hmm.adapters import adapt_dataframe_to_decoder
        from src.hmm.viterbi import decode_trip

        candidates = pd.DataFrame(_candidate_rows(self.candidates_path, trip_id))
        candidates["probability"] = self.scorer.predict_frame(candidates)
        candidate_arrays, score_arrays = adapt_dataframe_to_decoder(
            candidates.assign(step=candidates["observation_index"], arc_id=candidates["candidate_arc_id"]),
            len(timestamps),
            step_col="step",
            arc_col="arc_id",
            score_col="probability",
        )
        decoded = decode_trip(
            timestamps=timestamps,
            gps_xy=gps_xy,
            candidates=candidate_arrays,
            scores=score_arrays,
            network=network,
            params=self.params,
        )
        self._trip_inputs[trip_id] = (gps_xy, nodes_lonlat, network)
        route = decoded.pred_arc_ids.tolist()
        coordinates = [
            nodes_lonlat[int(network.arcs[arc_id, 0])].tolist()
            for arc_id in route
            if 0 <= arc_id < network.num_arcs
        ]
        return {
            "predicted_route": route,
            "route_coordinates": coordinates,
            "decoder_source": decoded.source,
            "segment_breaks": decoded.segment_breaks,
        }

    def evaluate_prediction(self, *, prediction: dict[str, Any], reference_route_path: Path) -> dict[str, float]:
        from src.hmm.evaluate import align_gps_to_route, compute_route_continuity

        route = np.loadtxt(reference_route_path, dtype=np.int64)
        route = np.atleast_1d(route)
        trip_id = str(prediction.get("trip_id", ""))
        if trip_id not in self._trip_inputs:
            return {}
        gps_xy, _, network = self._trip_inputs[trip_id]
        reference = align_gps_to_route(gps_xy, route, network)
        predicted = np.asarray(prediction.get("predicted_route", []), dtype=np.int64)
        valid = reference >= 0
        return {
            "hmm_accuracy": float(np.mean(predicted[valid] == reference[valid])) if np.any(valid) else 0.0,
            "hmm_continuity": compute_route_continuity(predicted, network),
        }


def create_pipeline() -> SavedArtifactPipeline:
    return SavedArtifactPipeline()


def _candidate_rows(path: Path, trip_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Candidate JSONL is missing: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("partition") == "evaluator_test":
                raise ValueError("evaluator_test candidate rows are forbidden")
            if row.get("partition") != "student_test" or str(row.get("trip_id")) != trip_id:
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"Missing features at {path}:{line_number}")
            flat = dict(row)
            flat.update(features)
            rows.append(flat)
    if not rows:
        raise ValueError(f"No student_test candidates found for trip {trip_id}")
    required = {"observation_index", "candidate_arc_id"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Candidate rows missing fields: {sorted(missing)}")
    return rows


def _load_inputs(track_path: Path, nodes_path: Path, arcs_path: Path):
    track = np.atleast_2d(np.loadtxt(track_path, dtype=np.float64))
    nodes = np.atleast_2d(np.loadtxt(nodes_path, dtype=np.float64))
    arcs = np.atleast_2d(np.loadtxt(arcs_path, dtype=np.int64))
    gps_lonlat = track[:, :2]
    nodes_lonlat = nodes[:, :2]
    zone = int((float(np.median(nodes_lonlat[:, 0])) + 180.0) / 6.0) + 1
    epsg = (32600 + max(1, min(60, zone))) if float(np.median(nodes_lonlat[:, 1])) >= 0 else (32700 + max(1, min(60, zone)))
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    node_x, node_y = transformer.transform(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
    gps_x, gps_y = transformer.transform(gps_lonlat[:, 0], gps_lonlat[:, 1])
    nodes_xy = np.column_stack((node_x, node_y))
    gps_xy = np.column_stack((gps_x, gps_y))
    return track[:, 2], gps_xy, gps_lonlat, nodes_lonlat, RoadNetwork(nodes_xy, arcs)

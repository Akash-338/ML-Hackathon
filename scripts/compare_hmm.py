"""Comprehensive HMM vs Classifier-Only Comparison Script.

Workflow:
1. Selects HMMParams (beta, emission_weight, invalid_transition_penalty) on training trips.
   Classifier scores may be in-sample, so this is not independent cross-validation.
2. Saves the optimal hyperparameters to configs/hmm_params.json and results/hmm/best_params.json.
3. Evaluates both Classifier-Only Top-1 (argmax) and Classifier + HMM Viterbi on:
   - Training trips
   - All 10 Student Test trips
4. Computes:
   - Observation-level Top-1 accuracy (via nearest-arc-on-route reference alignment)
   - Route continuity (fraction of consecutive distinct predicted arcs that are connected)
   - Candidate miss rate (theoretical error ceiling)
   - Runtime per trip (seconds)
5. Writes per-trip metrics, helped/hurt CSV, summary metrics JSON, provenance,
   and execution log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm.dev_candidates import generate_dev_candidates
from src.hmm.adapters import load_candidate_jsonl_scores
from src.hmm.evaluate import align_gps_to_route, evaluate_trip
from src.hmm.loader import TripData, load_trip_data
from src.hmm.network import RoadNetwork
from src.hmm.viterbi import DecodeResult, HMMParams, decode_trip


def setup_logger(log_file: Path) -> logging.Logger:
    """Configure logger to stream to both file and stdout."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("compare_hmm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def run_single_trip_eval(
    trip: TripData,
    params: HMMParams,
    classifier_candidates: Tuple[List[np.ndarray], List[np.ndarray]] | None = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Run classifier-only and HMM on a trip and compute all metrics."""
    network = RoadNetwork(trip.nodes_xy, trip.arcs)
    if classifier_candidates is None:
        cands, scores = generate_dev_candidates(trip.gps_xy, network, k=15)
    else:
        cands, scores = classifier_candidates
        if len(cands) < len(trip.gps_xy):
            cands = cands + [np.empty(0, dtype=np.int64)] * (len(trip.gps_xy) - len(cands))
            scores = scores + [np.empty(0, dtype=np.float64)] * (len(trip.gps_xy) - len(scores))

    # 1. Classifier-only argmax top-1
    t0 = time.time()
    top1_arcs = np.array([
        cands[t][int(np.argmax(scores[t]))] if len(cands[t]) > 0 else -1
        for t in range(len(cands))
    ], dtype=np.int64)
    top1_runtime = time.time() - t0

    # 2. Classifier + HMM Viterbi
    t0 = time.time()
    hmm_res = decode_trip(trip.timestamps, trip.gps_xy, cands, scores, network, params)
    hmm_runtime = time.time() - t0

    # 3. Ground truth alignment
    ref_arcs = None
    if trip.route is not None:
        ref_arcs = align_gps_to_route(trip.gps_xy, trip.route, network)

    m = evaluate_trip(
        trip_id=trip.trip_id,
        pred_arc_ids=hmm_res.pred_arc_ids,
        top1_arc_ids=top1_arcs,
        ref_arc_ids=ref_arcs,
        candidates=cands,
        network=network,
        top1_runtime_s=top1_runtime,
        hmm_runtime_s=hmm_runtime,
    )

    metrics_dict = {
        "trip_id": m.trip_id,
        "num_points": m.num_points,
        "top1_accuracy": m.top1_accuracy,
        "hmm_accuracy": m.hmm_accuracy,
        "delta_accuracy": m.delta_accuracy,
        "top1_continuity": m.top1_continuity,
        "hmm_continuity": m.hmm_continuity,
        "delta_continuity": m.delta_continuity,
        "candidate_miss_rate": m.candidate_miss_rate,
        "top1_runtime_s": m.top1_runtime_s,
        "hmm_runtime_s": m.hmm_runtime_s,
        "status": m.status,
    }
    return metrics_dict, top1_arcs, hmm_res.pred_arc_ids, ref_arcs


def tune_hyperparameters(
    train_trips: List[TripData],
    logger: logging.Logger,
    n_splits: int = 3,
    classifier_scores: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] | None = None,
) -> HMMParams:
    """Select decoder parameters on training trips using the supplied scores."""
    logger.info("Selecting HMM parameters on training trips (not independent validation).")
    if len(train_trips) < 2:
        logger.warning("Fewer than 2 training trips; using default HMMParams.")
        return HMMParams()
    n_splits = min(n_splits, len(train_trips))

    # Grid of candidate parameters
    param_grid = [
        {"beta": 5.0, "emission_weight": 1.0, "invalid_transition_penalty": 40.0},
        {"beta": 10.0, "emission_weight": 0.5, "invalid_transition_penalty": 50.0},
        {"beta": 10.0, "emission_weight": 1.0, "invalid_transition_penalty": 50.0},
        {"beta": 10.0, "emission_weight": 2.0, "invalid_transition_penalty": 50.0},
        {"beta": 20.0, "emission_weight": 1.0, "invalid_transition_penalty": 60.0},
        {"beta": 30.0, "emission_weight": 1.0, "invalid_transition_penalty": 80.0},
    ]

    trip_indices = np.arange(len(train_trips))

    best_score = -float("inf")
    best_config: Dict[str, Any] = param_grid[2]

    # Pre-generate candidate sets and networks for tuning speed
    logger.info(f"Pre-caching candidates for {len(train_trips)} tuning trips...")
    cached_trips = []
    for tr in train_trips:
        net = RoadNetwork(tr.nodes_xy, tr.arcs)
        if classifier_scores is None:
            cands, scs = generate_dev_candidates(tr.gps_xy, net, k=15)
        else:
            cands, scs = classifier_scores[tr.trip_id]
            if len(cands) < len(tr.gps_xy):
                missing = len(tr.gps_xy) - len(cands)
                cands = cands + [np.empty(0, dtype=np.int64) for _ in range(missing)]
                scs = scs + [np.empty(0, dtype=np.float64) for _ in range(missing)]
        ref = align_gps_to_route(tr.gps_xy, tr.route, net) if tr.route is not None else None
        cached_trips.append((tr, net, cands, scs, ref))

    for config in param_grid:
        params = HMMParams(**config)
        fold_scores = []

        for fold in range(n_splits):
            val_idx = trip_indices[fold::n_splits]
            val_accs = []
            val_conts = []

            for i in val_idx:
                tr, net, cands, scs, ref = cached_trips[i]
                res = decode_trip(tr.timestamps, tr.gps_xy, cands, scs, net, params)
                top1_arcs = np.array([
                    cands[t][int(np.argmax(scs[t]))] if len(cands[t]) > 0 else -1
                    for t in range(len(cands))
                ], dtype=np.int64)
                m = evaluate_trip(
                    trip_id=tr.trip_id,
                    pred_arc_ids=res.pred_arc_ids,
                    top1_arc_ids=top1_arcs,
                    ref_arc_ids=ref,
                    candidates=cands,
                    network=net,
                    top1_runtime_s=0.0,
                    hmm_runtime_s=0.0,
                )
                if m.hmm_accuracy is not None:
                    val_accs.append(m.hmm_accuracy)
                val_conts.append(m.hmm_continuity)

            mean_acc = float(np.mean(val_accs)) if val_accs else 0.0
            mean_cont = float(np.mean(val_conts)) if val_conts else 0.0
            # Composite validation objective: balance accuracy and continuity
            obj = 0.5 * mean_acc + 0.5 * mean_cont
            fold_scores.append(obj)

        cv_score = float(np.mean(fold_scores))
        logger.info(f"Params: {config} => training selection score: {cv_score:.4f}")

        if cv_score > best_score:
            best_score = cv_score
            best_config = config

    logger.info(f"Selected parameters: {best_config} (training selection score: {best_score:.4f})")
    return HMMParams(**best_config)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare HMM decoder vs Classifier Top-1.")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Directory with train trips")
    parser.add_argument("--test-dir", type=str, default="data/student_test", help="Directory with student_test trips")
    parser.add_argument("--num-train-tune", type=int, default=15, help="Number of train trips to use for CV tuning")
    parser.add_argument("--results-dir", type=str, default="results/hmm", help="Results directory")
    parser.add_argument("--configs-dir", type=str, default="configs", help="Configs directory")
    parser.add_argument("--candidate-jsonl", type=str, default="", help="Permitted candidate JSONL for real classifier scores")
    parser.add_argument("--classifier-artifacts", type=str, default="", help="Issue #2 classifier artifact directory")
    args = parser.parse_args()

    if bool(args.candidate_jsonl) != bool(args.classifier_artifacts):
        parser.error("Provide both --candidate-jsonl and --classifier-artifacts, or neither.")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = Path(args.configs_dir)
    configs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(results_dir / "run.log")
    logger.info("==================================================")
    logger.info("PathSnap Route Reconstruction: Classifier vs HMM")
    logger.info("==================================================")

    # 1. Load train trips for tuning
    train_dir = Path(args.train_dir)
    if not train_dir.is_dir():
        parser.error(
            f"Training partition is missing: {train_dir}. "
            "Provide --train-dir for permitted training data."
        )
    all_train_ids = sorted([
        d.name for d in train_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.track").exists()
    ])
    logger.info(f"Found {len(all_train_ids)} train trips in {train_dir}")

    # Use first N diverse train trips for tuning
    tune_ids = all_train_ids[:args.num_train_tune]
    logger.info(f"Loading {len(tune_ids)} train trips for parameter selection: {tune_ids}")
    train_trips = [load_trip_data(tid, train_dir) for tid in tune_ids]
    if not train_trips:
        parser.error("No training trips found for HMM parameter selection.")
    all_train_trips = [load_trip_data(tid, train_dir) for tid in all_train_ids]

    train_classifier_scores = None
    test_classifier_scores = None
    if args.candidate_jsonl and args.classifier_artifacts:
        logger.info("Loading real classifier probabilities from the upstream candidate/model contract")
        train_classifier_scores = load_candidate_jsonl_scores(
            args.candidate_jsonl,
            "train",
            args.classifier_artifacts,
            trip_ids=all_train_ids,
        )

    # 2. Select decoder parameters on training trips.
    best_params = tune_hyperparameters(
        train_trips,
        logger,
        n_splits=3,
        classifier_scores=train_classifier_scores,
    )

    # Save best parameters to configs/hmm_params.json and results/hmm/best_params.json
    best_params.to_json(configs_dir / "hmm_params.json")
    best_params.to_json(results_dir / "best_params.json")
    logger.info(f"Saved tuned parameters to {configs_dir / 'hmm_params.json'}")

    # 3. Evaluate on Student Test Trips (all 10)
    test_dir = Path(args.test_dir)
    if not test_dir.is_dir():
        parser.error(
            f"Student-test partition is missing: {test_dir}. "
            "Provide --test-dir for permitted student-test data."
        )
    test_ids = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.track").exists()
    ])
    logger.info(f"Found {len(test_ids)} student_test trips in {test_dir}: {test_ids}")
    if not test_ids:
        parser.error("No student-test trips found.")
    if args.candidate_jsonl and args.classifier_artifacts:
        test_classifier_scores = load_candidate_jsonl_scores(
            args.candidate_jsonl,
            "student_test",
            args.classifier_artifacts,
            trip_ids=test_ids,
        )

    all_records: List[Dict[str, Any]] = []
    helped_hurt_records: List[Dict[str, Any]] = []

    # Evaluate tuning train subset
    logger.info("--- Evaluating All 80 Training Trips ---")
    for tr in all_train_trips:
        classifier_candidates = (
            train_classifier_scores[tr.trip_id]
            if train_classifier_scores is not None
            else None
        )
        rec, top1_arcs, hmm_arcs, ref_arcs = run_single_trip_eval(
            tr,
            best_params,
            classifier_candidates,
        )
        rec["split"] = "train"
        all_records.append(rec)

        # Helped / Hurt span record
        helped_hurt_records.append({
            "split": "train",
            "trip_id": tr.trip_id,
            "num_points": rec["num_points"],
            "top1_accuracy": rec["top1_accuracy"],
            "hmm_accuracy": rec["hmm_accuracy"],
            "delta_accuracy": rec["delta_accuracy"],
            "top1_continuity": rec["top1_continuity"],
            "hmm_continuity": rec["hmm_continuity"],
            "delta_continuity": rec["delta_continuity"],
            "status": rec["status"],
            "explanation": (
                f"HMM accuracy {rec['delta_accuracy']:+.1%}, continuity {rec['delta_continuity']:+.1%}"
            ),
        })

    # Evaluate student_test trips
    logger.info("--- Evaluating Student Test Set ---")
    for tid in test_ids:
        tr = load_trip_data(tid, test_dir)
        classifier_candidates = (
            test_classifier_scores[tid]
            if test_classifier_scores is not None
            else None
        )
        rec, top1_arcs, hmm_arcs, ref_arcs = run_single_trip_eval(
            tr,
            best_params,
            classifier_candidates,
        )
        prediction_dir = results_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            prediction_dir / f"{tid}.npz",
            top1_arc_ids=top1_arcs,
            hmm_arc_ids=hmm_arcs,
            ref_arc_ids=(
                ref_arcs
                if ref_arcs is not None
                else np.full(len(tr.gps_xy), -1, dtype=np.int64)
            ),
        )
        rec["split"] = "student_test"
        all_records.append(rec)

        # Span detail on student test
        explanation = f"HMM accuracy {rec['delta_accuracy']:+.1%}, continuity {rec['delta_continuity']:+.1%}"
        if rec["delta_continuity"] > 0.20:
            explanation += "; strongly resolved disconnected classifier hops"

        helped_hurt_records.append({
            "split": "student_test",
            "trip_id": tr.trip_id,
            "num_points": rec["num_points"],
            "top1_accuracy": rec["top1_accuracy"],
            "hmm_accuracy": rec["hmm_accuracy"],
            "delta_accuracy": rec["delta_accuracy"],
            "top1_continuity": rec["top1_continuity"],
            "hmm_continuity": rec["hmm_continuity"],
            "delta_continuity": rec["delta_continuity"],
            "status": rec["status"],
            "explanation": explanation,
        })
        logger.info(
            f"Test Trip {tid}: Top1 Acc={rec['top1_accuracy']:.1%}, HMM Acc={rec['hmm_accuracy']:.1%} "
            f"(Delta={rec['delta_accuracy']:+.1%}), Top1 Cont={rec['top1_continuity']:.1%}, "
            f"HMM Cont={rec['hmm_continuity']:.1%} (Delta={rec['delta_continuity']:+.1%}), "
            f"Time={rec['hmm_runtime_s']:.2f}s, Status={rec['status']}"
        )

    # 4. Save CSV Outputs
    def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    write_csv(results_dir / "per_trip_metrics.csv", all_records)
    logger.info(f"Saved per-trip metrics to {results_dir / 'per_trip_metrics.csv'}")

    write_csv(results_dir / "helped_hurt.csv", helped_hurt_records)
    logger.info(f"Saved helped/hurt analysis to {results_dir / 'helped_hurt.csv'}")

    # 5. Compute Aggregate Summary for Student Test
    test_df = [r for r in all_records if r["split"] == "student_test"]

    def mean_field(records: List[Dict[str, Any]], field: str) -> float:
        values = [float(r[field]) for r in records if r[field] is not None]
        return float(np.mean(values)) if values else 0.0

    real_classifier_evaluation = bool(args.candidate_jsonl and args.classifier_artifacts)
    summary = {
        "evaluation_source": (
            "upstream_classifier_model_plus_hmm"
            if real_classifier_evaluation
            else "dev_distance_standin"
        ),
        "candidate_generator": (
            "src.hmm.adapters.load_candidate_jsonl_scores"
            if real_classifier_evaluation
            else "src.hmm.dev_candidates.generate_dev_candidates"
        ),
        "real_classifier_evaluation": real_classifier_evaluation,
        "evaluator_test_accessed": False,
        "num_student_test_trips": len(test_df),
        "total_test_observations": int(sum(r["num_points"] for r in test_df)),
        "mean_top1_accuracy": mean_field(test_df, "top1_accuracy"),
        "mean_hmm_accuracy": mean_field(test_df, "hmm_accuracy"),
        "mean_delta_accuracy": mean_field(test_df, "delta_accuracy"),
        "mean_top1_continuity": mean_field(test_df, "top1_continuity"),
        "mean_hmm_continuity": mean_field(test_df, "hmm_continuity"),
        "mean_delta_continuity": mean_field(test_df, "delta_continuity"),
        "mean_candidate_miss_rate": mean_field(test_df, "candidate_miss_rate"),
        "mean_top1_runtime_s": mean_field(test_df, "top1_runtime_s"),
        "mean_hmm_runtime_s": mean_field(test_df, "hmm_runtime_s"),
        "trips_helped": sum(r["status"] == "helped" for r in test_df),
        "trips_hurt": sum(r["status"] == "hurt" for r in test_df),
        "trips_neutral": sum(r["status"] == "neutral" for r in test_df),
        "best_params": best_params.__dict__,
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary metrics to {results_dir / 'summary.json'}")

    model_path = Path(args.classifier_artifacts) / "model.joblib" if real_classifier_evaluation else None
    metadata_path = Path(args.classifier_artifacts) / "metadata.json" if real_classifier_evaluation else None
    classifier_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path and metadata_path.is_file() else {}
    provenance = {
        "evaluation_source": summary["evaluation_source"],
        "candidate_generator": summary["candidate_generator"],
        "real_classifier_evaluation": summary["real_classifier_evaluation"],
        "evaluator_test_accessed": summary["evaluator_test_accessed"],
        "train_dir": str(train_dir),
        "student_test_dir": str(test_dir),
        "classifier_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest() if model_path else None,
        "classifier_model": classifier_metadata.get("selected_model", classifier_metadata.get("model")),
        "candidate_jsonl": str(args.candidate_jsonl) if real_classifier_evaluation else None,
        "tuning_trip_ids": tune_ids,
        "tuning_limitation": "Classifier scores on training trips may be in-sample; parameter selection is not independent cross-validation.",
        "notes": (
            "Scores were produced by the supplied classifier artifact "
            "from pathsnap.candidates.v1 train/student-test JSONL."
            if real_classifier_evaluation
            else (
                "Scores were generated by the distance-based development stand-in. "
                "No upstream model.joblib or candidate JSONL was available."
            )
        ),
    }
    with open(results_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    if provenance["evaluation_source"] == "dev_distance_standin":
        with open(results_dir / "standin_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    logger.info(f"Saved evaluation provenance to {results_dir / 'provenance.json'}")

    logger.info("==================================================")
    logger.info("STUDENT TEST AGGREGATE RESULTS SUMMARY")
    logger.info("==================================================")
    logger.info(f"Mean Classifier Top-1 Accuracy:   {summary['mean_top1_accuracy']:.2%}")
    logger.info(f"Mean HMM Viterbi Accuracy:         {summary['mean_hmm_accuracy']:.2%}")
    logger.info(f"Mean Delta Accuracy:               {summary['mean_delta_accuracy']:+.2%}")
    logger.info(f"Mean Classifier Route Continuity:  {summary['mean_top1_continuity']:.2%}")
    logger.info(f"Mean HMM Route Continuity:         {summary['mean_hmm_continuity']:.2%}")
    logger.info(f"Mean Delta Route Continuity:       {summary['mean_delta_continuity']:+.2%}")
    logger.info(f"Mean Candidate Miss Rate:          {summary['mean_candidate_miss_rate']:.2%}")
    logger.info(f"Mean HMM Runtime per Trip:         {summary['mean_hmm_runtime_s']:.3f} s")
    logger.info(f"Trips Helped: {summary['trips_helped']} / {len(test_df)} | Hurt: {summary['trips_hurt']} | Neutral: {summary['trips_neutral']}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()

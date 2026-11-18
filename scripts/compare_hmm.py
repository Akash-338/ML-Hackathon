"""Comprehensive HMM vs Classifier-Only Comparison Script.

Workflow:
1. Loads training trips and conducts Group-aware Cross-Validation (GroupKFold by trip)
   to tune HMMParams (beta, emission_weight, invalid_transition_penalty).
2. Saves the optimal hyperparameters to configs/hmm_params.json and results/hmm/best_params.json.
3. Evaluates both Classifier-Only Top-1 (argmax) and Classifier + HMM Viterbi on:
   - Training trips
   - All 10 Student Test trips
4. Computes:
   - Observation-level Top-1 accuracy (via nearest-arc-on-route reference alignment)
   - Route continuity (fraction of consecutive distinct predicted arcs that are connected)
   - Candidate miss rate (theoretical error ceiling)
   - Runtime per trip (seconds)
5. Writes per-trip metrics, helped/hurt CSV, summary metrics JSON, and execution log.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hmm.dev_candidates import generate_dev_candidates
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
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Run classifier-only and HMM on a trip and compute all metrics."""
    network = RoadNetwork(trip.nodes_xy, trip.arcs)
    cands, scores = generate_dev_candidates(trip.gps_xy, network, k=15)

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
) -> HMMParams:
    """Tune HMMParams via Group-aware Cross-Validation on training trips."""
    logger.info("Starting Group-aware Cross Validation for HMMParams on TRAIN...")

    # Grid of candidate parameters
    param_grid = [
        {"beta": 5.0, "emission_weight": 1.0, "invalid_transition_penalty": 40.0},
        {"beta": 10.0, "emission_weight": 0.5, "invalid_transition_penalty": 50.0},
        {"beta": 10.0, "emission_weight": 1.0, "invalid_transition_penalty": 50.0},
        {"beta": 10.0, "emission_weight": 2.0, "invalid_transition_penalty": 50.0},
        {"beta": 20.0, "emission_weight": 1.0, "invalid_transition_penalty": 60.0},
        {"beta": 30.0, "emission_weight": 1.0, "invalid_transition_penalty": 80.0},
    ]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    trip_indices = np.arange(len(train_trips))

    best_score = -float("inf")
    best_config: Dict[str, Any] = param_grid[2]

    # Pre-generate candidate sets and networks for tuning speed
    logger.info(f"Pre-caching candidates for {len(train_trips)} tuning trips...")
    cached_trips = []
    for tr in train_trips:
        net = RoadNetwork(tr.nodes_xy, tr.arcs)
        cands, scs = generate_dev_candidates(tr.gps_xy, net, k=15)
        ref = align_gps_to_route(tr.gps_xy, tr.route, net) if tr.route is not None else None
        cached_trips.append((tr, net, cands, scs, ref))

    for config in param_grid:
        params = HMMParams(**config)
        fold_scores = []

        for train_idx, val_idx in kf.split(trip_indices):
            val_accs = []
            val_conts = []

            for i in val_idx:
                tr, net, cands, scs, ref = cached_trips[i]
                res = decode_trip(tr.timestamps, tr.gps_xy, cands, scs, net, params)
                m = evaluate_trip(
                    trip_id=tr.trip_id,
                    pred_arc_ids=res.pred_arc_ids,
                    top1_arc_ids=res.pred_arc_ids,
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
        logger.info(f"Params: {config} => CV Score: {cv_score:.4f}")

        if cv_score > best_score:
            best_score = cv_score
            best_config = config

    logger.info(f"Optimal parameters found: {best_config} (CV Score: {best_score:.4f})")
    return HMMParams(**best_config)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare HMM decoder vs Classifier Top-1.")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Directory with train trips")
    parser.add_argument("--test-dir", type=str, default="data/student_test", help="Directory with student_test trips")
    parser.add_argument("--num-train-tune", type=int, default=15, help="Number of train trips to use for CV tuning")
    parser.add_argument("--results-dir", type=str, default="results/hmm", help="Results directory")
    parser.add_argument("--configs-dir", type=str, default="configs", help="Configs directory")
    args = parser.parse_args()

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
    all_train_ids = sorted([
        d.name for d in train_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.track").exists()
    ])
    logger.info(f"Found {len(all_train_ids)} train trips in {train_dir}")

    # Use first N diverse train trips for tuning
    tune_ids = all_train_ids[:args.num_train_tune]
    logger.info(f"Loading {len(tune_ids)} train trips for Group-CV hyperparameter tuning: {tune_ids}")
    train_trips = [load_trip_data(tid, train_dir) for tid in tune_ids]

    # 2. Hyperparameter Tuning via GroupKFold CV
    best_params = tune_hyperparameters(train_trips, logger, n_splits=3)

    # Save best parameters to configs/hmm_params.json and results/hmm/best_params.json
    best_params.to_json(configs_dir / "hmm_params.json")
    best_params.to_json(results_dir / "best_params.json")
    logger.info(f"Saved tuned parameters to {configs_dir / 'hmm_params.json'}")

    # 3. Evaluate on Student Test Trips (all 10)
    test_dir = Path(args.test_dir)
    test_ids = sorted([
        d.name for d in test_dir.iterdir()
        if d.is_dir() and (d / f"{d.name}.track").exists()
    ])
    logger.info(f"Found {len(test_ids)} student_test trips in {test_dir}: {test_ids}")

    all_records: List[Dict[str, Any]] = []
    helped_hurt_records: List[Dict[str, Any]] = []

    # Evaluate tuning train subset
    logger.info("--- Evaluating Train Subset ---")
    for tr in train_trips:
        rec, top1_arcs, hmm_arcs, ref_arcs = run_single_trip_eval(tr, best_params)
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
        rec, top1_arcs, hmm_arcs, ref_arcs = run_single_trip_eval(tr, best_params)
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
    df_all = pd.DataFrame(all_records)
    df_all.to_csv(results_dir / "per_trip_metrics.csv", index=False)
    logger.info(f"Saved per-trip metrics to {results_dir / 'per_trip_metrics.csv'}")

    df_hh = pd.DataFrame(helped_hurt_records)
    df_hh.to_csv(results_dir / "helped_hurt.csv", index=False)
    logger.info(f"Saved helped/hurt analysis to {results_dir / 'helped_hurt.csv'}")

    # 5. Compute Aggregate Summary for Student Test
    test_df = df_all[df_all["split"] == "student_test"]
    summary = {
        "num_student_test_trips": len(test_df),
        "total_test_observations": int(test_df["num_points"].sum()),
        "mean_top1_accuracy": float(test_df["top1_accuracy"].mean()),
        "mean_hmm_accuracy": float(test_df["hmm_accuracy"].mean()),
        "mean_delta_accuracy": float(test_df["delta_accuracy"].mean()),
        "mean_top1_continuity": float(test_df["top1_continuity"].mean()),
        "mean_hmm_continuity": float(test_df["hmm_continuity"].mean()),
        "mean_delta_continuity": float(test_df["delta_continuity"].mean()),
        "mean_candidate_miss_rate": float(test_df["candidate_miss_rate"].mean()),
        "mean_top1_runtime_s": float(test_df["top1_runtime_s"].mean()),
        "mean_hmm_runtime_s": float(test_df["hmm_runtime_s"].mean()),
        "trips_helped": int((test_df["status"] == "helped").sum()),
        "trips_hurt": int((test_df["status"] == "hurt").sum()),
        "trips_neutral": int((test_df["status"] == "neutral").sum()),
        "best_params": best_params.__dict__,
    }

    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary metrics to {results_dir / 'summary.json'}")

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
    logger.info(f"Trips Helped: {summary['trips_helped']} / 10 | Hurt: {summary['trips_hurt']} | Neutral: {summary['trips_neutral']}")
    logger.info("==================================================")


if __name__ == "__main__":
    main()

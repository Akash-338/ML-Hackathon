#!/usr/bin/env python3
"""Train and compare candidate classifiers using trip-grouped data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import platform
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.classifier.predict import ClassifierScorer, FEATURE_NAMES


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare candidate classifiers; evaluator_test is never read."
    )
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--student-test-jsonl", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/classifier"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/classifier"))
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--bayesian-trials", type=int, default=30)
    parser.add_argument("--disable-bayesian", action="store_true")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a deterministic capped sample and lightweight model settings for a quick smoke run.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=100_000,
        help="Maximum candidate rows used in --fast mode.",
    )
    return parser.parse_args()


def configure_logging(results_dir: Path) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_classifier")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(results_dir / "run.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def read_jsonl(path: Path, allowed_partition: str) -> pd.DataFrame:
    path_text = str(path).lower().replace("\\", "/")
    if "evaluator_test" in path_text or allowed_partition not in {"train", "student_test"}:
        raise ValueError("Evaluator-test data and labels are forbidden.")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing candidate JSONL: {path}. Generate it with Issue #1's "
            "build_data_pipeline.mjs from train/student_test only."
        )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("schema_version") != "pathsnap.candidates.v1":
                raise ValueError(f"Unsupported schema at {path}:{line_number}")
            partition = row.get("partition")
            if partition != allowed_partition:
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"Missing features at {path}:{line_number}")
            flat = dict(row)
            flat.update(features)
            flat["candidate_rank"] = row.get("candidate_rank")
            rows.append(flat)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"No candidate rows found in {path}")
    required = {"trip_id", "observation_index", "candidate_rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for name in FEATURE_NAMES:
        if name not in frame:
            frame[name] = np.nan
    return frame[list(FEATURE_NAMES)].apply(pd.to_numeric, errors="coerce")


def cap_training_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Deterministically cap rows while preserving trip and label coverage."""
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    trip_ids = frame["trip_id"].astype(str).unique()
    rows_per_trip = max(1, (max_rows + len(trip_ids) - 1) // len(trip_ids))
    samples = [
        group.sample(n=min(len(group), rows_per_trip), random_state=SEED)
        for _, group in frame.groupby("trip_id", sort=False)
    ]
    capped = pd.concat(samples, ignore_index=False).sort_index()
    capped = capped.iloc[:max_rows].copy()
    return capped


def make_models(fast: bool = False) -> Dict[str, Pipeline]:
    preprocessor = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    forest_estimators = 50 if fast else 300
    hist_iterations = 50 if fast else 200
    knn_neighbors = 5 if fast else 15
    return {
        "logistic_regression": Pipeline(
            preprocessor
            + [
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=SEED,
                        solver="lbfgs",
                    ),
                )
            ]
        ),
        "knn": Pipeline(
            preprocessor
            + [("model", KNeighborsClassifier(n_neighbors=knn_neighbors, weights="distance", p=2))]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=forest_estimators,
                        max_depth=12 if fast else None,
                        min_samples_leaf=1,
                        class_weight="balanced_subsample",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=forest_estimators,
                        max_depth=12 if fast else None,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.08,
                        max_iter=hist_iterations,
                        max_leaf_nodes=15 if fast else 31,
                        l2_regularization=1.0,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def positive_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def observation_metrics(frame: pd.DataFrame, probability: np.ndarray) -> Dict[str, float | int | None]:
    scored = frame.copy()
    scored["_probability"] = probability
    top_rows = (
        scored.sort_values(
            ["trip_id", "observation_index", "_probability"],
            ascending=[True, True, False],
        )
        .groupby(["trip_id", "observation_index"], sort=False)
        .head(1)
    )
    labels = pd.to_numeric(top_rows["label"], errors="coerce")
    top1 = float(accuracy_score(labels.fillna(0).astype(int), (labels == 1).astype(int))) if len(labels) else 0.0
    missed = 0
    total_observations = 0
    for _, group in scored.groupby(["trip_id", "observation_index"], sort=False):
        group_labels = pd.to_numeric(group["label"], errors="coerce").fillna(0)
        total_observations += 1
        if int(group_labels.sum()) == 0:
            missed += 1
    return {
        "top1_accuracy": top1,
        "candidate_miss_rate": 0.0
        if total_observations == 0
        else float(missed / total_observations),
        "route_continuity": None,
        "num_observations": int(scored[["trip_id", "observation_index"]].drop_duplicates().shape[0]),
    }


def candidate_row_metrics(frame: pd.DataFrame, probability: np.ndarray) -> Dict[str, Any]:
    y = pd.to_numeric(frame["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    y_pred = (probability >= 0.5).astype(int)
    result: Dict[str, Any] = {
        "top1_accuracy": observation_metrics(frame, probability)["top1_accuracy"],
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "candidate_miss_rate": observation_metrics(frame, probability)["candidate_miss_rate"],
        "route_continuity": None,
        "num_observations": observation_metrics(frame, probability)["num_observations"],
    }
    try:
        result["log_loss"] = float(log_loss(y, probability, labels=[0, 1]))
    except ValueError:
        result["log_loss"] = None
    try:
        result["roc_auc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) > 1 else None
    except ValueError:
        result["roc_auc"] = None
    return result


def cross_validate(
    model_name: str,
    model: Pipeline,
    train: pd.DataFrame,
    folds: int,
) -> Dict[str, Any]:
    X = feature_frame(train)
    y = pd.to_numeric(train["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = train["trip_id"].astype(str).to_numpy()
    if train["trip_id"].nunique() < folds:
        raise ValueError(
            f"Only {train['trip_id'].nunique()} distinct training trips; cannot make {folds} folds."
        )
    cv = GroupKFold(n_splits=folds)
    fold_records = []
    fold_times = []
    out_of_fold = np.zeros(len(train), dtype=float)
    for fold_number, (fit_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
        fitted = clone(model)
        started = time.perf_counter()
        fitted.fit(X.iloc[fit_idx], y[fit_idx])
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        probability = positive_probability(fitted, X.iloc[val_idx])
        inference_seconds = time.perf_counter() - started
        out_of_fold[val_idx] = probability
        val_frame = train.iloc[val_idx]
        metrics = candidate_row_metrics(val_frame, probability)
        fold_records.append({"fold": fold_number, **metrics, "fit_seconds": fit_seconds, "inference_seconds": inference_seconds})
        fold_times.append({"fit_seconds": fit_seconds, "inference_seconds": inference_seconds})
    overall = candidate_row_metrics(train, out_of_fold)
    overall.update(
        {
            "model": model_name,
            "cv_folds": folds,
            "group_column": "trip_id",
            "random_seed": SEED,
            "folds": fold_records,
        }
    )
    return overall


def optuna_tune(
    train: pd.DataFrame,
    folds: int,
    trials_count: int,
) -> tuple[Pipeline | None, list[dict[str, Any]]]:
    try:
        import optuna
    except ImportError:
        return None, []

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X = feature_frame(train)
    y = pd.to_numeric(train["label"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = train["trip_id"].astype(str).to_numpy()
    cv = GroupKFold(n_splits=folds)

    def objective(trial: Any) -> float:
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
                        max_depth=trial.suggest_int("max_depth", 4, 30),
                        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                        class_weight="balanced",
                        random_state=SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        fold_scores = []
        for fit_idx, val_idx in cv.split(X, y, groups):
            fitted = clone(model)
            fitted.fit(X.iloc[fit_idx], y[fit_idx])
            probabilities = positive_probability(fitted, X.iloc[val_idx])
            predictions = (probabilities >= 0.5).astype(int)
            fold_scores.append(
                f1_score(y[val_idx], predictions, average="macro", zero_division=0)
            )
        return float(np.mean(fold_scores))

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=trials_count, show_progress_bar=False)
    trial_records = [
        {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
        }
        for trial in study.trials
    ]
    params = study.best_params
    tuned = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=params["max_features"],
                    class_weight="balanced",
                    random_state=SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return tuned, trial_records


def write_comparison_plots(results_dir: Path, comparison: pd.DataFrame) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    metric_names = ["top1_accuracy", "macro_f1", "log_loss"]
    written = []
    for metric in metric_names:
        if metric not in comparison or comparison[metric].dropna().empty:
            continue
        plot = comparison.set_index("model")[metric].dropna().sort_values(ascending=(metric == "log_loss"))
        fig, ax = plt.subplots(figsize=(9, 5))
        plot.plot(kind="bar", ax=ax)
        ax.set_title(f"Training/CV {metric.replace('_', ' ').title()} by Model")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        path = results_dir / f"{metric}_comparison.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


def main() -> int:
    args = parse_args()
    logger = configure_logging(args.results_dir)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(args.train_jsonl, "train")
    student_test = read_jsonl(args.student_test_jsonl, "student_test")
    if train["label"].isna().all():
        raise ValueError("Training JSONL contains no labels; weak proxy labels are required.")
    train["label"] = pd.to_numeric(train["label"], errors="coerce")
    train = train.dropna(subset=["label"]).copy()
    train["label"] = train["label"].astype(int)
    if set(train["label"].unique()) != {0, 1}:
        raise ValueError("Training labels must contain both positive and negative candidates.")
    if args.fast:
        args.cv_folds = min(args.cv_folds, 2)
        args.disable_bayesian = True
        train = cap_training_rows(train, args.max_train_rows)
        if set(train["label"].unique()) != {0, 1}:
            raise ValueError("Fast-mode sampling removed one of the label classes.")
        logger.info(
            "Fast mode: capped training data to %d candidate rows across %d trips",
            len(train),
            train["trip_id"].nunique(),
        )
    if train["trip_id"].nunique() < args.cv_folds:
        raise ValueError("At least cv_folds distinct training trip IDs are required.")

    logger.info("Using %d training trips and %d candidate rows", train["trip_id"].nunique(), len(train))
    logger.info("Using %d student-test trips; route/ground-truth labels are not read", student_test["trip_id"].nunique())

    comparison_records = []
    fitted_models: Dict[str, Pipeline] = {}
    X_train = feature_frame(train)
    y_train = train["label"].to_numpy(dtype=int)

    for model_name, model in make_models(fast=args.fast).items():
        cv_record = cross_validate(model_name, model, train, args.cv_folds)
        comparison_records.append(cv_record)
        fitted = clone(model)
        fit_started = time.perf_counter()
        fitted.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        train_probability = positive_probability(fitted, X_train)
        inference_seconds = time.perf_counter() - inference_started
        fitted_models[model_name] = fitted
        cv_record["fit_seconds"] = fit_seconds
        cv_record["inference_seconds"] = inference_seconds
        cv_record["attempt_status"] = "succeeded"
        cv_record["config"] = {
            "pipeline_steps": list(fitted.named_steps),
            "step_parameters": {
                name: {
                    key: str(value)
                    for key, value in step.get_params(deep=False).items()
                }
                for name, step in fitted.named_steps.items()
            },
        }
        logger.info(
            "Model %s: macro_f1=%.4f top1=%.4f log_loss=%s",
            model_name,
            cv_record["macro_f1"],
            cv_record["top1_accuracy"],
            cv_record["log_loss"],
        )

    tuned_model = None
    trial_records: list[dict[str, Any]] = []
    if not args.disable_bayesian:
        tuned_model, trial_records = optuna_tune(train, args.cv_folds, args.bayesian_trials)
    if tuned_model is not None:
        tuned_cv = cross_validate("extra_trees_optuna", tuned_model, train, args.cv_folds)
        tuned_cv["attempt_status"] = "succeeded"
        tuned_cv["fit_seconds"] = None
        tuned_cv["inference_seconds"] = None
        comparison_records.append(tuned_cv)
        fitted_tuned = clone(tuned_model)
        fitted_tuned.fit(X_train, y_train)
        fitted_models["extra_trees_optuna"] = fitted_tuned
        logger.info(
            "Bayesian model extra_trees_optuna: macro_f1=%.4f top1=%.4f",
            tuned_cv["macro_f1"],
            tuned_cv["top1_accuracy"],
        )
    else:
        comparison_records.append(
            {
                "model": "extra_trees_optuna",
                "attempt_status": "skipped",
                "reason": "optuna unavailable or --disable-bayesian requested",
            }
        )

    comparison = pd.DataFrame(comparison_records)
    comparison.to_csv(args.results_dir / "model_comparison.csv", index=False)
    (args.results_dir / "model_comparison.json").write_text(
        json.dumps(comparison_records, indent=2), encoding="utf-8"
    )
    (args.results_dir / "bayesian_trials.json").write_text(
        json.dumps(trial_records, indent=2), encoding="utf-8"
    )
    if trial_records:
        pd.DataFrame(trial_records).to_csv(args.results_dir / "bayesian_trials.csv", index=False)

    succeeded = [
        record for record in comparison_records
        if record.get("attempt_status") == "succeeded"
        and record["model"] != "extra_trees_optuna"
    ]
    if not succeeded:
        raise RuntimeError("No classifier models succeeded.")
    best_initial = max(
        succeeded,
        key=lambda record: (
            record.get("macro_f1") if record.get("macro_f1") is not None else -1,
            record.get("top1_accuracy") if record.get("top1_accuracy") is not None else -1,
        ),
    )
    if tuned_model is not None:
        tuned_record = next(
            record for record in comparison_records if record["model"] == "extra_trees_optuna"
        )
        if tuned_record.get("macro_f1", -1) >= best_initial.get("macro_f1", -1):
            best_name = "extra_trees_optuna"
        else:
            best_name = str(best_initial["model"])
    else:
        best_name = str(best_initial["model"])

    best_model = fitted_models[best_name]
    model_path = args.artifacts_dir / "model.joblib"
    joblib.dump(best_model, model_path)
    fit_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    preprocessing_hash = hashlib.sha256(
        pickle.dumps(best_model.named_steps["imputer"], protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()

    student_test_predictions = student_test.copy()
    student_test_started = time.perf_counter()
    student_test_probability = positive_probability(best_model, feature_frame(student_test))
    student_test_inference_seconds = time.perf_counter() - student_test_started
    student_test_predictions["probability"] = student_test_probability
    top_student = (
        student_test_predictions.sort_values(
            ["trip_id", "observation_index", "probability"],
            ascending=[True, True, False],
        )
        .groupby(["trip_id", "observation_index"], sort=False)
        .head(1)
        .copy()
    )

    student_predictions_path = args.results_dir / "student_test_predictions.jsonl"
    with student_predictions_path.open("w", encoding="utf-8") as handle:
        for row in top_student.to_dict(orient="records"):
            record = {
                "schema_version": "pathsnap.classifier_predictions.v1",
                "trip_id": row["trip_id"],
                "partition": "student_test",
                "observation_index": int(row["observation_index"]),
                "candidate_arc_id": row["candidate_arc_id"],
                "probability": float(row["probability"]),
                "label": None,
            }
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    candidate_counts = top_student.groupby("trip_id").size()
    student_summary = {
        "partition": "student_test",
        "num_trips": int(student_test["trip_id"].nunique()),
        "num_observations": int(top_student.shape[0]),
        "mean_candidates_per_observation": float(
            student_test.groupby(["trip_id", "observation_index"]).size().mean()
        ),
        "top1_accuracy": None,
        "macro_f1": None,
        "log_loss": None,
        "candidate_miss_rate": None,
        "route_continuity": None,
        "inference_seconds": student_test_inference_seconds,
        "labels_used": False,
        "ground_truth_used": False,
        "candidate_coverage_evaluated": True,
    }

    metadata = {
        "schema_version": "pathsnap.classifier_artifacts.v1",
        "selected_model": best_name,
        "random_seed": SEED,
        "feature_names": list(FEATURE_NAMES),
        "cv_strategy": "GroupKFold",
        "group_column": "trip_id",
        "cv_folds": args.cv_folds,
        "training_trip_count": int(train["trip_id"].nunique()),
        "training_candidate_count": int(len(train)),
        "student_test_trip_count": int(student_test["trip_id"].nunique()),
        "label_policy": "training weak proxy labels from nearest route arc; no student-test labels",
        "bayesian_tuning": "optuna" if tuned_model is not None else "disabled_or_unavailable",
        "bayesian_trials": len(trial_records),
        "model_sha256": fit_hash,
        "preprocessing_sha256": preprocessing_hash,
        "sklearn_version": sklearn.__version__,
        "python_platform": platform.platform(),
        "prediction_interface": "ClassifierScorer.predict_frame / predict_jsonl",
        "forbidden_partitions": ["evaluator_test"],
    }
    (args.artifacts_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    config = {
        "random_seed": SEED,
        "cv_folds": args.cv_folds,
        "bayesian_trials_requested": args.bayesian_trials,
        "bayesian_trials_completed": len(trial_records),
        "selected_model": best_name,
        "model_artifact": str(model_path),
        "prediction_interface": "src.classifier.predict.ClassifierScorer",
    }
    (args.results_dir / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (args.results_dir / "student_test_summary.json").write_text(
        json.dumps(student_summary, indent=2), encoding="utf-8"
    )
    (args.results_dir / "timing_seed_config.json").write_text(
        json.dumps(
            {
                "random_seed": SEED,
                "selected_model_fit_seconds": float(
                    next(r for r in comparison_records if r["model"] == best_name).get("fit_seconds") or 0
                ),
                "student_test_inference_seconds": student_test_inference_seconds,
                "sklearn_version": sklearn.__version__,
                "python_platform": platform.platform(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plots = write_comparison_plots(args.results_dir, comparison)
    (args.results_dir / "selection_rationale.json").write_text(
        json.dumps(
            {
                "selected_model": best_name,
                "selection_rule": "Highest 5-fold trip-grouped out-of-fold Macro F1; ties favor Top-1 accuracy.",
                "cv_result": next(r for r in comparison_records if r["model"] == best_name),
                "student_test_note": "Only unlabeled candidate coverage/predictions are reported; no student-test route labels are opened.",
                "plots": plots,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Selected model: %s", best_name)
    logger.info("Saved model artifact: %s", model_path)
    logger.info("Saved student-test predictions: %s", student_predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

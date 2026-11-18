# PathSnap

PathSnap is the ML Hackathon map-matching project.

## Candidate classifier (Issue #2)

The classifier consumes the Issue #1 `pathsnap.candidates.v1` JSONL contract. It performs 5-fold
trip-grouped cross-validation over the 80 training trips, compares five distinct model families,
optionally runs Optuna Bayesian tuning, and evaluates unlabeled predictions on the 10
student-test trips. It never reads `evaluator_test`, `.route` files, or evaluator labels.

### Install

```powers
python -m pip install -r requirements.txt
```

Generate candidate JSONL with Issue #1, then run:

```powers
python scripts/train_classifier.py `
  --train-jsonl artifacts/data-pipeline/training_candidates.jsonl `
  --student-test-jsonl artifacts/data-pipeline/candidates.jsonl `
  --artifacts-dir artifacts/classifier `
  --results-dir results/classifier `
  --cv-folds 5 `
  --bayesian-trials 30
```

For a dependency-available run without Bayesian tuning:

```powers
python scripts/train_classifier.py `
  --train-jsonl artifacts/data-pipeline/training_candidates.jsonl `
  --student-test-jsonl artifacts/data-pipeline/candidates.jsonl `
  --disable-bayesian
```

### Training models

The comparison logs five attempted configurations:

- Logistic regression
- K-nearest neighbours
- Random forest
- Extra trees
- Histogram gradient boosting

Extra trees is also eligible for optional Optuna Bayesian tuning. The objective is mean
trip-grouped CV Macro F1. Every attempted model and completed trial is recorded.

### Outputs

- `artifacts/classifier/model.joblib`: complete preprocessing plus selected model pipeline
- `artifacts/classifier/metadata.json`: selected model, seed, schema, CV, and hashes
- `results/classifier/model_comparison.{csv,json}`: per-model CV metrics and timings
- `results/classifier/bayesian_trials.{csv,json}`: all completed tuning trials
- `results/classifier/student_test_predictions.jsonl`: top candidate per student-test observation
- `results/classifier/student_test_summary.json`: unlabeled candidate coverage and timing only
- `results/classifier/timing_seed_config.json`: seed, timing, and environment
- `results/classifier/*_comparison.png`: actual comparison plots
- `results/classifier/selection_rationale.json`: selected model and rationale
- `results/classifier/run.log`: complete run log

Student-test route files and labels are deliberately not opened. Therefore `top1_accuracy`,
Macro F1, LogLoss, and route continuity are `null` for student test by design. They are
measured only on training trip data; LogLoss is candidate-row LogLoss and Top-1 is
observation-level.

### Prediction interface

The saved artifact uses `src.classifier.predict.ClassifierScorer`:

```python
from src.classifier.predict import ClassifierScorer

scorer = ClassifierScorer("artifacts/classifier")
probabilities = scorer.predict_jsonl(candidate_rows)
top_rows = scorer.score_by_observation(candidate_dataframe)
```

Candidate JSONL rows and prediction DataFrames use trip-local `candidate_arc_id` and aligned
probability values, making the result directly usable by the HMM decoder and FastAPI service.

### Tests

```powers
pytest -q tests/test_classifier.py
```

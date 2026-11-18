# PathSnap

PathSnap - Machine Learning Hackathon

PathSnap is the ML Hackathon map-matching project for GPS map matching and route reconstruction on directed road networks.

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

## Team & Roles
- **Akash K:** Data ingestion, trajectory preprocessing, and spatial candidate arc generation.
- **Akash S:** Machine learning candidate classification and probability scoring.
- **Anand (Owner of Issue #3):** HMM / Viterbi route reconstruction decoder, network graph modeling, and route continuity optimization.
- **Amar:** FastAPI backend integration and UI dashboard.

---

## HMM Decoder (Issue #3)

The HMM Decoder reconstructs the most likely sequence of road arcs traversed by a vehicle from noisy GPS observations and candidate classification probabilities.

### Key Capabilities
- **Directed Road Network Graph (`src/hmm/network.py`):** Models trip-local directed arcs, edge adjacency ($a \to b$ if $a.\text{to\_node} == b.\text{from\_node}$), metric arc lengths in meters, and bounded Dijkstra shortest paths with LRU caching.
- **Vectorized Log-Space Viterbi (`src/hmm/viterbi.py`):** Pure NumPy implementation scaling to thousands of observations per second ($> 1000$ points/sec).
- **Physical Transition Dynamics:** Balances road network distance against GPS displacement; handles same-arc travel, connected turns, multi-hop routes, and applies soft penalties for disconnected leaps.
- **Robust Edge Case Handling:** Recovers seamlessly from empty candidate sets (emits `-1`), temporal gaps ($> 15$s), numerical underflow, loops, and single-observation trips.
- **Full Teammate Adapter (`src/hmm/adapters.py`):** Converts DataFrame, dict, or matrix candidate outputs from teammates into the decoder contract without modifying their code.

### Installation
```bash
pip install -r requirements.txt
```

### Running Tests
Run the comprehensive PyTest suite covering hand-crafted road networks and edge cases:
```bash
pytest -v tests/
```

### Running Comparison & Tuning
To tune hyperparameters via Group-aware Cross-Validation (GroupKFold by trip) on `data/train` and evaluate on `data/student_test`:
```bash
python scripts/compare_hmm.py
```
Outputs generated in `results/hmm/`:
- `results/hmm/per_trip_metrics.csv`
- `results/hmm/helped_hurt.csv`
- `results/hmm/summary.json`
- `results/hmm/best_params.json`
- `results/hmm/run.log`

### Generating Interactive Folium Maps
Generate a map with togglable layers (Raw GPS, Road Network, Reference Route, Classifier Top-1, and HMM Decoded Route):
```bash
python scripts/make_folium_map.py 00000000
python scripts/make_folium_map.py 00000001
```
Interactive maps are saved to `results/hmm/maps/<trip_id>.html`.

### Empirical Results (10 Student Test Trips)
| Metric | Classifier Top-1 | HMM Viterbi | Improvement |
| :--- | :---: | :---: | :---: |
| **Observation Accuracy** | 58.35% | **86.14%** | **+27.79%** |
| **Route Continuity** | 61.11% | **88.98%** | **+27.87%** |
| **Candidate Miss Rate** | - | 0.06% | - |
| **Mean Runtime per Trip** | 0.005s | 0.931s | Fast (< 1s) |
| **Trips Helped / Hurt** | - | **9 Helped / 1 Hurt\*** | - |

*\*Trip 00000056 had a minor -1.0% change in point accuracy due to intersection projection ties, but its route continuity improved dramatically by +36.3% (from 51.6% to 87.9%), completely eliminating erratic classifier jumping.*

For detailed API schemas, mathematical equations, and integration guidelines, see [docs/hmm_decoder.md](docs/hmm_decoder.md).

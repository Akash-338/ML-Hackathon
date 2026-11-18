# PathSnap – Evaluator Application

This branch contains the evaluator application scaffold for GitHub Issue #4.

## Components

- `api/main.py`: FastAPI health, trip-discovery, prediction, and server-side evaluation orchestration.
- `ui/app.py`: Streamlit evaluator UI.
- The real classifier, preprocessing artifacts, and HMM/Viterbi decoder are integrated through a pipeline adapter. No synthetic predictions are generated.

## Dataset layout

Set `PATHSNAP_DATA_ROOT` to a directory containing one folder per partition:

```text
data/
  student_test/
    00000000/
      00000000.track
      00000000.nodes
      00000000.arcs
      00000000.route
  evaluator_test/
    ...
```

The API defaults to `student_test` for development. Use `evaluator_test` only in the authorised final evaluation environment. Evaluator labels must remain server-side and must never be passed to the classifier/decoder or returned to the browser.

## Pipeline integration contract

Coordinate the exact model/artifact interface with Akash S and the decoder interface with Anand. Expose a factory configured by `PATHSNAP_PIPELINE_FACTORY=your_package.pipeline:create_pipeline`. The factory is deliberately mandatory: this repository does not ship a dummy model or generated predictions.

The factory should load the saved classifier/preprocessing artifacts and Anand's HMM parameters from paths supplied by its own configuration (for example, `PATHSNAP_MODEL_PATH`, `PATHSNAP_PREPROCESSOR_PATH`, and `PATHSNAP_HMM_PARAMS_PATH`). Keep those files outside git when they are large or private. The adapter must be importable from the API laptop's environment before starting Uvicorn.

The returned object must implement:

```python
def create_pipeline():
    return Pipeline()

class Pipeline:
    def predict_trip(self, *, trip_id, track_path, nodes_path, arcs_path) -> dict:
        # Run the real classifier + decoder using only these three inputs.
        return {
            "predicted_route": [...],
            "route_coordinates": [[lat, lon], ...],
        }

    def evaluate_prediction(self, *, prediction, reference_route_path) -> dict:
        # Optional server-only evaluation. This method is never called by the UI.
        return {"top1_accuracy": ..., "macro_f1": ...}
```

**Separation requirement:** `predict_trip` receives only the GPS track, road nodes, and road arcs. It never receives the `.route` reference file. The API may pass the reference-route path only to the separate server-side `evaluate_prediction` hook. Never include reference labels or reference-route arrays in the returned dictionary. The API filters common truth-field names as an additional safeguard.

Do not use evaluator-test labels for development, model selection, or tuning.

## Laptop 1 – API

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

PowerShell example:

```powershell
$env:PATHSNAP_DATA_ROOT="C:\\path\\to\\data"
$env:PATHSNAP_UI_PARTITION="student_test"
$env:PATHSNAP_PIPELINE_FACTORY="src.evaluator_pipeline:create_pipeline"
$env:PATHSNAP_CLASSIFIER_ARTIFACT_DIR="C:\\path\\to\\artifacts\\classifier"
$env:PATHSNAP_CANDIDATES_JSONL="C:\\path\\to\\artifacts\\data-pipeline\\candidates.jsonl"
$env:PATHSNAP_HMM_PARAMS_PATH="C:\\path\\to\\configs\\hmm_params.json"
$env:PATHSNAP_CORS_ORIGINS="http://localhost:8501"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Check `http://localhost:8000/health` and `http://localhost:8000/docs`.

## Laptop 2 – UI

Install requirements, then point the UI to Laptop 1's LAN IP:

```powershell
$env:PATHSNAP_API_URL="http://<LAPTOP-1-LAN-IP>:8000"
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501` on Laptop 2. Ensure both laptops are on the same network and the API port is allowed through Laptop 1's firewall.

For a quick two-laptop smoke check, Laptop 1 should run `Invoke-WebRequest http://localhost:8000/health`
and Laptop 2 should open the UI and confirm that `/trips` returns the ten reserved
`student_test` IDs. Select a trip and use the browser network/terminal logs to record
the real `POST /predict` response; do not copy labels or metrics from a reference file.

## API endpoints

- `GET /health`: API and pipeline configuration status.
- `GET /trips`: trip IDs in the configured partition.
- `POST /predict`: JSON body `{"trip_id":"00000000"}`; runs the injected classifier/decoder and optional server-side evaluation hook. IDs are validated as exactly eight digits.

A prediction request returns HTTP 503 until the real pipeline factory and saved classifier
artifact are configured. This is intentional: the evaluator does not generate dummy predictions.

## Evidence checklist

Capture screenshots only after running the connected implementation (the scaffold intentionally
cannot produce a successful prediction by itself):

- Streamlit UI with a real trip selected and result rendered.
- Laptop 1 terminal showing a successful API prediction request.
- Record actual dependency versions, model/decoder artifact paths, and test results.

The submission bundle should contain only checked-in source/docs plus verifiable generated
artifacts from the connected run: the ten-trip list, API log, UI screenshot, model and
decoder artifact manifest, and measured result/map files. If a run was not completed, leave
that evidence out and mark it as pending rather than fabricating a screenshot or metric.

Do not commit credentials, private keys, evaluator labels exposed to the client, or fabricated metrics.
# PathSnap
# PathSnap - Machine Learning Hackathon

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

## Data Pipeline (Issue #1)

Run the dependency-free Node.js pipeline from the repository root:

```powershell
node scripts/build_data_pipeline.mjs
```

The default input is `Data/Data`, with `train/` and `student_test/` trip folders. Override paths with `--data <directory>` and `--output <directory>`. Use `--expect-train 80 --expect-student-test 10` for a strict run against the supplied 80/10 partition split.

Run the dependency-free privacy and output regression test:

```powershell
node scripts/test_data_pipeline.mjs
```

The pipeline expects one folder per trip, named by trip ID, with files named `<trip-id>.track`, `.nodes`, `.arcs`, and `.route`. It parses whitespace- or comma-separated numeric rows without headers:

- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude
- `.arcs`: from-node index, to-node index
- `.route`: one local arc index per row

Arc IDs are zero-based line indices in each trip's `.arcs` file. Node IDs in `.arcs` index the trip's `.nodes` rows. GPS-to-arc distances use a local metric projection of longitude/latitude; final candidate distances are in metres. Candidates are the nearest road segments within 3 km, capped at eight per observation.

For training labels, each GPS point is matched independently to its nearest reference-route segment by point-to-segment distance. This does not pair GPS row `i` with route row `i`. The matched route arc is retained among the candidates, and only training rows receive `0`/`1` labels. Student-test `.route` files are never opened; their candidate labels are `null`.

The route match is a weak proxy label: it assigns each observation to the closest reference-route arc geometrically when that arc is within 500 m. More distant observations are left unlabeled. It does not claim temporal alignment or ground-truth GPS-to-arc labels.

Each candidate JSONL row uses schema `pathsnap.candidates.v1`, with trip and observation identifiers, the local `candidate_arc_id`, a `features` object, and the nullable training label. Features include GPS-to-arc distance, heading difference, arc length, progress along the arc, elapsed time, and GPS speed.

### Outputs

By default, generated files go under ignored `artifacts/data-pipeline/`:

- `candidates.jsonl`: model and decoder candidate/features rows
- `training_candidates.jsonl`: labeled candidate rows from training trips only
- `data_quality_report.json`: partition counts, row counts, malformed rows, timestamp gaps, coordinate bounds, and node/arc ID range checks
- `trip_summary.csv`: compact per-trip metrics
- `training_trip_<id>_map.svg`: actual GPS track and reference route from the first training trip
- `README.txt`: output field and label-policy note

Generated data and EDA artifacts are local outputs; keep the original dataset outside Git. The pipeline enumerates only `train/` and `student_test/`; it never enumerates, opens, parses, or emits `evaluator_test/` files or labels.

## HMM Decoder (Issue #3)

The HMM decoder reconstructs the most likely sequence of road arcs traversed by a vehicle from noisy GPS observations and candidate classification probabilities.

### Key Capabilities

- **Directed Road Network Graph (`src/hmm/network.py`):** Models trip-local directed arcs, edge adjacency, metric arc lengths in metres, and bounded Dijkstra shortest paths with LRU caching.
- **Vectorized Log-Space Viterbi (`src/hmm/viterbi.py`):** Pure NumPy implementation scaling to thousands of observations per second.
- **Physical Transition Dynamics:** Balances road network distance against GPS displacement; handles same-arc travel, connected turns, multi-hop routes, and disconnected leaps.
- **Robust Edge Case Handling:** Handles empty candidate sets, temporal gaps over 15 seconds, numerical underflow, loops, and single-observation trips.
- **Full Teammate Adapter (`src/hmm/adapters.py`):** Converts DataFrame, dict, or matrix candidate outputs into the decoder contract.

### Installation

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
pytest -v tests/
```

### Running Comparison & Tuning

To tune hyperparameters via Group-aware Cross-Validation (GroupKFold by trip) on `data/train` and evaluate on `data/student_test`:

```bash
python scripts/compare_hmm.py
```

Outputs are generated in `results/hmm/`, including per-trip metrics, helped/hurt analysis, a summary, best parameters, and a run log.

### Generating Interactive Folium Maps

```bash
python scripts/make_folium_map.py 00000000
python scripts/make_folium_map.py 00000001
```

Interactive maps are saved to `results/hmm/maps/<trip_id>.html`.

### Empirical Results (10 Student Test Trips)

> **Current artifact status:** the checked-in table is a development comparison
> generated by `src/hmm/dev_candidates.py`, not by Akash S's upstream classifier
> artifact. The corresponding stand-in snapshot is
> `results/hmm/standin_summary.json`, and its provenance is recorded in
> `results/hmm/provenance.json`. No real-classifier result is claimed here.
> The comparison script never reads `data/evaluator_test` or evaluator labels.

| Metric | Classifier Top-1 | HMM Viterbi | Improvement |
| :--- | :---: | :---: | :---: |
| **Observation Accuracy** | 58.35% | **86.14%** | **+27.79%** |
| **Route Continuity** | 61.11% | **88.98%** | **+27.87%** |
| **Candidate Miss Rate** | - | 0.06% | - |
| **Mean Runtime per Trip** | 0.005s | 0.931s | Fast (< 1s) |
| **Trips Helped / Hurt** | - | **9 Helped / 1 Hurt** | - |

For detailed API schemas, mathematical equations, and integration guidelines, see [docs/hmm_decoder.md](docs/hmm_decoder.md).

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
# PathSnap - Machine Learning Hackathon

GPS Map Matching and Route Reconstruction on Directed Road Networks.

---

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

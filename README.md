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
$env:PATHSNAP_PIPELINE_FACTORY="your_package.pipeline:create_pipeline"
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

A prediction request returns HTTP 503 until the real pipeline factory is configured. This is intentional: the scaffold does not generate dummy predictions.

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

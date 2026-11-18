# PathSnap – Evaluator Application

This repository currently contains the evaluator application scaffold for GitHub Issue #4.

## Components

- `api/main.py`: FastAPI endpoints for health, trip discovery, and prediction.
- `ui/app.py`: Streamlit evaluator UI.
- The actual classifier, preprocessing artifacts, and HMM/Viterbi decoder must be connected through the pipeline adapter described below. The scaffold intentionally does not fabricate predictions.

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

The UI partition defaults to `student_test` for development. Set it to `evaluator_test` only for the authorised final demonstration environment. Evaluator labels must remain server-side and must never be passed to the model or returned to the UI.

## Pipeline integration contract

Coordinate the exact model/artifact interface with Akash S and the decoder interface with Anand. Then provide a Python factory using this contract:

```python
def create_pipeline():
    return Pipeline()

class Pipeline:
    def predict_trip(self, *, trip_id: str, trip_dir: Path) -> dict:
        # Load/execute the real classifier and decoder.
        # Return JSON-serializable prediction data, e.g.:
        return {
            "predicted_route": [...],
            "metrics": {...},  # computed server-side from permitted reference data
            "route_coordinates": [[lat, lon], ...],
        }
```

Set `PATHSNAP_PIPELINE_FACTORY=your_package.pipeline:create_pipeline`. Do not put ground-truth labels in the return object. The API additionally filters common reference-label keys before serialization. Do not use evaluator-test labels for development, model selection, or tuning.

## Laptop 1 – API

From the repository root:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set environment variables (PowerShell example):

```powershell
$env:PATHSNAP_DATA_ROOT="C:\\path\\to\\data"
$env:PATHSNAP_UI_PARTITION="student_test"
$env:PATHSNAP_PIPELINE_FACTORY="your_package.pipeline:create_pipeline"
$env:PATHSNAP_CORS_ORIGINS="http://localhost:8501"
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Check `http://localhost:8000/health` and `http://localhost:8000/docs`.

## Laptop 2 – UI

Install requirements, then set the API host to Laptop 1's LAN IP:

```powershell
$env:PATHSNAP_API_URL="http://<LAPTOP-1-LAN-IP>:8000"
streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501
```

Open `http://localhost:8501` on Laptop 2. Ensure both laptops are on the same network and the API port is allowed through the Laptop 1 firewall.

## API

- `GET /health`: API and pipeline configuration status.
- `GET /trips`: trip IDs in the configured partition.
- `POST /predict`: JSON body `{"trip_id":"00000000"}`; executes the injected real pipeline.

A prediction request returns HTTP 503 until the real pipeline factory is configured. This is intentional: no dummy predictions are generated.

## Evidence checklist

Capture screenshots only after running the connected implementation:
- Streamlit UI with a real trip selected and result rendered.
- Laptop 1 terminal showing a successful API prediction request.
- Record actual dependency versions, model/decoder artifact paths, and test results.

Do not commit credentials, private keys, evaluator labels exposed to the client, or fabricated metrics.

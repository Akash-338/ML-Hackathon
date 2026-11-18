# PathSnap - Machine Learning Hackathon

GPS map matching and route reconstruction on directed road networks.

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

| Metric | Classifier Top-1 | HMM Viterbi | Improvement |
| :--- | :---: | :---: | :---: |
| **Observation Accuracy** | 58.35% | **86.14%** | **+27.79%** |
| **Route Continuity** | 61.11% | **88.98%** | **+27.87%** |
| **Candidate Miss Rate** | - | 0.06% | - |
| **Mean Runtime per Trip** | 0.005s | 0.931s | Fast (< 1s) |
| **Trips Helped / Hurt** | - | **9 Helped / 1 Hurt** | - |

For detailed API schemas, mathematical equations, and integration guidelines, see [docs/hmm_decoder.md](docs/hmm_decoder.md).

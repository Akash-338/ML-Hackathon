# PathSnap

PathSnap maps GPS observations to candidate road arcs, prepares training features, and reports dataset quality for the ML Hackathon.

## Data pipeline

Run the dependency-free Node.js pipeline from the repository root:

```powershell
node scripts/build_data_pipeline.mjs
```

The default input is `Data/Data`, with `train/` and `student_test/` trip folders. Override paths with `--data <directory>` and `--output <directory>`.

The pipeline expects one folder per trip, named by trip ID, with files named `<trip-id>.track`, `.nodes`, `.arcs`, and `.route`. It parses whitespace or comma separated numeric rows without headers:

- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude
- `.arcs`: from-node index, to-node index
- `.route`: one local arc index per row

Arc IDs are zero-based line indices in that trip's `.arcs` file. Node IDs in `.arcs` index the trip's `.nodes` rows. GPS-to-arc distances use a local metric projection of longitude/latitude; final candidate distances are in metres. Candidates are the nearest road segments within 3 km, capped at eight per observation.

For training labels, each GPS point is matched independently to its nearest reference-route segment by point-to-segment distance. This deliberately does not pair GPS row `i` with route row `i`. The matched route arc is retained among the candidates, and only training rows receive `0`/`1` labels. The student-test `.route` files are never opened; their candidate labels are `null`.

Each candidate JSONL row uses schema `pathsnap.candidates.v1`, with trip and observation identifiers, the local `candidate_arc_id`, a `features` object, and the nullable training label. Features include GPS-to-arc distance, heading difference, arc length, progress along the arc, elapsed time, and GPS speed. This JSONL shape is intended for loading with standard JSON-line readers in either the classifier or decoder.

## Outputs

By default, generated files go under ignored `artifacts/data-pipeline/`:

- `candidates.jsonl`: model and decoder candidate/features rows
- `data_quality_report.json`: partition counts, row counts, malformed rows, timestamp gaps, coordinate bounds, and node/arc ID range checks
- `trip_summary.csv`: compact per-trip metrics
- `training_trip_<id>_map.svg`: actual GPS track and reference route from the first training trip
- `README.txt`: output field and label-policy note

Generated data and EDA artifacts are local outputs; keep the original dataset outside Git.

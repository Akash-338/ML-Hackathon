# PathSnap

PathSnap maps GPS observations to candidate road arcs, prepares training features, and reports dataset quality for the ML Hackathon.

## Data pipeline

Run the dependency-free Node.js pipeline from the repository root:

```powershell
node scripts/build_data_pipeline.mjs
```

The default input is `Data/Data`, with `train/` and `student_test/` trip folders. Override paths with `--data <directory>` and `--output <directory>`. Use `--expect-train 80 --expect-student-test 10` for a strict run against the supplied 80/10 partition split.

Run the dependency-free privacy and output regression test:

```powershell
node scripts/test_data_pipeline.mjs
```

The pipeline expects one folder per trip, named by trip ID, with files named `<trip-id>.track`, `.nodes`, `.arcs`, and `.route`. It parses whitespace or comma separated numeric rows without headers:

- `.track`: longitude, latitude, timestamp
- `.nodes`: longitude, latitude
- `.arcs`: from-node index, to-node index
- `.route`: one local arc index per row

Arc IDs are zero-based line indices in that trip's `.arcs` file. Node IDs in `.arcs` index the trip's `.nodes` rows. GPS-to-arc distances use a local metric projection of longitude/latitude; final candidate distances are in metres. Candidates are the nearest road segments within 3 km, capped at eight per observation.

For training labels, each GPS point is matched independently to its nearest reference-route segment by point-to-segment distance. This deliberately does not pair GPS row `i` with route row `i`. The matched route arc is retained among the candidates, and only training rows receive `0`/`1` labels. The student-test `.route` files are never opened; their candidate labels are `null`.

The route match is a weak proxy label: it assigns each observation to the closest reference-route arc geometrically when that arc is within 500 m. More distant observations are left unlabeled. It does not claim temporal alignment or ground-truth GPS-to-arc labels. The report records this policy so model comparisons can describe it accurately.

Each candidate JSONL row uses schema `pathsnap.candidates.v1`, with trip and observation identifiers, the local `candidate_arc_id`, a `features` object, and the nullable training label. Features include GPS-to-arc distance, heading difference, arc length, progress along the arc, elapsed time, and GPS speed. This JSONL shape is intended for loading with standard JSON-line readers in either the classifier or decoder.

## Outputs

By default, generated files go under ignored `artifacts/data-pipeline/`:

- `candidates.jsonl`: model and decoder candidate/features rows
- `training_candidates.jsonl`: labeled candidate rows from training trips only, ready for classifier experiments
- `data_quality_report.json`: partition counts, row counts, malformed rows, timestamp gaps, coordinate bounds, and node/arc ID range checks
- `trip_summary.csv`: compact per-trip metrics
- `training_trip_<id>_map.svg`: actual GPS track and reference route from the first training trip
- `README.txt`: output field and label-policy note

The candidate feature contract is `pathsnap.candidates.v1`. A row contains `trip_id`, `partition`, source `observation_index`, `timestamp`, `observation` coordinates, a trip-local `candidate_arc_id`, `candidate_rank`, `features`, `label`, and `label_method`. The feature object contains metric distance to the arc, GPS heading difference, arc length, distance/fraction along the arc, elapsed timestamp, and GPS speed. Here is a compact example (values are illustrative):

```json
{"schema_version":"pathsnap.candidates.v1","trip_id":"00000001","partition":"train","observation_index":0,"timestamp":1,"observation":{"lon":8.595203,"lat":49.844053},"candidate_arc_id":281,"candidate_rank":0,"features":{"gps_to_arc_distance_m":7.2,"gps_heading_difference_deg":12.4,"arc_length_m":18.6,"distance_from_arc_start_m":4.1,"fraction_along_arc":0.22043,"time_delta_s":null,"gps_speed_mps":null},"label":1,"label_method":"nearest_route_arc_by_geodesic_segment_distance"}
```

Labels are weak proxy labels: each training observation is independently matched to its nearest reference-route segment in metric coordinates, only when within 500 m. This does not imply temporal alignment or ground-truth GPS-to-road matching. Student-test route files are not opened by the script. Trip folds should remain grouped by `trip_id` during downstream training and evaluation. No evaluator-test data is included in this pipeline.

Generated data and EDA artifacts are local outputs; keep the original dataset outside Git. The pipeline enumerates only `train/` and `student_test/`; it never enumerates, opens, parses, or emits `evaluator_test/` files or labels, and every artifact is safe to share without evaluator-test data.

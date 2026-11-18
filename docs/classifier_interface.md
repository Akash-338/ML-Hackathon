# Candidate classifier prediction contract

## Input

The classifier expects Issue #1 `pathsnap.candidates.v1` candidate JSONL. The Python entrypoint
reads:

- `training_candidates.jsonl` for weakly labeled training candidates
- `candidates.jsonl` or a separate student-test JSONL for development predictions

The entrypoint checks the schema and partition, and rejects `evaluator_test` paths. It does not
open student-test route files.

## Saved artifact

`artifacts/classifier/model.joblib` contains median imputation, feature scaling where applicable,
and the selected classifier pipeline.

`artifacts/classifier/metadata.json` records:

- selected model
- random seed
- feature names
- 5-fold `trip_id`-grouped CV configuration
- training trip/candidate counts
- Bayesian status and trial count
- model/preprocessing hashes
- sklearn and platform versions

## Python interface

```python
from src.classifier.predict import ClassifierScorer

scorer = ClassifierScorer("artifacts/classifier")
probabilities = scorer.predict_jsonl(candidate_rows)
top_candidates = scorer.score_by_observation(candidate_dataframe)
```

The probability is the positive-candidate probability. `score_by_observation` returns the
highest probability row for each `(trip_id, observation_index)`.

## Output JSONL

`results/classifier/student_test_predictions.jsonl` uses
`pathsnap.classifier_predictions.v1` with `trip_id`, `observation_index`, `candidate_arc_id`,
and `probability`. The row is aligned with the candidate contract and has `label: null`.

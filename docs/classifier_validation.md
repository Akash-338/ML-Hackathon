# Classifier validation evidence

## Reproducibility

The implementation was validated under Python 3.13 with the dependencies in
`requirements.txt`. The active Python 3.14 environment could not load NumPy because Windows
Application Control blocked `numpy/random/_bounded_integers` binaries.

Focused tests pass under Python 3.13:

```text
4 passed
```

## Real-data status

No generated `training_candidates.jsonl` or student-test candidate JSONL is committed or present
in this worktree. The Issue #2 entrypoint therefore could not perform the required real 80-trip
fit and 10-trip development evaluation during this session. It fails explicitly with a missing-
input message instead of fabricating results or producing synthetic result files.

Repository-history inspection also found no tracked `training_candidates.jsonl`,
`candidates.jsonl`, or `Data/Data` source tree. The 80-trip real comparison, CV metrics, tuning
trials, and plots therefore remain blocked by source-data availability, not by the classifier
implementation or Python environment.

The command for real validation is:

```powers
py -3.13 scripts/train_classifier.py `
  --train-jsonl artifacts/data-pipeline/training_candidates.jsonl `
  --student-test-jsonl artifacts/data-pipeline/candidates.jsonl `
  --artifacts-dir artifacts/classifier `
  --results-dir results/classifier `
  --cv-folds 5 `
  --bayesian-trials 30
```

The command uses only training trip IDs for fitting and trip-grouped CV. Student-test rows are
used only for candidate coverage and predictions; no route or ground-truth label is opened.

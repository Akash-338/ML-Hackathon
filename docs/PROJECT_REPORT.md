# PathSnap project report

## Overview

PathSnap reconstructs a likely road route from GPS observations and a directed road network. GPS measurements can be noisy, particularly near junctions or parallel roads. The system generates nearby road segment candidates, scores them with a classifier, and uses a hidden Markov model (HMM) with Viterbi decoding to choose a plausible sequence.

## Data and processing

Trips contain GPS tracks and road-network nodes and arcs. Reference routes support training labels and evaluation. The data pipeline generates candidate rows and features. Training labels are weak geometric proxies derived from the reference route, so they are not necessarily perfect ground truth. The evaluation aligns observations with reference arcs and compares predicted arcs against that alignment.

## System design

1. The JavaScript data pipeline builds candidate road segments and features for each observation.
2. A classifier estimates a positive score for each candidate. Features include distance to the arc, heading difference, arc length, position along the arc, speed, elapsed time and candidate rank.
3. The HMM combines candidate scores with road-network transition information. Viterbi decoding selects a sequence across observations, favoring plausible transitions.
4. The evaluator calculates accuracy and continuity. FastAPI exposes prediction and evaluation functionality, while the Streamlit interface displays GPS, road network and predicted routes.

## Classifier training

The training script supports logistic regression, K-nearest neighbours, random forest, extra trees and histogram gradient boosting. It uses trip-grouped cross-validation, keeping observations from a trip in the same fold, and selects by macro F1. Extra trees can optionally be tuned with Optuna. The currently saved metadata identifies histogram gradient boosting, with two cross-validation folds and 100,000 training candidate rows; tuning was disabled or unavailable.

The saved comparison reports 100% Top-1 accuracy for every classifier. This result came from an earlier calculation and should not be presented as valid: the current Top-1 calculation checks whether the selected candidate has a positive label. A fresh complete comparison is needed to establish the best classifier by accuracy. Macro F1 in the saved comparison is highest for histogram gradient boosting (0.8737), but this is a two-fold run and should not be confused with route accuracy.

## Reported HMM results

The saved HMM summary reports 10 student-test trips and 19,160 observations. Its mean per-trip classifier Top-1 accuracy is 82.11%, and mean HMM accuracy is 85.82%, an improvement of 3.71 percentage points. Mean continuity increases from 73.25% to 87.66%, a gain of 14.41 percentage points. Mean candidate miss rate is 0.21%, and mean HMM runtime is 0.404 seconds per trip. Eight trips improved and two worsened. These are averages across trips, not necessarily observation-weighted totals.

The HMM provenance describes a logistic classifier artifact, whereas the current classifier metadata identifies histogram gradient boosting. The evidence does not establish that the current artifact produced the saved HMM results. Record artifact hashes and rerun the comparison before attributing these numbers to a particular current classifier. The README also contains a separate 86.14% development stand-in result; it should not be presented as the accuracy of the real classifier pipeline.

## Verification and limitations

On this audit, the Python suite passed 21 tests and the JavaScript data pipeline privacy and output test passed. The local virtual environment executable could not be launched; the Python suite was run with the available system Python. Automated tests do not establish full correctness, independent generalization or reproducibility of historical results. The classifier comparison needs regeneration with the corrected metric, and the end-to-end UI and full real-data training run need separate verification. Existing result files and configuration had local modifications before this audit.

## Conclusion

The saved HMM experiment suggests that sequence decoding improves both accuracy and route continuity over its classifier baseline. A reproducible evaluation should preserve the exact classifier artifact, data split, decoder parameters and generated results together, then compare models using corrected metrics on held-out trips.

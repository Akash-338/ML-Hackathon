# Pull Request: Implement HMM/Viterbi route reconstruction (closes #3)

## Summary of Changes
Implements the complete HMM / Viterbi route reconstruction pipeline for PathSnap map matching on directed road networks:
1. **Per-Trip Directed Graph (`src/hmm/network.py`):**
   - Direct arc adjacency ($a \to b \iff a.\text{to\_node} == b.\text{from\_node}$).
   - Metric Euclidean arc lengths in meters.
   - Bounded Dijkstra shortest paths with LRU caching.
2. **Vectorized Log-Space Viterbi Decoder (`src/hmm/viterbi.py`):**
   - High-throughput NumPy implementation ($> 1000$ points/sec).
   - Stable contract: `decode_trip(timestamps, gps_xy, candidates, scores, network, params) -> DecodeResult`.
   - Emission: $w_{\text{emit}} \cdot \log(\text{score} + \epsilon)$.
   - Transition: balances network travel distance vs GPS displacement ($-\Delta / \beta$).
   - Soft penalties for unreachable candidate leaps; no $-\infty$ hard crash.
   - Robust edge cases: empty candidate handling (tags `-1` and restarts), temporal gaps ($> 15$s), numerical underflow, loops, and single-observation trajectories.
3. **Candidate & Classifier Adapters (`src/hmm/adapters.py`):**
   - Bridges DataFrame, dict, and matrix formats from Akash K / Akash S without modifying their files.
4. **Development Stand-in Generator (`src/hmm/dev_candidates.py`):**
   - Decoupled candidate extractor via `shapely.STRtree` point-to-segment distance and softmax scores.
5. **Evaluation & Comparison Pipeline (`src/hmm/evaluate.py`, `scripts/compare_hmm.py`):**
   - Group-aware Cross Validation (GroupKFold by trip) on `data/train`.
   - Optimal parameters saved to `configs/hmm_params.json`.
   - Reference route alignment via spatial nearest-arc projection.
6. **Interactive Folium Visualizer (`scripts/make_folium_map.py`):**
   - Generates interactive maps with togglable layers (GPS, Road Network, Ground Truth Route, Classifier Top-1, HMM Decoded Route).
7. **Comprehensive PyTest Suite (`tests/test_network.py`, `tests/test_viterbi.py`):**
   - 10 unit tests covering all required behaviors; 100% passing.

---

## Real-Classifier Results on 10 Student-Test Trips

This evaluation used the saved Histogram Gradient Boosting classifier and real candidate
probabilities. That classifier came from a fast run with 100,000 sampled training rows
and 2-fold trip-grouped CV. HMM parameters were tuned on the first five training trips.
Student-test `.route` files were used only for offline scoring; `evaluator_test` was not
accessed. Provenance is recorded in `results/hmm/provenance.json`.

| Trip ID | Points | Classifier Top-1 | HMM Viterbi | $\Delta$ Acc | Top-1 Continuity | HMM Continuity | $\Delta$ Cont. | Miss Rate | HMM Runtime | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `00000000` | 2,503 | 92.2% | 81.9% | -10.3% | 69.4% | 81.5% | +12.1% | 0.20% | 0.52s | hurt |
| `00000007` | 1,354 | 88.8% | 80.9% | -7.9% | 94.9% | 98.4% | +3.5% | 0.00% | 0.26s | hurt |
| `00000009` | 1,049 | 83.2% | 91.4% | +8.2% | 68.8% | 83.3% | +14.5% | 0.95% | 0.37s | helped |
| `00000040` | 2,995 | 86.7% | 88.9% | +2.2% | 80.8% | 92.4% | +11.5% | 0.00% | 0.64s | helped |
| `00000052` | 262 | 93.1% | 96.2% | +3.1% | 79.5% | 85.7% | +6.2% | 0.00% | 0.07s | helped |
| `00000056` | 3,958 | 80.5% | 80.8% | +0.3% | 67.2% | 88.2% | +21.0% | 0.05% | 0.75s | helped |
| `00000065` | 768 | 75.5% | 82.8% | +7.3% | 67.3% | 85.8% | +18.5% | 0.13% | 0.19s | helped |
| `00000076` | 1,486 | 77.3% | 91.5% | +14.1% | 62.2% | 84.1% | +21.8% | 0.67% | 0.43s | helped |
| `00000077` | 3,911 | 62.0% | 74.7% | +12.8% | 79.4% | 91.7% | +12.3% | 0.13% | 0.63s | helped |
| `00000091` | 874 | 81.7% | 89.1% | +7.4% | 62.8% | 85.5% | +22.6% | 0.00% | 0.18s | helped |
| **Mean per trip** | **19,160** | **82.11%** | **85.82%** | **+3.71 pp** | **73.25%** | **87.66%** | **+14.41 pp** | **0.21%** | **0.404s** | **8 helped / 2 hurt** |

---

## Optimal Tuned Parameters (`configs/hmm_params.json`)
```json
{
  "beta": 10.0,
  "emission_weight": 0.5,
  "invalid_transition_penalty": 50.0,
  "max_hops": 5,
  "max_meters": 2000.0,
  "gap_threshold_s": 15.0,
  "eps": 1e-06
}
```

## How to Verify
```bash
pytest -v tests/
python scripts/compare_hmm.py
python scripts/make_folium_map.py 00000000
python scripts/make_folium_map.py 00000001
```

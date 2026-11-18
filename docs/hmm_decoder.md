# HMM / Viterbi Route Reconstruction Decoder

**Owner:** Anand (Issue #3: *Implement HMM/Viterbi route reconstruction*)  
**Module Path:** `src/hmm/`  
**Configs:** `configs/hmm_params.json`  
**Test Suite:** `tests/test_viterbi.py`, `tests/test_network.py`  
**Comparison & Artifacts:** `results/hmm/`  

---

## 1. Overview & System Role

In map matching for GPS trajectories, raw GPS points contain spatial sensor noise, multipath distortion, and jitter. While a per-point candidate classifier (developed by Akash S) estimates probabilities for nearby candidate road arcs, taking the naive `argmax` (top-1) results in erratic hops between parallel or disconnected streets.

The **HMM / Viterbi Route Reconstruction Decoder** solves this by formulating route reconstruction as a Hidden Markov Model (HMM):
- **Hidden States:** The true directed road arc being traversed at observation $t$: $s_t \in \mathcal{C}_t$.
- **Observations:** Noisy GPS coordinates $(x_t, y_t)$ at timestamp $t$.
- **Emissions:** Classifier confidence that observation $t$ corresponds to candidate arc $k$.
- **Transitions:** Topological and metric feasibility of traveling between arc $a$ at $t-1$ and arc $b$ at $t$ through the road network.
- **Viterbi Decoding:** Solves for the globally optimal, physically continuous sequence of arcs in $\mathcal{O}(T \cdot K_{\text{prev}} \cdot K_{\text{curr}})$ time using vectorized log-space dynamic programming.

---

## 2. API Contract for Teammates (Amar / FastAPI Backend)

### 2.1 Function Signature
```python
from src.hmm import decode_trip, HMMParams, RoadNetwork, DecodeResult

def decode_trip(
    timestamps: Sequence[float] | np.ndarray,
    gps_xy: np.ndarray,
    candidates: Sequence[Sequence[int] | np.ndarray],
    scores: Sequence[Sequence[float] | np.ndarray],
    network: RoadNetwork,
    params: Optional[HMMParams] = None,
) -> DecodeResult:
    ...
```

### 2.2 Input Schema
| Argument | Type | Shape / Length | Description |
| :--- | :--- | :--- | :--- |
| `timestamps` | `np.ndarray` / `Sequence[float]` | `(T,)` | Monotonically ordered observation timestamps in seconds. |
| `gps_xy` | `np.ndarray` | `(T, 2)` | Metric projected coordinates $(x, y)$ in meters (e.g. UTM). |
| `candidates` | `Sequence[np.ndarray]` | Length `T` | List of 1D arrays; `candidates[t]` contains trip-local integer `arc_id`s for step $t$. |
| `scores` | `Sequence[np.ndarray]` | Length `T` | List of 1D arrays; `scores[t]` contains probabilities aligned 1:1 with `candidates[t]`. |
| `network` | `RoadNetwork` | Class instance | Trip-local directed road graph containing node coordinates, directed arcs, and adjacency. |
| `params` | `Optional[HMMParams]` | Dataclass / None | Hyperparameters. If `None`, defaults or `configs/hmm_params.json` are loaded. |

### 2.3 Output Schema (`DecodeResult`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `pred_arc_ids` | `np.ndarray` of shape `(T,)`, `int64` | The decoded sequence of traversed `arc_id`s. Set to `-1` for observations where decoding was impossible (e.g. empty candidate set). |
| `source` | `List[str]` of length `T` | Provenance per observation: `"viterbi"`, `"fallback_top1"`, or `"no_candidates"`. |
| `segment_breaks` | `List[int]` | Step indices where an independent trajectory segment began (due to time gap or restart). |
| `log_prob` | `float` | Total log-probability of the decoded path across segments. |

### 2.4 How to Call in Python / FastAPI
```python
from pathlib import Path
from src.hmm import load_trip_data, RoadNetwork, HMMParams, decode_trip
from src.hmm.dev_candidates import generate_dev_candidates

# 1. Load trip and projected road network
trip = load_trip_data("00000000", data_dir="data/student_test")
network = RoadNetwork(trip.nodes_xy, trip.arcs)

# 2. Get candidates and classifier probabilities (from Akash K / Akash S or dev stand-in)
candidates, scores = generate_dev_candidates(trip.gps_xy, network, k=15)

# 3. Load tuned parameters
params = HMMParams.from_json("configs/hmm_params.json")

# 4. Decode the route
result = decode_trip(
    timestamps=trip.timestamps,
    gps_xy=trip.gps_xy,
    candidates=candidates,
    scores=scores,
    network=network,
    params=params,
)

print(f"Decoded {len(result.pred_arc_ids)} points with log-prob {result.log_prob:.2f}")
print("Predicted Arc IDs (first 5):", result.pred_arc_ids[:5])
```

---

## 3. Mathematical Formulation (Viva Cheatsheet)

### 3.1 Hidden State Space
At observation time $t \in \{1, \dots, T\}$, the vehicle is on one of $K_t$ candidate arcs:
$$s_t \in \mathcal{C}_t = \{c_{t, 1}, c_{t, 2}, \dots, c_{t, K_t}\}$$

### 3.2 Emission Model
Given classifier output score $P_{\text{cls}}(s_t = c_{t, k} \mid z_t)$, the emission log-probability is:
$$\log P_{\text{emit}}(z_t \mid c_{t, k}) = w_{\text{emit}} \cdot \log\left(\text{score}_{t, k} + \epsilon\right)$$
Where $w_{\text{emit}}$ is `emission_weight` (default $1.0$) and $\epsilon = 10^{-6}$ ensures numerical safety against zero scores.

### 3.3 Transition Model
Vehicular motion is constrained by physical road topology. Between step $t-1$ and step $t$:
- GPS Euclidean displacement:
  $$d_{\text{gps}} = \|\mathbf{x}_t - \mathbf{x}_{t-1}\|_2$$
- Road network travel distance: $d_{\text{net}}(a, b)$ is computed by bounded shortest path on the directed road network.

The transition log-probability is:
$$\log P_{\text{trans}}(s_t = b \mid s_{t-1} = a) = \begin{cases} 
0.0 & \text{if } a = b \text{ (same arc)} \\
-\dfrac{|d_{\text{net}}(a, b) - d_{\text{gps}}|}{\beta} & \text{if } b \text{ is reachable from } a \text{ within } \text{max\_hops} \\
-P_{\text{invalid}} & \text{if } b \text{ is unreachable / disconnected}
\end{cases}$$
Where:
- $\beta$ is `beta` (meters scale factor, tuned to $10.0$ for the latest real-classifier run).
- $P_{\text{invalid}}$ is `invalid_transition_penalty` (soft penalty, tuned to $50.0$ for the latest real-classifier run).
- **Crucial Design Choice:** We use a finite soft penalty rather than $-\infty$. If GPS noise or map incompleteness breaks connectivity, the decoder degrades gracefully rather than failing or dropping observations.

### 3.4 Vectorized Viterbi Recurrence
Dynamic programming computes the maximum log-probability trellis:
$$V_t[j] = \max_{1 \le i \le K_{t-1}} \left( V_{t-1}[i] + \log P_{\text{trans}}(c_{t-1, i} \to c_{t, j}) \right) + \log P_{\text{emit}}(z_t \mid c_{t, j})$$
$$B_t[j] = \arg\max_{1 \le i \le K_{t-1}} \left( V_{t-1}[i] + \log P_{\text{trans}}(c_{t-1, i} \to c_{t, j}) \right)$$

Vectorized in NumPy via broadcasted addition:
$$\text{incoming}[i, j] = V_{t-1}[:, \text{None}] + T_{\text{mat}} \quad \text{of shape } (K_{\text{prev}}, K_{\text{curr}})$$
$$\text{backpointer}[j] = \text{argmax}(\text{incoming}[:, j]), \quad V_t = \max(\text{incoming}, \text{axis}=0) + E_t$$

---

## 4. Edge Cases, Gap Policy, and Fallbacks

1. **Empty Candidate Set at Observation $t$:**
   - Output: `pred_arc_ids[t] = -1`, `source[t] = "no_candidates"`.
   - Behavior: Immediately finalizes and backtracks the preceding segment $[t_{\text{start}}, t-1]$. The next observation with candidates starts a new segment.
2. **Temporal Gaps ($t_i - t_{i-1} > \text{gap\_threshold\_s}$):**
   - When GPS pings drop (e.g. tunnel, signal loss $> 15.0$ seconds), vehicle motion is no longer tightly coupled by 1Hz transition dynamics.
   - Behavior: The trajectory is split. The prior segment is backtracked, and step $t$ starts a new segment break (recorded in `segment_breaks`).
3. **Entire Column Unreachable / Numerical Underflow:**
   - If incoming probabilities are completely degenerate, the step restarts from pure emissions ($V_t = E_t$, `source = "fallback_top1"`).
4. **Single Point Trajectory ($T=1$):**
   - Selects the argmax candidate score, outputs `source = "viterbi"`, no crash.
5. **Loops and Repeated Arcs:**
   - Vehicles naturally stay on the same arc for multiple seconds ($a \to a \to a$) or traverse loops ($a \to b \to a$). Transitions for identical arcs incur $\Delta = 0$ penalty.

---

## 5. Coordinate Ordering and Metric Projection

- **Verification:** Inspection of `.track` and `.nodes` shows:
  - Column 0: Longitude ($X$)
  - Column 1: Latitude ($Y$)
  - Column 2: Timestamp in seconds (only in `.track`)
- **Metric Projection:** All coordinates are projected to Universal Transverse Mercator (UTM) via `pyproj.Transformer` with `always_xy=True`. The UTM EPSG zone is determined dynamically from the median coordinates of the trip:
  $$\text{zone} = \lfloor(\text{lon} + 180) / 6\rfloor + 1, \quad \text{EPSG} = 32600 + \text{zone} \text{ (North)}$$
  All Euclidean calculations operate in metric meters, never in degrees.

---

## 6. Ground-Truth Route Alignment & Limitations

### Alignment Method
`.route` contains the ground-truth sequence of traversed directed `arc_id`s, but lacks per-second GPS timestamps. To compute observation-level Top-1 accuracy, each GPS point is mapped to the spatially nearest arc present in `.route` using `shapely.STRtree(route_lines).query_nearest(pts, all_matches=False)`.

### Honest Viva Limitations
1. **Geometric Ambiguity at Intersections:** At tight switchbacks, overpasses, or cloverleafs, a GPS point may project marginally closer to a different arc of the reference route than the exact one the vehicle occupied at that second.
2. **Speed Profile Unawareness:** Nearest-arc spatial projection does not model vehicle acceleration or deceleration along the arc.

---

## 7. Latest Real-Classifier Results (Student Test Set)

This run uses the saved Histogram Gradient Boosting classifier and candidate probabilities
from the Issue #2 interface. The classifier was trained in fast mode on a deterministic
100,000-row sample with 2-fold trip-grouped CV; HMM parameters were tuned on the first
five training trips. The reported student-test scores use student-test `.route` files
for offline evaluation. The `evaluator_test` partition was not accessed. Run provenance
is recorded in `results/hmm/provenance.json`.

| Trip ID | Points | Top-1 Acc | HMM Acc | $\Delta$ Acc | Top-1 Cont | HMM Cont | $\Delta$ Cont | Miss Rate | HMM Time | Status |
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

## 8. Optimal Tuned Hyperparameters

Tuned via 3-fold GroupKFold CV on five training trips using real classifier candidate
probabilities. The best grid score was 0.8685. Parameters:
- `beta`: `10.0`
- `emission_weight`: `0.5`
- `invalid_transition_penalty`: `50.0`
- `max_hops`: `5`
- `max_meters`: `2000.0`
- `gap_threshold_s`: `15.0`
- `eps`: `1e-06`

Stored at `configs/hmm_params.json`.

---

## 9. Upstream Integration & Adapters

- **Development Stand-in:** `src/hmm/dev_candidates.py` provides spatial candidate search and simulated softmax emission probabilities.
- **Teammate Adapter:** `src/hmm/adapters.py` provides `adapt_dataframe_to_decoder`, `adapt_dict_to_decoder`, and `adapt_matrices_to_decoder` to automatically bridge Akash K and Akash S's pipeline outputs into the decoder contract without modifying their code.

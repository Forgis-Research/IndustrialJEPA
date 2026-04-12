# Lead-Time Analysis: MTS-JEPA Anomaly Prediction

## Critical Finding

**The majority of "correctly predicted" anomalies in MTS-JEPA's evaluation are actually CONTINUATION detections — the context window already contains anomalies.**

| Dataset | TRUE_PREDICTION | CONTINUATION | BOUNDARY | Total Anomalous Targets |
|---------|:-:|:-:|:-:|:-:|
| PSM | **15.5%** (45) | 84.5% (246) | 0% | 291 |
| MSL | **30.4%** (35) | 69.6% (80) | 0% | 115 |
| SMAP | **10.9%** (67) | 89.1% (548) | 0% | 615 |

### Definitions
- **TRUE_PREDICTION**: Context window is fully normal, target window contains anomaly → genuine early warning
- **CONTINUATION**: Context window already contains anomaly → detecting ongoing anomaly, not predicting
- **BOUNDARY**: Anomaly starts in last 20% of context → near-detection

### Implications

1. **MTS-JEPA's core claim is weakened**: "Anomaly prediction" is mostly continuation detection (70-89% of cases)
2. **Only 11-30% of predictions are genuinely proactive** — the model sees anomalous patterns in the context and (correctly) predicts they continue
3. **Fair evaluation should report AUC on TRUE_PREDICTION subset only** — this would show the real predictive capability
4. **SMAP is worst**: Only 10.9% true prediction → 89% of the "prediction" task is trivial for any model that detects anomalies in the input
5. **MSL is best**: 30.4% true prediction → most genuine predictive challenge

### Why This Matters for Our NeurIPS Extension

- A method with **causal masking** (our CC-JEPA) would be evaluated on the same metric breakdown
- If CC-JEPA scores better on the TRUE_PREDICTION subset while MTS-JEPA relies on CONTINUATION, that's a strong argument for causal architecture
- The lead-time-aware AUC (computed only on context-normal pairs) is the proper metric for genuine anomaly prediction

### Methodology

Analysis uses non-overlapping windows of T_w=100 from the official test splits. For each context-target pair (X_t, X_{t+1}):
1. Check if context window X_t contains any anomalous points (point-level labels)
2. Check if target window X_{t+1} contains any anomalous points (window-level label)
3. Categorize based on context anomaly status

No model predictions are used in this analysis — this is a property of the dataset and evaluation protocol.

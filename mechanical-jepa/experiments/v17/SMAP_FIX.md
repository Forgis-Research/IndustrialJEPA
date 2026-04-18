# SMAP/MSL Bug Diagnosis and Fix Plan

## Bugs found in V16 Phase 3

### Bug 1: Argument order swap (CRASH)
`v16/phase3_smap_100epochs.py` line 168:
```python
metrics = anomaly_metrics(labels, test_scores, threshold)  # WRONG
```
Should be:
```python
metrics = anomaly_metrics(test_scores, labels, threshold)  # correct
```
Function signature: `anomaly_metrics(scores, y_true, threshold)`.
This is the `KeyError: 'non_pa_f1'` crash.

### Bug 2: Wrong encoder for scoring
`data/smap_msl.py` line 254:
```python
h_full = model.encode_context(x_full, mask=full_mask)  # context encoder
```
Should use target encoder for proper JEPA prediction error. For SIGReg (shared encoder) this is technically OK. For EMA mode (V17), must use the EMA target encoder.

### Bug 3: Anomaly signal is inverted
Diagnostic from overnight session:
- Anomaly windows: mean score = 0.809
- Normal windows:  mean score = 0.869
Anomalies score LOWER → model predicts anomalous patterns as well as (or better than) normal.

**Root cause**: SMAP anomalies are systematic drifts, not random spikes. With 100 epochs of SIGReg training, the model learns to predict these patterns too. More training = worse anomaly detection (paradox).

**Hypothesis**: V15 20-epoch results were better BECAUSE the model was undertrained — it only learned normal-operation dynamics, not anomaly dynamics. This is actually the correct behavior: we WANT the model to only learn normal patterns so anomalies are surprising.

## Fix for V17

1. Fix argument order (trivial).
2. For EMA mode: score = ||predictor(h) - target_encoder(future)||, not context_encoder.
3. Early stopping on anomaly detection: stop pretraining when prediction error on a held-out normal set stabilizes. Don't overtrain.
4. Consider: train fewer epochs on SMAP (20-50, not 100) or use early stopping.
5. The LogUniform k change may help — short-horizon predictions (k=1-5) should be very accurate on normal data, making small deviations more detectable.

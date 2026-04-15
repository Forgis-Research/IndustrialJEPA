# V15 Metrics Report

## Summary of Metric Choices

### RUL (Time-to-Failure)

**Primary metric: RMSE**
RMSE is the universal currency for C-MAPSS comparisons. Every major paper
(STAR 2024, AE-LSTM, DC-SSL, CTTS) reports RMSE with RUL cap 125. This
is our primary metric for internal comparison and paper tables.

**Secondary metric: nRMSE**
Normalized RMSE = RMSE / RUL_range allows cross-domain comparison.
On C-MAPSS FD001, nRMSE = RMSE / 125 approximately.
Essential once we add FEMTO bearing data (RUL scale: thousands of cycles).

**When to report NASA score:** For safety-critical narratives only. The
asymmetric penalty (late: exp(e/10)-1, early: exp(-e/13)-1) is correct
operationally but unbounded and not directly comparable across papers.

**RA (Relative Accuracy):** Intuitive for operators but threshold-sensitive.
Not standard in ML literature. Report optionally.

### Anomaly Detection

**Primary metric: non-PA F1**
Point-Adjust (PA) inflates F1 by up to 30pp (demonstrated in Phase 0a).
PA rewards "any detection in a segment" - a model that fires once per
segment gets full credit. We report non-PA F1 as the honest number.

**Why also report PA F1:** Comparisons with literature (THOC, TranAD,
AnomalyTransformer, MTS-JEPA) require PA F1 for apples-to-apples comparison.
We report both clearly labeled.

**Secondary metric: AUC-PR**
Threshold-free, handles class imbalance (anomalies are rare ~1-5%). Best
for comparing different methods without threshold tuning.

**TaPR:** Segment-level credit with temporal buffer delta=0.1. More
operationally meaningful than point F1 for alarm systems (Kim et al. 2022).

### Threshold Exceedance (TTE)

**Primary metric: nRMSE**
Normalizes by max TTE to allow comparison across different event horizons.

**Secondary metric: RMSE**
Absolute cycles - operationally meaningful ("off by N cycles").

**Definition:** SPC 3-sigma rule, baseline = cycles 1-50 (healthy window).
Standard in industrial process control.

## C-MAPSS FD001 s14 Exceedance Analysis

Sensor s14 = corrected fan speed (Nc), known to be physics-relevant
(V14 found s14 is the attention concentration target during degradation).

Engines with s14 exceedance: 93/100

Results with ridge regression on hand-crafted features:
  - RMSE: 32.98 cycles (if finite)
  - nRMSE: 0.1182
  - n_train_samples: 8410
  - n_test_samples: 1068

**Interpretation:**
If fewer than 50% of engines have s14 exceedances (3-sigma), the TTE task
is sparse on FD001. This is expected - C-MAPSS engines degrade gradually
and s14 may not cross 3-sigma. Options:
  1. Use 2-sigma threshold (more sensitive)
  2. Use multiple sensors (first exceedance across any sensor)
  3. Focus TTE benchmark on SMAP/SWaT where anomalies are labeled

## Unified Evaluation Module

Located at: mechanical-jepa/evaluation/grey_swan_metrics.py

API:
  from evaluation.grey_swan_metrics import GreySwanEvaluator

  ev = GreySwanEvaluator(event_type='rul', rul_cap=125.0)
  metrics = ev.evaluate(predictions, targets)
  print(ev.summary(metrics))

Supports: 'rul', 'anomaly', 'tte' event types.
Implements: RMSE/nRMSE/NASA/RA, non-PA F1/PA-F1/AUC-PR/TaPR, TTE-RMSE/nRMSE.

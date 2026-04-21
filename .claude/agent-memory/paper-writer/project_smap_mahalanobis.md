---
name: SMAP Mahalanobis Finding (V18 Phase 4b)
description: Mahalanobis(PCA-10) on h_past achieves SMAP PA-F1 0.73 with same v17 encoder; the paper's anomaly story is now positive, not negative
type: project
---

V18 Phase 4b flipped the SMAP story. Using the SAME v17 pretrained encoder (no re-training), four different scoring geometries give very different anomaly results:

| Scoring              | non-PA F1 | PA-F1  | AUC-PR | Sign gap |
|----------------------|-----------|--------|--------|----------|
| L1 prediction error  | 0.038     | 0.219  | ---    | -0.61    |
| Representation shift | 0.057     | 0.593  | 0.115  | -0.60    |
| Trajectory div.      | 0.090     | 0.605  | 0.124  | -0.64    |
| Mahalanobis (PCA-10) | 0.100     | 0.733  | 0.173  | +1.70    |
| MTS-JEPA (paper ref) | ---       | 0.336  | ---    | ---      |

The Mahalanobis sign gap (+1.70) is CORRECT - anomalies really are far from training h_past.

**Why:** The prediction-error-as-anomaly-score reduction fails for RECURRENT anomalies because repeated anomaly morphologies become *more* predictable, not less. Distance-to-manifold scoring is the correct abstraction for recurrent regimes.

**How to apply:** 
- Lead the abstract and Contribution #4 with Mahalanobis PA-F1 0.73 vs MTS-JEPA 0.34.
- Keep non-PA F1 0.10 caveat and the single-seed / single-dataset limitation.
- The story: "one encoder, two scorings" - L1 prediction error for monotonic degradation (C-MAPSS), Mahalanobis for recurrent telemetry (SMAP).
- MSL re-evaluation and multi-seed robustness are explicit future work.
- Earlier V15 20-epoch number (PA-F1 0.625) was a superseded-checkpoint artefact and should NOT reappear in any draft.

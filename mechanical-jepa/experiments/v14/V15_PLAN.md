# V15 Plan

Consolidated from V14 overnight session findings (2026-04-14).

## What V14 established

### Positive architectural results

1. **Full-sequence target encoder (Phase 2)**: frozen -2.1 RMSE at 100%,
   wins at 20%/10%, regresses at 5%. Low-label-brittle but moderate-label win.
2. **Cross-sensor attention (Phase 3)**: frozen -2.8 RMSE at 100%
   (**new best 14.98**), brittle at low labels.
3. **Both architectural wins require pretraining** (Phase 3c: cross-sensor
   needs +21.5 RMSE of pretraining work at 10%).

### Physics-aligned attention pattern

During degradation, many temperature and pressure sensors
(s2, s3, s4, s11, s15) concentrate attention on **s14 (core speed Nc)**.
The five largest healthy-to-degradation attention shifts all point at s14.
Consistent with PCA PC1 dominance (47.6% variance).

### Honest negative results

1. **Phase 2 at 5% labels**: full-sequence REGRESSES +5.0 RMSE vs V2 frozen.
2. **Phase 3 at low labels**: cross-sensor is brittle (20% std 10.19).
3. **Prediction-error anomaly (Phase 5c.4)**: mean Spearman rho = +0.02
   across 5 engines - NOT a reliable zero-label degradation signal.
4. **AE-LSTM (SSL audit, Phase 5b)**: best-of-28 hyperparameter grid, not
   comparable to our 5-seed mean; we don't claim to beat it.

### Theory (Paper Section 6)

Three-part rationale: SFA bias, information-theoretic MI argument,
frozen-vs-E2E label-gradient-bias. Presented as principled sketch,
not formal theorems.

## V15 open directions

### High priority

1. **Unified architecture winning across budgets**. V2 is best at 5%,
   full-sequence at 20/10%, cross-sensor at 100%. Hypothesis: low-label
   fine-tuning needs *simpler* pretrained representations. Possible
   approaches:
   - Regularize cross-sensor to be more robust (sensor-token dropout,
     shorter sensor embedding, smaller d).
   - Ensemble: use cross-sensor at high labels, V2 at low labels.
   - Early-stop pretraining to limit representation richness.

2. **Cross-sensor generalization to FD003/FD004**. Does the physics-aligned
   attention (s\* -> s14 during degradation) replicate on other fault
   modes? If yes, strong multi-subset narrative.

3. **AE-LSTM head-to-head replication on our pipeline**. ~100 lines
   PyTorch per the SSL audit. Settle the 13.99 vs 14.23 question with
   matched splits/seeds/RUL-cap.

### Medium priority

4. **Dual-resolution predictor** from MTS-JEPA. Fine (local) + coarse
   (full-history compressed) branches. Could help long engines.

5. **Codebook regularization** (MTS-JEPA style). Deferred pending
   batch-size sensitivity study.

6. **Prediction-error anomaly with alternative metrics**. V14 tried
   L1 distance and got mean rho = +0.02. Alternatives to try:
   representation norm, per-dimension variance of prediction error,
   only evaluate during degradation phase, or ratio to target encoder
   variance.

7. **Cross-domain pretraining**: FD001+FD003 -> FD004. V11 showed
   FD002 -> FD001 worked weakly; FD003 -> FD001 is untested.

### Theoretical

8. **Formalize the SFA connection**. Verify on synthetic signals where
   one dimension is slow and others are fast. Promote Section 6's sketch
   to a formal proposition.

9. **Information bottleneck analysis**. The H.I. equivalence (R^2=0.926
   = frozen RMSE 17.81) suggests the encoder is extracting a 1-D health
   signal. Measure I(h_past; HI) vs I(h_past; other sensor dimensions)
   to formalize this.

## Session scope guidance

V15 should be a **focused rescue session for low-label robustness**.
Specifically:
- Goal: find an architecture that matches V2 at 5% (rmse <= 22) AND
  matches cross-sensor at 100% (rmse <= 15).
- Ways: sensor-token dropout during pretraining, mixing V2 and
  cross-sensor heads with an MoE-style gate, progressive training
  (start V2-like, add cross-sensor capacity).

## Status of V14 artifacts

All in `mechanical-jepa/experiments/v14/`:
- `phase2_full_sequence.py` + checkpoint + results JSON
- `phase2b_full_sequence_lowlabel.py` + results JSON
- `phase3_cross_sensor.py` + checkpoint + results JSON + attention maps JSON
- `phase3c_cross_sensor_fromscratch.py` + results JSON
- `phase4_plots.py` + 3 figure PNG/PDF
- `phase5c_prediction_error_anomaly.py` + results + figure
- `ssl_comparison_audit.md`
- `mtsjepa_comparison.md`
- `theory_draft.md`
- `RESULTS.md`
- `V15_PLAN.md` (this file)

Paper: `paper-neurips/paper.tex` with new Sections 5.4 (full-sequence),
5.4b (cross-sensor), 5.5 (from-scratch), Section 6 (theory). New figures
in `paper-neurips/figures/v14_*.pdf`.

Quarto notebook: `mechanical-jepa/notebooks/14_v14_analysis.qmd`.

Total commits this session: ~14, all pushed.

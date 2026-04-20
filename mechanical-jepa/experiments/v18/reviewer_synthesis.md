# V18 Phase 3: NeurIPS Reviewer Synthesis

**Date:** 2026-04-20
**Inputs:** 4 independent neurips-reviewer passes on `paper-neurips/paper.tex`, each blind to the others. All four gave **4/10 (weak reject)** with high confidence (4/5 each).

## Unanimous consensus issues (flagged by all 4)

### SEVERE-1. SMAP headline F1 62.5% is broken

Abstract + Contribution #4 + Table 1 cite "F1 62.5% vs MTS-JEPA 33.6%". Reviewers independently found:
 - Paper's own footnotes reveal this is point-adjusted (PA-F1), not non-PA.
 - Paper's declared primary metric is non-PA F1, which is 6.9% on SMAP (below the 7.1% random baseline) and 7.9% on MSL.
 - V16 100-epoch diagnostic shows anomaly-window prediction error is *below* normal-window error - score sign may invert.
 - The 62.5% is from a 20-epoch V15 checkpoint the authors themselves have flagged as superseded.

**Action:** Replace with v17 Phase 5 numbers (non-PA 0.038 / PA 0.219). Reframe Contribution #4 as a negative/auxiliary result or report AUROC/AUPRC (sign-invariant).

### SEVERE-2. SIGReg and CrossVar appear as contributions but are `\plannedc{}`

Contributions (ii) and (iii) in the intro are wrapped in draft-mode blue text. SIGReg has val-only numbers from a biased last-window protocol (RUL=1 for all val engines) and oscillates seed-to-seed (10.2 vs 11.6 on rerun). CrossVar attention does not transfer to FD003 per §5.6.

**Action:** Either deliver test-set numbers for both (budget-heavy) or demote to "preliminary/negative ablations" and list only causal trajectory JEPA as the architectural contribution.

### SEVERE-3. Single-subset (FD001) primary evaluation insufficient for "Forecast-Anything"

FD002 test RMSE 26.07 (vs STAR 15.87), FD004 RMSE 29.35 (vs STAR 15.87). FD003 not in main body. SWaT is "future work" with `\placeholder{}` cells. Only one anomaly benchmark (SMAP) where the claim is retracted per SEVERE-1.

**Action:** Either run full label-efficiency sweep on FD003 + FD004 (Phase 6 addresses this partially) or retitle to "Causal JEPA for Label-Efficient Turbofan RUL". Don't submit with the foundation-model framing.

### MAJOR-4. No paired significance tests on the crossover claim

"Frozen at 5% beats STAR" is 21.53 ± 2.0 vs 24.55 ± 6.4 with 5 seeds. Unpaired t ≈ 1.0, p > 0.3. STAR's σ=6.4 fully envelops FAM's mean. Similar margins at 10% (18.66 vs 18.72, effectively zero).

**Action:** Use paired Wilcoxon / paired t-test on matched engine-subset splits. Report p-values, bootstrap 95% CI in Figure 4 and Table 2. Soften abstract language accordingly ("within 1σ" or "indistinguishable under paired test" if non-significant).

### MAJOR-5. STAR replication gap (12.19 vs paper 10.61) is unexplained

1.6 RMSE gap at 100% labels, larger than FAM's std. If replication is broken at 100%, the 5%-labels comparison inherits the bias. Reviewers suggest: run STAR with their HP search, explain the gap in a footnote, or use paper's 10.61 as ground truth.

**Action:** Investigate STAR reproduction. If closing the gap is infeasible, footnote with the replication protocol and acknowledge the caveat in the abstract.

### MAJOR-6. No head-to-head with TS-JEPA / MTS-JEPA on C-MAPSS

Paper claims "first JEPA for RUL" but doesn't run TS-JEPA or MTS-JEPA on FD001. This is the most-direct competitor class and the code is public.

**Action:** (Either) port TS-JEPA / MTS-JEPA to C-MAPSS (~2 days eng) OR narrow the claim to "first JEPA variant to beat supervised STAR at low labels" with the empirical caveat. Phase 4 in v18 partially addresses the comparison narratively.

## Consensus issues (flagged by 2-3 reviewers)

### MAJOR-7. Frozen probe protocol is opaque; some earlier runs were broken

V17 Phase 2 already corrected this (WD=1e-2, val n_cuts=10, yielding 15.53 vs v11's 17.81). Reviewers notice inconsistency: Table 2 says "best probe RMSE 16.9 at ep 45" in one appendix and 17.81 in another. Full-sequence row's 15.70 ± 0.21 suspiciously tight.

**Action:** Re-audit Table 2 / Table 3 / Appendix tables under the honest protocol (V18 Phase 0 + 1a). Document the protocol in one place in §5.1. V18 Phase 0 honest re-probe of V2 (15.73 ± 0.14) is the anchor.

### MAJOR-8. "Zero-label anomaly" framing uses test data for threshold calibration

§5.4: 95th percentile of scores from the first 10% of *test* data. This is not truly zero-label. Baselines (MTS-JEPA, TS2Vec) may be sweeping thresholds.

**Action:** Reframe as "label-free pretraining + semi-supervised threshold" or calibrate from training set only. Report AUROC/AUPRC alongside F1 for sign/threshold invariance.

### MAJOR-9. Missing LSTM / TS2Vec / PatchTST baselines on FD001

The SSL contribution claim needs these. Only AE-LSTM (from third-party paper) is cited as SSL reference.

**Action:** Run TS2Vec frozen probe on C-MAPSS FD001 at matching label budgets (~1 day eng). A supervised 2-layer LSTM at 5% labels is also needed.

### MINOR-10. Title "Forecast-Anything" overclaims

Given results on 2 datasets × 1 strong task + 1 broken task, the "any event type" framing is aspirational.

**Action:** Retitle. Candidate: "Causal Latent Trajectory Prediction for Label-Efficient Turbofan RUL".

## Additional findings per reviewer

**Reviewer 1** (JEPA/SSL-focused): Flagged `\plannedc{}` placeholders as "red flag" for unfinished work; recommended removing them.

**Reviewer 2** (Skeptical): Emphasized that E2E (14.23) and Frozen (17.81) stories are ambiguous - which is "the SSL claim"? Questioned whether E2E benefit is really from SSL vs compute.

**Reviewer 3** (Prognostics practitioner): Flagged missing NASA Scoring Function - STAR reports NASA-S=195; FAM reports 395.7 in appendix. 2x worse on operational metric than on RMSE. Also flagged RUL cap (125) not ablated.

**Reviewer 4** (SSL/RL-focused): Questioned sign stability of prediction-error anomaly score; recurrent anomalies may be *more* predictable than novel normals.

## Prioritized revision list

### Must-do before submission (blocking):
1. **Fix SMAP headline** - replace 62.5% with v17 honest numbers; reframe contribution #4.
2. **Demote SIGReg / CrossVar** from contributions list unless test-set numbers are delivered.
3. **Add paired significance tests** on the crossover claim.
4. **Reconcile frozen probe protocol** - use v18 Phase 0/1a numbers consistently.
5. **Explain STAR reproduction gap** or use paper numbers.

### Should-do (highly improves):
6. Run v18 Phase 1b E2E honest-protocol numbers (in progress).
7. Run FD003/FD004 label-efficiency sweep (Phase 6).
8. Add NASA-S metric to RUL tables.
9. Compare TS-JEPA/MTS-JEPA on FD001.
10. Retitle to match scope.

### Nice-to-have:
11. TS2Vec baseline.
12. Supervised LSTM at matched budgets.
13. Reconstruction-error anomaly baseline on SMAP.

## Bottom line

4/4 weak rejects with high confidence (4/5). Paper has a real result (causal JEPA inductive bias for cross-machine transfer) but the packaging overclaims. With focused revision on items 1-5 above + at least one additional benchmark (FD003 or FD004), this could move to a 6 (weak accept). Without those, a resubmission to a workshop is the more realistic venue.

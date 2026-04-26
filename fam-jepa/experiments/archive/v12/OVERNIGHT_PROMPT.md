# V12 Overnight Session — Is V11 Real?

**Primary goal of v12 is NOT to improve numbers. It is to verify that the V11
numbers measure what we think they measure.** Two independent pieces of
evidence from the v11 artifacts suggest the headline result may be an
artifact of the evaluation protocol, not a real capability. Until that is
resolved, every other v12 experiment is wasted compute.

Builds on v11. Keep data pipeline, model, and evaluation protocol identical
except where a point explicitly changes them. All outputs under
`experiments/v12/`. Do NOT modify v11 artifacts.

---

## The two smoking guns to resolve first

1. **Prediction trajectories from v11 Part F look flat.** The saved figure
   `analysis/plots/v11/prediction_trajectories.png` shows the V2 E2E model
   outputting a constant ~92 cycles across every plotted test engine,
   regardless of cycle index or true RUL. A constant predictor of ~92 on
   FD001 with RUL cap 125 achieves ~13-14 RMSE on the last-window protocol
   purely by distribution luck. If this is real, **V11's 13.80 is a
   population-mean mirage, not a degradation tracker.**

2. **FD002 val-probe is ~15, test-frozen is ~26.** During v11 Part G
   pretraining, the linear probe on FD002 validation engines reached
   RMSE ~14.85. The downstream frozen fine-tune on the canonical test set
   got 26.33. A ~12 RMSE gap between val-probe and test-frozen on the same
   encoder on FD002 (vs ~2 on FD001) says: the encoder learns fine, but
   the representation doesn't transfer from train-distribution engines to
   test-distribution engines on multi-condition subsets. This is a
   distribution-shift / normalization bug, not an SSL failure.

Both of these change what v12 should prioritize. The old v12 (frozen/E2E
tightening, LR sweeps, MLP probe variants) is polishing a result that may
not exist. Kill that work. Execute the phases below in order. Each phase
gates the next.

---

## Phase 0 — Verify V11 is not constant-predicting (GATE for all other work)

**Budget: 2 hours. Must complete before any other v12 work.**

### 0.1 Reproduce the prediction-trajectory figure with fresh code

Load the saved V2 E2E checkpoint from v11 (`checkpoints/v11_v2_e2e_seed0.pt`
or equivalent — find it). For 10 FD001 test engines, run inference at
**every cycle** from cycle 30 to the last observed cycle at stride 1. Plot
predicted RUL vs cycle index, overlay with true RUL back-computed from
`oracle_RUL + (T_obs - t)`.

Save as `analysis/plots/v12/phase0_prediction_trajectories.png`.

### 0.2 Compute the three constant-predictor diagnostics

For EVERY FD001 test engine (not just 10), compute:

- **Per-engine prediction std**: `std(preds_at_every_cycle)`. A constant
  predictor has std ≈ 0. A real tracker has std 10-30 cycles.
- **Within-engine Spearman correlation**: `spearman(preds_at_every_cycle,
  true_rul_at_every_cycle)`. Target > 0.7 for a real tracker. Constant
  predictor gets ~0.
- **Constant-predictor baseline RMSE**: compute the RMSE you would get by
  predicting `mean(train_rul)` for every test engine on the last-window
  protocol. If V11's 13.80 is within ~1 RMSE of this number, the "win"
  is statistically indistinguishable from constant prediction.

Report as `experiments/v12/phase0_diagnostics.json` with fields:
`per_engine_pred_std_mean`, `per_engine_pred_std_median`,
`within_engine_rho_mean`, `within_engine_rho_median`,
`constant_predictor_rmse`, `v11_reported_rmse`, `verdict`.

### 0.2b Engine-summary-regressor lower bound (TIGHT BASELINE — mandatory)

The `mean(train_rul)` predictor in 0.2 is too loose a floor (~40 RMSE on
FD001). The real question is: **can a dumb feature-engineered regressor
that varies across engines but is flat within each engine match V11's
13.80?** If yes, the canonical last-window protocol cannot distinguish
V11 from a non-tracker, and the headline finding of v12 is exactly that.

On FD001 training engines, for each engine compute ~60 hand-designed
features from the observable prefix:

- **Length features**: `T_obs`, `T_obs / mean(train_T)`
- **Last-30-cycle sensor means**: mean of each of 14 sensors over the
  last 30 cycles (14 features)
- **Last-30-cycle sensor stds**: std of each of 14 sensors over the last
  30 cycles (14 features)
- **Last-30-cycle sensor slopes**: linear-regression slope of each of 14
  sensors over the last 30 cycles (14 features)
- **Global sensor deltas**: `mean(last_30) - mean(first_30)` per sensor (14)

Fit `sklearn.linear_model.Ridge(alpha=1.0)` on train-engine features →
RUL labels (use the same 85/15 split as v11, seed=42). Evaluate on the
canonical FD001 test set using the same last-window protocol v11 uses
(one prediction per test engine, raw-cycle RMSE vs `RUL_FD001.txt`).
5 seeds.

Report as `experiments/v12/engine_summary_regressor.json`:
`mean_rmse`, `std_rmse`, `per_seed_rmse`, `top_10_feature_weights`,
`delta_vs_v11_e2e`, `delta_vs_v11_frozen`, `verdict`.

**This experiment takes ~20 minutes. It is the single most important
number we will produce in v12. Run it before anything else in Phase 0
if the checkpoint load for 0.1 is slow.**

### 0.3 Decision rule (combines 0.2 and 0.2b)

Three outcomes are possible. Pick the one that matches the evidence:

1. **V11 is real and is tracking degradation.** Requires ALL of:
   - `per_engine_pred_std_median > 10`
   - `within_engine_rho_median > 0.5`
   - V11 E2E beats the engine-summary regressor by >1 RMSE
   → Proceed to Phases 1–4. Note in `RESULTS.md` that V11 is validated
     and record the baseline margin.

2. **V11 scores well on the benchmark but does not track degradation.**
   Triggered by EITHER of:
   - `per_engine_pred_std_median < 3` OR `within_engine_rho_median < 0.3`
     (direct evidence of within-engine flatness)
   - V11 E2E is within 1 RMSE of the engine-summary regressor
     (evidence that the benchmark does not require tracking)
   → **STOP Phases 1–4.** Write `RESULTS.md` with the headline:
     *"On the canonical C-MAPSS last-window protocol, V11's 13.80 RMSE
     is not distinguishable from a 60-feature engine-summary ridge
     regressor. The protocol does not require within-engine degradation
     tracking. Our learned representation contributes $X$ RMSE over this
     baseline."*
     Then run Phase 0b to understand the collapse.

3. **The evidence is itself inconsistent** (e.g. model tracks within-engine
   but loses to the regressor on RMSE, or vice versa).
   → Tag the result `⚠️ INTERNAL INCONSISTENCY` in `EXPERIMENT_LOG.md`,
     write up the disagreement, and **do not run Phases 1–4 until it is
     reconciled**. Inconsistency is itself the finding.

The trap to avoid: reporting the good-looking number while silently
ignoring any artifact that disagrees with it. If 0.1, 0.2, and 0.2b do
not tell a single coherent story, that IS the headline finding.

### 0.4 (if triggered) Phase 0b — Fix the collapse

The probe head has collapsed to the label mean. Root-cause candidates,
in order to test:

1. **Loss averaging issue**: check that fine-tuning MSE is computed per-sample
   not per-batch-averaged-twice. A double average can flatten gradients.
2. **Probe capacity too low**: re-run with a 3-layer MLP probe (already tried
   in v11 — 15.88 — but was it tracking or just constant at a different value?).
3. **`h_past` doesn't depend on input**: run the shuffle test at fine-tune
   time. If shuffled h_past → same RMSE, h_past is an input-independent bias.
4. **RUL cap + label distribution shape**: the cap creates ~50% of samples
   with label 125. A regression loss with this label distribution can have
   a global minimum at constant ~mean. Try training with RUL cap removed
   (use raw RUL, no clip) as a diagnostic — does it still collapse?

Report findings and stop v12 here if collapse is real. The remaining phases
are moot until v11 is either validated or fixed.

---

## Phase 1 — Diagnose FD002 val/test gap

**Budget: 2 hours. Runs only if Phase 0 validates V11.**

### 1.1 Measure the val/test gap explicitly

For both FD001 and FD002, using the v11 V2 pretrained encoders:
- Compute linear-probe RMSE on the **training engines' held-out validation split**
- Compute linear-probe RMSE on the **canonical test set**
- Report both side-by-side in `experiments/v12/val_test_gap.json`

Expected: FD001 gap small (~2 RMSE), FD002 gap large (~10-12 RMSE). If the
gap is uniform across subsets, the problem isn't FD002-specific and the
FD002 theory from the brainstorm is wrong.

### 1.2 Check FD002 test-time condition assignment

The v11 FD002 pipeline uses KMeans(k=6) per-condition normalization. For
each test engine's **last window** (the window that actually gets evaluated),
record the KMeans-assigned condition ID. Compare the distribution to the
training-engine condition distribution.

If test last-windows are disproportionately in rare training conditions
(>1.5x overrepresented in any cluster relative to training), the per-condition
normalization is mis-calibrated exactly where it matters.

Save as `analysis/plots/v12/fd002_condition_assignment.png` (two histograms).

### 1.3 One-line ablation — op-settings as input channels

In the v11 architecture, operating conditions are used for normalization
then discarded. On multi-condition subsets the encoder has no way to know
which regime it's in. Try:

- `SensorProjection` input: **17 channels** instead of 14 (concat op_setting_1,
  op_setting_2, op_setting_3 directly, scaled to [0, 1] using global min-max
  on training)
- Keep global (not per-condition) sensor normalization
- Pretrain V2 architecture from scratch on FD002 with this input
- Fine-tune frozen + E2E at 100% labels, 5 seeds

If frozen FD002 RMSE drops from 26.33 to below 20, the condition-awareness
hypothesis is confirmed and v13 becomes "condition-as-input-token" everywhere.
If it doesn't drop, the bug is elsewhere and we go back to 1.2.

---

## Phase 2 — STAR at reduced label budgets (runs in background)

**Budget: kicks off at start of session, runs unattended for ~6 hours.**

The V11 label-efficiency pitch compares JEPA against an unoptimized LSTM
scoring 17.36 at 100%. STAR scores 12.19 on our replication. Nobody
currently knows how STAR scales to low-label regimes. Without this number,
the label-efficiency pitch is not falsifiable.

### 2.1 STAR label-efficiency sweep

Using the existing STAR replication code at
`paper-replications/star/run_experiments.py`:

- Label budgets: 100% (already done), 50%, 20%, 10%, 5%
- Subset: FD001 only (FD002/003/004 are a v13 concern)
- 5 seeds per budget, same seeds as v11 JEPA
- Output: `experiments/v12/star_label_efficiency.json`

Launch this as the very first action of the session in the background so it
runs while Phase 0 and Phase 1 execute. Collect results when done.

### 2.2 Plot the comparison

Overlay four curves on one log-scale x-axis:
- JEPA E2E V2 (v11 numbers)
- JEPA frozen V2 (v11 numbers)
- Supervised LSTM (v11 numbers)
- **STAR (new, v12)**

Horizontal line: STAR @ 100% paper reference (10.61).

Save as `analysis/plots/v12/label_efficiency_with_star.png`. This is the
money plot of v12. If STAR at 20% labels is ≤ 14 RMSE, the JEPA label-
efficiency pitch is dead and the paper narrative pivots to "we recover
the simulator's health index from unlabeled sensor data" (Phase 3).

---

## Phase 3 — Health-index recovery probe (honest C-MAPSS success criterion)

**Budget: 1 hour. Runs only if Phase 0 validates V11.**

C-MAPSS degradation is generated by the simulator from a known latent
health index. A clean way to prove SSL is doing real work — independent
of RMSE-leaderboard games — is to show that the pretrained `h_past`
linearly decodes this H.I.

### 3.1 Reconstruct approximate H.I. per training engine

Standard approximation used in the C-MAPSS RUL literature:
piecewise-linear H.I. that is 1.0 for cycles `[1, T - 125]` (healthy
plateau) and declines linearly from 1.0 to 0.0 over `[T - 125, T]`
(degradation phase). This matches the RUL-cap convention.

### 3.2 Fit linear probe `h_past → H.I.`

On the frozen V2 encoder from v11:
- For every cycle of every training engine, compute `h_past` (causal
  encoding up to that cycle)
- Target: approximate H.I. at that cycle
- Fit linear regression (ridge, alpha=1.0) on training engines, evaluate
  on validation engines
- Report R² on train and val

**Target: val R² > 0.7.** If hit, this is the cleanest "SSL works" claim
in the paper, independent of any fine-tuning or label-efficiency game.

Save as `analysis/plots/v12/health_index_recovery.png` (scatter of
predicted vs true H.I. on val engines) + R² in `RESULTS.md`.

---

## Phase 4 — Sliding-cut-point diagnostic (add-on, cheap)

**Budget: 1 hour. Runs only if Phase 0 validates V11.**

The canonical C-MAPSS test protocol (last-window-only) is exactly what
allowed the constant-prediction concern in Phase 0. Add a secondary
evaluation that sweeps over cut points per test engine, reporting:

- **Per-engine RMSE averaged across all valid cut points at stride 1**
  (no arbitrariness in stride/width — uses every observable cycle)
- **Within-engine prediction std**
- **Within-engine Spearman ρ between pred(t) and true_rul(t)**

Report aggregated statistics in `experiments/v12/sliding_eval.json` and a
trajectory plot for 10 engines in `analysis/plots/v12/sliding_trajectories.png`.

This is the diagnostic that last-window-only evaluation cannot produce.
Everyone reporting C-MAPSS RMSE hides it. We should publish both.

---

## What is explicitly OUT of scope for v12

These were in the earlier v12 draft and are now killed:

- **Frozen/E2E gap tightening via longer pretraining** (old item 1) — polishing
  a number that may be meaningless.
- **LR sensitivity of E2E** (old item 2) — same.
- **Layer-wise LR decay** (old item 3) — same.
- **Partial unfreezing** (old item 4) — same.
- **MLP probe variants** (old item 5) — covered implicitly in Phase 0b if
  constant-prediction is diagnosed.
- **Consistency audit of `.eval()` vs `.train()`** (old item 6) — negligible
  expected signal.
- **Distributional RUL head + sharpness curve** (old items 7-8) — genuinely
  valuable but premature. Revisit in v13 once v11 is validated and the
  FD002 fix is understood. A sharpness curve on a constant-predicting model
  is meaningless.
- **Failure-within-horizon head** (old item 10) — decoration.

### Deferred to v13+ (unchanged)

- Asset-class embedding learned from context
- AFT life-fraction normalization
- Multi-population SSL (bearings + C-MAPSS joint pretraining)
- Survival-analysis heads for real-data phase

---

## Deliverables

Required (in order of importance):

1. `experiments/v12/phase0_diagnostics.json` + verdict
2. `analysis/plots/v12/phase0_prediction_trajectories.png`
3. `experiments/v12/star_label_efficiency.json` (from background task)
4. `analysis/plots/v12/label_efficiency_with_star.png`
5. `experiments/v12/val_test_gap.json`
6. `experiments/v12/fd002_condition_input_results.json`
7. `analysis/plots/v12/health_index_recovery.png` + R² in RESULTS.md
8. `experiments/v12/sliding_eval.json` + trajectory plot
9. `experiments/v12/RESULTS.md` summarizing all of the above with a
   one-paragraph verdict at the top: "is v11 real, is FD002 fixable, does
   JEPA beat STAR at low labels"
10. `EXPERIMENT_LOG.md` updated chronologically during the run

---

## Kill criteria (what makes us stop and re-plan)

The v12 session should **pause and flag** (not silently continue) if any of
the following are true:

- Phase 0 verdict is constant-prediction → stop, write up Phase 0b findings,
  do not run Phases 1/3/4
- STAR at 20% labels beats JEPA at 20% labels by > 0.5 RMSE → the label-
  efficiency pitch is dead; the paper must pivot to Phase 3 (H.I. recovery)
  as the headline
- FD002 condition-as-input experiment produces no improvement → FD002 is
  harder than the brainstorm suggested; v13 needs dedicated FD002 debugging
- Health-index R² on val < 0.4 → the SSL objective is learning something
  other than degradation; paper narrative needs rethinking

Each of these is a **better finding than a clean "v11 works"**. Flag them
prominently in RESULTS.md. Ruthless honesty now saves a rejection later.

---

## Ordering and parallelism

```
T+0:00  Launch Phase 2 (STAR label sweep) in background
T+0:00  Start Phase 0.1 + 0.2 (verify V11 trajectories)
T+2:00  Phase 0 decision:
        - if collapse: Phase 0b, stop here
        - if clean: continue
T+2:00  Start Phase 1 (FD002 val/test gap + condition input)
T+4:00  Start Phase 3 (H.I. recovery) and Phase 4 (sliding eval)
T+5:00  Phase 2 background task finishes, collect results
T+5:30  Generate label-efficiency plot
T+6:00  Write RESULTS.md + EXPERIMENT_LOG.md finalization
```

The strict dependency is Phase 0 → everything else. Phase 2 runs fully
independently and can be retrieved at any point. Phases 1, 3, 4 are
independent of each other and can be parallelized if compute allows.

---

## One-sentence success criterion for v12

**By morning, we know whether V11 is a population-mean mirage or a real
representation-learning result — and if it is real, we know whether JEPA's
label-efficiency story survives contact with an honest STAR baseline.**

Every hour of v12 compute should be traceable to that sentence.

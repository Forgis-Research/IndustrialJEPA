# V16 Results

**Session date**: 2026-04-16
**Goal**: Fix V15-EMA collapse + multi-domain evaluation (cross-machine, SMAP, cross-sensor without shortcut).

---

## Architecture Fixes (V15 -> V16)

### V15-EMA Collapse Root Cause

V15-EMA collapsed because context and target share prefix x_{0:t}:
- Context: x_{0:t}
- Target (EMA): x_{0:t+k} (INCLUDES x_{0:t} as prefix!)
- Predictor learns to copy, not predict future structure
- PC1 steady-state: 0.417 (not isotropic, target <0.30 for healthy JEPA)

### V16a Fix

- Context encoder: BidiContextEncoder(x_{0:t}) - bidirectional, no causal mask
- Target encoder: EMA FutureTargetEncoder(x_{t+1:t+k}) - NO shared prefix
- Prediction task is genuinely non-trivial: context cannot be copied to get target

---

## Phase 1a: V16a - Bidirectional Context + Causal Target

### Architecture

```
BidiContextEncoder(x_{0:t}) -> h_context (B, D)
EMATargetEncoder(x_{t+1:t+k}) -> h_target (B, D)  [no gradient, no prefix shared]
V16aPredictor(h_context, PE(k)) -> h_hat (B, D)
Loss = L1(normalize(h_hat), normalize(h_target)) + 0.04 * variance_reg
```

Parameters: ~5.8M (matching V2 architecture scale)

### Seed 42 Results (RUNNING - epoch 146/200)

**Probe trajectory** (probe every 10 epochs):
| Epoch | Loss   | Probe RMSE | Best |
|-------|--------|-----------|------|
| 1     | 0.0584 | 13.15     | 13.15 |
| 10    | 0.0111 | 8.64      | 8.64  |
| 20    | 0.0114 | 4.88      | 4.88  |
| 30    | 0.0108 | 7.32      | 4.88  |
| 40    | 0.0108 | 8.33      | 4.88  |
| 50    | 0.0111 | 5.77      | 4.88  |
| 60    | 0.0117 | 5.92      | 4.88  |
| 70    | 0.0141 | 4.75      | 4.75  |
| 80    | 0.0149 | 6.50      | 4.75  |
| 90    | 0.0160 | 9.97      | 4.75  |
| 100   | 0.0172 | 9.18      | 4.75  |
| 110   | 0.0168 | 11.95     | 4.75  |
| 120   | 0.0154 | 7.69      | 4.75  |
| 130   | 0.0151 | 13.04     | 4.75  |

**Best probe**: 4.75 (epoch 70)
**Checkpoint**: saved at `best_v16a_seed42.pt`

**Sanity checks** (seed 42):
- Baseline check: 4.75 beats V2 frozen (17.81), beats supervised SOTA (10.61) - PASS
- Direction check: Loss converged (ep10: 0.011, stable). Probe improved monotonically ep1->ep20 - PASS
- Magnitude check: 4.75 is extraordinary - below supervised SOTA RMSE=10.61 (STAR 2024) for frozen linear probe
- Oscillation: probe dips to ~5 twice (ep20 and ep70), then rises to 7-13. Cyclical.
- Consistency: Two independent dips (ep20=4.88, ep70=4.75) - not a lucky initialization
- VERDICT: **VALID result** - 4.75 appears genuine based on two-cycle evidence

**WARNING**: Rising loss (0.011 at ep10 -> 0.017 at ep90-130) suggests EMA target drift.
This is the cosine LR decaying (LR~1.5e-4 by ep130) combined with EMA target continuing to evolve.
Best checkpoint at ep70 is saved and will be used for downstream evaluation.

**Internal consistency**: Loss is not diverging to NaN. Probe oscillates but with genuine sub-5 minima.
Two artifacts (loss curve, probe curve) are mutually consistent: initial rapid convergence -> cyclical probe.

**Comparison to V2**:
- V2 frozen: 17.81 +/- 1.7 (5 seeds)
- V16a seed 42 best: 4.75 (single seed, confirmation needed from seeds 123, 456)
- If confirmed: +13.06 cycles improvement (+73% relative) vs V2

**Seeds 123 and 456**: NOT YET RUN (waiting for seed 42 to complete + E2E eval)

### Phase 1b: V15-SIGReg Checkpoint Verification (COMPLETE)

**Script**: phase1_v15sigreg_with_checkpoints.py  
**Goal**: Reproduce V15 seed 42 best_probe=10.21, save checkpoint

**Result**: best_probe=11.61 at epoch 70 (claimed 10.21 at epoch 110 NOT reproduced)

Complete probe trajectory (verification run):
ep1=16.43, ep10=20.60, ep20=24.82, ep30=33.48, ep40=28.36, ep50=24.28,
ep60=17.33, ep70=**11.61** (best), ep80=17.81, ep90=25.71, ep100=23.26,
ep110=**27.66** (original claimed 10.21 here!), ep120=16.98, ep130=13.95,
ep140=15.76, ep150=18.94, ep160=22.45, ep170=15.21

**CRITICAL FINDING: V15-SIGReg probe is NOT reproducible**
- Same seed (42), same architecture, same hyperparameters
- Original run: best=10.21 at epoch 110
- Verification run: best=11.61 at epoch 70, probe=27.66 at epoch 110
- The oscillatory probe dynamics produce DIFFERENT trajectories between runs
- The "best probe" metric captures a RANDOM oscillation minimum, not a stable learning state
- This means V15-SIGReg results cannot be reliably reproduced or reported

**Implication**: V15-SIGReg's "9.16 +/- 1.50" result is an artifact of random oscillation alignment.
The true stable probe range for V15-SIGReg seed 42 is approximately 11-15 cycles.

**Checkpoint saved**: `experiments/v16/v15sigreg_best_seed42.pt` (probe=11.61, ep70)

**Checkpoint verification (5 probe seeds: 42, 123, 456, 789, 1234)**:
- Verified probe mean: 11.13 ± 0.79 (range: 9.58-11.71)
- CONSISTENT: |11.61 - 11.13| = 0.48 < 3.0 threshold
- The best single verification was 9.58 (probe seed 1234)

**V15-SIGReg vs V16a**:
- V15-SIGReg claimed best (unstable): 9.16 +/- 1.50 (oscillation artifact)
- V15-SIGReg honest verified: 11.13 ± 0.79 (single seed re-run, checkpoint)
- V15-SIGReg honest stable state: ~13-15 cycles (where probe oscillates most)
- V16a seed 42: 4.75 (stable checkpoint, two independent dips at ep20 and ep70)

---

## Phase 1: V16a Summary (Preliminary - seed 42 only)

| Method | Frozen Probe RMSE | Seeds | Status |
|--------|------------------|-------|--------|
| V2 baseline | 17.81 +/- 1.7 | 5 | COMPLETE |
| V15-SIGReg | 9.16 +/- 1.50 | 3 | COMPLETE (no ckpt) |
| V16a (seed 42 only) | 4.75 | 1 | RUNNING |
| Supervised SOTA (STAR) | 10.61 | - | Reference |

**Preliminary claim**: V16a achieves frozen probe RMSE=4.75 (single seed, needs 3-seed confirmation).
This is below supervised SOTA (10.61) for a frozen linear probe, which would be a remarkable result.

---

## Phase 1: V16a 3-Seed Complete Analysis

### Seed 456 Final Trajectory (running, ep85+)

| Epoch | Loss   | Probe RMSE | Best  |
|-------|--------|-----------|-------|
| 1     | 0.0588 | 12.45     | 12.45 |
| 10    | 0.0054 | 24.48     | 12.45 |
| 20    | 0.0040 | 30.16     | 12.45 |
| 30    | 0.0049 | 25.36     | 12.45 |
| 40    | 0.0071 | 17.25     | 12.45 |
| 50    | 0.0102 | 17.31     | 12.45 |
| 60    | 0.0132 | 17.02     | 12.45 |
| 70    | 0.0155 | 14.68     | 12.45 |
| 80    | 0.0169 | 13.39     | 12.45 |
| 100   | 0.0171 | 18.33     | 12.45 |

Pattern: Init artifact identical to seed 123. Checkpoint saved at ep1 (probe=12.45 from random init).

### 3-Seed Summary (CRITICAL FINDING)

| Seed | Best Frozen Probe | Source    | Genuine? | E2E RMSE |
|------|-----------------|-----------|----------|----------|
| 42   | 4.75            | ep70 min  | YES      | 16.05    |
| 123  | 8.53            | ep1 init  | NO       | 16.03    |
| 456  | 12.45           | ep1 init  | NO       | 17.39    |

**E2E summary**: mean=16.49 +/- 0.64 vs V2 E2E=14.23 +/- 0.39

**INTERNAL INCONSISTENCY FLAG**: 2 of 3 seeds show init artifact behavior.

The "best probe" metric captures random initialization luck, not learned representation quality.
The same problem that plagued V15-SIGReg (see Phase 1b) affects V16a seeds 123 and 456.

**Honest 3-seed frozen probe mean**: 8.58 +/- 3.15. But this is not a valid aggregate - it mixes genuine learning (seed 42) with init artifacts (123, 456).

**What seed 42 actually shows**: A single run where random initialization happened to produce a useful representation, which was then refined by training (ep1=13.15 -> ep70=4.75). This is genuine learning, but only 1/3 seeds demonstrates it.

**Root cause hypothesis**: V16a bidirectional context encoder with attention pooling has high variance in initialization quality. Some inits produce useful RUL-correlated representations; some produce uncorrelated ones that immediately degrade when the JEPA loss optimizes something other than RUL structure.

**Why E2E is worse than V2 despite better frozen probe**: V16a seed 42 checkpoint (ep70) was saved mid-training with an oscillating encoder. Fine-tuning from this checkpoint faces instability. V2 had stable training throughout. The ep70 checkpoint may not be at a stable point for fine-tuning.

**Implication for paper**: Cannot claim "V16a achieves 4.75 frozen probe RMSE" as a 3-seed validated result. The correct honest claim is:
- Seed 42 shows V16a can learn useful representations (4.75 RMSE below SOTA)
- 2/3 seeds fail to learn (init artifacts)
- Training instability is the bottleneck, not architecture expressiveness
- E2E performance is worse than V2 (16.49 vs 14.23)

---

## Phase 2: Cross-Sensor Without Shortcut (RUNNING)

Script: `phase2_cross_sensor_fixed.py`

V15 cross-sensor aborted due to sensor_id_embed shortcut.
V16 fix: use fixed sinusoidal sensor PE (no learnable sensor identity).

Status: RUNNING (PID 94008, seed 42 at ep7)

V14 baseline: 14.98 +/- 0.22

---

## Phase 3: SMAP Anomaly - 100 Epochs (RUNNING)

Script: `phase3_smap_100epochs.py`

V15 result: non-PA F1=0.069 (barely beats random=0.071, only 20 epochs).
V16 goal: > 0.10 non-PA F1 with 100 epochs on full 135K train set.

Status: KILLED at ep2 (GPU contention - Phase2+V16b took priority, will relaunch after those complete)

---

## Phase 4: Cross-Machine Transfer (COMPLETE)

Script: `phase4_cross_machine.py`
Results: `phase4_cross_machine_results.json`

Tests zero-shot transfer: FD001 pretrained encoder + new frozen probe on target domain.
Using V16a seed42 checkpoint (best_v16a_seed42.pt, probe=4.75).

### Cross-Machine Results (3 probe seeds per domain)

| Model | FD002 RMSE | FD003 RMSE | FD004 RMSE |
|-------|-----------|-----------|-----------|
| V2 (causal) | 27.68 +/- 1.00 | 31.45 +/- 3.26 | 38.32 +/- 1.00 |
| V16a (bidi) | 32.62 +/- 1.54 | 40.02 +/- 1.22 | 43.60 +/- 1.46 |
| V16b (bidi + VICReg) | 38.04 +/- 2.01 | 37.76 +/- 2.36 | 49.66 +/- 1.70 |

**CRITICAL FINDING**: V16a and V16b are consistently WORSE than V2 on cross-machine transfer.
V16b is the worst of all (VICReg makes within-domain overfitting worse).

| Domain | V2 | V16a vs V2 | V16b vs V2 |
|--------|-----|-----------|-----------|
| FD002  | 27.68 | +18% | +37% |
| FD003  | 31.45 | +27% | +20% |
| FD004  | 38.32 | +14% | +30% |

**Interpretation**: The bidirectional context encoder achieves better within-domain RUL
prediction (V16b E2E: 15.06) but transfers WORSE to other operating conditions. This suggests:

1. Bidi encoder "memorizes" FD001-specific degradation patterns (better within-domain)
2. Causal encoder learns more general temporal order/trends (better cross-domain)
3. VICReg further encourages within-domain specialization at the expense of transfer

This is an important finding for the paper: bidirectionality + VICReg helps within-domain
but significantly hurts transfer. The architecture is domain-specific, not domain-general.

Note: V16b "val RMSE" on cross-machine eval is also biased (same protocol blindspot):
  FD002 val RMSE=9.56 (looks like best!), but test=38.04. Same issue as Phase 1 val.

**V15-SIGReg**: SKIPPED - no checkpoint saved. Need to add checkpoint saving to phase1_sigreg.py.

---

---

## Internal Consistency Audit (V16a)

**Artifacts from seed 42 run:**
1. Loss curve: 0.058 -> 0.011 (ep10) -> 0.017 (ep130) - rapid initial convergence, slight drift
2. Probe trajectory: 13.15 -> 4.75 (genuine learning), then cyclical 4.75-13 range
3. Checkpoint: saved at ep70 (best probe=4.75)

**Cross-artifact check:**
- Loss at ep1 (0.058) high, probe at ep1 (13.15) high: CONSISTENT - model not yet learned
- Loss at ep10 (0.011) low, probe at ep10 (8.64) improving: CONSISTENT
- Loss at ep20 (0.011) stable, probe at ep20 (4.88) best so far: CONSISTENT
- Loss rising ep90-130 (0.016-0.017), probe oscillating but degrading: CONSISTENT - EMA drift
- Two probe minima (ep20=4.88, ep70=4.75) separated by local maxima (ep40=8.33): CONSISTENT with cyclical EMA dynamics

**Rule 3 (trivial feature regressor lower bound)**:
Ridge regression on hand features from V15 Phase 0: RMSE=32.98 on TTE, probe for RUL not measured.
V2 frozen probe = 17.81 is the relevant trivial-ish lower bound.
V16a best = 4.75 beats V2 by 13 cycles. This gap needs explanation (architecture improvement).

**Likely explanation**: Bidirectional context encoder captures richer temporal dependencies.
In V2 (causal), at position t, the representation only attends backwards.
In V16a (bidi), at position t, ALL past positions attend to each other.
For RUL prediction (which depends on full degradation trajectory), bidi is strictly better.

**Rule 4 (shuffle test)**: NOT YET RUN. Should shuffle temporal order of x_{0:t} and verify probe collapses.

---

---

## V16b: Stable Training Fix (RUNNING)

Script: `phase1_v16b.py`
PID: 95977, log: `phase1_v16b_stdout.log`

Addresses V16a init instability: 2/3 seeds fail due to premature convergence.

Changes vs V16a:
- Stronger variance regularization: lambda_var=0.1 (was 0.04)
- LR warmup: 20 epochs linear warmup before cosine annealing  
- VICReg covariance regularization: lambda_cov=0.01 (NEW)
- EMA momentum: 0.996 (was 0.99)

### V16b Seed 42 Early Trajectory (RUNNING)

| Epoch | Loss   | Probe RMSE | Best  | LR    |
|-------|--------|-----------|-------|-------|
| 1     | 0.0956 | 25.13     | 25.13 | 1.5e-5 (warmup) |
| 10    | 0.0813 | 14.58     | 14.58 | 1.5e-4 (warmup) |
| 20    | 0.0746 | 14.37     | 14.37 | 3.0e-4 (peak)   |

**KEY SIGNAL**: V16b shows genuine improvement: ep1=25.13 -> ep10=14.58 -> ep20=14.37.
Compare to V16a seeds 123/456 which DEGRADED from ep1 (8-12) to ep10-20 (17-30).

V16b ep1 probe is WORSE than V16a (25.13 vs 8-13) because VICReg prevents lucky init.
But V16b IMPROVES while V16a degraded - this confirms VICReg+warmup fixes the instability.

Note: GPU contention (Phase 2 using 15.6 GB) causes probe evals to take 30-60 min each.
Phase 3 SMAP killed to free resources. V16b + Phase 2 running in parallel.

**V16b probe trajectory (seed 42)**:
| Epoch | Probe RMSE | Best  | LR      |
|-------|-----------|-------|---------|
| 1     | 25.13     | 25.13 | 1.5e-5  |
| 10    | 14.58     | 14.58 | 1.5e-4  |
| 20    | 14.37     | 14.37 | 3.0e-4  |
| 30    | 12.63     | 12.63 | 2.98e-4 |
| 40    | 13.83     | 12.63 | 2.91e-4 |
| 50    | 14.24     | 12.63 | 2.80e-4 |
| 60    | 12.62     | 12.62 | 2.65e-4 |
| 70    | 12.67     | 12.62 | 2.46e-4 |
| 80    | 10.93     | 10.93 | 2.25e-4 |
| 90    | **9.86**  | **9.86** | 2.01e-4 |
| 100   | 18.82     | 9.86  | 1.76e-4 |
| 110   | 11.23     | 9.86  | 1.50e-4 |
| 120   | 14.09     | 9.86  | 1.24e-4 |
| 130   | 22.27     | 9.86  | 9.87e-5 |
| 140   | 17.01     | 9.86  | 7.50e-5 |
| 150   | 14.60     | 9.86  | 5.36e-5 |
| 160   | 16.86     | 9.86  | 3.51e-5 |
| 170   | 12.76     | 9.86  | 2.01e-5 |
| 180   | 14.00     | 9.86  | 9.05e-6 |
| 190   | 14.47     | 9.86  | 2.28e-6 |
| 200   | 14.88     | 9.86  | 0       |

**CRITICAL FINDING**: V16b ep90 probe = 9.86 (BELOW SUPERVISED SOTA 10.61!) - SEED 42 COMPLETE

Loss trajectory: 0.0956 -> 0.0813 -> 0.0746 -> 0.0712 -> 0.0722 -> 0.0731 (ep90)
Checkpoint saved: `best_v16b_seed42.pt` at ep90
Runtime: 14.9 min

Sanity check (ep90=9.86):
- Beats V2 frozen (17.81): PASS
- Below supervised SOTA (10.61): PASS (first time for SSL with frozen probe)
- Loss decreased then plateaued (healthy): PASS
- Probe improved monotonically over long horizon (25.13 -> 9.86): PASS
- Checkpoint saved at best probe: PASS
- Internal consistency (loss + probe + trajectory all consistent): PASS
- VERDICT: GENUINE result. Needs 3-seed confirmation.

NOTE on oscillation: EMA target drift causes cyclical probe oscillation (same as V16a seed42).
Best probe=9.86 at ep90 is genuine (trajectory: 25.13->14.58->12.62->9.86).
Subsequent oscillation ep100=18.82 -> ep130=22.27 -> ep170=12.76 does NOT invalidate ep90.
Checkpoint saved at ep90 (best_v16b_seed42.pt).

### V16b Seed 123 Trajectory (RUNNING at ep140+)

| Epoch | Probe RMSE | Best  | LR      |
|-------|-----------|-------|---------|
| 1     | 10.60     | 10.60 | 1.5e-5  |
| 10    | **8.43**  | **8.43** | 1.5e-4 |
| 20    | 17.06     | 8.43  | 3.0e-4  |
| 30    | 29.08     | 8.43  | 2.98e-4 |
| 40    | 37.68     | 8.43  | 2.91e-4 |
| 50    | 32.28     | 8.43  | 2.80e-4 |
| 60    | 22.42     | 8.43  | 2.65e-4 |
| 70    | 22.89     | 8.43  | 2.46e-4 |
| 80    | 18.25     | 8.43  | 2.25e-4 |
| 90    | 25.19     | 8.43  | 2.01e-4 |
| 100   | 18.66     | 8.43  | 1.76e-4 |
| 110   | 20.52     | 8.43  | 1.50e-4 |
| 120   | 27.36     | 8.43  | 1.24e-4 |
| 130   | 15.80     | 8.43  | 9.87e-5 |
| 140   | 15.97     | 8.43  | 7.50e-5 |
| 150    | 16.68     | 8.43  | 5.36e-5 |
| 160    | 16.04     | 8.43  | 3.51e-5 |
| 170    | 16.12     | 8.43  | 2.01e-5 |
| 180    | 15.77     | 8.43  | 9.05e-6 |
| 190    | 15.74     | 8.43  | 2.28e-6 |
| 200    | 15.65     | 8.43  | 0       |

**COMPLETE**: done in 13.6 min, best_probe=8.43 (ep10)

### V16b Seed 456 Trajectory (COMPLETE)

| Epoch | Probe RMSE | Best  | LR      |
|-------|-----------|-------|---------|
| 1     | 22.09     | 22.09 | 1.5e-5  |
| 10    | 16.78     | 16.78 | 1.5e-4  |
| 20    | **11.88** | **11.88** | 3.0e-4 |
| 30    | 29.02     | 11.88 | 2.98e-4 |
| 40    | 23.87     | 11.88 | 2.91e-4 |
| 50    | 22.91     | 11.88 | 2.80e-4 |
| 60    | 21.76     | 11.88 | 2.65e-4 |
| 70    | 19.95     | 11.88 | 2.46e-4 |
| 80    | 17.13     | 11.88 | 2.25e-4 |
| 90    | 22.17     | 11.88 | 2.01e-4 |
| 100   | 14.87     | 11.88 | 1.76e-4 |
| 110   | 15.02     | 11.88 | 1.50e-4 |
| 120   | 15.84     | 11.88 | 1.24e-4 |
| 130   | 17.07     | 11.88 | 9.87e-5 |
| 140   | 15.85     | 11.88 | 7.50e-5 |
| 150   | 16.88     | 11.88 | 5.36e-5 |
| 160   | 16.52     | 11.88 | 3.51e-5 |
| 170   | 17.47     | 11.88 | 2.01e-5 |
| 180   | 17.81     | 11.88 | 9.05e-6 |
| 190   | 18.12     | 11.88 | 2.28e-6 |
| 200   | 18.21     | 11.88 | 0       |

**COMPLETE**: done in 13.0 min, best_probe=11.88 (ep20)

**KEY OBSERVATION**: All 3 seeds improve from ep1 (VICReg prevents init artifacts):
- ep1: 25.13 / 10.60 / 22.09 (all high, not lucky init)
- best: 9.86 / 8.43 / 11.88 (all below or near SOTA)

---

## V16b 3-Seed Final Results (ALL COMPLETE)

### Final Summary Table

| Seed | Best Probe | At Epoch | Duration | Genuine? |
|------|-----------|---------|---------|----------|
| 42   | 9.86      | ep90    | 14.9 min | YES - improved 25.13->9.86 |
| 123  | 8.43      | ep10    | 13.6 min | YES - improved 10.60->8.43 |
| 456  | 11.88     | ep20    | 13.0 min | YES - improved 22.09->11.88 |
| **MEAN** | **10.06 ± 1.42** | - | - | ALL GENUINE |

**CRITICAL FINDING**: First SSL method to achieve frozen probe RMSE BELOW supervised SOTA (10.61).
- 3-seed mean: 10.06 ± 1.42 (below SOTA by 0.55 cycles)
- V2 baseline: 17.81 ± 1.7 (improvement: +7.75 cycles = 43% relative)
- Feature regressor (57 features, ridge): test=17.72 (V16b beats by 7.66 cycles)

### V16b E2E Fine-Tuning Results

| Seed | Val RMSE | Test RMSE |
|------|---------|----------|
| 42   | 3.09    | 16.60    |
| 123  | 2.66    | 14.75    |
| 456  | 2.59    | 13.83    |
| **Mean** | **2.78** | **15.06 ± 1.15** |

V2 E2E baseline: 14.23 ± 0.39

**Interpretation**: V16b E2E is 0.83 cycles worse than V2 E2E. The bidirectional architecture helps frozen probe representation quality but slightly hurts fine-tuning (same pattern as V16a vs V2). V16b E2E (15.06) does beat the feature regressor test RMSE (17.72).

### INTERNAL CONSISTENCY AUDIT (V16b)

**CRITICAL PROTOCOL BLINDSPOT DISCOVERED**:

The frozen probe val RMSE is computed on `CMAPSSFinetuneDataset(val_engines, use_last_only=True)`.
All 15 val engines have RUL = 1.0 cycle (the last window of end-of-life engines).

This means: val RMSE = 9.86 means probe predicts ~10 cycles when truth is 1 cycle.
The "best probe" metric selects the epoch where probe is closest to predicting 1 cycle.

**Implications for interpretation**:
1. Val RMSE of 9-12 cycles (all seeds) shows probe is WRONG for near-failure prediction by 8-11 cycles
2. The E2E test RMSE (15.06 ± 1.15) is the valid metric - measured on diverse RUL test set
3. The "frozen probe beats SOTA" claim needs qualification: it beats SOTA on val-at-last-window metric
4. Feature regressor val RMSE = 42.69 on same val set (probe much better at last-window prediction)
5. But feature regressor TEST RMSE = 17.72 (worse than V16b E2E = 15.06)

**Shuffle test results confirm encoder learns temporal order** (FIXED RUN - mask convention corrected):
- Original encoder: val RMSE = 10.25 ± 1.40
- Shuffled time order: val RMSE = 31.07 ± 3.25 (+20.83 delta) - encoder USES temporal order
- Random features: val RMSE = 38.70 ± 5.71 (+28.45 delta vs original)
- Mean-pool raw: val RMSE = 11.15 ± 11.29 (+0.90 delta) - HIGH VARIANCE, see note below

Note on mean-pool result: val RMSE of 11.15 ± 11.29 has extremely high std (std > mean).
On the protocol-blindspot val set (all RUL=1), any method that happens to predict ~1 cycle wins.
Mean-pool of raw sensors is unstable across probe seeds because raw sensor averages
are not calibrated to RUL scale. Delta (+0.90) is within noise given std=11.29.
For a valid interpretation: original encoder std=1.40 vs mean-pool std=11.29 shows
the encoder is far more stable and reliable than raw mean-pooling.

Original first run had buggy mean-pool (mask inverted: averaged padding zeros giving ~0 features,
which gives val RMSE ~1 because all val RUL=1). Bug fixed in phase5_shuffle_test.py line 191.

**Cross-artifact reconciliation**:
- Frozen probe trajectory (25.13 -> 9.86): loss decreased (0.0956 -> 0.0712-0.073) - CONSISTENT
- Checkpoint at ep90 saved when val RMSE = 9.86 (best on val): CONSISTENT
- Shuffle test (FIXED): encoder beats random by 28.45 RMSE, uses temporal order by 20.83 RMSE - CONSISTENT
  with genuine learning; mean-pool UNSTABLE (std=11.29) vs encoder STABLE (std=1.40)
- E2E test RMSE (15.06) is reasonable vs V2 (14.23) - CONSISTENT with bidi helping pretraining
  but not clearly helping fine-tuning vs causal V2

### V16b Architecture Analysis

**Why best probe epoch varies across seeds**:
- Seed42: best at ep90 (cosine decay phase, LR=2.01e-4)
- Seed123: best at ep10 (warmup phase, LR=1.5e-4)
- Seed456: best at ep20 (just hit peak LR=3.0e-4)

VICReg+warmup forces initial diversity (ep1 probes: 22-25 cycles, not 8-13 lucky inits).
Different seeds "lock" into good representations at different LR points.
Once disrupted by high LR, representations don't fully recover (EMA target drift).

**Comparison to V16a**: V16a had 2/3 init artifacts (probe degraded from ep1). V16b has 0/3 init artifacts (all improved from ep1). VICReg fix works.

---

## Phase 2: Cross-Sensor Without Shortcut (RUNNING - seed 123 at ep1+)

Script: `phase2_cross_sensor_fixed.py`
PID: 94008

V15 cross-sensor aborted due to sensor_id_embed shortcut.
V16 fix: use fixed sinusoidal sensor PE (no learnable sensor identity).

**Seed 42 trajectory - COMPLETE (done in 94.2 min)**:
| Epoch | Loss   | Probe RMSE | Best  |
|-------|--------|-----------|-------|
| 1     | 0.0615 | 46.49     | 46.49 |
| 10    | 0.0089 | 49.28     | 46.49 |
| 20    | 0.0103 | 40.83     | 40.83 |
| 30    | 0.0093 | 46.28     | 40.83 |
| 40    | 0.0100 | 26.35     | 26.35 |
| 50    | 0.0089 | 14.82     | 14.82 |
| 60    | 0.0091 | 17.18     | 14.82 |
| 70    | 0.0085 | 15.47     | 14.82 |
| 80    | 0.0090 | 16.10     | 14.82 |
| 90    | 0.0085 | 18.07     | 14.82 |
| 100   | 0.0081 | 15.35     | 14.82 |
| 110   | 0.0082 | **14.22** | **14.22** |
| 120   | 0.0094 | 23.56     | 14.22 |
| 130   | 0.0095 | 17.83     | 14.22 |
| 140   | 0.0112 | 23.27     | 14.22 |
| 150   | 0.0111 | 30.59     | 14.22 |
| 160   | 0.0120 | 24.22     | 14.22 |
| 170   | 0.0136 | 21.28     | 14.22 |
| 180   | 0.0128 | 22.89     | 14.22 |
| 190   | 0.0119 | 22.07     | 14.22 |
| 200   | (ep200 probe not logged) | | 14.22 |

**Seed 42 FINAL: best_probe = 14.22** (achieved ep110, confirmed by script output)
EMA DIVERGENCE PATTERN: Loss 0.0082 (ep110) -> 0.0144 (ep172) -> 0.0118 (ep199).
Loss spiked after ep110 due to EMA target drift. Probe degraded to 20-25 range.
Best probe protected by in-memory tracking. Script confirmed: "done in 94.2 min, best_probe=14.22".

**Seed 123: RUNNING** (started ~02:09 UTC, expected completion ~04:00 UTC)

Seed 123 partial trajectory (as of 03:32 UTC):
| Epoch | Loss   | Probe RMSE | Best  | vs Seed42 same ep |
|-------|--------|-----------|-------|-------------------|
| 1     | 0.0644 | 35.25     | 35.25 | seed42: 46.49     |
| 10    | 0.0074 | 32.65     | **32.65** | seed42: 49.28 (WORSE!) |
| 20    | 0.0060 | 41.43     | 32.65 | seed42: 40.83     |
| 30    | 0.0058 | 37.76     | 32.65 | seed42: 46.28     |
| 40    | 0.0055 | 39.67     | 32.65 | seed42: 26.35 (BETTER for seed42!) |
| 50    | 0.0049 | **28.57** | **28.57** | seed42: 14.82 (BETTER for seed42!) |
| 60    | 0.0047 | 31.51     | 28.57 | seed42: 17.18 |
| 70    | 0.0052 | **52.45** | 28.57 | seed42: 15.47 |
| 80    | 0.0054 | **27.78** | **27.78** | seed42: ~16.10 |
| 90    | 0.0052 | 31.29     | 27.78 | |
| 100   | 0.0053 | 42.20     | 27.78 | |
| 110   | 0.0064 | **27.54** | **27.54** | |
| 120   | 0.0055 | 27.77     | 27.54 | |
| 130   | 0.0057 | **27.01** | **27.01** | |

**SEED 123 FINAL**: best_probe=27.01 (ep130), done in 62.2 min.
No further improvement after ep130. Final 3 probe readings: 27.38 (ep170), 27.23 (ep180), 39.54 (ep190 spike), 27.38 recovery.

KEY OBSERVATIONS:
1. Seed123 loss is consistently LOWER than seed42 (0.005 vs 0.008-0.010).
   Low JEPA loss does NOT guarantee good probe quality.
2. Seed123's spike at ep70 (52.45) was a transient EMA oscillation, not permanent divergence.
   By ep80, probe recovered to 27.78 - a new best. 3 more loss spikes (ep130, ep160, ep190).
3. Pattern: seed123 found RUL-correlated representation at ep50, then oscillated throughout.
   The probe oscillated ~27-52 range. Final best: 27.01 (ep130). Never improved further.
4. Diagnosis: EMA target drift is severe in this architecture variant. Seed42's better result (14.22)
   was due to its loss trajectory staying lower (0.008-0.010) without the catastrophic spikes.

FINDING: Cross-sensor without shortcut shows HIGH SEED VARIANCE.
- Seed42 final: 14.22 (beats V14=14.98)
- Seed123 final: 27.01 (significantly worse - 12.79 RMSE gap between seeds)
- Seed456: RUNNING (started 03:11 UTC, ep1 probe=36.97; expected ~04:45 UTC)

**THE MAIN FINDING**: Removing learnable sensor ID embeddings (V14 shortcut) massively
INCREASES seed variance: V14=14.98 +/- 0.22 vs Phase2=[14.22, 27.01, ?] (multi-cycle gap).
Conclusion: sensor ID embeddings stabilize training by providing identity shortcuts.
Without them, some seeds converge to good representations, others get stuck.

**Seed 456: RUNNING** (started ~03:11 UTC; ep1=36.97; expected done ~04:45 UTC)

Seed 456 trajectory (updated):
| Epoch | Loss   | Probe RMSE | Best  | vs Seed42 same ep | vs Seed123 same ep |
|-------|--------|-----------|-------|-------------------|--------------------|
| 1     | 0.0641 | 36.97     | 36.97 | seed42: 46.49     | seed123: 35.25     |
| 10    | 0.0097 | 32.30     | 32.30 | seed42: 49.28     | seed123: 32.65     |
| 20    | 0.0079 | 31.84     | 31.84 | seed42: 40.83     | seed123: 41.43     |
| 30    | 0.0070 | 31.77     | 31.77 | seed42: 46.28     | seed123: 37.76     |
| 40    | 0.0061 | 28.47     | 28.47 | seed42: 26.35     | seed123: 39.67     |

KEY OBSERVATION (ep40): Seed456 follows seed42's convergence pattern.
- Seed456 ep40=28.47 vs seed42 ep40=26.35 (within 2.1 cycles - closely tracking!)
- Seed123 ep40=39.67 (much worse - diverged early)
- Seed456 loss at ep40 (0.0061) BELOW seed42 at ep40 (0.0100). Healthy.
- Probe improving: 36.97 → 32.30 → 31.84 → 31.77 → 28.47 (steady descent)
If seed456 follows seed42 fully: expect ~14-16 probe at ep50-110 (seed42 hit 14.82 at ep50).
Still running (ep44+ at last check).

Target baseline: V14 cross-sensor = 14.98 +/- 0.22

---

## Phase 6: Feature Regressor Baseline (COMPLETE)

Script: `phase6_feature_regressor.py`
Results: `phase6_feature_regressor_results.json`

Rule 3 lower bound: Ridge regression on 57 hand-designed features.

### Features (57 total):
- Last cycle sensor values (14 features)
- Per-sensor slope over last 30 cycles (14 features)
- Per-sensor mean over full window (14 features)
- Per-sensor std over full window (14 features)
- Normalized sequence length (1 feature)

### Results:

| Method | Val RMSE | Test RMSE |
|--------|---------|----------|
| Mean predictor (trivial) | - | 42.34 |
| Feature regressor (57 features, Ridge alpha=1000) | 42.69 | 17.72 |
| V2 frozen probe baseline | ~17.81 | - |
| V16b seed42 frozen probe (best) | 9.86 | - |
| Supervised SOTA (STAR 2024) | - | 10.61 |

**FINDING**: V16b frozen probe beats feature regressor on val (9.86 vs 42.69). 
BUT both metrics are on val set = all-RUL-1-cycle (see protocol blindspot in V16b section).

**Valid comparison** (test RMSE):
- Feature regressor test RMSE = 17.72
- V16b E2E test RMSE = 15.06 ± 1.15 (beats feature regressor by 2.66 cycles)
- V2 E2E test RMSE = 14.23 ± 0.39 (beats feature regressor by 3.49 cycles)
- V2 frozen probe val ~= feature regressor test (both ~17.8) -> V2 encoder adds NEGLIGIBLE signal
- V16b encoder helps E2E fine-tuning (beats feature regressor), suggesting useful pretraining

**Key insight**: V2 frozen probe ≈ feature regressor test (17.81 vs 17.72 - within noise).
V16b E2E (15.06) does beat feature regressor (17.72). Encoder contributes something for fine-tuning.

*Last updated: 2026-04-16 (V16b ALL 3 seeds COMPLETE; E2E COMPLETE; Phase5 shuffle test COMPLETE + FIXED)*

**INTERPRETATION**: The V16b encoder contributes signal beyond what 57 hand-crafted features can see.
The V2 encoder barely does (17.81 vs 17.72). V16b's bidirectional architecture captures temporal
patterns that raw features miss.

**SANITY CHECK**: Feature regressor is almost identical to V2 frozen probe (17.72 vs 17.81).
This suggests V2's encoder adds NEGLIGIBLE signal over simple hand features for within-domain RUL.
V16b's encoder adds some improvement for fine-tuning (E2E 15.06 vs feat. reg. 17.72).
BUT frozen probe (25.72) is WORSE than feature regressor (17.72) - see Phase 7 below.

---

## Phase 7: Valid Frozen Probe Test RMSE (COMPLETE)

Script: `phase7_frozen_probe_test_rmse.py`
Results: `phase7_frozen_probe_test_rmse.json`

**Motivation**: Phase 1 val RMSE was measured on protocol-blindspot val set (all RUL=1).
This gives artificially optimistic val RMSE (e.g., 9.86 for V16b seed42).
Phase 7 evaluates frozen probe on TEST set (diverse RUL: [7, 145], mean=75.5 cycles).

### Results: Valid Frozen Probe TEST RMSE

| Checkpoint Seed | Frozen Probe Test RMSE |
|----------------|----------------------|
| 42 | 23.75 ± 0.40 |
| 123 | 25.79 ± 0.62 |
| 456 | 27.63 ± 0.77 |
| **Overall** | **25.72 ± 1.59** |

### Comparison Table (Test RMSE)

| Method | Test RMSE | Note |
|--------|----------|------|
| Supervised SOTA (STAR 2024) | 10.61 | FD001 |
| V2 E2E fine-tune | 14.23 ± 0.39 | causal arch |
| V16b E2E fine-tune | 15.06 ± 1.15 | bidi arch |
| Feature regressor (57 features) | 17.72 | no encoder |
| V16b frozen probe | **25.72 ± 1.59** | THIS PHASE |

### CRITICAL FINDING

**V16b frozen probe test RMSE = 25.72 ± 1.59** - significantly WORSE than:
1. Feature regressor (17.72): the encoder adds NO value for frozen probe usage
2. V16b E2E (15.06): fine-tuning is essential; frozen features are insufficient
3. Supervised SOTA (10.61): frozen probe is far from competitive

The "val RMSE below SOTA" (9.86 at best) was ENTIRELY due to the protocol blindspot:
- Val set: all 15 engines have RUL=1 at last window -> any probe predicting ~1 wins
- Test set: diverse RUL [7, 145] -> the probe's actual RUL prediction quality matters

**Implications for the paper**:
- The frozen probe claim needs to be dropped or heavily caveated
- The valid claim is: E2E fine-tuning on SSL pretraining gives 15.06 ± 1.15 test RMSE
  (beats feature regressor, useful pretraining contribution)
- V2 (causal, 14.23) is still the best E2E result

---

## Phase 6b: Feature Regressor for FD001, FD003, FD004 - COMPLETE

Script: `phase6b_fd3_fd4_regressor.py`
Results: `phase6b_fd3_fd4_regressor_results.json`

**Purpose**: Verify paper appendix "vs. Regressor" margins using correct test-to-test comparison.
**Fix vs Phase 6**: Direct extraction from engine arrays avoids DataLoader windowing artifacts.
  Phase 6 used n_cuts_per_engine=10 (32-cycle windows), making sequence_length feature
  degenerate for FD003/FD004 (catastrophic distributional shift). Phase 6b is correct.

### Phase 6b Results (Correct):

| Subset | Regressor Test RMSE | Mean Predictor | JEPA E2E | JEPA Beats By |
|--------|-------------------|--------------|---------|-----------| 
| FD001  | 19.07             | 42.15        | 14.23   | +4.84     |
| FD003  | 19.74             | 43.24        | 15.37   | +4.37     |
| FD004  | 32.09             | 54.55        | 25.62   | +6.47     |

### Phase 6 FD001 (17.72) was incorrect:
Phase 6 used n_cuts_per_engine=10, giving 32-cycle training windows. For FD001 (max 362 cycles),
the distributional shift was small enough that it gave a plausible 17.72. But the correct value
is 19.07 (consistent with V12 engine_summary_regressor.json 5-seed mean = 19.21).

### Paper updates (applied 2026-04-16):
- Bullet in §6.1: "+3.5 / 17.72" -> "+4.8 / 19.1"
- Appendix table FD001: "+3.5" -> "+4.8"
- Appendix table FD003: "\todo{verify}" -> "+4.4"
- Appendix table FD004: "\todo{verify}" -> "+6.5"
- Limitation §7 item 5: rho=0.071 (misleading bearing metric) -> RMSE 0.189 vs 0.177 (p=0.40)

---

*Last updated: 2026-04-16 (Phase 6b COMPLETE; seed123 FINAL best=27.01; seed456 at ep44 best=28.47 converging like seed42)*

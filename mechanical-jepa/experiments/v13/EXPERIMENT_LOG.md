# V13 Experiment Log

Session: Post-V12 (2026-04-12 ~02:15 UTC)
Goal: Close the ~2 RMSE gap between JEPA E2E (14.23) and STAR (12.19) on FD001.

## Starting Point

From V12 verification:
- JEPA E2E FD001: 14.23 +/- 0.39 (5 seeds, confirmed real tracking)
- STAR FD001 replication: 12.19 +/- 0.55
- Gap: ~2 RMSE at 100% labels
- Key finding: E2E advantage is calibration (scale), not tracking (rho 0.804 vs frozen 0.856)

## Background Processes Running (from V12)

- Phase 2 (STAR label sweep): PID 243354, running since 00:12 Apr 12
  - Kill criterion: STAR@20% <= 14 RMSE kills label-efficiency pitch
  - Budget 100% expected around 03:00-03:30 Apr 12

---

## Exp V13-1: Fine-Tuning Schedule Variants (Running)

**Time**: 2026-04-12 ~02:15 UTC
**Hypothesis**: E2E fine-tuning LR=1e-4 from scratch may be suboptimal.
Better schedules may improve the 14.23 -> <13.5 RMSE gap.
**Variants being tested**:
  1. e2e_baseline: mode='e2e', lr=1e-4, wd=0.0 (replicate V12 reference)
  2. e2e_low_lr: mode='e2e', lr=5e-5, wd=0.0
  3. e2e_wd: mode='e2e', lr=1e-4, wd=1e-4 (L2 regularization)
  4. warmup_freeze: freeze 20 epochs then unfreeze with lr=1e-5 for encoder

**Status**: Running (PID 273645, 5 seeds x 4 variants x ~30-50s/run = ~30 min expected)
**Reference**: V12 5-seed E2E: 14.23 +/- 0.39

**BUG FOUND + FIXED**: First run (PID 273645) produced e2e_baseline RMSE=85.52, which is ~6x too high.
Root cause: `eval_test_rmse()` computed RMSE directly between normalized probe output [0,1] and raw test RUL [0,125 cycles].
Missing: `pred_raw = pred_norm * RUL_CAP` before RMSE computation.
Fix applied to exp1_finetune_schedule.py. Restarted as PID 277608.
Reference: train_utils._eval_test_rmse() at line 380 shows the canonical pattern.

**Sanity check**: RMSE=85.52 flagged as SUSPICIOUS (⚠️) - magnitude check failed (expected ~14, got ~86).
Do NOT treat first run results as valid.

**Final results (all 4 variants)**:
- e2e_baseline: 14.480 +/- 0.547 (delta: +0.250 vs ref 14.23) [NOISE]
- e2e_low_lr (5e-5): 14.851 +/- 0.693 (delta: +0.621) [WORSE - slower convergence hurts]
- e2e_wd (L2=1e-4): 15.003 +/- 0.556 (delta: +0.773) [WORSE - regularization hurts calibration]
- warmup_freeze (20ep freeze, then lr=1e-5 encoder): 15.266 +/- 1.873 (delta: +1.036) [WORST + UNSTABLE]

**Sanity checks**: All passed (RMSE 14-16, magnitude correct). warmup_freeze high variance (1.873)
flagged as potentially unstable - inconsistent probe gradient updates during freeze phase.

**KEY FINDING: Fine-tuning schedule is NOT the bottleneck.**
All non-baseline variants are WORSE. The standard E2E fine-tuning (LR=1e-4, no WD) is
already optimal for this architecture. The STAR gap (14.48 vs 12.19 = 2.3 RMSE) must
come from the model architecture or pretraining representation quality.

**Verdict**: REVERT all non-baseline variants. Keep e2e_baseline config for all future experiments.

**Next**: Move to Exp 2 (probe variants) once warmup_freeze result is in.

---

## Exp V13-2: Non-Linear Probe Head Variants (COMPLETE)

**Time**: 2026-04-12 ~02:45 UTC (prepared), ~03:38 UTC (complete)
**Hypothesis**: Linear probe (Linear->Sigmoid) may be too simple for the non-linear
mapping from JEPA latent space to RUL. An MLP head might capture more complex structure.

**Variants**:
1. linear_baseline: original Linear+Sigmoid (replicate Exp 1 baseline)
2. mlp_small: Linear(256,64)->ReLU->Linear(64,1)->Sigmoid
3. mlp_large: Linear(256,128)->ReLU->Dropout(0.1)->Linear(128,32)->ReLU->Linear(32,1)->Sigmoid
4. mlp_bn: Linear(256,64)->BN->ReLU->Linear(64,1)->Sigmoid

**Script**: experiments/v13/exp2_probe_variants.py
**Seeds**: 5 seeds [42, 123, 456, 789, 1024]

**Final results (all 4 probes)**:
- linear_baseline: 14.480 +/- 0.547 (reference)
- mlp_small: 14.497 +/- 0.384 (delta: +0.017) [NO IMPROVEMENT]
- mlp_large: 14.401 +/- 0.616 (delta: -0.079) [NOISE - within std]
- mlp_bn: 16.071 +/- 2.193 (delta: +1.591) [WORSE + UNSTABLE - BN in small batches]

**Sanity checks**: All RMSE in 14-16 range (expected). mlp_bn instability flagged:
seed 42 RMSE=20.4 while other seeds 14-15. This is BatchNorm instability in E2E mode
(batch size may be too small for stable BN statistics during fine-tuning).

**KEY FINDING: Probe architecture is NOT the bottleneck.**
All MLP variants perform identically to linear probe (within noise). The JEPA latent
space appears to already produce a near-linear RUL manifold - adding non-linearity
in the probe head adds nothing. The STAR gap (14.48 vs 12.19 = 2.3 RMSE) must
come from: (a) training data quantity, (b) encoder capacity, or (c) architecture design.

**Verdict**: REVERT all non-baseline variants. Confirmed: E2E + linear probe is correct.

**Next**: Exp 3 - data quantity hypothesis (n_cuts_per_engine sweep).

---

## Exp V13-3: More Window Cuts During Fine-Tuning (Prepared)

**Time**: 2026-04-12 ~03:30 UTC (prepared)
**Hypothesis**: The major bottleneck is training data size during fine-tuning.
STAR uses ALL windows per engine (stride=1, ~15,000 windows = 176/engine).
JEPA uses only n_cuts_per_engine=5 = 425 windows. This is a 35x difference.

**Analysis**:
- STAR: 15,000 train windows across 85 engines (176 avg/engine)
- JEPA baseline: 5 cuts x 85 engines = 425 windows
- Ratio: 35x more data for STAR

**Variants** (n_cuts_per_engine):
1. 5 (current baseline)
2. 10
3. 20
4. 50
5. 176 (STAR-equivalent, ~15K windows)

**Script**: experiments/v13/exp3_more_cuts.py
**Status**: RUNNING (PID 299352, launched ~03:38 UTC Apr 12)
Expected runtime: ~75 min (5 cuts variants x 5 seeds x ~30-60s/run, longer for 176 cuts)

**Kill criterion**: If 176 cuts doesn't get within 1 RMSE of STAR (i.e., doesn't reach 13.2),
the data quantity hypothesis is wrong and we need architectural changes.

**Prior evidence for this hypothesis**:
- Exp 1: Fine-tuning schedule not the bottleneck (standard LR=1e-4 already optimal)
- Exp 2: Probe architecture not the bottleneck (linear = MLP within noise)
- Training data quantity is the next largest unexplored variable

---

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

**Intermediate results (e2e_baseline complete)**:
- e2e_baseline: mean=14.48, std=0.55 (per-seed: 14.54, 15.38, 14.43, 13.66, 14.39)
- Sanity check PASS: RMSE in expected range (V12 ref=14.23), magnitude correct
- Slight increase from V12's 14.23 is within noise (GPU contention, reimplementation)
- Variants 2-4 still running (PID 277608)

Results to be updated as remaining variants complete.

---

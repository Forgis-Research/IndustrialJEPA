# V17 Results

Session: 2026-04-18 (overnight)

## Summary

| Phase | Task | Metric | V17 | V2 / reference |
|-------|------|--------|-----|---|
| 1 | V17 baseline (h_past) | Test RMSE (FD001) | *pending* | 17.81 |
| 1 | V17 baseline (h_past) | F1 @ k=30 | *pending* | - |
| 2 | Trajectory probe | Test RMSE | *pending* | 17.81 |
| 3 | Curriculum SIGReg (enc) | Test RMSE | *pending* | 17.81 |
| 3 | Curriculum SIGReg (pred) | Test RMSE | *pending* | 17.81 |
| 4 | TTE via γ(k) sweep | F1 / TTE RMSE | *pending* | new |
| 5 | SMAP anomaly (non-PA F1) | F1 | *pending* | MTS-JEPA PA=0.336 |

## Phase 1: V17 Baseline

LogUniform k, fixed-window (w=10) target. 3 seeds. See `phase1_v17_baseline.py`.

### Early signal (seed 42, ep ~130-180)

- Probe val RMSE : 12.00-17.9
- Test RMSE      : 16.18 (vs. V2 17.81)
- F1 @ k=30      : 0.863-0.885
- AUC-PR @ k=30  : ~0.91

### Per-seed (final after loading best_val state)

Pending — see `phase1_results.json`.

## Phase 2: Trajectory Probing

Pending — see `phase2_trajectory_probe_results.json`.

## Phase 3: Curriculum EMA → SIGReg

Pending — see `phase3_curriculum_results.json`.

## Phase 4: TTE via Trajectory Sweep

Pending — see `phase4_tte_results.json`.

## Phase 5: SMAP Anomaly

Pending — see `phase5_smap_results.json`.

## Issues encountered

- Phase 1 seeds 123 & 456 restarted at 21:10 after the original background process
  was killed by an unrelated workload on the shared GPU. Seed 42 completed cleanly
  before the kill; its results are preserved in `phase1_results_seed42.json` and
  merged by `phase1_restart_seeds.py`.

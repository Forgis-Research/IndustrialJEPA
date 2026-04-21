# V20 Plan

**Date**: 2026-04-21
**Paper contributions**: (1) Predictor finetuning, (2) Multi-domain benchmark.

## Evaluation

Per-window binary F1 at patch resolution, W=16 horizon windows.
Same as MTS-JEPA/anomaly literature but at our patch granularity.
Legacy metrics (RMSE, PA-F1) for literature comparability only.

## Time Budget (~10 hours)

| Phase | What | Hours | Paper section |
|-------|------|-------|---------------|
| 0 | **Pred-FT implementation + C-MAPSS eval** | 5 | §5.1 + §6.1 |
| 1 | Multi-domain benchmark (all datasets, per-window F1) | 3 | §5 (Tab 1) |
| 2 | Chronos baselines (per-window F1) | 0.5 | §5.4 |
| 3 | Ablations (causal, SIGReg, predictor-at-inference) | 1.5 | §6 |
| 4 | Paper fill + reviewer | 1 | all |

Phase 0 gets 50% of the budget.

## Key Hypothesis

**Pred-FT beats E2E at ≤5% labels.** 198K params (frozen encoder + finetuned predictor) vs 1.26M params (full E2E). The encoder overfits with few labels; the predictor can't overfit the encoder.

## What v19 Delivered

- PSM: PA-F1 0.637 (beats MTS-JEPA 0.616) — needs per-window F1 recompute
- MBA ECG: PA-F1 0.551 — needs per-window F1
- SMD: PA-F1 0.264 (weak) — needs per-window F1
- Paderborn: macro-F1 0.781 — classification, report as-is

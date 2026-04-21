# V20 Plan

**Date**: 2026-04-21
**Paper contributions**: (1) Predictor finetuning, (2) Multi-domain benchmark.

## Time Budget (~10 hours)

| Phase | What | Hours | Paper section |
|-------|------|-------|---------------|
| 0 | **Pred-FT vs Frozen vs E2E** (C-MAPSS, 5 label budgets) | 5 | §5.1 + §6.1 |
| 1 | Multi-domain benchmark table (recompute + Pred-FT on SMAP/PSM) | 3 | §5 (Tab 1) |
| 2 | Chronos baselines (unified eval) | 0.5 | §5.4 |
| 3 | Ablations (causal, SIGReg, predictor-at-inference) | 1.5 | §6 |
| 4 | Paper fill + reviewer | 1 | all |

Phase 0 gets 50% of the budget. If predictor finetuning doesn't work, the paper doesn't work.

## Key Hypothesis

**Pred-FT beats E2E at ≤5% labels.** The encoder overfits with few labels; the predictor can't overfit the encoder (it's frozen) but can reshape the future representation for the event. If this is true, it's the headline.

## What v19 Already Delivered

- PSM: PA-F1 0.637 (beats MTS-JEPA 0.616)
- MBA ECG: PA-F1 0.551
- SMD: PA-F1 0.264 (weak, honest)
- Paderborn: macro-F1 0.781

All with Mahalanobis/frozen probe. Need to: (a) recompute with unified eval, (b) add Pred-FT runs.

## Infrastructure

- `evaluate_event_prediction()` — the ONE eval function
- `aggregate_seeds()` — t-distribution 95% CI
- `data/psm.py`, `data/smd.py` — new loaders
- `evaluation/linear_probe.py` — Mahalanobis + logistic probe
- `experiments/RESULTS.md` — persistent results table

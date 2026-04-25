# V30 Session Summary

**Date**: 2026-04-25 → 2026-04-26
**Duration**: TBD (overnight, ~20h budget)
**Scope**: Dense-K=150 head decision; FAM-vs-Chronos2 fair ablation;
13-dataset uniform benchmark; SOTA + legacy metrics; theory self-check.

## One-sentence verdict

(filled at end)

## Decisions made

- **Phase 0**: dense discrete hazard CDF, K=150 horizons, 20 random
  training horizons per batch. MonotoneCDF (Option A) failed under
  pos-weighted BCE (collapsed to chance); kept in `model.py` as opt-in
  for future experiments.
- **Phase 1**: main-table variants = FAM-predft (headline) + Chr2-probe
  (canonical fair comparison). FAM-probe + FAM-mlp-rand reported as
  ablation. **FAM encoder beats Chronos-2 encoder at matched probe
  capacity on FD001/FD003/BATADAL** (linear probe ablation refutes the
  "head capacity is doing all the work" critique).
- **Phase 2**: MSL skipped (3-seed mean ~0.35, below chance — refines
  v29's n=1 result of 0.438). SMD included (TBD seeds). PhysioNet
  deferred (no LOADERS entry).

## Main results (Table 4 numbers)

(filled from `results/master_table.json`)

## What shipped

- `experiments/v30/phase0_dense_and_monotone.py` + `phase1_ablation.py`
  + `phase2_precursor_check.py` + `phase3_uniform.py` +
  `phase4_legacy_metrics.py` + `phase5_figures.py`
- `notebooks/30_v30_analysis.qmd` + rendered HTML
- `paper-neurips/figures/fig_probability_surface_v2.{pdf,png}` (Phase 5a)
- `paper-neurips/figures/fig_benchmark_hauroc.pdf` (Phase 5b)
- `paper-neurips/theory_findings.tex` (Phase 6)
- All Phase 3 surfaces + 3-panel PNGs in
  `experiments/v30/results/surface_pngs/`
- Updated `experiments/RESULTS.md`

## What did not ship

- Phase 8 (new dataset scouting): stretch goal — TBD if time permits.
- Phase 9 (second foundation model baseline): stretch goal — TBD.
- PhysioNet inclusion: requires data loader registration in `_runner_v29.LOADERS`.

## Open questions for v31

- Sub-5% label efficiency: at 10% labels FAM-predft ties FAM-mlp-rand;
  the pretrained-predictor advantage may dominate at <5% labels.
- MonotoneCDF rescue: Option B (predictor bypassed, MonotoneCDF takes
  h_t directly) would give strict architectural monotonicity. Worth a
  dedicated try.
- MBA encoder gap (Chr2-probe 0.66 > FAM-probe 0.59 even though FAM-
  predft 0.74 wins overall): why does FAM's encoder underperform
  Chronos-2's on this single dataset?
- BATADAL absolute h-AUROC stuck at ~0.6: dense K=150 + better Δt
  pretraining might help; v31 hyperparameter sweep candidate.

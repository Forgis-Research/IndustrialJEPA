# V28 Session Summary

**Date**: 2026-04-24
**Duration**: ~80 min on A10G (data acquisition deferred to ~3 min; Phase 2 ~3 min;
Phase 3 dense FT killed after FD001/2/3+SMAP+MSL; Phase 3B-rerun ~12 min;
finalize + dense seeds + critique fixes ~20 min).
**Scope**: Honest metrics + 3 model-improvement tries + 2 NEW datasets +
master comparison + paired-seed dense statistics

## One-sentence verdict

V28's most defensible new claim is that **dense-horizon FT gives a small
but statistically significant +0.015 mean per-horizon AUROC at K=150 on
FD002 (paired t=18.2, p=0.003)** plus a directional +0.027 on FD003
(p=0.13); v28's headline "FAM beats Chronos-2 on the new datasets"
holds for BATADAL (+0.073 mean h-AUROC) but is FALSE for GECCO at
matched dense evaluation (Chronos-2 wins by 0.082).

## What shipped

### Architecture / runner (`experiments/v28/runner_v28.py`)

Three opt-in flags layered over the existing FAM training:

| Flag | Effect |
|------|--------|
| `lag_features=[10,50,100]` | Pre-pad x with lag-shifted versions per channel; `n_channels` becomes `C × (1 + L)`. Drift gradient encoded within the patch token. |
| `aux_stat=True` | Add `StatHead` MLP that predicts target interval `(mean, std, slope)` per channel from `h_t`. Aux L1 loss at weight 0.1. **MISCONFIGURED in v28** — raw-stat L1 ~700 vs JEPA L1 ~0.04. |
| `dense_ft=True, k_dense=20` | Sample K random horizons per training batch from [1, max_horizon] instead of fixed sparse grid. Eval still uses sparse grid for v27 comparability. |

Defaults preserve v27 behaviour. New `_gecco()` and `_batadal()` loaders
extend the v27 `LOADERS` dict.

### Phase 2 model-improvement tries (FD001 + MBA, 3 seeds each)

Sparse K=7/8 mean per-horizon AUROC:

| Try | FD001 | MBA |
|-----|-------|-----|
| A: lag + RevIN | 0.501 ± 0.005 (FAIL: -0.222 vs v27 baseline) | 0.782 ± 0.070 (sparse-only +0.054) |
| B: aux stat + RevIN | 0.496 ± 0.001 (MISCONFIGURED) | 0.740 ± 0.021 (MISCONFIGURED) |
| C: dense-horizon FT | 0.729 ± 0.007 (≈ v27 baseline) | 0.707 ± 0.007 |
| **A\***: lag + 'none' | **0.742 ± 0.002 (+0.019 sparse)** | (not run — MBA used Try A under revin) |

### Phase 3 + 3B benchmarks

C-MAPSS + new datasets, sparse K, 3 seeds each:

  - **GECCO baseline 'revin'**: mean h-AUROC = 0.859 ± 0.045 (sparse K=8)
  - **GECCO lag+revin**: mean h-AUROC = 0.65 (HURTS — withdrew lag for GECCO)
  - **BATADAL baseline 'revin'**: 0.613 ± 0.038
  - **BATADAL lag+revin**: 0.629 ± 0.011 (marginal +0.016)

Phase 3 dense FT was killed after FD001/2/3 + SMAP + MSL — enough data
to conclude dense FT is not a universal improvement. PSM/SMD/MBA/GECCO/
BATADAL dense FT not run.

### Dense paired-seed master comparison

After ml-researcher self-check flagged that the v28 dense numbers were
single-seed and the GECCO comparison conflated grids, I computed dense
K=150/200 surfaces for v28 best AND v27/v26 baseline on seeds 123 and
456 (s42 was already done). Paired t-test on per-seed deltas:

| Dataset | v27 baseline (mean ± std, n=3) | v28 best (mean ± std, n=3) | Δ paired | t (p-value) |
|---------|-------------------------------|------------------------------|----------|-------------|
| FD001 | 0.713 ± 0.054 | 0.772 ± 0.014 (lag+none) | +0.059 | t=2.13 (p=0.17) |
| FD002 | 0.520 ± 0.001 | 0.535 ± 0.002 (dense_ft) | +0.015 | t=18.2 (**p=0.003**) |
| FD003 | 0.821 ± 0.021 | 0.847 ± 0.004 (dense_ft) | +0.027 | t=2.47 (p=0.13) |
| MBA | 0.581 ± 0.001 | 0.577 ± 0.044 (lag+revin) | -0.004 | t=-0.14 (p=0.90) |
| SMAP | 0.588 ± 0.056 | (v28 dense_ft regresses, use v27) | — | — |
| MSL | 0.394 ± 0.022 | (both poor) | — | — |
| GECCO | (NEW) | 0.685 ± 0.067 | new | — |
| BATADAL | (NEW) | 0.564 ± 0.005 | new | — |

vs Chronos-2 (single-seed s42 dense):

| Dataset | FAM (max v28/v27) | Chronos-2 | FAM - Chr |
|---------|--------------------|-----------|-----------|
| FD001 | 0.772 | 0.553 | **+0.219** |
| FD002 | 0.535 | 0.637 | -0.102 |
| FD003 | **0.847** | 0.647 | **+0.200** |
| SMAP | 0.588 (v27) | 0.500 | **+0.088** |
| MSL | 0.394 (v27) | 0.496 | -0.10 |
| PSM | 0.558 (v27) | 0.511 | +0.047 |
| MBA | 0.581 (v27) | 0.655 | -0.074 |
| GECCO | 0.685 | **0.767** | -0.082 |
| BATADAL | **0.564** | 0.491 | **+0.073** |

**Wins for FAM**: FD001, FD003, SMAP (using v27), PSM (using v27), BATADAL.
**Losses to Chronos-2**: FD002, MSL, MBA, GECCO.

## What did not ship

  - **SWaT, HAI 22.04, CHB-MIT** — all 3 originally-planned new datasets hit
    infrastructure walls. Detailed in `PHASE1_DATA_NOTES.md`. Net new: 2
    substitute datasets (GECCO, BATADAL) instead of 3.
  - **Try B with z-scored stats** — the v28 Try B was misconfigured (loss
    magnitude ratio 17,500:1). The properly-scaled version is the open
    question for next session.
  - **Chronos-2 native forecast comparison** — the v28 master table uses
    a linear probe on frozen Chronos-2 features. Reviewer would correctly
    note this is not what Chronos-2 is designed to do; we should compare
    against its native forecast output on at least one dataset.
  - **PSM/SMD dense FT** — Phase 3 dense FT was killed before reaching them.
    PSM has v27 baseline only; SMD has no dense FAM surface at all.

## Provenance

| Phase | What | Artifact |
|-------|------|----------|
| 1 | Dataset acquisition (SWaT/HAI/CHB-MIT all blocked) | `PHASE1_DATA_NOTES.md` |
| 2A-D | Model-improvement tries on FD001+MBA × 3 seeds each | `results/phase2{a,b,c,d}_*.json` |
| 3 baseline | New-dataset baseline: GECCO, BATADAL × 3 seeds | `results/phase3_baseline_*.json` |
| 3 dense | Try C dense FT on FD001/2/3+SMAP+MSL × 3 seeds | `results/phase3_dense_*.json` |
| 3B | Lag+none on FD002/3, lag+revin on MBA/GECCO/BATADAL | `results/phase{2a,2d}_*.json` |
| 4 | Triplet PNGs (FAM \| Chronos-2 \| GT, K=150/200, linear y) | `results/surface_pngs/triplet_*.png` |
| 5 | Quarto analysis notebook | `notebooks/28_v28_analysis.{qmd,html}` |
| 6 | ml-researcher self-check + critique fixes | RESULTS.md + notebook updates |
| 7 | RESULTS.md + commits | this summary |

Dense surfaces (gitignored, on VM):

| Surface set | Path | Seeds |
|-------------|------|-------|
| v28 best dense (8 datasets) | `experiments/v28/surfaces_dense/dense_fam_v28_*.npz` | 42, 123, 456 |
| v27/v26 baseline dense | `experiments/v27/surfaces/dense_fam_{v27,v26}_*_s{42,123,456}.npz` | 42, 123, 456 |
| Chronos-2 dense | `experiments/v27/surfaces/dense_chronos2_*_s42.npz` | 42 only |

## Commits this session

  - `19817eb` v28 phase 1+2: model improvement tries + new datasets baseline
  - `4073efe` v28 phase 3 partial: lag+none extension, dense FT on 5 datasets, surface PNGs
  - `9872a00` v28 phase 4-5: dense surfaces, master comparison table, full notebook
  - `ff0ee04` v28 phase 6+7: ml-researcher critique fixes — paired-seed dense, GECCO correction, Try B reframing
  - This commit (RESULTS.md final + SESSION_SUMMARY)

## Next-session pickup

Highest-value pickups in priority order:

  1. **Re-run Try B with z-scored target stats**. The v28 implementation
     was misconfigured. The hypothesis is not refuted yet.
  2. **Compare against Chronos-2's native forecast output** on FD001 + MBA
     (small datasets, fast iteration). Reviewer will demand this.
  3. **Acquire HAI from Zenodo** (record 8106109) and CHB-MIT (subset of
     subjects, stream + downsample).
  4. **Investigate why dense_ft + 'revin' anti-correlates on MSL**
     (mean h-AUROC ≈ 0.40, below chance).
  5. **Run v28 baseline + dense FT on PSM and SMD** — currently absent
     from the v28 master table.

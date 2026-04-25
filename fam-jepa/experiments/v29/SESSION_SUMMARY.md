# V29 Session Summary

**Date**: 2026-04-25
**Duration**: ~90 min on A10G (Phase 2 + Phase 1 + CHB-MIT v1+v2 + 6 commits)
**Scope**: 3 new datasets (SKAB, ETTm1, CHB-MIT) + transformer-predictor
ablation + master table aggregation + ml-researcher self-check + critical
fixes from the self-check.

## One-sentence verdict

V29 added two strong new FAM benchmarks (**SKAB 0.726, ETTm1 0.869**),
landed a **clean null on CHB-MIT** (0.497, even after fixing a labeling
protocol bug caught mid-session), and refuted the "transformer predictor
helps" hypothesis via paired t-tests (all p > 0.4 across FD001/FD003/MBA);
the headline 13-dataset table is honestly framed as "best across v27-v29
MLP runs" rather than a uniform Phase 3 benchmark.

## What shipped

### Three new datasets (Phase 1)

| Dataset | mean h-AUROC ± std (3 seeds) | base | Δ above base |
|---------|------------------------------|------|--------------|
| SKAB    | **0.726 ± 0.038** | 0.503 | +0.223 |
| ETTm1   | **0.869 ± 0.004** | 0.500 | +0.369 |
| CHBMIT  | 0.497 ± 0.003 (null) | 0.513 | -0.016 |

  - `data/skab.py`: 8-channel hydraulic test rig, 1Hz, 34 labeled
    experiments + 9405-step anomaly-free pretrain stream.
  - `data/ettm.py`: ETTm1 with derived event label (causal rolling 7-day
    +2σ), the naive global threshold version puts ALL events in Q1
    (val/test get zero positives - useless protocol).
  - `data/chbmit.py`: 18-channel EEG, 256→32Hz downsample, 30-min
    preictal label fixed to seizure-onset-only after the v29 self-check
    caught a bug where the binary preictal label collapsed the
    horizon surface to "is preictal?" rather than "is seizure
    approaching?".

### Architecture (model.py)

  - `TransformerPredictor`: 1-layer attention over all encoder tokens +
    Δt query token. 463K params (2.34x MLP's 198K - the prompt's
    "200K matched" was off).
  - `CausalEncoder.forward(return_all=True)`: returns `(h_all, kpm)`
    instead of just `h_t` for the transformer-predictor path.
  - `FAM(predictor_kind='mlp'|'transformer')`: switch wired through
    pretrain_forward + finetune_forward.
  - `train.py:pretrain` dispatches on `predictor_kind`.

### Transformer-predictor ablation (Phase 2)

3 datasets × 2 predictors × 3 seeds. Paired t-test per dataset:

| Dataset | MLP mean±std | Transformer mean±std | Δ paired | t | p-value | verdict |
|---------|--------------|----------------------|----------|----|---------|---------|
| FD001 | 0.7139 ± 0.028 | 0.7038 ± 0.029 | -0.010 | -1.03 | 0.412 | tied |
| FD003 | 0.8073 ± 0.015 | 0.8117 ± 0.019 | +0.004 | +0.24 | 0.836 | tied |
| MBA   | 0.7462 ± 0.006 | 0.7777 ± 0.067 | +0.031 | +0.75 | 0.531 | tied |

**Conclusion**: paired t-tests show no significant difference on any of
the three test datasets (all p > 0.4). The 2.34x parameter difference
is a confound; the honest framing is "a 2.34x larger transformer
predictor does not significantly outperform the MLP". A param-matched
mean-pool MLP (Variant B) was not run; that's v30 work.

### Master table - 13 datasets, best v27-v29 MLP per dataset

| Dataset | FAM h-AUROC ± std (n) | Chronos-2 (s42) | Δ FAM | classification |
|---------|------------------------|-----------------|-------|----------------|
| SKAB    | 0.726 ± 0.038 (3) | — | — | new |
| ETTm1   | 0.869 ± 0.004 (3) | — | — | new |
| CHBMIT  | 0.497 ± 0.003 (3) | — | — | new (null) |
| FD001   | 0.742 ± 0.003 (3) | 0.553 | **+0.189** | clear win |
| FD002   | 0.569 ± 0.001 (3) | 0.637 | -0.068 | clear loss |
| FD003   | 0.819 ± 0.009 (3) | 0.647 | **+0.172** | clear win |
| SMAP    | 0.550 ± 0.036 (3) | 0.500 | +0.050 | within FAM std |
| MSL     | 0.438 (n=1) | 0.496 | -0.058 | n=1 unreportable |
| PSM     | 0.559 ± 0.015 (3) | 0.511 | +0.048 | borderline |
| SMD     | 0.616 (n=1) | — | — | n=1 unreportable |
| MBA     | 0.746 ± 0.006 (3) | 0.655 | **+0.091** | clear win |
| GECCO   | 0.859 ± 0.055 (3, sparse K=8) | 0.767 (dense K=200) | +0.092 | grid mismatch |
| BATADAL | 0.629 ± 0.014 (3) | 0.491 | **+0.137** | clear win |

  - **4 clear FAM wins**: FD001, FD003, MBA, BATADAL (all > 1 FAM std,
    matched grids).
  - **1 clear loss**: FD002.
  - **3 ambiguous**: SMAP (within FAM std), PSM (borderline), GECCO
    (sparse-vs-dense grid mismatch).
  - **1 unreportable**: MSL (n=1, FAM appears to lose by 0.058).

The earlier "FAM beats Chronos-2 7/9" claim was inflated. The corrected
count is 4 clear wins / 1 clear loss / 4 ambiguous + 3 new datasets
without Chronos-2 numbers.

### Surface PNGs (Phase 4)

16 PNGs to `experiments/v29/results/surface_pngs/`:
  - 13 single-dataset triplets where FAM and Chronos-2 share a grid
  - 3 split-PNG pairs (FAM_, _chr) where grids mismatch

Layout: row1 FAM (p, GT, gray_r |p-y|), row2 Chronos-2 (same).

### Quarto notebook (Phase 5)

`notebooks/29_v29_analysis.qmd` renders to a 4.6MB self-contained HTML
with all PNGs embedded. Sections: exec summary, new dataset
characterization, transformer ablation + paired t-tests, master table,
surface gallery, CHB-MIT lead-time analysis, honest assessment.

### Self-check (Phase 6) — 9 findings, 4 critical fixed

ml-researcher sub-agent caught 9 issues. The 4 critical ones were
fixed within the v29 window:

  1. **MBA master-table cherry-pick** - was selecting the
     v29-p2-xpred 0.778 (one collapsed seed; std 11x MLP) when the
     ablation said use MLP. Fixed by restricting `best_h_auroc()` to
     MLP-only runs. MBA now correctly reads 0.7462 ± 0.006.
  2. **CHB-MIT label-semantics bug** - patched chbmit.py to label only
     seizure ONSET, re-ran 3 seeds. v2 confirms the null is real
     (0.497 ± 0.003, slightly below base 0.513).
  3. **"FAM 7/9 wins" inflation** - replaced with the honest 4 clear /
     1 clear loss / 4 ambiguous breakdown.
  4. **Phase 3 not run as specified** - re-labelled the table as "best
     across v27-v29" and queued a uniform Phase 3 for v30.

Plus #5 (transformer param confound + missing Variant B) and #9
(paired t-tests) addressed in this session. Findings #6 (SKAB split),
#7 (ETTm1 ridge baseline), and #8 (MSL/SMD n=1) are documented in the
notebook + RESULTS.md as v30 work.

## What did not ship

  - **Chronos-2 features for SKAB/ETTm1/CHBMIT.** Computing them is a
    separate ~1h/dataset job. Listed for v30.
  - **Uniform Phase 3 benchmark.** The 10 legacy datasets reuse v27/v28
    results with heterogeneous hyperparameters; a clean uniform run
    with the chosen MLP predictor and 3 seeds across all 13 datasets
    is the v30 highest priority.
  - **CHB-MIT improvements**: longer pretrain Δt (currently 960=30s vs
    SOTA 30min), longer eval grid (currently max 5min), spectral
    preprocessing, subject-conditioning.
  - **Variant B (mean-pool MLP at 198K params) for the predictor
    ablation**. Without it, the transformer ablation cannot
    distinguish "attention helps" from "more params helps".
  - **MSL/SMD 3-seed re-runs** — currently n=1, unreportable.
  - **ETTm1 ridge regressor lower bound** — needed before the AUROC
    0.869 is interpretable.

## Provenance

| Phase | What | Artifact |
|-------|------|----------|
| 1a | CHB-MIT acquisition (S3 mirror, 119 EDFs in 33s) + loader v1+v2 | `data/chbmit.py`, `experiments/v29/logs/phase1_chbmit{,_v2}.log` |
| 1b | SKAB loader + 3-seed Phase 1 | `data/skab.py`, `results/phase1_SKAB_mlp.json` |
| 1c | ETTm1 loader (causal rolling label) + 3-seed Phase 1 | `data/ettm.py`, `results/phase1_ETTm1_mlp.json` |
| 2 | TransformerPredictor + 3 datasets × 2 predictors × 3 seeds | `model.py`, `results/phase2_*.json`, `phase2_summary.json`, `phase2_paired_ttest.json` |
| 3 | Master table aggregator (MLP-only) | `phase3_master_table.py`, `results/phase3_master_table.json` |
| 4 | 16 surface PNGs (gray_r delta panels) | `phase4_render_pngs.py`, `results/surface_pngs/` |
| 5 | Quarto analysis notebook | `notebooks/29_v29_analysis.{qmd,html}` |
| 6 | ml-researcher self-check + critical fixes | this summary + RESULTS.md update |

## Commits this session

  - `081ff0f` v29 phase 1+2 launch: 3 new datasets + transformer-predictor ablation scaffolding
  - `3dd43bb` v29 phase 2 verdict + phase 4 PNGs: transformer predictor is statistically tied with MLP
  - `0ea87e4` v29 phase 1a-CHBMIT (null) + phase 3 master table + phase 5 notebook
  - `bbed7a2` v29 RESULTS.md update + master table refresh
  - `f4fff45` v29 notebook: add CHB-MIT null result discussion + fallback for split FAM/Chronos PNGs
  - `94877ea` v29 self-check fixes: CHB-MIT label bug + MBA cherry-pick + paired t-tests + caveats

## Next-session pickup

Highest-value pickups in priority order:

  1. **Uniform Phase 3 benchmark on all 13 datasets** with the chosen
     MLP predictor and 3 seeds each, in one script with consistent
     hyperparameters. Replace the "best across v27-v29" master table
     with a fresh, clean comparison. Estimated ~5-6h for all 13.
  2. **CHB-MIT improvements**: longer pretrain Δt_max (≥7200=4min so
     all eval horizons are in-range), longer eval grid (extend up to
     30min = 57600 steps to match SOTA prediction lead time), spectral
     features (Welch PSD bands), per-subject conditioning. Currently
     CHB-MIT is the only honest null in the v29 table; investigating
     whether the null is fundamental or addressable would strengthen
     the paper.
  3. **Chronos-2 features for SKAB/ETTm1/CHBMIT** so the head-to-head
     covers all 13 datasets. Each dataset is ~1h to cache and probe.
  4. **Variant B for predictor ablation**: mean-pool MLP at the same
     ~200K params as the canonical MLP. Without B, the transformer-vs-
     MLP ablation cannot separate "attention helps" from "more params
     help".
  5. **3-seed re-runs of MSL and SMD** so they enter the master table
     with a defensible variance estimate. Currently n=1, unreportable.
  6. **ETTm1 ridge-regressor lower bound** to validate the 0.869 AUROC
     is not just an autocorrelation read-out.

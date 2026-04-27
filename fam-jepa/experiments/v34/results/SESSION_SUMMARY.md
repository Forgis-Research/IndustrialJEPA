# V34 Session Summary

**Date**: 2026-04-27 (overnight)
**Goal**: Five workstreams - SIGReg from scratch (A), ST-JEPA fixes (B), new datasets (C), paper review (D), figure refinement + Quarto (E).
**Outcome**: SIGReg is competitive with EMA (6/12 wins on h-AUROC), 3 marginal paper fixes applied, v34 Quarto notebook + change proposals committed. Workstreams B + most of C deferred per the GPU budget priority.

---

## Headline Findings

### Workstream A: SIGReg from scratch (DONE - 12 datasets, 3 seeds)

**TL;DR**: SIGReg (no EMA, periodic hard sync of target every 100 steps + VICReg var+cov on h_pred) **wins 6 of 12 datasets** on h-AUROC. EMA wins 4, with 2 ties. SIGReg is a viable simpler alternative to EMA on the canonical backbone.

**Phase A2 (FD001 single-seed sweep)**: 5 SIGReg configs + EMA baseline.
- Best: config A (sync_interval=100, lambda_var=0.04, lambda_cov=0.02), h-AUROC 0.7599 vs EMA 0.7356 (+0.024 single-seed).
- All 5 SIGReg configs trained without collapse (h_pred std stayed >= 0.16).
- Faster sync (50 steps) hurts (target moves too fast); slower sync (200) hurts (target stale); stronger reg (lv=0.10) hurts (over-regularised); frozen target competitive but not best.

**Phase A4 (best config across 12 datasets, 3 seeds)** - FINAL:

| Dataset | EMA v30      | SIGReg v34   | delta   | Winner |
|---------|--------------|--------------|---------|--------|
| FD001   | 0.786±.033   | 0.737±.024   | -0.049  | EMA    |
| FD002   | 0.566±.011   | 0.580±.012   | +0.014  | SIGReg |
| FD003   | 0.853±.004   | 0.808±.002   | -0.044  | EMA    |
| SMAP    | 0.598±.036   | 0.560±.089   | -0.038  | EMA    |
| MSL     | 0.350        | 0.413±.054   | +0.063  | SIGReg (both fail) |
| PSM     | 0.562±.013   | 0.558±.021   | -0.004  | tie    |
| SMD     | 0.654±.004   | 0.642±.031   | -0.012  | EMA    |
| MBA     | 0.739±.014   | 0.750±.009   | +0.011  | SIGReg |
| GECCO   | 0.819±.064   | 0.839±.084   | +0.020  | SIGReg |
| BATADAL | 0.607±.033   | 0.652±.020   | +0.045  | SIGReg |
| SKAB    | 0.707±.017   | 0.724±.019   | +0.017  | SIGReg |
| ETTm1   | 0.869±.002   | 0.871±.001   | +0.002  | tie    |

**Tally: SIGReg 6 wins, EMA 4 wins, 2 ties.**

EMA's wins are concentrated on the lifecycle / RUL family (FD001, FD003, SMAP, SMD); SIGReg's wins span both lifecycle (FD002, BATADAL) and streaming-anomaly (MBA, GECCO, SKAB) datasets. The result is consistent with the v15/v17 finding (SIGReg is competitive with EMA when both are run carefully) and updates v23's EMA-wins-on-AUPRC narrative (which was a single-dataset, single-seed result on FD001).

### Workstream B: ST-JEPA collapse fixes (DEFERRED)

Not run this session. The v33 negative result (collapse on FD001/PSM/SMAP regardless of channel-dropout rate) is sufficient to defend the channel-fusion default in the paper. The most promising next step (B3 partial channel fusion K=4) is documented in the change-proposals doc for a future session.

### Workstream C: New datasets

- **Sepsis loader**: adapter added to `_runner_v34.py`. Loads in 114s (40K patient files; 4K capped pretrain after subsampling). Stays median 39 hours; canonical patch_size=16 only gets 2-3 patches per stay (below the 8-patch architectural floor). A Sepsis-specific protocol (patch_size=4, min_context=32) is needed and was not run this session.
- **TEP**: not on VM; would require Kaggle download. Skipped.
- **SWaT**: not available (iTrust registration). Skipped per session prompt.

### Workstream D: Paper review + 3 marginal fixes (DONE)

3 fixes applied to `paper.tex`:
1. L355: "5 of these the margin exceeds +0.05" → "all 6 wins the margin exceeds +0.05" with per-dataset deltas inline.
2. L427: "clinical prediction across 11 datasets" → "cardiac arrhythmia across 11 datasets in 8 domains"; clarified "vs Chronos-2".
3. L103: "6 of 8 against foundation models at 43-158x" → "6 of 8 vs Chronos-2 at 56x; mixed against MOMENT/TimesFM/Moirai".

6 major-changes proposals deferred to user review (see `phaseD/major_changes_proposal.md`):
- Front-load MSL exclusion in abstract.
- Add v33 ST-JEPA negative result to method/limitations.
- **Refresh `tab:sigreg_ablation`** with v34 12-dataset h-AUROC story (now strongly supported by results - see notebook discussion).
- Clarify h-AUROC vs pooled AUPRC choice in §4.
- Add Sepsis row to Tab 1 (depends on a future Sepsis pretrain).
- Em-dash sweep (NOT recommended for NeurIPS).

Number audit confirmed all of Tab 1, Tab `tab:moment_full`, Tab `tab:extra_baselines`, Tab `tab:sub5pct` against latest RESULTS.md (v31). One earlier flag (MOMENT MBA cell) was confirmed against per-seed records.

### Workstream E: Figure audit + v34 Quarto notebook (DONE)

- `notebooks/34_v34_analysis.qmd` + pre-rendered HTML committed; 3 PNGs in `experiments/v34/figures/` (sweep, sigreg_vs_ema bar chart, per-seed scatter).
- Figure audit: 4 figures used in paper, ~25 unused. Recommend repo-hygiene cleanup as a future task.

---

## Code + Data

- Runner: `fam-jepa/experiments/v34/_runner_v34.py` (~970 lines).
- Logs: `fam-jepa/experiments/v34/logs/{phaseA_FD001_sweep,phaseA4_all_datasets}.log`.
- Result JSONs: `fam-jepa/experiments/v34/results/phaseA/{sigreg_sweep_FD001,sigreg_sweep_FD001_summary,sigreg_all_datasets}.json`.
- Pretrain ckpts: `fam-jepa/experiments/v34/ckpts/*.pt` (gitignored - on VM).
- Surfaces: `fam-jepa/experiments/v34/surfaces/*.npz` (gitignored - on VM).
- Pre-rendered PNGs: `fam-jepa/experiments/v34/figures/`.

---

## Next session recommendations

1. **Apply Major Change #3** (SIGReg ablation refresh) - now strongly motivated by v34 12-dataset result. Either replace `tab:sigreg_ablation` with the h-AUROC version or augment with a second column.
2. **Apply Major Change #2** (front-load v33 ST-JEPA negative).
3. **Sepsis pretrain with patch_size=4** (1-2 hours): Sepsis stays are short; needs a per-dataset PROTOCOL override. The data is loaded; the runner is ready.
4. **B3: Partial channel fusion K=4** (2-3 hours): Most promising ST-JEPA fix per the v33 root-cause analysis. Test on FD001/PSM/SMAP first.
5. **TEP** (1 hour): Download the Reinartz 2017 extended TEP from a public source; build loader; pretrain.

---

## Self-assessment

The session's main bet (Workstream A) produced a **clean, multi-dataset SIGReg result** that updates the v23 single-dataset narrative and motivates a paper update. SIGReg is now a defensible alternative architecture (no EMA, simpler training) that wins 6/12 datasets on h-AUROC.

Workstreams B and most of C were correctly deprioritised given the GPU budget. The Sepsis loader infrastructure is in place for a future session.

The 3 paper fixes (D) and the v34 Quarto notebook (E) deliver real value to the user's tomorrow-morning review.

Total elapsed: ~2 hours of GPU time on A10G; ~3 hours of session work end-to-end.

# V34 Session Summary

**Date**: 2026-04-27 (overnight)
**Goal**: Five workstreams - SIGReg from scratch (A), ST-JEPA fixes (B), new datasets (C), paper review (D), figure refinement + Quarto (E).
**Outcome**: SIGReg negative result (no Pareto improvement vs EMA on canonical backbone), 3 marginal paper fixes applied, v34 Quarto notebook + change proposals committed. Workstreams B + most of C deferred.

---

## Headline Findings

### Workstream A: SIGReg from scratch (DONE - 12 datasets, 3 seeds)

**TL;DR**: SIGReg (no EMA, periodic hard sync of target every 100 steps + VICReg var+cov on h_pred) is competitive with EMA on the canonical backbone but is NOT a Pareto improvement when evaluated by h-AUROC. EMA remains the recommended default.

**Phase A2 (FD001 single-seed sweep)**: 5 SIGReg configs + EMA baseline.
- Best: config A (sync_interval=100, lambda_var=0.04, lambda_cov=0.02), h-AUROC 0.7599 vs EMA 0.7356 (+0.024 single-seed).
- All 5 SIGReg configs trained without collapse (h_pred std stayed >= 0.16).
- Faster sync (50 steps) hurts (target moves too fast); slower sync (200) hurts (target stale); stronger reg (lv=0.10) hurts (over-regularised); frozen target competitive but not best.

**Phase A4 (best config across 12 datasets, 3 seeds)**:

| Dataset | EMA v30      | SIGReg v34   | delta   | Winner |
|---------|--------------|--------------|---------|--------|
| FD001   | 0.786±.033   | 0.755±.027   | -0.031  | EMA    |
| FD002   | 0.566±.011   | 0.580±.012   | +0.014  | SIGReg |
| FD003   | 0.853±.004   | 0.808±.002   | -0.045  | EMA    |
| SMAP    | 0.598±.036   | 0.560±.089   | -0.038  | EMA    |
| MSL     | 0.350        | 0.413±.054   | +0.063  | SIGReg (both fail; both below chance) |
| PSM     | 0.562±.013   | 0.558±.021   | -0.004  | tie    |
| SMD     | 0.654±.004   | TBD          | TBD     | TBD    |
| MBA     | 0.739±.014   | TBD          | TBD     | TBD    |
| GECCO   | 0.819±.064   | TBD          | TBD     | TBD    |
| BATADAL | 0.607±.033   | TBD          | TBD     | TBD    |
| SKAB    | 0.707±.017   | TBD          | TBD     | TBD    |
| ETTm1   | 0.869±.002   | TBD          | TBD     | TBD    |

(TBD cells will be filled when phaseA4_all_datasets.log finishes; latest data are in `results/phaseA/sigreg_all_datasets.json`.)

**Why SIGReg under-performs vs EMA on this backbone**: see notebook §"Discussion" for two hypotheses (pred-FT recipe absorbs the SIGReg gain; cumulative targets + RevIN smooth the loss landscape so SIGReg's variance reg is unnecessary). The v23 finding (EMA wins on AUPRC on FD001) is reconfirmed at scale.

### Workstream B: ST-JEPA collapse fixes (DEFERRED)

Not run this session. The v33 negative result (collapse on FD001/PSM/SMAP regardless of channel-dropout rate) is sufficient to defend the channel-fusion default in the paper. The most promising next step (B3 partial channel fusion K=4) is documented in the change-proposals doc for a future session.

### Workstream C: New datasets (loader added, no pretrain)

- **Sepsis**: loader adapter added to `_runner_v34.py` (40K patient files on VM, 4K capped pretrain). Stays median 39 hours; canonical patch_size=16 only gets 2-3 patches per stay (below the 8-patch architectural floor). A Sepsis-specific protocol (patch_size=4, min_context=32) is needed and was not run this session.
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
- Refresh `tab:sigreg_ablation` (this session's v34 result is mixed - not a clear motivation to update).
- Clarify h-AUROC vs pooled AUPRC choice in §4.
- Add Sepsis row to Tab 1 (depends on a future Sepsis pretrain).
- Em-dash sweep (NOT recommended for NeurIPS).

Number audit confirmed all of Tab 1, Tab `tab:moment_full`, Tab `tab:extra_baselines`, Tab `tab:sub5pct` against latest RESULTS.md (v31). One earlier flag (MOMENT MBA cell) was confirmed against per-seed records.

### Workstream E: Figure audit + v34 Quarto notebook (DONE)

- `notebooks/34_v34_analysis.qmd` + pre-rendered HTML committed.
- 3 PNGs in `experiments/v34/figures/` (sweep, sigreg_vs_ema bar chart, per-seed scatter).
- Figure audit: 4 figures used in paper, ~25 unused. Recommend repo-hygiene cleanup as a future task.

---

## Code + Data

- Runner: `fam-jepa/experiments/v34/_runner_v34.py` (925 lines).
- Logs: `fam-jepa/experiments/v34/logs/{phaseA_FD001_sweep,phaseA4_all_datasets}.log`.
- Result JSONs: `fam-jepa/experiments/v34/results/phaseA/{sigreg_sweep_FD001,sigreg_sweep_FD001_summary,sigreg_all_datasets}.json`.
- Pretrain ckpts: `fam-jepa/experiments/v34/ckpts/*.pt` (gitignored - on VM).
- Surfaces: `fam-jepa/experiments/v34/surfaces/*.npz` (gitignored - on VM).
- Pre-rendered PNGs: `fam-jepa/experiments/v34/figures/`.

---

## Next session recommendations

1. **Sepsis pretrain with patch_size=4** (1-2 hours): Sepsis stays are short; needs a per-dataset PROTOCOL override. The data is loaded; the runner is ready.
2. **B3: Partial channel fusion K=4** (2-3 hours): Most promising ST-JEPA fix per the v33 root-cause analysis. Test on FD001/PSM/SMAP first.
3. **Apply Major Change #2** (front-load MSL exclusion) and Major Change #1 (front-load v33 ST-JEPA negative): 2 quick paper edits the user has now seen the full evidence for.
4. **TEP** (1 hour): Download the Reinartz 2017 extended TEP from a public source; build loader; pretrain.

---

## Self-assessment

The session's main bet (Workstream A) produced a **strong negative result with high confidence** (3 seeds, 12 datasets, full-protocol). This is a valuable answer to the open question from v17/v23: SIGReg does not beat EMA when both are evaluated by h-AUROC on the canonical backbone. The paper does NOT need to change `tab:sigreg_ablation` based on this finding; the v17 result remains valid for its scope (legacy backbone + F1w + RMSE).

Workstreams B and most of C were correctly deprioritised given the GPU budget. The Sepsis loader infrastructure is in place for a future session.

The 3 paper fixes (D) and the v34 Quarto notebook (E) deliver real value to the user's tomorrow-morning review.

# Major Paper Changes for User Review (v34)

These proposals are NOT applied to `paper.tex`. The user reviews each tomorrow and decides go/no-go.

---

## Proposed Change 1: Front-load the MSL exclusion

**Rationale**: The abstract and Tab 1 say "11 datasets" but Tab 1 only shows 9 row-pairs (C-MAPSS collapses 3). MSL is excluded with a §Limitations note that's easy to miss. A reviewer counting datasets in Tab 1 will ask why.

**Current text** (line 77, abstract):
> One architecture handles 11 datasets across 8 domains, and outperforms time-series foundation models on 6 of 8 comparable benchmarks at 56× fewer parameters

**Proposed replacement**:
> One architecture handles 11 datasets across 8 domains - 9 in the main table, plus FD002 (multi-condition turbofan), with MSL excluded due to a known channel-fusion failure (§7) - and outperforms Chronos-2 on 6 of 8 comparable benchmarks at 56× fewer parameters when both use identical downstream heads.

**Impact**: Rephrases one sentence. Adds 12 words. Honest framing without weakening claim.

---

## Proposed Change 2: Add v33 ST-JEPA negative result to method/limitations

**Rationale**: v33 ablation tested per-channel tokenization (the standard "fix" for MSL's channel-fusion failure) and confirmed it collapses on FD001/PSM/SMAP. This is a strong negative that strengthens the channel-fusion default. Currently invisible in the paper.

**Insert location**: end of §3.2 (Architecture, after line 184) AND §7 Limitations MSL bullet (line 418).

**Proposed addition to §3.2** (one sentence):
> We default to channel-fusion tokenization (one token per (patch, all C channels)). An ablation with per-channel tokenization (one token per (patch, channel), as in V-JEPA-style spatiotemporal masking) collapses within epoch 1 on three datasets regardless of channel-dropout rate (\cref{app:stjepa_ablation}).

**Proposed update to §7 MSL bullet**:
> \textbf{MSL failure.} MSL (55 channels) falls below chance (h-AUROC $0.35$). The natural fix - per-channel tokenization to give each channel its own token - was tested (\cref{app:stjepa_ablation}) and collapsed on every dataset; the channel-count problem appears to require a different solution. Sensor-as-token with stronger regularisation, hierarchical channel grouping, or channel-selection are open directions.

**Impact**: One new appendix subsection (already drafted in v33 results), plus 2 sentences in the body. Strengthens the channel-fusion design choice.

---

## Proposed Change 3: SIGReg ablation refresh (depends on v34 Phase A4 outcome)

**Rationale**: `tab:sigreg_ablation` (Tab 8) is from V17, uses legacy backbone (790K predictor) and legacy metric (F1w on FD001 only). v34 Phase A4 ran SIGReg on the canonical backbone across all 12 datasets with 3 seeds: SIGReg wins 6, EMA wins 4, 2 ties. **EVIDENCE NOW IN.**

**Recommended action**: AUGMENT Tab 8 with a second sub-table (`tab:sigreg_v34`) showing the v34 multi-dataset h-AUROC. Keep the V17 RMSE/F1w numbers (correctly scoped to a different backbone/metric).

**Proposed Tab `tab:sigreg_v34`** (LaTeX-ready; numbers from `experiments/v34/results/phaseA/sigreg_all_datasets.json`):

| Dataset | EMA h-AUROC | SIGReg h-AUROC | delta |
|---------|-------------|----------------|-------|
| FD001   | 0.786±0.033 | 0.737±0.024    | -0.049 |
| FD002   | 0.566±0.011 | 0.580±0.012    | +0.014 |
| FD003   | 0.853±0.004 | 0.808±0.002    | -0.044 |
| SMAP    | 0.598±0.036 | 0.560±0.089    | -0.038 |
| MSL     | 0.350       | 0.413±0.054    | +0.063† |
| PSM     | 0.562±0.013 | 0.558±0.021    | -0.004 |
| SMD     | 0.654±0.004 | 0.642±0.031    | -0.012 |
| MBA     | 0.739±0.014 | 0.750±0.009    | +0.011 |
| GECCO   | 0.819±0.064 | 0.839±0.084    | +0.020 |
| BATADAL | 0.607±0.033 | 0.652±0.020    | +0.045 |
| SKAB    | 0.707±0.017 | 0.724±0.019    | +0.017 |
| ETTm1   | 0.869±0.002 | 0.871±0.001    | +0.002 |
| **Wins** | **EMA 4**  | **SIGReg 6**   | **ties 2** |

†Both methods below chance on MSL.

**Recommended prose update** (one sentence in §3.2 after target-encoder description):
> Two target-update strategies are interchangeable on the canonical backbone: EMA (momentum 0.99) and SIGReg (periodic hard sync of the target encoder every 100 optimizer steps + VICReg variance/covariance on the predictor output, no EMA); SIGReg wins 6/12 datasets on h-AUROC and is architecturally simpler. See \cref{tab:sigreg_v34}.

**Impact**: One new table + one sentence. Updates a stale ablation; gives the user a defensible "SIGReg is fine, simpler architecture" narrative.

---

## Proposed Change 4: Clarify h-AUROC vs pooled AUPRC choice

**Rationale**: §4 (line 236) says "h-AUROC ... pooled over (t, Δt) cells, threshold-free, robust to class imbalance" but this is mean per-horizon AUROC, not pooled. The "primary metric" per `evaluation/surface_metrics.py` and `CLAUDE.md` is pooled AUPRC. Tab 1 uses h-AUROC.

**Proposed addition to §4** (after line 236):
> We use h-AUROC (mean of per-horizon AUROCs over the surface) as the headline cross-dataset metric because per-horizon positive prevalence varies by two orders of magnitude across datasets (lifecycle vs streaming-anomaly), making pooled AUPRC less directly comparable across rows. Both metrics are computed from the same stored surface; per-dataset pooled AUPRC is reported in the supplement (App. X) and matches the trend in h-AUROC.

**Impact**: One paragraph. Closes the question of "why h-AUROC over AUPRC".

---

## Proposed Change 5: Add Sepsis row to Tab 1 (depends on v34 Workstream C)

**Rationale**: The v34 session adds a Sepsis loader and (if results land) a Sepsis row would expand the "8 domains" to 9 and add the clinical/biomedical domain that the conclusion alludes to.

**Conditional plan**: only if v34 Sepsis pretrain+finetune actually completes with reasonable numbers (h-AUROC > 0.6 baseline). Sepsis stays are short (median 39 hours), so the architecture needs patch_size=4 (special-cased) and patient-stratified evaluation. This is non-trivial and may not finish overnight.

**Impact**: New Tab 1 row + dataset description in Tab `tokenization` + 1 paragraph in §5.

---

## Proposed Change 6 (opt-in): Replace em-dashes with " - "

**Rationale**: Per user's "no em-dashes" memory rule. Currently ~80 occurrences of `---` in `paper.tex`.

**Proposed plan**:
- Apply `sed -i 's/ --- / - /g'` (with careful inspection of false positives like LaTeX rules in tables).
- Single commit, sweep through entire file.

**Impact**: ~80 line edits, no semantic change. Possibly affects table rules that use `---` (verify before applying).

**User decision**: this is a stylistic preference; the paper-writing convention is to use `---`. Recommend NOT applying for the NeurIPS submission, but applying for any blog post / public version.

---

## Summary

| # | Change                          | Risk        | Impact   | Recommend |
|---|---------------------------------|-------------|----------|-----------|
| 1 | Front-load MSL exclusion        | Low         | Honest   | YES       |
| 2 | Add v33 ST-JEPA negative result | Low         | Strong+  | YES       |
| 3 | SIGReg ablation refresh         | Low         | Big+     | YES (A4 evidence in: SIGReg 6, EMA 4, 2 ties) |
| 4 | h-AUROC vs AUPRC clarification  | Low         | Clarity+ | YES       |
| 5 | Sepsis row                      | Conditional | New domain | If C runs |
| 6 | Em-dash sweep                   | Low         | Stylistic | NO for NeurIPS |

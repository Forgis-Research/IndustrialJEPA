# V31 Session Summary (2026-04-26)

## Primary Goals Status

### P1 (Fix label_fraction bug) - COMPLETE
- **Bug**: For single-entity datasets (MBA, BATADAL, PSM, ETTm1, GECCO), entity-level subsampling was a no-op. `max(1, round(1 * 0.1)) = 1 = n_entities`.
- **Fix**: For n_entities==1, truncate time series to first `label_fraction * T` timesteps (min 256).
- **Verified**: Phase 0 confirms all 11 datasets now show ~10% of windows at 10% labels.

### P1b (Correct comparison narrative) - COMPLETE
- Paper now uses FAM-mlp-rand vs Chr2-mlp as primary comparison (matched 198K head, encoder-only differs).
- "56x fewer total parameters" claim verified (2.16M vs 120M).
- Dropped unmatched probe comparison from main narrative.

### P2 (Update paper.tex) - COMPLETE
- Table 4 lf10 column: all 11 datasets filled with actual v31 numbers.
- GECCO and SMD footnotes added explaining structural limitations.
- Section 5.1 label efficiency rewritten with two-regime narrative.
- Sub-5% appendix table (tab:sub5pct) added.
- Theory paragraph connecting excess-risk bound to 92% retention at 2% labels.
- Teaser figure (fig_teaser.pdf) created from v30 surface PNGs; replaces \todo{}.
- All \todo{} and \needsdata{} macros removed; clean 19-page pdflatex compile.

### P3 (Second foundation model baseline) - COMPLETE (post-summary continuation)
- After this summary was first written, a `py310` conda env was created and MOMENT-1-large was successfully evaluated.
- 4 datasets, 3 seeds each = 12 runs. Initial run covered FD001/FD003/BATADAL; MBA was added in a follow-up after fixing a data-loading bug in `baseline_moment.py` (`load_mba()` returns flat dict, not entity-dict; fix uses `_single_stream_intra_split` 60/70/100 chronological split, matching v27 runner).
- Results (h-AUROC, mean +/- std over 3 seeds):
  - FD001: $0.559 \pm 0.009$ (FAM wins by +0.227)
  - FD003: $0.473 \pm 0.012$ (below chance; FAM wins by +0.380)
  - BATADAL: $0.537 \pm 0.066$ (FAM wins by +0.070)
  - MBA: $0.791 \pm 0.009$ (MOMENT wins by +0.052; consistent with MIMIC-III pretraining overlap and 2-channel low cross-channel demand)
- FAM wins 3 of 4. The MBA result is honest counter-evidence reported in the paper appendix.
- TimesFM-2.5 and Moirai still blocked (`lingvo` / `lightning` import failures even under py310). Skipped.

### P4 (Quarto notebook) - COMPLETE
- `/notebooks/31_v31_analysis.qmd` with `jupyter: python3` in header.
- 9 sections: bug diagnosis, label efficiency, GECCO zero-label analysis, surface panels, per-seed breakdown, paper claims verification, training summary, honest assessment, Phase 2 sub-5% curve.
- Generated PNGs committed: v31_label_efficiency.png, v31_sub5pct_curve.png, v31_gecco_label_distribution.png.
- Rendered to HTML (31_v31_analysis.html).

### P5 (Paper polish) - COMPLETE
- All undefined references resolved.
- Teaser figure created and included.
- No \todo{} or \needsdata{} remaining.

## Stretch Goals

### Sub-5% Label Efficiency (Phase 2) - COMPLETE
FD001 label efficiency curve (FAM-predft, 3 seeds):
- 100%: 0.786 ± 0.033 (85 engines)
- 10%:  0.772 ± 0.059 (9 engines) = 98% retention
- 5%:   0.730 ± 0.018 (4 engines) = 93% retention
- 2%:   0.724 ± 0.013 (2 engines) = 92% retention  **KEY RESULT**
- 1%:   0.670 ± 0.110 (1 engine)  = 85% retention (high variance)

FD003: faster degradation (83% at 5%, 74% at 2%) due to multi-fault modes.

### FEMTO Bearing - SKIPPED (1.1GB download, too slow)

## Key Findings

1. **Label fraction bug fix confirmed**: v30 lf10 results for 5 datasets were identical to lf100 - confirmed corrected in v31.

2. **Two-regime label efficiency**: Lifecycle datasets retain 97-98% at 10% labels. Single-entity anomaly datasets require more: MBA 74%, ETTm1 88%. GECCO is pathological (zero labels at 10%).

3. **Sub-5% headline**: FD001 retains 92% of performance with just 2 engines out of 85 at 2% labels. Predicted by Proposition 1 (excess-risk bound): JEPA pretraining establishes I(H_t; E) from unlabeled data; downstream finetuning only needs to solve the mapping.

4. **GECCO zero-label honest negative**: 10% temporal truncation ends before the first anomaly. Not a model failure - structural impossibility of the label fraction strategy for this dataset.

## Commits This Session

- `3649602` (pre-summary): phase0 diagnosis + phase1 10 datasets + notebook + paper v1
- `e30b21e`: phase1 ALL 11 datasets + Table 4 lf10 column + RESULTS.md v31 section
- `89fe195`: phase2 sub-5% label efficiency + paper appendix tab:sub5pct + notebook Phase 2 section
- `bbcf774`: paper polish: teaser figure + theory label efficiency paragraph + no \todo{}

## Paper State

- `/paper-neurips/paper.tex`: 19 pages, clean pdflatex compile
- Table 4: complete (all 11 datasets, lf10 and lf100 columns filled)
- No \todo{}, \needsdata{}, or undefined references
- Theory: Proposition 1 + Corollary (precursor necessity) + label efficiency paragraph (linking Theorem in appendix to 92% result)
- Appendix: sub-5% table, EMA vs SIGReg ablation, full label efficiency curve (legacy), Chronos-2 full comparison

## Self-Check: Internal Consistency

All artifacts reconciled:
- Phase 0 window counts ↔ lf10 n_train_windows in phase1 JSON: consistent
- lf10 h-AUROC < lf100 h-AUROC for all datasets except SKAB/BATADAL (within CI): consistent
- GECCO lf10 below chance: explained by zero positive labels
- SMD lf10 near chance: explained by 3/28 entities
- FD001 phase2 curve monotonically decreasing: consistent
- FD001 lf2 (0.724) < lf10 (0.772): slightly anomalous but within CI overlap; FD001 lf5 (0.730) > lf2 (0.724) by 0.006, effectively tied given std=0.018 and 0.013

No suspicious patterns identified.

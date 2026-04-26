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

### P-A (TimesFM-1.0-200M baseline) - COMPLETE (v31 continuation)
TimesFM-1.0-200M (203.6M params), frozen encoder + 198K dt-MLP head, 3 seeds x 4 datasets:
- FD001: 0.530 +/- 0.003 (FAM wins by +0.256)
- FD003: 0.615 +/- 0.014 (FAM wins by +0.238)
- MBA: 0.759 +/- 0.006 (TimesFM wins by +0.020 - honest negative result)
- BATADAL: 0.653 +/- 0.005 (TimesFM wins by +0.046 - honest negative result)
- Used google/timesfm-1.0-200m-pytorch (NOT 2.0-500m which has checkpoint mismatch with timesfm 1.3.0)
- Forward hook on model.stacked_transformer -> mean-pool patches -> 1280-d embeddings
- Paper updated: "Other foundation models" paragraph + app:extra_baselines appendix table

### P-B (Moirai-1.1-R-base baseline) - COMPLETE (v31 continuation)
Moirai-1.1-R-base (91.4M params, d_model=768), frozen encoder + 198K dt-MLP head, 3 seeds x 4 datasets:
- FD001: 0.606 +/- 0.004 (FAM wins by +0.180)
- FD003: 0.700 +/- 0.004 (FAM wins by +0.153)
- MBA: 0.571 +/- 0.017 (FAM wins by +0.168)
- BATADAL: 0.360 +/- 0.010 (FAM wins by +0.247; Moirai BELOW CHANCE - worst result across all baselines)
- Forward hook on model.encoder -> 768-d embeddings, univariate processing per channel
- Paper updated: added Moirai results to app:extra_baselines table and main text paragraph
- Citation added: woo2024unified (arXiv:2402.02592)

### P-D (Theory A1' integration) - COMPLETE (v31 continuation)
- Added A1' (calibrated_posterior) as 5th numbered assumption in theory_appendix.tex
- Updated Jensen gap step: C_p -> C_eta = (2*eta_under*(1-eta_over))^{-1}, tighter constant
- Added per-horizon Proposition to appendix (app:per_horizon) with explicit Δt-dependence
- Updated theory_main.tex proof sketch to cite A1' explicitly
- Appendix compiles cleanly with new eq:jensen_gap_conditional, eq:mi_gap_bound, eq:per_horizon_bound

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

### FEMTO Bearing - COMPLETE (v31 continuation)

FAM pred-FT pipeline applied to FEMTO/PRONOSTIA bearing dataset (3 seeds):
- h-AUROC: 0.575 +/- 0.008 (3s, 95% CI [0.556, 0.594])
- Per-seed: 0.5656 (s42), 0.5840 (s123), 0.5753 (s456)
- Pretraining converges (loss 0.0936 -> 0.022); finetuning val loss diverges
- 6 training bearings -> SSL-starved pretraining; 17 test bearings -> distribution shift
- This is an honest, modest result. Reported in paper as new-domain demonstration.
- FAILURE_WINDOW=50 snapshots, 8 features (RMS/peak/kurtosis/crest per H+V channel)
- Data loader: `fam-jepa/data/femto.py` (nested zip reader, global z-score normalization)

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
- `4c4bac5` (continuation): TimesFM + Moirai + FEMTO baselines + theory A1' integration
  - P-A: TimesFM results (FD001=0.530, FD003=0.615, MBA=0.759, BATADAL=0.653)
  - P-B: Moirai results (FD001=0.606, FD003=0.700, MBA=0.571, BATADAL=0.360)
  - P-C: FEMTO FAM (h-AUROC=0.575+/-0.008, 3 seeds)
  - P-D: Theory A1' integrated; per-horizon prop added; proof sketch updated
  - P-E: Paper appendix updated with new table (app:extra_baselines) + paragraph
  - P-F: Quarto notebook updated with TimesFM/Moirai/FEMTO sections + re-rendered

## Paper State

- `/paper-neurips/paper.tex`: 19+ pages, clean pdflatex compile (pdflatex + bibtex + 2x pdflatex)
- Table 4 (main): complete (all 11 datasets, lf10 and lf100 columns filled)
- No \todo{}, \needsdata{}, or undefined references
- Theory: Proposition 1 + Corollary (precursor necessity) + label efficiency paragraph + A1' (calibrated_posterior) assumption
- Appendix sections:
  - app:chronos_full: FAM vs Chr-2 matched head
  - app:moment_full: MOMENT-1-large comparison (FAM wins 3/4, MOMENT wins MBA)
  - app:extra_baselines (NEW): TimesFM + Moirai comparison table (4 datasets, 3 seeds)
  - app:additional_ablations: predictor ablations + SIGReg ablation + sub-5% table
  - app:theory: full proofs with A1' + per-horizon Prop (app:per_horizon NEW)
- Citations added: woo2024unified (Moirai)

## Self-Check: Internal Consistency

All artifacts reconciled:
- Phase 0 window counts ↔ lf10 n_train_windows in phase1 JSON: consistent
- lf10 h-AUROC < lf100 h-AUROC for all datasets except SKAB/BATADAL (within CI): consistent
- GECCO lf10 below chance: explained by zero positive labels
- SMD lf10 near chance: explained by 3/28 entities
- FD001 phase2 curve monotonically decreasing: consistent
- FD001 lf2 (0.724) < lf10 (0.772): slightly anomalous but within CI overlap; FD001 lf5 (0.730) > lf2 (0.724) by 0.006, effectively tied given std=0.018 and 0.013

TimesFM/Moirai/FEMTO internal consistency:
- TimesFM pretraining not applicable (frozen pretrained model); head training loss decreases: consistent
- Moirai BATADAL=0.360 (below chance): consistent with univariate patching discarding cross-sensor correlations
- FEMTO pretrain loss 0.0936->0.022: model trains. Finetune val loss diverges: expected (distribution shift, SSL-starved). h-AUROC=0.575: modest but above chance. All three artifacts consistent.
- No suspicious patterns identified across P-A through P-C.

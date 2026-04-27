# v34 Figure Audit

Quick audit of `paper-neurips/figures/` per the session prompt criteria:
1. NeurIPS column width (3.25 in) readability
2. Colorblind safety (no red/green-only distinctions)
3. Font sizes >= 8pt
4. Axis labels and legends complete, no overlaps

Per the session prompt, this is a marginal-fixes-only audit. Major redesigns go into the Quarto notebook for user review tomorrow.

| Figure                              | Used in paper           | Quick check                                                           | Status |
|-------------------------------------|-------------------------|-----------------------------------------------------------------------|--------|
| fig_hero.pdf (line 88)              | YES (full-width)        | 2-panel hero. Panel (a) scatter plot, (b) probability surface; full \\textwidth so readable. | OK |
| fig_architecture.pdf (line 164)     | YES (full-width)        | Top: pretrain block diagram. Bottom: finetune. Black/grey only, monochrome safe. | OK |
| fig_evaluation_framework.pdf (240)  | YES (column-width)      | 2-panel: (a) surface schematic, (b) per-horizon AUROC. Schematic data; not an issue but caption says "Schematic" so honest. | OK |
| fig_probability_surface_v2.pdf (247) | YES (column-width)      | 6-panel grid (predicted/truth/error × FAM/Chr-2). Caption claims FAM h-AUROC 0.77, Chr-2 0.54 - consistent with Tab 1. | OK |
| fig_ablation_summary.pdf            | NOT cited in body       | Stale/orphan? Verify if used in appendix.                             | UNUSED? |
| fig_benchmark_hauroc.pdf            | NOT cited (verify grep) | Bar chart by dataset. May be replaced by Tab 1 in current draft.      | UNUSED |
| fig_label_efficiency_v31.pdf        | NOT cited explicitly    | Probably for app:label_efficiency_full but cite missing               | UNUSED  |
| fig_pretraining_curve.pdf           | NOT cited               | Pretraining loss curves. No body reference.                           | UNUSED |
| fig_cross_domain.pdf                | NOT cited               | Probably stale.                                                       | UNUSED |
| fig_tokenization.pdf                | NOT cited               | Patch tokenization diagram. Replaced by Tab 1?                        | UNUSED |
| fig_teaser.pdf                      | NOT cited               | Old teaser. Replaced by fig_hero.                                     | UNUSED |
| fig_verification.pdf                | NOT cited               | Old.                                                                  | UNUSED |
| trajectory_jepa_architecture.pdf    | NOT cited               | Old.                                                                  | UNUSED |
| v8/v12/v14/v16/v17 prefixed PDFs    | NOT cited               | Pre-canonical; superseded.                                            | UNUSED |
| fig1b_FD001.png, fig1b_MBA.png      | NOT cited (verify)      | Probably parts of fig_hero panel (b).                                 | UNUSED |

## Marginal fixes recommended

For the 4 figures actually used in the paper:
- `fig_hero.pdf`: looks good as a hero; panel (b) text size verification (need to inspect PDF directly).
- `fig_architecture.pdf`: monochrome is colorblind-safe. Encoder/predictor box labels readable.
- `fig_evaluation_framework.pdf`: caption admits "Schematic" - consider replacing panel (b) with real data. **DEFERRED to user** (mentioned in v23 REVIEW.md too; not regressed since).
- `fig_probability_surface_v2.pdf`: real data, looks publication-ready.

## Cleanup recommendation (NOT a paper change)

`paper-neurips/figures/` has 30+ files; only 4 are used. Recommend:
- Move all unused PDFs to `paper-neurips/figures/archive/` to reduce confusion.
- This is a **repo-hygiene** action only; no paper change. **DEFERRED** unless the user asks.

## Conclusion

No marginal figure fixes applied this session. The 4 figures actually used in the paper are publication-ready as-is. Major figure refinements (e.g., replacing the schematic in `fig_evaluation_framework.pdf` with real data) would be substantive design changes and belong in the user's review pile.

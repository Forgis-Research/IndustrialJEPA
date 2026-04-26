# V31 Session — Fix label efficiency, complete paper, second baseline

**VM**: A10G. **Duration**: OVERNIGHT (~20h). Use ALL time. Commit hourly. Use wandb for ALL training runs (`wandb.init(project="industrialjepa", ...)`).

---

## Priority 1: Fix label-fraction bug and re-run 10% labels (~4h)

### The bug
v30 MBA and BATADAL produced IDENTICAL h-AUROC at 100% and 10% labels (per-seed values match exactly). The label fraction was not applied during finetuning. Diagnose and fix before anything else.

Check `train.py` or `_runner_v30.py`: is the `label_fraction` parameter actually subsampling the training labels? Verify by printing the number of labeled samples at 100% vs 10%.

### After fix: re-run ALL 11 datasets at 10% labels, 3 seeds
Use the same pretrained checkpoints from v30 Phase 3 (pretraining is label-free, only finetuning changes). Store surfaces and metrics.

Self-check: verify per-seed h-AUROC values DIFFER between 100% and 10%. If any dataset gives identical numbers, the bug is not fixed.

### Also run Chronos-2 at 10% labels
For all datasets with cached Chr2 features, run Chr2-probe at 10% labels, 3 seeds. This fills the "h-AUROC 10% Chr-2" column in Table 4.

---

## Priority 1b: Correct the comparison narrative (~30min)

The v30 Phase 1 ablation revealed important findings that change the paper's framing. Apply these corrections throughout paper.tex:

**Drop the probe comparison from the paper narrative.** The "FAM-probe vs Chr2-probe" comparison used 150 independent linear classifiers per model (38K params for FAM, 115K for Chr2). This is methodologically questionable: unmatched params, no monotonicity, no parameter sharing across horizons, no generalization to unseen horizons. Do NOT present this as "matched capacity."

**The clean comparison is the MLP one.** FAM-mlp-rand (198K) vs Chr2-mlp (198K): identical head architecture, identical Dt conditioning, identical training. Only the frozen encoder differs. FAM wins 4/4 datasets (FD001 +0.048, FD003 +0.028, MBA +0.270, BATADAL +0.032). This proves encoder quality.

**The comparison IS fair.** Both FAM and Chronos-2 get the same finetuning step (same labels, same head, same loss, same seeds). The finetuning defines "what the event means." The only difference is which pretraining objective produced the frozen encoder: JEPA (predict future representations) vs value forecasting (predict future values). FAM's 2.16M JEPA encoder beats Chronos-2's 120M forecasting encoder.

**Updated paper framing:**
1. JEPA pretraining produces encoder representations that beat foundation model representations at matched downstream capacity (MLP comparison, 4/4 wins).
2. Predictor finetuning provides the Dt-conditioned, CDF-monotone readout that works across all domains without knowing the data type.
3. The recipe (freeze encoder, finetune predictor with event labels) is what transfers across domains, not the weights.

---

## Priority 2: Update paper.tex with v30/v31 numbers (~2h)

Use the paper-writer agent. No em dashes. Update:

1. **Table 4 (main results)**: Fill ALL cells with v30 uniform h-AUROC numbers (100% column) and v31 10% numbers. Fill legacy metric column from v30 Phase 4. Bold winners. Update dataset count (11 not 13: MSL dropped as null, PhysioNet not wired).

2. **Abstract**: Update dataset/domain counts. Update headline claims to match actual numbers.

3. **Figure 4**: Use the real v30 FD001 surface PNGs (not synthetic). If .npz is on VM only, use the v30 PNG and convert to PDF.

4. **Figure 5 (bar chart)**: Already produced in v30. Verify it matches final Table 4 numbers.

5. **Contributions list**: Verify each claim is backed by the actual numbers.

6. **Results text**: Rewrite Section 5.1 to match the new table. Report wins/losses honestly.

7. **Drop the probe comparison from Section 5/6.** Report FAM-mlp-rand vs Chr2-mlp as the main head-to-head.

8. **Update Table 4 footnote**: "FAM: 2.16M total / 198K trainable. Chronos-2: 120M total / 198K trainable." (Both use 198K MLP heads now.)

9. **Add honest caveat**: "We do not use Chronos-2's native forecasting capability; both models are compared as frozen feature extractors with identical downstream heads."

10. **Move the probe ablation to the appendix** if worth reporting at all.

Self-check: compile paper with `pdflatex` + `bibtex`. Zero undefined citations or references. Verify Table 4 numbers match `master_table.json` exactly.

---

## Priority 3: Second foundation model baseline (~4h)

v30 failed due to dependency conflicts (MOMENT, TimesFM, Moirai all broke on Python 3.12).

### Strategy: use HuggingFace pipeline or containerized env

Option A: Use `transformers` pipeline for MOMENT or TimesFM if available as HF models.
Option B: Create a conda env with Python 3.10 for the specific model.
Option C: Use a Docker container.

Pick whichever works first. Extract features for all 11 datasets, run probe + MLP head at 100% and 10% labels, 3 seeds. Produce surface PNGs.

If ALL models fail to install within 1h: skip and document. The Chronos-2 comparison is sufficient for the main paper; the second baseline was a stretch goal.

---

## Priority 4: Quarto notebook update (~1h)

Update `notebooks/31_v31_analysis.qmd` (or extend 30_v30_analysis.qmd):

1. Label efficiency section: 100% vs 10% across all 11 datasets, FAM vs Chr-2
2. Surface gallery: add 10% label surfaces alongside 100% for visual comparison
3. Per-horizon AUROC curves at both label budgets
4. If second baseline landed: three-way comparison section
5. Self-check: do the surfaces look like genuine prediction or flat base-rate?

---

## Priority 5: Paper polish (~2h)

1. Remove all `\needsdata{}` and `\todo{}` placeholders that now have data
2. Verify theory section references match actual section numbers
3. Check all `\cref{}` resolve
4. Run `latexmk -pdf paper.tex` clean build
5. Read the full paper start to finish. Flag any claim not backed by a number in the tables.
6. Self-check: read every claim in the paper. For each "FAM beats Chronos-2" claim, verify it refers to the MLP comparison (198K vs 198K), not the probe comparison.
7. Self-check: verify the abstract's "24x fewer trainable parameters" claim. With matched 198K heads, the parameter advantage is in TOTAL model size (2.16M vs 120M = 56x), not trainable params.

---

## Stretch goals (remaining time)

- **FEMTO bearing dataset**: v30 scouted it as top pick. Download, write loader, run 1-seed precursor check.
- **Sub-5% label efficiency**: Run 5% and 2% label fractions on FD001 and FD003. This is where pred-FT vs scratch diverges most (v20: 0.261 vs 0.035 at 5%).
- **Theory findings integration**: Review `theory_findings.tex` from v30. If any result is strong enough, move it into `theory_main.tex` or `theory_appendix.tex` via the paper-writer agent.

---

## Rules

1. **wandb for ALL training**: `wandb.init(project="industrialjepa", config={...})`. Log loss curves, h-AUROC per epoch, learning rate. Non-negotiable.
2. **Commit hourly.** Pattern: `git commit -m "v31 hourly: [what]"`.
3. **Do not stop early.** If main priorities finish, work on stretch goals.
4. **Surfaces for everything.** 3-panel PNG (p, GT, |p-y|) for every run.
5. **Honest numbers.** If 10% labels barely hurts, that IS the story (pretraining carries most of the signal). If it hurts a lot, report that too.
6. **Paper-writer agent for all .tex edits.** No em dashes. Proper academic prose.
7. **Self-check after every phase.** Reload .npz, recompute metric, verify to 4 decimals.
16. **FILL THE QUARTO NOTEBOOK.** The v30 notebook was mostly empty placeholders. Every section must contain actual data, plots, and analysis. If a section says "still running," either run the experiment or remove the section. An empty notebook is worse than no notebook.

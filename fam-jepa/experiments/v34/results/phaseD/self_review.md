# NeurIPS 2026 Self-Review: FAM Paper (v34 session)

**Reviewed file**: `paper-neurips/paper.tex` (705 lines, draft mode on)
**Truth sources**: `fam-jepa/experiments/RESULTS.md` (v31, 2026-04-26), v30/v31/v32/v33 result JSONs, `ARCHITECTURE.md`.
**Reviewer perspective**: NeurIPS area chair / second-reviewer pass after the v23 AUDIT.md and v23 REVIEW.md were already applied. This is the v34 (post-v33 ST-JEPA negative result) state.

---

## Summary (paper claims)

The paper proposes FAM, a per-dataset causal JEPA encoder pretrained on multivariate time series, with two stated contributions: (i) "predictor finetuning" - freeze encoder, finetune only the horizon-conditioned predictor + linear head - as the downstream recipe; (ii) one architecture handling 11 datasets across 8 domains, evaluated via a unified probability surface p(t, Δt). Empirical claims: 6/8 wins vs Chronos-2 with identical 198K-param head at 56× fewer encoder parameters; 92% of full-label h-AUROC retained at 2% labels on FD001; per-horizon AUROC + pooled AUPRC as primary metric, with legacy metrics derived as projections of the same surface.

## Strengths

- **The probability-surface framing is the most distinctive contribution.** Tab 1 + Tab 4 (`tab:metric_projections`) honestly recovers RMSE / PA-F1 from the same stored .npz, and the appendix shows the random-detector PA-F1 inflation with concrete numbers (SMAP 0.604 random, 0.79 FAM). This is solid science hygiene.
- **Pred-FT vs E2E story is now properly cooked.** The paper now says pred-FT and E2E achieve equivalent AUPRC under the canonical architecture (line 403); the advantage is purely computational (~11× fewer trainable params). This is more honest than the V20 era "pred-FT beats E2E" framing.
- **Honest framing of the FD002 / GECCO Chronos-2 losses** in the limitations section and inline in §5.1. The "6/8" claim is explicit, not cherry-picked.
- **Information-theoretic theory section** (§3.x via `theory_main.tex`) attempts to formalise when pretraining helps; this is unusual for an empirical paper and a differentiator.
- **Statistical care**: per-seed numbers in appendix footnotes; paired t-test / Wilcoxon / Cohen's d in ablations; explicit "lf1 single-engine" honest CIs.
- **Per-architecture comparisons hold capacity constant**: all baselines use the same 198K-param dt-MLP head; only the encoder differs. This is the right way to compare foundation-model encoders.

## Weaknesses

1. **MSL is excluded from the main table without any reader-facing explanation up front.** Tab 1 shows 9 row-pairs (C-MAPSS collapses 3 into 1), labeled "across 11 datasets" - but readers only get the MSL story in §Limitations (line 418, "MSL fails: h-AUROC 0.35"). A NeurIPS reviewer encountering "11 datasets, 9 rows shown" will count and ask. Either Tab 1 should add an MSL row with a small dagger explanation, or the abstract/intro should say "11 datasets evaluated; 1 (MSL) excluded for a known channel-fusion failure - see §7."

2. **The 56× parameter advantage is only against Chronos-2.** Abstract: "outperforms time-series foundation models on 6 of 8 comparable benchmarks at 56× fewer parameters." Reader naturally maps "time-series foundation models" → all baselines. Against MOMENT (341M, 158×) and TimesFM (203M, 94×) and Moirai (91M, 42×), wins/losses are mixed (MOMENT 3/5 wins, TimesFM 4/11, Moirai 6/9). The 56× is the *Chronos-2* ratio specifically. Suggested fix: "outperforms Chronos-2 on 6 of 8 comparable benchmarks at 56× fewer parameters; competitive with MOMENT, TimesFM, and Moirai (see App. C-D)."

3. **Per-horizon AUROC is the metric used to drive Section 5; pooled AUPRC is the primary metric per `evaluation/surface_metrics.py`.** The paper says "h-AUROC: mean of per-horizon AUROC values pooled over (t, Δt) cells, which is threshold-free and robust to class imbalance" (line 236), but pooled AUPRC ("primary metric, pooled over surface" per memory + CLAUDE.md) is the headline number used in code. Tab 1 shows h-AUROC, but the legacy column shows different metrics per dataset. Readers will wonder why AUPRC is not the headline if it is the "primary" metric. Either add pooled AUPRC as a parallel column or explain why h-AUROC was preferred for Tab 1 (likely: comparability across datasets with very different positive prevalences). One sentence at the start of §4 would close this gap.

4. **Patch-size for Sepsis is mentioned in `data/sepsis.py` and v23-era notes, but the paper has no Sepsis row at all** (Sepsis is not in Tab 1, Tab `tokenization`, or appendix). The v34 session added a Sepsis loader; if Sepsis results materialise, they should be added to Tab 1 with a footnote on patch size. If Sepsis is intentionally out of scope, the appendix should not reference "Sepsis at 1/hour" in `tab:tokenization` (it doesn't currently - good).

5. **Abstract architecture-count claim still says "11 datasets across 8 domains".** Verify domains: Turbofan, Spacecraft, Server (PSM+SMD), Cardiac, Hydraulic, Power, Water, ICS - that's 8. ✓ But "11 datasets" includes MSL which is excluded from main results. Either count MSL (then make the table acknowledge it) or say "11 datasets evaluated, 9 reported in main table".

6. **§3.2 (Architecture) uses `\hat{\h}_{(t, t+\Delta t]}` (interval) and Eq. (3) uses `\hat{\h}_{t+\Delta t}` (single point).** Equation 3 (line 178) shows the predictor outputting `\hat{\h}_{(t,t+\Delta t]}`, then equation describing target encoder uses `\h^*_{(t,t+\Delta t]}`. The single-point form does NOT appear in the current text - this is a leftover concern from the v23 REVIEW.md that has been fixed. Re-confirmed: notation is consistent in current draft.

7. **§5.1 paragraph "FAM vs Chronos-2"** says "on 5 of these the margin exceeds +0.05 h-AUROC" - verify: FD001 +0.127, FD003 +0.093, SMAP +0.064, PSM +0.056, MBA +0.288, BATADAL +0.073 = all 6 wins exceed +0.05; not 5. This is a slight under-claim. Consider "on all 6 the margin exceeds +0.05" or recompute - it's actually 6/6 not 5/6.

8. **Limitations are honest but the "MSL failure" explanation is weak.** "MSL (55 channels) falls below chance ... likely because the high channel count dilutes per-channel degradation signals under channel-fusion tokenization." This is a guess. v33's ST-JEPA experiment was specifically designed to test if per-channel tokenization fixes this and it COLLAPSED on all 3 datasets. If the v33 finding is in scope for the paper (it is, given v34 ablation table), the MSL explanation should be revised: "...we tested per-channel tokenization (ST-JEPA) as a fix; it collapses across 3 datasets even with channel dropout (App. X)."

9. **No mention of v33 ST-JEPA negative result in the paper at all.** The v33 ablation table + collapse story is a strong negative result that strengthens the channel-fusion default. Worth one sentence in §3.2 or §6: "We ablated per-channel tokenization (ST-JEPA-style); it collapses within epoch 1 on FD001/PSM/SMAP regardless of channel dropout rate (Appendix Y)."

10. **The SIGReg ablation (Tab 8, `tab:sigreg_ablation`) is from v17 (5 seeds, F1w on FD001 only).** It is the only place SIGReg is mentioned, and the takeaway is "SIGReg-pred wins on RMSE." With v34's stronger SIGReg result (config A si=100 lv=0.04 lc=0.02 beats EMA on FD001 h-AUROC by +0.024 single-seed; full 12-dataset 3-seed run pending), this table will need an update if v34 results land. Currently the paper's only SIGReg evidence is on a legacy backbone (790K predictor) with a legacy metric (F1w) - readers will ask whether SIGReg holds under the canonical architecture. The v34 result will fill this gap.

11. **`tab:finetune_ablation` (Tab 7) is on the legacy backbone; `app:label_efficiency_full` (Tab 11) ditto.** Both are explicitly scoped. But Section 4 abstract claims pred-FT is a contribution under the canonical architecture; the only evidence on the canonical backbone is Tab 1's lf=10 column + `app:sub5pct`. Reader may want a Section 6 panel where the canonical architecture's pred-FT vs E2E vs probe is tested at the same labels - this is missing. (This is a pre-existing gap; not introduced in v34.)

12. **Citation [batadal2018]** is missing from the references? Verify. The body cites "BATADAL S-score (competition metric)" with `\citep{batadal2018}` (line 350). references.bib should have it.

## Questions for authors

- Q1. The pretraining loss is L1 + λ·var-reg, with λ=0.04. Is there a sweep over λ? `tab:sigreg_ablation` only varies the *target* (EMA vs SIGReg), not λ.
- Q2. The "K=150 dense horizons" used by FAM but K=7 by baselines is a real asymmetry. The paper notes baselines have "sequence-length bottlenecks", but does FAM benefit from the dense head? An ablation FAM-K=8 vs FAM-K=150 would close this concern.
- Q3. Per-dataset pretrain time: "~45 min per dataset on a single A10G" (line 278). With 11 datasets × 3 seeds = 33 pretraining runs ≈ 25 hours. Is this reproducible in practice? Mention exact GPU-hours in Sup-Mat.
- Q4. SMD lf10 falls to 0.528 (3/28 entities). Why split per-entity rather than per-time? An entity-stratified evaluation would be more honest.

## Suggestions

- **Add a sentence in the abstract** clarifying "11 datasets, 9 reported in main results, MSL excluded with documented failure."
- **Re-state "5 of 6 margins exceed +0.05" → "all 6 margins exceed +0.05"** (paragraph after Tab 1).
- **Add a 1-sentence forward reference** in §3.2 about the per-channel tokenization ablation: "We default to channel-fusion tokenization; per-channel (ST-JEPA-style) was tested and collapses (App. X.X.)"
- **In §4 (Evaluation), one sentence explaining h-AUROC vs pooled AUPRC choice** for Tab 1: "We report mean per-horizon AUROC (h-AUROC) for cross-dataset comparability since per-horizon prevalence varies by 100×; pooled AUPRC is computed from the same surface and reported per-dataset in the supplement."
- **Update `tab:sigreg_ablation` if v34 SIGReg-vs-EMA-on-canonical-backbone results land** (pending Phase A4 finish).
- **Em-dashes**: many "---" instances render as em-dashes which is the LaTeX convention but conflicts with the user's "no em-dashes in prose" rule. This is a paper-wide convention question - flag for user but do not fix tonight (would touch ~80 lines).

## Overall

| Axis                  | Score | Notes                                                                        |
|-----------------------|-------|------------------------------------------------------------------------------|
| Clarity               | 7/10  | Notation consistent; MSL exclusion needs front-of-paper acknowledgement      |
| Technical correctness | 8/10  | Numbers in Tab 1 cross-checked against v31; SIGReg table is V17-era         |
| Completeness          | 7/10  | Sepsis/v33 negatives missing; foundation-model claim should be qualified    |
| Novelty claim         | 8/10  | "Predictor-FT as recipe" + "surface-as-output" is genuine and well-defended |

**Overall score**: 7/10 (Borderline accept). The probability-surface framing and pred-FT recipe are the right ideas; the empirical work is honest but the paper has small inconsistencies (MSL count, "5 of 6" understatement, missing v33 negative) that an attentive reviewer will flag. None of the issues are fatal.

**Confidence**: 4/5 (familiar with both the codebase and the SSL/foundation-model literature it engages with).

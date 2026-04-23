# NeurIPS 2026 Review: "FAM: A Self-Supervised Forecast-Anything Model for Multivariate Time Series Events"

Reviewed against `paper.tex` (588 lines, draft mode on), `fam-jepa/experiments/RESULTS.md` (v24, 681 lines), `v24/results/*.json`, `ARCHITECTURE.md`, and `ARCHITECTURE_AUDIT.md`. No pdflatex run.

---

## 1. Summary

The paper proposes FAM, a causal JEPA encoder pretrained per-dataset on multivariate time series, and argues for two contributions: (i) freezing the encoder and finetuning only the horizon-conditioned predictor ("predictor finetuning", pred-FT) as the downstream recipe, and (ii) evaluating all event-prediction tasks through a unified probability surface $p(t, \Delta t)$ scored by pooled AUPRC, from which legacy metrics (RMSE, PA-F1) are derived as projections. Empirically the authors claim competitive results on nine datasets across five domains (turbofan, spacecraft, server, cardiac, ICU sepsis) with one architecture, and a head-to-head against a frozen amazon/chronos-2 foundation encoder that is framed as "ties on turbofan, wins on spacecraft." Label-efficiency experiments argue pred-FT beats end-to-end finetuning at <=10% labels. The work lives entirely in `fam-jepa/` with surfaces stored as .npz for metric recomputation.

---

## 2. Strengths

- **The pooled-AUPRC-over-surface framing is genuinely useful.** Motivating it against Brier/PA-F1 with concrete counter-evidence (PA inflation numbers, appendix app:pa_f1) is compelling, and storing surfaces as .npz for recomputation is real science hygiene.
- **Predictor-finetuning as a named recipe is a clean, novel framing.** The audit doc correctly calls this out as the paper's core methodological contribution; the cost/benefit vs. probe_h and E2E is clearly articulated with a matched-seed paired analysis.
- **The Chronos-2 foundation-model comparison is honest** in Section 5.5 and in the abstract's "not dominant" framing; the authors do NOT claim to beat a 120M foundation model on every dataset, which is refreshingly calibrated.
- **Stats discipline is good**: paired $t$-tests, Wilcoxon, Cohen's $d$, collapse counts, and $N{=}10$ seed extensions on the key label-efficiency cells.
- **Monotonicity as a diagnostic rather than a constraint** is the right design choice and is flagged as such.
- **The "legacy metrics as projections of the surface" table (tab:metric_projections)** is a useful conceptual unification.

---

## 3. Weaknesses

- **Section 5.3 (sec:anomaly_results) is entirely stale.** Lines 316-319 still describe Mahalanobis-distance scoring as the anomaly method, with SMAP AUPRC 0.192, MSL 0.203, PSM 0.413, SMD 0.091, MBA 0.663. Tab 1 (the main benchmark) reports v24 pred-FT numbers that are 2-3x higher (SMAP 0.395, MSL 0.187, PSM 0.425, SMD 0.236, MBA 0.947). A reviewer reading front-to-back will notice the inconsistency within six pages and will be unable to tell which numbers to believe. This is the single most damaging issue in the draft.
- **Abstract scope does not match RESULTS.md.** Abstract claims "9 datasets in 5 domains." v24 phase 11 added GECCO 2018 (water quality), BATADAL (ICS cyber), and PhysioNet 2012 (ICU mortality). Either those three rows are in scope (so the paper is 12/8) or they are not (so RESULTS.md is ahead of the paper and the abstract should be hedged). The "cross-domain" framing loses a lot if the three newest domains are silently dropped.
- **The "33% of parameters" claim in Contribution 1 is wrong** under v24 numbers. Line 78: "freezing a JEPA encoder and finetuning only the predictor (33\% of parameters)". With v24's ~198K predictor and ~2.16M total, pred-FT is ~9%. 33% is the stale V17 ratio (790K/2.37M). Abstract line 74 says "$\approx$\,198K predictor parameters plus a $\approx$\,513 LayerNorm + linear head ... far cheaper than end-to-end finetuning ($\approx$\,2.16M parameters)" which is correct - but Contribution 1 contradicts it on the same page.
- **The Chronos-2 framing in the abstract is cherry-picked.** Abstract: "a frozen amazon/chronos-2 encoder ... matches FAM pred-FT on FD001 within $0.0003$ AUPRC ... on SMAP FAM beats Chronos-2 by $+0.110$ AUPRC." But RESULTS.md shows Chronos-2 BEATS FAM on FD002 (+0.008) and FD003 (+0.028) with multi-seed runs. Figure 4's caption does acknowledge this - the abstract and the body text (Sec 5.5) do not. Pretending to be honest while omitting the two C-MAPSS losses is worse than either honest posture.
- **Sepsis "matching or exceeding domain-specific supervised models" is over-claiming.** Abstract line 59. FAM sepsis AUROC is 0.802; InceptionTime/MGP-AttTCN SOTA is 0.85-0.87. The paper body (line 309) admits "below the InceptionTime/MGP-AttTCN 0.78-0.85 AUROC SOTA but within range." These two sentences contradict each other.
- **Event prediction vs. event detection is conflated throughout.** The paper's setting is "given history up to $t$, predict event in $(t, t{+}\Delta t]$" (prediction). Most anomaly-detection benchmarks (SMAP/MSL/SMD/PSM) label each timestep as in-segment or not, i.e. detection. The paper silently re-frames this as prediction by thresholding "event happens inside the future interval", but Sec 5.3 even acknowledges "Anomaly 'prediction' on SMAP is mostly within-segment continuation, not lead-time" (line 509). This matters because Tab 1 compares FAM's prediction-framed AUPRC to MTS-JEPA's detection-framed PA-F1, and then reports both as if they were on the same axis. The paper needs one paragraph that cleanly states the setting distinction and defends why the comparison is still meaningful. Right now it slides between the two.
- **Section 5.4 describes Paderborn results (macro-F1 0.781) but Paderborn does not appear in Tab 1 or in RESULTS.md anywhere.** This is either leftover text from an earlier draft (most likely) or an unverifiable claim. Either way it should not ship.
- **Legacy-metric column in Tab 1 is footnoted as "F1 at best threshold (honest)" but the C-MAPSS F1 interpretation is unclear.** C-MAPSS's standard legacy metric is RMSE, not F1. Tab 1 gives `F1 0.840 ± 0.000` for FD001 with no explanation of what a per-window F1 means on a continuous RUL target. Readers from the RUL literature will not understand this number without a formal definition near Eq. (2).
- **Ablation Section 6 is explicitly labeled v20/V17 legacy** (line 357 scope note) which is defensible, but then Section 6.5 "Architecture Sensitivity" (line 495) does not actually vary any architectural hyperparameter - it re-summarises frozen_multi's failure. It reads as a placeholder for a sweep that was not run.
- **No MSL/PSM/SMD Chronos-2 numbers** in the Chronos table (tab:chronos); the paper promises "pending" but MBA is actually done (3 seeds in RESULTS.md) and the figure already uses it. The Chronos table is stale relative to the figure.
- **Figure 2 (fig_evaluation_framework.pdf)** caption line 225 says "Illustrative data." A NeurIPS reviewer will take this as license to distrust the figure. Replace with real data from the stored v24 surface or drop the per-horizon subplot.
- **Figure 1 (fig_architecture_ema.pdf)** caption does not tell the reader that the target encoder is bidirectional with attention pooling (mentioned only in prose at line 160). Equation 2 on page 3 uses $\hat{\h}_{(t, t+\Delta t]}$ but Eq (3) then writes $\hat{\h}_{t+\Delta t}$ (single point) - the notation flips between an interval summary and a timestep embedding and is not defined.
- **No ablation pins down what pred-FT actually buys.** The finetune-mode table compares pred-FT to probe_h, frozen_multi, E2E, scratch. The table that would actually support Contribution 1 is: (pretrained predictor, finetuned) vs. (randomly-initialised predictor of the same size, finetuned) vs. (linear head on h_past of equivalent FLOPs). Without the middle row you cannot tell whether the predictor's *pretrained* weights are doing anything or whether a 198K-parameter MLP head happens to work well.

---

## 4. Specific issues with line numbers

1. **Line 59 (abstract), Line 309 (end of Sec 5.1):** Abstract claims "matching or exceeding domain-specific supervised models ... (ICU sepsis, at $1.4\%$ prevalence)." Body admits Sepsis is below SOTA. One of them must change - sepsis AUROC is 0.802 vs InceptionTime 0.868 per RESULTS.md line 41. Suggested fix: "matching on turbofan/anomaly/cardiac; sub-SOTA but non-trivial on sepsis."
2. **Line 59 (abstract):** "Against a multivariate foundation-model baseline ... matches FAM pred-FT on FD001 within $0.0003$ AUPRC." Omits that Chronos-2 beats FAM on FD002 (+0.008) and FD003 (+0.028). RESULTS.md lines 70-72 show this with multi-seed runs. Either acknowledge or qualify to "on FD001."
3. **Line 59, Line 93, Line 309, Line 522:** "nine datasets spanning five domains." v24 phase 11 added GECCO, BATADAL, PhysioNet 2012 (three more datasets, three arguably new domains). Either update count to 12/8 and include those rows, or explicitly scope as "nine selected in the main benchmark; see Appendix X for three additional domains."
4. **Line 74:** "$\approx$\,198K predictor parameters plus a $\approx$\,513 LayerNorm + linear head, this is more expressive than a linear probe on $\h_\text{past}$ alone but far cheaper than end-to-end finetuning ($\approx$\,2.16M parameters)." The E2E count 2.16M is the v24 total-params; trainable during E2E would be the same. Flag "33% of parameters" in the next clause as stale.
5. **Line 78 (Contribution 1):** "freezing a JEPA encoder and finetuning only the predictor (33\% of parameters)." With v24 the predictor is 198K/2.16M = 9%, not 33%. 33% is the V17 number. Replace with "$\approx$\,9% of parameters."
6. **Line 116:** "$\W \in \R^{d \times (S \cdot P)}$" uses $S$ that is nowhere defined. The other symbols are $C$ (channels) and $P$ (patch). Should be $\R^{d \times (C \cdot P)}$ per ARCHITECTURE.md Sec 1.
7. **Line 127, Line 577 (Appendix Preprocessing):** SMAP window in the method table is "sliding 512 steps (32 tokens)"; in app:preprocessing the same row says "100 steps". Pick one. ARCHITECTURE.md line 190 says 512; v22 entity-split experiments used 100. Paper cannot say both.
8. **Line 181, Line 180 (Tab:finetune_modes):** Parameter counts for the three modes give Predictor-FT = "$\approx$\,198K" and E2E = "$\approx$\,2.16M" - consistent with ARCHITECTURE.md, good. But Probe row has "513" for LN+linear, whereas ARCHITECTURE.md and Tab 2 use 257 for the probe_h case in the legacy ablation (line 365). Unify.
9. **Line 187 (Eq. 3):** $p(t, \Delta t) = \sigma(\mathbf{w}^\top \hat{\h}_{t+\Delta t} + b)$. But $\hat{\h}$ was defined in Eq. 2 as $\hat{\h}_{(t,t+\Delta t]}$, an interval summary. This notation flip is not explained.
10. **Line 225 (fig:auroc_curve caption):** "(b)~AUPRC($\Delta t$) curve: per-horizon discrimination quality. Illustrative data." "Illustrative" is not acceptable in a final NeurIPS figure. Use a real per-horizon curve from a stored v24 surface, or drop the subplot.
11. **Line 232 (fig:probability_surface):** Fine. One good real-data figure is better than two illustrative ones.
12. **Line 238-241 (fig:cross_domain caption):** "FAM beats Chronos-2 on stream-anomaly and cardiac datasets (SMAP $+0.110$, MBA $+0.030$)." RESULTS.md line 77: FAM MBA 0.947, Chronos-2 MBA 0.918, so FAM-Chronos2 = +0.029 (rounds to +0.030). OK. But RESULTS.md table in the same section says the delta column is -0.029, which is Chronos-2-FAM. The FIG caption uses FAM-minus-Chronos2 convention; the RESULTS.md table uses the opposite. Add a one-line sign convention to figure caption. Also "FD001 tie, FD002 +0.008, FD003 +0.028" all correctly reflect RESULTS.md. Good, but the abstract is silent on FD002/FD003.
13. **Line 304 (Tab:benchmark, Sepsis row):** "AUROC $0.85$ InceptionTime (v23 ref)". This citation points to a v23 ref but (a) the row data is v24 phase 6 not v23, (b) the SOTA AUROC is 0.868 per Chen+19 / RESULTS.md line 57, not 0.85. Fix citation and update 0.85 -> 0.868. Also note that PhysioNet 2012 (mortality) is NOT the same task as PhysioNet 2019 Sepsis - one is mortality-at-48h and the other is sepsis-onset prediction. RESULTS.md keeps them as separate rows (lines 41 and 57). Paper Tab 1 names it "Sepsis (ICU)" with SOTA "InceptionTime 0.85" which conflates the two.
14. **Line 309 ("Summary" paragraph):** "v22 pre-canonical pipeline" deltas quoted inline (SMAP +0.105, PSM +0.008, etc.) match RESULTS.md line 43-46. Good. But "C-MAPSS from $\pm 0.009\text{--}0.016$ to $\pm 0.001\text{--}0.009$" - RESULTS.md line 33-35 gives v24 stds of 0.001, 0.002, 0.009, so "0.001-0.009" is correct. Keep.
15. **Line 318-319 (Sec 5.3 anomaly body):** Entirely stale. Describes Mahalanobis protocol (v18/v19), reports Mahalanobis AUPRCs, cites chronological-split failure on PSM. All contradicted by Tab 1 and by the v24 pred-FT protocol. This whole subsection needs to be rewritten for v24 or deleted in favor of a pointer to Tab 1 and to the Limitations entry about PSM Mahalanobis fallback.
16. **Line 322-324 (Sec 5.4 Paderborn):** Paderborn results (macro-F1 0.781, accuracy 0.783) appear here but not in RESULTS.md, not in `v24/results/`, not in Tab 1. Cannot verify. Remove or supply a results JSON and add the row.
17. **Line 342-345 (Tab:chronos):** Shows FD001 and SMAP only. RESULTS.md lines 71-77 have multi-seed Chronos-2 runs on FD002 (1 seed), FD003 (3 seeds), MBA (3 seeds). Add those rows or explain the scope choice.
18. **Line 343-344 (Tab:chronos SMAP row):** "$\Delta t{=}1$ AUPRC $\uparrow$ ... pending ... pending." The JSON `baseline_chronos2_SMAP.json` contains per-horizon AUPRC at $\Delta t{=}1$: 0.305. Not pending. Fill in.
19. **Line 349 (end of Sec 5.5):** "Remaining anomaly-dataset Chronos-2 cells (MSL, PSM, SMD, MBA) are \placeholder{pending}." MBA is complete (3 seeds, AUPRC 0.918 ± 0.002). Update list to "(MSL, PSM, SMD)" and add MBA to the table/figure.
20. **Line 385-386 (Tab:finetune_ablation):** "pred_ft (790K p)" and "e2e (2.37M p)" are V17 legacy counts. Scope note at 357 justifies keeping them, but the paper elsewhere claims the canonical model has 198K/2.16M. The ablation table is the ONLY place a reader sees the 790K/2.37M numbers - make sure the scope note is very visible (currently it is a line of italics tucked under a section heading).
21. **Line 428-429 (Tab:label_efficiency caption):** "V21 re-evaluates with AUPRC." RESULTS.md lines 141-163 show v21 AUPRC results already. Update the caption to reflect that both protocols are available and this table is the F1w legacy cut.
22. **Line 498 (Sec 6.5 Architecture Sensitivity):** "A full depth/width sweep is out of scope for this session." This is not a sensitivity analysis, it is a placeholder. Either delete the subsection or replace with one concrete sweep (e.g. $d \in \{128, 256, 384\}$, which the authors presumably have compute for).
23. **Line 507-515 (Sec 7 Limitations):** "Per-dataset pretraining, not cross-domain. A foundation model would pretrain on a union." This is fine but does not engage with the elephant in the room: Chronos-2 IS a cross-domain foundation model and it ties/wins on turbofan. The Limitations should say so.
24. **Line 522 (Conclusion):** "with $5\%$ labels on C-MAPSS, beats a $3\times$-larger end-to-end baseline on AUPRC." At 5% labels pred-FT vs E2E $p = 0.114$ (not significant, Tab 6 line 440). The stronger claim is 10% labels ($p = 0.023$). Change the conclusion to 10% or reframe as "matches or beats."
25. **Line 534 (Appendix app:window_sensitivity):** "A full window-sensitivity sweep ... is deferred to a future version." v23 phase 5+6 ran a patch-size sweep on SMAP with L in {1, 5, 10, 20} and found L=10 wins with +0.068 AUPRC (RESULTS.md lines 597-617). That is a window sweep. Include it.
26. **Line 546-552 (app:decisions):** Every decision-source cell says "v17 phase 1". The v24 canonical architecture cites ARCHITECTURE.md. Update provenance column.
27. **Line 558 (app:extended):** Points to "v20 phase \{0,1,2,3,5,6\}*.json". Paper's main numbers are v24. Add v24 phase pointers or rename the appendix to say "ablation evidence."
28. **Line 580-581 (app:preprocessing):** "Paderborn 1 vibration z-score on train 1024 steps" - ties to the (unsupported) Paderborn section at line 322. Remove with Paderborn or supply the data.
29. **Consistent stale tokens to grep-and-check:** "V17" / "v17" appears in Sec 6 scope note and decision log (acceptable); "790K" appears in Sec 1 contribution and in ablation tables (acceptable in tables, wrong in Sec 1); "2.37M" appears in Sec 6 ablations only (acceptable); "$d_{ff}=1024$" appears in Sec 6.5 (acceptable as legacy mention); "Chronos-T5-tiny" appears in tab:chronos caption as "superseded" and in the ablation tab in RESULTS.md - the paper draft only mentions it in one caption (fine). One remaining stale: line 369 still cites "e2e: full backbone finetuned (2.37M)" in an *enumerated list that describes the canonical protocol*, not in a legacy table, which is confusing. Clarify or move to the legacy scope.
30. **Line 74 vs line 180:** "$\approx$\,513 LayerNorm + linear head" (abstract area) vs probe/linear-head sizes in Tab 2 that use 513 (LN + linear) and 257 in tab:finetune_ablation row probe_h. Standardise: `LN(256)+Linear(256->1)` = 256 + 256 + 1 = 513 params. The 257 is only `Linear(256->1)` without LN. Pick a convention and use it everywhere.

---

## 5. Questions for the authors

1. You report pooled AUPRC as the primary metric but your anomaly-benchmark baselines (MTS-JEPA, AT, DCdetector) report PA-F1 or segment-F1 on a fundamentally different task framing (detect an in-progress anomaly), not prediction of an event in a future interval. How do you defend the claim in Tab 1 that your AUPRC numbers are "matching or exceeding" those PA-F1 numbers? Is the comparison even well-defined?
2. Sec 5.5 and the abstract frame Chronos-2 as "tie on FD001, loss on SMAP." RESULTS.md shows Chronos-2 WINS on FD002 and FD003 with 1 and 3 seeds respectively. Why are those results absent from the table, figure, and abstract?
3. What exactly is "F1" in the Tab 1 turbofan rows? C-MAPSS is a regression task. Is this F1 over the per-horizon event-in-future-interval labels induced by the RUL cap of 125? If so, please define it in Sec 4 next to Eq. 2. If not, explain.
4. Paderborn (Sec 5.4) is not in RESULTS.md or v24/results/. Did this experiment happen? If yes, where is the JSON? If no, why is it in the paper?
5. Sec 5.3's anomaly paragraph describes a Mahalanobis-on-encoder pipeline, but Tab 1 reports v24 pred-FT numbers. Which is the actual method for the paper's headline anomaly numbers? If pred-FT, why is the v18/v19 Mahalanobis description still in the body text?
6. The "predictor finetuning" contribution: have you run the ablation where the predictor is randomly initialised and finetuned (same 198K params, no pretraining)? Without that row, the table does not separate "predictor MLP as a head" from "predictor pretraining transfers". This is the cleanest test of Contribution 1.
7. What is the sepsis test-time protocol that yields AUPRC 0.186 at 1.4% prevalence in v24, compared to the v23 phase 4 result of AUPRC 0.096 at the same prevalence? RESULTS.md line 41 does not explain the delta. Different cohort? Different min_context? A reader will notice the 2x swing.
8. The probability surface's interval arithmetic ($p(t, b) - p(t, a)$) only works if monotonicity holds. Your monotonicity violation rates on Phase 11 datasets (BATADAL up to 34%, PhysioNet 2012 up to 25%) are NOT "$\leq 0.06$" as claimed at line 309. How do you reconcile?
9. What is the label-regime crossover between pred-FT and E2E on anomaly datasets? All label-efficiency evidence is on C-MAPSS; the claim "label-efficient" is only supported there. Do pred-FT's advantages at low labels generalise to SMAP/PSM?
10. If you drop the "5 domains" framing and include the Phase 11 datasets (GECCO water, BATADAL ICS, PhysioNet 2012 mortality), the cross-domain AUPRC range is 0.11-0.95 - much wider than the "0.186-0.947" range shown in the abstract. Why hide the three datasets where the method is weakest?

---

## 6. Revise-and-resubmit checklist (24-hour priority)

**P0 (reject-grade issues if not fixed):**
1. Rewrite Sec 5.3 (anomaly prediction) from scratch against v24 pred-FT. Remove all Mahalanobis references (SMAP 0.192, MSL 0.203, SMD 0.091, MBA 0.663, PSM 0.413) or move them into a clearly-labeled legacy subsection in the appendix. Currently the section is stale to the point of contradicting Tab 1.
2. Fix the "33% of parameters" in Contribution 1 (line 78) to "$\approx$\,9%".
3. Add the Chronos-2 FD002/FD003/MBA rows to Tab:chronos and Fig 4; rewrite the abstract's Chronos-2 framing to acknowledge Chronos-2 wins on FD002 (+0.008) and FD003 (+0.028).
4. Delete or substantiate the Paderborn section (5.4). No JSON, no row.
5. Reconcile sepsis "matching or exceeding SOTA" in abstract with "below SOTA" in body. Pick the honest framing.

**P1 (accept-grade but visibly weak):**
6. Clarify event-prediction vs event-detection framing in a dedicated paragraph in Sec 4 or early Sec 5. Justify the Tab 1 comparison.
7. Replace "Illustrative data" in Fig 2 with a real per-horizon AUPRC curve from a stored v24 surface.
8. Fix notation flip $\hat{\h}_{(t,t+\Delta t]}$ (Eq 2) vs $\hat{\h}_{t+\Delta t}$ (Eq 3). Define precisely.
9. Fix the $\W \in \R^{d \times (S \cdot P)}$ undefined $S$ on line 116 - should be $C$.
10. Fix the SMAP window-size inconsistency: 512 steps (line 127) vs 100 steps (line 577).
11. Decide whether Phase 11 (GECCO, BATADAL, PhysioNet 2012) is in scope. If yes, add to Tab 1, update counts. If no, state the scoping rule.
12. Fix the sepsis SOTA citation: 0.868 (Chen+19), not 0.85, and disambiguate PhysioNet 2012 mortality from PhysioNet 2019 sepsis (these are different tasks).

**P2 (polish):**
13. Fill in the SMAP $\Delta t{=}1$ AUPRC cell in Tab:chronos (data is in the JSON at 0.305).
14. Reconcile probe head parameter counts (257 vs 513) between Tab 1 and Tab 2.
15. Update line 428 Tab:label_efficiency caption (v21 AUPRC data is already available).
16. Update line 534 app:window_sensitivity to include the v23 phase 5+6 SMAP patch-size sweep (L=10 wins +0.068 AUPRC).
17. Update line 546 app:decisions "Source" column to reflect ARCHITECTURE.md where v24 overrides v17.
18. Fix line 522 conclusion: the 5% crossover is not statistically significant ($p=0.114$). Move to 10% ($p=0.023$) or weaken to "matches or beats."
19. Add a sanity-check ablation: pretrained-and-finetuned predictor vs. random-init-and-finetuned predictor at the same parameter count. This is the missing evidence for Contribution 1.
20. Add the monotonicity-violation-rate column for Phase 11 datasets to the limitations paragraph that claims "$\leq 0.06$ on every seed of every dataset" (line 309) - at least one Phase 11 dataset violates this bound.

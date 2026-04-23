# paper.tex Audit vs v24 Canonical Results

**Date**: 2026-04-23
**Paper**: `/home/sagemaker-user/IndustrialJEPA/paper-neurips/paper.tex`
**Truth sources**:
 - `/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/RESULTS.md` (v24 section, lines 21-46; Chronos-2 table, lines 48-72)
 - `/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v24/results/*.json`
 - `/home/sagemaker-user/IndustrialJEPA/fam-jepa/ARCHITECTURE.md` (canonical ~2.16M params, P=16 global)
 - `/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v24/SESSION_PROMPT.md`

The paper text is currently calibrated against v17 / v20 / v21 / v22 numbers. v24 is a new run using the canonical `fam-jepa/model.py` + `fam-jepa/train.py` codebase - same overall architecture family but with a cleaner implementation (RevIN, cumulative target, P=16 global, min-context floor). Variance is uniformly 1 - 30x tighter than v22. FD003 regresses. A Chronos-2 (multivariate, 768-d) baseline now matches FAM on FD001 within 0.001 AUPRC, inverting the "FAM beats Chronos-T5-tiny" narrative.

Everything below is organised per-claim with paper line numbers, paper value, v24 value, and a discrepancy flag.

---

## 1. Abstract (lines 58-60)

| Claim | Paper value | v24 value | Flag |
|---|---|---|---|
| L59 "turbofan, mean across FD001-003" pooled AUPRC | 0.932 | mean(0.926, 0.908, 0.766) = **0.867** | STALE (FD003 drops 0.932 -> 0.766) |
| L59 "spacecraft, mean across SMAP/MSL" pooled AUPRC | 0.197 | mean(0.395, 0.187) = **0.291** | STALE (SMAP jumped; framing still works) |
| L59 "server, mean across PSM/SMD" pooled AUPRC | 0.252 | mean(0.425, 0.236) = **0.331** | STALE (both datasets up) |
| L59 "cardiac, MBA" pooled AUPRC | 0.663 | **0.947** | STALE (MBA jumped +0.284) |
| L59 "FAM pred-FT beats Chronos-T5-tiny ... +0.044 AUPRC, -8.5 RMSE" | +0.044 / -8.5 | Chronos-2 matches FAM (Delta = -0.0003 AUPRC, +0.010 AUROC). The old Chronos-T5-tiny number (v21) still exists but is against a weaker baseline. | STORY INVERTED |
| L74 "2.37M parameters" (also L147, L180, L263, L327, L502) | 2.37M | **2.16M** (ARCHITECTURE.md line 236, "Total ~2.16M") | STALE globally |
| L74 predictor params "~790K" | 790K | **~198K** (ARCHITECTURE.md line 234-238) | STALE globally |

Abstract is left intact for a separate pass per the user's instructions. All numbers in this row are flagged only.

---

## 2. Method / architecture prose

| Line | Claim | Paper value | v24 value | Flag |
|---|---|---|---|---|
| L74 | Predictor ~790K params, FAM 2.37M E2E, 257 linear head | 790K / 2.37M / 257 | 198K / 2.16M / 513 (LayerNorm + Linear(256->1)) | STALE |
| L112 | "Patches. P=16 ... fixed for all datasets with sufficient temporal resolution (sample rate >= 1/min); for coarsely sampled data (Sepsis at 1/hour), we use P=1" | OK | ARCHITECTURE.md agrees (line 222) | OK - no change needed |
| L147 | Encoder d=256, 2 layers, 4 heads | OK | ARCHITECTURE.md line 232: d=256, L=2, H=4, d_ff=256 (paper says d_ff=1024 at L479) | PARTIAL - d_ff is 256 in v24, not 1024 |
| L160 | EMA tau=0.99 | OK | OK | OK |
| L179 | Predictor-FT tunes "790K params"; linear head "linear 257" ; E2E "2.37M" | 790K / 257 / 2.37M | 198K / 513 / 2.16M | STALE (update table) |
| L181 | Frozen probe 257 params | 257 | ARCHITECTURE.md has event head = LayerNorm+Linear = 513 params in v24; probe_h in v21 RESULTS.md still shows 2.6K but v20 shows 257. | PARTIAL - depends on which probe |
| L444 | "2.37M backbone" | 2.37M | 2.16M | STALE |
| L479 | "d_ff = 1024, 2.37M parameters" | d_ff=1024 / 2.37M | ARCHITECTURE.md line 232: d_ff = **256**, total **~2.16M** | STALE - d_ff is wrong |

---

## 3. Main Benchmark Table (Tab 1, lines 268-291)

All 8 dataset rows are v21 / v22 numbers. v24 supersedes them.

| Line | Dataset | Paper AUPRC | v24 AUPRC | Paper legacy | v24 F1-best (non-PA) | v24 PA-F1 | Flag |
|---|---|---|---|---|---|---|---|
| L280 | FD001 | 0.945 +/- 0.016 | **0.926 +/- 0.001** | RMSE 17.1 +/- 4.6 | 0.840 +/- 0.000 | n/a | STALE; RMSE also drifts (see below) |
| L281 | FD002 | 0.955 +/- 0.009 | **0.908 +/- 0.002** | RMSE 12.4 +/- 1.3 | 0.829 +/- 0.001 | n/a | STALE |
| L282 | FD003 | 0.932 +/- 0.010 | **0.766 +/- 0.009** | RMSE 16.2 +/- 1.9 | 0.747 +/- 0.006 | n/a | STALE; REGRESSION |
| L284 | SMAP | 0.290 +/- 0.042 | **0.395 +/- 0.010** | F1 0.440 +/- 0.003 | 0.454 +/- 0.005 | 0.808 +/- 0.017 | STALE (up, variance 4x tighter) |
| L285 | MSL | 0.237 +/- 0.077 | **0.187 +/- 0.007** | F1 0.330 +/- 0.022 | 0.332 +/- 0.000 | 0.788 +/- 0.016 | STALE (small regression, variance 10x tighter) |
| L286 | PSM | 0.417 +/- 0.113 | **0.425 +/- 0.006** | F1 0.519 +/- 0.006 | 0.536 +/- 0.000 | 0.929 +/- 0.022 | STALE (variance 19x tighter) |
| L287 | SMD | 0.196 +/- 0.025 | **0.236 +/- 0.015** | F1 0.262 +/- 0.030 | 0.273 +/- 0.015 | 0.844 +/- 0.030 | STALE |
| L288 | MBA | 0.784 +/- 0.024 | **0.947 +/- 0.001** | F1 0.725 +/- 0.024 | 0.860 +/- 0.003 | 1.000 +/- 0.000 | STALE (+0.163; variance 24x tighter) |
| MISSING | Sepsis | not in table | **0.186 +/- 0.004** | n/a | 0.287 +/- 0.001 | n/a | NEW ROW (P=1 exception; AUROC 0.802 +/- 0.003) |

RMSE in v24: FD001 cross=38.7 +/- 0.03, expected=39.6 +/- 0.3 (phase10_legacy.json). These are **worse** than v22's 17.1 +/- 4.6 - v24 did not reoptimise the legacy-RMSE derivation. This is a real regression on RMSE and the paper's RMSE comparison to STAR (10.61) gets worse, not better, under v24. Flag for a follow-up.

---

## 4. Main Table Summary Prose (lines 293)

Every per-dataset number quoted in the big summary paragraph (L293) is stale for the same reason as Tab 1. The passage also says:
 - "SMAP/MSL/PSM retain sub-chance ranking (AUROC <= 0.51) on the unseen tail" -> v24 AUROCs: SMAP 0.594, MSL 0.472, PSM 0.566. MSL still sub-chance; SMAP/PSM now above chance. **Partially stale.**
 - "SMD/MBA with milder shift show clear pred-FT signal (AUROC 0.66/0.75)" -> v24: SMD 0.680, MBA 0.896. **Stale in the positive direction.**
 - "monotonicity violation <= 0.01 in all runs" -> v24 mono violation rates mostly 0.0 (FD001 all seeds 0.0; sepsis 1e-5). **OK.**

---

## 5. C-MAPSS / RUL section (lines 295-298, sec:rul_results)

| Claim | Paper | v24 | Flag |
|---|---|---|---|
| L298 FAM E2E @ 100% RMSE 14.96 +/- 1.16 (5 seeds) | 14.96 | v24 phase2 reports pred-FT only, no E2E; v22 kept this as v20 number | OK (v20 ref; not replaced by v24). **Orphaned if sec:finetune_ablation is removed.** |
| L298 "AUROCw of 0.994 at 100% labels" | 0.994 | n/a in v24; v21 AUROCs were ~0.99 on FD001 | OK legacy |
| L298 "pred-FT at 5% labels F_w = 0.26 vs STAR 24.55" | 0.26 | not rerun in v24 | OK legacy (v20 metric) |

Whole section depends on v20 F1w. Flag as "v20 legacy section; retain as context" or remove.

---

## 6. Anomaly Prediction section (lines 300-303, sec:anomaly_results)

Entire paragraph uses v18/v19 Mahalanobis numbers plus v22 pred-FT. All specific numerical claims:
 - L303 "SMAP AUPRC 0.192 +/- 0.007" -> v24 **0.395 +/- 0.010** (Mahal vs pred-FT are different methods; this paragraph is describing Mahal first then pred-FT)
 - L303 "MSL 0.203 +/- 0.029" -> v24 MSL 0.187 +/- 0.007
 - L303 "PSM 0.413 +/- 0.035" -> v24 0.425 +/- 0.006
 - L303 "FAM beats the MTS-JEPA PA-F1 baseline on SMAP (0.951 vs 0.336), MSL (0.849 vs 0.336), PSM (0.910 vs 0.616)" -> v24 PA-F1s are 0.808 / 0.788 / 0.929. Paper's 0.951 / 0.849 / 0.910 do not match any source I can find in RESULTS.md (may be from an older stream-mode Mahalanobis). **STALE - numbers differ from v24.**
 - L303 "SMD under shared encoder yields 0.091 +/- 0.010 AUPRC (PA-F1 0.644)" -> v24 SMD 0.236 +/- 0.015 AUPRC, PA-F1 0.844. **STALE; v24 improves.**
 - L303 "MBA reaches AUPRC 0.663 +/- 0.078 (PA-F1 0.914)" -> v24 MBA 0.947 +/- 0.001 AUPRC, PA-F1 1.000 +/- 0.000. **STALE; v24 jumps.**

This entire section needs a rewrite once the main table is updated. Flagged for a follow-up pass.

---

## 7. Chronos section (lines 310-332, sec:foundation_baselines)

**The whole story inverted.** The v24 Chronos-2 baseline (frozen 768-d multivariate encoder + linear probe, same labeled data / same splits / same horizons / same BCE loss as FAM pred-FT) matches FAM on FD001 within 0.001 AUPRC.

| Claim | Paper | v24 | Flag |
|---|---|---|---|
| L313, L317 | "Chronos-T5-tiny 8.4M vs FAM 2.37M" | Chronos-2 (768-d multivariate) has not been param-counted in RESULTS.md; FAM is 2.16M | STALE (model/family changed; param count changed) |
| L324 Chronos AUPRC | 0.901 +/- 0.002 | **Chronos-2: 0.925 +/- 0.000** (v24 chronos2 agg JSON) | STALE |
| L324 Chronos AUROC | 0.980 +/- 0.001 | **0.929 +/- 0.002** (this is AUROC which is now lower) | STALE |
| L324 Chronos RMSE | 25.58 +/- 3.71 | not reported in v24 Chronos-2 agg JSON | PARTIAL (pending) |
| L325-327 FAM probe_h / pred_ft / E2E lines | v21 numbers (0.928 / 0.945 / 0.962) | v24 has **FAM AUPRC 0.926 +/- 0.001** (only pred-FT reported; no v24 probe_h, no v24 E2E yet) | STALE + incomplete |
| L317, L332 caption and prose | "FAM beats Chronos-T5-tiny by +0.027 / +0.017 / +0.061 AUPRC" | v24 FAM vs Chronos-2 on FD001: **Delta AUPRC = -0.0003 (Chronos-2 marginally higher or tied)**, **Delta AUROC = +0.010 (Chronos-2 higher)**. Chronos-2 is *better* at short horizons (dt=1 AUPRC 0.41 vs 0.09); tied overall. RESULTS.md line 64-69. | STORY INVERTED |
| L317 "FAM 2.37M" | 2.37M | 2.16M | STALE |

This section needs a wholesale rewrite: the headline is no longer "small domain-specific FAM beats larger generalist"; it is "a general multivariate foundation model matches a domain-specific FAM on FD001, and Chronos-2 is actually *stronger* at short horizons." The user asked me not to fix this pass.

Note: the paper currently cites Chronos (Ansari et al. 2024) which is Chronos-T5. Chronos-2 is a newer multivariate model - the bibtex may need a new entry too.

---

## 8. Ablations (lines 336-479)

All ablation tables (tab:finetune_ablation, tab:label_efficiency, tab:sigreg_ablation) use v20 F1w as primary. They are consistent internally. v24 did not rerun these. Options per user:
 - Mark as "v20 legacy reference" in captions and keep.
 - Remove and replace with a pointer to the v21/v22/v24 appendix.

No v24-specific discrepancies since v24 did not touch these experiments. The main risk is that the paper's abstract and intro cite the $p = 0.023, N = 10$ pred-FT-vs-E2E result at 10% labels (v20 F1w) -> this statistic is not reproduced under v21 AUPRC (RESULTS.md lines 138-142 show the v21 10-seed AUPRC version has $p = 0.054$). That is already flagged in RESULTS.md line 144-151 but not in the paper.

---

## 9. Limitations (lines 484-495)

| Line | Claim | v24 status | Flag |
|---|---|---|---|
| L493 | "Chronos-T5-tiny outperforms FAM's frozen probe at the same parameter-matched task despite being the same order of magnitude in size; a cross-domain FAM pretrained at comparable scale (not done here) is the natural comparison." | DIRECTLY CONTRADICTS L332 which says FAM beats Chronos. v24 result: Chronos-2 ties on FD001. | LOGICAL INCONSISTENCY IN CURRENT PAPER |

This is an internal contradiction already present in the paper (predates v24). Should be harmonised when the Chronos section is rewritten.

---

## 10. Conclusion (lines 500-502)

 - L502 "2.37M-parameter model" -> 2.16M. STALE.
 - L502 "5% labels on C-MAPSS, beats a 3x-larger end-to-end baseline on AUPRC" -> the 3x factor was E2E vs pred-FT; v24 did not rerun this at 5% labels. OK as a v21 reference if the ablation tables remain, otherwise orphaned.
 - L502 "Amazon's Chronos-T5-tiny (8.4M)" -> v24 comparison is against Chronos-2 (multivariate); also tied not won. STALE.
 - L502 "anomaly detection PA-F1 above MTS-JEPA on SMAP, MSL, and PSM with a single architecture" -> v24 PA-F1: SMAP 0.808 vs MTS-JEPA 0.336 (OK); MSL 0.788 vs 0.336 (OK); PSM 0.929 vs 0.616 (OK). **True under v24.** No change needed.

---

## 11. Appendix (lines 509-566)

| Line | Claim | v24 | Flag |
|---|---|---|---|
| L514 | "$W = 16$ horizon windows" | v24 uses 7-8 horizons per dataset (ARCHITECTURE.md lines 140-146: FD001 {1,5,10,20,50,100,150}; anomaly {1,5,10,20,50,100,150,200}; sepsis {1,2,3,6,12,24,48}) | STALE - horizon scheme changed entirely |
| L525 | Tokenizer / patch values in the decision log | v24 = P=16 fixed globally (P=1 for Sepsis). Paper already mentions this at L112/L120. | OK |
| L540 | "v20 phase 0, Phase 5, Phase 6" | v20/21 references are fine; the v24 / Chronos-2 paths need to be added | PARTIAL - needs new appendix rows for v24 surfaces |
| L545 | "SMAP 0.793 PA vs 0.038 non-PA" etc. (Mahalanobis inflation numbers) | v24 pred-FT SMAP PA 0.808 / non-PA 0.212; PSM PA 0.929 / non-PA 0.112; SMD PA 0.844 / non-PA 0.154. Same qualitative message, different numbers. | STALE but qualitatively aligned |

---

## 12. Parameter counts - summary list

Every "2.37M" should become "~2.16M" when we update comprehensively: lines 74, 147 (implicitly), 180, 263, 288 caption, 293 (summary), 317 (chronos caption), 327 (chronos table), 444 (cross-subset consistency), 479, 502. The "790K predictor" should become "~198K". Count of occurrences via grep was sanity-checked but not enumerated here.

---

# Revision plan

Group the discrepancies into ordered edit chunks. Each chunk is a coherent edit that leaves the paper compilable and self-consistent.

1. **(This pass) Main Results Table + summary paragraph.** Replace Tab 1 (L268-291) with v24 rows. Add a dedicated PA-F1 column (v24 reports PA-F1 cleanly for all 5 anomaly datasets). Add Sepsis row with P=1 note. Restate legacy column as "F1-best (non-PA)". Rewrite the one-paragraph summary (L293) to cite v24 numbers and note (a) variance uniformly 1-30x tighter than v22, (b) FD003 regression, (c) MBA jump. Leave everything else intact.

2. **(Next pass) Abstract.** Recompute the domain-mean AUPRC numbers (turbofan 0.867, spacecraft 0.291, server 0.331, cardiac 0.947). Rewrite the Chronos sentence from "FAM beats Chronos-T5-tiny" to "FAM matches Chronos-2 zero-shot + probe within 0.001 AUPRC on FD001; per-dataset SSL is competitive with a multivariate foundation model rather than dominant." Update "2.37M" -> "~2.16M" once here; update globally in a later pass.

3. **(Next pass) Chronos section (L310-332).** Rewrite from scratch. New headline: "On FD001 a zero-shot Chronos-2 + linear probe (trained on the same labels FAM uses for pred-FT) matches FAM pred-FT within 0.001 AUPRC (Chronos-2 0.925 +/- 0.000 vs FAM 0.926 +/- 0.001)." Caveat: Chronos-2 is better at dt=1 (AUPRC 0.41 vs 0.09), FAM/Chronos-2 converge at longer horizons. Full anomaly-dataset Chronos-2 sweep is pending (5/6 cells "pending" in RESULTS.md). Update Tab 3 accordingly. This reframes the contribution: our pred-FT recipe still provides a way to specialise, not just a way to beat foundation models.

4. **(Next pass) Abstract / conclusion / limitations harmonisation.** Remove the contradiction between L332 (FAM beats Chronos) and L493 (Chronos beats FAM). Under v24 the honest framing is: FAM matches Chronos-2 on FD001; Chronos-2 pending on anomaly. Drop the "$+0.044$ / $-8.5$" claim in the abstract.

5. **(Next pass) Anomaly section (L300-303).** Rewrite with v24 pred-FT numbers. Current paragraph has SMAP/MSL/PSM with AUROC "sub-chance" framing which is no longer fully true under v24 (SMAP 0.594, PSM 0.566 are now above chance; MSL 0.472 is still sub-chance). Emphasise that PA-F1 now cleanly exceeds MTS-JEPA on all 4 comparable anomaly datasets with tight variance.

6. **(Optional) Ablation tables (tab:finetune_ablation, tab:label_efficiency, tab:sigreg_ablation).** Either annotate captions as "v20 legacy protocol, F1w metric; v21 AUPRC reruns in Appendix" or move to appendix. v24 did not rerun.

7. **(Global) Parameter-count fix.** Replace "2.37M" with "~2.16M" and "790K predictor" with "~198K" across all prose. $d_{ff} = 1024$ -> $d_{ff} = 256$ at L479. Fix "W=16 horizons" in appendix to reflect per-dataset horizon sets.

8. **(Optional) Appendix pointers.** Add `experiments/v24/surfaces/*.npz` and `experiments/v24/results/*.json` to L540 so the v24 provenance is findable.

Order: 1 (this pass), then 2+3+4 as one "Chronos / abstract" pass (they reference each other), then 5, then 6, then 7, then 8. The main table rewrite (this pass) unlocks everything downstream because every number in the rewrite section-5-prose and Chronos-section is dominated by the table's values.

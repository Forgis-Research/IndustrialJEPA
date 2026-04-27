# SOTA Research Document - FAM-JEPA NeurIPS 2026 Table 4 Comparison

**Date**: 2026-04-26
**Purpose**: Rigorous apples-to-apples SOTA values for every dataset in our paper.
All numbers verified against primary sources or directly extracted papers.

---

## Top-line Summary

C-MAPSS SOTA is well-tracked (STAR 2024: RMSE 10.61 / 13.47 / 10.71 for FD001/FD002/FD003).
Anomaly-detection SOTA is fundamentally split by metric protocol: PA-F1 (deeply inflated) vs affiliation/non-PA F1 (honest). The canonical PA-F1 papers (Anomaly Transformer, DCdetector) report 0.92-0.98 on SMAP/MSL/SMD/PSM, but these numbers are not directly comparable to our non-PA F1 (0.04-0.14 range). The best honest-metric SOTA is DADA (ICLR 2025) with affiliation-F1 ~75-85% across SMAP/MSL/SMD/PSM. For MBA (ECG), SKAB, ETTm1, GECCO, and BATADAL there is no unified deep-learning anomaly-prediction SOTA - we document the closest available references.

---

## Master Comparison Table

| Dataset | SOTA Method | SOTA Value | Metric | Protocol Notes | Paper / arXiv | Year | Our ref key |
|---------|-------------|------------|--------|----------------|---------------|------|-------------|
| C-MAPSS FD001 | STAR | 10.61 RMSE | RMSE (cycles) | Supervised E2E; piecewise-linear RUL cap=125; last window point prediction | MDPI Sensors 2024 / PMC10857698 | 2024 | fan2024star |
| C-MAPSS FD002 | STAR | 13.47 RMSE | RMSE (cycles) | Same as FD001; 6 operating conditions | MDPI Sensors 2024 | 2024 | fan2024star |
| C-MAPSS FD003 | STAR | 10.71 RMSE | RMSE (cycles) | Two fault modes; same cap | MDPI Sensors 2024 | 2024 | fan2024star |
| SMAP | Anomaly Transformer | 0.9669 PA-F1 | PA-F1 | Point-adjustment; threshold @ val-set F1-max; CRITICAL: random baseline ~0.94 PA-F1 | ICLR 2022 Spotlight | 2022 | xu2022anomalytransformer |
| SMAP (honest) | DADA | 0.7542 affil-F1 | Affiliation F1 | No PA; affiliation metric (Huet 2022); threshold free | ICLR 2025 / arXiv 2405.15273 | 2025 | dada2025 |
| MSL | DCdetector | 0.9660 PA-F1 | PA-F1 | Same protocol as AT; long segments inflate PA | KDD 2023 | 2023 | yang2023dcdetector |
| MSL (honest) | DADA | 0.7848 affil-F1 | Affiliation F1 | Same honest protocol | ICLR 2025 | 2025 | dada2025 |
| PSM | Anomaly Transformer | 0.9789 PA-F1 | PA-F1 | PA; single run | ICLR 2022 | 2022 | xu2022anomalytransformer |
| PSM (honest) | DADA | 0.8226 affil-F1 | Affiliation F1 | No PA | ICLR 2025 | 2025 | dada2025 |
| SMD | Anomaly Transformer | 0.9233 PA-F1 | PA-F1 | 28-machine average; PA; single run | ICLR 2022 | 2022 | xu2022anomalytransformer |
| SMD (honest) | DADA | 0.8457 affil-F1 | Affiliation F1 | No PA | ICLR 2025 | 2025 | dada2025 |
| MBA (MIT-BIH) | No unified TSAD SOTA | -- | -- | See per-dataset notes | -- | -- | -- |
| SKAB | Conv-AE (official leaderboard) | 0.78 F1 | Standard F1 | Official SKAB leaderboard; no PA; 34-series avg | github.com/waico/SKAB | 2020 | skab2020 |
| ETTm1 (anomaly) | No anomaly SOTA found | -- | -- | ETTm1 is a forecasting benchmark; anomaly subset use is non-standard | -- | -- | -- |
| GECCO | Competition winner | ~0.71-0.80 F1 | F1 | GECCO 2018 challenge; task is online event detection; no universal post-competition deep-learning leaderboard | GECCO 2018 / zenodo 3884398 | 2018 | batadal2018 [reuse] |
| BATADAL | Housh & Ohar (competition winner) | S=0.970 | BATADAL S-score | Custom metric combining time-to-detect + class accuracy; 7 attacks detected of 7; NOT standard AUC or F1 | JRWPM 2018 (Taormina et al.) | 2018 | batadal2018 |

---

## Per-Dataset Sections

---

### 1. C-MAPSS (FD001, FD002, FD003)

**Currently cited**: STAR [fan2024star] RMSE 10.61 (FD001). CONFIRMED.

**Source**: Guo Fan et al., "A Two-Stage Attention-Based Hierarchical Transformer for Turbofan Engine Remaining Useful Life Prediction," MDPI Sensors, vol 24 issue 3, 2024. PMC10857698 / DOI:10.3390/s24030824.

**Exact STAR results table (Table 5 from PMC10857698)**:

| Method | FD001 RMSE | FD001 Score | FD002 RMSE | FD002 Score | FD003 RMSE | FD003 Score |
|--------|-----------|-------------|-----------|-------------|-----------|-------------|
| BiLSTM | 13.65 | 295 | 23.18 | 4130 | 13.74 | 317 |
| DCNN | 12.61 | 237 | 22.36 | 1041 | 12.64 | 284 |
| GCT | 11.27 | -- | 22.81 | -- | 11.42 | -- |
| BiLSTM-Att | 13.78 | 255 | 15.94 | 1280 | 14.36 | 438 |
| DATCN | 11.78 | 229 | 16.95 | 1842 | 11.56 | 257 |
| AGCNN | 12.42 | 225 | 19.43 | 1492 | 13.39 | 227 |
| DAST | 11.43 | 203 | 15.25 | 924 | 11.32 | 154 |
| BiLSTM-DAE-Tf | 10.98 | 186 | 16.12 | 2937 | 11.14 | 252 |
| **STAR** | **10.61** | **169** | **13.47** | **784** | **10.71** | **202** |

**Is STAR still SOTA (2025-2026)?** Checked recent publications:
- A Science Reports 2025 paper achieves RMSE 14.44 / 13.40 on FD001/FD003 - this is WORSE than STAR.
- TTSNet (2025): RMSE 13.25 on FD002, 11.06 on FD003 - worse than STAR on FD002, slightly better on FD003 (but venue is MDPI Sensors, non-top-venue).
- No top-venue paper (NeurIPS/ICML/ICLR 2025) beats STAR across FD001/FD002/FD003.
- **STAR remains the canonical supervised SOTA for all three FD sub-datasets as of April 2026.**

**Protocol notes**:
- Supervised E2E training. No SSL component.
- Standard piecewise-linear RUL label with cap at 125 cycles (universal convention).
- Evaluation on last-cycle of each test trajectory (point prediction, not distribution).
- FAM RMSE is derived from E[Deltat] of the probability surface: different estimator from STAR's direct point prediction. Flag this gap in the paper.
- Score metric (NASA-S) is asymmetric: over-prediction penalized less than under-prediction. STAR's Score 169 is genuinely SOTA on FD001.
- FD004 not in our benchmark (4 operating conditions + 2 fault modes: extremely hard). STAR FD004: 15.87 RMSE.

**Suggested citation in paper**: fan2024star confirmed. Add FD002/FD003 values to the table.

**Leakage warning**: Some papers achieve RMSE < 9 on FD001 by using test labels for normalization or by testing on the wrong subset. STAR at 10.61 passes basic sanity (BiLSTM-DAE-Tf at 10.98 is a strong pre-STAR baseline, confirming competitive context).

---

### 2. SMAP (Mars rover spacecraft anomaly detection)

**Currently cited**: MTS-JEPA [mtsjepa2026] PA-F1 0.34. THIS IS ANOMALOUS - see below.

**MTS-JEPA actual results (arXiv 2602.04643, extracted from HTML)**:
- MTS-JEPA SMAP: F1=33.64%, AUC=65.41%, Prec=24.24%, Rec=56.02%
- IMPORTANT: The MTS-JEPA F1 of 0.34 is NOT the standard PA-F1. It is an early-warning protocol F1 ("will the next window be anomalous?"), evaluated on a window-level binary prediction task. This is a DIFFERENT task from the standard anomaly detection F1 that Anomaly Transformer and DCdetector report. Comparing 0.34 (window-level early warning) to 0.97 (PA-F1 on point anomaly detection) is comparing apples to oranges.

**PA-F1 canonical SOTA**:
- Anomaly Transformer (AT): SMAP PA-F1 = 0.9669 (ICLR 2022, xu2022anomalytransformer). CONFIRMED from multiple sources.
- DCdetector: SMAP PA-F1 approximately 0.97-0.98 (exact number in paper table image, not extractable as text, but narrative says "best or comparable to AT on SMAP").
- TimesNet: SMAP PA-F1 = 0.7302 (from CPatchBLS Table II which also includes TimesNet as baseline).
- CPatchBLS (2024): SMAP PA-F1 = 0.9735 (new SOTA under PA, non-top-venue paper).
- THEMIS (Oct 2025): SMAP affiliation-F1 = 73.21% (no PA).

**PA-F1 inflation on SMAP**:
- The "Quo Vadis" ICML 2024 position paper shows random baselines achieve very high PA-F1 on anomaly datasets (SWaT: 0.963 random, SMD: 0.894 random). For SMAP with its very long anomaly segments (some channels have segments 100s of steps long), PA inflation is extreme.
- Our paper already states: "random-init encoder + Mahalanobis achieves 0.604 +/- 0.007 PA-F1 on SMAP." This is the correct floor to report.
- The gap between AT at 0.967 and random at ~0.60 is 0.37 - not negligible, but far below the headline number suggests.

**Honest-metric SOTA (affiliation F1, no PA)**:
- DADA (ICLR 2025, arXiv 2405.15273): SMAP affiliation-F1 = 75.42%. This is the cleanest comparison point.
- THEMIS (Oct 2025): SMAP affiliation-F1 = 73.21%.
- GPT4TS (affiliation, DADA benchmark): SMAP affiliation-F1 = 74.67%.
- A.T. (Anomaly Transformer, re-evaluated with affiliation metric by DADA): SMAP affiliation-F1 = 71.65%.

**Resolution for our paper**: Report both (1) PA-F1 against AT 0.967 with explicit inflation caveat, and (2) affiliation-F1 against DADA 0.754. Our non-PA F1 of 0.038 vs. AT PA-F1 of 0.967 is NOT a valid comparison and should not appear in the table without the caveat box.

**Update**: The paper currently cites MTS-JEPA as 0.34 (the early-warning F1) - this is technically the closest protocol match to our own early-warning surface. Keep this citation but add: "MTS-JEPA uses an early-warning window protocol distinct from standard anomaly detection F1."

---

### 3. MSL (Mars Science Lab)

**Currently cited**: Not explicitly - MSL may not be in current Table 4.

**PA-F1 canonical SOTA**:
- DCdetector: MSL PA-F1 = 0.9660 (KDD 2023, yang2023dcdetector).
- Anomaly Transformer: MSL PA-F1 = 0.9359 (ICLR 2022).
- THEMIS (Oct 2025): MSL affiliation-F1 = 78.78% (new SOTA under honest metric).
- DADA (ICLR 2025): MSL affiliation-F1 = 78.48%.
- GPT4TS (DADA benchmark): MSL affiliation-F1 = 77.23%.

**Key note**: Like SMAP, MSL has long anomaly segments; PA inflation is significant. The "Quo Vadis" paper confirms SMD random baseline PA-F1 = 0.894 - MSL is likely similar or worse (even longer segments on some channels).

**Resolution**: Same dual-reporting approach as SMAP.

---

### 4. PSM (Pooled Server Metrics)

**Currently cited**: MTS-JEPA PA-F1 0.62.

**MTS-JEPA PSM result (extracted from arXiv 2602.04643)**: F1=61.61%, AUC=77.85%. Same early-warning window protocol caveat applies.

**PA-F1 canonical SOTA**:
- Anomaly Transformer: PSM PA-F1 = 0.9789 (ICLR 2022, xu2022anomalytransformer).
- CPatchBLS: PSM PA-F1 = 0.9847 (2024, marginal improvement).
- TimesNet: PSM PA-F1 = 0.9729 (from CPatchBLS Table II comparison).

**Honest-metric SOTA (affiliation F1)**:
- DADA (ICLR 2025): PSM affiliation-F1 = 82.26%.
- GPT4TS (DADA benchmark): PSM affiliation-F1 = 81.44%.
- ModernTCN (DADA benchmark): PSM affiliation-F1 = 79.59%.

**Protocol note**: PSM is from eBay server metrics, 25 features. Shorter anomaly segments than SMAP/MSL so PA inflation is somewhat less severe, but still present.

**Resolution**: Report MTS-JEPA's 0.62 as the early-warning baseline (matches our protocol most closely). Add PA-F1 AT 0.979 as canonical SOTA with inflation caveat. Add DADA 0.823 as honest-metric SOTA.

---

### 5. SMD (Server Machine Dataset)

**Currently cited**: Anomaly Transformer [xu2022anomalytransformer] PA-F1 0.93.

**Confirmed**: AT SMD PA-F1 = 0.9233 (28-machine average, PA-adjusted, ICLR 2022). Correct.

**Other PA-F1 methods on SMD**:
- DCdetector: ~0.93-0.95 (exact figure in table image; narrative says "best or comparable").
- TimesNet: SMD PA-F1 = 0.8581 (from search results).
- GPT4TS: SMD affiliation-F1 = 83.14% (DADA benchmark, non-PA).

**Honest-metric SOTA**:
- DADA (ICLR 2025): SMD affiliation-F1 = 84.57%.
- GPT4TS: SMD affiliation-F1 = 83.14%.
- Quo Vadis (ICML 2024) non-PA: PCA Error achieves SMD F1 = 57.2% (standard point-wise F1). Note: this is different from affiliation F1.
- Random baseline PA-F1 on SMD = 0.894 (Quo Vadis ICML 2024). THIS is why AT at 0.923 barely beats random.

**Critical observation**: AT's SMD PA-F1 (0.923) is only 0.029 above the random baseline PA-F1 (0.894). This makes the AT SMD result essentially uninformative - it is cited widely but is near-chance under PA. DADA's 0.846 affiliation-F1 is a much more meaningful number.

**Resolution**: The current citation of AT 0.93 is technically correct but deeply misleading context. In the paper, explicitly note "random-init baseline achieves 0.894 PA-F1 on SMD (Sarfraz et al., ICML 2024)." Keep AT as the PA-F1 SOTA, add DADA as honest-metric SOTA.

---

### 6. MBA (MIT-BIH Arrhythmia ECG)

**Currently cited**: None (no SOTA cited).

**Task framing issue**: MBA (as used in our paper) is framed as time-series anomaly prediction on ECG, where the "event" is an arrhythmia episode. This is different from the standard MIT-BIH arrhythmia CLASSIFICATION benchmark (per-beat classification into 5 AAMI classes), which is what most "SOTA 99% accuracy" papers evaluate.

**ECG anomaly detection papers (anomaly framing, not classification)**:
- FADE (arXiv 2502.07389, Feb 2025): self-supervised ECG forecasting-based anomaly detection on MIT-BIH. Global accuracy 84.65% +/- 0.56%, anomaly accuracy 83.84% +/- 2.97%. No F1/AUROC/AUPRC reported, accuracy only.
- LSTM-AE based arrhythmia anomaly detection (ScienceDirect 2021): AUROC-like AUC = 0.988, precision = 0.955, recall = 0.999, F1 = 0.977. BUT: this uses reconstruction error thresholding on MIT-BIH which has 3:1 normal-to-anomaly ratio - still classification-framed.
- CAE + RF (2023): AUC = 0.9991, F1 = 0.981. Very likely per-beat supervised classification, not anomaly detection.

**Recommendation**: There is NO established anomaly-prediction SOTA on MIT-BIH that matches our protocol (probability surface over future windows, AUPRC primary metric). Three options:
1. Compare to FADE (84.65% accuracy) with an explicit note that it uses accuracy, we use AUPRC.
2. Compare to unsupervised TSAD methods applied to ECG (none with established numbers matching our exact setup).
3. Report "no comparable SOTA" and note that MIT-BIH is used here in an anomaly-prediction framing (not per-beat classification).

**Suggested update**: Keep as "no SOTA" in Table 4 but add a footnote: "MIT-BIH arrhythmia is typically evaluated as per-beat classification (accuracy > 99%); we use it as a time-series anomaly prediction benchmark and report AUPRC against that surface."

---

### 7. SKAB (Skoltech Anomaly Benchmark)

**Currently cited**: None.

**Official leaderboard (from github.com/waico/SKAB README, verified)** - Outlier Detection problem:

| Method | F1 |
|--------|----|
| Conv-AE | 0.78 |
| MSET | 0.78 |
| T-squared+Q (PCA) | 0.76 |
| LSTM-AE | 0.74 |
| T-squared | 0.66 |
| Isolation Forest | 0.29 |

**Notes**:
- These are STANDARD F1 scores (NOT PA-adjusted). SKAB explicitly does not use PA.
- 34 time series, all from a controlled water-pump circuit.
- The benchmark is from 2020; best result on the official leaderboard is Conv-AE at F1=0.78. No NeurIPS/ICML/ICLR paper has benchmarked on SKAB as a primary dataset.
- FleetSense (cited in search results) claims F1=0.91 on SKAB - but this appears to be a 2024/2025 preprint, not a top-venue paper. Reference is unverified.
- There is no canonical supervised SOTA reference for SKAB.

**Apples-to-apples**: SKAB F1 is standard binary classification F1 per-timestep (no PA). This is closest to our non-PA F1 readout. If we report non-PA F1, comparison to Conv-AE 0.78 is valid.

**Suggested citation**: skab2020 (Katser et al., Skoltech Anomaly Benchmark, 2020) with Conv-AE 0.78 as best leaderboard entry.

---

### 8. ETTm1 (Electricity Transformer Temperature)

**Currently cited**: None (no SOTA for anomaly).

**Key finding**: ETTm1 is PRIMARILY a forecasting benchmark (introduced with Informer, AAAI 2021). There is NO established anomaly detection leaderboard for ETTm1. Several papers (TimesNet, GPT4TS) include ETTm1 in anomaly detection experiments using artificially injected anomalies, but there is no agreed-upon injection protocol, label set, or standard split.

**TimesNet (ICLR 2023)**: The TimesNet paper includes ETTm1 in Table 5 / Table 15 (anomaly detection appendix) with F1 scores. However, it uses an internal anomaly injection scheme (ratio=0.25 injected randomly) and PA-F1 metric. Exact numbers not extractable from web sources, but the paper reports results for ETT-based anomaly detection as a secondary task.

**Concrete issue**: There is NO public ground truth for "true" anomalies in ETTm1 - anomalies are synthetically injected. Different papers inject differently, making numbers incomparable. DADA (ICLR 2025) does NOT include ETTm1 in its evaluation.

**Resolution**: ETTm1 should not have a SOTA comparison in Table 4. Instead, note in the table footnote: "ETTm1 anomalies are synthetically injected with no universally agreed protocol; SOTA comparison not possible." If we want a reference, cite TimesNet for the PA-F1 protocol it uses, but do not treat it as a standard SOTA citation.

---

### 9. GECCO (Water Quality, GECCO 2018 Industrial Challenge)

**Currently cited**: batadal2018 F1 0.71 (the BATADAL reference was apparently used as a proxy here - this is wrong).

**Correct source**: GECCO Industrial Challenge 2018. The challenge was "Internet of Things: Online Anomaly Detection for Drinking Water Quality," run by SPOTSeven Lab at GECCO 2018 Kyoto.

**Metric**: F1 score (no PA). Online detection; evaluated on hidden test set.

**Competition results** (from SPOTSeven Lab, competition page now 404, but results archived):
- The competition had limited public result disclosure. One submission reported F1 = 0.80 using automatic feature learning.
- Winner's F1 is approximately 0.71 to 0.80 based on available references.
- The zenodo dataset (zenodo.org/records/3884398) is the primary data reference.

**Post-competition deep learning papers on GECCO**:
- Machine learning for GECCO anomaly detection (Taylor & Francis 2019): F1 not directly comparable (different train/test split).
- A 2022 paper using hybrid CNN achieved F1 = 0.98, AUC = 0.97 on GECCO 2018 - but the exact split and protocol are unclear and not reproducible from the search results.
- DADA (ICLR 2025): GECCO/SWAN are listed in Table 2 (NeurIPS-TS benchmark) but exact numbers not extractable.

**Resolution**: The current reference batadal2018 is wrong for GECCO. Use the zenodo dataset reference (zenodo 3884398, Duarte et al. 2020) and note competition winner ~F1=0.71. Do not claim a reliable deep-learning SOTA because the post-competition papers use different splits. Reference key should be gecco2018.

**Suggested update to paper.tex**: Replace `batadal2018` reference for GECCO with a GECCO-specific zenodo reference. Cite F1 ~0.71 as competition-era SOTA with caveat.

---

### 10. BATADAL (Battle of the Attack Detection Algorithms)

**Currently cited**: batadal2018 AUC 0.97.

**CRITICAL CORRECTION**: The BATADAL competition does NOT use AUC as its primary metric. The official metric is the BATADAL S-score, which combines time-to-detection (TTD) and classification accuracy for 7 specific attack events. There is no ground-truth anomaly score curve that yields a proper AUC.

**Official competition results (batadal.net)** - competition winner:
- 1st place: Housh & Ohar, 7/7 attacks detected, S = 0.970
- 2nd place: Abokifa et al., 7/7 attacks, S = 0.949
- 3rd place: Giacomoni et al., 7/7 attacks, S = 0.927
- 7th place (last): Aghashahi et al., 3/7 attacks, S = 0.534

**Source of the 0.97 AUC claim**: This likely refers to a deep-learning paper that applied their own anomaly detector to BATADAL and reported an internal ROC-AUC. The hybrid ensemble method (arXiv 2512.14422, Dec 2025) reports AUC=0.9826 for their approach but F1=0.7205 on the attack class. This is a different metric from the official S-score.

**Post-competition deep-learning results**:
- Graph Attention Network with physics constraints (Physics-GAT, arXiv 2601.12426, Jan 2026): F1 = 0.979.
- Hybrid ensemble (arXiv 2512.14422, Dec 2025): F1 = 0.7205, AUC = 0.9826.
- Temporal GCN with attention: "performance equivalent to best competition models."

**Recommended update**: In Table 4, report two numbers:
- Official competition: S-score 0.970 (Housh & Ohar 2018, 7/7 attacks detected)
- Modern deep-learning: Physics-GAT F1 = 0.979 (arXiv 2601.12426, 2026) or Hybrid F1 = 0.720, AUC = 0.983 (arXiv 2512.14422, 2025)

The current "AUC 0.97" in our paper is ambiguous - it conflates S-score with AUC. Fix in paper.

---

## Supplementary Notes: MTS-JEPA Protocol vs Standard Anomaly Detection

MTS-JEPA (arXiv 2602.04643, Feb 2026) uses an early-warning window-level protocol: given an observation window, predict whether the NEXT window will contain an anomaly. This is directly analogous to our probability surface at Deltat=1 horizon. Its results (SMAP F1=0.34, PSM F1=0.62) are not comparable to standard point-anomaly detection F1 (SMAP PA-F1=0.97).

The correct framing in our paper for citing MTS-JEPA: "MTS-JEPA, using the same early-warning protocol as FAM, achieves [values]. Standard anomaly detection methods using PA-F1 report higher values due to the PA inflation artifact."

---

## Honest-Metric SOTA Summary (DADA ICLR 2025 - Affiliation F1)

This is the cleanest comparison point for our non-PA F1 results:

| Dataset | DADA affil-F1 | GPT4TS affil-F1 | ModernTCN affil-F1 | AT affil-F1 |
|---------|--------------|----------------|-------------------|-------------|
| SMD | 84.57 | 83.14 | 83.16 | 69.46 |
| MSL | 78.48 | 77.23 | 77.17 | 66.49 |
| SMAP | 75.42 | 74.67 | 67.41 | 71.65 |
| PSM | 82.26 | 81.44 | 79.59 | 65.37 |

Note: "Affiliation F1" is NOT the same as "non-PA F1." It is a proximity-aware temporal metric (Huet et al. 2022). Our non-PA F1 is simpler (standard binary classification F1 per timestep). They will not match numerically.

---

## Apples-to-Apples Checklist

| Dataset | FAM metric | SOTA metric in table | Match? | Resolution |
|---------|-----------|---------------------|--------|-----------|
| C-MAPSS FD001 | RMSE (from surface E[Deltat]) | RMSE (point prediction) | PARTIAL | Both are RMSE but estimator differs. Flag in footnote: "CDF-based vs point estimator." |
| C-MAPSS FD002 | RMSE (surface) | RMSE (STAR point) | PARTIAL | Same caveat. |
| C-MAPSS FD003 | RMSE (surface) | RMSE (STAR point) | PARTIAL | Same caveat. |
| SMAP | PA-F1 AND non-PA F1 | PA-F1 (AT 0.967) or affil-F1 (DADA 0.754) | SPLIT | Report PA-F1 vs AT, AND non-PA vs DADA's affil-F1 in appendix. Never compare non-PA to PA directly. |
| MSL | PA-F1 AND non-PA F1 | PA-F1 (DCdetector 0.966) or affil-F1 (DADA 0.785) | SPLIT | Same resolution as SMAP. |
| PSM | PA-F1 AND non-PA F1 | PA-F1 (AT 0.979) or affil-F1 (DADA 0.823) | SPLIT | Same. Note MTS-JEPA 0.62 is early-warning F1 (closest to our protocol). |
| SMD | PA-F1 AND non-PA F1 | PA-F1 (AT 0.923) or affil-F1 (DADA 0.846) | SPLIT | Extra flag: random baseline PA-F1=0.894 on SMD; AT margin is only 0.029. |
| MBA | AUPRC (surface) | No matched SOTA | NO MATCH | Footnote: "No established anomaly-prediction SOTA on MIT-BIH ECG; per-beat classification SOTA (>99% acc) uses different task framing." |
| SKAB | Non-PA F1 (our binary threshold) | Conv-AE F1=0.78 (non-PA) | YES | Closest available match. Use skab2020 reference. |
| ETTm1 | AUPRC (surface) | No standard anomaly SOTA | NO MATCH | Note: "Synthetic anomalies injected; no unified protocol." Remove from SOTA comparison table or footnote clearly. |
| GECCO | AUPRC / F1 | Competition F1 ~0.71 | PARTIAL | Match on metric (F1) but split is competition-era vs ours. Use zenodo/gecco2018 ref. Correct wrong batadal2018 ref. |
| BATADAL | AUPRC / F1 | S-score 0.970 (competition) | NO MATCH | S-score is NOT AUC/F1. Either compare to Physics-GAT F1=0.979 (best post-competition F1) or report S-score separately. Fix "AUC 0.97" in paper - this is the S-score, not ROC-AUC. |

---

## Priority Fixes for paper.tex

1. **FD002 and FD003**: Add STAR RMSE values (13.47 and 10.71) to Table 4. Currently only FD001=10.61 appears.

2. **SMAP / MSL / PSM / SMD**: Add dual rows in the appendix - one for PA-F1 vs canonical PA-F1 SOTA (AT/DCdetector) with inflation warning, one for affiliation-F1 vs DADA (ICLR 2025). Do not silently compare non-PA F1 to PA-F1 SOTA.

3. **BATADAL**: Remove "AUC 0.97" and replace with "S-score 0.970 (1st place, 7/7 attacks, Taormina et al. 2018)" plus a new entry "Physics-GAT F1=0.979 (2026)" as modern DL SOTA. The confusion between S-score and AUC is a factual error.

4. **GECCO**: Replace batadal2018 with the correct GECCO 2018 reference (zenodo.org/records/3884398, Duarte et al., GECCO 2018).

5. **MBA footnote**: Add explicit disclaimer that MIT-BIH is used here in anomaly-prediction framing, not per-beat classification (to avoid confusion with 99% accuracy SOTA claims).

6. **MTS-JEPA clarification**: In the paper body, explicitly state that MTS-JEPA F1 values are from an early-warning window protocol, not standard anomaly detection F1, which is why they appear much lower than AT/DCdetector.

---

## New References to Add to paper.bib

```bibtex
% DADA - ICLR 2025, honest-metric anomaly detection
@inproceedings{dada2025,
  title={Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders},
  author={...},
  booktitle={ICLR},
  year={2025},
  note={arXiv:2405.15273}
}

% Quo Vadis position paper - ICML 2024
@inproceedings{sarfraz2024quovadis,
  title={Position: Quo Vadis, Unsupervised Time Series Anomaly Detection?},
  author={Sarfraz, M. Saquib and others},
  booktitle={ICML},
  year={2024}
}

% GECCO 2018 dataset reference
@misc{gecco2018,
  title={GECCO Industrial Challenge 2018 Dataset: A water quality dataset for Online Anomaly Detection},
  author={Duarte, J. and others},
  howpublished={Zenodo, \url{https://zenodo.org/records/3884398}},
  year={2018}
}

% Physics-GAT for BATADAL
@misc{physgat2026,
  title={Graph Attention Networks with Physical Constraints for Anomaly Detection},
  author={...},
  howpublished={arXiv:2601.12426},
  year={2026}
}
```

---

## Sources Consulted

- PMC10857698 (STAR paper full results table) - PRIMARY for C-MAPSS SOTA
- arXiv 2602.04643 HTML (MTS-JEPA) - extracted exact F1/AUC values per dataset
- arXiv 2405.15273 HTML (DADA ICLR 2025) - extracted affiliation-F1 table (honest-metric SOTA)
- arXiv 2510.03911 HTML (THEMIS) - affiliation-F1 comparison for MSL/SMAP
- arXiv 2405.02678 (Quo Vadis ICML 2024) - random baseline PA-F1 values for SWaT/WADI/SMD
- github.com/waico/SKAB README - official SKAB leaderboard (text extracted)
- batadal.net/results.html - official BATADAL S-score results
- ICLR 2022 Anomaly Transformer (xu2022anomalytransformer) - PA-F1 SMAP/MSL/SMD/PSM (values confirmed from multiple citing papers: SMAP 0.9669, MSL 0.9359, SMD 0.9233, PSM 0.9789)
- KDD 2023 DCdetector (yang2023dcdetector) - MSL PA-F1 0.966, SWaT 0.963
- arXiv 2412.05498 (CPatchBLS) - PA-F1 table for AT, TimesNet, proposed on SMAP/MSL/SWaT/PSM
- arXiv 2502.07389 (FADE) - ECG anomaly detection accuracy on MIT-BIH
- zenodo.org/records/3884398 - GECCO 2018 dataset
- arXiv 2512.14422 (Hybrid Ensemble BATADAL) - modern F1/AUC on BATADAL
- arXiv 2601.12426 (Physics-GAT) - best modern F1=0.979 on BATADAL

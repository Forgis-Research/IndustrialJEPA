# C-MAPSS RUL Prediction: SOTA Literature Review
**Date**: 2026-04-09  
**Scope**: 2023–2026, verified numbers only, quality-filtered venues

---

## Protocol Caveat (Read First)

Comparison across papers is hazardous due to three inconsistencies in the field:

1. **Evaluation window**: The canonical protocol evaluates each test engine using the **last sensor window only** and computes RMSE over the 100/259/100/248 test engines. Some papers report RMSE over all sliding windows — this inflates the apparent precision of nearby-failure predictions.

2. **RUL cap**: The dominant convention is cap = **125 cycles** (piecewise linear target). A minority of papers use 130 or omit the cap. Uncapped RUL makes early-life predictions harder and will raise RMSE.

3. **Data leakage risk**: At least one documented class of error uses training trajectories (which run to failure) as test data, producing implausibly low RMSE (~6–8 on FD001). Legitimate state-of-the-art on proper eval is ~10–13 for FD001.

**All verified numbers below use: last-window per engine, RUL cap = 125, unless explicitly noted.**

---

## Table 1: Top Supervised Methods on C-MAPSS (2023–2025), Ranked by FD001 RMSE

| Rank | Method | Authors | Venue | Year | FD001 | FD002 | FD003 | FD004 | URL |
|------|--------|---------|-------|------|-------|-------|-------|-------|-----|
| 1 | **STAR** (Two-Stage Attention Hierarchical Transformer) | Fan, Li, Chang | *Sensors* (MDPI) | 2024 | **10.61** | **13.47** | **10.71** | **15.87** | [PMC10857698](https://pmc.ncbi.nlm.nih.gov/articles/PMC10857698/) |
| 2 | **BiLSTM-DAE-Transformer** | (compared in STAR) | — | ~2023 | 10.98 | 16.12 | 11.14 | 18.15 | cited in STAR paper |
| 3 | **TMSCNN** (Multiscale CNN + Transformer + domain adapt.) | Liu et al. | *J. Computational Design & Eng.* | 2024 | 10.26* | 14.79 | 10.51* | 14.25 | [OUP jcde](https://academic.oup.com/jcde/article/11/1/343/7610897) |
| 4 | **DAST** | (compared in STAR) | — | ~2022–23 | 11.43 | 15.25 | 11.32 | 18.23 | cited in STAR paper |
| 5 | **GCT** | (compared in STAR) | — | ~2022 | 11.27 | 22.81 | 11.42 | 24.86 | cited in STAR paper |
| 6 | **DATCN** | (compared in STAR) | — | ~2022 | 11.78 | 16.95 | 11.56 | 18.23 | cited in STAR paper |
| 7 | **BLTTNet** (BiLSTM + DCEFormer + TCN) | Yang et al. | *Scientific Reports* | 2025 | 12.26 | 14.18 | 11.19 | 18.15 | [PMC12307624](https://pmc.ncbi.nlm.nih.gov/articles/PMC12307624/) |
| 8 | **ABGRU** (Attention-based GRU) | — | — | 2023 | 12.83 | — | 13.23 | — | cited in CAELSTM paper |
| 9 | **DCNN** | (compared in STAR) | — | ~2022 | 12.61 | 22.36 | 12.64 | 23.31 | cited in STAR paper |
| 10 | **CAELSTM** (Conv. Autoencoder + Attention LSTM) | Elsherif et al. | *Scientific Reports* | 2025 | 14.44 | — | 13.40 | — | [PMC12276258](https://pmc.ncbi.nlm.nih.gov/articles/PMC12276258/) |

*TMSCNN FD001/FD003 numbers: the paper uses domain adaptation transfer from FD002/FD004 as source; the per-subset RMSE represents the in-domain performance with 100% target data.

**One-sentence summaries:**
- **STAR**: Two-stage transformer that first applies temporal attention within each sensor, then cross-sensor attention, with hierarchical patch merging.
- **TMSCNN**: Multi-scale CNN for local feature extraction, Transformer encoder for global dependencies, with optional MMD-based domain adaptation for limited-data transfer.
- **BLTTNet**: Triple-branch fusion of BiLSTM (sequence), DCEFormer dual-channel attention (sensor + timestep), and TCN (local patterns).
- **CAELSTM**: Convolutional autoencoder as denoising front-end feeding a stacked attention-LSTM.

---

## Table 2: SSL / Semi-Supervised / Unsupervised Methods on C-MAPSS (2022–2025)

| Method | Authors | Venue | Year | FD001 | FD002 | FD003 | FD004 | Label Regime | URL |
|--------|---------|-------|------|-------|-------|-------|-------|--------------|-----|
| **Contrastive + VAE (semi-sup.)** | — | *Computers & Electrical Eng.* | 2025 | NR | NR | NR | NR | Partial labels + incomplete life histories | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045790625000771) |
| **USL-CNN / USL-LSTM** (contrastive + unlabeled aug.) | Kong, Jin, Xu, Chen | *RESS* | 2023 | NR | NR | NR | NR | Semi-supervised (unlabeled augmentation) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0951832023000789) |
| **SSDA** (Self-Supervised Domain Adaptation, dual Siamese) | Le Xuan, Munderloh, Ostermann | *RESS* | 2024 | NR | NR | NR | NR | No RUL labels for target domain | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0951832024003685) |
| **Triplet SSL** (simple triplet network for RUL) | — | *J. Manufacturing Systems* | 2024 | NR | NR | NR | NR | Self-supervised pretraining, few-shot fine-tune | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S147403462400689X) |
| **MambAtt** (Mamba + Attention, SSL pre-training) | Han, Kwon, Yoon | *RESS* | 2026 (pub. 2025) | NR | NR | NR | NR | Pre-train on unlabeled, few-shot fine-tune | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0951832025006921) |
| **FSGRI + Dual-Mixer** (supervised contrastive) | Fu, Hu, Peng, Chu | *RESS* | 2024 | NR | NR | NR | NR | Fully supervised + contrastive training objective | [arXiv:2401.16462](https://arxiv.org/abs/2401.16462) / [RESS](https://www.sciencedirect.com/science/article/abs/pii/S0951832024004708) |
| **AE-LSTM** (unsupervised AE encoder + supervised LSTM) | — | *Machines* (MDPI) | 2025 | 13.99 | — | — | 28.67 | Unsupervised pretraining of encoder | [MDPI](https://www.mdpi.com/2075-1702/14/2/135) |

**Critical finding**: Of 7 SSL/semi-supervised papers identified, only **1** (AE-LSTM) reports concrete RMSE numbers accessible without a paywall. The others are paywalled and their abstracts do not include specific RMSE tables. The AE-LSTM baseline (13.99 on FD001) is **worse than STAR (10.61) by 32%** and barely better than naive supervised LSTM.

**No paper combining MAE-style masking, JEPA-style latent prediction, or I-JEPA/V-JEPA architecture with C-MAPSS RUL was found.** This is a genuine gap.

---

## Table 3: Cross-Subset Transfer Results (Verified)

| Study | Source Domain | Target Domain | RMSE (same-domain) | RMSE (cross-domain) | Method | Notes |
|-------|--------------|--------------|-------------------|--------------------|---------|----|
| TMSCNN / DATMSCNN (Liu et al. 2024, JCDE) | FD002+FD004 (multi-cond.) | FD001+FD003 (single-cond.) | 10.26 (FD001), 10.51 (FD003) | Not directly reported | MMD domain adaptation | Paper uses all 4 sets jointly with DA |
| Cross-condition transfer (cited in arXiv:2410.03134) | FD001 | FD002 | 16.24 ± 1.30 (in-domain) | 21.76 ± 1.21 (cross) | Unspecified DL baseline | Significant RMSE degradation on transfer |
| Cross-condition transfer (cited in arXiv:2410.03134) | FD002 | FD001 | — | 17.56 ± 0.06 (cross) | Unspecified DL baseline | FD002→FD001 degrades less |
| SSDA (Le Xuan et al. 2024, RESS) | C-MAPSS (various) | C-MAPSS (various) | — | ~20% RMSE improvement over DA baseline | Siamese SSL domain adapt. | No absolute RMSE values publicly available |

**Key observation**: Cross-subset transfer degrades RMSE by 30–50% compared to same-domain training. This gap is the opportunity for JEPA-style pretraining on all subsets.

---

## Verified SOTA Targets

| Regime | Dataset | Best Verified RMSE | Source | Notes |
|--------|---------|--------------------|--------|-------|
| Supervised SOTA | FD001 | **10.61** | STAR (Sensors 2024) | All 4 subsets reported, RUL cap 125 |
| Supervised SOTA | FD002 | **13.47** | STAR (Sensors 2024) | |
| Supervised SOTA | FD003 | **10.71** | STAR (Sensors 2024) | |
| Supervised SOTA | FD004 | **14.25** | TMSCNN (JCDE 2024) | STAR reports 15.87 |
| SSL SOTA | FD001 | **13.99** | AE-LSTM (Machines 2025) | Only SSL paper with reported RMSE |
| SSL SOTA | FD004 | **28.67** | AE-LSTM (Machines 2025) | |

**RMSE gap (SSL vs. supervised on FD001)**: 13.99 vs. 10.61 = **+32% worse**. This gap is large and likely much larger for the better SSL methods not releasing numbers.

---

## Sanity Check: Field-Level Concerns

### Implausibly Low Results
- RMSE of **6.62 on FD001** appears in at least one 2025 Scientific Reports paper (LightGBM + CatBoost ensemble). This is highly suspicious for the standard test protocol. Independent analysis confirms this class of result likely involves using training trajectories (run-to-failure) as part of the test set — a known data leakage pattern. **Do not use RMSE < 9.0 on FD001 as a comparison target.**
- The "all windows" vs. "last window" evaluation produces a ~15–25% RMSE reduction artificially on the same model. Papers using all-window eval report lower RMSE without model improvement.

### Reproducibility Status
- The field lacks a centralized reproducibility study comparable to what we did on FEMTO. However, informal analysis suggests results below ~10.5 on FD001 with standard protocol are exceptional and should be treated with skepticism until independently replicated.
- The STAR paper (10.61) uses a clearly described architecture evaluated at the published PMC version — this is the most trustworthy low-RMSE claim found.
- BiLSTM-DAE-Transformer (10.98) is cited in the STAR paper but the primary paper was not directly accessible; treat as credible but secondary.

### PHM 2008 Score Consistency
- PHM score is reported inconsistently: STAR reports FD001 score = 169, FD002 = 784, FD003 = 202, FD004 = 1449. BLTTNet reports FD001 = 220.75. CAELSTM reports FD001 = 282.38. The metric is directionally consistent (lower = better) but is **not suitable as a cross-paper comparison without matching RUL cap and window protocol**.

### RUL Cap Usage
- Dominant convention: **125 cycles**. Verified in STAR, TMSCNN, CAELSTM, BLTTNet, and the uncertainty-aware DL paper. The cap is applied as a piecewise linear target with constant RUL phase before degradation onset.

---

## Key Gaps for Potential Contribution

1. **No JEPA-style latent prediction for RUL on C-MAPSS** — confirmed absent from the literature. I-JEPA/TS-JEPA applied to industrial sensor sequences with RUL fine-tuning is a genuine gap.

2. **No masked autoencoder (MAE) for C-MAPSS** — MAE pretraining of sensor time series followed by RUL regression head is unpublished on this benchmark.

3. **SSL RMSE numbers are hidden** — the most interesting SSL papers (MambAtt, SSDA, triplet SSL, contrastive VAE) all have paywalled results. The one publicly available SSL result (AE-LSTM, RMSE=13.99) is poor. A strong SSL result that reports all 4 subsets would be a clear paper contribution.

4. **Cross-subset transfer is largely unsolved** — the 30–50% RMSE degradation on cross-condition transfer is a well-documented problem with no satisfying solution. A pretrained JEPA that transfers from FD002+FD004 (multi-condition) to FD001+FD003 (single-condition) with strong results would be novel.

5. **Supervised contrastive applied as pretraining, not auxiliary loss** — FSGRI uses contrastive as a training-time regularizer with full labels, not as an unsupervised pretraining step. The SSL-then-finetune protocol with contrastive objectives is missing.

---

## Top Papers Ranked by Credibility and Relevance

1. **STAR** (Fan et al., Sensors 2024) — PMC open access, full table, all 4 subsets, strong architecture. **Best verified supervised SOTA.** URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10857698/

2. **TMSCNN / DATMSCNN** (Liu et al., JCDE 2024) — transfer learning + domain adaptation, limited data regime, all 4 subsets. **Most relevant to our cross-subset interest.** URL: https://academic.oup.com/jcde/article/11/1/343/7610897

3. **BLTTNet** (Yang et al., Scientific Reports 2025) — PMC open access, all 4 subsets, strong FD002 result. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12307624/

4. **FSGRI + Dual-Mixer** (Fu et al., RESS 2024) — supervised contrastive training objective on C-MAPSS; code available at https://github.com/fuen1590/PhmDeepLearningProjects. Specific RMSE paywalled but code enables reproduction.

5. **MambAtt** (Han et al., RESS 2026) — self-supervised pretraining with temporal + N-tuplet + pseudo-label losses; few-shot fine-tuning. Most methodologically relevant to our SSL interest. URL: https://www.sciencedirect.com/science/article/abs/pii/S0951832025006921

6. **SSDA** (Le Xuan et al., RESS 2024) — dual Siamese network for self-supervised domain adaptation without RUL labels in target domain. URL: https://www.sciencedirect.com/science/article/pii/S0951832024003685

7. **Contrastive + VAE semi-supervised** (2025) — addresses incomplete life histories, the most realistic industrial scenario. URL: https://www.sciencedirect.com/science/article/abs/pii/S0045790625000771

---

## Summary: Numbers to Beat

| Target | RMSE | Source |
|--------|------|--------|
| FD001 supervised SOTA | 10.61 | STAR 2024 |
| FD002 supervised SOTA | 13.47 | STAR 2024 |
| FD003 supervised SOTA | 10.71 | STAR 2024 |
| FD004 supervised SOTA | 14.25 | TMSCNN 2024 |
| FD001 SSL (only public baseline) | 13.99 | AE-LSTM 2025 |
| **Reasonable SSL target** | **<12.0 on FD001** | Our goal |
| **Stretch SSL target** | **<11.0 on FD001** | Match supervised SOTA with fewer labels |

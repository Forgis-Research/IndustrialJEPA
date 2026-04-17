# Paper Quality Checklist

Audit date: 2026-04-09
Paper: `paper-neurips/paper.tex`
Ground truth: `mechanical-jepa/v8/RESULTS.md`, `mechanical-jepa/v8/results/hybrid_experiment.json`, `mechanical-jepa/v8/EXPERIMENT_LOG.md`

---

## 1. Figures Assessment

### Current state: ZERO figures in the paper. This is a critical gap for NeurIPS.

A typical NeurIPS paper has 4-7 figures. The absence of any figures will hurt reviewer impressions severely -- figures are the first thing reviewers scan.

### Existing plots available (PDF, publication-ready format)

All located in `mechanical-jepa/notebooks/plots/`:

| File | Description | Usable? |
|------|-------------|---------|
| `v8_rul_comparison.pdf` | Bar chart of RMSE across all methods | YES - main results figure |
| `v8_cross_dataset.pdf` | Cross-dataset transfer results | YES - transfer results figure |
| `v8_encoder_analysis.pdf` | PCA/correlation analysis of encoder representations | YES - mechanistic analysis figure |
| `v8_latent_trajectories.pdf` | Latent space trajectories over bearing lifetime | YES - qualitative analysis figure |
| `v8_pretrain_history.pdf` | JEPA pretraining loss curve | YES - appendix figure |

### Recommended figure plan (6 figures minimum)

#### Figure 1: Architecture Diagram (Section 3, after Sec 3.1)
- **Status: NEEDS TO BE CREATED**
- Content: Three-panel diagram showing (a) JEPA pretraining with masked patches, (b) temporal contrastive learning with triplet loss, (c) hybrid fusion architecture
- This is the single most important missing figure. Every NeurIPS methods paper needs an architecture diagram.
- Style: Use TikZ or a clean vector drawing tool (draw.io exported to PDF). Match NeurIPS column width (textwidth = ~5.5in). Use consistent color coding: blue for JEPA components, orange for contrastive, green for hybrid.
- Position: Should be `\begin{figure*}[t]` spanning full width, immediately after the Method section opener or after Problem Formulation.

#### Figure 2: Main Results Bar Chart (Section 4.2)
- **Status: USE `v8_rul_comparison.pdf`** (may need reformatting)
- Content: Grouped bar chart of RMSE for all 11 methods, color-coded by category (naive, end-to-end, handcrafted, self-supervised, hybrid)
- Publication-quality checklist: Verify font size >= 8pt, axis labels present, legend readable, error bars visible, bolded best result
- Position: Near Table 2 or replacing it. If both table and figure are kept, consider moving the figure to be the primary and the table to the appendix, or vice versa.
- Note: Reviewers often prefer tables for exact numbers and figures for visual comparison -- having both is fine if space permits.

#### Figure 3: Cross-Dataset Transfer (Section 4.3)
- **Status: USE `v8_cross_dataset.pdf`** (may need reformatting)
- Content: Grouped bars or heatmap showing within-dataset vs cross-dataset RMSE for elapsed time, JEPA, and contrastive
- Position: Near Table 3
- Publication-quality: Ensure the within vs cross contrast is visually obvious (e.g., dashed separator line, different shading)

#### Figure 4: Encoder Representation Analysis (Section 5.1)
- **Status: USE `v8_encoder_analysis.pdf`**
- Content: PCA scatter plots showing JEPA vs contrastive vs handcrafted latent spaces, colored by RUL or spectral centroid
- This is the key mechanistic insight figure. It should show visually why JEPA has low spectral centroid correlation (0.071) and contrastive has high (0.856).
- Position: Near Table 4 (encoder comparison table)
- Publication-quality: Use consistent colormap (e.g., viridis for RUL). Annotate correlation values directly on subplot.

#### Figure 5: Latent Trajectories (Section 5.1 or 5.2)
- **Status: USE `v8_latent_trajectories.pdf`**
- Content: Latent space trajectories over bearing lifetime showing how representations evolve from healthy to faulty
- Position: After Figure 4, to reinforce the mechanistic story
- Publication-quality: Use line plots with time/RUL on x-axis, ensure multiple episodes are shown with different line styles

#### Figure 6: Pretraining History (Appendix)
- **Status: USE `v8_pretrain_history.pdf`**
- Content: JEPA training loss over epochs showing oscillation and early convergence at epoch 2
- Position: Appendix (supports the claim about pretraining instability in Limitations)

### Figures that should be created new

1. **Architecture diagram (Figure 1)** -- highest priority, no existing version
2. **Grey swan conceptual figure** (optional, Introduction) -- a schematic showing the data paradox: abundant healthy data, scarce failure trajectories. Could use a timeline showing multiple healthy episodes and only 1-2 failure trajectories. This would make the "grey swan" framing more concrete.
3. **Dataset overview** (optional, Section 4.1 or Appendix) -- example raw vibration waveforms from healthy vs degraded bearings across FEMTO and XJTU-SY, showing the visual difference that the model learns to detect.

### Where to place figures in the paper

| Figure | Section | LaTeX position | Priority |
|--------|---------|---------------|----------|
| Architecture diagram | Sec 3 (Method) | `\begin{figure*}[t]` after Sec 3.1 | CRITICAL |
| Main results bar chart | Sec 4.2 | `\begin{figure}[t]` near Tab 2 | HIGH |
| Cross-dataset transfer | Sec 4.3 | `\begin{figure}[t]` near Tab 3 | HIGH |
| Encoder analysis | Sec 5.1 | `\begin{figure*}[t]` near Tab 4 | HIGH |
| Latent trajectories | Sec 5.1-5.2 | `\begin{figure}[t]` | MEDIUM |
| Pretrain history | Appendix | `\begin{figure}[h]` | LOW |

---

## 2. Citation Audit

### Entries with `author={various}` -- ALL NEED REAL AUTHOR NAMES

There are **14 bib entries** with "various" as author (either `author={various}` or `author={Zhang, various}` etc.). This is unacceptable for submission. Every entry needs complete author lists.

| Bib Key | Title | Issue |
|---------|-------|-------|
| `tsjepa2024` | TS-JEPA | `author={various}` -- need real authors |
| `mtsjepa2026` | MTS-JEPA | `author={various}` -- need real authors |
| `zhang2022tcntransformer` | Multi-Head Dual Sparse Self-Attention | `author={Zhang, various}` -- need full author list |
| `chen2024cnugrumha` | CNN-GRU-MHA for Bearing RUL | `author={Chen, various}` -- need full author list |
| `li2020fewshot` | Deep Transfer Learning Survey | `author={Li, various}` -- need full author list |
| `openMAE2025` | OpenMAE | `author={various}` -- need real authors |
| `ttsnet2025` | TTSNet | `author={various}` -- need real authors |
| `mdsct2024` | Conv Attention + Enhanced Transformer | `author={various}` -- need real authors |
| `nomi2024` | Neural ODE for Bearing RUL | `author={various}` -- need real authors |
| `greyswan_weather2024` | AI Weather Models Grey Swan | `author={various}` -- need real authors |
| `greyswan_factory2026` | Gray Swan Factory | `author={various}` -- need real authors |
| `greyswan_supplychain2025` | Supply Chain Grey Swans | `author={various}` -- need real authors |
| `physics_augment_rul2024` | Physics-Informed Data Augmentation | `author={various}` -- need real authors |
| `physics_rul_review2024` | Physics-Informed RUL Review | `author={various}` -- need real authors |

### Incomplete entries (missing venue, DOI, or pages)

| Bib Key | Issue |
|---------|-------|
| `lecun2022path` | `journal={OpenReview}` -- this is a position paper, not a journal. Should be `howpublished={\url{https://openreview.net/pdf?id=BZ5a1r-kVsf}}` or `journal={Technical Report}` |
| `wang2024brainjepa` | `author={Wang, Zijian and others}` -- "and others" is not proper BibTeX. Use full author list or at least first 3+ authors with "and others" |
| `nam2026cjepa` | `author={Nam, Hyunji and others}` -- same issue |
| `wang2018xjtusybearing` | Key says 2018 but `year={2020}` -- inconsistent |
| `chen2018neural` | Uses `booktitle={NeurIPS}` but declared as `@article` -- should be `@inproceedings` |
| `li2018dcnn` | Uses `booktitle={Reliability Engineering \& System Safety}` but declared as `@inproceedings` -- this is a journal article, should be `@article` with `journal=` field |
| `mdsct2024` | DOI has `e38xxx` placeholder -- need real DOI |
| `nomi2024` | DOI has `007286` -- verify this is correct |
| `ansari2024chronos` | Still arXiv -- Chronos was published; check for proceedings version |
| `rasul2024lagllama` | Still arXiv -- check if accepted at a venue |
| `balestriero2025lejepa` | arXiv only -- verify if published at a venue |

### Missing citations that should be included

The following important papers are not cited but are relevant:

1. **MAE (He et al., CVPR 2022)** -- Masked Autoencoders Are Scalable Vision Learners. The paper compares against masked reconstruction approaches but doesn't cite the foundational MAE paper.
2. **BYOL / VICReg / Barlow Twins** -- The paper uses variance regularization to prevent collapse, which relates to the VICReg line of work (Bardes et al., ICLR 2022). Should cite at least one collapse prevention method.
3. **DINO / DINOv2** -- The EMA target encoder is identical to DINO's architecture. Should acknowledge this lineage.
4. **PatchTST** is cited but the paper doesn't cite **iTransformer (Liu et al., ICLR 2024)** which is a more recent and widely-cited time series Transformer.
5. **Timer / TimesFM v2 / Moirai** -- More recent time series foundation models (2024-2025) that strengthen the "foundation models solve a different problem" argument.

### Citation format issues

- The paper uses `\citep` and `\citet` correctly throughout (good).
- `wang2018xjtusybearing` is cited as the XJTU-SY dataset but the year mismatch (key=2018, year=2020) may cause confusion.

---

## 3. Accuracy Verification

### RMSE values in the main results table (Table 2)

All values cross-checked against `RESULTS.md`:

| Method | Paper RMSE | RESULTS.md RMSE | Match? |
|--------|-----------|-----------------|--------|
| Constant mean | 0.290 | 0.290 | YES |
| Elapsed time only | 0.224 | 0.224 | YES |
| Envelope RMS + LSTM | 0.287 | 0.287 | YES |
| Random JEPA + LSTM | 0.221 | 0.221 | YES |
| End-to-end CNN-LSTM | 0.195 | 0.195 | YES |
| CNN-GRU-MHA | 0.185 | 0.185 | YES |
| HC + LSTM | 0.177 | 0.177 | YES |
| JEPA + LSTM | 0.189 | 0.189 | YES |
| HC + MLP | 0.085 | 0.085 | YES |
| Transformer + HC | 0.070 | 0.070 | YES |
| **JEPA + HC Transformer** | **0.055** | **0.0553** | ROUNDED (see below) |

### Standard deviations

| Method | Paper std | Source std | Match? |
|--------|----------|-----------|--------|
| Envelope RMS + LSTM | 0.001 | 0.001 | YES |
| Random JEPA + LSTM | 0.008 | 0.008 | YES |
| CNN-LSTM | 0.005 | 0.005 | YES |
| CNN-GRU-MHA | 0.005 | 0.005 | YES |
| HC + LSTM | 0.016 | 0.016 | YES |
| HC + MLP | 0.004 | 0.004 | YES |
| Transformer + HC | 0.006 | 0.006 | YES |
| JEPA + LSTM | 0.015 | 0.015 | YES |
| JEPA + HC Transformer | 0.004 | 0.0041 | ROUNDED (acceptable) |

### Cross-dataset transfer table (Table 3)

All values cross-checked against `RESULTS.md`:

| Value | Paper | RESULTS.md | Match? |
|-------|-------|------------|--------|
| FEMTO->FEMTO elapsed | 0.027 | 0.027 | YES |
| FEMTO->FEMTO JEPA | 0.113 +/- 0.011 | 0.113 +/- 0.011 | YES |
| FEMTO->FEMTO contrastive | 0.142 +/- 0.012 | 0.142 +/- 0.012 | YES |
| XJTU->XJTU elapsed | 0.159 | 0.159 | YES |
| XJTU->XJTU JEPA | 0.195 | 0.195 | YES |
| XJTU->XJTU contrastive | 0.214 | 0.214 | YES |
| FEMTO->XJTU elapsed | 0.367 | 0.367 | YES |
| FEMTO->XJTU JEPA | 0.280 +/- 0.007 | 0.280 +/- 0.007 | YES |
| FEMTO->XJTU contrastive | 0.227 +/- 0.015 | 0.227 +/- 0.015 | YES |
| XJTU->FEMTO elapsed | 0.336 | 0.336 | YES |
| XJTU->FEMTO JEPA | 0.403 | 0.403 | YES |
| XJTU->FEMTO contrastive | 0.309 +/- 0.007 | 0.309 +/- 0.007 | YES |

### Percentage improvements

| Claim | Calculation | Paper value | Correct? |
|-------|------------|-------------|----------|
| Hybrid vs time-only | (0.224-0.055)/0.224 | 75.5% | YES (75.4% from rounded, 75.3% from exact 0.0553; paper says 75.5%) |
| Hybrid vs Transformer+HC | (0.070-0.055)/0.070 | 21.4% | **DISCREPANCY** (see below) |
| JEPA vs time-only | (0.224-0.189)/0.224 | 15.8% | YES (15.6% exactly, rounds to 15.8% -- actually (0.224-0.189)/0.224 = 0.1563 = 15.6%) |
| Cross-dataset contrastive vs time | (0.367-0.227)/0.367 | 38% | YES (38.1%) |
| Contrastive vs JEPA (FEMTO->XJTU) | (0.280-0.227)/0.280 | 18.8% | YES (18.9%) |

### THE 21.4% vs 20.7% DISCREPANCY

This is the most important accuracy issue in the paper.

**Ground truth** (from `hybrid_experiment.json`):
- Transformer+HC RMSE: 0.0697
- Hybrid RMSE: 0.0553
- Improvement: (0.0697 - 0.0553) / 0.0697 = 20.66% ~ **20.7%**

**Paper claims** (using rounded RMSE values):
- Transformer+HC RMSE: 0.070
- Hybrid RMSE: 0.055
- Improvement: (0.070 - 0.055) / 0.070 = 21.43% ~ **21.4%**

The paper rounds both RMSE values to 3 decimal places, then recomputes the percentage from the rounded values, inflating the improvement from 20.7% to 21.4%.

**Recommendation**: Use 21.4% if the table reports 0.070 and 0.055 (internally consistent with the rounded values shown). OR report 4 significant figures in the table (0.0553 and 0.0697) and use 20.7%. The current approach is defensible but the abstract and contributions should match the table. Currently the paper uses 21.4% everywhere, which is consistent with the table values. However, the EXPERIMENT_LOG.md and the git commit message say "20.7%". Choose one and be consistent. If a reviewer checks the JSON, they will find 20.7%.

**Safest fix**: Report RMSE to 4 significant figures: 0.0553 +/- 0.0041 and 0.0697 +/- 0.006, and use 20.7%. Alternatively, keep 3 decimal places and note that the percentage is approximate.

### The 75.5% calculation

- From rounded: (0.224 - 0.055) / 0.224 = 75.4%
- From exact: (0.224 - 0.0553) / 0.224 = 75.3%
- Paper says 75.5%

This is off by 0.1-0.2 percentage points. The `hybrid_experiment.json` reports `vs_elapsed_time_pct: 75.5`, so the JSON itself uses this value. But recomputing: (0.224 - 0.0553) / 0.224 = 0.75313 = 75.3%. The JSON may have computed it from slightly different baseline values.

**Recommendation**: Verify the exact elapsed-time baseline RMSE used. If it is 0.2244 (rounded to 0.224), then (0.2244 - 0.0553)/0.2244 = 75.4%. The difference is negligible but worth confirming.

### The 15.8% claim for JEPA vs time-only

- (0.224 - 0.189) / 0.224 = 0.15625 = 15.6%
- RESULTS.md says 15.8%
- Paper says 15.8%

This is slightly inflated (15.6% vs 15.8%). The discrepancy is small but worth noting. Likely computed from non-rounded values.

### p-values

| Claim | Paper p-value | RESULTS.md p-value | Match? |
|-------|--------------|-------------------|--------|
| JEPA+LSTM vs elapsed time | 0.010 | 0.010 | YES |
| Random JEPA vs elapsed time | 0.44 (abstract) / "not significant" | 0.44 / 0.435 | YES |
| JEPA+LSTM vs HC+LSTM | 0.40 | 0.402 | YES (rounded) |
| Contrastive vs JEPA (FEMTO->XJTU) | <0.001 | <0.001 | YES |

### Architecture parameters

| Parameter | Paper | Source | Match? |
|-----------|-------|--------|--------|
| Encoder: 4.0M params | 4.0M | Sec 3.2 | YES (stated in method) |
| Predictor: 0.9M params | 0.9M | Sec 3.2 | YES |
| Pretraining windows | 33,939 | RESULTS.md: 33,939 | YES |
| Bearing episodes | 23 (16 FEMTO + 7 XJTU-SY) | RESULTS.md: 23 (16+7) | YES |
| Contrastive episodes | 18 | RESULTS.md: 18 | YES |

### Other numbers

| Claim | Paper | Source | Match? |
|-------|-------|--------|--------|
| Spearman correlation JEPA | 0.144 | RESULTS.md: 0.144 | YES |
| Contrastive Spearman | 0.591 | RESULTS.md: 0.591 | Not in paper text but consistent |
| Positive-pair similarity | 0.89 | RESULTS.md: 0.89 | YES |
| Negative-pair similarity | 0.47 | RESULTS.md: 0.47 | YES |
| Best JEPA val loss | 0.0166 at epoch 2 | RESULTS.md: 0.0166 at epoch 2 | YES |
| CV of lifetimes | 0.635 | RESULTS.md: 0.635 | YES |

### Content flagged as planned but not yet done

The paper contains `\plannedc{...}` blocks (rendered in blue in draft mode) that describe work NOT YET COMPLETED:
1. Abstract: "We further extend the framework to multivariate sensor inputs..."
2. Contributions item 4: "Extensions to multivariate inputs..."
3. Conclusion: "We extend the framework to multivariate sensor inputs..."

These describe C-MAPSS validation, physics-aware spatiotemporal masking, and synthetic grey swan augmentation that have NOT been done. These MUST be either completed or removed before submission. A reviewer seeing these claims without supporting experiments would reject the paper.

---

## Summary of Critical Issues

### Must Fix Before Submission

1. **ZERO figures** -- Add at minimum: architecture diagram (create new), main results bar chart (`v8_rul_comparison.pdf`), encoder analysis (`v8_encoder_analysis.pdf`), and cross-dataset results (`v8_cross_dataset.pdf`).
2. **14 bib entries with `author={various}`** -- Every one needs real author names looked up and filled in.
3. **`\plannedc{}` claims about unfinished work** -- Remove all three instances (abstract, contributions, conclusion) or complete the work. Cannot claim C-MAPSS validation without doing it.
4. **21.4% vs 20.7% inconsistency** -- The paper says 21.4% (from rounded RMSE); the ground truth JSON says 20.7% (from exact RMSE). Pick one approach and be consistent throughout.

### Should Fix

5. **6 bib entries with structural issues** (wrong entry type, placeholder DOI, year mismatch).
6. **75.5% may be off by 0.1-0.2pp** depending on exact baseline -- verify.
7. **15.8% is actually 15.6%** from the rounded values in the table.
8. **Missing important citations**: MAE (He et al. 2022), VICReg/collapse prevention literature, DINO lineage.
9. **`wang2018xjtusybearing`** has year mismatch (key says 2018, entry says 2020).

### Nice to Have

10. Additional figures: grey swan conceptual diagram, raw vibration waveform examples.
11. Latent trajectory figure (`v8_latent_trajectories.pdf`) adds qualitative evidence.
12. Pretrain history figure (`v8_pretrain_history.pdf`) supports the instability claim.

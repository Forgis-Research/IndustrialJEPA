---
name: Bearing RUL SOTA Literature Review (April 2026)
description: Best published results on FEMTO/XJTU-SY RUL, standard evaluation protocol, self-supervised RUL approaches, PHM 2012 metric, single-window formulation gap
type: project
---

Focused review conducted April 2026 for positioning IndustrialJEPA against SOTA.

## Standard Evaluation Protocol (Critical to Understand)

### How Published Papers Define the RUL Regression Target

The dominant convention (used by ~90% of papers):

  RUL_normalized(t) = (T_total - t) / T_total

This yields a value in [0, 1] where 1.0 = brand new, 0.0 = failure. Some papers use a
piecewise-linear label that caps at 1.0 for the "healthy phase" before a detected
degradation onset point, then decays linearly to 0.

### What Is Actually Predicted

Almost all papers do **point-to-point (P2P) regression on the full trajectory**: given
every window in a run-to-failure trajectory, predict the normalized RUL at each timestep.
This is the standard benchmark task — the model sees sequential windows and outputs a
prediction for each one.

A minority of papers do **sequence-to-point (S2P)**: given a subsequence of windows,
predict the RUL at the last timestep.

**Neither paradigm is what IndustrialJEPA's task formulation describes.** The project
task is: given ONE short window (0.1s-1.3s), without any temporal context or ordering
information, predict RUL as a percentage. This is a truly different setup.

### End-of-Life (EOL) Threshold (FEMTO/PRONOSTIA)

The PRONOSTIA rig stops each test when vibration amplitude exceeds 20 g. This is the
EOL marker. RUL is measured backward from this point.

### Dataset Splits (FEMTO)

Three operating conditions (sub-datasets). Sub-datasets 1 and 2: 2 training runs + 5 test
runs each. Sub-dataset 3: 1 test run. Standard practice uses sub-datasets 1 and 2 for
comparison; sub-dataset 3 is rarely included due to small size.

### Input Window

The standard window size in the `rul-datasets` library is 2560 samples. At 25.6 kHz
this is exactly 0.1 seconds. Papers vary widely — many use extracted statistical features
(RMS, kurtosis, etc.) over a 10-second recording interval rather than raw samples.

---

## FEMTO/PRONOSTIA SOTA Results (Normalized RUL in [0,1])

Metric is normalized RMSE (nRMSE) and MAE, where values are on the 0-1 scale.
Lower is better.

| Method | Year | RMSE (FEMTO) | MAE (FEMTO) | Notes |
|---|---|---|---|---|
| CNN-GRU-MHA (transfer learning) | 2024 | **0.0443** | -- | Applied Sci 14/19/9039; transfer learning, small sample |
| Bi-LSTM-Transformer + EMD | 2025 | 0.0563 | 0.0469 | Applied Sci 15/17/9529; XJTU primary, FEMTO also tested |
| TCN-Transformer parallel | 2025 | ~0.12 (inferred) | -- | Sensors 25/11/3571 PMC12158285; FEMTO+XJTU; 14.6% RMSE improvement over TCN-SA |
| MDSCT (Multi-scale Separable Conv+Transformer) | 2024 | 0.124 | 0.100 | PMC11481647; FEMTO+XJTU |
| HIWSAN (Health Indicator-Weighted Subdomain Align.) | 2025 | ~0.12 | ~0.10 | Sensors 25/15/4536; cross-condition transfer |
| CVT (baseline in MDSCT paper) | 2024 | 0.131 | 0.112 | |
| TCN-SA (baseline in MDSCT paper) | 2024 | 0.148 | 0.114 | |
| DCNN (baseline) | 2024 | 0.175 | 0.145 | On XJTU-SY |
| TcLstmNet-CBAM | 2025 | ~3.1 absolute min | -- | Scientific Reports; PHM2012+XJTU-SY; different scale |

The CNN-GRU-MHA result of 0.0443 appears to be among the best reported on FEMTO as of
early 2025. The MDSCT paper (Heliyon/PMC 2024) provides the cleanest apples-to-apples
comparison table.

Note: results are NOT fully comparable across papers due to:
- Different subsets of FEMTO used (some use all 3 conditions, some just 1-2)
- Different degradation onset detection methods (affects how many windows are labeled)
- Different normalization conventions

## XJTU-SY SOTA Results

| Method | Year | RMSE (XJTU-SY) | MAE (XJTU-SY) | Notes |
|---|---|---|---|---|
| Bi-LSTM-Transformer + EMD | 2025 | **0.0563** | 0.0469 | Applied Sci 15/17/9529 |
| CNN-GRU-MHA (transfer) | 2024 | 0.0693 | -- | Applied Sci 14/19/9039 |
| TCN-Transformer parallel | 2025 | -- | -- | Sensors 25/11/3571; XJTU also tested |
| MDSCT | 2024 | 0.160 | 0.134 | PMC11481647 |
| SAGCN-SA | 2024 | 0.170 | -- | Best in that paper |
| TCN-SA (baseline) | 2024 | 0.194 | 0.158 | |
| CVAER (envelope spectrum + VAE) | 2024 | 28.47 min | -- | Absolute time, not normalized |

The XJTU-SY dataset is less standardized than FEMTO — papers sometimes report per-bearing
results, sometimes averages over condition groups.

---

## The PHM 2012 Challenge Scoring Function

The original 2012 challenge used an **asymmetric score function** based on percent error,
not RMSE. The formula (from Nectoux et al. 2012 and widely cited):

  For each bearing i:
    Er_i = (RUL_predicted - RUL_true) / RUL_true * 100   (percent error)

    if Er_i <= 0 (early prediction):  score_i = -log(1 - Er_i/100)^C1   [penalizes less]
    if Er_i > 0  (late prediction):   score_i = log(1 + Er_i/100)^C2   [penalizes more]

  Final score = (1/N) * sum(score_i)   where N = 11 test bearings

The exact penalty coefficients C1, C2 are in the original PDF (not fully accessible via
web). Late predictions (predicting longer life than actual) are penalized more heavily,
reflecting the safety-critical nature of maintenance.

The competition winner achieved a total score of ~0.28.

Crucially: the PHM 2012 metric is NOT normalized RMSE — it is a percent-error-based
score function applied to a **single final RUL estimate** per bearing (made at a specific
fraction of the bearing's life). This is conceptually different from the whole-trajectory
RMSE that most recent papers report.

Post-2015 literature has largely abandoned the PHM score in favor of normalized
RMSE/MAE on the full trajectory, which is more interpretable and allows finer comparison.

---

## Self-Supervised / Representation Learning Approaches to RUL

### Existing Work

**DCSSL (2026)** — "A novel dual-dimensional contrastive self-supervised learning-based
framework for rolling bearing RUL prediction." Scientific Reports 2026.
Stage 1: Random cropping + timestamp masking to build positive pairs. Temporal-level
and instance-level contrastive loss. Stage 2: Fine-tune RUL regression head.
Dataset: FEMTO/PRONOSTIA. Claims to outperform supervised SOTA.
Relevance: Closest existing work to IndustrialJEPA's SSL + RUL framing.

**HNCPM (Hard Negative Contrastive Prediction Model, ~2023)** — Contrastive learning
with hard negative samples (selected by cosine similarity), GRU regression module, decoder
for reconstruction. Encoder learns representations, GRU predicts RUL.

**Contrastive SSL for incipient fault detection (Reliability Eng. 2022)** — Pretrains
encoder with contrastive objective on healthy vs degraded windows, then extracts health
indicator. Not direct RUL regression.

**Semi-supervised transfer learning (MDPI Sensors 2025)** — Anti-self-healing health
indicator + semi-supervised transfer. Some self-supervised elements but primarily
supervised RUL.

**RULSurv (2024)** — Survival analysis (CoxPH, RSF) on XJTU-SY with censoring-aware
training. Not self-supervised but novel framing with probabilistic output.
XJTU-SY C1 MAE: 12.6 minutes (Random Survival Forests). Different metric from nRMSE.

**CVAER (2024)** — Convolutional Variational Autoencoder for Regression. Uses averaged
envelope spectra (AES) as input. Trains one VAE + regression head per bearing lifecycle.
XJTU-SY best RMSE: 28.47 minutes. This is the closest thing to single-spectrum prediction:
each timestamp's AES is fed independently to predict RUL. But they still train on the
full trajectory (supervised on time-ordered windows).

### What Does NOT Exist

- JEPA + RUL: no paper exists (confirmed gap from earlier search)
- MAE pretraining specifically for bearing RUL regression (pretraining on healthy vibration,
  then fine-tuning on limited run-to-failure labels): not published
- JEPA pretrained on large multi-source vibration corpus, then fine-tuned for RUL: not published
- Any paper that explicitly treats RUL as a **position-agnostic single-window problem**

---

## The Single-Window RUL Problem: Does It Exist in Literature?

### The Direct Answer: No, Not as a Formal Formulation

No published paper defines the task as: "Given one short vibration window, with no
knowledge of where in the bearing's lifecycle this window comes from, predict RUL as a
percentage."

All existing methods require either:
1. **Sequential context**: A series of windows in temporal order, so the model can see
   the degradation trend
2. **Online position information**: The model knows which timestep in the lifecycle the
   current window corresponds to
3. **Degradation onset detection first**: The model first detects when degradation begins
   (a separate module), then predicts RUL from that onset point

### The Closest Existing Work

**CVAER (Convolutional VAER, 2024)**: Makes a RUL prediction from each individual envelope
spectrum. The model doesn't explicitly see history — it predicts from a single snapshot.
BUT the training procedure is supervised on full trajectories (the model learns the mapping
from spectrum shape to position in lifecycle), and the spectrum is time-averaged over 10s
recordings (not a raw 0.1s vibration burst). Published in PMC (Sensors/MDPI family), so
not a top-venue paper.

**DMW-Trans (IEEE 2023)**: Deep Multiscale Window-based Transformer that explicitly
argues single-window approaches fail because they miss cross-temporal information. It
proposes multiscale feature maps as a solution. This paper directly acknowledges the
single-window problem but treats it as a weakness to overcome, not a feature to exploit.

### Why the Single-Window Formulation Is Novel

The conventional wisdom (stated explicitly in DMW-Trans 2023 and the survey literature)
is that single-window RUL estimation is fundamentally limited — you need temporal context.
Our hypothesis: a sufficiently powerful pretrained representation (JEPA embeddings) can
encode enough physical information about degradation state from a single window to regress
RUL without temporal context. If true, this is both novel and practically significant:
it enables RUL estimation from a single accelerometer capture, with no historical data
and no assumption about where in the lifecycle the window sits.

---

## Cross-Dataset Transfer Results (FEMTO <-> XJTU-SY)

This is rare but does exist. Key papers:

**ERCDAN (Enhanced Residual Conv Domain Adaptation Network, Reliability Eng. & System Safety 2024)**
DOI: 10.1016/j.ress.2024.110516 (ScienceDirect). Uses CBAM + residual conv + MK-MMD domain adaptation.
Validates on PHM2012, XJTU-SY, and EBFL datasets in cross-machine transfer scenarios.
Does not report normalized RUL RMSE — uses a different scale.
Architecture: CNN feature extractor + CBAM + MK-MMD + RUL regressor.

**Cross-condition/cross-platform adversarial domain adaptation (Scientific Reports 2021, still widely cited)**
PMC8766616. Uses 12 transfer tasks between FEMTO conditions and XJTU-SY conditions.
First paper to explicitly frame cross-platform (FEMTO->XJTU) as a formal benchmark.

**CNN-Bi-LSTM Domain Adaptation (Sensors 2024)**
MDPI 1424-8220/24/21/6906. PHM2012 as source, tests cross-condition transfer within FEMTO.
Does not test cross-dataset (FEMTO->XJTU) directly.

**HIWSAN (Health Indicator-Weighted Subdomain Alignment Network, Sensors 2025)**
MDPI 1424-8220/25/15/4536. Cross-condition transfer, combined FEMTO+XJTU evaluation.
Average MAE 0.0989, average RMSE 0.1189 across both datasets (not separate per-dataset numbers).

Key insight: Cross-dataset transfer (train on FEMTO, test on XJTU-SY or vice versa) is an active but
small sub-field. No paper does it as a single-window JEPA transfer problem. Most use domain
adaptation (DANN, MMD), not representation pretraining. This is a gap IndustrialJEPA can fill.

## Top Papers to Compare Against

Priority papers to include as baselines in any IndustrialJEPA RUL paper:

### Tier 1: Must-Compare (Standard Baselines)

1. **MDSCT (Multi-scale Deep Separable Convolution Transformer, Heliyon 2024)**
   PMC11481647. Provides clean comparison table on both FEMTO and XJTU-SY.
   FEMTO nRMSE: 0.124. XJTU-SY nRMSE: 0.160.
   Why: Most recent clean benchmark with multiple baselines.

2. **CNN-GRU-MHA with Transfer Learning (Applied Sciences 2024)**
   MDPI 2076-3417/14/19/9039. nRMSE 0.0443 on FEMTO (best reported).
   Why: Current best result on FEMTO (if reproducible).
   STATUS: REPLICATED. Our implementation achieves 0.0416 avg RMSE (-6.1% vs paper).
   Protocol: 5 seeds, 11 unique FEMTO transfers, random 50/50 split (NOT chronological).

3. **DCSSL (Dual-Dimensional Contrastive SSL, Scientific Reports 2026)**
   Nature/s41598-026-38417-7. Self-supervised + RUL on FEMTO.
   Why: The only SSL-based RUL paper on FEMTO — direct methodological competitor.

### Tier 2: Important Context

4. **CVAER (Envelope Spectra + Probabilistic VAE, PMC 2024)**
   PMC11597903. XJTU-SY, 28.47 min RMSE.
   Why: Closest thing to per-spectrum (single-snapshot) prediction; establishes
   feasibility of the single-window formulation even if not framed that way.

5. **Bi-LSTM-Transformer + EMD (Applied Sciences 2025)**
   MDPI 2076-3417/15/17/9529. RMSE 0.0563 on XJTU-SY, validated on FEMTO.
   Why: Strong 2025 supervised baseline for fair comparison.

### Tier 3: Cite for Context

6. **Nectoux et al. 2012** — PRONOSTIA platform description, PHM challenge setup.
   Cite as: "Nectoux, P. et al. PRONOSTIA: An experimental platform for bearings
   accelerated degradation tests. IEEE PHM 2012 Challenge."
   HAL: hal-00719503.

7. **RULSurv (arXiv 2405.01614, 2024)** — Survival analysis framing, censoring-aware.
   Methodologically distinct; useful to cite as alternative probabilistic approach.

8. **OpenMAE (ACM IMWUT 2025)** — MAE pretraining on vibration signals.
   Closest foundation-model competitor on the pretraining side (not RUL specifically).

---

## Key Positioning Takeaways for IndustrialJEPA

### What We Would Be the First to Do

1. Apply JEPA pretraining to bearing RUL prediction (no prior work)
2. Define and solve the single-window position-agnostic RUL task (no prior formulation)
3. Show that JEPA embeddings carry sufficient degradation state information
   for single-window RUL regression

### How to Frame the Contribution

Existing narrative in the field: "Single-window RUL estimation is a known weakness —
you need temporal context to predict RUL."

Our counter-claim: "A JEPA representation learned from large multi-source vibration data
encodes physical degradation state in a single window's embedding. With no temporal
context, our model achieves competitive RUL prediction — suggesting the signal is rich
enough; the failure mode was lack of a strong enough representation."

### Target Metrics to Beat

- FEMTO nRMSE: beat 0.0443 (CNN-GRU-MHA) with no temporal context. If we get <0.10
  without temporal context vs their 0.0443 with full trajectory, the story is still strong.
- XJTU-SY nRMSE: beat 0.170 (SAGCN-SA) without temporal context.
- A meaningful result: showing that our single-window approach gets within 2x of
  trajectory-based SOTA would be a strong result justifying the new problem formulation.

**Why:** IndustrialJEPA must be positioned against these numbers so reviewers see
exactly where we stand relative to methods that use orders of magnitude more temporal
context.
**How to apply:** When writing the experiment section, explicitly state: "We compare
against Tier 1 baselines, noting that all baselines use full degradation trajectory
context while our method uses only a single window."

---

## Probabilistic RUL Methods (Added April 2026)

### Method Comparison for Uncertainty Quantification

Four main paradigms for producing a distribution rather than a point estimate:

**1. Heteroscedastic Neural Network (HNN)**
Output two scalars: predicted mean mu and log-variance log(sigma^2). Loss = NLL under
Gaussian. Cost: zero overhead vs standard regression — one forward pass, two output
neurons. Trains end-to-end. Can attach to any encoder+MLP head with a 2-line change.
Limitation: assumes Gaussian output; cannot model heavy-tailed or multimodal failure
distributions. Papers flag "pitfalls of heteroscedastic estimation" — overconfident when
data is non-Gaussian (arXiv 2203.09168). BUT: for RUL in [0,1] with near-Gaussian error
distribution, HNN is the lowest-overhead option.

**2. Monte Carlo Dropout (MCD)**
Keep dropout active at inference; run N forward passes (typically 30-100); treat the
sample mean as prediction, variance as uncertainty. Cost: N x inference cost. For LSTM
architectures, dropout already applied between LSTM layers, so zero architectural change
needed. Implementation: 3 lines of code. Quality: approximates a GP posterior. Weakness:
MCD uncertainty estimates are unreliable when dropout rate is too low or network is
over-parameterized (arXiv 2512.14851). Recent work (2025) shows improvements via stable
output layers. Practical guidance: use 50 forward passes, dropout rate 0.1-0.3.

**3. Deep Ensembles (DE)**
Train M independent models with different random seeds (typically M=5). Average predictions;
variance across predictions is uncertainty. Cost: M x training cost, M x inference cost.
Cannot run in parallel on single GPU. Quality: consistently BEST calibrated uncertainty
in benchmarks. Key finding from the UQ benchmark paper (Reliability Eng. & System Safety
2025, DOI: 10.1016/j.ress.2024.005854): DE > MCD > HNN for calibration on N-CMAPSS.
For deployment: DE is practical when M=5 and model is small (LSTM at 1M params is fine).

**4. Evidential Deep Learning (EDL)**
Output the parameters of a higher-order distribution (e.g., Normal-Inverse-Gamma) in one
forward pass. Decomposes uncertainty into aleatoric (data) + epistemic (model). Cost:
single forward pass, same as HNN. Quality: good in theory; in practice, EDL often
underestimates uncertainty and requires careful regularization. Paper: "Remaining Useful
Life Prediction with Uncertainty Quantification Using Evidential Deep Learning" (JAISCR
2025, on C-MAPSS). Best for: production systems where separate forward passes are
impossible.

**5. Quantile Regression (QR)**
Train a separate head for each quantile (e.g., q=0.1, 0.5, 0.9). Pinball loss per
quantile. Cost: K x output neurons but single forward pass. Gives prediction intervals
directly. Paper: "Quantile Regression Network for cross-domain bearing RUL prediction"
(ScienceDirect 2024); average interval coverage 91.25% on bearing datasets, interval
width 16.65%. State Space Model + Simultaneous Quantile Regression (SQR) paper (arXiv
2506.17018) shows SSM handles long sequences efficiently. Practical for IndustrialJEPA:
can replace single output neuron with 3-5 quantile outputs, minimal change.

**6. Gaussian Process (GP) for RUL**
Sparse GP with time-aware spatiotemporal kernel (SAGE Pub 2025). Full probabilistic
output. Cost: O(n^2) to O(nm^2) with sparse approximation. Does not compose easily with
deep encoder. Best use: as a final layer on top of frozen encoder embeddings (Deep Kernel
Learning). Paper: "Sparse Gaussian process regression with time-aware spatiotemporal
kernel for bearing RUL and uncertainty quantification" (SAGE 2025).

### Lowest Overhead Ranking for Encoder+LSTM Architecture

Ranked from cheapest to most expensive to add:
1. HNN — zero overhead, 2 output neurons, 1 loss function change
2. Quantile regression (3-5 quantiles) — zero overhead, K output neurons, K loss terms
3. EDL — zero overhead, 4 output neurons, specialized loss
4. MCD — N x inference, 3 lines of code, no training change
5. GP on frozen embeddings — O(nm^2), separate fitting step
6. Deep Ensembles — M x all costs

**Recommendation for IndustrialJEPA:** HNN as the primary approach (zero cost), with
MCD (N=50 passes) as a validation check that uncertainty estimates are consistent.

### Outputting P(failure | t) as a Curve Over Future Time

This is exactly what survival analysis produces. Two distinct paradigms:

**Survival Analysis (RULSurv framing)**
arXiv 2405.01614 (2024), code: github.com/thecml/rulsurv.
- Uses KL divergence of frequency spectrum as health indicator to detect fault onset
- Trains Cox Proportional Hazards, Weibull AFT, Random Survival Forest (RSF) on XJTU-SY
- Output: S(t) = P(T > t | x) — the probability the bearing survives past time t
- Equivalently: F(t) = 1 - S(t) = P(failure by time t | current sensor reading)
- RSF outperformed CoxPH and neural survival models; XJTU-SY C1 MAE: 12.6 minutes
- Key insight: survival function gives user a full curve; they pick their own risk
  threshold (e.g., replace when P(failure by end of shift) > 0.2)
- Limitation: requires knowing the reference time origin (when degradation started)

**Weibull Proportional Hazards Model (WPHM)**
Classical approach in PHM. KPCA-WPHM-SCNs (2024, TIMC journal): uses kernel PCA
health indicator + WPHM to estimate reliability function R(t). Then fits neural network
to the hazard rate curve. Produces both R(t) curve and point RUL estimate.

**Neural Network Survival Models**
DeepHit, DeepSurv: treat survival as classification over discrete time bins.
Output is a discrete probability distribution over failure times.
These can be bolted onto any encoder with a survival head.

**The Key Distinction for IndustrialJEPA**
If we output P(failure by time t) rather than RUL%, the end user can:
1. Set their own risk threshold (replace at P>0.3 vs P>0.1 depending on criticality)
2. Quantify the planning horizon (95% CI on time-to-failure)
3. Integrate with cost functions for optimal replacement scheduling

This framing is genuinely more useful than a single RUL% number and has not been
combined with JEPA pretraining. Novel contribution opportunity.

---

## Dataset Inventory: Run-to-Failure Episodes (Added April 2026)

### How Many Episodes Do Published Papers Use?

| Paper | Dataset | # Training Episodes | # Test Episodes |
|---|---|---|---|
| CNN-GRU-MHA (2024) | FEMTO | 4 (conditions 1+2) | 10 |
| MDSCT (2024) | FEMTO + XJTU-SY | 4 + 3 = 7 | 10 + 12 |
| DCSSL (2026) | FEMTO | 4 | 10 |
| RULSurv (2024) | XJTU-SY | cross-val, 15 total | cross-val |
| Typical bearing RUL paper | FEMTO or XJTU | 4-7 | 5-15 |

**The field routinely publishes with 4-7 training episodes.** 23 episodes is NOT small
by this community's standards — it is above average. However, FEMTO has only 4 training
episodes in the standard split. Papers regularly publish with those 4 episodes.

### Dataset Episode Counts (Corrected and Complete)

**FEMTO/PRONOSTIA (2012)**
- Total: 17 bearings (3 operating conditions)
  - Condition 1 (1800 rpm, 4 kN): 2 train + 5 test = 7 bearings
  - Condition 2 (1650 rpm, 4.2 kN): 2 train + 5 test = 7 bearings
  - Condition 3 (1500 rpm, 5 kN): 0 train + 3 test = 3 bearings
- Standard benchmark uses conditions 1+2 = 4 train, 10 test (14 bearings total)
- All 17 can be used if you include condition 3 and use cross-validation

**XJTU-SY (2019)**
- Total: 15 bearings (3 operating conditions, 5 bearings each)
  - Condition 1 (2100 rpm, 12 kN): 5 bearings
  - Condition 2 (2250 rpm, 11 kN): 5 bearings
  - Condition 3 (2400 rpm, 10 kN): 5 bearings
- All 15 run to failure. Standard split varies by paper (some use all 15 with LOO or
  cross-val; some use 3 conditions as train/val/test)
- Our V8 uses only 7 — this is leaving 8 episodes on the table. Can add all 15.

**IMS/NASA (2004)**
- 3 test runs (datasets 1, 2, 3), each with 4 bearings on a shaft = 12 bearing signals
- BUT only Bearing 3 in test 1, Bearing 3 in test 2, Bearing 1 in test 3, Bearing 3 in
  test 3 reached failure (4 actual run-to-failure trajectories in the strict sense)
- Many papers treat all 4 bearings per run as independent RUL episodes despite only
  some failing (the others are right-censored)
- Commonly used but benchmark results are messier than FEMTO/XJTU-SY
- Sampling: 1 second vibration snapshot every 10 minutes

**Additional Datasets (Beyond the Big Three)**

- **University of Ferrara (2024)**: 6 accelerated run-to-failure tests on self-aligning
  double-row ball bearings. Data descriptor: ScienceDirect DOI: 10.1016/j.dib.2024...
  Published 2024; not yet widely benchmarked.

- **Zenodo Run-to-Failure (Aimiyekagbon 2024)**: Ball bearings under time-varying load
  and speed conditions. DOI: 10.5281/zenodo.10805043. Novel: time-varying conditions.
  Size: not specified in search results.

- **Mendeley Run-to-Failure (2024)**: Single bearing, failed at 128 hours.
  DOI: 10.17632/5hcdd3tdvb.6. Very small — 1 episode.

- **Paderborn Zwoelf (2024 Zenodo)**: New Paderborn dataset with time-varying load/speed.
  Not the KA/KI fault classification dataset — a separate RUL-focused release.

- **PHM Society datasets**: Various challenge datasets. PHM 2012 = FEMTO. PHM 2023
  competition had a gearbox dataset, not bearing RUL specifically.

- **DIRG (Politecnico di Torino)**: Internal dataset occasionally published. Not
  confirmed as a standardized public benchmark.

- **MFPT (2013)**: Machinery Failure Prevention Technology. Has run-to-failure data but
  rarely used for RUL benchmarks. More common for fault classification.

**Total publicly available run-to-failure episodes (conservative count):**
FEMTO (17) + XJTU-SY (15) + IMS (4 strict / 12 liberal) + Ferrara (6) + others (~5-10)
= roughly 45-55 clean run-to-failure episodes exist in the literature.

### Is 23 Episodes Too Few?

No. Context:
- Standard FEMTO benchmark: 4 training episodes
- Standard XJTU-SY benchmark: 3-5 training episodes per condition
- Most top-venue RUL papers: 4-7 training episodes
- 23 episodes is more than any single-dataset paper uses for training

The real concern is **diversity of degradation trajectories**, not count:
- FEMTO 4 train episodes are all at 2 operating conditions (deterministic setup)
- Our 23 come from 2 datasets, 5 operating conditions — better coverage
- But all are inner/outer race failures; ball failures are rare in public data

### Data Augmentation for Run-to-Failure Episodes

Three approaches from the literature:

**1. Dynamic Time Warping (DTW) augmentation**
Stretch or compress a known trajectory to simulate faster/slower degradation.
Paper: "RUL Prediction by data augmentation technique based on DTW" (ScienceDirect 2020).
Produces virtual run-to-failure episodes with different lifespans. Widely cited.
Risk: DTW-stretched trajectories may not honor physical constraints.

**2. VAE-GAN synthesis**
Train a Variational Autoencoder + GAN on degradation feature sequences.
Paper: "Remaining Useful Life Prediction Method Enhanced by Data Augmentation and
Similarity Fusion" (MDPI 2024). Generates synthetic degradation time series.
Can increase episode count 3-5x. Most complex to implement.

**3. Degradation model + Sobol sampling**
Parametric degradation model (Paris law, Wiener process), sample parameters from
Sobol sequence. Each sample = one synthetic episode. Physics-grounded.
Produces unlimited synthetic episodes, but requires knowing the physics model.

**4. Window-level augmentation (within episodes)**
Jitter, time shift, magnitude warping within each window. Does NOT increase episode count
but increases per-episode sample diversity. Already standard practice.

**5. Autoregressive model augmentation**
"Data augmentation of multivariate sensor time series using autoregressive models"
(arXiv 2410.16419, Oct 2024). Fits AR model to healthy segment, extrapolates degradation.

Practical recommendation: DTW augmentation is lowest effort, well-cited, and produces
interpretable results. Can easily 3x episode count without changing architecture.
VAE-GAN is higher effort with marginal benefit given we have 23 real episodes.

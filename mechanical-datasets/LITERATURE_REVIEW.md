# Literature Review: State-of-the-Art in Mechanical Fault Diagnosis
## Is Mechanical-JEPA Justified?

**Date:** March 2026
**Purpose:** Honest assessment of what existing methods achieve and where the real gaps are

---

## Table of Contents
1. [Bearing Fault Classification (The "Easy" Benchmark)](#1-bearing-fault-classification)
2. [Cross-Domain / Transfer Learning](#2-cross-domain-transfer-learning)
3. [Remaining Useful Life (RUL) Prediction](#3-remaining-useful-life-prediction)
4. [Self-Supervised / Foundation Models for Vibration](#4-self-supervised-foundation-models)
5. [Action-Conditioned / Varying-Condition Models](#5-action-conditioned-models)
6. [The "Is JEPA Even Needed?" Question](#6-is-jepa-even-needed)
7. [Where the Real Gaps Are](#7-where-the-real-gaps-are)
8. [Verdict: Is Mechanical-JEPA Justified?](#8-verdict)

---

## 1. Bearing Fault Classification

### 1.1 CWRU Dataset: Effectively "Solved" (With Caveats)

The CWRU bearing dataset is the ImageNet of mechanical fault diagnosis. Nearly every paper reports results on it. The consensus: **it is saturated.**

| Method | Accuracy on CWRU | Year | Reference |
|--------|-----------------|------|-----------|
| 1D CNN (2-3 layers) | 99%+ | 2020 | Smith & Randall, Mech. Syst. Signal Process. |
| WDCNN (Zhang et al.) | 99.6% | 2017 | Zhang et al., Sensors |
| 1D Deep CNN (unbalanced classes) | **100%** (10 classes) | 2025 | Feisa, Structural Control and Health Monitoring |
| CNN-BiLSTM + Grey Wolf Opt. | **100%** | 2025 | Nature Scientific Reports |
| Hybrid TCN-Transformer | 99.1%+ | 2025 | J. Mechanical Science and Technology |
| ResNet-50 + SVM | 95.5% | 2024 | Nature Scientific Reports |
| Envelope Analysis + ML | ~95-97% | 2023 | IEEE Access (Khanam et al.) |
| Masked SSL + Swin Transformer | **100%** | 2025 | MDPI Machines |
| Random Forest + hand-crafted features | 92-96% | Various | Multiple studies |

**Key finding:** Simple 1D CNNs have achieved 99%+ on CWRU since ~2019. Multiple methods now report 100%. The dataset is solved for standard fault classification.

**Critical caveat (Smith & Randall, 2015; Braga et al., 2022):** Most CWRU results suffer from **data leakage**. The standard practice splits segments from the same continuous recording into train/test sets, meaning the network learns recording-specific artifacts rather than fault signatures. Additionally, the same physical bearing is re-measured at different speeds, allowing the model to learn rig-specific features. Braga et al. (Towards Better Benchmarking Using the CWRU Bearing Fault Dataset, Mech. Syst. Signal Process., 2022) showed that under controlled domain shift, **accuracy drops from 99% to 36-45%** with variance of 15-20%.

### 1.2 Paderborn University Dataset

The Paderborn dataset is harder: real damages (not just EDM-seeded), varying operating conditions, 32 damage states.

| Method | Accuracy on Paderborn | Year | Reference |
|--------|----------------------|------|-----------|
| Masked SSL + Swin Transformer | 99.53% | 2025 | MDPI Machines |
| Domain Adaptation Networks | 85-95% | 2024 | Various |
| Standard CNN | 90-95% | Various | Various |
| Traditional ML | 80-90% | Various | Various |

Paderborn is more realistic but still a controlled lab dataset. The real challenge begins with cross-domain transfer.

### 1.3 Simple/Traditional Methods

Envelope analysis + SVM on CWRU achieves ~95-97% (Khanam et al., IEEE Access, 2023). Hand-crafted statistical features (RMS, kurtosis, crest factor, spectral kurtosis) + Random Forest typically achieves 92-96% on CWRU. These are far from trivial -- for the standard (leaky) CWRU benchmark, even simple methods work well because the task is easy.

**Bottom line on classification:** Single-dataset bearing fault classification is essentially solved. The gap between a 2-layer CNN and a Transformer on CWRU is <2%. Any new method reporting only CWRU accuracy is not contributing meaningfully. The real test is generalization.

---

## 2. Cross-Domain / Transfer Learning

This is where things get interesting -- and where most methods break down.

### 2.1 Cross-Dataset Transfer (Same Component Type)

| Transfer Task | Method | Accuracy | Year | Reference |
|--------------|--------|----------|------|-----------|
| CWRU -> Paderborn | 1D-LDSAN (deep subdomain adapt.) | 82-90% | 2022 | PMC |
| CWRU -> Paderborn | Multiscale CNN + Adversarial Subdomain | ~85-92% | 2025 | Nondestructive Testing and Eval. |
| CWRU -> Paderborn | Transformer Transfer Learning Network | 80-90% | 2025 | Nature Scientific Reports |
| CWRU -> Paderborn (no adaptation) | Direct CNN transfer | **36-45%** | 2022 | Braga et al., MSSP |
| Various bearing datasets | Domain Adversarial + Structural Adj. | avg F1 ~96.6% | 2025 | PMC |
| Cross-condition (same dataset) | Spatiotemporal Feature Fusion DA | 90-95% | 2025 | MDPI Sensors |

**Key findings:**
- **Without domain adaptation**, cross-dataset accuracy drops catastrophically (from 99% to 36-45%)
- **With domain adaptation**, accuracies of 85-95% are achievable but require labeled or unlabeled target domain data
- **No method reliably achieves >95% cross-domain accuracy** without some access to target data
- Most reported "cross-domain" results are actually cross-condition (different speed/load on the same rig), which is much easier

### 2.2 Cross-Component Transfer (Bearing -> Gearbox)

This area is **severely under-researched**. Almost no papers attempt true cross-component transfer.

| Transfer Task | Method | Accuracy | Year | Reference |
|--------------|--------|----------|------|-----------|
| Bearing -> Gearbox | 1D Large-Conv DenseNet | ~96.45% bearing, 98.92% gearbox (separate) | 2025 | Nonlinear Dynamics |
| Cross-domain gearbox | Deep conv. transfer learning | Variable | 2020 | ResearchGate |

Most "cross-component" work trains and tests on the same component type but different machines. True bearing-to-gearbox transfer (learning fault physics from one and applying to another) is essentially unexplored. This is a genuine gap.

### 2.3 Foundation Models vs. Training from Scratch

There is **no established vibration foundation model** comparable to ImageNet-pretrained CNNs in vision. The closest approaches:
- Using CWT/STFT images with ImageNet-pretrained ResNets as feature extractors (achieves ~90-95% on CWRU, ~80-90% cross-domain)
- Few-shot transfer with Siamese networks pretrained on CWRU (Siamese-WDCNN, 2025)
- Self-supervised pretraining then fine-tuning (see Section 4)

**Bottom line on transfer:** Cross-domain generalization is the real unsolved problem. Current methods require target domain data for adaptation. A model that learns transferable fault physics without target data would be genuinely novel.

---

## 3. Remaining Useful Life (RUL) Prediction

### 3.1 Metrics

RUL prediction uses different metrics than classification:
- **RMSE** (Root Mean Squared Error) -- most common, but scale-dependent
- **MAE** (Mean Absolute Error)
- **PHM Score Function** -- asymmetric, penalizes late predictions more than early ones (safety-critical)
- **Percentage Error** -- normalized RMSE
- **Alpha-Lambda Accuracy** -- checks if predictions fall within alpha% of true RUL at lambda% of life

### 3.2 FEMTO / PHM 2012 Challenge

The PHM 2012 challenge dataset (PRONOSTIA/FEMTO) has 17 run-to-failure experiments across 3 operating conditions.

| Method | Avg RMSE | Year | Reference |
|--------|----------|------|-----------|
| CNN-GRU-MHA (transfer learning) | 0.0443 | 2024 | Applied Sciences |
| Transformer-Bi-LSTM | 0.0563 | 2025 | MDPI Applied Sciences |
| TcLstmNet-CBAM | ~0.05 | 2025 | Nature Scientific Reports |
| VMD + BiLSTM-CBAM | ~0.04-0.06 | 2025 | PubMed |
| Simple degradation indicator + regression | ~0.08-0.15 | Various | Various |
| PHM 2012 challenge winner (Sutrisno et al.) | Score: 0.31 | 2012 | PHM Society |

**Note:** RMSE values are often normalized differently across papers, making direct comparison difficult. The PHM Score is the official metric but many papers only report RMSE/MAE.

### 3.3 XJTU-SY Dataset

15 bearings across 3 operating conditions, complete run-to-failure data.

| Method | Avg RMSE | Year | Reference |
|--------|----------|------|-----------|
| Dynamic Temporal Attention + CT-MLP | Best in class | 2024 | Springer AIS |
| SAGCN-SA (graph + self-attention) | 0.170 | 2024 | Various |
| Transformer-Bi-LSTM | 0.0442 | 2025 | MDPI |
| CNN-GRU-MHA | 0.0693 | 2024 | Applied Sciences |
| CNN-based Health Indicator | 0.05-0.10 | Various | Various |

### 3.4 IMS Dataset

3 test runs, 4 bearings each, run-to-failure under constant 2000 RPM / 6000 lbs.

| Method | RMSE | Year | Reference |
|--------|------|------|-----------|
| Deep BiLSTM | 0.0281 | 2024 | AIMS ERA |
| SAE-LSTM | 0.0326 | 2022 | ScienceDirect |
| PSR-former | 1.031 (unnormalized) | 2022 | Nature Sci. Reports |
| Fractal Dimension + 1D-CNN | 0.0691 | 2022 | SAGE Journals |
| Deep Reinforcement Learning | Competitive | 2024 | AIP Publishing |

### 3.5 Can Simple Methods Compete?

**Yes, to a surprising degree.** Simple health indicator approaches (trending RMS, kurtosis, spectral features) combined with exponential/polynomial regression can achieve RMSE in the 0.08-0.15 range on FEMTO/XJTU-SY. The deep learning methods reduce this to 0.04-0.07 -- a meaningful but not dramatic improvement.

The real advantage of deep learning for RUL is:
1. **Automatic feature extraction** (no manual HI construction)
2. **Better handling of multiple operating conditions**
3. **Uncertainty quantification** (some methods)

But the gap between a well-tuned HI + regression baseline and SOTA deep learning is often only 30-50% relative RMSE reduction, not orders of magnitude.

**Bottom line on RUL:** Deep learning helps but does not revolutionize RUL prediction. The bigger challenge is **early detection** (predicting failure before obvious degradation indicators appear) and **cross-condition generalization** (training on one operating condition, predicting on another).

---

## 4. Self-Supervised / Foundation Models for Vibration

### 4.1 Vibration Foundation Models: They Basically Don't Exist Yet

As of early 2026, there is **no widely adopted vibration foundation model** analogous to GPT for text or DINO/MAE for vision. This is notable and represents a genuine gap.

**Why not?**
- Vibration data is highly domain-specific (bearing vs. gearbox vs. motor)
- Sampling rates and sensor configurations vary enormously (1 kHz to 100 kHz)
- No single large-scale vibration dataset exists (unlike ImageNet, LibriSpeech, etc.)
- The community is fragmented across mechanical engineering, signal processing, and ML

### 4.2 Self-Supervised Pretraining Approaches

| Method | Approach | Dataset | Result | Year | Reference |
|--------|----------|---------|--------|------|-----------|
| Masked SSL + Swin Transformer | Masked autoencoder on vibration spectrograms | CWRU, Paderborn | 100% (CWRU), 99.53% (Paderborn) | 2025 | MDPI Machines |
| Self-Supervised Progressive Learning | Multi-task SSL with progressive fine-tuning | Multiple | Superior to supervised baselines under limited labels | 2025 | Neurocomputing (ScienceDirect) |
| Contrastive + Attention (BYOL-style) | Temporal contrastive with attention | Various bearings | Improved domain adaptability | 2024-25 | ScienceDirect |
| Physics-Informed SSL | Domain generalization with physics constraints | Bearing datasets | Improved generalization | 2024 | Various |

### 4.3 Masked Autoencoders for Vibration

The masked autoencoder paradigm (from He et al., MAE, 2022 in vision) has been adapted to vibration:
- Pham et al. (2025): Masked SSL + Swin Transformer achieves 99.53% on Paderborn unsupervised, 100% on CWRU
- Several papers apply CWT to convert 1D vibration to 2D images, then use standard vision MAEs

**Limitation:** Most of these still evaluate on single-dataset classification (the easy task). Few evaluate on cross-domain transfer or RUL.

### 4.4 Contrastive Learning Approaches

- BYOL-style frameworks adapted for vibration temporal structure
- Multi-target contrastive pretraining with attention mechanisms
- Semi-supervised contrastive learning with limited labels

Results generally show **5-15% improvement over supervised baselines when labels are scarce** (<10% labeled data). When abundant labels are available, the gap narrows to 1-3%.

### 4.5 How Do They Compare to Supervised Baselines?

**Honest assessment:**
- On standard benchmarks (CWRU, Paderborn): SSL methods **match** supervised methods, sometimes marginally better
- Under **limited labels** (1-10% labeled data): SSL methods show **clear advantage** (5-15% improvement)
- For **cross-domain transfer**: SSL pretraining provides **modest improvement** (3-8%) over random initialization
- For **RUL prediction**: Very few SSL methods have been evaluated; no clear SOTA

**Bottom line on SSL:** Self-supervised pretraining helps, especially with limited labels, but no vibration-specific foundation model exists. There is a clear opportunity for a model pretrained on diverse vibration datasets.

---

## 5. Action-Conditioned / Varying-Condition Models

### 5.1 Speed-Varying Fault Diagnosis

This is a practical problem: real machines don't run at constant speed.

| Method | Approach | Result | Year | Reference |
|--------|----------|--------|------|-----------|
| STFDAN (Spatiotemporal Feature Fusion Domain Adaptive Network) | Domain adaptation for varying conditions | Improved cross-condition accuracy | 2025 | MDPI Sensors |
| FPOMRE (multitime-frequency ridge extraction) | Adaptive STFT + ridge extraction | Effective under variable speed | 2024 | Shock and Vibration (Wiley) |
| Deep Conv. Sparse Dictionary | Sparse coding for variable speed features | Robust fault extraction | 2024 | ScienceDirect |
| Order Tracking + CNN | Classical resampling + deep learning | Standard approach | Various | Various |
| Physics-Informed HMM | Speed-integrated Hidden Markov Model | Improved variable-speed diagnosis | 2025 | ASCE-ASME J. Risk and Uncertainty |

### 5.2 Models That Take Operating Conditions as Input

Most approaches treat operating conditions as **domain labels** for domain adaptation, not as explicit model inputs. Very few methods actually condition the model on speed/load:

- Order tracking (classical): resamples signal to angular domain, removing speed dependence
- Some CNN models take speed as an auxiliary input feature
- Physics-informed approaches embed speed in the governing equations

**This is a significant gap.** Almost no work treats operating conditions as an **action variable** that modulates the predicted signal dynamics (the way V-JEPA2 treats robot actions). A model that says "given this vibration state and a speed change from 1000 to 2000 RPM, predict the next vibration state" is essentially unexplored.

### 5.3 Prediction Under Changing Conditions

No existing work attempts to **predict** how vibration signals will change when operating conditions change. The literature focuses on **diagnosing** faults under varying conditions (classification), not **predicting** signal evolution (generation/forecasting).

**Bottom line on action-conditioning:** This is the clearest gap in the literature. Treating operating conditions as actions that modulate dynamics is novel in this field. Every existing approach either ignores condition changes or treats them as domain adaptation problems.

---

## 6. The "Is JEPA Even Needed?" Question

### 6.1 What's the Simplest Competitive Method?

For **CWRU classification**: Envelope analysis + SVM achieves ~95-97%. A 2-layer 1D CNN achieves 99%+. No complex architecture needed.

For **Paderborn classification**: A well-tuned ResNet or 1D CNN achieves ~95%+.

For **Cross-domain transfer**: Simple CNN + MMD/CORAL domain adaptation achieves ~80-90% (CWRU -> Paderborn).

For **RUL prediction**: Health indicator (RMS trending + kurtosis) + exponential regression achieves RMSE ~0.08-0.15. Not great, but functional.

### 6.2 Traditional Signal Processing

| Method | What It Does | When It Works | When It Fails |
|--------|-------------|---------------|---------------|
| Envelope Analysis (HFRT) | Demodulates bearing fault impulses | Localized faults with clear characteristic frequencies | Early-stage faults, distributed faults, heavy noise |
| Spectral Kurtosis | Finds optimal demodulation band | Similar to envelope but more automated | Requires some fault energy in observable band |
| Order Tracking | Removes speed variation effects | Variable speed diagnosis | Requires tachometer or speed estimate |
| Cepstral Analysis | Detects periodic modulation patterns | Gearbox faults, families of harmonics | Low SNR, early faults |
| Wavelet Packet Decomposition | Multi-resolution time-frequency analysis | Non-stationary signals | Requires wavelet selection |
| EMD/VMD | Adaptive decomposition | Non-stationary, nonlinear signals | Mode mixing, computational cost |

**These methods work well in practice** for obvious faults in controlled settings. An experienced vibration analyst with envelope analysis can diagnose most bearing faults. The challenge is automation, generalization, and early detection.

### 6.3 Does Deep Learning Meaningfully Outperform Traditional Methods?

**On benchmarks:** Yes, but the margin is small when the benchmark is easy (CWRU).

**On cross-domain tasks:** Yes, clearly. Traditional methods require re-tuning of demodulation bands, threshold settings, and feature selection for each new machine. Deep learning with domain adaptation handles this automatically (though imperfectly).

**On real-world industrial data:** The evidence is mixed. Several survey papers note:
- Most deep learning results are on lab data, not industrial data (PMC review, 2025)
- The gap between benchmark performance and industrial deployment is large
- Real industrial data has noise, missing data, class imbalance, and distribution shift that benchmarks don't capture
- "Most methods are formulated under the assumption of complete, balanced, and abundant data, which often does not align with real-world engineering scenarios" (Frontiers in Mechanical Engineering, 2025)

**Honest answer:** Deep learning provides clear advantages for (a) automation, (b) handling variable conditions, and (c) processing raw signals without hand-crafted features. But for a well-understood machine with a competent vibration analyst, traditional methods often suffice.

### 6.4 What Are the Actually Unsolved Problems?

Based on the literature review, these problems remain genuinely unsolved:

1. **Cross-domain generalization without target data**: Achieving >90% accuracy when deploying a model trained on lab data to a completely new industrial machine, with no labeled or unlabeled target data. Current best: ~36-45% without adaptation.

2. **Early fault detection**: Detecting incipient faults before degradation indicators (RMS, kurtosis) show obvious trends. Most methods work well only after ~50% of remaining life has passed.

3. **Cross-component transfer**: Learning fault physics from bearings and applying to gearboxes (or vice versa). Essentially unexplored.

4. **Prediction under varying conditions**: Forecasting how vibration will evolve when operating conditions change. Not just classifying faults under varying conditions, but actually predicting signal dynamics.

5. **Unified multi-source models**: A single model that handles diverse sensor configurations, sampling rates, and component types. Currently, each paper trains a bespoke model per dataset.

6. **Compound/novel fault detection**: Identifying fault types not seen during training. Open-set recognition for mechanical faults is in its infancy.

7. **Uncertainty quantification**: Knowing when the model doesn't know. Critical for safety applications but rarely addressed.

8. **Explainability**: Understanding what the model learned. Regulatory requirements in aviation, nuclear, etc. demand interpretable diagnostics.

---

## 7. Where the Real Gaps Are (And Where JEPA Could Fill Them)

### Gap 1: No Vibration Foundation Model
**Status:** Completely open. No model trained on diverse vibration datasets that can be fine-tuned for multiple downstream tasks.
**JEPA opportunity:** A self-supervised model pretrained on the 7+ bearing and 3+ gearbox datasets in Mechanical-Components could be the first.
**Difficulty:** Medium. Data exists. The challenge is handling different sampling rates and sensor configurations.

### Gap 2: Cross-Component Transfer
**Status:** Essentially unexplored. No work on bearing-to-gearbox transfer.
**JEPA opportunity:** If JEPA learns fundamental vibrational physics (impact-resonance-decay, amplitude modulation), these should transfer across components.
**Difficulty:** High. Bearing and gearbox fault signatures are physically different. Success would be genuinely novel.

### Gap 3: Action-Conditioned Prediction
**Status:** No existing work treats operating conditions as actions for predictive world modeling.
**JEPA opportunity:** This maps directly to V-JEPA2's action-conditioned prediction. "Given this vibration state and a speed ramp from 1000 to 2000 RPM, predict the next latent vibration state."
**Difficulty:** Medium-High. The 412 transition samples in Mechanical-Components dataset provide training data, but it's limited.

### Gap 4: Prediction-Based Early Detection
**Status:** Current RUL methods detect degradation after it's obvious. Predicting subtle changes before traditional indicators react is an open problem.
**JEPA opportunity:** A world model that predicts next-state vibration could detect anomalies when predictions diverge from observations. This is conceptually different from classification or regression -- it's prediction error monitoring.
**Difficulty:** High. Requires the model to learn normal dynamics accurately enough that subtle deviations are detectable.

### Gap 5: Generalization Without Target Data
**Status:** Domain adaptation methods require unlabeled target data. Zero-shot cross-domain is at 36-45%.
**JEPA opportunity:** A model that learns physics-grounded representations (not dataset-specific features) could generalize better. Physics don't change between labs.
**Difficulty:** Very high. This is arguably the hardest open problem in the field.

---

## 8. Verdict: Is Mechanical-JEPA Justified?

### What Mechanical-JEPA Should NOT Claim
- "We achieve higher accuracy on CWRU" -- irrelevant; CWRU is solved
- "We outperform CNNs on single-dataset classification" -- marginal gains not worth a new architecture
- "We apply JEPA to vibration data" -- novelty of application alone is insufficient

### What Would Justify Mechanical-JEPA

**Tier 1 (Strong justification -- if any one is achieved):**
1. A pretrained model that fine-tunes to >90% accuracy on a new bearing/gearbox dataset with <10 labeled samples (few-shot)
2. Cross-component transfer: train on bearings, achieve >80% on gearbox faults (or vice versa)
3. Action-conditioned prediction: given operating condition changes, predict latent vibration state evolution

**Tier 2 (Moderate justification -- needs multiple):**
4. Cross-dataset generalization (CWRU -> Paderborn) without target data, achieving >85% accuracy
5. Earlier RUL degradation detection than HI-based baselines (detect at 80% remaining life vs. 50%)
6. Single pretrained model that achieves competitive results across 3+ different datasets

**Tier 3 (Interesting but not sufficient alone):**
7. Self-supervised features that match supervised accuracy with 50% fewer labels
8. Improved RMSE on any single RUL benchmark
9. Better accuracy on any single classification benchmark

### Recommended Evaluation Protocol

To make a convincing case, Mechanical-JEPA should be evaluated on:

| Experiment | Dataset(s) | Baseline to Beat | Success Criterion |
|-----------|-----------|-------------------|-------------------|
| Standard classification | CWRU, Paderborn, MFPT | 1D CNN, ResNet | Match (not beat -- this is table stakes) |
| Cross-dataset transfer | Train CWRU -> Test Paderborn | Domain adaptation (MMD, DANN) | >85% without target data |
| Few-shot learning | All datasets, 1-10 labels/class | Siamese-WDCNN, ProtoNet | >80% with 5 labels/class |
| Cross-component transfer | Bearings -> Gearboxes | Train from scratch on gearbox | Any improvement |
| RUL prediction | FEMTO, XJTU-SY | CNN-GRU-MHA (RMSE ~0.044) | Competitive RMSE |
| Action-conditioned | Mendeley, MCC5-THU (transitions) | N/A (novel task) | Meaningful predictions under condition change |
| Representation quality | All datasets | t-SNE/UMAP visualization | Clean separation of fault types AND conditions |

### The Honest Bottom Line

**Mechanical-JEPA is justified -- but only if it targets the right problems.** Specifically:

1. **The world-model / prediction angle is genuinely novel** for vibration. No one is doing latent-space prediction of vibration dynamics conditioned on operating parameters. This alone justifies the research direction.

2. **Cross-component transfer is unexplored territory.** If JEPA learns physics-grounded representations that transfer from bearings to gearboxes, that's a real contribution.

3. **A vibration foundation model is needed** and doesn't exist. Being first to pretrain on diverse vibration sources and demonstrate broad fine-tuning capability would be significant.

4. **Single-dataset classification is NOT a justification.** The field has 100+ papers achieving 99%+ on CWRU. Do not waste effort competing on this.

5. **The risk is real:** If Mechanical-JEPA only matches existing methods on standard benchmarks, it will be seen as "JEPA for the sake of JEPA." The architecture must demonstrate capabilities that simpler methods cannot achieve -- prediction, transfer, or few-shot generalization.

---

## Key References

### Surveys and Benchmarks
- Braga et al., "Towards Better Benchmarking Using the CWRU Bearing Fault Dataset," Mech. Syst. Signal Process., 2022 -- Critical analysis of CWRU data leakage
- "From Theory to Industry: A Survey of Deep Learning-Enabled Bearing Fault Diagnosis in Complex Environments," Expert Syst. Appl., 2025 -- Comprehensive DL survey
- "A Comprehensive Review of Deep Learning-Based Fault Diagnosis Approaches for Rolling Bearings," AIP Advances, Feb 2025
- "Fault Detection and Diagnosis in Industry 4.0: A Review on Challenges and Opportunities," Sensors, Jan 2025

### Classification SOTA
- Feisa, "1D Deep CNN for Bearings Under Unbalanced Health States," Structural Control and Health Monitoring, 2025 (100% on CWRU)
- Pham et al., "Unsupervised Bearing Fault Diagnosis Using Masked SSL and Swin Transformer," MDPI Machines, 2025 (100% CWRU, 99.53% Paderborn)
- Khanam et al., "Bearing Fault Diagnosis with Envelope Analysis and ML Using CWRU," IEEE Access, 2023

### Cross-Domain Transfer
- "A New Cross-Domain Approach Based on Multiscale CNN and Adversarial Subdomain Adaptation," Nondestructive Testing and Eval., 2025
- "Transformer-based Conditional Generative Transfer Learning for Cross-Domain Fault Diagnosis," Nature Scientific Reports, 2025
- "Domain Adversarial Transfer Learning Bearing Fault Diagnosis Model," PMC, 2025
- Zhao et al., DG-PHM repository: github.com/CHAOZHAO-1/DG-PHM (domain generalization for PHM)

### RUL Prediction
- "Remaining Useful Life Prediction Based on Transfer Learning Integrated with CNN-GRU-MHA," Applied Sciences, 2024 (RMSE 0.0443 on FEMTO)
- "Enhanced Bearing RUL Prediction Based on Dynamic Temporal Attention and Mixed MLP," Autonomous Intelligent Systems, 2024
- "Transformer-Bi-LSTM for RUL Prediction," MDPI Applied Sciences, 2025 (RMSE 0.0442 on XJTU-SY)
- "Deep BiLSTM for RUL Prediction," AIMS ERA, 2024 (RMSE 0.0281 on IMS)

### Self-Supervised Learning
- "Self-Supervised Progressive Learning for Fault Diagnosis Under Limited Labeled Data," Neurocomputing, 2025
- "Fault Diagnosis via Multi-Sensor Fusion with Auxiliary Contrastive Learning," Eng. Appl. of AI, 2025
- "Unsupervised Bearing Fault Diagnosis Using Masked Self-Supervised Learning and Swin Transformer," MDPI Machines, 2025

### Variable Conditions
- "Bearing Fault Diagnosis for Varying Operating Conditions Based on Spatiotemporal Feature Fusion," MDPI Sensors, 2025
- "Rolling Bearings Fault Diagnosis Under Variable Speed Based on Multitime-Frequency Ridge Extraction," Shock and Vibration, 2024
- "Physics-Informed Speed-Integrated HMM for Variable Operating Conditions," ASCE-ASME J., 2025
- "Deep Convolutional Sparse Dictionary Learning Under Variable Speed," ScienceDirect, 2024

### JEPA Architecture (Context)
- Nam et al., "Causal-JEPA: Learning World Models through Object-Level Latent Interventions," arXiv:2602.11389, Feb 2026
- Zhang et al., "ThinkJEPA: Empowering Latent World Models with Large VLM Reasoning," arXiv:2603.22281, Mar 2026
- Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," arXiv:2506.09985, 2025
- LeCun, "A Path Towards Autonomous Machine Intelligence," 2022

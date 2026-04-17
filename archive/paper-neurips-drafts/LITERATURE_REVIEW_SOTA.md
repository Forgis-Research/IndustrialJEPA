# Literature Review: Bearing RUL Prediction & Grey Swan Failure Forecasting (2024–2026)

**Compiled:** 2026-04-09
**Focus:** Absolute latest, reputable, validated SOTA for (1) bearing RUL prediction and (2) grey swan / rare mechanical failure prediction.

---

## 1. Bearing RUL Prediction — SOTA Benchmark Results

### 1.1 Best Reported Normalized RMSE (2024–2026)

| Method | Venue | Year | FEMTO nRMSE | XJTU-SY nRMSE | Architecture | SSL? | Transfer? |
|--------|-------|------|-------------|---------------|--------------|------|-----------|
| CNN-GRU-MHA + TL | MDPI Applied Sciences | 2024 | **0.044** | **0.069** | CNN + GRU + Multi-Head Attention | No | Yes (cross-condition) |
| DCSSL | Scientific Reports | 2026 | "SOTA" (number not public) | — | Dual contrastive SSL → fine-tune | **Yes** | No |
| 1D-DCAE + BiLSTM + TPA | — | 2024 | 0.079 | — | Autoencoder + BiLSTM + Temporal Attention | No | No |
| Mamba-SDP | J. Mech. Sci. Tech. | 2024 | ~0.11 (est.) | ~0.11 (est.) | Mamba + Channel FFT + Scaled Dot-Product Attn | No | No |
| MDSCT | Heliyon (PMC) | 2024 | 0.124 | 0.160 | Multi-scale Depthwise Sep Conv + Transformer | No | No |
| HIWSAN | — | 2024 | — | 0.119 | Cross-condition domain adaptation | No | Yes |
| ERCDAN | Reliability Eng. & System Safety | 2024 | ~0.13–0.15 | ~0.13–0.15 | Cross-machine domain adaptation | No | Yes (cross-machine) |
| SAGCN-SA | — | 2024 | — | 0.170 | Graph Conv + Self-Attention | No | No |
| DTA-MLP | Autonomous Intelligent Systems | 2025 | "SOTA" claimed | "SOTA" claimed | CWT + Channel-Temporal Mixed MLP + Dynamic Attn | No | No |
| TcLstmNet-CBAM | Scientific Reports | 2025 | ~10% better than CNN-LSTM | RMSE=3.12 (raw min) | TCN + LSTM + CBAM Attention | No | No |

**Caveat:** These numbers are not directly comparable across papers — different normalization schemes, train/test splits, and FPT identification rules. The "Rethinking RUL Prediction" paper (PHM Society 2025) explicitly flags this as the field's main weakness.

### 1.2 Architecture Landscape

| Architecture Family | Representative Paper(s) | Key Idea | Strengths | Weaknesses |
|---------------------|------------------------|----------|-----------|------------|
| **Hybrid CNN-LSTM-Transformer** | CNN-GRU-MHA, MDSCT, TcLstmNet-CBAM | CNN for local features, RNN for temporal, attention for global | Dominant paradigm; best raw numbers | Complex; no principled pretraining |
| **Mamba / State Space Models** | Mamba-SDP (2024), Enhanced Mamba (2025) | Selective state space for long-range dependencies | 7–10% over Transformers on long sequences; linear complexity | New; fewer benchmarks |
| **Graph Neural Networks** | SAGCN-SA | Graph structure captures inter-sensor dependencies | Principled multi-channel modeling | Worse raw RMSE than hybrids |
| **Physics-Informed (PINNs)** | PIDL-GP (Springer 2026), Hybrid Paris PINN | Paris crack-growth law (da/dN = C·DeltaK^m) in loss | Reduces data requirements; physically grounded | Only for known crack physics |
| **Self-Supervised Contrastive** | DCSSL (2026), CPC+Transformer (2024), Contrastive Siamese (2024) | Pretrain on unlabeled vibration → fine-tune for RUL | Best approach for sparse labels (10–30% gain) | Under-explored for RUL (most SSL focuses on fault classification) |
| **LLM / Foundation Model** | LM4RUL (arXiv 2025), PHM-GPT (Science China 2025) | Fine-tune pretrained language model for vibration RUL | Leverages massive pretraining; zero-shot potential | Very early stage; no rigorous FEMTO/XJTU-SY benchmarks yet |
| **Diffusion Models** | Defect-Guided Conditional Diffusion (ASCE-ASME 2025) | Generate synthetic degradation trajectories for augmentation | Addresses tiny-dataset problem directly | Not a predictor itself; augmentation pipeline |

### 1.3 Self-Supervised Learning for RUL — Detailed

| Paper | Venue | Year | SSL Method | Dataset | Key Result |
|-------|-------|------|-----------|---------|------------|
| DCSSL (Dual-Dimensional Contrastive SSL) | Scientific Reports | 2026 | Random cropping + timestamp masking; dual temporal + instance contrastive loss | FEMTO | Claims SOTA; first dual-level contrastive for RUL |
| Contrastive SSL + Nested Siamese | Reliability Eng. & System Safety | 2024 | Nested Siamese network; integrates RUL signal into pretraining | PRONOSTIA | Superior under sparse labeling |
| CPC + Transformer (non-full lifecycle) | Advanced Engineering Informatics | 2024 | Contrastive Predictive Coding encoder → Transformer decoder | FEMTO, XJTU-SY | Handles incomplete trajectories |
| Self-Supervised Domain Adaptation | Reliability Eng. & System Safety | 2024 | Degradation ordering as self-supervisory signal | Cross-condition | Cross-condition transfer |
| SSPCL (Momentum Contrast) | — | 2024 | MCL for instance-level discrimination | FEMTO-ST | Incipient fault detection |

### 1.4 Strategies for Handling Tiny Data (~6–15 Trajectories)

| Strategy | Representative Papers | Typical Gain | Maturity |
|----------|-----------------------|-------------|----------|
| Sliding windows (basic) | Everyone | Baseline requirement | Standard |
| Cross-condition transfer | CNN-GRU-MHA (2024) | 20–40% RMSE improvement | Mature |
| Cross-machine domain adaptation | ERCDAN (RESS 2024), JDATransformer (2025) | Works but degrades ~50% cross-dataset | Active research |
| SSL pretraining → fine-tune | DCSSL (2026), CPC+Transformer (2024) | 10–30% with sparse labels | Emerging |
| Diffusion-based augmentation | ASCE-ASME 2025 | Synthetic degradation trajectories | Early |
| Physics constraints (Paris law) | PIDL-GP (2026) | Reduces data needs via inductive bias | Niche |
| Weibull-informed ML | tvhahn/weibull-knowledge-informed-ml | Prior on lifetime distribution | Underused |

### 1.5 Datasets Used by SOTA

| Dataset | Bearings | Complete Run-to-Failure | Sampling Rate | Used By |
|---------|----------|------------------------|---------------|---------|
| FEMTO / PRONOSTIA (PHM 2012) | 17 | ~16 | 25.6 kHz | Nearly all papers |
| XJTU-SY | 15 | 15 | 25.6 kHz | Nearly all papers |
| IMS / NASA | 12 | 4–5 | 20 kHz | Some papers |
| **Total available** | **44** | **~35–36** | — | — |

No paper has meaningfully expanded the RUL benchmark beyond these ~36 trajectories.

---

## 2. Grey Swan Prediction — Rare Structural Failures

### 2.1 Core Grey Swan Papers

| Paper | Venue | Year | Domain | Key Finding |
|-------|-------|------|--------|-------------|
| Can AI predict OOD gray swan tropical cyclones? (Sun, Hassanzadeh et al.) | **PNAS** | 2025 | Weather/climate | AI confidently predicts weakening when facing unseen Cat-5 storms; dangerous false negatives. Partial cross-basin transfer for dynamically similar physics. |
| Gray Swan Factory (arXiv 2604.00348) | arXiv | 2026 | Weather/climate | Generative approach: perturb ordinary events toward extreme regimes to create synthetic grey swan training examples. |
| Multi-Phase Degradation Modeling with Jumps | IEEE/CAA J. Auto. Sinica | 2025 | Industrial | Models degradation with abrupt phase transitions and random jumps — formalizes the healthy→rapid-degradation transition. |

### 2.2 Adjacent Domains with Grey Swan Pattern

| Domain | Pattern | Key Paper | Venue/Year | Method |
|--------|---------|-----------|-----------|--------|
| **Battery degradation** | Months of slow capacity fade → sudden thermal runaway | Early prediction of Li-ion degradation with GPT | PMC, 2025 | Pretrained transformer on charge/discharge curves |
| **Battery degradation** | Long healthy → sudden failure | U-H-Mamba | ResearchGate, 2025 | Uncertainty-aware hierarchical Mamba; 146K+ cycles |
| **Battery self-discharge** | Latent risk → sudden thermal runaway | Accurate prediction using self-discharge signal | Wiley, 2024 | Self-discharge as early warning for sudden failure |
| **Wind turbine gearbox** | Months healthy → rapid gear failure | LLMs for wind turbine gearbox prognosis | MDPI Energies, 2025 | GPT-4o / DeepSeek-V3 on SCADA data |
| **Wind turbine bearing** | 12–24 month healthy → failure | Main bearing LSTM | IEEE, 2024 | LSTM on SCADA; long-lead-time prediction |

### 2.3 Foundation Models for Industrial Anomaly Detection

| Paper | Venue | Year | Key Contribution | Relevance to Grey Swan |
|-------|-------|------|------------------|----------------------|
| DADA (Adaptive Bottlenecks + Dual Adversarial Decoders) | **ICLR** | 2025 | Zero-shot time series anomaly detection; multi-domain pretrained | Evaluated on SWAN dataset (rare anomalies) |
| Zero-Shot Anomaly via Synthetic Data | **ICLR** | 2025 | Generates synthetic anomalies during pretraining | Populates grey-swan-equivalent scenarios |
| Adaptive Conformal Anomaly + Foundation Models | **ICLR** | 2026 | Conformal prediction + foundation model features | Calibrated uncertainty for rare failure detection |
| UniTS (Unified Multi-Task Time Series) | **NeurIPS** | 2024 | Multi-task model; 38 datasets; beats 12 forecasting + 20 classification models | Cross-task generalization relevant to transfer |
| MOMENT (Masked Time Series Foundation Model) | **ICML** | 2024 | Masked transformer; 5 tasks; zero-shot/few-shot | Not applied to bearing RUL yet — potential baseline |

### 2.4 LLM / Agent Approaches for PHM

| Paper | Venue | Year | Key Contribution |
|-------|-------|------|------------------|
| LM4RUL (Pre-Trained LLM for Bearing RUL) | arXiv | Jan 2025 | Two-stage LLM fine-tuning; "local scale perception" tokenization of vibration |
| PHM-GPT | Science China | 2025 | Unified LLM for anomaly detection + fault diagnosis; InsPHM-456k instruction tuning dataset |
| PHMForge (Agentic Benchmark) | arXiv | Apr 2026 | 75 PHM scenarios; Claude Sonnet 4 achieved 68% task completion; LLMs fail at multi-asset reasoning (42.7%) |
| Pulse Foundation Model for Bearing PHM | SSRN | 2025 | Pretrained LLM fine-tuned for fault diagnosis + RUL on 4 bearing datasets |

---

## 3. Meta-Analysis and Key Critiques

| Paper | Venue | Year | Key Critique |
|-------|-------|------|-------------|
| Rethinking RUL Prediction (Salinas-Camus, Goebel, Eleftheroglou) | PHM Society | 2025 | Over-optimization for RMSE ignores uncertainty, robustness, interpretability. BiLSTM achieves better RMSE but worse uncertainty calibration than stochastic models. |
| Towards Universal Vibration Analysis Dataset | arXiv | Apr 2025 | Calls for ImageNet-equivalent for vibration transfer learning. Current datasets too small and fragmented. |
| Small Data Challenges for PHM | AI Review (Springer) | 2024 | Systematic review of data scarcity strategies across PHM. |

---

## 4. Identified Research Gaps

| Gap | Current State | Opportunity |
|-----|---------------|-------------|
| **JEPA-style latent prediction for RUL** | Does not exist. DCSSL (2026) is closest but uses contrastive reconstruction, not predictive coding in latent space. | First paper to do "predict masked future representations" for degradation. |
| **Grey swan framing for bearing prognostics** | No paper has explicitly framed bearing RUL as grey swan prediction with OOD generalization + uncertainty quantification. | Novel framing with theoretical backing from PNAS 2025 cyclone paper. |
| **Cross-machine SSL transfer for RUL** | Best cross-dataset results degrade ~50%. | 7.2B-datapoint pretraining corpus across 16 sources is larger than anything published. |
| **Physics-informed + SSL** | Paris-law PINNs and contrastive SSL are entirely separate communities. | Combine JEPA predictive coding with physics constraints. |
| **Foundation model fine-tuned on bearing RUL** | MOMENT, UniTS, Chronos exist for general time series; nobody has fine-tuned them on FEMTO/XJTU-SY with proper evaluation. | Straightforward baseline comparison. |
| **Standardized RUL evaluation protocol** | Different papers use incompatible normalization, splits, FPT rules. | Propose and adhere to a rigorous protocol. |

---

## 5. Full Reference List

### Bearing RUL — Architecture Papers

1. CNN-GRU-MHA + Transfer Learning. MDPI Applied Sciences, 2024. FEMTO nRMSE=0.044, XJTU-SY nRMSE=0.069.
2. MDSCT (Multi-scale Depthwise Separable Conv + Transformer). Heliyon / PMC11481647, 2024. FEMTO nRMSE=0.124, XJTU-SY nRMSE=0.160.
3. TcLstmNet-CBAM (TCN + LSTM + Attention). Scientific Reports, 2025. doi:10.1038/s41598-025-98845-9.
4. DTA-MLP (Dynamic Temporal Attention + Mixed MLP). Autonomous Intelligent Systems (Springer), Jan 2025. doi:10.1007/s43684-024-00088-4.
5. SAGCN-SA (Self-Attention + Adaptive Graph Conv). 2024. XJTU-SY nRMSE=0.170.
6. Mamba-SDP (Mamba + Channel FFT + Attention). J. Mechanical Science and Technology, Dec 2024. doi:10.1007/s12206-025-1114-4. PHM Score=0.82.
7. Enhanced Mamba + Multi-Head Attention. Scientific Reports, Feb 2025. doi:10.1038/s41598-025-91815-1.

### Bearing RUL — Self-Supervised Learning

8. DCSSL (Dual-Dimensional Contrastive SSL for RUL). Scientific Reports, 2026. doi:10.1038/s41598-026-38417-7.
9. Contrastive SSL + Nested Siamese for Sparse Labeled Prognostics. Reliability Engineering & System Safety, 2024. HAL: hal-04808670v1.
10. CPC + Transformer for Non-Full Lifecycle RUL. Advanced Engineering Informatics, 2024. doi:10.1016/j.aei.2024.102721 (approx).
11. Self-Supervised Domain Adaptation for RUL. Reliability Engineering & System Safety, 2024. doi:10.1016/j.ress.2024.103685 (approx).

### Bearing RUL — Transfer Learning & Domain Adaptation

12. ERCDAN (Cross-Machine RUL). Reliability Engineering & System Safety, 2024. doi:10.1016/j.ress.2024.100516 (approx).
13. JDATransformer (Cross-Domain RUL). Engineering Applications of AI, 2025. doi:10.1016/j.engappai.2025.116173 (approx).
14. HIWSAN (Cross-Condition). 2024. XJTU-SY nRMSE=0.119.

### Bearing RUL — Physics-Informed

15. Physics-Informed Deep Learning + Gaussian Processes for FEMTO. Integrating Materials and Manufacturing Innovation (Springer), Feb 2026. doi:10.1007/s40192-026-00441-w.
16. Hybrid Paris PINN for CFM56-7B Engine Fatigue. ResearchGate, 2025.

### Bearing RUL — LLM / Foundation Model

17. LM4RUL (Pre-Trained LLM for Bearing RUL Transfer). arXiv:2501.07191, Jan 2025.
18. PHM-GPT (Unified LLM for PHM). Science China, 2025.
19. Pulse Foundation Model for Bearing PHM. SSRN, 2025.
20. PHMForge (Agentic Benchmark for PHM). arXiv:2604.01532, Apr 2026.

### Bearing RUL — Diffusion / Augmentation

21. Defect-Guided Conditional Diffusion for Bearing RUL. ASCE-ASME J. Risk and Uncertainty, 2025.
22. Gear RUL via Diffusion-Generated Health Index. Scientific Reports, 2024.

### Grey Swan Prediction

23. Can AI predict OOD gray swan tropical cyclones? Sun, Hassanzadeh et al. PNAS, 2025. doi:10.1073/pnas.2420914122. arXiv:2410.14932.
24. Gray Swan Factory: Making Extreme Events from Ordinary Cyclones. arXiv:2604.00348, Apr 2026.
25. Multi-Phase Degradation Modeling with Jumps. IEEE/CAA J. Automatica Sinica, 2025. doi:10.1109/JAS.2024.124791.

### Foundation Models for Time Series / Anomaly

26. DADA (General Time Series Anomaly Detector). ICLR 2025.
27. Zero-Shot Time Series Anomaly via Synthetic Data. ICLR 2025. OpenReview: Z4T26VztkU.
28. Adaptive Conformal Anomaly Detection + Foundation Models. ICLR 2026.
29. UniTS (Unified Multi-Task Time Series). NeurIPS 2024.
30. MOMENT (Masked Time Series Foundation Model). ICML 2024. arXiv:2402.03885.

### Battery / Adjacent Domain Grey Swans

31. Early prediction of Li-ion degradation with GPT. PMC, 2025.
32. U-H-Mamba for Li-ion Battery RUL. ResearchGate, 2025. 146K+ cycles.
33. Accurate battery prediction using self-discharge signal. Wiley, 2024.
34. LLMs for Wind Turbine Gearbox Prognosis. MDPI Energies, 2025.
35. Wind Turbine Main Bearing LSTM. IEEE, 2024.

### Meta-Analysis / Critiques

36. Rethinking RUL Prediction. Salinas-Camus, Goebel, Eleftheroglou. PHM Society, 2025.
37. Towards Universal Vibration Analysis Dataset. arXiv, Apr 2025. arXiv:2504.11581.
38. Small Data Challenges for PHM. AI Review (Springer), 2024. doi:10.1007/s10462-024-10820-4.

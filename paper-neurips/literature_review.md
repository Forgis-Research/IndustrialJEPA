# Deep Literature Review: Self-Supervised Learning for Mechanical Grey Swan Prediction

**Date**: 2026-04-09
**Scope**: JEPA, SSL for time series, RUL prediction, grey swan / rare event prediction
**Papers reviewed**: 50+ (from 3 parallel research agents + existing reviews)

---

## 1. Joint Embedding Predictive Architectures (JEPA)

### 1.1 I-JEPA (Assran et al., CVPR 2023)
- **Key idea**: Self-supervised image representation learning by predicting representations of masked image blocks from visible context in a shared latent space.
- **Method**: ViT encoder + lightweight predictor. Multi-block masking with EMA target encoder. Predictions in representation space avoid pixel-level reconstruction bias.
- **Results**: Competitive with MAE on ImageNet linear probing (73.4% top-1); outperforms pixel-reconstruction methods on semantic tasks; better data efficiency than contrastive methods.
- **Gap**: Image-only. No temporal modeling. Our work extends JEPA to 1D vibration time series with temporal structure.

### 1.2 V-JEPA (Bardes et al., TMLR 2024)
- **Key idea**: Video JEPA with spatiotemporal masking — 90% masking forces the model to learn motion and dynamics rather than static appearance.
- **Method**: ViT encoder processes visible spatiotemporal patches. Predictor operates in latent space. 90% masking with spatiotemporal tube structure.
- **Results**: SOTA on action recognition (Kinetics-400, Something-Something v2) among self-supervised methods. Frozen encoder evaluation demonstrates learned dynamics.
- **Gap**: Video domain. Our 62.5% masking is lower — adopted from V-JEPA's insight that high masking prevents context-average shortcuts.

### 1.3 V-JEPA 2 (Meta, 2025)
- **Key idea**: Scaled-up V-JEPA with 1M hours of video pretraining. Demonstrates that JEPA pretraining scales efficiently to massive datasets.
- **Method**: Same core architecture as V-JEPA but at much larger scale. Downstream adaptation for understanding, prediction, and planning.
- **Results**: Only 62 hours of robot data needed for action-conditioned planning. Strong zero-shot and few-shot transfer.
- **Gap**: Vision-scale data. Our domain has far less data but also far less visual complexity. Validates the pretraining-then-finetune paradigm we adopt.

### 1.4 Brain-JEPA (Wang et al., NeurIPS 2024 Spotlight)
- **Key idea**: JEPA for fMRI brain dynamics with gradient-based positioning and domain-specific masking strategies.
- **Method**: Gradient positioning replaces standard positional encodings with brain-region-aware embeddings. Spatiotemporal masking across brain regions. EMA target encoder.
- **Results**: SOTA on brain disorder diagnosis (88% sex classification, MSE 0.50 for age prediction). Cross-ethnic transfer: 66%.
- **Gap**: Closest analogue to our work — applies JEPA to a non-vision domain with physics-informed structure. Our work applies JEPA to mechanical vibration (much higher temporal resolution, different physics).

### 1.5 C-JEPA / Causal-JEPA (Nam et al., arXiv 2026)
- **Key idea**: Object-level masking as latent interventions — mask entire objects to force reasoning about inter-object interactions.
- **Method**: Frozen object-centric encoder → object slots → mask entire object trajectories → predict from other objects. Identity anchors preserve object tracking.
- **Results**: +20% on counterfactual reasoning (CLEVRER). 88.67% on Push-T planning. 8x faster MPC than patch-based methods.
- **Gap**: Inspires our planned physics-group masking — masking entire sensor modalities to force cross-modal prediction.

### 1.6 TS-JEPA (NeurIPS Workshop on Time Series, 2024)
- **Key idea**: First JEPA adaptation for time series. Temporal masking with L1 loss.
- **Method**: Adapts I-JEPA architecture to 1D with temporal patch masking. Uses 70% masking. EMA target encoder.
- **Results**: Improvements over masked autoencoder baselines on time series classification (UEA archive).
- **Gap**: General time series classification, not mechanical prognostics. No transfer learning evaluation. No RUL prediction.

### 1.7 MTS-JEPA (arXiv, 2026)
- **Key idea**: Multivariate time series JEPA with codebook regularization to prevent representation collapse.
- **Method**: Dual-resolution encoding (fine dynamics + coarse trends). Vector-quantized codebook bottleneck forces discrete latent states. Cross-scale prediction.
- **Results**: Improved performance on multivariate classification and forecasting benchmarks.
- **Gap**: General multivariate time series. No mechanical/prognostics application. Codebook regularization is relevant to our JEPA instability finding — we observe oscillation after epoch 2.

---

## 2. Self-Supervised Learning for Time Series

### 2.1 TS2Vec (Yue et al., AAAI 2022)
- **Key idea**: Universal time series representation via hierarchical contrastive learning across temporal and instance dimensions.
- **Method**: Augmentation via timestamp masking and random cropping. Contrastive loss at multiple temporal granularities.
- **Results**: SOTA on 125 UEA datasets for classification. Strong forecasting on ETT, Electricity, Weather.
- **Gap**: General-purpose representations. Not optimized for degradation dynamics. No RUL evaluation.

### 2.2 TNC (Tonekaboni et al., ICLR 2021)
- **Key idea**: Temporal Neighborhood Coding — exploits stationarity of temporal neighborhoods as a self-supervised signal.
- **Method**: Positive pairs from stationary windows (detected by ADF test). Negative pairs from non-stationary transitions.
- **Results**: Improvements on clinical time series (PhysioNet, HAR).
- **Gap**: Assumes stationarity within windows. Bearing degradation is fundamentally non-stationary. Our temporal contrastive approach explicitly leverages non-stationarity.

### 2.3 CoST (Woo et al., ICLR 2022)
- **Key idea**: Contrastive learning of disentangled seasonal and trend representations for time series forecasting.
- **Method**: Frequency-domain and time-domain contrastive losses disentangle periodic and trend components.
- **Results**: SOTA on long-term forecasting benchmarks.
- **Gap**: Forecasting-focused. The seasonal/trend decomposition could be relevant for separating periodic vibration from degradation trend, but not explored for prognostics.

### 2.4 TS-TCC (Eldele et al., IJCAI 2021)
- **Key idea**: Temporal and contextual contrasting with weak and strong augmentations.
- **Method**: Cross-view prediction between weakly and strongly augmented versions. Temporal contrasting via autoregressive prediction.
- **Results**: Strong results on HAR, sleep staging, fault diagnosis.
- **Gap**: Evaluated on fault diagnosis (classification), not RUL prediction. No cross-dataset transfer.

### 2.5 SimMTM (Dong et al., NeurIPS 2023)
- **Key idea**: Simple pre-training framework for masked time-series modeling. Reconstructs masked points from multiple masked versions to recover manifold structure.
- **Method**: Multiple random masks of same series → aggregate predictions → reconstruct. Avoids learning simple interpolation.
- **Results**: SOTA on forecasting and classification with self-supervised pretraining.
- **Gap**: Reconstruction-based (predicts raw values, not representations). No mechanical/RUL application.

### 2.6 PatchTST (Nie et al., ICLR 2023)
- **Key idea**: Patching + channel independence for Transformer-based time series forecasting. Self-supervised pretraining via masked patch prediction.
- **Method**: Divide time series into patches. Channel-independent processing. Masked self-supervised pretraining.
- **Results**: SOTA on long-term forecasting. Channel independence paradox: independent channels often beat channel-mixing models.
- **Gap**: Forecasting-focused, channel-independent. Our multivariate extension will explicitly model cross-channel dependencies (physics groups).

### 2.7 TimeMAE (Cheng et al., 2023)
- **Key idea**: Masked autoencoders for time series with decoupled representation learning.
- **Method**: Separate encoding of visible and masked patches. Reconstruction target in representation space.
- **Results**: Competitive on UEA classification benchmarks.
- **Gap**: General classification. No mechanical/prognostics application.

### 2.8 TF-C (Zhang et al., NeurIPS 2022)
- **Key idea**: Self-supervised contrastive pretraining via time-frequency consistency. Aligns representations across time and frequency domains.
- **Method**: Separate time-domain and frequency-domain encoders. Contrastive loss aligns their representations.
- **Results**: Cross-domain transfer on medical and mechanical datasets. +15% on EMG-to-EEG transfer.
- **Gap**: Closest to our contrastive approach. Evaluated on fault classification, not RUL. Does not address the specific degradation-aware representation learning we target.

---

## 3. Foundation Models for Time Series

### 3.1 TimesFM (Das et al., Google, ICML 2024)
- **Key idea**: Decoder-only foundation model for zero-shot time series forecasting.
- **Method**: Trained on a large corpus of real-world and synthetic time series. Decoder-only Transformer with input patching.
- **Results**: Competitive with supervised approaches on Monash, ETT, and other benchmarks without any fine-tuning.
- **Gap**: Forecasts next values, not degradation indicators. No mechanical/prognostics evaluation.

### 3.2 Chronos (Ansari et al., Amazon, 2024)
- **Key idea**: Tokenizes time series values into bins and trains language model architectures for probabilistic forecasting.
- **Method**: Value tokenization → T5-based architecture → next-token prediction. Gaussian mixture augmentation.
- **Results**: Strong zero-shot performance, competitive with task-specific models on 42 datasets.
- **Gap**: Value-level forecasting, not representation learning. Our approach learns representations of degradation state, not forecasts of next values.

### 3.3 Chronos-2 / Chronos-Bolt (Amazon, 2025)
- **Key idea**: Efficient version of Chronos with improved tokenization and smaller model sizes.
- **Method**: Improved architecture for faster inference. Better tokenization scheme.
- **Results**: Comparable to Chronos with significantly lower compute.
- **Gap**: Same fundamental limitation — forecasts values, not degradation-relevant features.

### 3.4 MOMENT (Goswami et al., ICML 2024)
- **Key idea**: Family of open time-series foundation models supporting multiple tasks (forecasting, classification, anomaly detection, imputation).
- **Method**: Masked pretraining on diverse time series. Multi-task fine-tuning. T5-based architecture.
- **Results**: Competitive across all four tasks. Open-source model family.
- **Gap**: General-purpose. Anomaly detection capability is closest to our prognostics task, but does not address RUL prediction or degradation trajectory modeling.

### 3.5 Lag-Llama (Rasul et al., 2024)
- **Key idea**: Foundation model for probabilistic time series forecasting using LLaMA architecture with lag features.
- **Method**: LLaMA architecture with lag-based tokenization. Trained on diverse time series.
- **Results**: Strong probabilistic forecasting with uncertainty quantification.
- **Gap**: Forecasting paradigm. Probabilistic aspect is relevant (uncertainty in RUL predictions), but architecture is designed for next-step prediction.

### 3.6 UniTS (Gao et al., NeurIPS 2024)
- **Key idea**: Unified model across multiple time series tasks with a single architecture.
- **Method**: Multi-task pretraining. Prompt-based task specification. Shared backbone.
- **Results**: Competitive across forecasting, classification, anomaly detection.
- **Gap**: General-purpose multi-task model. Does not address the specific structure of degradation data (run-to-failure episodes).

---

## 4. Remaining Useful Life Prediction

### 4.1 Classical Deep Learning Approaches

#### DCNN for RUL (Li et al., Reliability Engineering & System Safety, 2018)
- **Key idea**: Deep convolutional neural network for direct RUL prediction from sensor data.
- **Method**: Multi-layer CNN with time-window features. Applied to C-MAPSS.
- **Results**: FD001 Score: 274, FD002: 10,412, FD003: 284, FD004: 12,466.
- **Gap**: Fully supervised, no pretraining. Single-dataset evaluation.

#### LSTM for RUL (Zheng et al., IEEE PHM, 2017)
- **Key idea**: Long Short-Term Memory networks capture temporal dependencies in degradation trajectories.
- **Method**: LSTM with sliding window input. Applied to C-MAPSS.
- **Results**: FD001 Score: 338, competitive with CNN approaches at the time.
- **Gap**: Fully supervised, no transfer learning capability.

### 4.2 Modern Architectures

#### TCN-Transformer (various, 2022-2024)
- **Key idea**: Combine temporal convolutional networks for local feature extraction with Transformers for long-range dependencies.
- **Method**: TCN backbone → Transformer encoder → RUL prediction head. Multi-head attention over temporal features.
- **Results**: Current SOTA on C-MAPSS (specific numbers vary by paper; best reported FD001 RMSE ~11-12, Score ~200-250).
- **Gap**: Fully supervised. Requires extensive labeled degradation data.

#### CNN-GRU-MHA (Chen et al., Applied Sciences, 2024)
- **Key idea**: Combining CNN, GRU, and multi-head attention for bearing RUL prediction.
- **Method**: CNN for spatial features → GRU for temporal → multi-head attention for weighting.
- **Results**: nRMSE 0.044 on FEMTO dataset (single-dataset evaluation).
- **Gap**: Single dataset. Handcrafted features. No cross-dataset transfer. Our CNN-GRU-MHA baseline achieves RMSE 0.185 on the harder mixed FEMTO+XJTU task.

### 4.3 Self-Supervised Methods for RUL (Key Gap)

This is the critical gap our paper addresses. Despite the success of self-supervised learning in NLP and vision, **very few papers apply SSL to RUL prediction**:

- **TF-C (Zhang et al., NeurIPS 2022)**: Demonstrates time-frequency contrastive pretraining for fault classification, but does not evaluate on RUL.
- **TS2Vec applied to PHM**: Some works use TS2Vec representations for anomaly detection, but not for quantitative RUL prediction.
- **Domain adaptation**: DANN and CORAL have been applied to cross-domain fault diagnosis, but these are supervised transfer methods, not self-supervised pretraining.

**Our contribution fills this gap**: We are the first to systematically evaluate JEPA and temporal contrastive pretraining for bearing RUL prediction, including cross-dataset transfer evaluation.

### 4.4 Bearing RUL Benchmarks

#### FEMTO/PRONOSTIA (Nectoux et al., IEEE PHM 2012)
- 17 bearings, 3 operating conditions, accelerated degradation
- Standard benchmark for bearing prognostics
- Most published results are single-dataset, making cross-dataset comparison difficult

#### XJTU-SY (Wang et al., IEEE Trans. Reliability, 2020)
- 15 bearings, 3 operating conditions, natural degradation
- Complementary to FEMTO (different test rig, different lifetime distributions)
- Rarely used for cross-dataset evaluation

#### NASA IMS (2007)
- 3 run-to-failure tests, 4-8 channels each
- Long-duration runs (35 days)
- Primarily used for anomaly detection rather than RUL prediction

### 4.5 C-MAPSS Turbofan (Saxena et al., 2008)
- **FD001-FD004**: Standard train/test splits, 21 sensor channels
- **Evaluation metrics**: Score (asymmetric, penalizes late predictions more) and RMSE
- **Current SOTA** (approximate, varies by paper):
  - FD001: RMSE ~11-12, Score ~200-250
  - FD002: RMSE ~17-20, Score ~1500-3000
  - FD003: RMSE ~12-13, Score ~250-350
  - FD004: RMSE ~19-22, Score ~2000-4000
- **Gap**: All published methods are fully supervised. Self-supervised pretraining for C-MAPSS is unexplored.

---

## 5. Grey Swan / Rare Event Prediction

### 5.1 Grey Swan Concept

The term "grey swan" (sometimes "gray rhino") refers to events that are:
- **Rare**: Low probability in any given time window
- **Physically plausible**: Follow known physics, not fundamentally unforeseeable
- **High-impact**: Catastrophic consequences when they occur
- **Predictable in principle**: Given enough data and the right model

In mechanical systems, grey swans include:
- Bearing seizure after extended operation
- Fatigue crack propagation to critical length
- Lubrication failure cascade
- Combined-mode failures (e.g., inner race + rolling element)

### 5.2 Predictive Maintenance with Limited Failure Data

#### Transfer Learning for Fault Diagnosis (Li et al., Neurocomputing, 2020)
- **Key idea**: Systematic review of deep transfer learning for machinery fault diagnosis.
- **Method**: Domain adaptation (DANN, CORAL, MMD), fine-tuning, few-shot approaches.
- **Results**: Transfer learning improves cross-domain fault diagnosis by 5-20% typically.
- **Gap**: Classification focus, not RUL. Transfer is between operating conditions, not datasets.

#### Few-Shot Fault Detection
- Multiple papers (2020-2024) apply meta-learning (MAML, ProtoNet) to fault diagnosis with limited labels
- Typically achieve 80-90% accuracy with 5-10 labeled examples per class
- **Gap**: Classification, not RUL regression. No self-supervised pretraining.

### 5.3 Synthetic Data and Physics-Informed Methods

#### Physics-Informed Neural Networks (Raissi et al., JCP, 2019)
- **Key idea**: Embed physical laws as constraints in neural network training.
- **Method**: Loss function includes PDE residuals. No labeled data needed for physics loss.
- **Results**: Accurate solution of forward and inverse problems.
- **Gap**: Continuous physics. Mechanical degradation is more complex (stochastic, multi-mode).

#### GAN-Based Augmentation for Fault Diagnosis
- Several papers (2021-2024) use GANs to generate synthetic fault data
- WGAN-GP and conditional GANs most common
- Typically improve classification accuracy by 3-8% when real data is scarce
- **Gap**: Classification focus. Quality of synthetic degradation trajectories (not just snapshots) is unexplored.

#### Digital Twins for Prognostics
- Simulation-based approaches create virtual replicas of physical systems
- Physics-based models (FEM, multi-body dynamics) generate synthetic degradation data
- **Gap**: Require detailed physical models. Our approach learns from data, complementing physics-based simulation.

#### Neural ODEs for Degradation (Chen et al., NeurIPS 2018)
- **Key idea**: Continuous-depth networks parameterize ODE dynamics.
- **Method**: Replace discrete layers with continuous dynamics. ODE solver for forward/backward pass.
- **Results**: Compact models for irregular time series.
- **Gap**: Could model degradation dynamics, but requires labeled degradation trajectories. Our self-supervised approach learns degradation representations without explicit physics models.

### 5.4 World Models for Physical Systems

LeCun's "path towards autonomous machine intelligence" (2022) envisions world models that predict future states in latent space — exactly the JEPA paradigm. For mechanical systems, a "mechanical world model" would:
1. Learn a latent space where degradation state is geometrically encoded
2. Predict how that state evolves given operating conditions
3. Enable planning (e.g., optimal maintenance scheduling)

No prior work achieves this for mechanical systems. Our work represents a first step: the temporal contrastive encoder learns a latent space where proximity to failure is partially encoded (PC1 correlation with RUL: 0.648).

---

## 6. Key New Papers from Deep Search (Agent Findings)

### 6.1 Direct Competitors

#### DCSSL (Shen et al., Scientific Reports, 2026)
- **Most direct competitor**: Dual-dimensional contrastive SSL for bearing RUL on FEMTO
- Temporal-level + instance-level contrastive pretraining → fine-tune RUL head
- Claims to outperform supervised SOTA on FEMTO
- **Key differences from us**: Contrastive (not JEPA), requires positive pair design, uses full trajectory context, no cross-dataset transfer evaluation

#### RmGPT (Wang et al., IEEE IoT Journal, 2025)
- GPT-style generative pretraining for rotating machinery (68.5M parameters)
- 4 token types: Signal, Prompt, Time-Frequency, Fault
- 92% one-shot 16-class fault classification on CWRU+Paderborn
- **Key differences**: Generative (predicts raw tokens), no RUL evaluation, no cross-dataset transfer

#### OpenMAE (ACM IMWUT, 2025)
- MAE pretraining on 5M vibration samples from Raspberry Shake seismic network
- FreqCutMix augmentation for frequency-domain data mixing
- +23% downstream accuracy improvement
- **Key differences**: MAE (not JEPA), evaluated on activity recognition not fault diagnosis/RUL

#### LeJEPA/SIGReg (Balestriero & LeCun, arXiv 2025)
- Proves optimal embedding distribution is isotropic Gaussian
- SIGReg: single tractable regularizer replaces EMA/stop-gradient/VICReg
- 79.0% ImageNet-1k linear probe with ViT-H/14
- **Relevance**: Our V8 architecture uses variance regularization; SIGReg is the principled replacement

### 6.2 New RUL SOTA Results

#### TTSNet (Sensors, 2025) — C-MAPSS SOTA
- Transformer + TCN + Self-Attention, three parallel branches
- FD001: RMSE 11.02, Score 194.6 (current best)
- FD002: RMSE 13.25, Score 874.1

#### MDSCT (Heliyon, 2024) — Clean Bearing Comparison
- Multi-scale depth-wise separable conv + Transformer
- FEMTO: RMSE 0.124, XJTU-SY: RMSE 0.160
- Uses full trajectory context (vs our single-window approach)

#### Bi-LSTM-Transformer + EMD (Applied Sciences, 2025)
- XJTU-SY: RMSE 0.0563 (among best reported)
- EMD preprocessing reduces non-stationarity

#### NOMI — Neural ODE for Bearing RUL (Advanced Engineering Informatics, 2024)
- Neural ODE exploits time-invariant latent dynamics for cross-domain transfer
- Outperforms RNN/LSTM domain adaptation on FEMTO→XJTU-SY
- Complementary but different from JEPA: models continuous dynamics vs single-window representations

### 6.3 Grey Swan Literature

#### "Can AI Weather Models Predict Out-of-Distribution Gray Swan Tropical Cyclones?" (arXiv, 2024)
- AI weather models (FourCastNet, GraphCast, GenCast) consistently fail on grey swan extremes
- Strong conceptual anchor for our grey swan framing

#### "Gray Swan Factory" (arXiv, 2026)
- Generates grey swans via gradient optimization through differentiable physics
- Direct methodological analog for our physics-informed synthetic augmentation

#### Grey Swan Supply Chain Framework (IJPR, 2025)
- Formally distinguishes grey swans (structurally predictable) from black swans (epistemically unknowable)
- Best definitional citation for our terminology

#### Physics-Informed Data Augmentation for RUL (RESS, 2024)
- Achieves competitive RUL prediction with ZERO real failure data
- System identification on healthy data → physics-of-failure simulation → synthetic trajectories
- Strongest precedent for our planned synthetic augmentation

#### GAN + Inverse PINN (PLOS ONE, 2025)
- Conditional GAN + physics constraints for rare fault generation
- Strong gains for fault categories with <5% representation

#### Digital Twin Bearing Fault Diagnosis (SHM, 2025)
- Hertz contact model generates synthetic vibration → domain adaptation
- >94% accuracy with only 5 real labeled samples per class

---

## 7. Summary: Key Gaps Our Work Fills

| Gap | Status | Our Contribution |
|-----|--------|-----------------|
| JEPA for mechanical systems | No prior work | First JEPA application to vibration/prognostics |
| Self-supervised pretraining for RUL | Only DCSSL (contrastive, 2026) | First JEPA-based RUL; no augmentation design needed |
| Cross-dataset bearing RUL transfer | Unexplored without domain adaptation | First demonstration with temporal contrastive learning |
| Single-window RUL prediction (no trajectory context) | No prior work with SSL | All competitors use full temporal trajectory |
| Mechanistic analysis of SSL encoders for degradation | No prior work | JEPA vs contrastive: waveform texture vs spectral dynamics |
| Hybrid SSL + handcrafted for prognostics | No prior work | JEPA+HC achieves +75.5% vs time-only |
| Grey swan framing for self-supervised prognostics | No prior work in PHM | Connecting data scarcity paradox to SSL pretraining |
| JEPA vs contrastive for different transfer regimes | No prior work | JEPA wins within-dataset, contrastive wins cross-dataset |

---

## References

*Full BibTeX entries in `latex/references.bib`*

1. Assran et al. (2023). I-JEPA. CVPR.
2. Bardes et al. (2024). V-JEPA. TMLR.
3. Wang et al. (2024). Brain-JEPA. NeurIPS (Spotlight).
4. Nam et al. (2026). C-JEPA. arXiv.
5. TS-JEPA (2024). NeurIPS Workshop.
6. MTS-JEPA (2026). arXiv.
7. Yue et al. (2022). TS2Vec. AAAI.
8. Tonekaboni et al. (2021). TNC. ICLR.
9. Woo et al. (2022). CoST. ICLR.
10. Eldele et al. (2021). TS-TCC. IJCAI.
11. Dong et al. (2023). SimMTM. NeurIPS.
12. Nie et al. (2023). PatchTST. ICLR.
13. Cheng et al. (2023). TimeMAE. arXiv.
14. Zhang et al. (2022). TF-C. NeurIPS.
15. Das et al. (2024). TimesFM. ICML.
16. Ansari et al. (2024). Chronos. arXiv.
17. Goswami et al. (2024). MOMENT. ICML.
18. Rasul et al. (2024). Lag-Llama. arXiv.
19. Gao et al. (2024). UniTS. NeurIPS.
20. Li et al. (2018). DCNN for RUL. RESS.
21. Zheng et al. (2017). LSTM for RUL. IEEE PHM.
22. Chen et al. (2024). CNN-GRU-MHA. Applied Sciences.
23. Nectoux et al. (2012). PRONOSTIA. IEEE PHM.
24. Wang et al. (2020). XJTU-SY. IEEE Trans. Reliability.
25. Saxena et al. (2008). C-MAPSS. IEEE PHM.
26. LeCun (2022). Path towards autonomous machine intelligence.
27. Lei et al. (2018). Machinery health prognostics review. MSSP.
28. Zhao et al. (2019). Deep learning for machine health monitoring. MSSP.
29. Li et al. (2020). Deep transfer learning for fault diagnosis. Neurocomputing.
30. Raissi et al. (2019). PINNs. JCP.
31. Chen et al. (2018). Neural ODEs. NeurIPS.
32. Taleb (2007). The Black Swan. Random House.
33. Smith & Randall (2021). CWRU benchmark analysis. MSSP.
34. Zhang et al. (2022). TCN-Transformer. RESS.
35. V-JEPA 2 (Meta, 2025).
36. Shen et al. (2026). DCSSL. Scientific Reports.
37. Wang et al. (2025). RmGPT. IEEE IoT Journal.
38. OpenMAE (2025). ACM IMWUT.
39. Balestriero & LeCun (2025). LeJEPA/SIGReg. arXiv.
40. TTSNet (2025). Sensors.
41. MDSCT (2024). Heliyon.
42. NOMI Neural ODE (2024). Advanced Engineering Informatics.
43. Grey Swan Weather (2024). arXiv.
44. Grey Swan Factory (2026). arXiv.
45. Grey Swan Supply Chain (2025). IJPR.
46. Physics-Informed Augmentation for RUL (2024). RESS.
47. GAN + Inverse PINN (2025). PLOS ONE.
48. Digital Twin Bearing Diagnosis (2025). SHM.
49. Weak Supervision PHM Survey (2025). WIREs DMKD.
50. Physics-Informed RUL Review (2024). MSSP.

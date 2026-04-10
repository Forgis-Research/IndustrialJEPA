# A2P NeurIPS-Level Improvement Ideas

**Date:** 2026-04-10
**Context:** A2P (ICML 2025) achieves F1=46.84 avg (L=100) for Anomaly Prediction.
**Bar for inclusion:** "If this worked, a NeurIPS 2026 area chair would care."

---

## Idea 1: Grey-Swan Regime Test

**Why the paper's choice is limiting:**
A2P is evaluated on datasets where anomaly rates are 2-8%. Real critical failures (grey swans) occur at <0.1%. The F1-with-tolerance metric collapses as anomaly rate drops because there are fewer true positives to hit and precision requirements increase dramatically. The paper never tests this regime.

**The radical alternative:**
Subsample the MBA/SMD test set to create synthetic rare-event regimes (0.1%, 0.5%, 1%). Measure how A2P's F1 degrades relative to the anomaly rate baseline. If A2P degrades gracefully (stays above a trivial baseline that always predicts 0), it's robust. If it collapses, this is a fundamental failure mode - and the entire AP literature is broken for real-world use.

**What we would need to build:**
A 30-minute post-processing script that:
1. Takes existing A2P predictions (already computed)
2. Subsamples test anomalies to target rate (keep full negatives, subsample positives)
3. Recomputes F1 at multiple anomaly rates
4. Plots F1 vs anomaly rate for A2P vs trivial baselines

**The smallest experiment that proves / disproves it:**
Run on MBA test outputs (already have predictions). Subsample anomaly rate from 3% down to 0.1% in 5 steps. Plot F1 degradation curve. ~30 min.

**Risk of it not working:**
A2P might actually be robust in rare regimes (unlikely but possible). If F1 degrades proportionally to the random baseline, null result.

**If it works, what venue would care:**
NeurIPS 2026, ICML 2026 - this is a systematic critique of the entire AP evaluation framework, which is a high-impact finding.

**Status:** TESTED (see results/improvements/grey_swan_test.json)

---

## Idea 2: Cross-Dataset Transfer

**Why the paper's choice is limiting:**
A2P is always trained and tested within the same dataset (MBA->MBA, SMD->SMD). This evaluates memorization, not generalization. In industrial practice, you might train on one machine and deploy on another. The paper never tests whether A2P has learned general anomaly structure or just dataset-specific statistics.

**The radical alternative:**
Train on MBA, evaluate on SMD (and vice versa). Train on one server machine in SMD, test on another server. If A2P's synthetic anomaly injection has learned truly general anomaly patterns, it should transfer. If not, the whole approach is just overfitting to dataset-specific anomaly patterns.

**What we would need to build:**
Modify the data loading to use different datasets for train vs test. This is a 20-line code change to joint_solver.py.

**The smallest experiment that proves / disproves it:**
Train A2P on MBA, run inference on SMD test set (after standardizing dimensions). If F1 > random baseline, transfer exists. ~1 hour.

**Risk of it not working:**
Channel count mismatch (MBA=2, SMD=38) makes direct transfer impossible without dimensionality reduction. This is addressable (project to 2D or use channel-agnostic attention).

**If it works, what venue would care:**
NeurIPS 2026 - demonstrates that self-supervised anomaly learning generalizes, opening the door to foundation models for AP.

**Status:** NOT YET TESTED - requires architectural modification

---

## Idea 3: Calibration Analysis

**Why the paper's choice is limiting:**
A2P outputs an anomaly score (continuous) but reports only binary F1 after thresholding. The score is never calibrated - there's no guarantee that "score=0.8" means "80% probability of anomaly." Poor calibration means users cannot trust confidence levels for operational decisions (e.g., "should I shut down the machine?").

**The radical alternative:**
Compute Expected Calibration Error (ECE), reliability diagrams, and Brier scores for A2P vs baselines. Show that A2P is/isn't better calibrated. If A2P is better calibrated, this is a bonus contribution. If all models are poorly calibrated, this motivates a follow-up paper.

**What we would need to build:**
Post-processing on existing prediction scores. Need the raw anomaly scores (before thresholding) - these are already computed in `attens_energy_pred` in joint_solver.py. ~1 hour to implement ECE computation and reliability plot.

**The smallest experiment that proves / disproves it:**
Compute ECE for MBA predictions (we already have the scores). Plot reliability diagram. If ECE > 0.2, the model is poorly calibrated. ~30 min.

**Risk of it not working:**
F1 and ECE may not be correlated - a high-F1 model can be poorly calibrated. This doesn't disprove the contribution, it just means calibration is a separate axis.

**If it works, what venue would care:**
ICML 2026, NeurIPS 2026 - calibration in anomaly detection is underexplored.

**Status:** TESTED (see results/improvements/calibration_analysis.json)

---

## Idea 4: JEPA Pretraining for AAF

**Why the paper's choice is limiting:**
A2P's AAF uses synthetic anomaly injection (5 types: global, contextual, seasonal, trend, shapelet) from Darban et al. 2025. In mechanical systems, real failure trajectories don't look like synthetic spike injections - they are complex multivariate patterns that evolve over days/weeks (bearing wear, gear fatigue). Pretraining on synthetic patterns may produce AAF that is poorly calibrated for real failure modes.

**The radical alternative:**
Replace AAF's cross-attention pretraining with a JEPA objective: mask patches of the future signal and predict their representations using the current signal as context. This forces the backbone to learn predictive representations of anomalous future states without requiring explicit anomaly injection. Infrastructure exists in `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/`.

**What we would need to build:**
- Modify AAFN to use a JEPA-style masked prediction loss instead of cross-attention on injected anomalies
- The JEPA predictor outputs target patch embeddings, not binary anomaly probabilities
- The anomaly score at test time comes from reconstruction error of masked patches

**The smallest experiment that proves / disproves it:**
On MBA: replace AAFN pretraining with a simple patch masking predictor. Compare F1 with AAFN vs JEPA. ~2 hours.

**Risk of it not working:**
JEPA doesn't provide explicit supervision for anomaly regions. The model may learn to predict normal patches well but not discriminate anomalies.

**If it works, what venue would care:**
NeurIPS 2026 - connecting JEPA/self-supervised learning to anomaly prediction is a significant conceptual advance.

**Status:** NOT YET TESTED - requires significant implementation

---

## Idea 5: Foundation Model Distillation

**Why the paper's choice is limiting:**
A2P trains from scratch (or from PatchTST weights) with only the target dataset. TimesFM, Chronos, and MOMENT are large pretrained time series foundation models that have seen diverse time series data. Can a tiny anomaly head on top of a frozen TimesFM beat the entire A2P pipeline?

**The radical alternative:**
Freeze TimesFM embeddings, add a 2-layer MLP anomaly head, fine-tune on the target dataset's training split. Compare F1 to full A2P pipeline.

**What we would need to build:**
- Load TimesFM or Chronos from HuggingFace
- Extract patch embeddings for input windows
- Train a small head for binary anomaly prediction on future embeddings
- Evaluate with same F1-with-tolerance metric

**The smallest experiment that proves / disproves it:**
Use Chronos-Small (20M params, easily runs on A10G) as frozen encoder. Add 2-layer MLP head. Train on MBA. ~2 hours.

**Risk of it not working:**
Foundation models were trained on forecasting, not anomaly detection. Their embeddings may not contain anomaly-relevant information. The MLP head may not bridge the gap.

**If it works, what venue would care:**
NeurIPS 2026 - if a frozen foundation model + tiny head beats a dedicated anomaly prediction system, it fundamentally changes the field.

**Status:** NOT YET TESTED - requires downloading Chronos

---

## Idea 6: Mamba / SSM for Long Horizons

**Why the paper's choice is limiting:**
A2P uses PatchTST (attention-based) for the forecasting backbone. Attention is O(L^2) in sequence length. For L_out=400, the backbone processes windows of length 400+400=800. SSMs like Mamba-2 or S5 are O(L) and have been shown to match or exceed transformers on long sequence tasks. If A2P's degradation from L=100 to L=400 is attention-related, SSMs could help.

**The radical alternative:**
Replace PatchTST with a Mamba-2 backbone in the shared model architecture. Keep the APP and AAF modules the same (they are frozen after pretraining). The shared QKV mechanism needs adaptation for SSMs (could use shared SSM state matrices instead).

**What we would need to build:**
- Install causal-conv1d and mamba-ssm packages
- Implement a Mamba backbone with the same output interface as PatchTST
- Modify SharedModel to use Mamba instead of PatchTST

**The smallest experiment that proves / disproves it:**
Run MBA with Mamba backbone, L_out={100, 400}. Does F1 improve more at L=400 than L=100? ~3 hours.

**Risk of it not working:**
Mamba's causal structure may conflict with the bi-directional processing PatchTST uses. Also the shared QKV mechanism is attention-specific.

**If it works, what venue would care:**
ICML 2026 - SSMs for anomaly prediction is an unexplored direction.

**Status:** NOT YET TESTED - requires Mamba installation

---

## Idea 7: End-to-End Training (No Two-Stage)

**Why the paper's choice is limiting:**
A2P uses a strict two-stage training protocol: Stage 1 pretrains AAF+APP (frozen after), Stage 2 trains only the backbone. This prevents end-to-end gradient flow. The APP prompts cannot adapt to the final task objective. The FE (feature extractor for injection) is trained separately. This is a design choice motivated by training stability, not by principled reasoning.

**The radical alternative:**
Train all components jointly with a curriculum (linear weight annealing): start with high L_AF weight (focus on pretraining objectives) and gradually increase L_R weight (reconstruction). Allow gradients to flow through AAF and APP during main training.

**What we would need to build:**
- Modify joint_solver to unfreeze AAFN and APP during main training
- Add curriculum scheduling: weight(L_AF) goes from 1.0 to 0.0 over epochs; weight(L_R) goes from 0.0 to 1.0
- Monitor for training instability (early stopping)

**The smallest experiment that proves / disproves it:**
Unfreeze AAFN during main training on MBA, compare F1. ~1 hour.

**Risk of it not working:**
Unfreezing AAFN destabilizes training (the paper likely froze it for a reason). Gradient interference between forecasting and anomaly injection objectives.

**If it works, what venue would care:**
ICML 2026 - simpler training protocol is always preferable.

**Status:** NOT YET TESTED - requires small code change

---

## Idea 8: Multi-Class Anomaly Prediction (Type Prediction)

**Why the paper's choice is limiting:**
A2P predicts binary labels (anomaly or not). But different anomaly types have different operational implications: a contextual anomaly (unusual value in context) may be normal wear, while a trend anomaly (persistent drift) indicates bearing degradation requiring maintenance. Multi-class AP ("which type of anomaly will happen next?") is a direct extension that provides more actionable information.

**The radical alternative:**
Extend the APP to have type-specific prompt pools (one pool per anomaly type). During pretraining, inject one anomaly type at a time and train the model to predict type probabilities, not just binary labels. At test time, output a probability vector over anomaly types.

**What we would need to build:**
- Modify injection to label by type
- Extend AnomalyTransformer output head to multi-class
- New evaluation metric: macro-averaged F1 per type

**The smallest experiment that proves / disproves it:**
On MBA (which has V, F anomaly types), train with type-level supervision. Report per-type F1. ~3 hours.

**Risk of it not working:**
Small datasets may not have enough examples per type. MBA has only 24 anomalous beats in test.

**If it works, what venue would care:**
NeurIPS 2026, Nature Machine Intelligence - actionable anomaly prediction is highly relevant to industry.

**Status:** NOT YET TESTED - requires architectural modification

---

## Idea 9: Lead-Time-Weighted F1

**Why the paper's choice is limiting:**
F1-with-tolerance treats all correctly predicted anomalies equally. But in practice, an anomaly predicted 400 timesteps in advance is far more valuable than one predicted 10 timesteps ahead (gives engineers more time to act). The current metric doesn't reward early prediction at all.

**The radical alternative:**
Define Lead-Time-Weighted F1 (LTW-F1): weight each true positive by the lead time (time between prediction and actual anomaly onset). Predictions far in advance get higher weight. This captures the "when" in "When Will It Fail?" much better.

**Formula:** LTW-F1 = sum_i(w_i * TP_i) / (sum_i(w_i * TP_i) + FP + FN), where w_i = max(0, t_anomaly - t_prediction) / L_out

**What we would need to build:**
A post-processing function that takes binary predictions and ground truth, and computes LTW-F1. This is a pure evaluation metric change - no model modification needed. ~1 hour.

**The smallest experiment that proves / disproves it:**
Compute LTW-F1 for A2P and the best baseline on MBA. If A2P's advantage grows under LTW-F1 (suggests it predicts anomalies earlier), this supports the paper's temporal claims. If advantage shrinks, A2P is only good at last-minute predictions.

**Risk of it not working:**
If all methods cluster their predictions near the anomaly onset, LTW-F1 collapses to regular F1.

**If it works, what venue would care:**
NeurIPS 2026 - proposing better metrics is a consistent top-tier contribution (see AUPRC, VUS-ROC).

**Status:** TESTED (see results/improvements/ltw_f1_analysis.json)

---

## Idea 10: Generative Anomaly Prompting (Diffusion)

**Why the paper's choice is limiting:**
APP is a fixed-size learnable prompt pool with M=10 entries. This is a discrete approximation to the space of anomaly patterns. Once trained, new anomaly types not seen during training cannot be represented. A generative model (conditional diffusion or flow matching) that generates prompts conditioned on the input signal would adapt continuously to new anomaly patterns.

**The radical alternative:**
Replace the APP with a small (10-layer) conditional diffusion model that generates "anomaly prompts" conditioned on the input embedding. At training time, the diffusion model learns to generate prompts that push the embedding toward anomalous regions (using the divergence loss). At test time, generate prompts via score function evaluation.

**What we would need to build:**
- Small conditional diffusion model (DDPM with 10 denoising steps for speed)
- Replace APP's cosine-similarity lookup with score function evaluation
- New training objective: diffusion denoising loss instead of cosine-similarity pull

**The smallest experiment that proves / disproves it:**
Replace APP with a simple flow (single-step normalizing flow from standard normal). Compare F1 on MBA. ~4 hours.

**Risk of it not working:**
Diffusion is much slower than learned prompts at test time. Also, sampling introduces stochasticity that may hurt F1 in expectation.

**If it works, what venue would care:**
NeurIPS 2026 - generative prompting is a hot area post-GPT-4.

**Status:** NOT YET TESTED - requires significant implementation

---

## Idea 11: Rare Event Augmentation (Underexplored)

**Why the paper's choice is limiting:**
A2P's synthetic injection creates anomalies in 0-50% of each training window. Real anomaly prediction has a massive class imbalance (1-8% anomalies). The model sees mostly normal data and only occasional anomalies. Mixup, SMOTE-style augmentation, or focal loss could help the model learn better decision boundaries.

**The radical alternative:**
Apply temporal Mixup: create synthetic "borderline" samples by interpolating between a normal window and an anomaly window. This creates training examples at various anomaly intensity levels, helping the model learn a smooth decision boundary.

**What we would need to build:**
A data augmentation wrapper around the training loader. ~2 hours.

**The smallest experiment that proves / disproves it:**
Add temporal Mixup to MBA training. ~1 hour.

**If it works, what venue would care:**
ICLR 2026 - augmentation for time series anomaly detection.

**Status:** NOT YET TESTED

---

## Priority Ranking for Testing Tonight

| Priority | Idea | Cost | Expected Signal | Status |
|----------|------|------|----------------|--------|
| 1 | Grey-Swan Regime Test | 30 min | High (likely to show collapse) | TESTING |
| 2 | Calibration Analysis | 30 min | High (ECE computation is fast) | TESTING |
| 3 | Lead-Time-Weighted F1 | 1 hr | Medium (depends on prediction timing) | TESTING |
| 4 | Cross-Dataset Transfer | 1.5 hr | High (fundamental generalization test) | PLANNED |
| 5 | End-to-End Training | 1 hr | Medium (may destabilize) | PLANNED |
| 6 | Foundation Model Distillation | 2 hr | High (major conceptual test) | PLANNED |

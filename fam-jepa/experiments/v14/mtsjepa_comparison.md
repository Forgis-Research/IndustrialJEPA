# Trajectory JEPA vs. MTS-JEPA: Methodological Comparison

**Date**: 2026-04-14
**Purpose**: Precise architectural diff for NeurIPS submission positioning and V15 planning.
**Sources**: paper-replications/mts-jepa/ (CRITICAL_REVIEW.md, models.py, train_utils.py, RESULTS.md,
NEURIPS_REVIEW.md, EXPERIMENT_LOG.md); v12/v13 IndustrialJEPA experiments.

---

## 1. Side-by-Side Architectural Diff

| Design choice | Ours (Trajectory JEPA) | MTS-JEPA (He et al. 2026) |
|---|---|---|
| **Task** | Regression: predict RUL (remaining useful life) as a scalar, trained end-to-end or frozen + linear head | Binary classification: predict whether the next T_t-length window contains any anomaly |
| **Prediction target** | Future latent h_future: a single attention-pooled vector produced by bidirectional EMA encoder over the target segment | Two-resolution code distributions: fine Pi_hat^fine (P x K) over next window patches; coarse Pi_hat^coarse (1 x K) global summary of next window |
| **Masking / splitting strategy** | Temporal causal split: past segment of variable length feeds online encoder; future segment feeds EMA encoder; boundary is a hard causal cut with no overlap | Fixed context window X_t of length T_c followed by fixed target window X_{t+1} of length T_t; non-overlapping adjacent windows; no variable-length masking |
| **Prediction horizon** | Stochastic: k ~ U[5, 30] timesteps; predictor must learn multi-step extrapolation under uncertainty | Fixed: one window ahead (T_t = 100 for all datasets); no horizon randomization |
| **Encoder architecture** | Causal Transformer, d=256, L=2, 1.26M params; causal masking in self-attention; online branch only (unidirectional) | Channel-independent Transformer, d=256, L=6, ~5-8M params (paper); CNN residual tokenizer (2 blocks) creates patch-level tokens before attention; bidirectional attention within patches |
| **Tokenization** | Raw time-series segment fed directly; positional encoding over raw timesteps | Residual CNN (2 blocks, AdaptiveAvgPool1d) converts each patch (length L=20) to a single d-dim token before Transformer; patch-level positional embeddings |
| **Collapse prevention** | EMA momentum 0.99 on target encoder; no codebook; L1 loss in latent space provides direct gradient signal | Soft codebook (K=128 prototypes, temperature tau=0.1) + dual entropy regularization (minimize sample entropy, maximize batch entropy) + bidirectional alignment losses (L_emb, L_com); EMA on both encoder AND codebook (rho=0.996) |
| **Loss function** | L1(h_pred, h_EMA_target); total = mean over prediction horizon samples | L_pred = lambda_f(KL_fine + gamma * MSE_fine) + lambda_c * KL_coarse; L_code = lambda_emb * L_emb + lambda_com * L_com + lambda_ent_sample - lambda_ent_batch; L_rec = lambda_r(epoch) * MSE_reconstruction; 7+ loss weights, lambda_r annealed 0.5 -> 0.1 |
| **Reconstruction auxiliary** | None; pure predictive objective | MSE reconstruction of context patches from soft-quantized embeddings; annealed to reduce influence over training |
| **Multi-scale / multi-resolution** | Single resolution; variable prediction horizon k implicitly explores multiple timescales but no explicit coarse branch | Explicit dual resolution: fine view (P=5 patches of L=20) for local structure; coarse view (1 patch covering full window via downsampling) for global context; separate predictors for each resolution |
| **Cross-variate attention** | None; encoder operates on the full multivariate sequence jointly via causal Transformer (all channels fused from the start) | Channel-independent: encoder processes each variable separately (batch dimension B*V); variables never interact inside the encoder; variable-wise max-pooling at downstream evaluation stage (crude aggregation) |
| **Downstream evaluation** | Frozen encoder + linear regression head for RUL; also end-to-end fine-tuning; evaluated at 5 label fractions (5-100%); regression metric RMSE | Frozen encoder + frozen codebook -> (B, P*K) flattened code distributions via variable-wise max-pooling -> MLP classifier (2 hidden layers) -> binary cross-entropy; threshold tuned on validation to maximize F1; binary classification metric F1 + AUC |
| **Anomaly scoring via prediction error** | Not applied; prediction error not used as an anomaly score in any published run | Not directly; anomaly score comes from the downstream MLP classifier trained on code distributions, NOT from prediction error. The JEPA prediction error itself is not thresholded as an anomaly detector |
| **Normalization** | RevIN applied once on the full training split (z-score per-channel) | RevIN applied per window pair (context + target share the same RevIN stats computed from context); more local normalization |
| **Parameter count** | 1.26M (full), 0.58M (encoder only) | ~5-8M (paper config d=256, L=6); our reduced replication: 1.47M at d=128, L=3 |
| **Training stability** | Stable with EMA + L1; no codebook collapse reported across 5 seeds | Fragile at small batch size: KL divergence term causes early stopping at epoch 1-7 with batch=32; paper requires batch=128 for stable training; codebook perplexity ~K (uniform) in our replication = no discriminative code learning |
| **Datasets** | C-MAPSS FD001-FD004 (turbofan RUL), 100 train engines, up to 362 cycles | MSL, SMAP, SWaT, PSM (space telemetry, water treatment, server metrics); anomaly detection benchmarks repurposed for prediction |
| **Physical interpretability** | H.I. R^2=0.926 (latent trajectory predicts health index); degradation monotonicity validates representations | Codebook code activation patterns claimed as "degradation regime" fingerprints; interpretability is qualitative (Figure 3/4); no physical unit ground truth |

---

## 2. What Can WE Learn from MTS-JEPA?

### (a) Codebook Regularization

**What they do.** K=128 learnable prototypes on the unit hypersphere. Soft assignment via temperature-scaled cosine similarity. Dual entropy: minimize per-sample entropy (sharp assignments) while maximizing batch entropy (diverse code usage). Bidirectional alignment loss prevents encoder drift from the codebook manifold. Theory: non-collapse lower bound on Tr(Cov(z)).

**Would it help us?**

Argument for: Our encoder latent space is currently supervised only by L1 prediction error. A codebook could (a) prevent representation collapse in the frozen-encoder regime (v12: frozen RMSE=17.81 vs E2E=14.23, a 25% gap), (b) provide interpretable "degradation regime" codes aligned with C-MAPSS operating conditions, (c) enable anomaly scoring without labels by measuring code distribution shifts.

Argument against: Our replication showed codebook utilization was uniform (~100% of K codes active, perplexity ~K) at batch_size=32 - the codebook learned nothing discriminative. At our scale (1.26M params, small C-MAPSS dataset of ~18k samples), the codebook might not reach the critical mass needed for meaningful regime learning. The paper's ablation (no codebook -> near-collapse on MSL) was not reproduced under our constraints: on PSM, removing the codebook *improved* AUC (59.9 vs 51.0), contradicting the theoretical motivation.

**Feasibility**: Moderate (2-3 day implementation). The main engineering risk is codebook initialization and entropy weight tuning. Requires gradient accumulation to simulate larger batch sizes.

**Actionable verdict for V14**: Do NOT add for V14. Our current bottleneck is the frozen-encoder RMSE gap, not representation diversity. Adding a codebook with 7+ new hyperparameters without resolving the batch-size sensitivity issue would likely destabilize training. Flag for V15 as a targeted investigation: implement with gradient accumulation (effective batch=128) and measure codebook perplexity carefully before claiming regime learning.

---

### (b) Dual-Resolution Predictor

**What they do.** Fine predictor: 2-layer Transformer operating on P=5 patch code distributions, predicts next window's fine-grained code distribution patch-by-patch. Coarse predictor: learnable query token + cross-attention over fine codes -> single global prediction of next window's full-history summary. The coarse branch uses a learnable query that attends to fine patch tokens as keys and values, then passes through 2-layer Transformer. Effectively compresses P=5 local contexts into 1 global context.

**Would it help us?**

The physical interpretation is compelling: for RUL on vibration/turbofan data, local fault frequencies (fine scale) and macro-trend degradation (coarse scale) are both informative. Our current predictor is a 2-layer MLP that takes (h_past, horizon k), which collapses both scales into a single vector.

Concrete hypothesis: a dual-resolution predictor might improve the frozen-encoder result specifically (17.81 -> target 14.5), because the coarse branch would explicitly represent monotonic degradation trends while the fine branch represents local fluctuations.

Counter-argument: the CMAPSS sequences are short (median ~120 cycles), so with our variable horizon k ~ U[5, 30], the "coarse" scale only spans 1-6 windows. This is more useful for long, complex signals like SWaT (hours of telemetry). The gain might be marginal for CMAPSS.

**Feasibility**: Moderate (2 days). Can add a coarse head to the existing predictor without touching the encoder. The coarse branch needs a downsampled view of the input sequence, which requires a minor change to the data pipeline.

**Actionable verdict for V14**: Candidate for V14 Phase 2. Implement as an add-on to the current predictor (keep fine as default path, add coarse branch with 0.5x loss weight). Ablate on FD001 only with 3 seeds before expanding. Low architectural risk since we keep the existing encoder unchanged.

---

### (c) Anomaly Score from Prediction Error (Quickest to Test)

**What they do.** MTS-JEPA does NOT use prediction error directly as an anomaly score. Their downstream classifier is trained on the codebook code distributions, not on ||z_pred - z_EMA||. However, our CRITICAL_REVIEW noted this as a gap and opportunity: the natural anomaly score in a JEPA framework IS the prediction error.

**Would it help us?**

For C-MAPSS, the relevant quantity is not binary anomaly but continuous degradation. However, the prediction error ||h_pred - h_EMA|| as a function of cycle could be a proxy for degradation rate. Hypothesis: prediction error increases monotonically as the engine degrades, because the encoder's learned representation of "normal operation" becomes progressively harder to predict correctly near end of life.

This is directly testable on our V12/V13 checkpoints without any retraining. We have the trained encoder and EMA target encoder. We can compute per-cycle prediction errors on the test set and compute Spearman's rho between prediction error and true RUL (expected: negative correlation).

If rho < -0.3 (prediction error rises as RUL falls), this opens a zero-shot anomaly detection application: no fine-tuning required, just measure how surprised the predictor is.

**Feasibility**: Quick (half-day). Use existing V13 checkpoint. Write a 40-line evaluation script that computes per-cycle prediction error across all test engines and correlates with ground truth RUL.

**Actionable verdict for V14**: DO THIS FIRST in V14. It is the lowest-cost diagnostic with potentially high narrative value for the paper ("JEPA prediction error = degradation surprise signal, usable without labels"). If rho < -0.4, add a figure to the NeurIPS paper showing prediction error curves overlaid with RUL ground truth. This directly addresses MTS-JEPA's weakness (anomaly score not used) and adds a zero-shot PHM application.

---

### (d) Cross-Domain Pretraining

**What they do.** Table 2 / Table 7: pretrain on MSL + SMAP + PSM, evaluate frozen on SWaT (held out). Performance drops ~5-10% F1 but remains well above chance. This demonstrates representation transferability across anomaly domains.

**Would it help us?**

For NeurIPS, cross-domain pretraining on C-MAPSS is directly relevant: pretrain on FD001+FD002+FD003 turbofan data, freeze encoder, evaluate on FD004 (different degradation modes, more operating conditions). Our V13 results already show FD002 underperforms FD001 significantly (RMSE gap), suggesting domain shift is a real factor.

Stronger version: pretrain on C-MAPSS + FEMTO bearings + PRONOSTIA, evaluate frozen on C-MAPSS. This would directly demonstrate "one model, multiple machine types," which is the foundation model claim.

The computational cost is manageable: our encoder is 1.26M params, pretraining on CMAPSS takes ~20 minutes per run.

**Feasibility**: Moderate (1-2 days for FD001-3->FD004; 3-5 days for multi-dataset foundation model experiment). The FD001-3->FD004 experiment is feasible in V14. The multi-dataset version requires FEMTO/PRONOSTIA data preprocessing.

**Actionable verdict for V14**: Add FD001+FD002+FD003 -> FD004 cross-domain evaluation as Phase 3. This directly addresses the FD002/FD004 gap observed in V13 (is the gap about cross-domain or architecture?). For the full multi-dataset version, plan for V15.

---

## 3. What Can MTS-JEPA Learn from Us?

### 3.1 Representation Quality Diagnostic Suite

MTS-JEPA reports F1 and AUC. These scalar metrics are protocol-blind to whether the encoder is doing anything. Our diagnostic suite catches failure modes that a single metric misses:

**Shuffle test (input ordering).** Permute the temporal order of the input sequence and re-evaluate. If performance does not degrade, the encoder is not exploiting temporal structure - it is functioning as a bag-of-timesteps. MTS-JEPA never reports this. Given that their channel-independent encoder processes each variable's P=5 patches with learnable positional embeddings, a shuffle of the 5 patch positions would reveal whether temporal ordering matters. Given that anomalies often manifest as temporal pattern changes (not statistical marginal changes), this test has high diagnostic power.

**Tracking rho (per-sequence Spearman correlation).** For RUL prediction, the tracking rho measures within-sequence monotonicity. For anomaly prediction, the analogous metric is: within each anomaly run (from onset to end), does the model's anomaly probability increase monotonically? MTS-JEPA does not report this. Their Figure 3 shows a few representative windows but no systematic per-anomaly-event tracking curve.

**From-scratch ablation.** Compare frozen encoder (pretrained) vs. random encoder (same architecture, random weights) on the downstream task. MTS-JEPA does not report what a random-encoder MLP achieves on their PSM/MSL F1. In our MTS-JEPA replication, removing the codebook MODULE (which is close to a random-encoder baseline in terms of information content) improved PSM AUC from 51.0 to 59.9, which is an alarming result: the "w/o codebook module" variant outperforming the full model suggests the encoder representation itself is not contributing beyond what untrained features would provide.

**Feature-regressor baseline.** For C-MAPSS RUL, a ridge regression on (mean, std, last-value, per-channel slope over last 10 cycles, sequence length) achieves RMSE around 18-22 on FD001. Our model at 14.23 clearly beats this. MTS-JEPA never runs this baseline. The question of "what does a 10-line feature extractor achieve on PSM F1?" is unanswered. Given that PSM has known recurring failure patterns with strong autocorrelation, a sliding-window feature regressor might achieve F1 well above 40%.

**Length-vs-content diagnostic.** For RUL prediction, we explicitly test whether performance comes from sequence length alone (longer observation window = lower RUL). We verify that our encoder performance exceeds a length-only oracle. For MTS-JEPA's anomaly prediction task, the analogous test is: does a baseline that predicts "anomalous" whenever the context window has any unusual statistical property (high variance, spike count, etc.) achieve similar F1? This is not reported.

### 3.2 Label-Efficiency Evaluation

MTS-JEPA evaluates on a fixed 60/20/20 split of the test set for downstream classifier training. There is no experiment asking: how much labeled anomaly data does the SSL representation actually save? Our V13 experiments systematically evaluate 5%, 10%, 20%, 50%, 100% label fractions and show JEPA frozen encoder outperforms a from-scratch supervised model at 5% and 10% labels. This is the commercially relevant claim for industrial applications: "I have 2 anomaly events, can I build a detector?" MTS-JEPA makes no such claim despite targeting industrial use cases.

This is also where their evaluation has a structural weakness: with 60% of test data for downstream training, they have access to a large labeled set that narrows the gap between SSL and supervised baselines. A 5-20% label efficiency experiment would tell a more honest story.

### 3.3 Honest Feature-Regressor Baseline Comparison

As noted above: the paper does not compare against a simple non-neural baseline on F1. Reviewers at NeurIPS expect to see: "what does a logistic regression on window statistics (mean, std, max, autocorrelation) achieve?" If the answer is "comparable F1," then the SSL pretraining contributes nothing. The paper's inclusion of baselines like DeepSVDD and K-Means (which solve a different task) inflates the apparent advantage. The correct baseline structure is:
1. Trivial non-learning (predict all-normal -> F1=0)
2. Feature engineering + logistic regression (honest lower bound)
3. Supervised deep learning (upper bound on representation quality)
4. SSL methods (where MTS-JEPA sits)

MTS-JEPA paper goes from (1) to (4), skipping (2) and leaving (3) only partially addressed via detection-method baselines. Our paper should make all four explicit.

### 3.4 Run-to-Failure Structural Supervision Honesty

MTS-JEPA's "anomaly prediction" framing conflates genuine early warning with continuation detection. Our lead-time analysis (RESULTS.md) shows:
- PSM: 84.5% of "predicted anomalies" are continuation detections (context window also anomalous)
- MSL: 69.6% continuation
- SMAP: 89.1% continuation

Only 11-30% of anomalous target windows have fully-normal context windows (TRUE_PREDICTION cases). The reported F1 is dominated by continuation detection performance, not genuine early warning. This is analogous to the "within-sequence flatness" problem in RUL: a model that outputs a constant prediction gets good last-window RMSE without learning anything.

Our C-MAPSS evaluation is immune to this failure mode by construction: RUL ground truth is a continuous label that decreases monotonically, and our per-sequence tracking rho explicitly measures within-sequence discrimination. The degradation signal is never contaminated by "continuation" cases because each training sequence IS the run-to-failure trajectory.

For NeurIPS, we should make this structural advantage explicit: "Unlike anomaly prediction benchmarks where most 'predictions' are continuations of already-active anomalies, our RUL task requires genuine forward prediction of degradation state from a window where the machine is still nominally operational."

---

## 4. Concrete Recommendation for V15

V14 should close the immediate engineering gaps on C-MAPSS. V15 should be the positioning move relative to MTS-JEPA. Specific recommendations:

### V14 (immediate, within current session)

1. **Prediction-error anomaly score diagnostic** (half-day, highest priority). Use the V13 encoder checkpoint to compute per-cycle ||h_pred - h_EMA_target|| for all FD001 test engines. Compute Spearman rho between prediction error and ground truth RUL. If rho < -0.3, this adds a zero-shot PHM capability and a key paper figure at no extra training cost. This directly distinguishes us from MTS-JEPA which does not exploit its own prediction error.

2. **Dual-resolution predictor add-on** (2 days). Add a coarse predictor branch with 0.5x loss weight. The coarse branch compresses the past context via mean-pooling (not a learnable query initially - keep it simple) and predicts the future EMA target independently of the fine branch. Ablate on FD001 (3 seeds). Keep only if RMSE improves by > 0.5 over current 14.23 baseline.

3. **Cross-domain evaluation FD001+FD002+FD003 -> FD004** (1 day). Pretrain on union of FD001+FD002+FD003, evaluate frozen on FD004. Compare against training from scratch on FD004 only. This is the "foundation model" experiment for V14 and directly addresses our largest gap (FD004 underperformance in V13).

### V15 (planning horizon)

1. **Codebook regularization for regime learning** (2-3 days). Implement soft codebook with gradient accumulation (effective batch=128). Monitor codebook perplexity throughout training; only report "regime learning" if perplexity is well below K. Map learned codes to C-MAPSS operating conditions (FD002/FD004 have 6 conditions each). The target claim: each code corresponds to a degradation regime distinguishable in physical space.

2. **Multi-dataset pretraining** (3-5 days). Add FEMTO bearings (PRONOSTIA) to the pretraining corpus alongside C-MAPSS. Evaluate frozen encoder on both tasks. If cross-task transfer is positive, this is the headline "foundation model" result for NeurIPS.

3. **Honest early-warning evaluation protocol** (1 day, high narrative value). Borrow our lead-time analysis framework from the MTS-JEPA replication and apply it to any anomaly benchmark we evaluate on. Report TRUE_PREDICTION vs. CONTINUATION fraction explicitly. Offer this as a new evaluation standard that MTS-JEPA should adopt. Position as a methodological contribution in the paper.

4. **Label-efficiency baseline suite** (1 day). Add a feature-engineering baseline (ridge regression on 10 hand-crafted features) to every downstream evaluation. Report the margin between our SSL method and this baseline. A convincing NeurIPS story requires that our SSL features contribute signal BEYOND what a 10-line feature extractor provides, measured at 5% and 100% label fractions.

5. **Statistical rigor pass** (1 day). For all key results: 5 seeds reported throughout paper, paired t-test between Trajectory JEPA and best baseline, Cohen's d, 95% CI. MTS-JEPA was critiqued for missing this; we should not be.

---

## 5. Summary Assessment Table

| Dimension | Trajectory JEPA (ours) | MTS-JEPA (He et al.) | Winner |
|---|---|---|---|
| Task relevance to industry | High: direct RUL regression on gold-standard PHM benchmarks | Medium: anomaly prediction on IT/space telemetry | Ours |
| Architectural simplicity | High: 2-layer causal Transformer + MLP predictor | Low: 6-layer CI encoder + codebook + 2 predictors + decoder + 7 losses | Ours |
| Theoretical grounding | None published (opportunity) | Non-collapse and stability bounds | MTS-JEPA |
| Multi-scale modeling | No | Yes (fine + coarse dual predictor) | MTS-JEPA |
| Cross-variate attention | Yes (fused in encoder) | No (channel-independent) | Ours |
| Evaluation rigor | High (tracking rho, shuffle test, feature baseline, label efficiency) | Low (F1/AUC only, no feature baseline, no lead-time analysis) | Ours |
| Physical interpretability | H.I. R^2=0.926 quantitative | Qualitative code activation figures | Ours |
| Replication reproducibility | Full: 5-seed RMSE on FD001 confirmed | Partial: 15-40% F1 gap at reduced scale | Ours |
| Training stability | Stable (5 seeds, low variance) | Fragile (KL divergence, early stopping epoch 1-7 at small batch) | Ours |
| Foundation model generality | FD001 validated; cross-domain planned | Cross-domain table included | Tie (now), MTS-JEPA (published) |
| NeurIPS narrative strength | Strong if anomaly score diagnostic succeeds | Weak (evaluation protocol has fundamental issues flagged by our review) | Ours (with V14 additions) |

---

## Appendix A: MTS-JEPA Architecture Quick Reference

Encoder: CNNTokenizer (residual 2-block CNN, AdaptiveAvgPool1d -> d_model per patch) + 6-layer pre-norm Transformer (8 heads, d_ff=4*d_model, dropout=0.1) + 3-layer MLP projection head (d_model -> 64 -> 32 -> d_out). Channel-independent: processes B*V inputs in parallel.

Codebook: 128 prototypes, L2-normalized, cosine similarity, temperature tau=0.1, softmax soft assignment. Dual entropy: minimize E[H(p)] + maximize H(E[p]). Bidirectional alignment: stop-gradient MSE in both directions.

Fine predictor: 2-layer Transformer (4 heads, d_ff=128) operating on (B*V, P, K) code distributions.

Coarse predictor: learnable query token + cross-attention over fine codes as keys/values, then 2-layer Transformer. Output: (B*V, 1, K) single global prediction.

Reconstruction decoder: 3-layer MLP z -> patch_length, applied to soft-quantized context embeddings.

Total loss: L_pred + L_code + L_rec with 7+ weight hyperparameters. Reconstruction weight annealed 0.5 -> 0.1 linearly.

EMA rho=0.996 applied to both encoder and codebook after each optimizer step.

Downstream: flatten variable-wise max-pooled code distributions (B, P*K) -> MLP classifier (2 hidden layers, 256 and 128 dims).

## Appendix B: Our Replication Gap Summary

Configuration: d=128, K=64, L=3 (vs paper: d=256, K=128, L=6). Batch=32 (vs paper: 128). Effective parameter count ~1.47M (vs ~5-8M paper). KL scale reduced to 0.1 to prevent loss explosion.

Results on 5 seeds:
- MSL F1: 20.57 +/- 5.81 (paper: 33.58, gap: -38.7%)
- PSM F1: 50.68 +/- 3.64 (paper: 61.61, gap: -17.7%)
- SMAP F1: 24.10 +/- 2.22 (paper: 33.64, gap: -28.3%)

Primary failure mode: KL divergence instability causes best-epoch at 1-7, before meaningful representation learning. Codebook utilization uniform (~100%) -> no discriminative regime learning.

Critical finding from lead-time analysis: 70-89% of "correctly predicted" anomalous target windows have anomalous context windows (continuation detection, not true early warning). This is a property of the evaluation protocol independent of model quality.

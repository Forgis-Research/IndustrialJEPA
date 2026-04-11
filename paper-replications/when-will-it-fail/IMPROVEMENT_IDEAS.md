# A2P Improvement Ideas - NeurIPS-Level

**Generated:** 2026-04-10  
**Context:** A2P (Park et al., ICML 2025) frames Anomaly Prediction as a two-stage transformer pipeline (AAF pretrain + main training) with a synthetic anomaly injection scheme and learnable prompt pool.

The ideas below are ordered from most radical to most incremental. Each card follows the spec format from OVERNIGHT_PROMPT.md.

---

## Idea 1: JEPA-Pretrained AAF for Industrial Degradation

**Why the paper's choice is limiting:**
A2P's AAF is pretrained on five synthetic anomaly types (global spike, contextual, trend, shapelet, seasonal - following Darban et al. 2025). These are point anomalies and distributional shifts appropriate for ECG and server telemetry. Industrial mechanical degradation does NOT look like this: bearing wear produces broadband spectral elevation over thousands of cycles; gearbox faults develop as sidebands around mesh frequency. The synthetic injection assumption is the single weakest link in A2P for industrial transfer.

**The radical alternative:**
Replace the synthetic injection pretraining for AAF with a self-supervised JEPA objective trained on real run-to-failure trajectories. Specifically: treat the signal segment leading up to a known failure event as the "latent target" and train the AAF cross-attention to predict the latent representation of the anomaly-bearing future segment from the latent representation of the normal-looking prior segment - in representation space, not signal space. This is exactly the JEPA paradigm: predict missing context in latent space, not in pixel/signal space. The IndustrialJEPA infrastructure at `../../mechanical-jepa/` already has the encoder and JEPA training loop.

**What we would need to build:**
1. Adapter that converts IndustrialJEPA's FEMTO bearing embeddings to the A2P backbone dimension (d_model=256).
2. Modified `AAFN.py` where the MSE supervision signal is replaced by cosine-similarity loss in JEPA embedding space.
3. Mixed pretraining schedule: JEPA-AAFN on FEMTO/PRONOSTIA, then A2P main training on target dataset.

**The smallest experiment that proves or disproves it:**
Extract JEPA embeddings from bearing signals leading up to 5 FEMTO failure events. Train a linear probe to predict "time-to-failure bucket" from the embedding. If the JEPA representation is informative about upcoming failure (AUROC > 0.75), then using it as a supervision signal for AAF is justified. Estimated cost: 30 min on a GPU with existing JEPA checkpoints.

**Risk of it not working:**
JEPA representations may encode the nature of vibration patterns but not the temporal order of degradation (they are non-causal by default). Also, FEMTO -> MBA/WADI domain gap is large; the pretrained AAF may not transfer.

**Venue if it works:** NeurIPS 2026 main track - bridges JEPA SSL with industrial PHM and AP tasks in a principled way.

---

## Idea 2: Replace APP with Flow-Matching Anomaly Generator

**Why the paper's choice is limiting:**
The Anomaly Prompt Pool (APP) is a fixed pool of M=10 learnable (key, prompt) pairs. Prompt pools have well-known failure modes: (a) collapse - multiple keys converge to similar regions; (b) staleness - the pool is frozen after pretraining so cannot adapt to novel anomaly patterns at test time; (c) nearest-neighbour brittleness - if the input signal has an anomaly type not well-represented by any key, the top-N selection returns irrelevant prompts. The divergence loss L_D mitigates collapse but does not solve staleness or brittleness.

**The radical alternative:**
Replace the static APP with a conditional flow-matching model that generates anomaly prompts on-the-fly given the current input signal. The generator takes a normal signal embedding as condition and produces a sample from the anomaly distribution via continuous normalizing flow. This yields infinitely diverse anomaly prompts that adapt to the input signal's statistics at every forward pass. Conceptually: the APP learns "what anomalies look like in this signal's neighborhood" rather than "what anomalies look like in general".

**What we would need to build:**
1. A small conditional flow-matching network (e.g., 2-layer MLP with FiLM conditioning) that maps (noise, signal_embedding) -> anomaly_prompt.
2. Training objective: replace L_D with a flow-matching loss that maximizes likelihood of anomaly embeddings conditioned on reconstructed (normal) embeddings.
3. Inference: sample 3 prompts from the generator; replace the cosine-similarity selection step.

**The smallest experiment that proves or disproves it:**
On MBA: swap the APP (pool_size=10, frozen after pretrain) for a 2-layer conditional MLP that generates prompts from Gaussian noise conditioned on the FE [CLS] embedding. Measure whether train diversity of generated prompts (average pairwise cosine distance) is higher than APP and whether F1 improves. Estimated cost: 2-3 hours of coding + 1 GPU hour.

**Risk of it not working:**
Flow-matching requires a good conditional distribution - if the feature extractor embedding space does not cleanly separate normal from anomaly regimes, the generator will produce useless prompts. Also, end-to-end training of the generator with the backbone is more complex than the two-stage frozen-pretrain approach.

**Venue if it works:** ICLR 2027 or NeurIPS 2026 - generative anomaly prompting is a new paradigm.

---

## Idea 3: Grey-Swan Regime - Rare Anomaly Failure Mode

**Why the paper's choice is limiting:**
All four A2P benchmarks (MBA, Exathlon, SMD, WADI) have anomaly rates of roughly 5-15% of test timesteps. These are not rare events - they are common enough that standard thresholding (top anormly_ratio percentile of energy) reliably finds them. In real industrial and medical scenarios, catastrophic failures are rare: machine breakdowns may occur in 0.1-1% of operational hours, arrhythmia events in 0.05-0.5% of ECG recording. The paper never tests this regime.

**The radical alternative:**
Systematically evaluate A2P at anomaly rates {10%, 5%, 1%, 0.5%, 0.1%} by subsampling the MBA test set (or using the FEMTO bearing dataset which has <1% anomaly in the operating signal). Measure F1 collapse curve. Then propose a calibration-aware threshold that uses Platt scaling or temperature scaling on the anomaly score distribution to maintain calibrated P(anomaly | score) across anomaly rates.

**What we would need to build:**
1. A test-set subsampling script that artificially reduces anomaly prevalence while preserving temporal structure.
2. Platt scaling post-processor: train a logistic regression on held-out scores to map energy -> P(anomaly).
3. Evaluation loop over anomaly rates.

**The smallest experiment that proves or disproves it:**
Subsample MBA test set to 0.1% anomaly rate (remove 98% of anomaly segments at random, keeping normal data intact). Run A2P inference with original threshold (anormly_ratio=0.1%). Measure F1. **Estimated cost: 15 minutes once MBA results are available.** This is the cheapest high-value probe in the whole list.

**Risk of it not working:**
Subsampling changes test distribution statistics in ways that may confound results. A more principled version requires a truly rare-anomaly dataset.

**Venue if it works:** NeurIPS 2026 (applied ML track) - "When Will It Fail? Rarely" - a direct rebuttal/extension of the paper with industrial relevance.

---

## Idea 4: Lead-Time-Weighted F1 Metric

**Why the paper's choice is limiting:**
F1-with-tolerance gives equal credit for predicting an anomaly 1 timestep in advance vs 400 timesteps in advance. In real applications, EARLY prediction has exponentially more value: a warning 400 steps ahead allows corrective action; a warning 2 steps ahead is useless. The tolerance window in the paper's metric gives credit for "predicting within tol of the hit" but does NOT reward earliness. The paper's F1 actually peaks at larger L_out (58.89 at L_out=400 vs 46.84 at L_out=100) - but this is because longer windows have more anomaly overlap, not because predictions are genuinely earlier.

**The radical alternative:**
Define Lead-Time-Weighted F1 (LTF1): for each correctly predicted anomaly timestep, multiply the credit by `exp(-alpha * delta_t)` where `delta_t` is how close the prediction was to the current time. Correct early predictions (low delta_t because prediction is far in future) receive higher weight. This aligns the metric with operational value.

**What we would need to build:**
1. Modify `detection_adjustment()` to output per-prediction lead times.
2. Implement LTF1 as a new metric function.
3. Rerun A2P and baselines on MBA with LTF1 - expect that A2P's advantage over baselines may shrink or grow depending on how far ahead it actually predicts correctly.

**The smallest experiment that proves or disproves it:**
Compute LTF1 from existing A2P saved predictions (no retraining needed). Compare LTF1 ranking with F1-tolerance ranking across baselines. If the ranking changes substantially, this is evidence the existing metric is masking early-warning quality differences. Cost: 1-2 hours of scripting.

**Risk of it not working:**
LTF1 is harder to optimize for (non-differentiable, temporal weighting complicates gradient flow). May be better as a metric paper than a method paper.

**Venue if it works:** NeurIPS 2026 (evaluation and benchmarks) or a dedicated evaluation workshop.

---

## Idea 5: State-Space Model Backbone (Mamba-2) for Long Horizons

**Why the paper's choice is limiting:**
A2P uses a PatchTST/AnomalyTransformer transformer backbone. At L_out=400, the total sequence processed is L_in + L_out = 800 timesteps. Standard attention is O(n^2) in sequence length. More importantly, the AnomalyTransformer's anomaly scoring is based on local attention pattern discrepancy - this becomes noisy and loses signal at L_out=400. The paper's improvement from L_out=100 to L_out=400 (58.89 vs 46.84) is encouraging but likely limited by attention's difficulty with very long-range dependencies.

**The radical alternative:**
Replace the PatchTST forecasting branch and AnomalyTransformer AD branch with a shared Mamba-2 (or S5) SSM. SSMs are O(n) in sequence length, handle long horizons naturally via selective state updates, and have shown strong results on long-sequence time series (Gu & Dao 2024). The shared-backbone architecture maps cleanly: a single Mamba state space captures both forecasting-relevant dynamics and anomaly-detection-relevant deviation from the learned normal manifold.

**What we would need to build:**
1. `shared_model_mamba.py` replacing the transformer QKV sharing with Mamba-2 block sharing.
2. Compatible embedding layers (Mamba uses 1D convolution token mixing, not patch embedding).
3. Modify `joint_solver.py` to use the new backbone.

**The smallest experiment that proves or disproves it:**
Run A2P with Mamba backbone vs original on MBA at L_out=400 only (where long-horizon matters most). Measure F1 and training time. Cost: 1-2 days of implementation + 2 GPU hours.

**Risk of it not working:**
Mamba's selective state update may not capture the anomaly-detection attention discrepancy mechanism that A2P relies on for scoring. Also, the pretrained APP uses AnomalyTransformer's attention maps explicitly; replacing it requires rethinking the scoring function.

**Venue if it works:** ICML 2026 - SSM for AP is a clean method contribution.

---

## Idea 6: Multi-Class Anomaly Prediction

**Why the paper's choice is limiting:**
A2P produces a single binary label: "is this future timestep anomalous?" It conflates all anomaly types into one class. In practice, knowing WHAT KIND of failure is imminent is as valuable as knowing WHEN it will happen. A cardiac arrhythmia prediction system that can distinguish "supraventricular tachycardia" from "PVC" is clinically actionable; a binary AP system is not. Similarly, in industrial settings, knowing whether a bearing is heading toward inner-race fault vs outer-race fault determines which maintenance action to take.

**The radical alternative:**
Extend AP to a multi-class prediction task: given X_in, predict O in {0, 1, ..., K} for K anomaly types. The injection module already supports 5 types (global, contextual, trend, shapelet, seasonal) - extend this to K classes and train the AAF with a softmax cross-entropy loss instead of binary BCE. The APP keys can then be class-specific (K*N top-K selection instead of top-N).

**What we would need to build:**
1. Modify `AAFN.py` to output K-class probabilities instead of binary.
2. Modify `injection.py` to preserve anomaly type labels through the pipeline.
3. Modify threshold and F1 computation for multi-class.
4. A dataset with labeled anomaly types (MBA has two types: SVT and PVC - already multi-class ground truth available in MIT-BIH SVDB).

**The smallest experiment that proves or disproves it:**
On MBA with 2 anomaly classes (SVT vs PVC): train multi-class A2P. Measure macro-F1 and compare to binary F1. Cost: 3-4 hours coding + 1 GPU hour. MBA is the ideal dataset because it already has two labeled irregular rhythms.

**Risk of it not working:**
The 5 synthetic anomaly types may not align with the 2 real anomaly types in MBA. Transfer from synthetic-type supervision to real-type prediction requires the model to generalize anomaly concepts, not just anomaly locations.

**Venue if it works:** Nature Machine Intelligence or NeurIPS 2026 applied track.

---

## Idea 7: Foundation Model Zero-Shot Baseline

**Why the paper's choice is limiting:**
A2P's baselines are ALL task-specific trained models (PatchTST + AnomalyTransformer pairs). No zero-shot or foundation model baseline is included. This omission is significant given that TimesFM (Google), Chronos (Amazon), and MOMENT (CMU) were published before the ICML 2025 submission. If a zero-shot foundation model + a single linear anomaly head beats A2P's carefully pretrained two-stage pipeline, the contribution is severely undermined.

**The radical alternative:**
Zero-shot AP baseline: use TimesFM or Chronos to forecast X_hat_out from X_in (no fine-tuning), then run the AnomalyTransformer on X_hat_out to detect anomalies. Compare this zero-shot pipeline against A2P. If this "lazy baseline" scores within 5 F1 points of A2P, then A2P's two-stage training provides minimal benefit over what pretrained forecasting already gives.

**What we would need to build:**
1. A thin wrapper calling TimesFM/Chronos API for X_hat_out prediction on MBA/SMD.
2. Standard AnomalyTransformer inference on X_hat_out.
3. Evaluation with same F1-tolerance metric.

**The smallest experiment that proves or disproves it:**
Run TimesFM (0-shot, 512M params) + AT on MBA. Compare F1 against A2P's 67.55. Cost: 1-2 hours of scripting, no GPU training required. TimesFM is available via HuggingFace.

**Risk of it not working:**
TimesFM is trained on diverse TS data but not explicitly on anomaly-bearing signals; it may smooth out the anomalies in its forecasts, making the downstream AD harder than on A2P's anomaly-aware forecasts.

**Venue if it works:** This is more of a benchmark/negative result paper. Could be published as a discussion paper at a workshop or as part of a larger benchmark survey.

---

## Idea 8: End-to-End Training with Loss Curriculum

**Why the paper's choice is limiting:**
A2P uses a strict two-stage pipeline: Stage 1 pretrain AAF+APP -> freeze -> Stage 2 train backbone. Freeze-pretrain pipelines are brittle: the pretrained modules may not be optimally calibrated for the Stage 2 gradient flow, and the frozen AAF may produce sub-optimal anomaly probability weighting as the backbone evolves. There is no theoretical guarantee that freezing is optimal; it is a practical choice to avoid interference.

**The radical alternative:**
End-to-end training with a curriculum loss schedule. Start with `alpha * (L_AAF + L_D + L_F) + (1-alpha) * (L_R + L_AF)` where alpha decays from 1 to 0 over training. This avoids the hard boundary between pretraining and main training while still emphasizing the pretraining signal early. Optionally use low learning rate on AAF/APP in Stage 2 instead of freezing completely.

**What we would need to build:**
1. Modify `run.py` to run a single training loop instead of two sequential stages.
2. Add an `alpha_schedule` parameter that controls the curriculum decay.
3. Remove the `for p in solver.AAFN.parameters(): p.requires_grad = False` freeze step.

**The smallest experiment that proves or disproves it:**
On MBA: run A2P with curriculum (alpha: 1.0 -> 0.0 over 10 total epochs) vs original two-stage (5+5 epochs). Compare F1. Cost: 1 GPU hour, minimal code changes.

**Risk of it not working:**
The pretrain->freeze design may be intentional for stability: the L_D (divergence) loss operates on attention maps, which change rapidly during backbone training and could destabilize the prompt pool if trained jointly.

**Venue if it works:** Workshop paper or ablation contribution in a follow-up ICML submission.

---

## Idea 9: Cross-Dataset Transfer Test

**Why the paper's choice is limiting:**
A2P is always evaluated on the dataset it was trained on. This is a closed-world assumption: the model learns the statistics of MBA (ECG frequency, channel correlations) and is tested on MBA. This is fine for a method paper but limiting for real-world applicability. A system that can only predict anomalies in signals it has seen before has limited industrial value - equipment types change, new machines are deployed continuously.

**The radical alternative:**
Zero-shot cross-dataset transfer: train A2P on MBA, evaluate on SMD (38D server telemetry). The domain gap is large (ECG vs server CPU metrics) but the structural task is identical (predict future anomaly locations). If A2P has learned a general "anomaly relationship" via its synthetic injection scheme, it should transfer partially. If it has memorized MBA-specific spike patterns, it will fail completely.

**What we would need to build:**
1. No model changes - just evaluate the MBA-trained model on SMD test set (after dimensional projection: 2D -> 38D requires a learned linear adapter OR evaluate on a single SMD channel).
2. Alternative: few-shot transfer - train on MBA, fine-tune for 1-5 epochs on SMD train, evaluate on SMD test.

**The smallest experiment that proves or disproves it:**
Extract the MBA-trained backbone's anomaly scoring function. Apply it to SMD test data (using a linear projection layer). Measure F1. Compare against full SMD training (F1=36.29). Cost: 30-60 min once MBA model is trained.

**Risk of it not working:**
Dimensional mismatch (2 -> 38 channels) requires a projection layer that introduces significant domain shift. A fairer test is few-shot rather than zero-shot.

**Venue if it works:** NeurIPS 2026 - "Anomaly Prediction without Dataset-Specific Training" is a strong applied contribution.

---

## Idea 10: Calibration Analysis (ECE + Reliability Diagram)

**Why the paper's choice is limiting:**
A2P produces binary anomaly predictions with no associated probability estimate. The anomaly "score" is a raw energy value (MSE * attention_discrepancy) that is never calibrated. For any deployment scenario - clinical, industrial, financial - a decision-maker needs a probability estimate, not just a flag. "80% chance of failure in the next 5 minutes" is actionable; "anomaly: yes" is not.

**The radical alternative:**
Post-hoc calibration via Platt scaling: fit a logistic regression on a held-out calibration set to map energy scores to [0,1] probabilities. Evaluate with Expected Calibration Error (ECE) and reliability diagrams. Compare calibrated vs raw-threshold F1 and show that calibrated probabilities enable risk-adjusted decision making (e.g., deploy a cost-sensitive threshold that minimizes expected cost of missed alarms vs false alarms).

**What we would need to build:**
1. Split each dataset's test set 50/50 into calibration and evaluation.
2. Fit `LogisticRegression(C=1.0)` on calibration energy scores.
3. Compute ECE in 10 equal-width probability bins.
4. Plot reliability diagram.

**The smallest experiment that proves or disproves it:**
From existing A2P test outputs (energy scores + labels), compute ECE. If ECE > 0.10, calibration is broken and Platt scaling is needed. Cost: 30 min once energy scores are saved (add a `np.save('energy_scores.npy', ...)` line to `test_from_predicted()`).

**Risk of it not working:**
If A2P's energy scores already have reasonable calibration (ECE < 0.05), the contribution is limited. But given the architecture (no explicit calibration objective), poor calibration is the expected outcome.

**Venue if it works:** ML in Healthcare (NeurIPS workshop) or a combined metric+method NeurIPS paper.

---

## Idea 11: Anomaly Prediction as Regression (Time-to-Failure)

**Why the paper's choice is limiting:**
A2P predicts binary labels O in {0,1}^L_out. This formulation answers "IS timestep i anomalous?" but not "HOW MANY STEPS until the next anomaly?" The regression formulation - predicting time-to-failure (TTF) as a continuous value - is both harder and more informative. It is also the standard in PHM (Prognostics and Health Management) where RMSE on TTF is the canonical metric (as in our DCSSL replication). The binary classification AP formulation and the regression PHM formulation are solving the same underlying problem with different output heads.

**The radical alternative:**
Replace A2P's binary output head with a continuous TTF regression head. Train with Huber loss on the distance to the next anomaly event. At inference, threshold the TTF prediction (e.g., "flag if TTF < 50 steps") to recover binary predictions. Evaluate both TTF RMSE and binary F1.

**What we would need to build:**
1. New output head in `shared_model.py`: scalar TTF predictor per timestep.
2. Modified loss in `joint_solver.py`: L_TTF = Huber(predicted_TTF, true_TTF).
3. Evaluation: TTF RMSE + binary F1 from thresholded TTF.

**The smallest experiment that proves or disproves it:**
On MBA: extract true TTF from test labels (easy: distance to next 1 in the label sequence). Train a TTF head as a post-processing step on top of frozen A2P representations. Measure RMSE. If RMSE < trivial baseline (predict mean TTF), the representation is informative about lead time. Cost: 1-2 hours.

**Risk of it not working:**
MBA anomaly segments are contiguous runs of many timesteps (arrhythmia episodes last 1-30s). TTF within an anomaly segment is 0, which dominates the loss. Careful handling of in-anomaly timesteps needed.

**Venue if it works:** Bridges AP and PHM communities - strong fit for NeurIPS 2026 applied ML track.

---

## Idea 12: Spectral-Domain AAF for Mechanical Signals

**Why the paper's choice is limiting:**
AAF operates on raw time-domain signals. For mechanical systems (bearings, gearboxes), anomalies manifest primarily in the frequency domain: bearing defect frequencies, sideband modulation around mesh frequency, harmonic distortion. Pretraining AAF on time-domain synthetic injection misses the spectral signature of mechanical degradation entirely. An AAF that understands "elevated spectral energy at 3x ball-pass frequency" would be far more powerful for industrial AP than one that understands "time-domain spike".

**The radical alternative:**
Spectral AAF: compute the short-time Fourier transform (STFT) of X_in, inject synthetic anomalies in frequency domain (narrowband elevation, harmonic modulation), train the cross-attention between spectral maps of X_in and X_out. The backbone receives both time-domain and spectral inputs via a dual-domain embedding.

**What we would need to build:**
1. Modified `injection.py` with spectral injection types (narrowband, harmonic).
2. Dual-domain embedding layer: concatenate FFT magnitude with time-domain token embedding.
3. Spectral `AAFN.py`: cross-attention between spectral representations.

**The smallest experiment that proves or disproves it:**
On FEMTO bearing data: compare vanilla A2P F1 (if FEMTO AP labels can be extracted) vs spectral-AAF A2P. Cost: 2-3 days (significant new code). This is the most expensive idea in the list.

**Risk of it not working:**
STFT introduces time-frequency uncertainty tradeoff. Short windows lose temporal resolution; long windows lose frequency resolution. Also, standard TSAD benchmarks (MBA, WADI) are not mechanical signals - this improvement would only show on industrial/mechanical datasets.

**Venue if it works:** NeurIPS 2026 industrial ML track, or IEEE Transactions on Industrial Electronics.

---

## Summary: Cheapest Two Experiments

| Rank | Idea | Cost | Expected Signal |
|------|------|------|-----------------|
| 1 | Grey-Swan regime (Idea 3) | 15 min once MBA runs | Near-certain F1 collapse; quantify the curve |
| 2 | Calibration / ECE (Idea 10) | 30 min once energy scores saved | Near-certain poor calibration; ECE estimate |
| 3 | Foundation model zero-shot (Idea 7) | 1-2 hours, no training | May reveal A2P is not worth the complexity |
| 4 | Cross-dataset transfer (Idea 9) | 30 min once MBA trained | Reveals generalization limits |

**Recommendation:** Run Idea 3 and Idea 10 first. Both are post-hoc probes on A2P outputs (no retraining). They directly attack the two biggest practical gaps: rare anomalies and calibration for deployment.

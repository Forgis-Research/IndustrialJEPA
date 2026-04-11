# Overnight Autoresearch Prompt: Mechanical-JEPA V4

## Deep Understanding, Proper Metrics, RUL/Prognostics, HF Dataset, Simplification

Use this prompt with the ml-researcher agent for overnight autonomous research.

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project.

## Context

Working directory: `/home/sagemaker-user/IndustrialJEPA/mechanical-jepa`

### Where We Stand (after V2 + V3 runs)

**Architecture (V2 — current best):**
- Encoder: 4-layer transformer, embed_dim=512, patch_size=256, 16 patches per window
- Predictor: 4-layer transformer, predictor_dim=256, sinusoidal pos encoding
- Loss: L1 on L2-normalized predictions vs EMA targets + variance reg (λ=0.1)
- mask_ratio=0.625 (10 of 16 patches masked)
- Total: ~5M parameters

**Results:**
- CWRU 4-class linear probe: 82.1% ± 5.4% (3 seeds), best seed 89.7%
- CWRU MLP probe: 96.1%
- IMS binary transfer: +8.8% ± 0.7% over random init
- IMS 3-class transfer: +7.6% ± 1.8%
- Paderborn transfer @20kHz: +14.7% ± 0.8%
- Transfer efficiency: 142% (cross-domain beats self-pretrain!)
- wav2vec2 (94M params) = 77.2% vs our JEPA (5M) = 87.1%

**Predictor collapse was fixed** in V2 via high mask ratio + sinusoidal pos + L1 + var_reg.
The root cause: at mask_ratio=0.5, the predictor could predict the context-average as a
valid shortcut. At 0.625, only 6 of 16 patches are visible — the context average no longer
approximates the target average, forcing position-specific predictions.

**Open questions this run must answer:**
1. Do we actually NEED all five V2 fixes, or is mask_ratio alone sufficient?
2. What do SOTA methods use for collapse prevention? Is there something simpler?
3. What are the RIGHT success metrics for a vibration foundation model?
4. Can we do RUL prediction / prognostics, not just fault classification?
5. What does the HF Mechanical-Components dataset give us?
6. Can we get a failure probability distribution, not just a point prediction?

## FILES TO READ FIRST

Read ALL before running experiments:

1. `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` — ALL 35 experiments, every result
2. `autoresearch/mechanical_jepa/LESSONS_LEARNED.md` — critical insights including
   predictor collapse root cause, transfer boundary rule, spectral input findings
3. `autoresearch/LITERATURE_REVIEW.md` — JEPA SOTA review (I-JEPA, V-JEPA, TS-JEPA,
   MTS-JEPA, Brain-JEPA, C-JEPA) with collapse prevention table
4. `jepa-lit-review/jepa_sota_review.md` — C-JEPA + ThinkJEPA analysis
5. `mechanical-jepa/src/models/jepa_v2.py` — V2 model (the current best)
6. `mechanical-jepa/train_v2.py` — V2 training with all CLI flags
7. `mechanical-jepa/ims_transfer.py` — IMS transfer evaluation
8. `mechanical-jepa/paderborn_transfer.py` — Paderborn transfer (with resampling)
9. `mechanical-jepa/src/data/bearing_dataset.py` — data pipeline
10. `mechanical-jepa/quick_diagnose.py` — predictor collapse diagnostic

All paths relative to `/home/sagemaker-user/IndustrialJEPA/`.

## DATASET SCALE REMINDER

| Dataset | Windows | Classes | Sampling Rate |
|---------|---------|---------|---------------|
| CWRU | ~2,400 | 4 (H/OR/IR/Ball) | 12kHz |
| IMS | ~3,000 | continuous degradation | 20kHz |
| Paderborn | ~600 (3 bearings) | 3 (H/OR/IR) | 64kHz |
| HF Mechanical-Components | ~8K? (bearings+gearboxes) | multiple sources | varies |

This is TINY. Keep models small. Regularise heavily. Iterate fast at 30 epochs
before committing to 100-epoch runs.

## HF TOKEN & ENV

```bash
source /home/sagemaker-user/IndustrialJEPA/.env
# or manually: export HF_TOKEN="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"
```

### =======================================================================
### ROUND 1: DEEP LITERATURE REVIEW & SUCCESS METRICS (90 min)
### =======================================================================

This round is CRITICAL. Do not rush it. The experiments that follow depend on
having the right metrics and knowing what SOTA actually achieves.

#### 1A. Deep Literature Review — Collapse Prevention

Search the web thoroughly for:

1. **How does I-JEPA (Assran et al., CVPR 2023) prevent predictor collapse?**
   - Read the actual paper, not summaries
   - What is their predictor depth, dim, pos encoding?
   - Do they use variance regularization? VICReg? Something else?
   - What mask ratio do they use and why?

2. **SigReg / SigLIP / other collapse prevention**
   - Search: "SigReg self-supervised", "sigmoid loss self-supervised learning"
   - Search: "representation collapse prevention 2024 2025 2026"
   - Is there something simpler than our 5-part fix?
   - VICReg, Barlow Twins, BYOL, DINO — what do they each do differently?
   - Which approach is simplest while being effective?

3. **Is high mask ratio ALONE sufficient?**
   - In I-JEPA, what mask ratio do they use?
   - In MAE (He et al., 2022), they use 75% masking — is that enough to
     prevent collapse without other tricks?
   - Can we drop sinusoidal pos, L1, var_reg if mask ratio is high enough?

4. **Online learning / continual learning for SSL**
   - Can JEPA or similar models learn incrementally on new data?
   - Search: "continual self-supervised learning", "online SSL", "streaming SSL"
   - If JEPA can learn online, then few-shot transfer is less important because
     the model can just keep pretraining on new domain data
   - If NOT, then zero-shot and few-shot transfer metrics are critical

5. **RUL prediction with self-supervised features**
   - Search: "remaining useful life self-supervised", "RUL SSL", "prognostics SSL"
   - Search: "remaining useful life transformer", "RUL foundation model"
   - How do current SOTA methods approach RUL?
   - What metrics do they use? (RMSE, MAE, score function, timeliness)
   - C-MAPSS / PHM datasets — what are SOTA numbers?

6. **Failure probability / hazard functions**
   - Search: "failure probability estimation bearing", "hazard function machinery"
   - Search: "survival analysis predictive maintenance", "Weibull distribution RUL"
   - Gaussian processes for uncertainty in RUL prediction
   - Conformal prediction for RUL bounds
   - Key question: can JEPA embeddings be used as input to a survival model?

7. **Spectral energy as a prognostic indicator**
   - Search: "spectral energy prediction bearing fault", "vibration energy forecasting"
   - Is predicting future spectral energy a good proxy for fault progression?
   - What metric measures this best? RMSE? MAPE? Event-based F1?

#### 1B. Define Success Metrics (WRITE THESE DOWN)

After the literature review, define precisely:

**Fault Classification Metrics:**
- Macro F1-score (not accuracy — handles class imbalance)
- Per-class F1 (healthy, outer_race, inner_race, ball)
- We have been using accuracy — SWITCH to F1 for all future experiments
- Reproduce: what does SOTA get on CWRU 4-class? What split do they use?
  (Many papers report >99% but use random window splits, not bearing splits!)

**Transfer Metrics:**
- Zero-shot: freeze encoder, train linear probe on target domain
- Few-shot (N=20, 50, 100): same but with limited labeled data
- Transfer gain: F1(pretrained) - F1(random_init), must be positive in 3/3 seeds
- Compare to SOTA transfer results in literature

**RUL / Prognostics Metrics:**
- If predicting a continuous RUL value:
  - RMSE, MAE on RUL predictions
  - Score function (asymmetric — late predictions penalised more than early)
  - Timeliness: how early can we detect impending failure?
- If predicting a failure probability distribution:
  - Calibration (predicted prob vs observed frequency)
  - Sharpness (narrower distributions are better)
  - Coverage (do 90% prediction intervals contain 90% of actual failures?)
- If predicting event windows (will failure happen in next K hours?):
  - Precision, Recall, F1 for the "failure imminent" class
  - Early warning time: how far in advance is the alert?

**Spectral Energy Forecasting Metrics:**
- If predicting future spectral energy from current window:
  - RMSE between predicted and actual energy
  - But also: does the prediction trend upward before failures?
  - Event-based F1: if spectral energy crossing a threshold = alarm,
    does the forecast predict these alarm events?

**What matters MOST for a real industrial deployment?**
1. Low false alarm rate (high precision on "fault" class)
2. Early detection (high recall, with lead time)
3. Generalisation to unseen equipment (transfer)
4. Uncertainty quantification (know when you don't know)

Write all of this to a new file: `SUCCESS_METRICS.md` in `autoresearch/mechanical_jepa/`.
This file becomes the reference for all future experiments.

#### 1C. SOTA Reproduction Targets

Search for and record SOTA numbers on our exact datasets:

**CWRU (bearing split, not random window split!):**
- What methods and what accuracies?
- Typical range with proper bearing splits: 85-95% (not the inflated 99%)
- Note the EXACT split protocol used

**IMS:**
- RUL prediction SOTA
- Anomaly detection SOTA
- What features do people use? (RMS, kurtosis, spectral, learned?)

**Paderborn:**
- Cross-domain CWRU→Paderborn SOTA: published 88-93%
- We got +14.7% transfer gain — how does this compare?

**Record all this in SUCCESS_METRICS.md with citations.**

### =======================================================================
### ROUND 2: ARCHITECTURE SIMPLIFICATION (90 min)
### =======================================================================

The V2 fix uses 5 simultaneous changes. That's too many — we don't know which
are necessary. Simplify the architecture to the minimum that prevents collapse.

#### 2A. Targeted Ablation

Run at 30 epochs, seed 42, FAST iteration. Use F1-score from now on (not accuracy).

**Experiment 2A-1: Mask ratio 0.625 ONLY (remove all other V2 fixes)**
- mask_ratio=0.625, learnable pos encoding, MSE loss, no var_reg, predictor_depth=2
- Does high mask ratio alone prevent collapse?
- Run quick_diagnose.py immediately after

**Experiment 2A-2: Mask ratio 0.625 + sinusoidal (remove L1, var_reg)**
- If 2A-1 still collapses, add back sinusoidal
- Does mask ratio + sinusoidal suffice?

**Experiment 2A-3: Mask ratio 0.75 ONLY**
- Higher mask ratio might be enough on its own
- No other fixes

**Experiment 2A-4: V2 but with MSE instead of L1**
- Everything V2 except keep MSE loss
- Isolate whether L1 actually matters

**Experiment 2A-5: V2 but predictor_depth=2 instead of 4**
- Does predictor depth matter when mask ratio is high?

**Decision gate:** Find the MINIMAL config that prevents collapse (spread_ratio > 0.1)
AND achieves ≥80% accuracy. This becomes V3 — simpler is better.

#### 2B. If something simpler from literature works

If the literature review (Round 1) reveals a simpler collapse prevention mechanism
(e.g., just using high mask ratio like I-JEPA, or a simple regulariser):

**Experiment 2B-1: Implement the literature approach**
- Whatever I-JEPA actually uses
- Test at 30 epochs

**Experiment 2B-2: Compare to V2**
- Same evaluation protocol
- Is it simpler AND as effective?

### =======================================================================
### ROUND 3: SWITCH TO F1-SCORE & RE-EVALUATE (60 min)
### =======================================================================

All prior experiments used accuracy. F1 is the right metric for imbalanced classes.

#### 3A. Implement F1 Evaluation

Modify the evaluation code to report:
- Macro F1 (primary metric)
- Per-class F1 (breakdown)
- Confusion matrix
- Keep accuracy for backward compatibility

Add this to `train_v2.py` and any evaluation scripts.

#### 3B. Re-evaluate Best Models with F1

Take the existing best checkpoints and re-evaluate with F1:
- V2 best (seed 123, 89.7% accuracy) — what's the F1?
- Random init baseline — what's its F1?
- Any V3 simplified model from Round 2

This gives us proper baseline numbers going forward.

### =======================================================================
### ROUND 4: RUL PREDICTION & PROGNOSTICS (120 min)
### =======================================================================

This is new territory. IMS is a run-to-failure dataset — perfect for RUL.

#### 4A. Design the RUL Task

IMS Test 1 and Test 2 are run-to-failure experiments:
- Test 1: 2156 files over ~35 days, bearing 3+4 failed
- Test 2: 984 files over ~7 days, bearing 1 failed

**RUL definition:**
- For each file at time t, RUL = (time_of_failure - t)
- Normalise to [0, 1] where 1 = start of run, 0 = failure
- OR: use absolute hours remaining

**Approach 1: JEPA embeddings → RUL regression**
- Extract JEPA embeddings for each IMS file (using best CWRU-pretrained encoder)
- Train a small regression head (Linear or 2-layer MLP) to predict RUL
- Evaluate: RMSE, MAE, score function
- Compare: JEPA features vs hand-crafted (RMS, kurtosis, spectral) vs random init

**Approach 2: Spectral energy prediction**
- For each window, compute spectral energy (sum of FFT magnitudes)
- Track how this evolves over the run
- Can JEPA predict the NEXT window's spectral energy from the current one?
- This is a regression task: predict scalar from embedding
- The key: does spectral energy rise before failure? If yes, predicting it = early warning

**Approach 3: Health indicator from embeddings**
- Compute embedding distance from "healthy" cluster center
- As bearing degrades, embeddings should drift from healthy → failure region
- Track this distance over time — does it monotonically increase before failure?
- No labels needed! This is a ZERO-SHOT prognostic indicator

**Approach 4: Failure probability distribution**
- Train a survival model on JEPA embeddings
- Output: probability distribution over time-to-failure
- Options:
  a) Weibull distribution: p(t) = (k/λ)(t/λ)^(k-1) exp(-(t/λ)^k)
     - Predict (k, λ) from embeddings
  b) Gaussian process: nonparametric, gives uncertainty bounds
  c) Simple approach: discretise time into bins, predict P(failure in bin i | embedding)
     - This is just multi-class classification on time bins
  d) Conformal prediction: distribution-free uncertainty bounds on RUL

Start with Approach 3 (zero-shot, no training needed) and Approach 1 (simple regression).
Try Approach 4 if time permits.

#### 4B. Implement and Run

**Experiment 4B-1: Zero-shot health indicator (Approach 3)**
- Extract JEPA embeddings for all IMS files in temporal order
- Compute centroid of first 25% (healthy) embeddings
- Track cosine distance from centroid over time
- Plot distance vs time — does it correlate with degradation?
- Compute rank correlation (Spearman) between distance and time-to-failure
- Do this for BOTH IMS Test 1 and Test 2
- Compare: JEPA-pretrained vs random init encoder

**Experiment 4B-2: RUL regression (Approach 1)**
- Use JEPA embeddings as features
- Train linear regression (RUL from embedding)
- Split: first 60% of run for training, last 40% for test
- Evaluate: RMSE, MAE, and the asymmetric score:
  score = exp(-d/13) if d < 0 (early), exp(d/10) if d > 0 (late)
  where d = predicted_RUL - actual_RUL
- Compare: JEPA vs random init vs RMS/kurtosis baseline
- 3 seeds

**Experiment 4B-3: Spectral energy tracking (Approach 2)**
- Compute per-window spectral energy for IMS runs
- Does it increase before failure? Plot it
- Can JEPA embeddings predict next-window spectral energy? (regression)
- Early warning: how many hours before failure does energy spike?

**Experiment 4B-4: Failure probability (Approach 4, if time permits)**
- Discretise remaining life into bins: >75%, 50-75%, 25-50%, <25%
- Train classifier on JEPA embeddings to predict which bin
- This gives a rough probability distribution over remaining life
- Evaluate with calibration plot

#### 4C. Critical Analysis

- Does JEPA actually add value over simple signal statistics (RMS, kurtosis)?
- Is the zero-shot health indicator (embedding distance) monotonic?
- How early can we detect impending failure?
- Does the CWRU-pretrained encoder give better RUL features than random init?
  (This tests whether classification pretraining helps with prognostics)

### =======================================================================
### ROUND 5: HF MECHANICAL-COMPONENTS DATASET (120 min)
### =======================================================================

The dataset `Forgis/Mechanical-Components` is on HuggingFace with:
- `bearings/` (5 parquet files) — CWRU, MFPT, FEMTO, Mendeley, XJTU-SY sources
- `bearings/extra_cwru_mfpt.parquet`, `bearings/extra_ims.parquet`
- `gearboxes/` (4 parquet files)
- `source_metadata/` (1 parquet file)

#### 5A. Download and Explore

```python
import os
os.environ['HF_TOKEN'] = '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'
from datasets import load_dataset

# Load bearings config
ds_bearings = load_dataset("Forgis/Mechanical-Components", "bearings",
                           token=os.environ['HF_TOKEN'])
print(ds_bearings)
print(ds_bearings['train'].column_names)
print(ds_bearings['train'][0])  # first sample

# Load gearboxes
ds_gearboxes = load_dataset("Forgis/Mechanical-Components", "gearboxes",
                            token=os.environ['HF_TOKEN'])
print(ds_gearboxes)
```

#### 5B. Thorough Sanity Checks (MUST DO ALL)

1. **Schema**: What columns? Signal data format? Labels? Metadata?
2. **Signal quality**: Shape, dtype, NaN/inf, value ranges, plot a few signals
3. **Label distribution**: Samples per source, per fault type, per component
4. **Sampling rates**: What rates? Are they consistent within a source?
5. **Source breakdown**: How many samples from CWRU vs MFPT vs others?
6. **Overlap check**: Does it contain the EXACT same CWRU bearings we use locally?
   If yes, which ones? We must exclude overlapping data from transfer experiments.
7. **Gearbox data**: What's in it? Fault types? Signal format? Usable?
8. **Signal lengths**: Are all signals the same length? Or variable?

**Document everything in the experiment log.**

#### 5C. Experiments (only if sanity checks pass)

**Experiment 5C-1: New-source bearing pretraining**
- Identify bearing sources NOT in our local data (MFPT, FEMTO, Mendeley, XJTU-SY)
- Pretrain JEPA on these new-source bearings (unsupervised)
- Test: does pretraining on diverse sources improve transfer to CWRU/IMS/Paderborn?
- MUST resample all data to common sampling rate first
- 3 seeds

**Experiment 5C-2: Cross-component transfer (MOST IMPORTANT)**
- Pretrain JEPA on bearing data
- Evaluate on gearbox fault classification (zero-shot linear probe)
- This is THE test of generality: do vibration features transfer across component types?
- Compare: bearing-pretrained JEPA vs random init on gearbox task
- If this works, it's a strong result for a general vibration foundation model
- 3 seeds

**Experiment 5C-3: Gearbox self-pretrain (upper bound)**
- Pretrain JEPA on gearbox data
- Linear probe on gearbox fault classification
- Compare to Experiment 5C-2 (bearing→gearbox transfer)

**Experiment 5C-4: All-source pretraining**
- Pretrain on ALL available data (all bearings + all gearboxes)
- Test on held-out sources
- Does massive diversity help or hurt? (Recall: multi-source hurt in Exp 33)

#### 5D. Cross-Component Analysis

- Do bearing and gearbox embeddings cluster differently?
- t-SNE/PCA of embeddings from both — do fault types cluster across components?
- What vibration features are shared? What's different?
- Is the gearbox signal fundamentally different (gear mesh frequencies vs bearing defect frequencies)?

### =======================================================================
### ROUND 6: DEEPER UNDERSTANDING OF WHY COLLAPSE HAPPENS (60 min)
### =======================================================================

This is a RESEARCH question, not just an engineering fix. Understand the mechanism.

#### 6A. Visualise the Collapse

Create visualisations showing EXACTLY what happens:

1. **Prediction heatmap**: For a single input, show predicted embedding for each
   masked position. In collapsed model: all rows look identical. In fixed model:
   each row is different.

2. **Positional embedding analysis**: Extract and plot the learned positional
   embeddings from V1 (collapsed) vs V2 (fixed). In V1, do they collapse to
   similar vectors? What's the pairwise cosine similarity matrix?

3. **Loss landscape analysis**: Train V1 (collapsed) and V2 (fixed) for 30 epochs.
   Track per-position prediction loss. In V1, is the loss the same for all positions?
   In V2, do some positions have higher loss than others?

4. **Gradient flow through predictor**: In V1, do the positional embeddings receive
   meaningful gradients? Or do they receive near-zero gradients because the loss
   doesn't depend on position?

#### 6B. The Mathematical Argument

Write up a clear explanation (for the notebook):

- Why mask_ratio=0.5 allows collapse (context average ≈ global average for
  symmetric masking)
- Why mask_ratio>0.5 prevents it (context average ≠ target average when context
  is a small biased sample)
- What the loss landscape looks like in both cases
- Connection to information theory: at mask_ratio=0.5, predicting the mean has
  low information content but acceptable loss. At higher mask ratio, the mean
  prediction has higher loss, forcing content-specific predictions.

#### 6C. Does the Encoder ACTUALLY Learn Better with Fixed Predictor?

This is the key question. We know:
- V1 (collapsed predictor): 80.4% CWRU accuracy
- V2 (fixed predictor): 82.1% CWRU accuracy
- But IMS transfer: V1 +2.4% vs V2 +8.8%

**Experiment 6C-1: Embedding quality comparison**
- Extract embeddings from V1 and V2 for the SAME test data
- Compare: embedding variance, dimension utilisation, cluster separation
- t-SNE: do V2 embeddings show cleaner fault clusters?
- Linear separability: V2 should have higher F1 with linear probe

**Experiment 6C-2: What features does V2 learn that V1 doesn't?**
- Gradient-weighted analysis: which input features (time steps) drive the
  embedding most?
- Does V2 attend to different parts of the signal than V1?
- Attention map comparison between V1 and V2

### =======================================================================
### ROUND 7: ONLINE / CONTINUAL LEARNING EXPLORATION (60 min)
### =======================================================================

If JEPA can learn online (incrementally on new data), then the transfer story
becomes less important — you just keep pretraining on new-domain data.

#### 7A. Quick Experiment: Continual Pretraining

**Experiment 7A-1: CWRU pretrain → continue on IMS**
- Load best CWRU checkpoint
- Continue JEPA pretraining on IMS data for 20 more epochs (no labels)
- Test IMS performance
- Compare to: CWRU-only pretrain, IMS-only pretrain, random init

**Experiment 7A-2: Does catastrophic forgetting happen?**
- After continuing on IMS, test CWRU performance
- If CWRU accuracy drops significantly: catastrophic forgetting is a problem
- If CWRU accuracy holds: the model successfully learned both domains

**Experiment 7A-3: Sequential pretraining across all domains**
- Pretrain on CWRU (100ep) → continue on IMS (50ep) → continue on Paderborn (50ep)
- Test on all three datasets
- Is sequential pretraining better or worse than joint pretraining?

#### 7B. Implications

- If continual pretraining works: the deployment story is "pretrain on lab data,
  then keep learning on field data"
- If it doesn't: the deployment story is "collect diverse lab data, pretrain once,
  deploy frozen encoder + fine-tune probes"

### =======================================================================
### ROUND 8: COMPREHENSIVE JUPYTER NOTEBOOK (120 min)
### =======================================================================

Create a NEW notebook: `notebooks/04_v4_comprehensive_analysis.ipynb`

This notebook must be SELF-CONTAINED — someone reading it should understand
everything without reading any other file. It should include:

#### Section 1: Problem Statement & Architecture
- What is JEPA? (1 paragraph + diagram)
- What is predictor collapse? (1 paragraph + before/after figure)
- Our architecture (encoder + predictor + EMA target)

#### Section 2: Success Metrics
- Table of metrics (F1, transfer gain, RUL RMSE, etc.)
- SOTA comparison from literature
- Our targets vs achieved

#### Section 3: Predictor Collapse Deep Dive
- Visualisations from Round 6
- Mathematical explanation (why mask_ratio matters)
- Before/after diagnostics
- Minimal fix identification (from Round 2)

#### Section 4: Classification Results
- CWRU F1 results (all configs)
- Per-class breakdown
- Confusion matrices
- V1 vs V2 vs V3 (simplified) comparison

#### Section 5: Cross-Dataset Transfer
- Complete transfer matrix (from V3 run + new results)
- Frequency standardisation results
- Zero-shot vs few-shot curves
- Comparison to literature SOTA transfer numbers

#### Section 6: RUL Prediction & Prognostics
- Zero-shot health indicator (embedding distance over time)
- RUL regression results
- Spectral energy tracking
- Failure probability distribution (if completed)
- Comparison to baseline methods

#### Section 7: Cross-Component Transfer (HF Dataset)
- Bearing → gearbox transfer results
- t-SNE of cross-component embeddings
- What works, what doesn't

#### Section 8: Online / Continual Learning
- Does continual pretraining work?
- Catastrophic forgetting analysis
- Deployment implications

#### Section 9: Architecture Simplification
- What's the minimal architecture that works?
- Ablation table
- Comparison to literature approaches

#### Section 10: Conclusions & Next Steps
- Summary table of all key results
- What JEPA is good for (and not good for)
- Honest limitations
- Concrete next steps

**IMPORTANT:** Every figure should have:
- Clear title and axis labels
- Legend
- Error bars where applicable (3 seeds)
- Caption explaining what to look for

Every claim should have:
- Numbers with uncertainty
- Statistical significance (p-value or CI)
- Comparison to relevant baseline

Save all plots to `notebooks/plots/v4_*.png`.

### =======================================================================
### ROUND 9: DOCUMENTATION & COMMIT (30 min)
### =======================================================================

#### 9A. Update Experiment Log
- Log ALL new experiments (continue numbering from Exp 35)
- Full configs, results, verdicts, insights

#### 9B. Update Lessons Learned
- Add RUL/prognostics findings
- Add HF dataset findings
- Add simplification findings
- Update collapse prevention section with literature findings

#### 9C. Update SUCCESS_METRICS.md
- Fill in achieved values next to targets
- Mark which targets were met

#### 9D. Commit and Push
```bash
git add -A && git commit -m "Exp 36+: V4 overnight — RUL, HF dataset, simplification, metrics" && git push
```
Commit after each major round, not just at the end.

## GLOBAL RULES

### Experiment Discipline
- **ALWAYS use wandb** (never pass --no-wandb). Project: 'mechanical-jepa'
- **Log EVERY experiment** to EXPERIMENT_LOG.md
- **3+ seeds** for any claim. 1 seed for exploration only.
- **Use F1-score** (macro) as primary metric from now on, not accuracy
- **Run diagnostics** (quick_diagnose.py) after every training run
- **30 epochs first**, scale up only what works
- **Compare to V2 baseline** always

### Iterate Fast, Then Validate
- 30-epoch single-seed runs to see signals
- If signal is promising: 100-epoch 3-seed validation
- If signal is negative: move on immediately, don't try to salvage
- Budget: ~60% of time on exploration (30ep), ~40% on validation (100ep, 3 seeds)

### Simplicity Principle
- Fewer moving parts = more robust
- If mask_ratio alone prevents collapse, drop everything else
- Don't add complexity unless it provides measurable benefit
- The best architecture is the simplest one that achieves the target metrics

### Code Quality
- Add CLI flags for new features
- Keep backward compatibility
- F1 evaluation must be added cleanly, not hacked in

### Commit Protocol
- Commit after each round
- Message format: "Exp N: [brief finding]"
- Push after each round

### Self-Criticism
After every experiment, ask:
- Is this real or noise? (Check error bars across 3 seeds)
- Fair comparison? (Same seeds, epochs, eval protocol)
- Could RMS + logistic regression explain this? (Always check simple baseline)
- Does the F1-score tell a different story than accuracy?
- Would this survive peer review?

### Stopping Conditions
Stop and write final summary when:
1. All 9 rounds complete
2. You've been running 12+ hours
3. Irrecoverable error
4. Disk full (check: `df -h /home/sagemaker-user`)

**Priority if running out of time:**
- Round 1 (literature + metrics): MUST DO — everything else depends on this
- Round 2 (simplification): MUST DO — understand what's necessary
- Round 3 (F1 switch): MUST DO — need proper metrics
- Round 4 (RUL / prognostics): HIGH — new capability, compelling story
- Round 5 (HF dataset): HIGH — cross-component is the big test
- Round 6 (collapse understanding): MEDIUM — deepens the narrative
- Round 7 (online learning): MEDIUM — practical deployment angle
- Round 8 (notebook): MUST DO — all results must be presented clearly
- Round 9 (documentation): MUST DO — always commit clean

### Expected Outputs

After a successful overnight run:

1. **SUCCESS_METRICS.md** — clear targets with SOTA comparisons and achieved values
2. **Minimal architecture** — simplest config that prevents collapse
3. **F1 results** — all models re-evaluated with proper F1 metric
4. **RUL prediction results** — zero-shot health indicator + regression results
5. **HF cross-component results** — bearing→gearbox transfer (or clear negative result)
6. **Collapse visualisations** — heatmaps, embedding comparisons
7. **Comprehensive notebook** — 04_v4_comprehensive_analysis.ipynb with all results
8. **Updated experiment log** — Exp 36+ with all new results
9. **All experiments on wandb**
10. **Everything committed and pushed**

Good luck. Be thorough. Understand deeply before optimising. Use the whole night.
```

---

## Pre-Flight Checklist

- [ ] GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Disk: `df -h /home/sagemaker-user` (need >15GB free)
- [ ] WandB: `python -c "import wandb; print(wandb.api.api_key[:8])"`
- [ ] CWRU data: `ls mechanical-jepa/data/bearings/raw/cwru/`
- [ ] IMS data: `ls mechanical-jepa/data/bearings/raw/ims/`
- [ ] V2 checkpoint: `ls mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt`
- [ ] HF token: `grep HF_TOKEN /home/sagemaker-user/IndustrialJEPA/.env`
- [ ] scipy: `python -c "from scipy.io import loadmat; print('OK')"`
- [ ] sklearn: `python -c "from sklearn.metrics import f1_score; print('OK')"`
- [ ] Git clean: `git status`

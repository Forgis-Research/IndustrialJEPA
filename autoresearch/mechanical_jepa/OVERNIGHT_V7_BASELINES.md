# Overnight V7: Comprehensive Baseline Establishment

**Goal**: Establish thorough, validated baselines across 4 task families on the HF Mechanical-Components dataset. These baselines define what our self-supervised model must beat.

**Agent**: ml-researcher
**Estimated duration**: 6-10 hours
**Output**: `mechanical-jepa/baselines/` directory + Jupyter notebook + JSON results

---

## Context

We are building a self-supervised model (JEPA-family) for mechanical vibration time series. It will be trained on the HuggingFace dataset `Forgis/Mechanical-Components` (~12,000 samples, 16 sources, bearings + gearboxes).

The model's value proposition is that it learns representations useful for:
1. **Latent forecasting** — predicting future spectral energy density / energy envelope / meaningful proxy from chaotic accelerometer signals
2. **Forecasting** — predicting future values for more predictable signals (temperature, low-frequency sampled data)
3. **Anomaly detection/prediction** — detecting out-of-distribution or degrading behavior
4. **Anomaly classification** — classifying fault types (inner race, outer race, ball, gear crack, etc.)

Before training ANY model, we need to know what trivial/standard/SOTA baselines achieve on each task. This session establishes those.

---

## Dataset Access

```python
from datasets import load_dataset
TOKEN = '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'

bearings = load_dataset("Forgis/Mechanical-Components", "bearings", split="train", token=TOKEN)
gearboxes = load_dataset("Forgis/Mechanical-Components", "gearboxes", split="train", token=TOKEN)
sources = load_dataset("Forgis/Mechanical-Components", "source_metadata", split="train", token=TOKEN)
```

Key fields per sample:
- `signal`: vibration array (variable length, typically 64k-88k samples)
- `n_channels`: 2-8
- `source_id`: 'cwru', 'femto', 'xjtu_sy', 'ims', 'paderborn', 'mafaulda', 'mcc5_thu', etc.
- `health_state`: 'healthy' or 'faulty'
- `fault_type`: 'inner_race', 'outer_race', 'ball', 'gear_crack', etc.
- `fault_severity`: float or None
- `rpm`: int or None
- `episode_id`: str or None (for run-to-failure tracking)
- `episode_position`: float 0-1 (normalized degradation position, 0=start, 1=failure)
- `rul_percent`: float or None (remaining useful life %)
- `sampling_rate`: varies by source (5120-66667 Hz)

**Run-to-failure sources**: FEMTO (3569 samples), XJTU-SY (1370), IMS (1256) — these have `episode_id` and `episode_position`.

**V2 training spec**: All signals should be resampled to 12,800 Hz, windowed to 16,384 samples (1.28s). See `mechanical-datasets/V2_TRAINING_SPEC.md`.

---

## Phase 0: Deep Research & Brainstorming (1-2 hours)

Before writing ANY code, spend significant time thinking and researching. This is the most important phase.

### 0A: Literature Search
Search the web for SOTA on each task family applied to mechanical vibration data:
- What metrics does the community use for bearing/gear fault classification?
- What are SOTA anomaly detection methods for vibration? (Isolation Forest? Autoencoders? DAGMM? Deep SAD?)
- What forecasting methods work for chaotic sensor time series? (Not just ARIMA — think spectral domain, wavelet, reservoir computing)
- What does "latent forecasting" even mean for accelerometers? Research energy-based proxies: spectral energy density, envelope spectrum energy, Hilbert transform amplitude, RMS trending, kurtosis trending
- What baselines do papers like MOMENT, Chronos, TimesFM, MOIRAI use?

### 0B: What Proxies Are Meaningful?
For accelerometers (chaotic, broadband noise), raw signal forecasting is meaningless — you can't predict the next vibration sample. But you CAN predict:
- **Spectral energy density evolution**: How the frequency content changes over time (degradation signature)
- **Band energy trends**: Energy in defect-frequency bands (BPFO, BPFI, BSF) evolving over bearing life
- **RMS / kurtosis trending**: Statistical health indicators changing over hours/days
- **Envelope spectrum peaks**: Characteristic defect frequencies appearing/growing

Research which of these are:
1. Actually predictable (not just noise)
2. Meaningful for predictive maintenance
3. Computable from the HF dataset (which sources have temporal ordering?)

### 0C: What Tasks Make Sense?
Think critically about what tasks are well-posed given our dataset:
- Classification is well-posed (we have labels)
- Anomaly detection: which sources have enough healthy data for training? Do we need one-class or semi-supervised?
- Forecasting: only run-to-failure sources (FEMTO, XJTU-SY, IMS) have temporal ordering. Is ~6000 samples enough?
- Latent forecasting: what latent space? PCA? Autoencoder? The forecast target itself needs definition.

### 0D: Boil It Down
After brainstorming, write a 1-page summary of exactly which tasks and metrics you'll benchmark, and why. Save as `mechanical-jepa/baselines/TASK_DEFINITIONS.md`. Be ruthless — cut anything that doesn't have a clear, implementable definition.

---

## Phase 1: Data Loading & Task Setup (1 hour)

### 1A: Unified Data Loader
Create `mechanical-jepa/baselines/data_utils.py`:
- Load from HF, resample to 12,800 Hz, window to 16,384 samples
- For each task, define train/test splits:
  - **Classification**: source-disjoint (train on CWRU+FEMTO+MAFAULDA, test on Paderborn+Ottawa) AND within-source splits
  - **Anomaly detection**: train on healthy-only from selected sources, test on faulty
  - **Forecasting**: temporal split within run-to-failure episodes (first 80% train, last 20% test)
- Handle the diversity: different sampling rates, channel counts, signal lengths
- Store split definitions in JSON for reproducibility

### 1B: Feature Extraction Toolkit
Create `mechanical-jepa/baselines/features.py`:
- Time domain: RMS, peak, crest factor, kurtosis, skewness, shape factor, impulse factor, clearance factor
- Frequency domain: spectral centroid, spectral spread, spectral entropy, band energies (adaptive to sampling rate)
- Envelope analysis: Hilbert transform envelope, envelope spectrum peaks
- Time-frequency: STFT energy per band, wavelet packet energy
- Derived health indicators: RMS trend, kurtosis trend, spectral kurtosis

---

## Phase 2: Anomaly Classification Baselines (1.5 hours)

This is the task we've benchmarked most — extend it properly.

### 2A: Trivial Baselines
- **Majority class**: Always predict most common class
- **Random (stratified)**: Random predictions matching class distribution  
- **Nearest centroid**: Per-class mean of handcrafted features, classify by distance

### 2B: Standard ML Baselines
- **Logistic Regression** on handcrafted features (with StandardScaler)
- **Random Forest** (200 trees) on handcrafted features
- **XGBoost** on handcrafted features
- **SVM (RBF kernel)** on handcrafted features
- **1-NN with DTW distance** on raw signals (if computationally feasible on a subset)

### 2C: Deep Learning Baselines
- **1D CNN** (supervised, from scratch) — we have this, re-run on unified data
- **1D ResNet** (supervised) — standard architecture for time series classification
- **Transformer** (supervised) — we have this
- **InceptionTime** — SOTA for time series classification (ensemble of Inception modules)

### 2D: Transfer Baselines (most important)
For each method, evaluate:
- In-domain F1 (train and test on same source)
- Cross-domain F1 (train on source A, test on source B) — this is where self-supervised should shine
- Few-shot (N=1,5,10,20) on target domain

### Metrics
- Macro F1 (primary), accuracy, per-class F1
- 3 seeds minimum (42, 123, 456)
- All results to `mechanical-jepa/baselines/results/classification_baselines.json`

---

## Phase 3: Anomaly Detection Baselines (1.5 hours)

### 3A: Problem Setup
- **Training**: Only healthy samples (one-class learning)
- **Testing**: Mix of healthy + faulty (binary: is this anomalous?)
- **Sources for evaluation**: Use sources with clear healthy/faulty labels
- **Cross-domain**: Train healthy-only on source A, detect anomalies on source B

### 3B: Trivial Baselines
- **RMS threshold**: Flag if RMS > μ + 3σ of training healthy RMS
- **Kurtosis threshold**: Flag if kurtosis > μ + 3σ (impulse faults increase kurtosis)
- **Spectral energy threshold**: Flag if energy in defect-frequency bands exceeds threshold
- **Constant predictor**: Always predict healthy (gives you the class imbalance baseline)

### 3C: Standard Baselines
- **Isolation Forest** on handcrafted features
- **One-Class SVM** on handcrafted features  
- **LOF (Local Outlier Factor)** on handcrafted features
- **Mahalanobis distance** on handcrafted features
- **PCA reconstruction error** — fit PCA on healthy, flag high reconstruction error

### 3D: Deep Learning Baselines
- **Autoencoder** (1D CNN): Train to reconstruct healthy signals, flag high reconstruction error
- **VAE**: Same but with KL divergence as additional anomaly score
- **Deep SVDD**: One-class deep learning (map healthy to a hypersphere center)

### Metrics
- AUROC (primary), AUPRC, F1 at optimal threshold, F1 at fixed FPR=5%
- Separate: in-domain and cross-domain anomaly detection
- All results to `mechanical-jepa/baselines/results/anomaly_detection_baselines.json`

---

## Phase 4: Forecasting Baselines (1.5 hours)

### 4A: Define Forecasting Tasks Carefully

**Raw signal forecasting is NOT the goal** — accelerometer signals are chaotic/stochastic, predicting the next sample is meaningless. Instead, define meaningful forecasting targets:

1. **RMS trajectory forecasting**: Given RMS values for windows t=1..T, predict RMS at t=T+1..T+H
   - Use run-to-failure episodes (FEMTO, XJTU-SY, IMS)
   - This captures degradation trending
   
2. **Spectral energy density forecasting**: Given band energies for windows t=1..T, predict band energies at t=T+1..T+H
   - Defect frequencies grow as bearing degrades
   
3. **Health indicator forecasting**: Given a derived health indicator (HI) trajectory, predict future HI
   - HI could be: RMS, kurtosis, spectral kurtosis, or a learned representation

4. **RUL estimation** (if time): Given current window features, predict remaining useful life
   - Regression task, not forecasting per se

### 4B: Trivial Baselines
- **Last value**: Predict HI(t+1) = HI(t)
- **Moving average**: Predict HI(t+1) = mean(HI(t-k:t))
- **Linear extrapolation**: Fit line to last k points, extrapolate
- **Constant mean**: Predict mean of all training HI values

### 4C: Standard Baselines
- **ARIMA / auto-ARIMA** on HI trajectories
- **Exponential smoothing** (Holt-Winters)
- **Random Forest regression** on handcrafted features → next HI value
- **Ridge regression** on rolling features → next HI value

### 4D: Deep Learning Baselines (if time)
- **LSTM** on HI trajectories
- **1D CNN regressor** on raw windows → next-window HI

### Metrics
- RMSE, MAE, R², Spearman correlation (with true degradation)
- Horizon: 1-step, 5-step, 10-step ahead
- All results to `mechanical-jepa/baselines/results/forecasting_baselines.json`

---

## Phase 5: Latent Forecasting Baselines (1 hour)

This is the most speculative task — define it carefully.

### 5A: What is Latent Forecasting?
The idea: compress each time window into a latent vector, then forecast future latent vectors. The value of this over raw forecasting is that the latent space captures semantically meaningful structure (fault signatures, degradation state) rather than noise.

### 5B: Latent Space Definitions
Try multiple latent spaces:
1. **PCA latent space**: Fit PCA on handcrafted features, use first k components
2. **Autoencoder latent space**: Train AE on healthy signals, use bottleneck
3. **Random projection**: Random linear projection of handcrafted features (sanity check)

### 5C: Forecast in Latent Space
For each latent space:
- Given z(t-k:t), predict z(t+1)
- Methods: linear regression, MLP, LSTM
- Evaluate: MSE in latent space, AND decode back to interpretable metrics (does predicted latent vector correspond to correct health state?)

### 5D: Evaluate Meaningfulness
The key question: is latent forecasting better than just forecasting handcrafted features directly?
- Compare: forecast HI directly vs forecast latent → decode to HI
- If latent forecasting adds nothing, note that honestly

### Metrics
- Latent MSE, downstream classification accuracy from predicted latents, correlation with true HI
- All results to `mechanical-jepa/baselines/results/latent_forecasting_baselines.json`

---

## Phase 6: Jupyter Notebook Walkthrough (1 hour)

Create `mechanical-jepa/notebooks/07_baseline_establishment.ipynb`:

### Structure
1. **Dataset Overview**: Load HF data, show statistics per source, visualize sample waveforms
2. **Task 1 — Classification**: Show baseline results table, confusion matrices for best method, cross-domain transfer matrix
3. **Task 2 — Anomaly Detection**: ROC curves, detection rate vs false alarm rate, cross-domain comparison
4. **Task 3 — Forecasting**: RMS/HI trajectory plots with predictions overlaid, horizon comparison
5. **Task 4 — Latent Forecasting**: Latent space visualization (t-SNE/UMAP), forecast trajectories in latent space
6. **Summary Table**: All baselines across all tasks, what a self-supervised model needs to beat

### Key Figures (save as PDF+PNG in notebooks/plots/)
- `fig_baseline_classification_matrix.{pdf,png}` — Cross-source transfer matrix for classification
- `fig_baseline_anomaly_roc.{pdf,png}` — ROC curves for anomaly detection methods
- `fig_baseline_forecasting_trajectories.{pdf,png}` — Health indicator forecasting examples
- `fig_baseline_summary.{pdf,png}` — Summary comparison across all tasks

---

## Phase 7: Consolidate & Document (30 min)

Create `mechanical-jepa/baselines/BASELINE_RESULTS.md`:
- Complete results tables for all 4 task families
- Per-task: what's trivial, what's standard, what's hard to beat
- Clear "bar to clear" for each task — what does our self-supervised model need to achieve to be interesting?
- Honest assessment: which tasks are well-posed, which need more thought?

---

## Execution Notes

### Environment
- Running on SageMaker with GPU (CUDA available)
- Python environment has: torch, numpy, scipy, sklearn, xgboost, datasets (HuggingFace), matplotlib, seaborn
- Install anything else needed via pip

### Code Quality
- All results backed by JSON files (no "log only" numbers)
- 3 seeds minimum for all stochastic methods
- Clear train/test splits, no data leakage
- Reusable data loading code (we'll use the same splits when evaluating our model later)

### Paths
- Working directory: `/home/sagemaker-user/IndustrialJEPA/`
- New code: `mechanical-jepa/baselines/`
- Results: `mechanical-jepa/baselines/results/`
- Notebook: `mechanical-jepa/notebooks/07_baseline_establishment.ipynb`
- Figures: `mechanical-jepa/notebooks/plots/`

### HF Token
```python
TOKEN = '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'
```

### Priority
If running low on time, prioritize in this order:
1. Phase 0 (research) — most valuable, prevents wasted effort
2. Phase 2 (classification) — extends existing work
3. Phase 3 (anomaly detection) — new task, high value
4. Phase 4 (forecasting) — meaningful but harder to define
5. Phase 6 (notebook) — consolidation
6. Phase 5 (latent forecasting) — most speculative

### Iteration Protocol
After each phase, review results critically:
- Do the numbers make sense? (e.g., random forest shouldn't beat neural nets on easy tasks AND hard tasks)
- Are there data leakage red flags? (e.g., suspiciously high cross-domain performance)
- Is the evaluation fair? (same splits, same preprocessing)
- What's surprising? Investigate surprises — they're either bugs or insights.

Commit results after each major phase so progress is saved.

### Final Validation Loop (MANDATORY — do this AFTER all phases complete)

Once all baselines are implemented and results collected, loop over every single baseline and metric and ask two questions:

**Question 1: Is the metric meaningful & correctly implemented?**
For each metric in each task:
- Is this the metric the community actually uses for this task? (e.g., AUROC for anomaly detection, not accuracy on imbalanced data)
- Is the implementation correct? Check edge cases: class imbalance handling, macro vs micro averaging, threshold selection, normalization
- Does the metric capture what we care about? (e.g., F1 is meaningless if the test set is balanced — accuracy would be equivalent and simpler)
- Are there standard metrics we're MISSING that reviewers would expect? (e.g., EER for anomaly detection, MAPE for forecasting)
- For forecasting: are we evaluating at the right granularity? (per-episode vs pooled, per-horizon vs averaged)
- For anomaly detection: are we reporting at a fair operating point? (optimal threshold is cheating — use fixed FPR or cross-validated threshold)

If any metric fails these checks: fix it, re-run, and update the JSON.

**Question 2: Do we include the SOTA baseline?**
For each task, search the web for the current SOTA method and ask:
- **Classification**: What's the best published cross-domain bearing fault detection method? (DANN? CORAL? Some 2025 method?) Is there a method that uses self-supervised pretraining + fine-tuning that we should compare against? Are we missing a key baseline that ICML/NeurIPS reviewers will ask about?
- **Anomaly detection**: What's SOTA for unsupervised/one-class anomaly detection on vibration data? Is there a deep learning method that's become standard (e.g., DAGMM, Deep SAD, DROCC)? Are there domain-specific methods (envelope analysis + threshold) that practitioners actually use?
- **Forecasting**: What's SOTA for health indicator prognostics? Is there a transformer-based or foundation-model-based method we should include? What about physics-informed methods (Paris' law, degradation models)?
- **Latent forecasting**: Is anyone else doing this? If not, is our framing novel or just reinventing something with a different name?

For each task, if there's a clear SOTA that we're missing:
1. Implement it (or a faithful approximation)
2. Run it on the same splits with the same seeds
3. Add to the results JSON and notebook
4. If it can't be implemented in reasonable time, document WHY and note it as a known gap

Update `BASELINE_RESULTS.md` with a "Validation Notes" section documenting every check performed and its outcome. Be honest about gaps — it's better to know what we're missing than to discover it during review.

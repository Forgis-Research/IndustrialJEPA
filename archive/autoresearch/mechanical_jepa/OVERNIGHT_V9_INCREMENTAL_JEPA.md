# Overnight V9: Data-First JEPA — Ruthless Dataset Analysis, Clean Repo, Incremental Experiments

**Goal**: Fix the foundation before building higher. The JEPA pretraining instability (epoch 2 collapse) is a DATA problem, not a hyperparameter problem. Before any model changes, do a thorough dataset analysis, harmonize preprocessing, and be ruthlessly honest about which data sources are compatible. Then restructure the repo cleanly. Then run incremental experiments.

**Agent**: ml-researcher  
**Estimated duration**: 4 hours  
**Output**: Dataset analysis report, restructured repo, trained models, plots (PCA + t-SNE), RESULTS.md, **Quarto notebook (`notebooks/09_v9_data_first.qmd`) — REQUIRED DELIVERABLE**

**REQUIRED DELIVERABLES** (the session is NOT done without these):
1. `data/analysis/COMPATIBILITY_REPORT.md` — dataset compatibility verdict
2. `experiments/v9/RESULTS.md` — full results table with statistical tests
3. `experiments/v9/EXPERIMENT_LOG.md` — every experiment logged
4. `notebooks/09_v9_data_first.qmd` — complete Quarto walkthrough with embedded plots and hardcoded tables (engine: markdown, self-contained: true, follow format of `notebooks/08_rul_jepa.qmd`)
5. All plots saved to `analysis/plots/`

---

## CRITICAL: Read Before Starting

Read these files first to understand the current state:
- `mechanical-jepa/v8/RESULTS.md` — current best results
- `mechanical-jepa/v8/EXPERIMENT_LOG.md` — what was tried, including pretraining instability
- `mechanical-jepa/v8/data_pipeline.py` — current data loading (8 sources, resampling, windowing)
- `mechanical-jepa/v8/pretrain.py` — pretraining loop
- `mechanical-datasets/datasets_inventory.md` — dataset inventory
- `mechanical-datasets/dataset_temporal_analysis.md` — temporal properties

**V8 key results to beat**:
- JEPA+LSTM in-domain: RMSE=0.189 ±0.015
- Hybrid JEPA+HC in-domain: RMSE=0.055 ±0.004
- Contrastive cross-domain FEMTO→XJTU: RMSE=0.227 ±0.015
- Published SOTA (CNN-GRU-MHA, 2024): FEMTO nRMSE=0.044

**V8 key problem**: JEPA pretraining loss is minimal at epoch 2 of 100, then oscillates. 98% of training is wasted. The 8 pretraining sources are too heterogeneous — different machines, sampling rates, signal characteristics, amplitude ranges, frequency content. No amount of EMA tuning fixes bad data.

---

## Part A: Repo Restructuring (30 min)

### A.1: Create the clean structure

The repo should read like a foundation model pipeline: data → pretraining → downstream → evaluation.

Create this structure inside `mechanical-jepa/`:

```
mechanical-jepa/
├── README.md                          # Updated: clear project overview, quick start, results summary
│
├── data/
│   ├── analysis/                      # Phase B outputs: per-source analysis reports, plots
│   │   └── plots/                     # Spectral plots, amplitude distributions, etc.
│   ├── preprocessing.py               # Unified preprocessing pipeline (resample, normalize, window)
│   ├── loader.py                      # Data loading from HuggingFace
│   ├── registry.py                    # Source registry: metadata, native SR, channel map, domain
│   └── configs/                       # YAML configs: which sources to use, preprocessing params
│       ├── pretrain_all.yaml          # All 8 sources
│       ├── pretrain_bearings.yaml     # Bearing-only sources
│       └── pretrain_compatible.yaml   # Only sources that pass compatibility check (Phase B)
│
├── pretraining/
│   ├── jepa.py                        # JEPA architecture (from v8/jepa_v8.py)
│   ├── train.py                       # Pretraining loop (from v8/pretrain.py)
│   ├── masking.py                     # Masking strategies (random, contiguous block, multi-block)
│   └── configs/                       # Pretraining hyperparameters
│
├── downstream/
│   ├── rul/
│   │   ├── models.py                  # LSTM, TCN-Transformer, MLP heads
│   │   ├── train.py                   # RUL training loop
│   │   ├── baselines.py              # All RUL baselines (from v8/rul_baselines.py)
│   │   └── evaluate.py               # RUL metrics (RMSE, uncertainty, calibration)
│   ├── classification/                # Future: fault classification
│   └── anomaly/                       # Future: anomaly detection
│
├── analysis/
│   ├── embeddings.py                  # PCA, t-SNE, correlation analysis
│   ├── visualize.py                   # Plotting utilities
│   └── plots/                         # Generated figures
│
├── notebooks/                         # Quarto walkthroughs (already exists, keep as-is)
│
├── experiments/
│   ├── v8/                            # Archive of V8 results (move from v8/)
│   │   ├── RESULTS.md
│   │   ├── EXPERIMENT_LOG.md
│   │   └── results/                   # JSON results
│   └── v9/
│       ├── RESULTS.md
│       ├── EXPERIMENT_LOG.md
│       └── results/
│
├── configs/                           # Top-level experiment configs
│
└── archive/                           # Old scripts, v1-v7 artifacts
    └── (move loose scripts here: diagnose_jepa.py, freq_masking.py, etc.)
```

### A.2: Move and consolidate code

1. **Move** `v8/jepa_v8.py` → `pretraining/jepa.py` (rename class to JEPA, remove v8 references)
2. **Move** `v8/data_pipeline.py` → split into `data/loader.py` + `data/preprocessing.py`
3. **Move** `v8/pretrain.py` → `pretraining/train.py`
4. **Move** `v8/rul_baselines.py` → `downstream/rul/baselines.py`
5. **Move** `v8/rul_model.py` → `downstream/rul/models.py`
6. **Move** `v8/evaluate.py` → `downstream/rul/evaluate.py`
7. **Move** `v8/analyze.py` → `analysis/embeddings.py`
8. **Move** V8 results/logs to `experiments/v8/`
9. **Archive** loose scripts in `mechanical-jepa/` root → `archive/`
10. **Keep** `notebooks/` and `baselines/` as-is for now

### A.3: Create source registry

Create `data/registry.py` — a single source of truth for all datasets:

```python
SOURCES = {
    'cwru': {
        'native_sr': 12000,
        'channels': ['drive_end', 'fan_end'],
        'primary_channel': 0,
        'domain': 'bearing_fault_classification',
        'has_run_to_failure': False,
        'machine_type': 'motor_bearing',
        'load_conditions': [0, 1, 2, 3],  # HP
        'fault_types': ['normal', 'inner_race', 'outer_race', 'ball'],
        'n_samples': 40,
        'notes': 'Pre-seeded faults, no degradation trajectory',
    },
    'femto': {
        'native_sr': 25600,
        'channels': ['horizontal_accel', 'vertical_accel'],
        'primary_channel': 0,
        'domain': 'bearing_run_to_failure',
        'has_run_to_failure': True,
        'machine_type': 'ball_bearing',
        'load_conditions': ['1800rpm_4kN', '1650rpm_4.2kN', '1500rpm_5kN'],
        'n_episodes': 17,
        'snapshot_interval_s': 10,
        'snapshot_duration_s': 0.1,
        'notes': 'PHM 2012 challenge. 0.1s snapshots every 10s.',
    },
    # ... same for: xjtu_sy, ims, mfpt, paderborn, ottawa, mafaulda
}
```

### A.4: Update README.md

Rewrite `mechanical-jepa/README.md` with:
- One-paragraph project description
- Architecture diagram (text)
- Current best results table
- Quick start: how to run dataset analysis, pretraining, downstream
- Directory structure explanation
- Link to Quarto notebooks for detailed walkthroughs

---

## Part B: Ruthless Dataset Analysis (60 min)

This is the most important part of the entire overnight session. Create `data/analysis/dataset_compatibility.py` that produces a comprehensive report.

### B.1: Per-source signal characterization

For EACH of the 8 sources, compute and save:

**Time-domain statistics** (per window, then aggregate):
- Mean, std, min, max amplitude
- RMS energy
- Kurtosis (bearing damage indicator)
- Crest factor (peak / RMS)
- Skewness

**Frequency-domain statistics**:
- Power spectral density (PSD) — plot the average PSD per source
- Spectral centroid (center of mass of frequency spectrum)
- Spectral bandwidth
- Dominant frequency peaks (top 5)
- Energy distribution across bands: 0-500Hz, 500-2kHz, 2-5kHz, 5kHz+
- Nyquist frequency after resampling to 12.8kHz: 6.4kHz for all

**Distribution properties**:
- Amplitude histogram per source (overlay all 8 on one plot)
- QQ-plot against Gaussian
- Is the signal stationary within a window? (variance ratio of first half vs second half)

**Save**: One summary plot per source + one comparison plot with all 8 sources overlaid. Save to `data/analysis/plots/`.

### B.2: Cross-source compatibility analysis

**The key question**: Which sources are similar enough to pretrain together without the encoder being pulled in conflicting directions?

Compute pairwise similarity between all 8 sources:

1. **PSD divergence**: KL divergence between average PSDs (lower = more similar)
2. **Amplitude distribution overlap**: Wasserstein distance between amplitude distributions
3. **Spectral centroid difference**: |centroid_A - centroid_B| in Hz
4. **Kurtosis difference**: |kurtosis_A - kurtosis_B|
5. **RMS energy ratio**: max(rms_A, rms_B) / min(rms_A, rms_B) — should be close to 1

Produce an **8×8 compatibility matrix** heatmap for each metric. Save to `data/analysis/plots/compatibility_matrix.png`.

### B.3: Preprocessing audit

For each source, check:
1. **After resampling to 12.8kHz**: Is there aliasing? (energy above Nyquist before resampling)
2. **After instance normalization**: Do the distributions look similar across sources?
3. **After windowing to 1024 samples**: Any edge effects? Truncation artifacts?

Does instance normalization (zero-mean, unit-std per window) actually make the sources compatible? Or are there deeper structural differences (different spectral shapes, different kurtosis distributions) that normalization doesn't fix?

### B.4: Compatibility verdict

Based on B.1-B.3, produce a **compatibility report** in `data/analysis/COMPATIBILITY_REPORT.md`:

```markdown
# Dataset Compatibility Report

## Summary Table
| Source | PSD shape | Amplitude dist | Spectral centroid | Kurtosis | Compatible with bearing RUL? |
|--------|-----------|---------------|-------------------|----------|------------------------------|
| CWRU   | ...       | ...           | ...               | ...      | YES/NO/MARGINAL              |
| FEMTO  | ...       | ...           | ...               | ...      | YES (primary target)         |
| ...    | ...       | ...           | ...               | ...      | ...                          |

## Recommended Source Groups
- **Group A (bearing degradation)**: FEMTO, XJTU-SY, IMS — share run-to-failure structure
- **Group B (bearing faults)**: CWRU, MFPT, Ottawa — static fault classification
- **Group C (industrial)**: Paderborn, MAFAULDA — different machine types

## Recommendation for pretraining
[Based on analysis: which groups to combine, which to exclude, what preprocessing needed]
```

### B.5: Smart preprocessing pipeline

Based on the analysis, implement preprocessing in `data/preprocessing.py` that goes beyond simple instance normalization:

Possible steps (implement what the analysis shows is needed):
1. **Bandpass filter** to a common frequency range (if some sources have energy in bands others don't)
2. **Spectral whitening** (flatten the average PSD per source, so all sources have similar spectral shape)
3. **Amplitude standardization** by source (not just per-window, but per-source statistics)
4. **Kurtosis-based filtering** (remove windows that are pure noise or have artifacts)

**IMPORTANT**: Whatever preprocessing you do, verify it doesn't destroy the RUL-relevant information (spectral centroid shift, increasing kurtosis during degradation). Before and after preprocessing, compute correlation of features with RUL% for FEMTO and XJTU-SY episodes.

---

## Part C: Rebuild Pretraining on Clean Data (45 min)

### C.1: Pretrain on compatible sources only

Using the compatibility verdict from Part B, create `data/configs/pretrain_compatible.yaml` listing only the sources that passed the compatibility check.

Retrain JEPA with:
- Only compatible sources
- Improved preprocessing from B.5
- Same architecture as V8 (no model changes yet — isolate the data improvement)
- 100 epochs, fixed EMA momentum 0.996

**Key evaluation**: Does the loss curve stabilize? Does the best checkpoint shift beyond epoch 2? If YES → the problem was data heterogeneity. If NO → there may be additional issues.

### C.2: Compare pretraining source groups

Run 3 pretraining experiments:
1. **All 8 sources** (V8 reproduction with new preprocessing)
2. **Compatible sources only** (from B.4)
3. **Bearing RUL sources only** (FEMTO + XJTU-SY + IMS)

For each: report loss curve shape, best epoch, best val loss, downstream JEPA+LSTM RMSE.

### C.3: Expand to all available episodes

Currently using only 7 of 15 XJTU-SY episodes. Fix this:
- Load ALL 15 XJTU-SY bearings (5 per condition × 3 conditions)
- Keep all 17 FEMTO bearings
- Total: 32 episodes (17 FEMTO + 15 XJTU-SY)
- 75%/25% episode-based split stratified by source
- This is the evaluation dataset for all downstream experiments

---

## Part D: TCN-Transformer — The Right Temporal Model (60 min)

### D.1: TCN-Transformer with handcrafted features (supervised SOTA baseline)

Implement the published SOTA architecture as a supervised baseline.

**Architecture** (following Sensors 2025 / Heliyon 2024):

```
Input: full episode sequence of features per snapshot
         ↓
    ┌────┴────┐
    ↓         ↓
   TCN     Transformer
 (local)   (global)
    ↓         ↓
    └────┬────┘  ← concatenation
         ↓
   MLP → RUL scalar (per timestep)
```

**TCN branch**: 4 layers, kernel 3, dilations {1,2,4,8}, hidden 64, weight norm + ReLU + dropout 0.2, residual connections.

**Transformer branch**: 2 encoder layers, 4 heads, d_model=64, FFN 256, dropout 0.1. **Positional encoding = elapsed time (continuous, in hours)**, not integer position.

**Fusion**: Concat → Linear(128,64) → ReLU → Linear(64,1) → RUL

**Input**: Handcrafted features (18-dim) per snapshot. Causal attention mask (can only attend to past).

**Training**: End-to-end supervised, AdamW lr=1e-3, 100 epochs, early stopping. 5 seeds.

### D.2: JEPA + TCN-Transformer (replace LSTM)

Feed frozen JEPA embeddings (from best encoder in Part C) into TCN-Transformer instead of LSTM. Compare to JEPA+LSTM.

### D.3: JEPA + Deviation-from-Baseline features

Explicitly encode "how different is the current state from the healthy baseline":

```python
z_baseline = mean(z_1, ..., z_K)       # healthy baseline (first K=10 snapshots)
z_deviation = z_t - z_baseline          # deviation vector (256-dim)
deviation_norm = ||z_deviation||_2      # scalar distance from healthy

input_t = [z_t, z_deviation, deviation_norm, elapsed_time, delta_t]  # 515 dim
```

Feed into TCN-Transformer. This addresses the elephant: during the long healthy phase, `deviation ≈ 0` tells the model "still healthy." When degradation starts, deviation grows and shows WHERE in embedding space the bearing is drifting.

### D.4: Hybrid — JEPA + HC + deviation (only if D.3 helps)

```python
input_t = [z_t, z_deviation, handcrafted_features, elapsed_time, delta_t]  # 532 dim
```

---

## Part E: JEPA Pretraining Improvements (30 min)

Only attempt these AFTER Part C confirms the data was the main problem.

### E.1: Contiguous block masking

Replace random patch masking with contiguous block masking. Mask a single contiguous block of 10 patches, block start position randomized per training step.

```python
block_start = random.randint(0, 16 - 10)
mask_indices = list(range(block_start, block_start + 10))
```

Retrain encoder, compare pretraining loss + downstream RMSE to C.1.

### E.2: Dual-channel raw + FFT

Input: (B, 2, 1024) — channel 0 = raw waveform, channel 1 = magnitude FFT. Patch projection maps 128 dims (64 raw + 64 FFT) → 256.

**Key question**: Does the dual-channel encoder learn spectral centroid-aligned features (Spearman correlation with RUL >0.3 vs current 0.144)?

---

## Part F: Probabilistic RUL Output (20 min)

### F.1: Heteroscedastic output (zero overhead)

Final MLP outputs 2 values: mean μ and log-variance log(σ²). Loss = Gaussian NLL:

```python
def nll_loss(mu, log_var, target):
    var = torch.exp(log_var)
    return 0.5 * (log_var + (target - mu)**2 / var).mean()
```

User can compute P(RUL < threshold) from the Gaussian CDF. End users pick their own risk threshold.

**Evaluate**: PICP at 90%, MPIW, calibration plot. Uncertainty should be higher during healthy phase (ambiguous) and near failure transition.

### F.2: Ensemble uncertainty

5-seed runs already form an ensemble. Report cross-seed mean and std. Compare to HNN uncertainty.

---

## Part G: Visualization and Reporting (30 min)

### G.1: Comprehensive plots

For the best model, produce (save all to `analysis/plots/`):

1. **PCA**: PC1 vs PC2, color=RUL%, markers=dataset source
2. **t-SNE**: Same, perplexity=30
3. **PCA colored by source**: Do embeddings cluster by source or by health?
4. **t-SNE colored by source**: Same
5. **Correlation heatmap**: Top embedding dims vs RUL, spectral centroid, kurtosis, RMS
6. **Uncertainty calibration**: Predicted σ vs actual error
7. **Degradation trajectories**: Embedding PC1 over episode time for 5 test episodes
8. **Deviation-from-baseline**: ||z_t - z_baseline|| over time for 5 episodes
9. **Side-by-side encoder comparison** (t-SNE): random / V8 JEPA / V9 JEPA / contrastive

### G.2: Results consolidation

Write `experiments/v9/RESULTS.md` with full results table:
```
| Exp | Change | RMSE ±std | vs V8 JEPA+LSTM | vs Time-Only | p-value | Notes |
```

### G.3: Quarto notebook

Create `notebooks/09_v9_data_first.qmd` (static document with embedded plots and tables). Sections:
1. The data problem: why V8 pretraining collapsed at epoch 2
2. Dataset compatibility analysis (with plots from Part B)
3. Which sources survived and why
4. Improved pretraining results
5. Full-history temporal model (TCN-Transformer)
6. Deviation-from-baseline: addressing the healthy phase
7. Probabilistic uncertainty output
8. All results with statistical tests
9. Comparison to published SOTA
10. Honest limitations

---

## Execution Order

1. **Part A**: Restructure repo (30 min)
2. **Part B**: Dataset analysis — THE MOST IMPORTANT PART (60 min)
3. **Part C.1-C.2**: Retrain on clean data, compare source groups (30 min)
4. **Part C.3**: Expand to 32 episodes + reproduce baseline (15 min)
5. **Part D.1**: TCN-Transformer supervised baseline (20 min)
6. **Part D.2**: JEPA + TCN-Transformer (15 min)
7. **Part D.3**: Deviation-from-baseline features (15 min)
8. **Part D.4**: Hybrid (only if D.3 helps) (10 min)
9. **Part E.1**: Contiguous block masking (15 min)
10. **Part E.2**: Dual-channel raw+FFT (15 min)
11. **Part F**: Probabilistic output (15 min)
12. **Part G**: Visualization + reporting (30 min)

**Time budget**: ~4h total. If running behind, skip E.2 and D.4 first.

---

## Validation Checklist (After EVERY Experiment)

- [ ] RMSE on held-out test episodes (never trained on)
- [ ] 5-seed average with std
- [ ] PCA + t-SNE plot saved
- [ ] Embedding Spearman correlation with RUL
- [ ] Pretraining loss curve saved (if pretraining changed)
- [ ] EXPERIMENT_LOG.md updated: hypothesis, change, result, keep/revert
- [ ] Paired t-test vs previous best, report p-value

---

## Error Recovery

- If pretraining diverges: halve LR, retry once. If still diverges, revert and log.
- If RMSE regresses >10%: revert immediately, log failure, move on.
- If any phase takes >1.5x budget: stop, save partial results, move on.
- If GPU OOM: reduce batch to 32, then 16.

---

## What NOT To Do

- Do NOT change multiple things at once. One variable per experiment.
- Do NOT skip Part B (dataset analysis). This is the foundation.
- Do NOT tune EMA momentum or LR schedule before fixing the data. Bad data in = bad features out.
- Do NOT spend >10 min on data loading bugs — fall back to V8 pipeline.
- Do NOT retrain temporal contrastive encoder (V8's is fine as comparison).
- Do NOT implement survival analysis (future V10).
- Do NOT reduce patch count below 12.
- Do NOT use piecewise-linear labels — linear RUL for V8 consistency.

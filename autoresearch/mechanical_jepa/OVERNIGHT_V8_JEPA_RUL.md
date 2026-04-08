# Overnight V8: JEPA Pretraining + RUL Pipeline — Full Implementation

**Goal**: Implement, train, evaluate, and iteratively improve a complete JEPA-based pipeline for bearing RUL% prediction. This is the main research contribution — self-supervised JEPA representations for remaining useful life estimation from vibration signals.

**Agent**: ml-researcher
**Estimated duration**: 10-14 hours (use the full night)
**Output**: Trained models, JSON results, notebook, updated baselines, literature-validated claims

---

## Executive Summary of Design Decisions

These decisions were made through extensive discussion and analysis. Do NOT revisit them — implement them.

1. **Single channel** (primary vibration accelerometer per source)
2. **12,800 Hz** target sampling rate (resample all sources)
3. **1024-sample window** (0.08s) — accommodates FEMTO's short snapshots (1,280 samples at 12.8kHz)
4. **16 patches of 64 samples** each, mask 10 (62.5%)
5. **RUL% is the right target** (not absolute time) — because actual time depends on usage/load
6. **Elapsed time is a valid input** to the RUL model, but the key metric is improvement OVER the elapsed-time-only baseline
7. **Envelope RMS** is the primary handcrafted health indicator for comparison
8. **Raw vibration input** for the main model; FFT as optional ablation channel
9. **JEPA encoder frozen** after pretraining; lightweight temporal model trained for RUL
10. **Episode lifetime variance** is what makes the task non-trivial — elapsed time alone fails when bearings have different lifetimes

---

## Architecture Specification

### JEPA Encoder (adapted from V2)

The existing V2 architecture needs modification for the new window size:

```
Current V2:  (B, 3, 4096) → 16 patches of 256 → 4-layer Transformer d=512
New V8:      (B, 1, 1024) → 16 patches of 64  → 4-layer Transformer d=256
```

**Encoder details:**
- Input: (B, 1, 1024) — single channel, 1024 samples at 12,800 Hz
- Patch embedding: Linear projection of 64-sample patches → d=256
- Positional encoding: Sinusoidal (fixed, not learnable — critical for avoiding collapse)
- Transformer: 4 layers, d=256, 4 heads, MLP ratio 4.0, dropout 0.1
- Output: (B, 16, 256) patch embeddings, or (B, 256) via mean-pool
- Parameters: ~1.3M (reduced from 5.1M due to smaller d and single channel)

**Why d=256 instead of 512**: Single channel (was 3), shorter patches (64 vs 256 samples), so the input dimensionality per patch is 64 (was 768). d=256 is proportional and prevents over-parameterization on a simpler input.

### JEPA Predictor

```
- Input projection: Linear(256 → 128)
- Positional encoding: Sinusoidal (128-dim, fixed)
- Mask tokens: Shared learnable token (128-dim)
- Transformer: 4 layers, d=128, 4 heads
- Output projection: Linear(128 → 256)
- Receives: 6 context patch embeddings + 10 mask token positions
- Predicts: 10 masked patch embeddings (256-dim each)
```

### EMA Target Encoder
- Deep copy of encoder, updated via EMA with momentum=0.996
- No gradients
- Processes all 16 patches (no masking) to produce prediction targets

### Loss Function
- **L1 loss** on L2-normalized predictions vs targets (NOT MSE — MSE causes mean-prediction collapse)
- **Variance regularization** (lambda=0.1): penalize low variance across predicted positions
- Total: `loss = L1(pred_norm, target_norm) + 0.1 * relu(0.1 - pred.var(dim=1).mean())`

### RUL Temporal Model (Stage 2)

Two variants to implement:

**Variant A — MLP (no history, tests single-snapshot quality):**
```
Input: [z_t (256-dim), elapsed_time (1-dim)] → concat → (257-dim)
MLP: 257 → 256 → ReLU → 128 → ReLU → 1
Output: RUL% ∈ [0, 1] (sigmoid)
```

**Variant B — LSTM (with history, main method):**
```
Per-step input: [z_t (256-dim), delta_t (1-dim)] → concat → (257-dim)
LSTM: input_size=257, hidden_size=128, num_layers=2, dropout=0.1
Per-step output: LSTM hidden → Linear(128 → 1) → sigmoid → RUL%

Additional global input to final layer: elapsed_time (concatenated with LSTM output)
Final: [lstm_out (128-dim), elapsed_time (1-dim)] → Linear(129 → 1) → sigmoid
```

**delta_t** = time in seconds since previous snapshot (handles irregular sampling: FEMTO 10s, XJTU-SY 60s, IMS ~600s).

---

## Dataset Specification

### HuggingFace Access
```python
from datasets import load_dataset
TOKEN = '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'
bearings = load_dataset("Forgis/Mechanical-Components", "bearings", split="train", token=TOKEN)
```

### Sources for Pretraining (Stage 1)
All bearing sources, resampled to 12,800 Hz, 1024-sample windows:

| Source | Native SR | Resampling | Samples | Windows (approx) |
|--------|-----------|------------|---------|-------------------|
| CWRU | 12,000 | ×16/15 | 60 | ~6,000 (long signals) |
| MFPT | 48,828 | ÷3.81 | 20 | ~2,000 |
| FEMTO | 25,600 | ÷2 | 3,569 | ~3,569 (1 per snapshot) |
| XJTU-SY | 25,600 | ÷2 | 1,370 | ~13,700 (long signals) |
| IMS | 20,480 | ×5/8 | 1,256 | ~6,000 |
| Paderborn | 64,000 | ÷5 | 384 | ~10,000 (long signals) |
| Ottawa | 42,000 | ÷3.28 | 180 | ~1,000 |
| MAFAULDA | 50,000 | ÷3.91 | 800 | ~4,000 |
| VBL | 20,000 | ÷1.56 | 800 | ~4,000 |
| SCA | 8,192* | ×25/16 | 2,663 | ~5,000 |

*SCA is 8,192 Hz — upsample to 12,800. Check if upsampling introduces artifacts.

**Exclude gearbox sources** (MCC5-THU, PHM2009, OEDI) — V6 showed mixing component types hurts.

**Total: ~55,000 windows for pretraining.** No labels used.

### Sources for RUL Evaluation (Stage 2)
Only sources with run-to-failure episodes and `episode_position`:

| Source | Episodes | Snapshots/episode | Snapshot interval | Total snapshots |
|--------|----------|-------------------|-------------------|-----------------|
| FEMTO | ~17 bearings | 100-2500 | 10s | ~3,569 |
| XJTU-SY | ~15 bearings | 50-300 | 60s | ~1,370 |

**Episode-based split**: Train on ~75% of episodes, test on ~25%. Same split across all methods.

**Critical: compute actual elapsed time per snapshot.** FEMTO: snapshot_index × 10s. XJTU-SY: snapshot_index × 60s. Store as metadata.

**Lifetime variance check**: Compute and report the min/max/mean/std of episode durations in seconds. This quantifies how much the time-only baseline will fail.

### Channel Selection (1 channel per source)
- CWRU: DE_accel (drive end — most diagnostic)
- FEMTO: channel 0 (horizontal accelerometer)
- XJTU-SY: channel 0 (horizontal)
- Paderborn: vibration_1
- IMS: channel with known failure bearing
- MAFAULDA: underhang_rad (radial, most fault-sensitive)
- Ottawa: accelerometer column 0
- VBL: accel_x
- SCA: first vibration channel
- MFPT: primary accelerometer

---

## Implementation Plan

### Phase 0: Data Pipeline (1.5 hours)

Create `mechanical-jepa/v8/data_pipeline.py`:

1. **Unified loader**: Load from HF parquet, extract channel 0, resample to 12,800 Hz
2. **Windowing**: Non-overlapping 1024-sample windows. For FEMTO (1,280 samples at 12.8kHz), take first 1024 samples. For longer signals, extract multiple windows.
3. **Instance normalization**: Per-window zero-mean, unit-variance
4. **Episode-aware loading for RUL**: Group by `episode_id`, sort by `episode_position`, compute `elapsed_time_seconds` and `delta_t_seconds`
5. **Split definitions**: Store in JSON. Pretraining uses all windows. RUL uses episode-based 75/25 split.
6. **Sanity checks**: Print per-source statistics (n_windows, signal duration, SR verification)

**Verify**: After resampling, plot a few waveforms before/after to check for artifacts. Especially for SCA (upsampling from 8,192 Hz).

### Phase 1: JEPA V8 Architecture (1 hour)

Create `mechanical-jepa/v8/jepa_v8.py`:

Adapt from `src/models/jepa_v2.py` with these changes:
- `n_channels=1` (was 3)
- `window_size=1024` (was 4096)
- `patch_size=64` (was 256)
- `embed_dim=256` (was 512)
- `predictor_dim=128` (was 256)
- Keep everything else: 4-layer encoder, 4-layer predictor, sinusoidal PE, L1 loss, var_reg=0.1, EMA momentum=0.996

Also implement `get_embeddings(x, pool='mean')` → (B, 256) for downstream use.

### Phase 2: JEPA Pretraining (2 hours)

Create `mechanical-jepa/v8/pretrain.py`:

- Load all bearing windows from Phase 0
- Train JEPA for 100 epochs (or until convergence)
- AdamW, lr=1e-4, cosine schedule, warmup 5 epochs
- Batch size 64
- Monitor: prediction loss, prediction variance, encoder embedding std
- **Collapse detection**: If prediction variance drops below 0.01, stop and debug
- Save checkpoint every 20 epochs
- Save best checkpoint (lowest validation prediction loss)

**Validation**: Hold out 10% of windows (random, not episode-based) for monitoring pretraining loss. This is NOT the RUL test set.

### Phase 3: Embedding Quality Check (30 min)

Before training the RUL model, verify the JEPA encoder produces useful embeddings:

1. **t-SNE/UMAP visualization**: Color by source_id, health_state, fault_type. Do clusters form?
2. **Linear probe**: Freeze encoder, train logistic regression on embeddings → health_state (healthy/faulty). Compare to random encoder.
3. **Envelope RMS correlation**: Compute Spearman correlation between mean embedding norm and envelope RMS. A good encoder should correlate with health indicators.

If embeddings show no structure → debug pretraining before proceeding.

### Phase 4: RUL Baselines (1.5 hours)

Create `mechanical-jepa/v8/rul_baselines.py`:

Implement ALL of these on the same episode split:

| # | Method | Encoder | Temporal | What it tests |
|---|--------|---------|----------|---------------|
| 1 | Constant mean | None | None | Trivial: always predict mean RUL% |
| 2 | Elapsed time only | None | Linear regression | Time-only baseline (how good can you do without vibration?) |
| 3 | Handcrafted + MLP | 18 features | MLP([feat, elapsed_time]) | Single-snapshot + handcrafted + clock |
| 4 | Handcrafted + LSTM | 18 features per step | LSTM over feature sequence | Trending over handcrafted features |
| 5 | Envelope RMS + LSTM | Scalar envelope RMS per step | LSTM over HI sequence | Classic prognostics approach |
| 6 | JEPA + MLP | Frozen V8 encoder | MLP([z_t, elapsed_time]) | JEPA single-snapshot + clock |
| 7 | **JEPA + LSTM** | Frozen V8 encoder | LSTM over z sequence | **Our main method** |
| 8 | End-to-end CNN-LSTM | Jointly trained | CNN encoder + LSTM | Supervised SOTA comparison |

For baselines 4, 5, 7, 8: include `delta_t` (seconds since last snapshot) as per-step input to handle irregular sampling.

For all LSTM-based methods: include `elapsed_time` as input to the final prediction layer.

**Seeds**: 42, 123, 456

### Phase 5: Evaluation & Metrics (1 hour)

Create `mechanical-jepa/v8/evaluate.py`:

**Primary metric**: RMSE on rul_percent (0-1 scale)

**All metrics to compute**:
- RMSE (primary — standard in literature)
- MAE (robust to outliers)
- R² (variance explained)
- Spearman correlation (ranking quality)
- **RMSE improvement over time-only baseline** (the metric that isolates vibration's contribution)
- **Per-episode RMSE** (some episodes are harder)
- **RMSE on short-lived vs long-lived episodes** (where does vibration help most?)
- **PHM 2012 asymmetric score** (for literature comparability, penalizes late predictions more)
- **Monotonicity of predicted HI** (does the predicted health decrease over time? measured as fraction of consecutive pairs where HI_t > HI_{t+1})

**Evaluation protocol**:
- Episode-based split: same split for all methods
- Report per-dataset (FEMTO, XJTU-SY) and combined
- 3 seeds with different model initializations (but SAME episode split)
- Report mean ± std across seeds
- All results to JSON

### Phase 6: FFT Ablation (1 hour)

Create `mechanical-jepa/v8/pretrain_fft.py`:

Same as Phase 2 but with 2-channel input:
- Channel 0: raw vibration (1024 samples)
- Channel 1: FFT magnitude (|FFT| of the raw signal, 513 bins → zero-padded to 1024)

Modify encoder: `n_channels=2`, patch embedding input dim = 64×2 = 128.

Train, evaluate with same protocol. Compare JEPA(raw) vs JEPA(raw+FFT) on all metrics.

**This tests**: Does explicit frequency information help the encoder learn better representations for RUL? Our hypothesis: FFT helps for RUL (degradation is frequency-domain phenomenon) even though it hurt for cross-domain classification in V6 (spectral features were dataset-specific).

### Phase 7: Review & Iterate (2+ hours)

**THIS IS THE MOST IMPORTANT PHASE. Do NOT skip it.**

After Phase 6, you have initial results. Now iterate:

#### Round 1: Sanity Check
- Does the time-only baseline confirm that episode lifetimes vary enough to make the task non-trivial?
- Does JEPA + LSTM (#7) beat handcrafted + LSTM (#4)? If not, why?
- Does adding elapsed_time to JEPA + MLP (#6) vs JEPA + LSTM (#7) show that temporal context helps?
- Is the pretraining converged? Check loss curves. Try more epochs if not.
- Are there episodes where ALL methods fail? Investigate — these may be data quality issues.

#### Round 2: Literature Comparison
Search the web for:
- "FEMTO PRONOSTIA RUL prediction 2024 2025 2026 SOTA"
- "self-supervised bearing RUL"
- "DCSSL bearing remaining useful life"
- "single window health indicator bearing"

For each relevant paper found:
1. What RMSE do they report on FEMTO?
2. What evaluation protocol do they use? (same episodes? piecewise-linear labels?)
3. Are we using the same label convention? (linear RUL vs clamped-at-1 during healthy phase)
4. Implement any missing SOTA baseline that's feasible

**Key paper to compare against**: DCSSL (Scientific Reports, 2026) — only SSL + RUL on FEMTO. If we beat this, the self-supervised comparison is won.

#### Round 3: Hyperparameter Sensitivity
If results are promising but not yet beating baselines:
- Try embed_dim = 128, 256, 512
- Try patch_size = 32, 64, 128
- Try mask_ratio = 0.5, 0.625, 0.75
- Try LSTM hidden_size = 64, 128, 256
- Try LSTM num_layers = 1, 2, 3
- Try learning rate for Stage 2: 1e-3, 5e-4, 1e-4

Log all configurations and results. Find the best combo.

#### Round 4: Analysis & Understanding
- **What does the JEPA embedding capture?** Compute correlation between each embedding dimension and handcrafted features (RMS, kurtosis, envelope RMS, spectral entropy). Which embedding dimensions are most predictive of RUL?
- **When does JEPA help most?** Plot per-episode RMSE for handcrafted vs JEPA. Is JEPA better on specific operating conditions or fault types?
- **Latent trajectory visualization**: For a test episode, plot z_t (projected to 2D via PCA) over time. Does the trajectory show a clear degradation path?

#### Round 5: Final Polish
- Re-run best configuration with 5 seeds (42, 123, 456, 789, 1024) for tighter confidence intervals
- Compute statistical significance (paired t-test) for key claims
- Generate publication-quality figures
- Write consolidated results document

### Phase 8: Notebook & Documentation (1 hour)

Create `mechanical-jepa/notebooks/08_rul_jepa.ipynb`:

1. **Motivation**: Why RUL% from vibration? Why is elapsed-time-alone insufficient?
2. **Data**: Episode lifetime distribution, show variance. Waveform examples at different degradation stages.
3. **JEPA Pretraining**: Loss curves, embedding visualization, collapse check
4. **Baselines Table**: All 8 methods, RMSE ± std, organized by method type
5. **Key Comparison**: JEPA+LSTM vs Handcrafted+LSTM (same temporal model, different encoder)
6. **FFT Ablation**: Raw vs Raw+FFT
7. **Per-Episode Analysis**: Where does JEPA help? Where does it fail?
8. **Latent Trajectories**: t-SNE/PCA of embeddings colored by RUL%
9. **Summary**: What JEPA adds, what it doesn't, honest limitations

Save figures as PDF+PNG in `mechanical-jepa/notebooks/plots/`.

---

## File Organization

```
mechanical-jepa/v8/
├── data_pipeline.py          # Unified data loading, resampling, windowing
├── jepa_v8.py                # JEPA V8 architecture (encoder + predictor + EMA)
├── pretrain.py               # Stage 1: self-supervised pretraining
├── pretrain_fft.py           # Stage 1 variant: raw + FFT input
├── rul_baselines.py          # All 8 RUL baselines
├── rul_model.py              # Stage 2: MLP and LSTM temporal models
├── evaluate.py               # Metrics computation
├── analyze.py                # Embedding analysis, correlation, visualization
├── results/
│   ├── pretrain_metrics.json
│   ├── rul_baselines.json
│   ├── rul_fft_ablation.json
│   ├── hyperparameter_sweep.json
│   └── final_results.json
├── checkpoints/
│   ├── jepa_v8_raw_best.pt
│   └── jepa_v8_fft_best.pt
└── RESULTS.md                # Consolidated results with analysis
```

---

## Execution Environment

- Running on SageMaker with GPU (CUDA)
- Python: torch, numpy, scipy, sklearn, xgboost, datasets (HuggingFace), matplotlib, seaborn
- Install anything else via pip
- Working directory: `/home/sagemaker-user/IndustrialJEPA/`
- HF token: `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`

## Existing Code to Reuse

- `mechanical-jepa/src/models/jepa_v2.py` — Adapt encoder/predictor architecture
- `mechanical-jepa/baselines/features.py` — Handcrafted feature extraction (18 features)
- `mechanical-jepa/baselines/data_utils.py` — HF data loading patterns, resampling
- `mechanical-jepa/baselines/results/*.json` — V7 baseline results for comparison

## Critical Constraints

1. **Do NOT use gearbox data** for pretraining — V6 proved it hurts bearing representations
2. **Do NOT use MSE loss** for JEPA — causes mean-prediction collapse. Use L1.
3. **Do NOT use learnable positional encoding** in predictor — causes position collapse. Use sinusoidal.
4. **Always include elapsed_time** as input to RUL models — but report improvement OVER time-only baseline
5. **Include delta_t** (seconds since last snapshot) for LSTM models — FEMTO=10s, XJTU-SY=60s
6. **All results backed by JSON** — no "log only" numbers
7. **3 seeds minimum** for all stochastic methods
8. **Commit after each major phase** so progress is saved

## Priority If Time-Constrained

1. Phase 0-1 (data + architecture) — foundation, must complete
2. Phase 2-3 (pretraining + quality check) — core contribution
3. Phase 4-5 (RUL baselines + evaluation) — the comparison
4. Phase 7 rounds 1-2 (sanity check + literature) — validation
5. Phase 6 (FFT ablation) — bonus
6. Phase 7 rounds 3-5 (hyperparameters + analysis) — polish
7. Phase 8 (notebook) — documentation

## Success Criteria

The session is successful if:
- [ ] JEPA V8 pretrains without collapse (prediction variance > 0.01 throughout)
- [ ] JEPA embeddings show structure (health_state linear probe >> random encoder)
- [ ] JEPA + LSTM beats handcrafted + LSTM on RUL RMSE (main claim)
- [ ] Results are compared against published SOTA numbers from literature
- [ ] All results in JSON with 3-seed statistics
- [ ] Honest assessment of where JEPA helps and where it doesn't

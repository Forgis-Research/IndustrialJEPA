# Overnight V10: Trajectory JEPA — Predict the Remaining Future from Full History

**Goal**: Replace patch-level JEPA (which learns waveform texture, not degradation) with a trajectory-level JEPA that encodes the full bearing history and predicts the latent representation of the remaining life until failure. Start simple, add complexity only when justified.

**Agent**: ml-researcher
**Estimated duration**: 4-5 hours
**Output**: HC feature analysis, trajectory JEPA implementation, trained models, plots, Quarto notebook

**REQUIRED DELIVERABLES** (the session is NOT done without these):
1. `experiments/v10/RESULTS.md` — full results table with statistical tests
2. `experiments/v10/EXPERIMENT_LOG.md` — every experiment logged
3. `notebooks/10_v10_trajectory_jepa.qmd` — complete Quarto walkthrough (engine: markdown, self-contained: true, follow format of `notebooks/08_rul_jepa.qmd` and `notebooks/09_v9_data_first.qmd`)
4. All plots saved to `analysis/plots/v10/`
5. `experiments/v10/hc_feature_analysis.md` — HC feature importance report

---

## CRITICAL: Read Before Starting

Read these files to understand the current state:

- `mechanical-jepa/experiments/v9/RESULTS.md` — V9 results (best: JEPA+LSTM RMSE=0.085)
- `mechanical-jepa/notebooks/09_v9_data_first.qmd` — V9 full writeup including dataset compatibility
- `mechanical-jepa/baselines/features.py` — all 18 HC features (FEATURE_NAMES list at line 185)
- `mechanical-jepa/data/loader.py` — data loading, `load_rul_episodes()`, `compute_handcrafted_features_per_snapshot()`
- `mechanical-jepa/pretraining/jepa.py` — current patch-level JEPA (V8 architecture)
- `mechanical-jepa/downstream/rul/models.py` — RULLSTM, RULMLP heads
- `mechanical-jepa/downstream/rul/train.py` — RUL training loop
- `mechanical-jepa/data/registry.py` — dataset metadata

**V9 key results (31 episodes: 16 FEMTO + 15 XJTU-SY, 24 train / 7 test)**:
- Elapsed time only: RMSE = 0.224
- V9 JEPA+LSTM (best): RMSE = 0.0852 ± 0.0014
- V9 Heteroscedastic LSTM: RMSE = 0.0868 ± 0.0023, PICP@90% = 0.910
- V8 Hybrid JEPA+HC: RMSE = 0.055 ± 0.004 (NOT re-run on V9 31-episode split)

**V9 key problems**:
1. JEPA embeddings have near-zero correlation with RUL (max dim corr = -0.121). Patch prediction learns waveform texture, not degradation.
2. The LSTM head does all the work — JEPA embeddings provide marginal signal over elapsed time.
3. The t-SNE/PCA plots show no RUL-based clustering — just random scattering of colors.
4. V8 Hybrid JEPA+HC (0.055) still beats everything, suggesting HC features carry most RUL signal.

**V9 DCSSL comparison error**: The notebook cites DCSSL RMSE=0.131 but this number does NOT appear in the DCSSL paper (Shen et al., Scientific Reports, 2026). The actual DCSSL result on FEMTO is RMSE=0.0822 (Table 4 of the paper). This needs to be corrected.

---

## Part A: HC Feature Importance Analysis (45 min)

The 18 HC features are listed in `baselines/features.py` line 185:
```
Time domain (8):  rms, peak, crest_factor, kurtosis, skewness, shape_factor, impulse_factor, clearance_factor
Frequency (7):    spectral_centroid, spectral_spread, spectral_entropy, band_energy_0_1kHz, band_energy_1_3kHz, band_energy_3_5kHz, band_energy_5_nyq
Envelope (3):     envelope_rms, envelope_kurtosis, envelope_peak
```

### A.1: Per-feature correlation with RUL

For all 31 episodes (FEMTO + XJTU-SY), compute each HC feature per snapshot. Then compute Spearman rank correlation of each feature with RUL% across all snapshots. Also compute per-episode correlations and report mean ± std.

Produce a bar chart of |Spearman ρ| for all 18 features, sorted by magnitude. Save as `analysis/plots/v10/hc_feature_correlations.png`.

### A.2: Feature ablation on HC+MLP baseline

Run the HC+MLP baseline (from `downstream/rul/baselines.py`) with different feature subsets, 5 seeds each:

1. **All 18 features** (V8/V9 baseline)
2. **Top-3 features** by correlation from A.1
3. **Top-5 features**
4. **Top-10 features**
5. **Spectral centroid only** (1 feature)
6. **Time-domain only** (8 features)
7. **Frequency-domain only** (7 features)

Use the same 31-episode, 24/7 train/test split as V9. Report RMSE ± std for each.

### A.3: Feature ablation on HC+LSTM baseline

Repeat the same subsets with the LSTM temporal head instead of MLP. This tests whether temporal modeling changes which features matter.

### A.4: Summary

Write `experiments/v10/hc_feature_analysis.md` with:
- Correlation table (all 18 features)
- Ablation results (MLP and LSTM)
- Recommendation: which features to keep going forward
- Key insight: how many features does it take to match the full-18 performance?

---

## Part B: Trajectory JEPA — Simplest Possible Version (90 min)

### The Core Idea

Instead of predicting masked patches within a single 0.08s window (V8/V9 JEPA), predict the **latent representation of the remaining bearing life** from the **full history up to the current time**.

This operates at the episode level, not the window level:
- **Tokens** = per-snapshot feature vectors (NOT raw waveform patches)
- **Context** = all snapshots from start to current time t
- **Target** = all snapshots from t+1 to failure T
- **Self-supervised objective**: predict target representation from context representation

### B.0: Prepare episode-level data

For each of the 31 run-to-failure episodes:
1. Extract per-snapshot features using the TOP features identified in Part A (e.g., top-5 HC features). Call this z_t ∈ R^F where F is the number of features kept.
2. Record timestamps τ_t in hours for each snapshot.
3. Store as a list of episodes, each episode = list of (z_t, τ_t) tuples.

**Important**: Start with HC features for z_t, NOT JEPA embeddings. HC features have proven RUL correlation (spectral centroid ρ=0.585). We can swap in learned embeddings later, but the trajectory JEPA should work first with good inputs.

### B.1: Architecture — Keep it minimal

```
Stage 1: Snapshot features (frozen, from Part A)
  Per snapshot: raw waveform → HC top-K features → z_t ∈ R^F
  
Stage 2: Trajectory JEPA
  Context encoder:  Transformer(z_1+pe(τ_1), ..., z_t+pe(τ_t)) → h_past
  Target encoder:   Transformer(z_{t+1}+pe(τ_{t+1}), ..., z_T+pe(τ_T)) → h_future  (EMA)
  Predictor:        MLP(h_past) → ĥ_future
  Loss:             ||ĥ_future - sg(h_future)||² + variance regularization
```

**Context encoder** — causal Transformer:
- Input: z_t ∈ R^F projected to d=64 via Linear, plus continuous-time sinusoidal PE
- 2 layers, 4 heads, d_model=64, MLP ratio=2, dropout=0.1
- Causal attention mask (each token only attends to itself and earlier tokens)
- Output: **last hidden state** h_t ∈ R^64 (the hidden state at position t summarizes the full past)

**Target encoder** — bidirectional Transformer (EMA copy of context encoder weights, but WITHOUT causal mask):
- Input: z_{t+1}...z_T projected to d=64, plus continuous-time PE
- Same architecture as context encoder but full (bidirectional) attention
- Output: **attention-pooled** over all future tokens → h_future ∈ R^64
- Attention pooling: learned query q ∈ R^64, weights = softmax(q · H / √64), output = weighted sum

**Predictor** — small MLP:
- Input: h_past ∈ R^64
- Linear(64, 128) → ReLU → Linear(128, 64)
- Output: ĥ_future ∈ R^64

**Continuous-time positional encoding**:
```python
def continuous_time_pe(timestamps_hours, d_model):
    """Sinusoidal PE from elapsed time in hours (not integer position)."""
    t = timestamps_hours.unsqueeze(-1)  # (seq_len, 1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(len(timestamps_hours), d_model)
    pe[:, 0::2] = torch.sin(t * div)
    pe[:, 1::2] = torch.cos(t * div)
    return pe
```

**EMA update**: θ_target ← m · θ_target + (1-m) · θ_context, with m=0.996

**Variance regularization** (anti-collapse, same as V8):
```python
var_reg = max(0, 1 - std(h_future, dim=0).mean())  # encourage spread across dims
```

### B.2: Training data generation

For each episode of length T, sample multiple cut points:
- Sample t uniformly from [5, T-3] (need at least 5 past tokens and 3 future tokens)
- Each cut gives one (context, target) training pair
- Per epoch: sample 10 cut points per episode → 24 episodes × 10 = 240 pairs/epoch
- Train for 200 epochs → 48,000 total pairs seen

Batch size: 1 (each example has different sequence length). Accumulate gradients over 8 examples before stepping.

Optimizer: AdamW, lr=3e-4, weight_decay=0.01, cosine schedule over 200 epochs with 20-epoch warmup.

### B.3: Downstream probe

After trajectory JEPA pretraining, freeze the context encoder and predictor. Train a linear probe:

```
Input:  ĥ_future = Predictor(ContextEncoder(z_1...z_t))  ∈ R^64
Output: RUL% = σ(Linear(ĥ_future))  ∈ [0, 1]
Label:  1 - τ_t / τ_T

Probe training: MSE loss, AdamW lr=1e-3, 100 epochs, 5 seeds
```

Also train the same probe on the context encoder output directly (without the predictor) as an ablation:
```
h_past → Linear → RUL%   (tests whether the encoder alone is enough)
```

### B.4: Baselines to compare against

Run all of these on the same 24/7 train/test split, 5 seeds:

1. **Elapsed time only**: RUL% = 1 - t/t_max_train (no learning)
2. **HC top-K + MLP** (from Part A, best subset)
3. **HC top-K + LSTM** (from Part A, best subset)
4. **V9 JEPA+LSTM** (RMSE=0.085, from V9 — just cite the number, don't re-run)
5. **Trajectory JEPA probe on ĥ_future** (B.3)
6. **Trajectory JEPA probe on h_past** (B.3 ablation)

### B.5: Sanity checks

Before reporting results, verify:
1. **Trajectory JEPA pretraining loss decreases** over epochs (not flat or diverging)
2. **h_future varies by RUL%**: compute Spearman correlation of PC1 of h_future with RUL%. Should be >> 0 if the target encoder captures degradation.
3. **Probe beats elapsed-time-only** by a meaningful margin. If it doesn't, the trajectory JEPA learned nothing beyond counting tokens.
4. **Token-count leakage test**: shuffle the z values within each context sequence (destroying temporal order but keeping sequence length). If probe RMSE barely changes, the model is just counting tokens.

---

## Part C: Trajectory JEPA Improvements — Only If B Works (60 min)

**IMPORTANT**: Only proceed to Part C if the trajectory JEPA from Part B shows promising results (beats HC+LSTM baseline, or shows clear learning signal in the pretraining loss + embedding quality). If B fails, skip C entirely and write up the negative result honestly.

### C.1: Heteroscedastic probe

Replace the linear RUL probe with a heteroscedastic version:
```python
Linear(64, 2)  →  (μ, log_σ²)
Loss: Gaussian NLL = 0.5 * (log_σ² + (y - μ)² / σ²)
```
Report RMSE, PICP@90%, MPIW. Compare to V9 heteroscedastic LSTM.

### C.2: Learned snapshot encoder (Stage 1 upgrade)

Replace HC features with a small learned snapshot encoder:
- Same PatchEmbed1D from `pretraining/jepa.py` (patch_size=64, embed_dim=64)
- 2-layer Transformer, d=64, mean pool → z_t ∈ R^64
- Train Stage 1 with patch-level JEPA on all compatible sources (V9 compatible_6 group)
- Then freeze Stage 1, re-run Stage 2 trajectory JEPA

Compare: HC features vs learned embeddings as Stage 1 input to trajectory JEPA.

### C.3: Binary failure classifier (alternative to RUL regression)

Instead of predicting RUL%, train a binary classifier:
```
Label = 1 if bearing fails within k hours, 0 otherwise
Input = Predictor(h_past, pe(k))  — predictor takes horizon as additional input

For training: sample random (t, k) pairs from episodes
  If τ_t + k ≥ τ_T → label = 1 (failure within horizon)
  If τ_t + k < τ_T → label = 0 (survives)

Loss: binary cross-entropy
```

At test time, sweep k to get survival curve. Report calibration.

**Note**: This requires modifying the predictor to accept horizon as input:
```python
Predictor: MLP(concat(h_past, pe(k_hours)))  → ĥ_future_k
           Input: 64 + 64 = 128 → Linear(128, 128) → ReLU → Linear(128, 64)
```

---

## Part D: Visualization and Analysis (45 min)

### D.1: Trajectory JEPA embedding plots

- PCA of h_past colored by RUL% (compare to V9 JEPA embedding PCA — should show much clearer gradient)
- PCA of ĥ_future colored by RUL%
- t-SNE of h_past colored by RUL% and by source (FEMTO vs XJTU-SY)
- Degradation trajectories: PC1 of h_past over normalized episode time for 5 test episodes

### D.2: Pretraining dynamics

- Trajectory JEPA loss curve over 200 epochs
- Variance regularization term over epochs
- h_future PC1 correlation with RUL% at different training checkpoints (does it improve?)

### D.3: HC feature analysis plots

- Bar chart of per-feature Spearman ρ with RUL (from Part A)
- Ablation results bar chart (different feature subsets)

### D.4: Results comparison bar chart

- All methods RMSE comparison with error bars
- Include V9 baselines for reference

---

## Part E: Quarto Notebook (30 min)

Write `notebooks/10_v10_trajectory_jepa.qmd` covering:

1. Motivation: why patch-level JEPA fails for RUL (t-SNE showing no clusters, low correlation)
2. HC feature analysis: which features matter, ablation results
3. Trajectory JEPA: architecture, training, the idea of predicting the latent future
4. Results table with all methods and statistical tests
5. Embedding quality comparison (V9 JEPA vs trajectory JEPA)
6. Honest limitations
7. All plots embedded with captions

**Format**: engine: markdown, self-contained: true, tables hardcoded in markdown (NOT computed live).

---

## Experiment Ordering and Time Budget

| Part | Experiment | Est. Time | Depends On |
|------|-----------|:---------:|:----------:|
| A.1 | HC feature correlations | 15 min | — |
| A.2 | HC+MLP ablations (7 subsets × 5 seeds) | 15 min | A.1 |
| A.3 | HC+LSTM ablations (7 subsets × 5 seeds) | 15 min | A.1 |
| A.4 | Write HC analysis report | 10 min | A.2, A.3 |
| B.0 | Prepare episode data with top-K features | 10 min | A.4 |
| B.1-B.2 | Trajectory JEPA pretraining | 30 min | B.0 |
| B.3 | Downstream probes (5 seeds) | 15 min | B.2 |
| B.4 | Baselines comparison | 15 min | A.4 |
| B.5 | Sanity checks | 10 min | B.3 |
| C.1 | Heteroscedastic probe | 10 min | B.3 (if B works) |
| C.2 | Learned Stage 1 encoder | 25 min | B.3 (if B works) |
| C.3 | Binary failure classifier | 15 min | B.3 (if B works) |
| D | Plots and analysis | 30 min | All above |
| E | Quarto notebook | 30 min | All above |
| | **Total** | **~4.5h** | |

---

## Anti-Patterns to Avoid

1. **Do NOT re-run V9 experiments**. Cite V9 numbers from `experiments/v9/RESULTS.md`.
2. **Do NOT use patch-level JEPA for trajectory tokens.** Start with HC features. The whole point is testing the trajectory-level architecture, not debugging bad embeddings.
3. **Do NOT build a complex model first.** Part B must use the simplest possible version (2-layer Transformer d=64, linear probe). Complexity comes in Part C only if B works.
4. **Do NOT skip sanity checks** (B.5). The token-count leakage test is critical.
5. **Do NOT report "improvement over V9" without noting the methodological differences** (different architecture, same data split).
6. **Do NOT proceed to Part C if Part B fails.** Write up the negative result honestly instead.
7. **Fix the DCSSL number**: The V9 notebook cites DCSSL RMSE=0.131 which is wrong. The actual number from the paper is RMSE=0.0822 (Shen et al., Sci Rep 2026, Table 4). Note this correction in the V10 writeup.

---

## What Success Looks Like

**Minimum viable result**: Trajectory JEPA probe beats elapsed-time-only AND the token-count shuffle test shows the model uses signal content (not just sequence length).

**Good result**: Trajectory JEPA probe matches or beats HC+LSTM (RMSE ~0.08-0.09 range).

**Great result**: Trajectory JEPA probe approaches V8 Hybrid JEPA+HC (RMSE ~0.055). This would mean the trajectory-level architecture captures degradation dynamics that the LSTM misses.

**The HC feature analysis is valuable regardless of trajectory JEPA results.** Even if the trajectory JEPA fails, knowing which 3-5 features carry most of the RUL signal is a concrete contribution.

# Overnight V11: Trajectory JEPA on C-MAPSS — Unsupervised Pretraining for Turbofan RUL

**Goal**: Pivot from bearings to NASA C-MAPSS turbofan engine data. Build a Trajectory JEPA variant that pretrains without failure-time labels (addressing the V10 "cheating" concern), then fine-tunes with minimal labels, targeting the label-efficiency story.

**Agent**: ml-researcher
**Estimated duration**: 6-7 hours
**Working directory**: `C:\Users\Jonaspetersen\dev\IndustrialJEPA\mechanical-jepa\experiments\v11`

**REQUIRED DELIVERABLES**:
1. `experiments/v11/data_analysis/` — dataset characterization figures + report
2. `experiments/v11/RESULTS.md` — full results table with statistical tests
3. `experiments/v11/EXPERIMENT_LOG.md` — every experiment logged
4. `notebooks/11_v11_cmapss_trajectory_jepa.qmd` — Quarto walkthrough (engine: markdown, self-contained: true)
5. `analysis/plots/v11/` — all figures
6. `experiments/v11/models.py`, `data_utils.py`, `train_utils.py`, `run_experiments.py` — clean implementation

---

## CRITICAL: Read Before Starting

1. **`autoresearch/mechanical_jepa/CMAPSS_SOTA_REVIEW.md`** — verified SOTA numbers, reproducibility warnings, and gaps we exploit
2. **`paper-replications/star/REPLICATION_SPEC.md`** — STAR architecture (supervised SOTA we compare against)
3. **`paper-replications/star/fan2024-star-sensors.pdf`** — the STAR paper (for data protocol details)
4. **`mechanical-jepa/experiments/v10/RESULTS.md`** — V10 bearing results (best Traj JEPA: RMSE 0.155, HC+LSTM Top-3: 0.025)
5. **`mechanical-jepa/notebooks/10_v10_trajectory_jepa.qmd`** — V10 full writeup
6. Check if STAR replication has been run at `paper-replications/star/results/RESULTS.md` — if yes, use those numbers; if no, use paper-reported numbers as targets

---

## Key Context from Prior Work

**Verified C-MAPSS targets (see CMAPSS_SOTA_REVIEW.md)**:

| Subset | Supervised SOTA (STAR 2024) | Only public SSL baseline (AE-LSTM 2025) |
|:------:|:---------------------------:|:---------------------------------------:|
| FD001  | 10.61 | 13.99 |
| FD002  | 13.47 | — |
| FD003  | 10.71 | — |
| FD004  | 15.87 (STAR) / 14.25 (TMSCNN) | 28.67 |

**The SSL gap**: 32% worse than supervised on FD001. **No JEPA-style or MAE-style paper exists on C-MAPSS** — confirmed absent from the literature review. This is our opportunity.

**V10 lessons carried forward**:
- Patch-level JEPA on bearings learned waveform texture, not degradation → failed for RUL
- Trajectory-level JEPA (V10) learned some degradation structure (h_past PC1 corr = 0.424 with RUL) but couldn't beat HC+LSTM Top-3 at 18 episodes
- The real bottleneck was **data**, not architecture
- **C-MAPSS has 100-260 training engines per subset** — 10× more than bearings
- This is where Trajectory JEPA should finally have enough data to breathe

**Design decisions locked in** (from user discussion):
1. **"Unsupervised" = pretraining does not use failure times** (episode-structure and cycle indices OK)
2. **Variable-horizon prediction**: predict h_future_k for k sampled from [5, 30] cycles ahead
3. **Multivariate input**: primary L=1 (cycle-as-token), ablation L=4 (patch-as-token). NO FFT (doesn't apply to C-MAPSS).
4. **Data scope**: start with FD001 only; expand to all 4 only if FD001 works
5. **Label budgets**: 100%, 50%, 20%, 10%, 5%
6. **Primary metric**: RMSE on last window per test engine, RUL cap 125
7. **Data analysis FIRST**, before any modeling
8. **Pre-flight sanity checks verified**: C-MAPSS at `C:\Users\Jonaspetersen\dev\OpenTSLM\data\cmapss\6. Turbofan Engine Degradation Simulation Data Set\`, 14 sensor selection matches STAR exactly (constant sensors s1,5,6,10,16,18,19 drop verified on FD001)

---

## Part A: C-MAPSS Dataset Characterization (60 min)

Before writing any model code, deeply understand the data. This phase is mandatory.

### A.0: Pre-flight sanity checks (ALREADY VERIFIED)

The following has been verified before this overnight session — use as sanity checks for your loader:

- **Data location**: `C:\Users\Jonaspetersen\dev\OpenTSLM\data\cmapss\6. Turbofan Engine Degradation Simulation Data Set\`
  (path contains a space — quote appropriately; use raw Python strings)
- **Files present**: `train_FD00{1,2,3,4}.txt`, `test_FD00{1,2,3,4}.txt`, `RUL_FD00{1,2,3,4}.txt`
- **Format**: Space-separated floats, 26 columns: `[engine_id, cycle, op_set_1, op_set_2, op_set_3, s1, s2, ..., s21]`. Load via `np.loadtxt(path)`.
- **FD001 verified facts**:
  - train_FD001.txt: shape (20631, 26), 100 engines
  - Engine length: min=128, max=362, mean=206.3
  - Op settings are CONSTANT on FD001 (std ≈ 0) → single operating condition confirmed
  - **Constant sensors to drop (verified)**: s1, s5, s6, s10, s16, s18, s19 (std ≈ 0)
  - **Informative sensors (verified, matches STAR paper exactly)**: s2, s3, s4, s7, s8, s9, s11, s12, s13, s14, s15, s17, s20, s21 (14 sensors)
  - Test set: 13096 rows for 100 engines (~131 cycles/engine avg, truncated before failure)
  - RUL_FD001.txt: 100 values (ground-truth RUL at last observed cycle per test engine)

Use these as assertions in your loader — if they don't match, your loader has a bug.

### A.1: Load and inventory

Files per subset: `train_FDXXX.txt`, `test_FDXXX.txt`, `RUL_FDXXX.txt`.

Produce a summary table:

| Subset | N train engines | N test engines | Avg cycles/engine | Min/Max cycles | N op conditions | N fault modes |
|:------:|:---------------:|:--------------:|:-----------------:|:--------------:|:---------------:|:-------------:|
| FD001  | 100 | 100 | ? | ? | 1 | 1 |
| FD002  | 260 | 259 | ? | ? | 6 | 1 |
| FD003  | 100 | 100 | ? | ? | 1 | 2 |
| FD004  | 249 | 248 | ? | ? | 6 | 2 |

Save as `data_analysis/inventory.md`.

### A.2: Episode length distribution

For each subset, plot histogram of training engine lengths (cycles to failure). Save as `data_analysis/episode_length_distributions.png` (4 subplots).

### A.3: Sensor informativeness analysis

For each of the 21 sensors on FD001 (easiest, single condition):
1. Compute Spearman rank correlation between the sensor value and "cycles since start" across all training engines
2. Rank sensors by |ρ|
3. Plot as a horizontal bar chart

Expected finding: sensors 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21 should have the strongest correlations (these are the 14 selected in STAR and other papers). Sensors 1, 5, 6, 10, 16, 18, 19 should be near zero (constant or uninformative).

**Verify this is true in our data**. If not, investigate why.

Save as `data_analysis/sensor_informativeness_fd001.png` and `data_analysis/sensor_informativeness.md`.

Repeat the analysis for FD002 (multi-condition). Report any differences.

### A.4: Operating condition clustering (FD002, FD004 only)

For FD002 and FD004, scatter-plot the 3 operating condition variables (columns op_setting_1, op_setting_2, op_setting_3). Use KMeans(n_clusters=6) to identify the 6 operating regimes. Color-code the scatter. Verify 6 distinct clusters exist.

Save as `data_analysis/operating_conditions_fd002.png`, `data_analysis/operating_conditions_fd004.png`.

### A.5: Per-condition sensor statistics

For FD002 and FD004, compute per-operating-condition mean and std of each selected sensor. Plot as heatmap (conditions × sensors). This shows how different the sensor baselines are across conditions.

**This is critical**: If the per-condition variation is large, global min-max normalization is a problem. This is a well-known issue in C-MAPSS multi-condition subsets.

Save as `data_analysis/per_condition_sensor_stats.png`.

### A.6: Degradation trajectories

Pick 5 representative training engines from each subset. For each, plot 3 informative sensors (e.g., sensors 2, 9, 14) vs cycle index. Shows the "healthy flat → degradation" trajectory shape.

Save as `data_analysis/degradation_trajectories.png` (4 × 5 grid).

### A.7: Cross-subset comparison

Overlay 10 engines from each subset on the same axes (sensor 2 vs cycle, sensor 9 vs cycle). Shows the domain shift between subsets.

Save as `data_analysis/cross_subset_comparison.png`.

### A.8: RUL label distributions

Plot histogram of:
1. Raw RUL (uncapped): distribution of (T - t) for all training samples
2. Capped RUL (min(T-t, 125)): distribution after clipping

Show how much of the data has RUL = 125 (the constant phase). Expected: 40-60% of samples.

Save as `data_analysis/rul_distribution.png`.

### A.9: Summary report

Write `data_analysis/CMAPSS_ANALYSIS.md` with all findings and figures linked. Include:
- Dataset inventory table
- Sensor selection justification (which 14 sensors and why)
- Normalization strategy recommendation (per-sensor vs per-condition)
- Potential pitfalls identified
- Any surprising findings

---

## Part B: Data Pipeline (45 min)

### B.1: Data loader

```python
CMAPSS_DATA_DIR = r"C:\Users\Jonaspetersen\dev\OpenTSLM\data\cmapss\6. Turbofan Engine Degradation Simulation Data Set"
SELECTED_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]  # 1-indexed
# Column indices in the raw file: [engine_id=0, cycle=1, op1=2, op2=3, op3=4, s1=5, ..., s21=25]
SENSOR_COLUMNS = [5 + (s - 1) for s in SELECTED_SENSORS]  # zero-indexed column positions

def load_cmapss_subset(subset: str) -> dict:
    """
    Load one C-MAPSS subset.
    
    Assertions for FD001 (use to sanity check your loader):
      - train shape = (20631, 26)
      - 100 train engines
      - engine length: min=128, max=362, mean=206.3
      - 100 test engines
      - 100 RUL values
    
    Returns dict with:
        'train_engines': list of np.ndarray — (T_i, 14) sensors per engine (after selection)
        'train_cycles':  list of np.ndarray — cycle numbers per engine
        'test_engines':  list of np.ndarray — sensors per test engine (truncated)
        'test_rul':      np.ndarray — ground truth RUL at last observed cycle per test engine
        'op_settings_train': list of np.ndarray — (T_i, 3) operating condition per engine
        'op_settings_test':  list of np.ndarray — same for test engines
    """
```

Extract the 14 selected sensors after loading. Verify shapes against the assertions above.

### B.2: Normalization

For FD001/FD003 (single condition): **per-sensor min-max on training data**.

For FD002/FD004 (multi-condition): **per-operating-condition per-sensor min-max**.

```python
def normalize_multi_condition(sensors, op_settings, train_stats):
    """
    For each timestep, identify the operating condition from op_settings,
    then apply that condition's normalization stats.
    """
```

Normalization stats must be computed on training engines only. Test engines are normalized using training statistics.

### B.3: RUL labels

```python
def compute_rul_labels(n_cycles: int, rul_max: int = 125) -> np.ndarray:
    """Piecewise linear RUL with cap."""
    rul = np.arange(n_cycles, 0, -1, dtype=np.float32)
    return np.minimum(rul, rul_max)
```

### B.4: Train/validation split

Hold out 15% of training engines (by engine_id, not by cycle) as validation. Use seed=42 for determinism. Applies to both pretraining (if same engines) and fine-tuning.

### B.5: Pretraining dataset (cut points)

```python
class CMAPSSPretrainDataset(Dataset):
    """
    Pretraining dataset for Trajectory JEPA.
    
    Each item:
        past: (t, 14) sensor history up to cycle t
        future: (k, 14) next k cycles
        k: int, horizon (sampled per item)
        t: int, cut point
    
    Samples n_cuts_per_epoch cut points per engine per epoch.
    Cut points sampled uniformly from [min_past, T - max_horizon].
    Horizons sampled uniformly from [min_horizon, max_horizon].
    
    CRITICAL: This dataset uses cycle indices and engine sequences ONLY.
    It does NOT use T_failure or per-cycle RUL labels.
    """
    def __init__(self, engines, n_cuts_per_epoch=20, 
                 min_past=10, min_horizon=5, max_horizon=30):
```

### B.6: Fine-tuning dataset (labeled)

```python
class CMAPSSFinetuneDataset(Dataset):
    """
    Fine-tuning dataset for RUL prediction.
    
    Each item:
        past: (t, 14) sensor history
        rul: float — RUL at cycle t (capped at 125, normalized to [0, 1])
    
    Uses FULL RUL labels. This is the supervised fine-tuning phase.
    """
```

### B.7: Test dataset

Test engines get truncated before failure. For each test engine, take the LAST window_length cycles (or left-pad if shorter than window_length). The test label is the ground-truth RUL at the last observed cycle, from `RUL_FDXXX.txt`.

---

## Part C: Trajectory JEPA Architecture for C-MAPSS (75 min)

**Key simplification vs V10**: C-MAPSS cycles are already preprocessed sensor readings (14-dim per cycle after sensor selection). No waveform, no ultrasonic content, no need for a CNN Stage 1 encoder.

### Patching decision (important)

**Should we patch the time series?** This is the bearings-vs-turbofan question.

For **bearings (V10)**, patching was essential because each snapshot was a 2560-sample raw waveform — individual samples are meaningless, patches of 64 samples capture one ball-pass event. Patching was unavoidable.

For **C-MAPSS**, each cycle is already one reduced/aggregated reading per sensor. A single cycle is *already* an informative multivariate feature vector (14-dim). Patching means grouping L consecutive cycles into one token.

Arguments for patching (L > 1):
- Reduces sequence length (longer engines have 362 cycles → quadratic attention gets expensive)
- Captures short-term trends within a patch (local context)
- Matches STAR's successful approach (they use ~L=4)

Arguments against patching (L = 1):
- Simpler
- Preserves fine-grained temporal resolution
- Each cycle already has rich information (14 sensors)

**Our approach: run both as ablations in the same session.**

- **Primary**: L = 1 (cycle-as-token). Simplest, ~100-362 tokens per engine. Test first.
- **Ablation**: L = 4 (STAR-style patching). ~25-90 tokens per engine. Run after L=1 works.

The sensor projection layer becomes:

```python
class SensorProjection(nn.Module):
    """
    Projects multivariate sensor readings into model dimension.
    Supports both L=1 (cycle-as-token) and L>1 (patch-as-token) via stacking.
    """
    def __init__(self, n_sensors=14, patch_length=1, d_model=128):
        super().__init__()
        self.patch_length = patch_length
        self.proj = nn.Linear(n_sensors * patch_length, d_model)
    
    def forward(self, x):
        # x: (B, T, n_sensors)
        B, T, S = x.shape
        L = self.patch_length
        if L > 1:
            # Trim T to multiple of L
            T_trim = (T // L) * L
            x = x[:, :T_trim, :]
            # Reshape into patches: (B, T/L, L*S)
            x = x.reshape(B, T_trim // L, L * S)
        return self.proj(x)  # (B, T_patches, d_model)
```

### FFT / frequency domain — skip for C-MAPSS

For bearings (V9 E.2), we tried a dual-channel raw+FFT input and it made things worse (downstream RMSE 0.112 vs 0.087 with raw). The FFT channel gave better embedding correlation but overfitting from doubled PatchEmbed dimensions killed downstream performance.

For C-MAPSS, **FFT is even less justified**:
1. Each cycle is already an aggregated scalar per sensor (no sub-cycle waveform exists)
2. The only "frequency" computable is cycle-to-cycle variation — which the Transformer already learns from the raw sequence
3. No ball-pass frequencies, no resonances, no periodic content at sub-cycle scale

**Decision: no FFT for C-MAPSS.** The time series is already in feature space.

**What we CAN optionally add** (not in the primary path): first-difference features `Δx_t = x_t - x_{t-1}`, which capture short-term rate of change per sensor. Concatenate to raw sensor values → 28-dim input instead of 14-dim. Run this as a secondary ablation if time permits.

### C.1: Sensor projection (Stage 1, minimal)

See `SensorProjection` above. L=1 primary, L=4 ablation.

No CNN, no ultrasonic waveform processing. The token at cycle t is just the 14-dim sensor vector → project to 128-dim.

### C.2: Continuous-time positional encoding

```python
def sinusoidal_pe(positions, d_model):
    """
    Sinusoidal PE indexed by cycle (integer, but treat as continuous).
    positions: (T,) or (B, T)
    Returns: (..., d_model)
    """
```

Use cycle index as position. For C-MAPSS cycles are integer (1, 2, 3, ...) so standard sinusoidal PE works directly.

### C.3: Context encoder (causal)

```python
class ContextEncoder(nn.Module):
    """
    Causal Transformer: processes full history up to cycle t.
    Output: h_past = last hidden state (d_model).
    """
    def __init__(self, d_model=128, n_layers=2, n_heads=4, dropout=0.1):
        # 2-layer Transformer with causal mask
    
    def forward(self, z_seq):
        # z_seq: (B, T, d_model) — full history
        # Returns: h_past (B, d_model) — last hidden state
```

### C.4: Target encoder (bidirectional, EMA)

```python
class TargetEncoder(nn.Module):
    """
    Bidirectional Transformer (no causal mask).
    Output: h_future = attention-pooled over future tokens.
    """
    def __init__(self, d_model=128, n_layers=2, n_heads=4, dropout=0.1):
        # 2-layer Transformer (no causal mask)
        self.attn_pool_query = nn.Parameter(torch.randn(d_model))
    
    def forward(self, z_future):
        # z_future: (B, k, d_model) — the next k cycles
        # Returns: h_future (B, d_model)
```

EMA update: `θ_target ← 0.996 · θ_target + 0.004 · θ_context`

### C.5: Predictor (horizon-aware)

```python
class Predictor(nn.Module):
    """
    Predicts h_future from h_past conditioned on horizon k.
    """
    def __init__(self, d_model=128, d_hidden=256):
        self.horizon_proj = nn.Linear(d_model, d_model)  # project PE(k)
        self.net = nn.Sequential(
            nn.Linear(2 * d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
        )
    
    def forward(self, h_past, k):
        # h_past: (B, d_model)
        # k: (B,) integer horizons
        k_pe = sinusoidal_pe(k.float(), self.d_model)  # (B, d_model)
        x = torch.cat([h_past, k_pe], dim=-1)
        return self.net(x)  # (B, d_model) predicted h_future
```

### C.6: Loss function

```python
def trajectory_jepa_loss(h_past, h_future, pred_future, lambda_var=0.01):
    """
    L2 prediction loss + variance regularization (anti-collapse).
    """
    pred_loss = F.mse_loss(pred_future, h_future.detach())
    
    # Variance regularization: encourage h_future to have unit variance per dim
    std = h_future.std(dim=0)
    var_loss = torch.relu(1.0 - std).mean()
    
    return pred_loss + lambda_var * var_loss
```

### C.7: Full model

```python
class TrajectoryJEPA(nn.Module):
    def __init__(self, n_sensors=14, d_model=128, n_layers=2, n_heads=4):
        self.sensor_proj = SensorProjection(n_sensors, d_model)
        self.context_encoder = ContextEncoder(d_model, n_layers, n_heads)
        self.target_encoder = TargetEncoder(d_model, n_layers, n_heads)
        self.predictor = Predictor(d_model)
        # EMA sync target_encoder with context_encoder initially
    
    def forward(self, past_sensors, future_sensors, k):
        z_past = self.sensor_proj(past_sensors) + pe_past
        z_future = self.sensor_proj(future_sensors) + pe_future
        
        h_past = self.context_encoder(z_past)
        with torch.no_grad():
            h_future = self.target_encoder(z_future)
        
        pred_future = self.predictor(h_past, k)
        return pred_future, h_future, h_past
    
    def update_ema(self, momentum=0.996):
        # Update target encoder params from context encoder
```

---

## Part D: Pretraining (60 min)

### D.1: Pretraining protocol

```python
config = {
    'n_layers': 2,
    'd_model': 128,
    'n_heads': 4,
    'lr': 3e-4,
    'weight_decay': 0.01,
    'batch_size': 8,  # episodes per batch (variable length)
    'n_epochs': 200,
    'ema_momentum': 0.996,
    'n_cuts_per_epoch': 20,
    'min_past': 10,
    'min_horizon': 5,
    'max_horizon': 30,
    'lambda_var': 0.01,
}
```

Train on FD001 training engines (85% used, 15% held out for validation).

### D.2: Pretraining loop

```python
for epoch in range(200):
    for batch in pretrain_loader:
        past, future, k = batch
        pred_future, h_future, h_past = model(past, future, k)
        loss = trajectory_jepa_loss(h_past, h_future, pred_future)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.update_ema()
    
    # Every 10 epochs: linear probe evaluation on validation
    if epoch % 10 == 0:
        probe_rmse = linear_probe_eval(model, val_engines_labeled)
        log(f"epoch={epoch}, loss={loss:.4f}, probe_rmse={probe_rmse:.2f}")
```

**Important**: The linear probe at epoch X uses the current context encoder to produce h_past for the validation set, then fits a linear RUL regressor on top. This is a proxy for pretraining quality — if probe_rmse decreases over epochs, the encoder is learning something useful.

### D.3: Sanity checks

After pretraining:
1. **h_past PC1 vs RUL correlation** on training data. Should be > 0.4 (V10 got 0.424 on bearings).
2. **h_past PCA plot** colored by RUL. Should show a visible gradient.
3. **Shuffle test**: randomly permute past tokens, re-run encoder, compute probe RMSE. If shuffled RMSE ≈ un-shuffled RMSE, the encoder is only counting tokens. Abort and reconfigure.
4. **Loss curve**: should decrease monotonically. If it diverges after epoch 10 (like V8/V9), investigate — shouldn't happen with variable horizon but be vigilant.

Save as `analysis/plots/v11/pretraining_diagnostics.png` (4-panel figure).

---

## Part E: Fine-tuning at Multiple Label Budgets (60 min)

### E.1: Fine-tuning protocol

For each label budget ∈ [100%, 50%, 20%, 10%, 5%]:

1. Sample that fraction of training engines (seeded, deterministic)
2. Freeze the context encoder
3. Train a linear probe: `Linear(d_model, 1) → sigmoid → RUL × 125`
4. Use MSE loss, Adam lr=1e-3, 100 epochs, early stopping with patience 20
5. Run 5 seeds per label budget
6. Evaluate on the full test set

### E.2: Also run fine-tuning with encoder unfrozen (E2E)

For each label budget, also run end-to-end fine-tuning (unfreeze encoder, lower lr = 1e-4). Compare to frozen-probe results.

### E.3: Supervised baseline at each label budget

For fair comparison: run a supervised LSTM or small Transformer trained from scratch at each label budget (no pretraining). This is the reference curve to beat.

### E.4: Results matrix

```
                      Test RMSE (mean ± std, 5 seeds)
Label fraction:    100%      50%      20%      10%      5%
Supervised LSTM:   ???       ???      ???      ???      ???
Traj JEPA frozen:  ???       ???      ???      ???      ???
Traj JEPA E2E:     ???       ???      ???      ???      ???
STAR (reference):  10.61     -        -        -        -
```

---

## Part F: Analysis and Visualization (45 min)

### F.1: Label efficiency plot (the money plot)

X-axis: label fraction (log scale: 5%, 10%, 20%, 50%, 100%)
Y-axis: test RMSE
Three curves: Supervised LSTM, Traj JEPA frozen, Traj JEPA E2E
Horizontal line: STAR reference (10.61)

Save as `analysis/plots/v11/label_efficiency.png`.

**The win scenario**: Traj JEPA at 20% labels matches Supervised LSTM at 100%. If this holds, we have a paper.

### F.2: h_past embedding visualization

PCA and t-SNE of h_past for all test engine (sliding window every 5 cycles). Color by RUL. Compare to V10 bearing plot.

Save as `analysis/plots/v11/h_past_pca.png`, `analysis/plots/v11/h_past_tsne.png`.

### F.3: Prediction trajectories

For 5 sample test engines, plot predicted RUL vs cycle (predictions at every cycle, with the Trajectory JEPA applied online). Overlay ground truth. Save as `analysis/plots/v11/prediction_trajectories.png`.

### F.4: Loss curves

Pretraining loss + probe RMSE over epochs. Save as `analysis/plots/v11/training_curves.png`.

### F.5: h_past correlation heatmap

Spearman correlation between top-5 PCA components of h_past and: (RUL, cycle_index, 14 sensor values). Shows what h_past actually encodes.

Save as `analysis/plots/v11/h_past_correlations.png`.

---

## Part G: Expansion to FD002 / Cross-Subset Transfer (60 min, OPTIONAL)

**Only if FD001 results show real signal (probe beats supervised LSTM baseline at 20% labels).**

### G.1: Run FD002 in-domain

Train Trajectory JEPA on FD002 training engines (multi-condition). Fine-tune. Compare to STAR's FD002 result (13.47).

### G.2: Cross-condition transfer

Pretrain on FD002+FD004 (multi-condition, lots of data, hard distribution). Fine-tune on FD001 with few labels. Compare to Traj JEPA pretrained on FD001 alone.

This is the **transfer learning story**: SSL pretraining on diverse data transfers better than training on the target subset alone.

---

## Part H: Quarto Notebook (30 min)

Write `notebooks/11_v11_cmapss_trajectory_jepa.qmd` with:

1. **Motivation**: Why C-MAPSS? Pivot from bearings story. SSL gap in the literature. What we're trying to prove.
2. **Dataset characterization**: Key findings from Part A (sensor informativeness, operating conditions, episode distributions). 3-4 figures.
3. **Method**: Trajectory JEPA adapted for C-MAPSS. Architecture diagram. Variable-horizon prediction motivation.
4. **Results**: Label efficiency curve (money plot). Embedding quality. Prediction trajectories.
5. **Benchmark table vs STAR and SSL literature**: Like we did in V10, bold only numbers we reproduced (STAR if the star replication is done, otherwise cite paper).
6. **Honest limitations**: What didn't work, what we don't know, where we need more work.
7. **Conclusion and next steps**.

Follow format conventions from `notebooks/08_rul_jepa.qmd`, `notebooks/09_v9_data_first.qmd`, `notebooks/10_v10_trajectory_jepa.qmd`. `engine: markdown`, `self-contained: true`, tables hardcoded.

---

## Experiment Ordering and Time Budget

| Part | Task | Est. Time | Depends On |
|------|------|:---------:|:----------:|
| A | Dataset characterization | 60 min | — |
| B | Data pipeline | 45 min | A |
| C | Trajectory JEPA architecture (with L configurable) | 75 min | B |
| Test | Test pipeline (5 epochs smoke test) | 15 min | C |
| D1 | Pretraining FD001, **L=1 (primary)** | 45 min | Test passes |
| E1 | Fine-tuning L=1 at 5 label budgets | 45 min | D1 |
| D2 | Pretraining FD001, **L=4 (ablation)** | 30 min | E1 done |
| E2 | Fine-tuning L=4 at 5 label budgets | 30 min | D2 |
| F | Analysis and visualization | 45 min | E1, E2 |
| G | FD002 / cross-subset transfer | 60 min | F (optional) |
| H | Quarto notebook | 30 min | All above |
| | **Total** | **~7h** | |

If running short on time: skip Part G, skip L=4 ablation (D2/E2), focus on A-F with L=1 only + H. The L=1 primary path is the main deliverable.

---

## Iteration Points

This session should include **two explicit iteration checkpoints**:

### Checkpoint 1 (after Part A)
After dataset characterization, log a summary in EXPERIMENT_LOG.md:
- What did we learn about the data?
- Are there surprises that change the design?
- Is per-operating-condition normalization needed for FD001? (Probably no — only 1 condition.)
- Are any of the 14 selected sensors actually uninformative on our data?

Update design choices before proceeding to Part B.

### Checkpoint 2 (after Part D)
After pretraining, log diagnostics:
- Did pretraining loss decrease smoothly?
- What is h_past PC1 ρ with RUL? (target: > 0.4)
- Did the shuffle test pass? (shuffled RMSE should be notably worse)

**If pretraining diagnostics are bad**: do NOT proceed to full fine-tuning. Debug:
- Check EMA update is working
- Check data pipeline (is the future actually from the same engine?)
- Check loss function (variance regularization not dominating)
- Reduce model size if overfitting
- Try longer max_horizon (e.g., [5, 50])

If still bad after debugging, write up negative result and stop.

---

## Anti-Patterns to Avoid

1. **Do NOT skip Part A**. The data analysis is where most bugs hide in new datasets.
2. **Do NOT use failure times in pretraining**. This is the core methodological claim. The CMAPSSPretrainDataset must not touch `T - t` or `rul_max`.
3. **Do NOT compare V11 RMSE to V10 RMSE directly**. V10 is on bearings + cut-point protocol. V11 is on C-MAPSS + last-window protocol. Different datasets, different metrics, different scales.
4. **Do NOT report RMSE < 10.0 on FD001 without investigation**. From the SOTA review: RMSE < 10 on FD001 is a red flag for data leakage. Verify train/test split is clean.
5. **Do NOT use sliding-window evaluation on test set**. Last-window-per-engine is the canonical protocol. Sliding-window can artificially lower RMSE by 15-25%.
6. **Do NOT train with batch_size > 16 for variable-length sequences**. Engine lengths differ a lot (128 to 362 cycles on FD001). Use small batches + gradient accumulation, or pad within a batch.
7. **Do NOT forget to save the best pretrained checkpoint**. Use the epoch with lowest validation probe RMSE, not the last epoch.
8. **Do NOT proceed to Part E if Part D diagnostics fail**. See Checkpoint 2 above.
9. **Do NOT skip the shuffle test**. It's the single most important sanity check — it tells you whether the model is using temporal information or just counting tokens.
10. **Do NOT compute PHM 2008 Score as a primary metric**. Use RMSE. Report Score only as a secondary number for consistency with the literature.

---

## Success Criteria

**Minimum viable result (MVP)**: 
- Pretraining loss decreases monotonically (no V8/V9-style collapse)
- h_past PC1 ρ with RUL > 0.4 on FD001 (better than V10 bearings)
- Shuffle test shows temporal signal is used
- Trajectory JEPA at 100% labels beats supervised LSTM trained from scratch at 100% labels by any margin

**Good result**:
- Traj JEPA frozen probe at 100% labels: FD001 RMSE ≤ 14.0 (within 30% of SOTA)
- Traj JEPA E2E at 100% labels: FD001 RMSE ≤ 12.5
- Label efficiency: Traj JEPA at 20% labels ≥ Supervised LSTM at 20% labels by ≥ 1.0 RMSE
- Establishes a defensible SSL baseline on C-MAPSS that exceeds AE-LSTM's public 13.99

**Great result**:
- Traj JEPA E2E at 100% labels: FD001 RMSE ≤ 11.5 (within 10% of SOTA)
- Label efficiency: Traj JEPA at 20% labels matches Supervised LSTM at 100% labels
- Cross-subset transfer from FD002+FD004 → FD001 works (Part G)
- This is the scenario for a NeurIPS submission: first SSL method to approach supervised SOTA with 5× label efficiency

**Breakthrough**:
- Traj JEPA E2E: FD001 RMSE ≤ 10.61 (matches STAR)
- AND label efficiency at ≤ 20%

Honestly, breakthrough is unlikely on the first try. Aim for Good with stretch Great.

---

## Key Architectural Caveats (from design discussion)

1. **Per-operating-condition normalization for FD002/FD004**: Critical. A known bug source. Must normalize per operating condition, not globally. For FD001/FD003, single-condition global normalization is fine.

2. **Variable-length sequence batching**: Engines have different lengths. Use pad_sequence + attention mask, or batch_size=1 with gradient accumulation. Do NOT truncate all engines to the same length — you'd throw away the short/long episode signal.

3. **Cycle 1 "cold start"**: At cycle 1, the context has only 1 token. The encoder must handle this. Verify by running an episode from cycle 5 onwards (skipping the degenerate 1-token case) or by padding with special tokens.

4. **Test-time evaluation**: Test engines are TRUNCATED before failure. The last window of cycles is the input; the ground truth is from `RUL_FDXXX.txt` (not computable from the sensor data alone). Do not try to "compute RUL" for test engines — read it from the ground truth file.

5. **RUL cap**: 125 is standard. Do not change. Any other choice breaks comparability with the literature.

6. **The "too-easy RUL = 125" problem**: Because RUL is capped, ~50% of training samples have label 125. The model could just predict 125 constantly and get a decent MSE. Check that predictions vary meaningfully, not constant.

7. **Early stopping on validation RMSE, not pretraining loss**: The pretraining loss is not directly tied to downstream performance. Save checkpoints by probe RMSE on the validation set.

8. **Deterministic data splits**: Use seed=42 for all train/val splits. Reproducibility matters.

---

## Recommended Initial Architecture

Small, simple, fast to iterate:

```
ContextEncoder:
  Input: (B, T, 14) sensors
  Linear: 14 → 128
  Add continuous-time PE
  Transformer: 2 layers, d=128, 4 heads, FFN ratio 2, dropout 0.1, CAUSAL
  Output: last hidden state → (B, 128)

TargetEncoder (EMA of ContextEncoder, but WITHOUT causal mask):
  Same Transformer, 2 layers, d=128, 4 heads
  Attention pool: learned query, attention over T tokens → (B, 128)

Predictor:
  Input: [h_past (128), pe(k) (128)] = 256
  Linear 256 → 256 → ReLU → Linear 256 → 128
  Output: predicted h_future (128)

Total params: ~1.2M
```

This is deliberately small. If it underfits we can scale up. Starting small catches architectural bugs before they cost us overnight compute.

---

## Questions to Answer in RESULTS.md

1. What is the Traj JEPA FD001 RMSE at 100%, 50%, 20%, 10%, 5% labels? How does it compare to a supervised LSTM trained at the same budget?
2. Is there a label budget at which Traj JEPA beats the supervised baseline? What is it?
3. What is the h_past quality (PC1 correlation with RUL, shuffle test result)?
4. Does the model use temporal ordering (shuffle test)?
5. What is the pretraining loss curve shape? Any instability?
6. What are the qualitative features of the predicted RUL trajectories (for 5 sample engines)?
7. How does our Traj JEPA compare to STAR (paper) and AE-LSTM (the only public SSL baseline)?
8. What's the story: "we beat SOTA" (unlikely), "we match SOTA with fewer labels" (possible), "we establish first JEPA baseline" (achievable)?
9. What are the top 3 things that would improve results if we had more time / data?
10. Is the pivot from bearings to C-MAPSS worth continuing, or should we pivot again?

# Overnight: STAR Replication (Fan et al., Sensors 2024)

**Goal**: Replicate the STAR paper's Table 5 results on all four C-MAPSS subsets. STAR is the current verified supervised SOTA for turbofan engine RUL prediction. This replication establishes a strong baseline we'll use for all future IndustrialJEPA work on C-MAPSS.

**Agent**: ml-researcher
**Estimated duration**: 5-6 hours
**Working directory**: `C:\Users\Jonaspetersen\dev\IndustrialJEPA\paper-replications\star`

**REQUIRED DELIVERABLES**:
1. `models.py` — STAR architecture in PyTorch
2. `data_utils.py` — C-MAPSS data loader with sensor selection, normalization, RUL labels
3. `train_utils.py` — Training loop with early stopping, evaluation
4. `run_experiments.py` — Main experiment runner for all 4 subsets
5. `test_pipeline.py` — Quick validation (5 epochs, verify no NaN, shape checks)
6. `results/` — Per-subset JSON results and plots
7. `RESULTS.md` — Final comparison table vs paper Table 5
8. `EXPERIMENT_LOG.md` — Every experiment logged

---

## CRITICAL: Read Before Starting

1. Read `REPLICATION_SPEC.md` in this directory — complete architectural specification
2. Read `fan2024-star-sensors.pdf` in this directory — the original paper (especially §3 Methodology, §4 Experiments)
3. Read `../cnn-gru-mha/` — successful replication pattern to follow
4. Read `../dcssl/` — partial replication, note the failure modes
5. Read `../../autoresearch/mechanical_jepa/CMAPSS_SOTA_REVIEW.md` — C-MAPSS SOTA context and reproducibility warnings

---

## Part A: Setup and Data Pipeline (60 min)

### A.1: Download C-MAPSS dataset

The NASA C-MAPSS dataset is publicly available. Check if already downloaded:
- Look in `C:\Users\Jonaspetersen\dev\IndustrialJEPA\mechanical-datasets\` for C-MAPSS files
- Look in system download locations
- If not present, download from the NASA Prognostics Data Repository

Expected files per subset (FD001, FD002, FD003, FD004):
- `train_FDXXX.txt`: training engine sensor readings
- `test_FDXXX.txt`: test engine sensor readings (truncated before failure)
- `RUL_FDXXX.txt`: ground-truth RUL at the final observation of each test engine

Raw format of train/test files:
```
[engine_id, cycle, op_setting_1, op_setting_2, op_setting_3, s1, s2, ..., s21]
```

### A.2: Data loader

```python
def load_cmapss_subset(subset: str, data_dir: Path):
    """
    Load one C-MAPSS subset.
    
    Returns:
        train_engines: list of (sensor_array, rul_array) per engine
        test_engines: list of (sensor_array,) per engine (no labels)
        test_rul: array of ground-truth RUL for each test engine (from RUL file)
    """
```

- Parse train file: group by engine_id, extract 14 selected sensors (columns 2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21 — 1-indexed from the sensor names in Table 2)
- Parse test file: same but stop before failure
- Parse RUL file: one value per test engine

### A.3: Normalization

Min-max per sensor, computed on training data only:
```python
x_min = train_sensors.min(axis=0)
x_max = train_sensors.max(axis=0)
normalized = (x - x_min) / (x_max - x_min + 1e-8)
```

For FD002/FD004: Also try **per-operating-condition normalization** as an ablation. The 6 operating conditions are determined by clustering the 3 op_setting columns. Keep per-sensor normalization as primary; per-condition as ablation.

### A.4: RUL labels (piecewise linear with cap)

```python
def compute_rul_labels(n_cycles, rul_max=125):
    """Piecewise linear RUL: constant at rul_max during healthy, linear decay."""
    rul = np.arange(n_cycles, 0, -1, dtype=np.float32)  # T, T-1, ..., 1
    rul = np.minimum(rul, rul_max)
    return rul
```

### A.5: Sliding window generator

```python
def create_windows(sensors, ruls, window_length):
    """
    Sliding windows with stride 1.
    Returns (N_windows, window_length, 14) inputs and (N_windows,) labels.
    Label = RUL at the LAST cycle of the window.
    """
```

For test engines: take ONLY THE LAST window of each engine. If engine has fewer cycles than window_length, pad with the first available value at the beginning (left-pad).

### A.6: Train/validation split

Hold out 15% of training engines for validation (seeded, deterministic). Do NOT split windows — split by engine.

---

## Part B: STAR Architecture Implementation (120 min)

Implement the architecture described in `REPLICATION_SPEC.md` step by step.

### B.1: Patch embedding

```python
class DimensionWisePatchEmbed(nn.Module):
    """
    Segment each sensor into K patches of length L, embed each patch.
    
    Input: (B, T, D) where T = window length, D = 14 sensors
    Output: (B, K, D, d_model) where K = T // L
    """
    def __init__(self, d_model, patch_length, n_sensors, max_patches):
        # Affine embedding: Linear(patch_length, d_model) applied per sensor
        # Positional embedding: learnable (max_patches, n_sensors, d_model)
```

### B.2: Two-stage attention encoder block

```python
class TwoStageAttentionEncoder(nn.Module):
    """
    Stage 1: Temporal attention within each sensor
    Stage 2: Sensor-wise attention at each temporal position
    
    Input: (B, K, D, d_model)
    Output: (B, K, D, d_model)
    """
    def forward(self, x):
        # Stage 1: reshape to (B*D, K, d_model), MHA over K
        # Stage 2: reshape to (B*K, D, d_model), MHA over D
```

Use standard `nn.MultiheadAttention` with batch_first=True for both stages. Each stage is a full Transformer encoder block (LayerNorm → MHA → residual → LayerNorm → FFN → residual).

### B.3: Patch merging

```python
class PatchMerging(nn.Module):
    """
    Merge adjacent patches along the K dimension: K → K/2.
    
    Input: (B, K, D, d_model)
    Output: (B, K//2, D, d_model)
    """
    def __init__(self, d_model):
        self.proj = nn.Linear(2 * d_model, d_model)
    
    def forward(self, x):
        # x[:, 0::2], x[:, 1::2] → concat → project
```

### B.4: Two-stage attention decoder block

```python
class TwoStageAttentionDecoder(nn.Module):
    """
    Self-attention (two-stage) + cross-attention to encoder output.
    
    Inputs:
        x_dec: (B, K, D, d_model) from previous decoder layer
        x_enc: (B, K, D, d_model) from encoder at same scale
    Output: (B, K, D, d_model)
    """
```

For the first scale's decoder, use a fixed sinusoidal positional encoding as the input (as described in the paper, Vaswani-style).

### B.5: Full STAR model

```python
class STAR(nn.Module):
    def __init__(self, n_sensors=14, d_model=128, n_heads=1, 
                 patch_length=4, window_length=32, n_scales=3, dropout=0.1):
        # Patch embedding
        # List of encoder blocks, one per scale
        # List of patch merging blocks, n_scales - 1 of them
        # List of decoder blocks, one per scale
        # MLP per scale + final concat MLP
    
    def forward(self, x):
        # x: (B, T, D)
        # Return predicted RUL (B,)
```

### B.6: Loss function

Standard MSE on normalized RUL:
```python
loss = F.mse_loss(pred, target / 125.0)  # normalize target to [0, 1]
```

When computing metrics, rescale predictions back to cycles (× 125).

---

## Part C: Training and Evaluation (60 min)

### C.1: Training loop

```python
def train_star(model, train_loader, val_loader, config):
    optimizer = Adam(model.parameters(), lr=config['lr'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'])
    
    best_val_rmse = float('inf')
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        # train one epoch
        # validate
        # early stopping on val RMSE with patience 20
```

### C.2: Evaluation

```python
def evaluate_last_window(model, test_engines, ground_truth_rul):
    """
    Evaluate on the last sliding window of each test engine.
    Compute RMSE and PHM 2008 Score.
    """
    preds = []
    for engine_sensors in test_engines:
        last_window = engine_sensors[-window_length:]
        if len(engine_sensors) < window_length:
            # Left-pad with first value
            pad = np.tile(engine_sensors[0:1], (window_length - len(engine_sensors), 1))
            last_window = np.concatenate([pad, engine_sensors], axis=0)
        pred = model(last_window).item() * 125.0
        pred = min(max(pred, 0), 125)  # clamp to valid range
        preds.append(pred)
    
    preds = np.array(preds)
    rmse = np.sqrt(np.mean((preds - ground_truth_rul) ** 2))
    score = compute_phm_score(preds - ground_truth_rul)
    return rmse, score


def compute_phm_score(d):
    """PHM 2008 asymmetric scoring function."""
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return score.sum()
```

### C.3: Per-subset hyperparameters

Use exactly the values from Table 4 of the paper:

```python
HYPERPARAMS = {
    'FD001': {'lr': 0.0002, 'batch': 32, 'window_length': 32, 'n_scales': 3, 'd_model': 128, 'n_heads': 1, 'patch_length': 4},
    'FD002': {'lr': 0.0002, 'batch': 64, 'window_length': 64, 'n_scales': 4, 'd_model': 64,  'n_heads': 4, 'patch_length': 4},
    'FD003': {'lr': 0.0002, 'batch': 32, 'window_length': 48, 'n_scales': 1, 'd_model': 128, 'n_heads': 1, 'patch_length': 4},
    'FD004': {'lr': 0.0002, 'batch': 64, 'window_length': 64, 'n_scales': 4, 'd_model': 256, 'n_heads': 4, 'patch_length': 4},
}
```

Patch length 4 gives K = window_length / 4, which for FD001 (T=32, S=3) gives K=8 → 4 → 2 → 1 (four values, three merges). For FD002/FD004 (T=64, S=4) gives K=16 → 8 → 4 → 2 → 1 (five values, four merges). For FD003 (T=48, S=1) gives K=12.

---

## Part D: Run All Subsets (90 min)

For each subset (FD001, FD002, FD003, FD004):

1. Load data
2. Create windows
3. Train STAR model with the per-subset hyperparameters
4. Run 5 seeds: [42, 123, 456, 789, 1024]
5. Evaluate on test set (last window per engine)
6. Report mean ± std of RMSE and Score

Save per-subset results as JSON:
```python
{
    "subset": "FD001",
    "hyperparams": {...},
    "paper_target": {"rmse": 10.61, "score": 169},
    "seeds": [...],
    "rmse_per_seed": [...],
    "score_per_seed": [...],
    "rmse_mean": 10.8,
    "rmse_std": 0.3,
    "score_mean": 180,
    "score_std": 15,
    "delta_rmse_pct": +1.8,
    "status": "EXACT"  # GOOD/EXACT/WORSE/BETTER based on thresholds
}
```

---

## Part E: Results Compilation (30 min)

### E.1: RESULTS.md

Generate a markdown table exactly like Table 5 of the paper:

```markdown
| Subset | Paper RMSE | Our RMSE ± std | Paper Score | Our Score ± std | Status |
|--------|:----------:|:--------------:|:-----------:|:---------------:|:------:|
| FD001  | 10.61 | ??? | 169 | ??? | ??? |
| FD002  | 13.47 | ??? | 784 | ??? | ??? |
| FD003  | 10.71 | ??? | 202 | ??? | ??? |
| FD004  | 15.87 | ??? | 1449 | ??? | ??? |
```

### E.2: RUL prediction plots

For each subset, plot predicted RUL vs true RUL for all test engines (sorted by true RUL), mirroring Figure 7 of the paper. Save to `results/plots/rul_FDXXX.png`.

### E.3: Honest assessment

In RESULTS.md, include a section "Honest assessment: did we replicate STAR?". Clearly state:
- Which subsets we matched within 10% (EXACT)
- Which we matched within 20% (GOOD)
- Which we missed (>20% worse: FAILED)
- Hypotheses for any failures (patch length choice, RUL cap, multi-condition handling, etc.)

---

## Part F: Test Pipeline (15 min)

Write `test_pipeline.py` that runs FIRST before any full experiments:
1. Load FD001 data (smallest, simplest)
2. Create STAR model with FD001 hyperparameters
3. Forward pass with batch of random data → verify output shape is (B,)
4. Run 5 epochs on real data → verify loss decreases
5. Print parameter count → sanity check size
6. Evaluate on test set → verify metrics are computed correctly (even if RMSE is bad)

Run `python test_pipeline.py` and confirm ALL checks pass before proceeding to Part D.

---

## Experiment Ordering

| Part | Task | Est. Time | Depends On |
|------|------|:---------:|:----------:|
| A | Data pipeline | 60 min | — |
| B | STAR architecture | 120 min | A |
| F | Test pipeline | 15 min | A, B |
| C | Training protocol | 60 min | F passes |
| D | Run all subsets (4 × 5 seeds) | 90 min | C |
| E | Results compilation | 30 min | D |
| | **Total** | **~6h** | |

---

## Anti-Patterns to Avoid

1. **Do NOT use random RUL cap**. Paper uses 125 (per convention). Do not guess 120 or 130.
2. **Do NOT skip sensor selection**. Using all 21 sensors will change the architecture behavior and break RMSE targets.
3. **Do NOT use sliding-window evaluation on test set**. Paper uses last-window per engine. Full-sliding eval can artificially lower RMSE by 15-25%.
4. **Do NOT train without early stopping**. The paper doesn't specify epoch count but validation-based early stopping is standard.
5. **Do NOT use per-cycle normalization** (online). Use train-set statistics only, applied consistently to train and test.
6. **Do NOT commit huge model checkpoints to git**. Save them under `results/checkpoints/` and rely on `.gitignore`.
7. **Do NOT proceed to Part D if Part F fails**. Debug the test pipeline first.
8. **Do NOT confuse patch_length with window_length**. window_length = total input length T, patch_length = L (per patch), K = T / L.

---

## Success Criteria

**EXACT replication**: All 4 subsets within 10% of paper RMSE
- FD001: ≤ 11.7 (paper: 10.61)
- FD002: ≤ 14.8 (paper: 13.47)
- FD003: ≤ 11.8 (paper: 10.71)
- FD004: ≤ 17.5 (paper: 15.87)

**GOOD replication**: All 4 subsets within 20% of paper RMSE
- FD001: ≤ 12.7
- FD002: ≤ 16.2
- FD003: ≤ 12.9
- FD004: ≤ 19.0

**FAILED**: Any subset off by more than 30%. Investigate, document failure modes, move on.

The purpose is to have a **strong verified supervised baseline** for future IndustrialJEPA work. If we can't replicate STAR exactly, a GOOD replication is still sufficient — we care about the architecture being correctly implemented more than exact numerical match.

# Overnight: CNN-GRU-MHA Replication (Yu et al., Applied Sciences 2024)

**Goal**: Replicate the CNN-GRU-MHA bearing RUL prediction paper. This is a supervised transfer learning method (NOT self-supervised) that achieves FEMTO average RMSE = 0.0443. We need this as a strong baseline for comparison with our JEPA and DCSSL work.

**Agent**: ml-researcher
**Estimated duration**: 3-4 hours
**Working directory**: `C:\Users\Jonaspetersen\dev\IndustrialJEPA\cnn-gru-mha-replication`

**REQUIRED DELIVERABLES**:
1. `models.py` — CNN-GRU-MHA architecture in PyTorch
2. `data_utils.py` — FEMTO + XJTU-SY data loading with DWT preprocessing
3. `train_utils.py` — Training loop with transfer learning protocol
4. `run_experiments.py` — Main experiment runner (all transfers)
5. `results/` — Per-transfer JSON results
6. `EXPERIMENT_LOG.md` — Every experiment logged
7. `RESULTS.md` — Final comparison table vs paper targets
8. `test_pipeline.py` — Quick validation script (5 epochs, verify no NaN)

---

## CRITICAL: Read Before Starting

1. Read `REPLICATION_SPEC.md` in this directory — complete architectural specification
2. Read `yu2024-cnn-gru-mha-applsci.pdf` in this directory — the original paper
3. Read `../dcssl-replication/data_utils.py` — FEMTO data loading (REUSE this for FEMTO loading, do NOT rewrite from scratch)
4. Read `../dcssl-replication/models.py` — TCN encoder reference (different architecture but same data pipeline patterns)
5. Read `../dcssl-replication/run_experiments.py` — experiment runner pattern to follow
6. Read `../dcssl-replication/REPLICATION_SPEC.md` — structure to follow

**IMPORTANT**: The FEMTO dataset is already downloaded and available. Check `../dcssl-replication/` for the data path. Reuse the data loading code — do NOT re-download or rewrite the FEMTO CSV parser.

---

## Part A: Setup and Data Pipeline (30 min)

### A.1: Data loading

Reuse FEMTO loading from `../dcssl-replication/data_utils.py`. Key functions to import or adapt:
- `load_bearing_snapshots()` — loads CSV files → (n_snapshots, 2560, 2)
- `BEARING_EOL_FILES` — number of snapshots per bearing

**This paper uses HORIZONTAL channel only** (channel 0). Extract it after loading.

### A.2: DWT preprocessing

The paper's preprocessing is specific and must be replicated exactly:

```python
import pywt

def dwt_denoise(signal, wavelet='sym8', level=3):
    """
    DWT noise reduction: decompose with sym8 at 3 levels,
    reconstruct from approximation coefficients only
    (suppress high-frequency detail coefficients).
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # Zero out detail coefficients (keep only approximation)
    for i in range(1, len(coeffs)):
        coeffs[i] = np.zeros_like(coeffs[i])
    return pywt.waverec(coeffs, wavelet)[:len(signal)]

def minmax_normalize(signal):
    """Min-max normalize to [0, 1]."""
    xmin, xmax = signal.min(), signal.max()
    if xmax - xmin < 1e-10:
        return np.zeros_like(signal)
    return (signal - xmin) / (xmax - xmin)
```

Apply per-snapshot: denoise → normalize → this IS the health indicator.

### A.3: RUL labels

**Linear labels** (NOT piecewise): Y_i = (N - i) / N where N = total snapshots.
This means RUL starts at ~1.0 and decreases linearly to 0.0 at failure.

### A.4: Transfer experiment data splits

From Table 2 in the paper. Create a config dict:

```python
FEMTO_TRANSFERS = {
    'test1': {
        'source': 'Bearing1_3',
        'targets': ['Bearing2_3', 'Bearing2_4', 'Bearing3_1', 'Bearing3_3']
    },
    'test2': {
        'source': 'Bearing2_3',
        'targets': ['Bearing1_3', 'Bearing1_4', 'Bearing3_3', 'Bearing3_3']
    },
    'test3': {
        'source': 'Bearing3_2',
        'targets': ['Bearing1_3', 'Bearing1_4', 'Bearing2_3', 'Bearing2_4']
    }
}

XJTU_TRANSFERS = {
    'exp1': {
        'source': 'Bearing1_3',
        'targets': ['Bearing2_3', 'Bearing3_2']
    },
    'exp2': {
        'source': 'Bearing2_3',
        'targets': ['Bearing1_3', 'Bearing3_2']
    }
}
```

**Target domain split**: Each target bearing is split 1:1 into train (with labels, for FC fine-tuning) and validation (for evaluation). Use first half for training, second half for validation (chronological split, not random).

---

## Part B: Model Implementation (45 min)

### B.1: CNN feature extractor

```
Input: (batch, 1, 2560) — single-channel denoised+normalized waveform

Block 1: Conv1d(1, 32, kernel=5, padding=2) → BN → ReLU → MaxPool1d(2)
Block 2: Conv1d(32, 64, kernel=5, padding=2) → BN → ReLU → MaxPool1d(2)
Block 3: Conv1d(64, 128, kernel=5, padding=2) → BN → ReLU → MaxPool1d(2)

MHA: MultiheadAttention(embed_dim=128, num_heads=2)
     (input: (batch, seq_len, 128) after permuting)

Block 4: Conv1d(128, 256, kernel=3, padding=1) → BN → ReLU → MaxPool1d(2)
Block 5: Conv1d(256, 512, kernel=3, padding=1) → BN → ReLU → MaxPool1d(2)
Block 6: Conv1d(512, 1024, kernel=3, padding=1) → BN → ReLU → MaxPool1d(2)

GAP: AdaptiveAvgPool1d(1) → squeeze → (batch, 1024)
```

After 6 conv blocks with 3 poolings of 2 each (6 total): 2560 / (2^6) = 40. So before GAP the feature map is (batch, 1024, 40). GAP reduces to (batch, 1024).

### B.2: GRU temporal model

The CNN extracts a feature vector per snapshot. For the full bearing sequence:

```
Input: (batch=1, seq_len=N_snapshots, 1024) — one feature per snapshot
GRU layer 1: input_size=1024, hidden_size=512
GRU layer 2: input_size=512, hidden_size=128
Output: last hidden state → (1, 128)
```

### B.3: FC head

```
Input: (batch, 128)
FC1: Linear(128, 64) → ReLU
FC2: Linear(64, 1) → Sigmoid (output in [0, 1])
```

### B.4: Loss function

```python
def cnn_gru_mha_loss(pred, target, model, alpha=1e-4):
    """RMSE + L1 regularization."""
    rmse = torch.sqrt(F.mse_loss(pred, target))
    l1 = sum(p.abs().sum() for p in model.parameters())
    return rmse + alpha * l1
```

### B.5: Complete forward pass

```python
class CNNGRU_MHA(nn.Module):
    def forward(self, x_sequence):
        """
        x_sequence: (1, N_snapshots, 2560) — full bearing life
        
        1. CNN extracts features per snapshot: (1, N, 2560) → (N, 1024)
        2. GRU processes sequence: (1, N, 1024) → (1, N, 128)
        3. FC predicts RUL per timestep: (N, 128) → (N, 1)
        """
        batch, N, L = x_sequence.shape
        # Process each snapshot through CNN
        feats = []
        for t in range(N):
            snap = x_sequence[:, t, :].unsqueeze(1)  # (1, 1, 2560)
            f = self.cnn(snap)  # (1, 1024)
            feats.append(f)
        feats = torch.stack(feats, dim=1)  # (1, N, 1024)
        
        # GRU over time
        gru_out, _ = self.gru(feats)  # (1, N, 128)
        
        # FC per timestep
        rul = self.fc(gru_out.reshape(-1, 128))  # (N, 1)
        return rul.squeeze(-1)  # (N,)
```

**Note on efficiency**: For long sequences (N > 1000), processing each snapshot individually through the CNN is slow. Batch the CNN forward pass:
```python
feats = self.cnn(x_sequence.reshape(-1, 1, L))  # (N, 1024)
feats = feats.unsqueeze(0)  # (1, N, 1024) for GRU
```

---

## Part C: Training Protocol (30 min)

### C.1: Source domain training

```python
# Train full model on source bearing
optimizer = Adam(model.parameters(), lr=0.001)
for epoch in range(60):
    pred = model(source_sequence)      # (N_source,)
    loss = cnn_gru_mha_loss(pred, source_rul, model, alpha=1e-4)
    loss.backward()
    optimizer.step()
```

### C.2: Transfer learning (freeze + fine-tune)

```python
# Freeze CNN + GRU
for param in model.cnn.parameters():
    param.requires_grad = False
for param in model.gru.parameters():
    param.requires_grad = False
# Only FC head trainable

# Fine-tune on first half of target bearing
target_train = target_sequence[:, :N//2, :]
target_train_rul = target_rul[:N//2]

optimizer = Adam(model.fc.parameters(), lr=0.001)
for epoch in range(100):
    pred = model(target_train)
    loss = cnn_gru_mha_loss(pred, target_train_rul, model.fc, alpha=1e-4)
    loss.backward()
    optimizer.step()

# Evaluate on second half
with torch.no_grad():
    pred_val = model(target_val)
    rmse = torch.sqrt(F.mse_loss(pred_val, target_val_rul)).item()
```

### C.3: Repeat 5 times

The paper says experiments are repeated 5 times and averaged. Use seeds [42, 123, 456, 789, 1024].

---

## Part D: Run All FEMTO Transfers (45 min)

Run all 12 FEMTO transfers from Table 2:

```
Test 1 (source: 1-3):  → 2-3, 2-4, 3-1, 3-3    (4 transfers)
Test 2 (source: 2-3):  → 1-3, 1-4, 3-3, 3-3    (4 transfers)
Test 3 (source: 3-2):  → 1-3, 1-4, 2-3, 2-4    (4 transfers)
```

For each transfer:
1. Load source bearing, preprocess (DWT + minmax), compute linear RUL labels
2. Load target bearing, preprocess, split 1:1
3. Train on source (60 iterations)
4. Freeze CNN+GRU, fine-tune FC on target train half (100 iterations)
5. Evaluate RMSE on target validation half
6. Repeat 5 seeds, report mean ± std

Save results as JSON:
```python
{
    "source": "Bearing1_3",
    "target": "Bearing2_3",
    "rmse_mean": 0.0463,
    "rmse_std": 0.002,
    "per_seed": [0.045, 0.047, ...],
    "paper_target": 0.0463,
    "predictions": [...],  # from best seed
    "ground_truth": [...]
}
```

---

## Part E: Run XJTU-SY Transfers (30 min)

**IMPORTANT**: Check if XJTU-SY data is available. If not, download it or skip this part and note it as TODO.

XJTU-SY uses a different format:
- Sampling rate: 25.6 kHz
- 32768 samples per snapshot (1.28 seconds)
- Snapshot every ~1 minute

Same preprocessing (DWT + minmax) but on longer snapshots.

Run 4 transfers from Table 5.

---

## Part F: Results Compilation (30 min)

### F.1: FEMTO comparison table

Generate markdown table comparing our replication to paper targets:

```
| Source | Target | Paper RMSE | Our RMSE ± std | Delta |
|--------|--------|:----------:|:--------------:|:-----:|
| 1-3    | 2-3    | 0.0463     | ???            | ???   |
...
| **Avg** |       | **0.0433** | ???            | ???   |
```

### F.2: Per-bearing RUL prediction plots

For each transfer, plot:
- True RUL curve (black line)
- Predicted RUL curve (red line)
- Similar to Figures 10-21 in the paper

Save to `results/plots/`.

### F.3: Loss curves

Plot training loss over iterations for each source domain training. Save to `results/plots/`.

### F.4: Write RESULTS.md

Complete results with:
- FEMTO Table 4 comparison
- XJTU-SY Table 5 comparison (if run)
- Loss curves summary
- Honest assessment: did we replicate the results?

---

## Part G: Test Pipeline (15 min)

Write `test_pipeline.py` that:
1. Loads one bearing (Bearing1_3)
2. Applies DWT + minmax preprocessing
3. Creates CNN-GRU-MHA model
4. Runs 5 training iterations
5. Verifies: no NaN, loss decreases, shapes are correct
6. Prints parameter count

Run this FIRST before any full experiments.

---

## Experiment Ordering

| Part | Task | Est. Time | Depends On |
|------|------|:---------:|:----------:|
| A | Data pipeline + preprocessing | 30 min | — |
| B | Model implementation | 45 min | — |
| G | Test pipeline (run FIRST) | 15 min | A, B |
| C | Training protocol | 30 min | G passes |
| D | FEMTO transfers (12 × 5 seeds) | 45 min | C |
| E | XJTU-SY transfers (4 × 5 seeds) | 30 min | C |
| F | Results compilation | 30 min | D, E |
| | **Total** | **~3.5h** | |

---

## Anti-Patterns to Avoid

1. **Do NOT rewrite FEMTO CSV loading from scratch.** Import from `../dcssl-replication/data_utils.py` or copy the relevant functions.
2. **Do NOT skip the DWT preprocessing.** This is a core part of the paper's method. Install pywt if needed: `pip install PyWavelets`.
3. **Do NOT use piecewise-linear RUL labels.** This paper uses simple linear labels Y_i = (N-i)/N. The DCSSL paper uses piecewise-linear with FPT — different protocol.
4. **Do NOT forget the 1:1 target split.** The target bearing is split chronologically: first half for FC fine-tuning, second half for evaluation.
5. **Do NOT use both channels.** The paper explicitly says "only vibration signals in the horizontal direction are used" (page 8).
6. **Do NOT batch multiple bearings.** Each transfer is one source → one target. Process as batch_size=1 for the GRU (one full bearing sequence).
7. **Embed paper target results in code** for automatic comparison (like DCSSL replication does).
8. **Use PyTorch, not TensorFlow.** The paper used TF 2.5 but we use PyTorch for consistency with the rest of IndustrialJEPA.

---

## What Success Looks Like

**Good replication**: Average FEMTO RMSE within 20% of paper target (0.0443 ± 0.009).
**Exact replication**: Average FEMTO RMSE within 10% of paper target.
**Failed replication**: Document which transfers diverge and hypothesize why (common: different DWT implementation, different RUL label construction, TF vs PyTorch numerical differences).

The value of this replication is having a strong supervised baseline on the EXACT same evaluation protocol, so we can fairly compare JEPA, DCSSL, and CNN-GRU-MHA.

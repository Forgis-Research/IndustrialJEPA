# Autoresearch: JEPA Time Series Forecasting

## Objective

Beat **iTransformer (0.386 MSE)** on ETTh1 benchmark using JEPA.

## Strategy: Focused Innovation

**Primary contribution**: JEPA + VICReg (stable latent prediction)
**Secondary contribution**: Cross-channel attention (unlike channel-independent PatchTST)

We test ONE change at a time. Keep what helps, discard what doesn't.

## Phase 1: Baseline (Current)

Run the baseline first to establish starting point:
```bash
python run.py --single
```

Record: `test_mse = ???`

## Phase 2: Quick Wins

Try these one at a time, in order:

### 2.1 More Training
```python
EPOCHS = 20  # was 10
```
Expected: Better convergence, ~10-20% improvement

### 2.2 Lower Learning Rate
```python
LEARNING_RATE = 1e-4  # was 1e-3
```
Expected: More stable training

### 2.3 Longer Context
```python
SEQ_LEN = 336  # was 96
```
Expected: More information for prediction

## Phase 3: Key Innovation #1 - VICReg

**Why**: Prevents JEPA collapse (from C-JEPA, NeurIPS 2024)

Add to the loss function in `train.py`:

```python
# After computing z_pred (the predicted latent)

# Variance loss: prevent collapse to constant
std = z_pred.std(dim=0)
var_loss = F.relu(1 - std).mean()

# Covariance loss: decorrelate dimensions
z_centered = z_pred - z_pred.mean(dim=0)
cov = (z_centered.T @ z_centered) / (z_pred.shape[0] - 1)
off_diag = cov - torch.diag(torch.diag(cov))
cov_loss = off_diag.pow(2).mean()

# Combined loss
total_loss = mse_loss + 0.04 * var_loss + 0.04 * cov_loss
```

Expected: More stable training, possibly better generalization

## Phase 4: Key Innovation #2 - Cross-Channel Attention

**Why**: iTransformer shows attention across variates helps (ICLR 2024)

Modify the encoder to add cross-channel attention:

```python
# After patch embedding, before temporal attention:
# (B, num_patches, d_model) -> transpose -> (B, num_features, d_model)
# Apply attention across features, then transpose back
```

This captures sensor dependencies that PatchTST misses.

## Phase 5: Architecture Tuning

Only if Phases 2-4 don't reach SOTA:

- `D_MODEL`: 256 → 512
- `E_LAYERS`: 3 → 4 or 6
- `PATCH_LEN`: 16 → 8 or 24
- `USE_JEPA`: True → False (compare to direct prediction)

## Success Criteria

| MSE | Status | Action |
|-----|--------|--------|
| > 0.50 | Poor | Check for bugs |
| 0.45-0.50 | Okay | Continue tuning |
| 0.414-0.45 | Good | Phase 3-4 innovations |
| 0.386-0.414 | **Competitive** | Fine-tune |
| < 0.386 | **SOTA!** | Stop, document |

## Rules

1. **One change at a time** - Know what works
2. **Keep what helps** - Don't revert improvements
3. **Log everything** - Results go to experiment_log.jsonl
4. **5 minutes max** - Fast iteration beats perfect runs
5. **Don't overcomplicate** - Simple + working > complex + broken

## Current Best

| Run | MSE | Changes |
|-----|-----|---------|
| baseline | ??? | Initial config |
| ... | ... | ... |

## References

- iTransformer: 0.386 MSE (our target)
- PatchTST: 0.414 MSE
- C-JEPA: VICReg for JEPA stability
- TS-JEPA: JEPA works for time series

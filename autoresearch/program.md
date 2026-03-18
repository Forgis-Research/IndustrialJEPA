# Autoresearch: JEPA Time Series Forecasting

## Objective

Achieve **state-of-the-art MSE on ETTh1 benchmark** using JEPA architecture.

| Target | MSE |
|--------|-----|
| SOTA (iTransformer) | 0.386 |
| Competitive (PatchTST) | 0.414 |
| **Our Goal** | < 0.386 |

## Dataset

**ETTh1** (Electricity Transformer Temperature)
- 7 variables, hourly data
- Input: 96 timesteps, Output: 96 timesteps
- Standard benchmark for long-term forecasting

## Your Task

1. **Run experiment**: `python run.py --single`
2. **Analyze results**: Check test_mse vs SOTA
3. **Modify train.py**: Try architecture/hyperparameter changes
4. **Repeat**: Until MSE < 0.386 or time runs out

## What You Can Modify

Only modify `train.py`. Key sections:

### Hyperparameters
```python
EPOCHS = 10             # More epochs = better convergence (but slower)
BATCH_SIZE = 32         # Larger = faster, smaller = more stable
LEARNING_RATE = 1e-3    # Try 1e-4 to 5e-3
SEQ_LEN = 96            # Input length (try 192, 336)
PRED_LEN = 96           # Prediction horizon
```

### Architecture
```python
D_MODEL = 256           # Model dimension (128, 256, 512)
N_HEADS = 8             # Attention heads (4, 8, 16)
E_LAYERS = 3            # Encoder depth (2, 3, 4, 6)
D_FF = 512              # FFN dimension (2x or 4x d_model)
DROPOUT = 0.1           # Regularization
```

### JEPA-Specific
```python
USE_JEPA = True         # Toggle JEPA vs direct prediction
LATENT_DIM = 128        # Latent space size
EMA_MOMENTUM = 0.99     # Target encoder momentum (0.99, 0.996, 0.999)
```

### Patch Encoding
```python
USE_PATCHES = True      # Patch-based (like PatchTST)
PATCH_LEN = 16          # Patch length (8, 16, 24)
STRIDE = 8              # Patch stride (patch_len // 2)
```

## Ideas to Try

### Quick Wins
1. Increase EPOCHS to 20 (more convergence)
2. Lower LEARNING_RATE to 1e-4 (more stable)
3. Increase SEQ_LEN to 192 or 336 (more context)

### Architecture Changes
1. Deeper encoder (E_LAYERS = 4 or 6)
2. Wider model (D_MODEL = 512)
3. Different patch sizes (PATCH_LEN = 8 or 24)
4. Disable JEPA (USE_JEPA = False) as baseline

### Advanced Ideas
1. Add cross-channel attention (modify TransformerEncoder)
2. Multi-scale patches (different patch sizes concatenated)
3. Learnable positional encoding
4. Layer normalization before/after attention
5. Separate encoder for each channel (channel-independent)
6. Add JEPA loss to total loss (weighted combination)

## Constraints

- Each run should complete in ~5 minutes
- Keep model parameters < 50M (fits in GPU memory)
- Don't modify prepare.py or run.py

## Success Metrics

| MSE | Status |
|-----|--------|
| > 0.50 | Poor |
| 0.45-0.50 | Okay |
| 0.414-0.45 | Good |
| 0.386-0.414 | **Competitive** |
| < 0.386 | **SOTA!** |

## Example Workflow

```bash
# Run baseline
python run.py --single

# Check result - suppose MSE = 0.52 (poor)
# Hypothesis: need more epochs

# Edit train.py: EPOCHS = 20
python run.py --single

# Check result - suppose MSE = 0.45 (good)
# Hypothesis: need more capacity

# Edit train.py: D_MODEL = 512, E_LAYERS = 4
python run.py --single

# Continue iterating...
```

## Notes

- RevIN is already implemented (handles distribution shift)
- Patch embedding is already implemented (like PatchTST)
- JEPA target encoder uses EMA (like BYOL/I-JEPA)
- Results are logged to experiment_log.jsonl

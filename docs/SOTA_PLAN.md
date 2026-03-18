# IndustrialJEPA SOTA Plan

Long-term plan to achieve state-of-the-art results on time series forecasting benchmarks.

## Goal

Build a JEPA-based foundation model that ranks on the **GIFT-Eval leaderboard**, demonstrating that Joint Embedding Predictive Architectures can compete with or beat current SOTA (Toto, Chronos-2, TimesFM 2.5).

## Current Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Architecture Search | **IN PROGRESS** | 5-min experiments on ETT |
| Phase 2: Benchmark Validation | Pending | Full ETT suite |
| Phase 3: Scale Up | Pending | Pre-train on Time-300B/LOTSA |
| Phase 4: GIFT-Eval Submission | Pending | Official leaderboard |

---

## Phase 1: Architecture Search (Current)

**Timeline**: 1-2 days
**Compute**: A10G (24GB), 5-min experiments
**Dataset**: ETTh1 (7 variables, 17K timesteps)

### Task
Multi-step prediction with horizons: 96, 192, 336, 720 steps

### Metric
MSE (Mean Squared Error) - standard for ETT benchmark

### SOTA Baselines (ETTh1, horizon 96)

| Model | MSE | MAE |
|-------|-----|-----|
| iTransformer | 0.386 | 0.405 |
| PatchTST | 0.414 | 0.419 |
| DLinear | 0.456 | 0.452 |
| Autoformer | 0.496 | 0.487 |

### Our Target
- **Competitive**: MSE < 0.45
- **Strong**: MSE < 0.40
- **SOTA**: MSE < 0.386

### Architecture Ideas to Try
- [ ] Multi-step JEPA predictor (predict 8, 16, 32 steps)
- [ ] Cross-channel attention (iTransformer style)
- [ ] Patch-based encoding (PatchTST style)
- [ ] RevIN normalization (handle distribution shift)
- [ ] Varying predictor depths
- [ ] Different masking strategies

---

## Phase 2: Benchmark Validation

**Timeline**: 1-2 days
**Compute**: A10G, full training runs (hours)

### Datasets
Full ETT suite:
- ETTh1, ETTh2 (hourly, 7 vars)
- ETTm1, ETTm2 (15-min, 7 vars)

Additional classic benchmarks:
- Electricity (321 vars)
- Weather (21 vars)
- Traffic (862 vars)

### Success Criteria
Beat published baselines on at least 3/4 ETT datasets.

---

## Phase 3: Scale Up

**Timeline**: 1-2 weeks
**Compute**: Multi-GPU (scale up credits)

### Pre-training Data Options

| Dataset | Size | Source |
|---------|------|--------|
| Time-300B | 309B datapoints | [HuggingFace](https://huggingface.co/datasets/Maple728/Time-300B) |
| LOTSA | 27B datapoints | [Salesforce](https://huggingface.co/datasets/Salesforce/lotsa_data) |
| GIFT-Eval Pretrain | 230B datapoints | Non-leaking corpus |

### Model Scaling

| Size | Parameters | Fits A10G? |
|------|------------|------------|
| Small | 10-50M | Yes |
| Medium | 100-200M | Yes (grad checkpointing) |
| Large | 500M-1B | Need multi-GPU |

---

## Phase 4: GIFT-Eval Submission

**Timeline**: 1 week after Phase 3
**Goal**: Top-10 on leaderboard

### Evaluation
- 28 datasets, 97 configurations
- Metrics: MASE, CRPS
- Zero-shot evaluation

### Current Leaderboard (as of 2025)

| Rank | Model | Avg Rank |
|------|-------|----------|
| 1 | Toto | 5.495 |
| 2 | Chronos-2 | ~7 |
| 3 | TimesFM 2.5 | ~8 |
| 4 | Moirai-MoE | ~10 |

### Our Differentiator
JEPA architecture with:
- True cross-channel modeling (current weakness of SOTA)
- Multi-step latent prediction (not autoregressive)
- Efficient inference (single forward pass)

---

## Key Challenges to Address

### 1. Varying Scales
Different variables have vastly different magnitudes (temperature vs accelerometer).

**Solutions**:
- RevIN (Reversible Instance Normalization)
- Per-channel normalization
- Learnable scale parameters

### 2. Rare Events
Anomalies/failures are extremely rare (1 in millions).

**Solutions**:
- Forecasting-based detection (predict normal, flag deviations)
- Synthetic anomaly injection during training
- Asymmetric loss functions

### 3. Distribution Shift
Statistics change over time (non-stationary).

**Solutions**:
- RevIN
- Adaptive normalization
- Test-time adaptation

### 4. Channel Dependencies
Current SOTA treats channels independently (missed opportunity).

**Solutions**:
- Cross-channel attention (iTransformer)
- Graph neural networks for spatial structure
- JEPA with multi-channel latent space

---

## Resources

### Benchmarks
- [GIFT-Eval Leaderboard](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
- [GIFT-Eval GitHub](https://github.com/SalesforceAIResearch/gift-eval)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

### Datasets
- [Time-300B](https://huggingface.co/datasets/Maple728/Time-300B)
- [LOTSA](https://huggingface.co/datasets/Salesforce/lotsa_data)
- [ETDataset](https://github.com/zhouhaoyi/ETDataset)

### Papers
- [iTransformer](https://arxiv.org/abs/2310.06625) - Attention across variates
- [PatchTST](https://arxiv.org/abs/2211.14730) - Patch-based transformer
- [Time-MoE](https://arxiv.org/abs/2409.16040) - Foundation model on Time-300B
- [Moirai](https://arxiv.org/abs/2402.02592) - Universal forecasting transformer

### Compute
- **Current**: SageMaker A10G (24GB)
- **Scale up**: Multi-GPU available (credits)

---

## Autoresearch Setup

### Quick Start (SageMaker)
```bash
# From local Windows
aws login --profile sagemaker
ssh sagemaker-space

# On SageMaker
cd ~/IndustrialJEPA/autoresearch
export ANTHROPIC_API_KEY="..."
claude "Follow program.md to improve val_loss on ETT benchmark"
```

### Monitor Progress
```bash
# Second SSH session
ssh sagemaker-space
cd ~/IndustrialJEPA/autoresearch
python run.py --leaderboard
tail -f experiment_log.jsonl
```

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-18 | Target GIFT-Eval | Most competitive leaderboard, active community |
| 2026-03-18 | Start with ETT | Small, well-benchmarked, fast iteration |
| 2026-03-18 | Multi-step prediction | Harder task, more meaningful than 1-step |
| 2026-03-18 | Use JEPA architecture | Novel approach, potential for cross-channel modeling |

---

## Next Actions

1. [ ] Update autoresearch to use ETTh1 dataset
2. [ ] Implement multi-step prediction (horizons 96, 192, 336, 720)
3. [ ] Add MSE metric comparable to published results
4. [ ] Run overnight autoresearch
5. [ ] Review results, identify best architecture
6. [ ] Scale to full ETT suite

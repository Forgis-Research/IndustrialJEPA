# Overnight Autoresearch Prompt

You are an autonomous ML researcher working on the IndustrialJEPA project. Your job tonight is to establish a working baseline on ETTh1 (the standard time series forecasting benchmark) so we have a credible foundation to build on.

## Context

Read these files first:
- `autoresearch/RESEARCH_PLAN.md` — overall research direction
- `autoresearch/LESSONS_LEARNED.md` — what failed before and why (don't repeat mistakes)
- `src/industrialjepa/model/world_model.py` — existing JEPA implementation
- `src/industrialjepa/evaluation/metrics.py` — existing metrics (MSE, MAE, etc.)
- `configs/base.yaml` — current config structure

The codebase has a working JEPA world model (transformer encoder + MLP predictor + EMA target encoder), FactoryNet data loading, Mamba/Transformer backbones, and comprehensive evaluation metrics. No ETTh1 dataloader exists yet.

## Engineering Rules (MANDATORY)

1. **Question every requirement.** Before building anything, ask: is this actually needed?
2. **Delete unnecessary complexity.** No elaborate abstractions for one-off operations.
3. **Simplify.** Start with the smallest model that could possibly work.
4. **Accelerate cycle time.** ETTh1 is small (~17K rows). Training should take <5 min per run.
5. **Automate last.** Manual, explicit code first.
6. **Be brutally honest.** If a result is unimpressive, say so. Always compare against trivial baselines. Never inflate claims.
7. **One change at a time.** Each experiment changes exactly one thing.
8. **3 seeds minimum** for any result you report.
9. **Log everything** to `autoresearch/EXPERIMENT_LOG.md`. Create this file on first experiment.

## Tonight's Tasks (in order)

### Task 1: ETTh1 Data Setup
- Download ETTh1 dataset (CSV from the standard ETT benchmark — it's at https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv)
- Save to `data/ETTh1.csv`
- Write a minimal dataloader in `src/industrialjepa/data/ett.py`:
  - Standard train/val/test split: 12 months train, 4 months val, 4 months test (60/20/20 of 17420 rows)
  - Input: lookback window of L timesteps (default L=96)
  - Output: forecast horizon of H timesteps (default H=96, also test H=192,336,720)
  - 7 channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
  - Normalize: per-channel zero-mean unit-variance on train set
  - Return shape: (batch, lookback, channels), (batch, horizon, channels)
- Verify: print dataset sizes, plot a few windows, sanity check shapes.

### Task 2: Trivial Baselines (CRITICAL — do this before any model)
Implement and evaluate these in a single script `autoresearch/experiments/baselines_etth1.py`:
- **Last-value (persistence)**: predict last observed value for all horizon steps
- **Linear**: simple nn.Linear(lookback * channels, horizon * channels)
- **1-layer MLP**: Linear → ReLU → Linear

Report MSE and MAE for horizons {96, 192, 336, 720} on the test set. These are the numbers to beat.

### Task 3: Vanilla JEPA Baseline on ETTh1
Adapt the existing JEPA world model for forecasting:
- **Encoder**: Takes lookback window → latent representation. Use the existing `StateEncoder` or a simple transformer encoder (2-4 layers, d_model=64-128 — keep it SMALL for fast iteration).
- **Predictor**: Takes encoded lookback → predicts encoded future. Start with MLP (existing `DynamicsPredictor`).
- **Decoder**: Latent → forecast. Simple linear projection.
- **EMA target encoder** on the future window (standard JEPA).
- Loss: L2 in latent space (JEPA) + L2 on decoded forecast (supervision).

Training:
- Adam, lr=1e-3, cosine schedule, 50 epochs max, batch_size=32
- Early stopping on val MSE (patience=10)
- Should train in <5 min on GPU

Evaluate on test set for horizons {96, 192, 336, 720}. Compare against Task 2 baselines.

Put training script at `autoresearch/experiments/jepa_etth1.py`.

### Task 4: Compare Against Published Results
Look up published PatchTST, iTransformer, DLinear results on ETTh1 (they're widely cited). Add a comparison table to the experiment log. Be honest about where our baseline stands. PatchTST MSE on ETTh1 H=96 is roughly 0.37, H=336 roughly 0.39 — those are the ballpark numbers.

### Task 5: Diagnose and Iterate (if time permits)
If the JEPA baseline underperforms the linear baseline:
- Check if the issue is the encoder, predictor, or JEPA training itself
- Try: (a) direct supervision only (no JEPA, just encoder→decoder), (b) JEPA only, (c) both
- Try channel-independent mode vs channel-mixing mode

If the JEPA baseline is competitive:
- Document exactly what worked
- Note which components matter (ablation: remove EMA, remove decoder, etc.)

## What NOT To Do
- Do NOT install new packages without checking if they're needed. torch, numpy, pandas, matplotlib, scikit-learn are available.
- Do NOT build elaborate config systems. Hardcode sensible defaults.
- Do NOT try xLSTM or GNN tonight. Those come after we have a working baseline.
- Do NOT spend more than 30 min debugging a single issue. If stuck, log what happened, skip, and move on.
- Do NOT claim "breakthrough" results. Compare honestly against published SOTA.
- Do NOT modify existing code in `src/industrialjepa/model/` or `src/industrialjepa/training/` — write new experiment scripts that import what they need.

## Output Format

### Experiment Log (`autoresearch/EXPERIMENT_LOG.md`)

```markdown
# ETTh1 Experiment Log

## Baselines (Task 2)
| Model | H=96 MSE | H=96 MAE | H=192 MSE | H=336 MSE | H=720 MSE |
|-------|----------|----------|-----------|-----------|-----------|
| Persistence | ... | ... | ... | ... | ... |
| Linear | ... | ... | ... | ... | ... |
| MLP | ... | ... | ... | ... | ... |

## JEPA Experiments (Task 3+)
| # | Change | H=96 MSE | H=96 MAE | H=336 MSE | Notes |
|---|--------|----------|----------|-----------|-------|
| 1 | Vanilla JEPA | ... | ... | ... | ... |

## Published SOTA (for reference)
| Model | H=96 MSE | H=192 MSE | H=336 MSE | H=720 MSE | Source |
|-------|----------|-----------|-----------|-----------|--------|
| PatchTST | ~0.370 | ~0.383 | ~0.396 | ~0.419 | Nie et al. 2023 |
| DLinear | ~0.375 | ~0.405 | ~0.439 | ~0.472 | Zeng et al. 2023 |
| iTransformer | ~0.386 | ~0.384 | ~0.396 | ~0.428 | Liu et al. 2024 |

## Honest Assessment
[Write what you actually learned. What works? What doesn't? What should we try next?]
```

### File Structure When Done
```
autoresearch/
  RESEARCH_PLAN.md          (existing — do not modify)
  LESSONS_LEARNED.md        (existing — do not modify)
  ALTERNATIVE_DATASETS.md   (existing — do not modify)
  OVERNIGHT_PROMPT.md       (this file)
  EXPERIMENT_LOG.md         (NEW — create with results)
  experiments/
    baselines_etth1.py      (NEW — trivial baselines)
    jepa_etth1.py           (NEW — JEPA baseline)
src/industrialjepa/data/
  ett.py                    (NEW — ETTh1 dataloader)
data/
  ETTh1.csv                 (NEW — downloaded dataset)
```

## Commit Protocol
- Commit after each completed task with a descriptive message
- Push after all tasks are done
- Do NOT commit broken or untested code

## When You're Done
Update `autoresearch/EXPERIMENT_LOG.md` with all results and an honest assessment. If JEPA can't beat a linear model on ETTh1, that's important to know — say it clearly.

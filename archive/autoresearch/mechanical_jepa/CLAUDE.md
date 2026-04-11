# Mechanical-JEPA: Bearing Fault Detection

Self-supervised learning for industrial bearing fault detection using JEPA (Joint Embedding Predictive Architecture).

## Quick Start

```bash
# 1. Setup (download CWRU dataset)
cd mechanical-jepa
bash setup_vm.sh

# 2. Verify installation
python train.py --epochs 5 --no-wandb

# 3. Full training
python train.py --epochs 30
```

## Key Files

| File | Purpose |
|------|---------|
| `program.md` | Research plan - READ THIS FIRST |
| `EXPERIMENT_LOG.md` | Results log - UPDATE AFTER EACH EXPERIMENT |
| `LESSONS_LEARNED.md` | Insights - UPDATE WHEN YOU LEARN SOMETHING |

## Core Hypothesis

JEPA learns transferable features for bearing fault detection by predicting masked patch embeddings. This is analogous to Brain-JEPA (NeurIPS 2024) for fMRI.

## Datasets

| Dataset | Status | Size | Use |
|---------|--------|------|-----|
| **CWRU** | ✅ Ready | 134MB | Classification benchmark |
| **IMS** | ✅ Ready | 6GB | Run-to-failure (RUL) |
| Paderborn | ⚠️ RAR files | 5GB | Multi-modal (future) |

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| JEPA Test Acc | >40% | 49.8% ✓ |
| vs Random Init | +5% | +19.8% ✓ |
| Multi-seed | 3+ seeds | 1 (TODO) |

## Commands

```bash
# Training
python train.py --epochs 30                # Default
python train.py --epochs 100 --seed 42     # Longer, fixed seed
python train.py --encoder-depth 6          # Deeper model
python train.py --mask-ratio 0.7           # More masking
python train.py --no-wandb                 # Without logging

# Evaluation
python train.py --eval-only --checkpoint checkpoints/jepa_xxx.pt
```

## Anti-Patterns

1. **Split by bearing** - NOT by window (prevents leakage)
2. **Multiple seeds** - 3+ for any claim
3. **Log everything** - Even failures are information

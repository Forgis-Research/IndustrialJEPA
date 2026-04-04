---
name: IndustrialJEPA Project Context
description: Mechanical-JEPA V6 COMPLETE: JEPA transfer gain +0.371 (2.6x better than supervised Transformer +0.144). JEPA@N=10 > Transformer@N=all on Paderborn. All key claims p<0.05, large effect sizes.
type: project
---

IndustrialJEPA is a research project on self-supervised learning (JEPA) for industrial time series.

## Mechanical-JEPA: Bearing Fault Detection

**Status as of 2026-04-04 (V6 complete)**

### CORRECTED V6 Results (JSON-backed, 3-seed, fixed Paderborn API bug)

| Method | CWRU F1 | Paderborn F1 | Transfer Gain | Source |
|--------|---------|-------------|---------------|--------|
| CNN Supervised | 1.000 ± 0.000 | 0.987 ± 0.005 | +0.457 ± 0.020 | transfer_baselines_v6_final.json |
| **JEPA V2 (ours)** | 0.773 ± 0.018 | **0.900 ± 0.008** | **+0.371 ± 0.026** | transfer_baselines_v6_final.json |
| Transformer Supervised | 0.969 ± 0.026 | 0.673 ± 0.063 | +0.144 ± 0.044 | transfer_baselines_v6_final.json |
| MAE (reconstruct) | 0.643 ± 0.144 | ~0.587 | ~+0.001 | baselines_comparison.json |
| JEPA V3 (SIGReg, no EMA) | 0.531 ± 0.008 | 0.540 ± 0.025 | +0.193 | log only |
| Random Init | ~0.412 | 0.529 ± 0.024 | 0.000 | transfer_baselines_v6_final.json |

**IMPORTANT**: V5 numbers were wrong due to Paderborn API bug (PaderbornDataset constructor mismatch).
Corrected: JEPA V2 Paderborn F1 is 0.900 (not 0.795), gain is +0.371 (not +0.453).
The qualitative finding is unchanged: JEPA > Transformer supervised for transfer.

### Few-Shot Transfer Curves (fewshot_curves.json)

| Method | N=10 | N=20 | N=50 | N=100 | N=all |
|--------|------|------|------|-------|-------|
| CNN Supervised | 0.989 | 0.992 | 0.989 | 0.990 | 0.990 |
| **JEPA V2** | **0.735** | 0.779 | 0.853 | 0.878 | **0.903** |
| Transformer Sup. | 0.510 | 0.545 | 0.609 | 0.638 | 0.689 |
| Random Init | 0.383 | 0.385 | 0.413 | 0.418 | 0.538 |

**KEY FINDING: JEPA@N=10 (0.735) > Transformer@N=all (0.689)**
p=0.034, Cohen's d=0.92 (large). This is the primary publishable figure.

### Statistical Significance

All key claims: p < 0.05, large Cohen's d (>0.9). See `mechanical-jepa/statistical_tests.py`.
- JEPA gain > 0: t=20.3, p=0.0012, d=11.73
- JEPA gain > Transformer gain: t=5.36, p=0.017, d=5.11
- JEPA@N=10 > Transformer@N=all: t=1.96, p=0.034, d=0.92

With n=3 seeds, p-values are approximate. For camera-ready: run 10 seeds for key claims.

### SF-JEPA Tradeoff (sfjepa_comparison.json)

No sweet spot: spectral auxiliary losses improve in-domain but hurt transfer.
| lambda_spec | CWRU F1 | Paderborn F1 | Transfer Gain |
|-------------|---------|-------------|---------------|
| 0.0 (pure JEPA) | 0.773 | 0.900 | +0.371 |
| 0.1 | 0.863 | 0.825 | +0.319 |
| 0.5 | 0.905 | 0.818 | +0.312 |

Finding: pure JEPA always maximizes transfer. SF-JEPA is useful only when in-domain accuracy > transfer.

### Cross-Component Transfer (partial, seed 42 only)

- CWRU bearings → MCC5-THU gearboxes: Gear-pretrained JEPA CWRU F1=0.506 (below random 0.542)
- Multi-source CWRU+Gear: CWRU F1=0.526 (still below random)
- Root cause: bearing impulses vs gear tooth-mesh modulation are physically distinct

### JEPA V2 Architecture (jepa_v2.py)

- Encoder: 4-layer Transformer, d=512, 4 heads, sinusoidal PE
- Input: (B, 3, 4096) → 16 patches of 256 samples
- Mask ratio: 0.625 (10 of 16 patches masked)
- EMA target encoder (momentum=0.996)
- Loss: L1 + variance regularization (lambda=0.1)
- Training: 100 epochs, AdamW lr=1e-4, cosine schedule
- Params: ~5.1M

**5 critical components** (all required, verified by ablation):
1. Sinusoidal PE in predictor (not learnable)
2. High mask ratio 0.625
3. L1 loss (not MSE)
4. Variance regularization lambda=0.1
5. EMA target encoder (not stop-gradient)

### Best Checkpoint

`mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt` (seed=123)
- Used as reference in multi-source and SF-JEPA experiments

### Key Files

- Training: `mechanical-jepa/train_v2.py` (main)
- Transfer audit: `mechanical-jepa/run_transfer_audit.py` (generates V6 JSON)
- Paderborn: `mechanical-jepa/paderborn_transfer.py` (use `create_paderborn_loaders`)
- Few-shot curves: `mechanical-jepa/fewshot_transfer_curves.py`
- SF-JEPA: `mechanical-jepa/train_sfjepa_fast.py` (use torch FFT, not scipy)
- Multi-source: `mechanical-jepa/multi_source_hf_pretrain.py`
- Stats tests: `mechanical-jepa/statistical_tests.py`
- Paper outline: `mechanical-jepa/PAPER_OUTLINE.md`
- Publication notebook: `mechanical-jepa/notebooks/06_v6_walkthrough.ipynb`
- Figures: `mechanical-jepa/notebooks/plots/fig1-6.{pdf,png}`
- Results: `mechanical-jepa/results/transfer_baselines_v6_final.json`, `fewshot_curves.json`, `sfjepa_comparison.json`
- Full results doc: `mechanical-jepa/CONSOLIDATED_RESULTS.md`

### Critical Bug Fixed in V6

`transfer_baselines.py` was calling `PaderbornDataset(root_dir=..., ...)` but the class expects
`bearing_dirs: list` (list of (path, label) tuples). Fixed by using `create_paderborn_loaders`.
All prior Paderborn F1 values were null/bogus. V6 has correct numbers.

### Datasets

- CWRU: `mechanical-jepa/data/bearings/` (134MB, 40 bearings, 4 fault classes, 12kHz)
- IMS raw: `/mnt/sagemaker-nvme/ims_raw/` (symlinked from `data/bearings/ims_raw/`)
  - Files: TAB-DELIMITED TEXT, load with `np.loadtxt(fpath)` NOT `np.fromfile()`
- Paderborn: `datasets/data/paderborn/` (K001/KA01/KI01, 64kHz → resample to 20kHz)
- HF gearboxes: `Forgis/Mechanical-Components`, token `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`
  - Use `pd.read_parquet('hf://...', storage_options={'token': TOKEN})` (not load_dataset)

### Disk Space

- Home disk: ~10GB free after cleanup
- NVMe: `/mnt/sagemaker-nvme/` (~200GB), used for IMS raw data and training logs
- If disk fills: delete `mechanical-jepa/__pycache__/` and `mechanical-jepa/wandb/`

**Why:** V6 was needed to fix the Paderborn API bug, get correct transfer numbers, and run few-shot curves.
**How to apply:** Use results from `transfer_baselines_v6_final.json` and `fewshot_curves.json` for any paper numbers.

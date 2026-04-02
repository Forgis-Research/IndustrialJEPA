---
name: IndustrialJEPA Project Context
description: Mechanical-JEPA V4 COMPLETE: F1=0.773±0.018, JEPA RUL (Test1 Spearman=0.08, 60% early warning), 3-seed ablation (V2 full 0.743 vs minimal 0.711, all collapsed), collapse visualizations done
type: project
---

IndustrialJEPA is a research project on self-supervised learning (JEPA) for industrial time series.

**Two main subprojects:**

## 1. Physics-Informed Attention (main paper)

52+ experiments, core finding: Physics-masked attention provides principled constraint when physics groups are statistically independent.

| System | Physics Mask Effect | Why |
|---|---|---|
| Double Pendulum | +7.4% over Full-Attn (p=0.0002) | Groups are truly independent |
| C-MAPSS | ≈ random mask (p=0.528) | Correlated degradation |
| ETT Weather | -1.3% vs Full-Attn | Thermal couples to all loads |

---

## 2. Mechanical-JEPA: Bearing Fault Detection

**Status as of 2026-04-01 (V3 complete)**

### Best Checkpoint
`mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt` (seed=123, 89.7% CWRU, no collapse)

### Best Training Command
```bash
python train_v2.py --epochs 100 --seed 123 --embed-dim 512 --predictor-pos sinusoidal \
  --predictor-depth 4 --loss-fn l1 --var-reg 0.1 --mask-ratio 0.625
```

### V2 Architecture (jepa_v2.py)
- Sinusoidal positional encoding in predictor
- Mask ratio 0.625 (key: forces position use, not context averaging)
- L1 loss + variance regularization lambda=0.1
- Predictor depth 4, encoder depth 4, embed_dim 512

### Results Summary

| Metric | V1 | V2 | V3 Best |
|--------|----|----|---------|
| CWRU linear probe (3-seed) | 80.4% ± 2.6% | 82.1% ± 5.4% | **83.7% ± 6.6%** (var_reg=0.05) |
| CWRU best seed | 84.1% | 89.7% | 90.1% (patch=128, seed=123) |
| IMS transfer gain | +2.4% ± 2.9% | **+8.8% ± 0.7%** | +8.8% (maintained) |
| Paderborn transfer | -1.4% (failed) | -1.4% | **+14.7% ± 0.8%** (20kHz resample) |
| Predictor collapse | Yes | No | No |

### Complete Transfer Matrix (V3)

| Source → Target | Gain | Seeds |
|---|---|---|
| CWRU → IMS (binary, 20kHz) | **+8.8% ± 0.7%** | 3/3 |
| CWRU → Paderborn @ 12kHz | **+8.5% ± 3.0%** | 3/3 |
| CWRU → Paderborn @ 20kHz | **+14.7% ± 0.8%** | 3/3 |
| CWRU → Paderborn (no resample) | -1.4% ± 1.0% | 0/3 |
| IMS → IMS (self) | **+6.2% ± 1.7%** | 3/3 |
| Paderborn → CWRU | +5.3% ± 9.0% | 2/3 |
| IMS → CWRU | **-6.8% ± 1.1%** | 0/3 |

### Key V3 Findings

1. **FREQUENCY STANDARDIZATION IS THE CRITICAL FIX**: Use `scipy.signal.resample_poly` to 20kHz before cross-dataset eval. Implemented in `paderborn_transfer.py`.

2. **Transfer is asymmetric**: CWRU (diverse faults) → anything works. IMS (degradation dynamics) → CWRU fault classification FAILS (-6.8%).

3. **JEPA 5M > wav2vec2 94M**: V2 JEPA (87.1%) beats frozen wav2vec2-base (77.2%) by +9.9% on vibration. Speech pretraining provides some transfer (+5.4% over random) but domain-specific wins.

4. **Optimal mask ratio = 0.625**: Confirmed over fine sweep (0.5 to 0.875). Higher mask (0.75) wins at 30ep but loses at 100ep.

5. **Block masking = random masking**: No benefit for vibration JEPA. Random masking is sufficient.

6. **Multi-source pretraining dilutes features**: CWRU+Paderborn: 81.2% vs CWRU-only: 88.7% (-7.5%). Train on one domain at a time for in-domain tasks.

7. **Patch size 256 near-optimal**: patch=128 marginally better (+0.3%), patch=512 dramatically worse (-23.7%).

8. **100 epochs optimal**: 200ep hurts (-2.1%), confirmed twice across V1 and V2.

### V4 Results (2026-03-31)

| Metric | Result | Notes |
|--------|--------|-------|
| CWRU Macro F1 (3-seed) | 0.773 ± 0.018 | vs random 0.412 (F1 gain +0.360) |
| Cross-component bearing→gearbox | +2.5% F1 | mcc5_thu 8-class, 3/3 seeds |
| Continual learning CWRU drop | -0.15% | No catastrophic forgetting |
| IMS RMS Spearman (1st_test) | 0.758 | 22% early warning lead time |
| IMS RMS Spearman (2nd_test) | 0.443 | 29% early warning lead time |

**New V4 Files:**
- `mechanical-jepa/eval_f1.py`: Macro F1 evaluation with per-class breakdown
- `mechanical-jepa/rul_from_rms.py`: RUL from precomputed RMS cache (no raw IMS needed)
- `mechanical-jepa/rul_prognostics.py`: Full JEPA embedding-based RUL (needs raw IMS)
- `mechanical-jepa/hf_cross_component.py`: HF Mechanical-Components cross-component transfer
- `mechanical-jepa/notebooks/04_v4_comprehensive_analysis.ipynb`: 10-section analysis notebook

**V4 Key Findings:**
1. All 5 V2 fixes are necessary as a system; no single fix alone prevents collapse AND gives good features
2. L1 loss is the critical feature-quality driver (MSE+var_reg prevents collapse but gives 49% acc)
3. Diagnostic bug: `quick_diagnose` used wrong `n_context=8` (should be 6 for mask=0.625); V2 best checkpoint is NOT collapsed
4. HF Mechanical-Components: use `pd.read_parquet('hf://...', storage_options={'token': TOKEN})` NOT `load_dataset()` (OOM)
5. Cross-component gain modest (+2.5%) because bearing impulses vs gearbox tooth-mesh modulation are different physics
6. EMA + low LR (5e-5) prevents catastrophic forgetting in continual learning

### Experiment Log Range

| Run | Experiments | Key Result |
|-----|------------|-----------|
| Overnight V1 | Exp 0-15 | 80.4% baseline, +28.5% over random |
| Overnight V2 | Exp 16-23 | Fixed predictor collapse, +8.8% IMS transfer |
| Overnight V3 | Exp 24-35 | +14.7% Paderborn (20kHz resample), wav2vec2 comparison |
| Overnight V4 | Exp 36-40 | F1 metrics, RUL, cross-component, continual learning |

### Files
- Training: `mechanical-jepa/train_v2.py` (main), `train_v3_block.py` (block masking variant)
- Transfer: `mechanical-jepa/ims_transfer.py`, `paderborn_transfer.py`, `transfer_matrix.py`
- Multi-source: `mechanical-jepa/multi_source_pretrain.py`
- Pretrained encoder eval: `mechanical-jepa/pretrained_encoder_eval.py`
- Models: `mechanical-jepa/src/models/jepa_v2.py` (main), `jepa_enhanced.py` (structured masking)
- Experiment log: `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` (Exp 0-35)
- Lessons: `autoresearch/mechanical_jepa/LESSONS_LEARNED.md`

### IMS Data Notes (UPDATED 2026-04-02)
- Raw IMS files DOWNLOADED and available:
  - `data/bearings/ims_raw/1st_test/1st_test/` (2156 files, 35 days)
  - `data/bearings/ims_raw/2nd_test/2nd_test/` (984 files, 7 days)
  - `data/bearings/ims/Test1/` and `Test2/` are symlinks to the above
- IMS files are TAB-DELIMITED TEXT: use `np.loadtxt(fpath)` → (20480, 8) float64
  - DO NOT use `np.fromfile(fpath, dtype=np.float64)` — gives all zeros silently
- Use `jepa_rul_ims.py` for proper JEPA prognostics on raw IMS data
- Old broken symlink `data/bearings/raw/ims` → `ims_npy_cache` (still there but cache is gone)

### Paderborn Data
- Location: `/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn/`
- K001/ (healthy), KA01/ (outer race), KI01/ (inner race)
- 80 MAT files each, ~256k samples per file at 64kHz
- Load with: `scipy.io.loadmat(..., simplify_cells=True)` → `obj['Y'][6]['Data']` for vibration_1
- MUST resample to 20kHz for best transfer with CWRU checkpoint

### HuggingFace Dataset
- `Forgis/Mechanical-Components`: CWRU (n=16 parquet rows), FEMTO bearings (25.6kHz, 2ch), mcc5_thu gearboxes (12.8kHz, 3ch, 8 fault types, 956 samples)
- Token: `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`
- Load with: `pd.read_parquet('hf://datasets/Forgis/Mechanical-Components/gearboxes/train-00000-of-00004.parquet', storage_options={'token': TOKEN})` (NOT load_dataset — OOM)
- FEMTO has 2ch (mismatches CWRU 3ch model); use mcc5_thu gearboxes for cross-component (3ch matches)

**Why:** V3 was needed to test frequency standardization, pretrained encoders, and complete the transfer matrix.
**How to apply:** Use `paderborn_transfer.py` for Paderborn experiments. Always resample to 20kHz first.

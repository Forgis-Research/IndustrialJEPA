# Session Handoff: Mechanical-JEPA Research

## Date: 2026-04-01

## What's Running Right Now

An **ml-researcher agent** is running overnight (V4 prompt). It was launched from this
session and will continue running even after account switch. Check its output at:
```
/tmp/claude-1000/-home-sagemaker-user-IndustrialJEPA/be1045f3-5a60-4d70-8422-9c926dd4451f/tasks/a7e71b7ea30fb7349.output
```

**V4 agent is working on (9 rounds):**
1. Deep literature review + SUCCESS_METRICS.md definition
2. Architecture simplification (find minimal collapse fix)
3. Switch to F1-score metric
4. RUL prediction / prognostics on IMS
5. HF Mechanical-Components dataset (cross-component transfer)
6. Predictor collapse deep understanding + visualisations
7. Online / continual learning experiments
8. Comprehensive Jupyter notebook (04_v4_comprehensive_analysis.ipynb)
9. Documentation + commit/push

## Project State

### Architecture
- **V2 is current best** — `src/models/jepa_v2.py`, trained via `train_v2.py`
- Encoder: 4-layer transformer, embed_dim=512, patch_size=256, 16 patches/window (~5M params)
- Predictor: 4-layer transformer, predictor_dim=256, sinusoidal pos encoding
- Loss: L1 on L2-normalized predictions + variance reg (λ=0.1)
- mask_ratio=0.625

### Results (V1 → V2 → V3)

| Metric | V1 | V2 | V3 finding |
|--------|----|----|------------|
| CWRU linear probe (3-seed) | 80.4% ± 2.6% | 82.1% ± 5.4% | — |
| CWRU best seed | 84.1% | 89.7% | — |
| IMS binary transfer | +2.4% ± 2.9% | +8.8% ± 0.7% | — |
| Paderborn transfer (no resample) | N/A | -1.4% | — |
| Paderborn transfer @20kHz | N/A | N/A | **+14.7% ± 0.8%** |
| Predictor collapse | Yes | Fixed | — |
| Transfer efficiency | 70% | 142% | — |
| wav2vec2 (94M) vs JEPA (5M) | — | — | 77.2% vs 87.1% |
| Block masking | — | — | No benefit |
| Multi-source pretrain | — | — | Hurts (-7.5%) |

### Key Findings
1. **Predictor collapse root cause**: mask_ratio=0.5 allows context-average shortcut
2. **Fix**: high mask ratio (0.625) is primary lever; sinusoidal pos, L1, var_reg help
3. **Frequency standardisation**: resampling to common rate enables Paderborn transfer
4. **Transfer is asymmetric**: CWRU→anything works; IMS→CWRU is negative
5. **Domain-specific small model beats large pretrained**: 5M JEPA > 94M wav2vec2

### Experiments Completed
- **Exp 0-35** logged in `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md`
- V4 overnight will continue from Exp 36+

### Key Files

| File | Purpose |
|------|---------|
| `mechanical-jepa/src/models/jepa_v2.py` | V2 model (current best) |
| `mechanical-jepa/train_v2.py` | V2 training script |
| `mechanical-jepa/ims_transfer.py` | Cross-dataset transfer eval |
| `mechanical-jepa/paderborn_transfer.py` | Paderborn transfer (with resampling) |
| `mechanical-jepa/quick_diagnose.py` | Predictor collapse diagnostic |
| `mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt` | Best V2 checkpoint (seed=123) |
| `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` | All 35 experiments |
| `autoresearch/mechanical_jepa/LESSONS_LEARNED.md` | Critical insights |
| `autoresearch/mechanical_jepa/OVERNIGHT_PROMPT_V4.md` | Current overnight prompt |
| `autoresearch/LITERATURE_REVIEW.md` | JEPA SOTA review |
| `jepa-lit-review/jepa_sota_review.md` | C-JEPA + ThinkJEPA analysis |
| `notebooks/03_results_analysis.ipynb` | Results notebook (V2 sections 18-22) |

### Git State
- Branch: `main`, up to date with origin
- Last commit: `daf8528` — Overnight prompt v4
- Working tree: clean (before V4 agent started making changes)

### Environment
- GPU: NVIDIA A10G
- Disk: ~20GB free
- Python env: conda, with torch, wandb, scipy, sklearn, huggingface datasets
- WandB project: `mechanical-jepa`
- HF token in `.env`: `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`

### Data
- CWRU: `mechanical-jepa/data/bearings/raw/cwru/` (ready)
- IMS: `mechanical-jepa/data/bearings/raw/ims/` (raw exists, npy cache cleared)
- Paderborn: `/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn/` (3 bearings)
- HF dataset: `Forgis/Mechanical-Components` (bearings + gearboxes, on HuggingFace)

## How to Resume

```bash
# 1. Check if V4 agent is still running or finished
cat /tmp/claude-1000/-home-sagemaker-user-IndustrialJEPA/be1045f3-5a60-4d70-8422-9c926dd4451f/tasks/a7e71b7ea30fb7349.output | tail -100

# 2. Check git status (agent may have committed new work)
cd /home/sagemaker-user/IndustrialJEPA
git log --oneline -10
git status

# 3. Check experiment log for new entries
grep "^### Exp" autoresearch/mechanical_jepa/EXPERIMENT_LOG.md | tail -10

# 4. Check if notebook was created
ls mechanical-jepa/notebooks/04_v4_comprehensive_analysis.ipynb

# 5. Check disk space
df -h /home/sagemaker-user
```

## What to Do Next (after V4 completes)

1. **Review V4 results** — read experiment log, check notebook, review wandb
2. **Verify claims** — re-run any suspicious results with different seeds
3. **Write paper draft** — we have enough results for a workshop paper:
   - Predictor collapse analysis (novel contribution)
   - Cross-dataset transfer with frequency standardisation
   - Comparison to pretrained models (wav2vec2)
   - RUL/prognostics (if V4 delivers)
   - Cross-component transfer (if V4 delivers)
4. **Scale up** — if HF dataset works, pretrain on ALL available data
5. **Deploy** — package as a pip-installable library with pretrained weights

## Overnight Prompts (for reference)

| Version | File | Focus |
|---------|------|-------|
| V2 | `autoresearch/mechanical_jepa/OVERNIGHT_PROMPT.md` | Fix predictor collapse, IMS transfer |
| V3 | (inline in conversation) | Frequency standardisation, pretrained encoders, transfer matrix |
| V4 | `autoresearch/mechanical_jepa/OVERNIGHT_PROMPT_V4.md` | RUL, HF dataset, simplification, F1 metrics |

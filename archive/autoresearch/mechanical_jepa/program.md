# Mechanical-JEPA: Bearing Fault Detection

**Goal:** Prove that JEPA (Joint Embedding Predictive Architecture) learns transferable features for industrial bearing fault detection, analogous to Brain-JEPA's success on fMRI data.

**Success metric:** Linear probe on JEPA embeddings > Random init embeddings + 5%

---

## Context

Brain-JEPA (NeurIPS 2024 Spotlight) showed that JEPA learns useful features from fMRI time series for brain age prediction and cognitive tasks. We adapt this approach to vibration signals from industrial bearings.

**Key insight:** Self-supervised learning on masked patch prediction forces the encoder to learn semantic features (fault signatures) rather than low-level statistics.

---

## Phase 1: Establish Baselines (First 2 hours)

### 1.1 Verify Current Implementation

```bash
cd mechanical-jepa
python train.py --epochs 5 --no-wandb  # Quick smoke test
```

**Checks:**
- [ ] Training runs without errors
- [ ] Loss decreases
- [ ] Linear probe runs

### 1.2 Baselines to Establish

| Baseline | Description | Expected |
|----------|-------------|----------|
| Random Guessing | 4-class uniform | 25% |
| Random Init | Linear probe on untrained encoder | 25-35% |
| JEPA (30 epochs) | Current implementation | ~50% |
| Supervised | End-to-end with labels | 80-95% (ceiling) |

### 1.3 Multi-seed Validation

Run current best (30 epochs) with 3 seeds:
```bash
python train.py --epochs 30 --seed 42
python train.py --epochs 30 --seed 123
python train.py --epochs 30 --seed 456
```

**Report:** mean ± std for test accuracy

---

## Phase 2: Systematic Improvements (Hours 2-6)

### 2.1 Training Duration

| Experiment | Epochs | Hypothesis |
|------------|--------|------------|
| Exp A1 | 50 | More training helps |
| Exp A2 | 100 | Diminishing returns? |
| Exp A3 | 200 | Overfitting check |

### 2.2 Model Architecture

| Experiment | Change | Hypothesis |
|------------|--------|------------|
| Exp B1 | encoder_depth=6 | More capacity |
| Exp B2 | embed_dim=512 | Larger embeddings |
| Exp B3 | predictor_depth=4 | Stronger predictor |

### 2.3 Masking Strategy

| Experiment | Mask Ratio | Hypothesis |
|------------|------------|------------|
| Exp C1 | 0.3 | Less masking = easier task |
| Exp C2 | 0.7 | More masking = harder, better features? |
| Exp C3 | Block masking | Temporal structure |

### 2.4 Patch Size

| Experiment | Patch Size | Patches | Hypothesis |
|------------|------------|---------|------------|
| Exp D1 | 128 | 32 | Finer granularity |
| Exp D2 | 512 | 8 | Coarser, captures full cycles |

### 2.5 Data

| Experiment | Data | Hypothesis |
|------------|------|------------|
| Exp E1 | CWRU + IMS | More data helps |
| Exp E2 | IMS only (RUL) | Different task |

---

## Phase 3: Analysis & Documentation (Hours 6+)

### 3.1 Ablation Study

For the best configuration, ablate:
1. Encoder depth (4 vs 2 vs 6)
2. Mask ratio (0.3 vs 0.5 vs 0.7)
3. EMA decay (0.99 vs 0.996 vs 0.999)

### 3.2 Visualization

1. **t-SNE by fault type** - Do fault types cluster?
2. **t-SNE by bearing** - Features should NOT cluster by bearing (generalization)
3. **Loss curves** - Learning dynamics
4. **Attention maps** - What does the model attend to?

### 3.3 Final Notebook

Create `notebooks/03_results_analysis.ipynb` with:
- Clear explanation of JEPA for bearings
- All baseline comparisons (table)
- Best result with 5 seeds (mean ± std)
- t-SNE visualizations
- Confusion matrix
- Conclusions and future work

---

## Success Criteria

### Required (MUST achieve):
- [ ] JEPA test accuracy > 40% (vs ~30% random init)
- [ ] Improvement consistent across 3+ seeds
- [ ] Clear t-SNE clustering by fault type

### Stretch (NICE to have):
- [ ] JEPA test accuracy > 60%
- [ ] Cross-bearing generalization (test within 10% of train)
- [ ] Meaningful attention patterns

---

## Code Locations

```
mechanical-jepa/
├── train.py                    # Main training script
├── setup_vm.sh                 # VM setup (dataset download)
├── requirements.txt            # Dependencies
├── src/
│   ├── data/bearing_dataset.py # Data loader (stratified split)
│   └── models/jepa.py          # JEPA model
└── notebooks/
    ├── 01_bearing_faults_analysis.ipynb  # Data exploration
    ├── 02_jepa_training.ipynb            # Training walkthrough
    └── 03_results_analysis.ipynb         # Final results (create this)
```

---

## Commands Reference

```bash
# VM Setup
cd mechanical-jepa
bash setup_vm.sh           # Quick (CWRU only)
bash setup_vm.sh --full    # Full (CWRU + IMS)

# Training
python train.py --epochs 30                    # Default
python train.py --epochs 100 --seed 42         # Longer, fixed seed
python train.py --epochs 100 --no-wandb        # Without logging
python train.py --encoder-depth 6              # Deeper encoder
python train.py --mask-ratio 0.7               # More masking

# Evaluation only
python train.py --eval-only --checkpoint checkpoints/jepa_xxx.pt
```

---

## Current Best Result

| Metric | Value | Seeds |
|--------|-------|-------|
| JEPA Test Accuracy | 49.8% | 1 |
| Random Init Accuracy | ~30% | 1 |
| Improvement | +19.8% | - |

**Status:** Needs multi-seed validation

---

## Anti-Patterns to Avoid

1. **Data leakage** - Split by bearing_id, not by window
2. **Single seed** - Always run 3+ seeds for claims
3. **Test set tuning** - Don't tune hyperparams on test
4. **Comparing unfairly** - Same split for all methods
5. **Ignoring failures** - Log negative results too

---

## Overnight Checklist

Before starting:
- [ ] `setup_vm.sh` completed successfully
- [ ] Smoke test passed (`python train.py --epochs 5`)
- [ ] WandB logging works (or `--no-wandb` if not)
- [ ] Git is clean, changes committed

During run:
- [ ] Log each experiment to EXPERIMENT_LOG.md
- [ ] Commit after each successful improvement
- [ ] Push every 5 experiments

Stopping conditions:
- [ ] All Phase 1-3 experiments complete
- [ ] Beat 60% accuracy
- [ ] Run out of ideas
- [ ] Hit irrecoverable error

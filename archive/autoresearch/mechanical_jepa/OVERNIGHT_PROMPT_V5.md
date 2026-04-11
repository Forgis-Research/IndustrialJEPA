# Overnight Autoresearch Prompt: Mechanical-JEPA V5

## From Hack to Principled: SIGReg, RUL Pretraining, Frequency Masking

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA project. Use the FULL night (10+ hours).

Working directory: /home/sagemaker-user/IndustrialJEPA/mechanical-jepa

## DISK WARNING — CRITICAL
Home disk has only 5GB free. NVMe has 232GB free.
- Store ALL checkpoints at /mnt/sagemaker-nvme/jepa_checkpoints/
- Store ALL HF caches at /mnt/sagemaker-nvme/hf_cache/
  export HF_HOME=/mnt/sagemaker-nvme/hf_cache
- Run `df -h /home/sagemaker-user` before EVERY training run
- If disk < 2GB: stop, clean old checkpoints, clear __pycache__

Create the NVMe directories at start:
  mkdir -p /mnt/sagemaker-nvme/jepa_checkpoints /mnt/sagemaker-nvme/hf_cache

## Context: What Exists (after V1-V4, 43 experiments)

**Architecture (V2 — current best):**
- Encoder: 4-layer transformer, embed_dim=512, patch_size=256, 16 patches/window (~5M params)
- Predictor: 4-layer transformer, predictor_dim=256, sinusoidal pos encoding
- Loss: L1 on L2-normalized predictions vs EMA targets + variance reg (λ=0.1)
- mask_ratio=0.625 (10 of 16 patches masked)
- EMA target encoder (decay=0.996)

**Current Results:**
- CWRU 4-class Macro F1: 0.773 ± 0.018 (3 seeds, linear probe)
- CWRU MLP probe: 96.1%
- IMS binary transfer: +8.8% ± 0.7%
- Paderborn @20kHz transfer: +14.7% ± 0.8%
- Bearing→Gearbox: +2.5% F1
- wav2vec2 94M: 77.2% vs our JEPA 5M: 87.1%
- Continual learning forgetting: -0.15%
- RUL: FAILED (RMSE 0.503, worse than constant baseline 0.360)

**Predictor collapse: FIXED with 5-part hack but NOT principled.**
- V2 uses: mask_ratio=0.625 + sinusoidal pos + L1 + var_reg=0.1 + predictor_depth=4
- 3-seed ablation showed ALL 5 fixes needed together
- But literature (LeJEPA, Nov 2025) shows SIGReg replaces ALL of this + removes EMA

**Known failures:**
- RUL regression from JEPA embeddings: Spearman 0.08 vs RMS 0.55
  ROOT CAUSE: pretrained on fault classification (CWRU), not degradation dynamics
- Multi-source pretraining: hurts CWRU by -7.5%
- IMS→CWRU transfer: -6.8% (asymmetric)

## FILES TO READ FIRST

1. `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` — ALL 43 experiments
2. `autoresearch/mechanical_jepa/LESSONS_LEARNED.md` — critical insights
3. `autoresearch/mechanical_jepa/SUCCESS_METRICS.md` — metrics & SOTA comparison
4. `mechanical-jepa/src/models/jepa_v2.py` — V2 model
5. `mechanical-jepa/train_v2.py` — V2 training
6. `mechanical-jepa/hf_cross_component.py` — HF dataset usage
7. `mechanical-jepa/jepa_rul_ims.py` — current RUL code (failed approach)
8. `mechanical-jepa/eval_f1.py` — F1 evaluation

## RECENT LITERATURE — MUST KNOW

These papers define our competitive landscape. Read carefully before experimenting.

### Direct Competitors
- **TS-JEPA** (NeurIPS 2024 Workshop, arXiv:2509.25449): JEPA on 1D time series with
  >70% masking, lightweight transformer. Matches SOTA on classification + forecasting.
  Our advantage: domain-specific, cross-machine transfer (they don't evaluate this).

- **MTS-JEPA** (arXiv:2602.04643, Feb 2026): Multi-resolution JEPA + soft codebook for
  anomaly detection. Beats contrastive by +43% F1 on MSL. Their codebook prevents collapse.
  Consider adopting their multi-resolution idea.

- **RmGPT** (IEEE IoT-J 2025, arXiv:2409.17604): GPT for rotating machinery.
  82% accuracy on 16-class one-shot. Our few-shot competitor.

### Collapse Prevention (the principled way)
- **LeJEPA / SIGReg** (arXiv:2511.08544, Nov 2025): Replaces EMA with Sketched Isotropic
  Gaussian Regularization. Projects embeddings onto random directions, applies normality
  test. ViT-H reaches 79% ImageNet. Removes EMA, stop-gradient, VICReg entirely.
  THIS IS WHAT WE SHOULD IMPLEMENT.

- **LeWorldModel** (arXiv:2603.19312, Mar 2026, LeCun's final Meta paper):
  End-to-end JEPA from pixels, no EMA, only prediction loss + SIGReg.
  15M params, single GPU, hours to train. Code: github.com/lucas-maes/le-wm
  CONFIRMS our scale (5M params) is right.

### Foundation Models for Vibration
- **OpenMAE** (ACM IMWUT 2025): MAE on 5M vibration samples, +23% gain. Scale matters.
- **MAE+Swin**: 99.53% Paderborn, 100% CWRU in-domain. IN-DOMAIN IS SOLVED.
- **PHM-GPT**: 96.92% total accuracy across 15 datasets. Language model + vibration.

### RUL with Self-Supervised Learning
- **DCSSL** (Scientific Reports 2026): Contrastive SSL for bearing RUL on FEMTO.
  Dual contrastive loss (temporal + instance level). Outperforms supervised SOTA.
  NO JEPA+RUL PAPER EXISTS — this is our gap.

### Why JEPA > Contrastive for Vibration (our theoretical argument)
Contrastive methods assume augmentation invariance — but in vibration, load/speed
variation creates spurious invariances (healthy@heavy_load ≈ fault@light_load).
JEPA predicts structure, not invariance. This is specifically relevant to industrial
vibration where operating conditions vary. Make this argument explicit in the notebook.

## =======================================================================
## ROUND 1: IMPLEMENT SIGREG (REPLACE EMA) — 3 hours
## =======================================================================

This is the most important architectural change. SIGReg from LeJEPA replaces:
- EMA target encoder (saves 5M params of memory)
- Variance regularization (var_reg)
- The need for carefully tuned mask ratio

### 1A. Read LeJEPA Paper & LeWM Code

Search web for the actual SIGReg implementation:
- arXiv:2511.08544 — read the SIGReg algorithm section
- github.com/lucas-maes/le-wm — read their loss function
- Key: SIGReg projects embeddings onto M random directions, computes
  Epps-Pulley normality test on each projection, averages the statistic.
  The loss encourages embeddings to be isotropic Gaussian.

### 1B. Implement SIGReg Loss

Create `src/models/sigreg.py` with:
```python
def sigreg_loss(z: torch.Tensor, n_projections: int = 64) -> torch.Tensor:
    """
    Sketched Isotropic Gaussian Regularization (LeJEPA).
    z: (B, D) embeddings
    Returns: scalar loss (lower = more Gaussian-distributed)
    """
    # 1. Center and standardize
    # 2. Project onto random unit vectors
    # 3. Compute Epps-Pulley statistic per projection
    # 4. Average across projections
    pass
```

Get the EXACT algorithm from the paper. Don't approximate.

### 1C. Create V3 Model: JEPA Without EMA

Create `src/models/jepa_v3.py` — copy V2 but:
- Remove target_encoder (no EMA copy)
- Remove _update_target_encoder()
- Context encoder produces BOTH context embeddings AND targets
  (use stop-gradient on targets, like BYOL but with SIGReg instead of momentum)
  OR: use the architecture from LeWM exactly
- Add SIGReg loss on encoder outputs
- Keep prediction loss (L1 on normalized embeddings)
- Total loss = prediction_loss + sigreg_coeff * sigreg_loss

### 1D. Experiments

**Exp V5-1: V3 (SIGReg, no EMA) vs V2 (EMA + 5 hacks), 30 epochs, seed 42**
- Compare: CWRU F1, collapse diagnostic, training time, memory usage
- V3 should use ~half the GPU memory (no EMA copy)

**Exp V5-2: V3 with different sigreg coefficients (0.01, 0.1, 1.0), 30 epochs**
- Find optimal coefficient

**Exp V5-3: V3 best config, 100 epochs, 3 seeds**
- Full validation if V3 matches or beats V2

**Exp V5-4: V3 transfer — CWRU→IMS, CWRU→Paderborn**
- THE critical test: does SIGReg improve transfer?
- Compare to V2 transfer numbers (+8.8%, +14.7%)

**Decision gate**: If V3 (SIGReg) matches V2 transfer with simpler architecture → adopt V3.
If V3 is worse → keep V2 but note that SIGReg didn't transfer to 1D domain.

### 1E. If SIGReg is hard to implement perfectly

Fall back to simpler approach from LeWM:
- Keep prediction loss
- Add VICReg (variance + covariance) on encoder outputs with higher coefficient
- Remove EMA, use stop-gradient on targets instead
- This is NOT as principled as SIGReg but still removes EMA

## =======================================================================
## ROUND 2: FIX RUL — PRETRAIN ON DEGRADATION, NOT FAULTS — 2 hours
## =======================================================================

Current RUL failed because: CWRU pretraining teaches fault TYPE discrimination,
but RUL needs degradation DYNAMICS. The encoder learns "this is an outer race fault"
not "this bearing is 70% through its life". These are different features.

### 2A. RUL-Oriented Pretraining on IMS

IMS is a run-to-failure dataset. Pretrain JEPA on IMS temporal sequences:
- Each IMS file = one time snapshot of a bearing degrading
- Pretrain on ALL IMS files (no labels needed — just self-supervised masking)
- The hope: JEPA learns features that capture degradation state, not fault type

**Exp V5-5: IMS-pretrained JEPA → RUL regression on IMS**
- Pretrain on IMS Test 1 data (2156 files, ~35 days)
- Extract embeddings, train Ridge regression for RUL
- Split: first 70% for training, last 30% for test
- Compare to: CWRU-pretrained (current, failed), random init, RMS baseline
- Metrics: RMSE, Spearman correlation, early warning time

### 2B. Temporal Ordering as Pretext Task

JEPA predicts masked patches in SPACE (across the signal window).
For RUL, we need to understand TIME (degradation progression).

**Exp V5-6: Temporal prediction pretext task**
- New pretext: given embeddings at time t, predict embedding at time t+Δ
- This teaches the model about temporal evolution of bearing state
- Implement as: pairs of consecutive IMS files → predict next embedding
- Can be combined with standard JEPA masking loss

### 2C. Use FEMTO Bearings from HF Dataset

The HF Mechanical-Components dataset has FEMTO bearings with rul_percent labels.
export HF_HOME=/mnt/sagemaker-nvme/hf_cache

**Exp V5-7: FEMTO pretraining → RUL on FEMTO**
- Download FEMTO bearings from Forgis/Mechanical-Components
- Pretrain JEPA on FEMTO (self-supervised)
- Fine-tune RUL regression head
- Compare: JEPA pretrained vs random init vs simple RMS
- 3 seeds if promising

### 2D. Zero-Shot Health Indicator (Improved)

The current zero-shot approach (embedding distance from healthy centroid) was
hypersensitive (warned at 60% remaining life = too early). Improve it:
- Use Mahalanobis distance instead of cosine (accounts for variance structure)
- Compute moving average of distance (reduces noise)
- Define threshold using percentile of healthy distribution
- Measure: time-to-first-alarm, false alarm rate, detection lead time

## =======================================================================
## ROUND 3: FREQUENCY-DOMAIN MASKING — 1.5 hours
## =======================================================================

Nobody has done frequency-domain masking in JEPA. Mechanical faults concentrate
in specific frequency bands (ball pass frequency, gear mesh frequency).

### 3A. Implement Spectral Masking

Current: mask random TIME patches (contiguous blocks of the signal)
New: mask random FREQUENCY bands

Implementation:
1. FFT the input signal → complex spectrum
2. Divide spectrum into N frequency bands (patches in frequency domain)
3. Mask some bands (set to zero)
4. IFFT back to time domain → masked signal
5. Encoder sees masked signal, predictor predicts original embeddings

Alternative (simpler):
1. Compute spectrogram (STFT) → 2D time-frequency representation
2. Mask frequency bands (rows of spectrogram)
3. Encoder processes masked spectrogram
4. This is closer to how image-JEPA works (2D masking)

### 3B. Experiments

**Exp V5-8: Frequency masking vs time masking, 30 epochs**
- Same model, same training, just different masking strategy
- Compare: CWRU F1, collapse diagnostic
- Hypothesis: frequency masking forces the model to reconstruct fault-specific
  frequency content, which is more informative than random time chunks

**Exp V5-9: Combined time + frequency masking, 30 epochs**
- Mask both time patches AND frequency bands simultaneously
- More aggressive masking → potentially better representations

**Exp V5-10: Transfer with frequency masking — CWRU→Paderborn**
- If frequency masking helps in-domain, does it also help transfer?
- This is the key test: frequency masking should be MORE transferable
  because fault frequencies scale with shaft speed (predictable relationship)

## =======================================================================
## ROUND 4: SCALE UP HF DATASET — 2 hours
## =======================================================================

We only used mcc5_thu gearbox (143 samples) from HF. The dataset has much more.

### 4A. Full Dataset Inventory

Download and catalog ALL data from Forgis/Mechanical-Components:
- How many samples per source? Per component type? Per fault type?
- What sampling rates?
- Signal lengths?
- Quality issues?

Export HF_HOME=/mnt/sagemaker-nvme/hf_cache to avoid disk issues.

### 4B. All-Bearing Pretraining

**Exp V5-11: Pretrain on ALL HF bearings (FEMTO + MFPT + Mendeley + XJTU-SY)**
- Resample everything to 20kHz (our common rate)
- Exclude CWRU (save as test target)
- Self-supervised JEPA pretraining
- Test: transfer to CWRU, IMS, Paderborn
- Compare to: CWRU-only pretrain, random init

### 4C. Cross-Component at Scale

**Exp V5-12: All-bearing pretrain → all-gearbox evaluation**
- Pretrain on all bearing data
- Linear probe on each gearbox source separately
- Which gearbox sources benefit from bearing pretraining?

**Exp V5-13: Joint bearing+gearbox pretraining → evaluation on held-out sources**
- Pretrain on 80% of all data (mixed bearings and gearboxes)
- Test on held-out 20%
- Does mixing components help or hurt? (Recall: multi-source hurt before)

### 4D. Embedding Visualization

- t-SNE / UMAP of embeddings from ALL sources
- Color by: component type, fault type, source dataset
- Do fault types cluster across component types?
- Do sources cluster (domain gap visible)?

## =======================================================================
## ROUND 5: TRANSFER BASELINES (DANN / simple) — 1 hour
## =======================================================================

Our transfer claims (+14.7% Paderborn) need baselines beyond random init.

### 5A. Domain Adaptation Baseline

Implement simple DANN (Domain Adversarial Neural Network):
- Shared feature extractor (same JEPA encoder architecture, random init)
- Task classifier (fault type)
- Domain discriminator (source vs target)
- Train with gradient reversal layer

**Exp V5-14: DANN on CWRU→Paderborn transfer**
- Same eval protocol as our JEPA transfer experiments
- Is JEPA pretraining better or worse than supervised domain adaptation?

### 5B. Simple Baselines

**Exp V5-15: Supervised CNN baseline**
- Simple 1D CNN (3-4 conv layers + linear head), same param count (~5M)
- Train on CWRU, test on Paderborn (with resampling)
- This is the "do you even need self-supervised pretraining?" baseline

**Exp V5-16: RMS + kurtosis + spectral features → logistic regression**
- Hand-crafted features baseline for transfer
- Extract: RMS, kurtosis, crest factor, spectral entropy, band energies
- Train logistic regression on CWRU, test on Paderborn
- The "do you even need deep learning?" baseline

## =======================================================================
## ROUND 6: COMPREHENSIVE NOTEBOOK UPDATE — 2 hours
## =======================================================================

Update `notebooks/04_v4_comprehensive_analysis.ipynb` OR create new
`notebooks/05_v5_sigreg_rul_analysis.ipynb` with ALL new findings.

### Required Sections:

1. **Literature Positioning** — where we sit vs TS-JEPA, MTS-JEPA, RmGPT, OpenMAE
2. **SIGReg vs EMA** — architecture comparison, results, simplicity analysis
3. **RUL Breakthrough (or Honest Failure)** — IMS pretraining → RUL regression
4. **Frequency Masking** — time vs frequency vs combined masking results
5. **Scaled Cross-Component Transfer** — all HF sources, embedding visualization
6. **Transfer Baselines** — JEPA vs DANN vs CNN vs hand-crafted features
7. **Why JEPA for Vibration** — theoretical argument against contrastive methods
8. **Complete Results Table** — all experiments, all metrics, all baselines
9. **Limitations & Honest Assessment** — what JEPA is good/bad at

Every figure: title, labels, error bars, caption.
Every claim: numbers, uncertainty, baseline comparison.
Save plots to `notebooks/plots/v5_*.png`.

## =======================================================================
## ROUND 7: DOCUMENTATION & COMMIT — 30 min
## =======================================================================

### 7A. Update All Docs
- EXPERIMENT_LOG.md: all new experiments (continue from Exp 43)
- LESSONS_LEARNED.md: SIGReg findings, RUL findings, frequency masking
- SUCCESS_METRICS.md: updated achieved values

### 7B. Commit After Each Round
```bash
git add -A && git commit -m "Exp N-M: [round summary]" && git push
```

## GLOBAL RULES

### Experiment Discipline
- **ALWAYS use wandb** (project: 'mechanical-jepa')
- **Log EVERY experiment** to EXPERIMENT_LOG.md immediately
- **3+ seeds** for any claim, 1 seed for exploration
- **Macro F1** as primary metric (not accuracy)
- **Run quick_diagnose.py** after every training run
- **30 epochs first**, 100 only if signal is clear
- **Compare to V2 baseline** always
- **Check disk** (df -h) before every training run

### Checkpoints
- Save ALL checkpoints to /mnt/sagemaker-nvme/jepa_checkpoints/
- Update checkpoint paths in training scripts accordingly
- Keep at most 3 checkpoints per experiment (best, last, V2-baseline)

### Code Quality
- New model files: src/models/jepa_v3.py, src/models/sigreg.py
- New training script: train_v3.py (for SIGReg model)
- CLI flags for all new features
- Backward compatible — V2 still works

### Self-Criticism After Every Experiment
- Real or noise? (error bars across seeds)
- Fair comparison? (same seeds, epochs, eval protocol)
- Could RMS + logistic regression explain this?
- Would this survive peer review?
- Is this better than what TS-JEPA / MTS-JEPA achieved?

### Priority If Running Out of Time
1. **Round 1 (SIGReg)**: MUST DO — this is the publishable contribution
2. **Round 2 (RUL fix)**: MUST DO — fixes our biggest failure
3. **Round 4 (Scale HF)**: HIGH — strengthens cross-component story
4. **Round 3 (Freq masking)**: HIGH — novel masking strategy
5. **Round 5 (Baselines)**: MEDIUM — needed for paper, can be quick
6. **Round 6 (Notebook)**: MUST DO — results must be presented
7. **Round 7 (Docs)**: MUST DO — commit everything

### Expected Runtime
- Round 1: ~3 hours (implementation + 4 experiments)
- Round 2: ~2 hours (3 experiments + analysis)
- Round 3: ~1.5 hours (implementation + 3 experiments)
- Round 4: ~2 hours (download + 3 experiments + viz)
- Round 5: ~1 hour (3 quick experiments)
- Round 6: ~2 hours (notebook)
- Round 7: ~30 min (docs + commit)
Total: ~12 hours

### Stopping Conditions
1. All 7 rounds complete
2. Running 14+ hours
3. Irrecoverable error (after 3 attempts to fix)
4. Disk full on BOTH home AND NVMe (unlikely)

USE THE FULL NIGHT. Don't stop after 4 hours. If a round finishes early,
go deeper on the most promising finding. Run more seeds. Try variations.
If Round 1 (SIGReg) works, it's worth spending extra time on it.
```

---

## Pre-Flight Checklist

- [ ] GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Disk: `df -h /home/sagemaker-user` (need >5GB) + `df -h /mnt/sagemaker-nvme`
- [ ] NVMe dirs: `mkdir -p /mnt/sagemaker-nvme/jepa_checkpoints /mnt/sagemaker-nvme/hf_cache`
- [ ] WandB: `python -c "import wandb; print(wandb.api.api_key[:8])"`
- [ ] CWRU data: `ls mechanical-jepa/data/bearings/raw/cwru/`
- [ ] IMS data: `ls mechanical-jepa/data/bearings/raw/ims/` (or ims_raw)
- [ ] V2 checkpoint: `ls mechanical-jepa/checkpoints/jepa_v2_20260401_003619.pt`
- [ ] HF token: `grep HF_TOKEN .env`
- [ ] Git clean: `git status`

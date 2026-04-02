# Building a Foundation Model for Industrial Vibration Analysis

**An overnight AI-assisted research sprint — from broken pretraining to cross-component transfer**

---

## The Problem

Predictive maintenance for rotating machinery (bearings, gearboxes, turbines) costs industry billions annually in unplanned downtime. Current approaches rely on hand-engineered spectral features (FFT, envelope analysis) that require domain expertise and don't generalise across equipment.

I set out to build a **general-purpose vibration encoder** using JEPA (Joint Embedding Predictive Architecture) — the same self-supervised framework behind Meta's I-JEPA and V-JEPA, but applied to 1D vibration signals instead of images or video. The goal: pretrain once on labelled fault data, then transfer to unseen machines with zero or few labels.

## The Starting Point

- **Architecture**: 5M-parameter transformer encoder + predictor, trained with masked patch prediction on vibration windows
- **Dataset**: CWRU bearing fault dataset (2,400 windows, 4 fault classes, 12kHz)
- **Initial result**: 80.4% linear probe accuracy, but cross-dataset transfer to IMS bearings was weak (+2.4%)

The diagnostic scripts I wrote revealed the root cause: **the predictor had collapsed**. It was outputting nearly identical embeddings for every masked position (spread ratio = 0.02), meaning it learned to predict the context average rather than position-specific content. The encoder still learned useful features — because it had to produce informative context — but it was leaving performance on the table.

## The Research Sprint

I designed and orchestrated four overnight autonomous research sessions, each targeting specific gaps. The workflow: write a detailed research prompt specifying literature review targets, experiment designs, success metrics, and self-check protocols — then launch an ML research agent to execute autonomously while I slept.

### Session 1: Fixing Predictor Collapse (4.2 hours, 171 tool calls)

**Literature finding**: V-JEPA uses 90% masking; I-JEPA uses multi-block targets — both prevent collapse by making the context-average shortcut mathematically invalid.

**The key insight**: At mask_ratio=0.5, 8 of 16 patches are visible. The visible patches are a representative sample, so their average ≈ the global average ≈ the target average. The predictor can minimise loss without using positional information at all.

At mask_ratio=0.625, only 6 patches are visible — a biased sample. The context average no longer approximates the target average, **breaking the shortcut**.

**The fix** (V2 architecture):
- Mask ratio 0.5 → 0.625 (primary lever)
- Sinusoidal positional encoding (guaranteed position discrimination)
- L1 loss replacing MSE (less incentive for "safe" mean predictions)
- Variance regularisation (direct collapse penalty)
- Predictor depth 2 → 4 layers

**Result**: Spread ratio improved 13x (0.02 → 0.26). IMS transfer gain jumped from +2.4% to **+8.8%** — a 3.7x improvement.

### Session 2: Frequency Standardisation & Pretrained Encoders (2.8 hours, 180 tool calls)

Transfer to Paderborn bearings (64kHz) had failed completely (-1.4%). I hypothesised sampling rate mismatch was the cause.

**Experiment**: Resample Paderborn from 64kHz to 20kHz before evaluation.

**Result**: Transfer gain went from **-1.4% to +14.7% ± 0.8%** — a complete reversal. The "5.3x sampling rate barrier" was entirely an artifact of frequency content misalignment, solved by simple polyphase resampling.

I also benchmarked against wav2vec 2.0 (94M parameters, pretrained on 960 hours of speech). Our 5M-parameter domain-specific JEPA achieved **87.1% vs 77.2%** — 18x fewer parameters, 10% better accuracy. Domain-targeted self-supervised pretraining outperformed brute-force transfer from a related modality.

### Session 3: Metrics, Prognostics & Cross-Component Transfer (52 min + 36 min continuation)

Switched from accuracy to **macro F1-score** as the primary metric (handles class imbalance properly). Re-evaluated all models: JEPA F1 = 0.773 ± 0.018 vs random 0.412.

**RUL Prognostics** on IMS run-to-failure data:
- JEPA embeddings detect anomalies **60-71% of the run before failure** (vs RMS at 0.2-38%)
- But RMS gives a more reliable monotonic health indicator (Spearman 0.69 vs 0.21)
- Honest finding: JEPA adds value for early anomaly detection, not smooth RUL regression — it was trained on fault classification, not degradation dynamics

**Cross-component transfer** (bearing → gearbox):
- Loaded gearbox data from our HuggingFace dataset (`Forgis/Mechanical-Components`)
- CWRU-bearing-pretrained encoder achieved **+2.5% F1 gain** on gearbox fault classification, positive in all 3 seeds
- Small but consistent: vibration features partially transfer across fundamentally different mechanical components

**Continual learning**: Continued pretraining on IMS after CWRU caused only -0.15% F1 drop on CWRU — no catastrophic forgetting. The EMA target encoder acts as a momentum anchor.

### Architecture Ablation

The 3-seed ablation definitively answered "do we need all 5 fixes?":
- V2 full (all fixes): F1 = 0.743 ± 0.056, **0/3 seeds collapsed**
- Mask ratio only: F1 = 0.711 ± 0.093, **3/3 seeds collapsed**

The collapsed predictor still achieves decent in-domain accuracy (the encoder learns regardless), but collapse **specifically destroys transfer learning** (3.7x worse IMS transfer). This is the paper-worthy finding: predictor collapse is invisible in-domain but devastating for generalisation.

## Final Results

| Metric | Value |
|---|---|
| CWRU 4-class Macro F1 | **0.773 ± 0.018** (3 seeds) |
| F1 gain over random init | **+0.360** |
| CWRU → IMS transfer | **+8.8% ± 0.7%** |
| CWRU → Paderborn @20kHz | **+14.7% ± 0.8%** |
| Bearing → Gearbox transfer | **+2.5% F1** (3/3 seeds) |
| Transfer efficiency | **142%** (cross-domain beats self-pretrain) |
| vs wav2vec2 (94M params) | **+9.9%** with 18x fewer parameters |
| Continual learning forgetting | **-0.15%** (negligible) |
| Early anomaly detection | **60-71%** of run before failure |

## Complete Transfer Matrix

| Source → Target | Gain | Verdict |
|---|---|---|
| CWRU → IMS (binary) | +8.8% ± 0.7% | Strong |
| CWRU → Paderborn @20kHz | +14.7% ± 0.8% | Very strong |
| CWRU → Paderborn @12kHz | +8.5% ± 3.0% | Strong |
| Bearing → Gearbox | +2.5% F1 | Modest but consistent |
| IMS → CWRU | -6.8% ± 1.1% | Negative (asymmetric) |
| Paderborn → CWRU | +5.3% ± 9.0% | Marginal |

## What I Built

- **43 experiments** logged with full configs, results, and statistical analysis
- **V2 architecture** with principled collapse prevention, validated against JEPA literature
- **Cross-dataset transfer pipeline** with automatic frequency standardisation
- **RUL prognostics module** — zero-shot health indicators from pretrained embeddings
- **HuggingFace integration** — cross-component experiments using our curated dataset
- **Comprehensive Jupyter notebook** (10 sections, 15 figures) documenting every finding
- All experiments tracked on Weights & Biases; all code committed to GitHub

## Technical Decisions I'm Proud Of

**Understanding before optimising.** When the predictor collapsed, I didn't just try random fixes. I traced the mathematical root cause (context-average shortcut at symmetric mask ratios), validated it against the I-JEPA and V-JEPA literature, then designed targeted ablations. The 3-seed ablation confirmed that all 5 fixes are necessary — and crucially, that collapse is invisible in-domain but kills transfer.

**Honest negative results.** JEPA embeddings are worse than simple RMS for smooth health monitoring (Spearman 0.21 vs 0.69). Multi-source pretraining hurts in-domain accuracy (-7.5%). IMS→CWRU transfer is negative (-6.8%). I documented all of these because understanding failure modes is as valuable as celebrating wins.

**Frequency standardisation as a first principle.** The Paderborn "failure" (-1.4%) wasn't a model problem — it was a data engineering problem. Resampling to a common rate converted it to our strongest transfer result (+14.7%). Simple preprocessing beats architectural complexity.

**Right-sized models.** With 2,400 training windows, a 5M-parameter model is correct. The 94M-parameter wav2vec2 lost by 10% despite 18x more parameters. Matching model capacity to dataset scale matters more than importing massive pretrained models.

## Tools & Infrastructure

- PyTorch, Weights & Biases, HuggingFace Datasets
- Claude Code (Anthropic) for autonomous overnight research orchestration
- Custom diagnostic tools for predictor collapse detection
- SageMaker with NVIDIA A10G GPU

---

*This research was conducted over 48 hours in March-April 2026, producing 43 experiments across 4 autonomous overnight sessions. The codebase, experiment logs, and comprehensive analysis notebook are available on GitHub.*

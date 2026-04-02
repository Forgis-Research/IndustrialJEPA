# Claude Code Session: Autonomous Overnight ML Research

**Project**: Mechanical-JEPA — Self-supervised learning for industrial bearing fault detection
**Duration**: 4 overnight sessions (48 hours total)
**Result**: 43 experiments, predictor collapse fix, 3.7x transfer improvement

---

## Inspired by Karpathy's Autoresearch

This work is inspired by [Andrej Karpathy's autoresearch concept](https://github.com/karpathy/autoresearch) — using AI agents to run ML experiments autonomously overnight.

**What makes my setup different**:
- **Custom ml-researcher agent**: I built a specialized subagent within Claude Code specifically for ML research. It has domain knowledge about experiment design, statistical validation (3-seed protocols), and knows when to dig into literature vs. run ablations.
- **Structured task orchestration**: The agent writes to `EXPERIMENT_LOG.md` and `LESSONS_LEARNED.md` as it works, creating a persistent research narrative across sessions.
- **Self-correction loops**: The agent runs sanity checks after each experiment (loss decreasing? accuracy > chance? all seeds positive?) and pivots when results are unexpected.
- **HuggingFace dataset curation**: A separate `data-curator` agent built and uploaded a 7,985-sample mechanical vibration dataset overnight — the ml-researcher then used it for cross-component experiments.

The result: not just "run these hyperparameters" automation, but genuine research — diagnosing predictor collapse, reading the I-JEPA literature, designing targeted fixes, and validating with proper statistics.

---

## Session Overview

I used Claude Code to orchestrate autonomous overnight ML research sessions. The workflow: write detailed research prompts specifying experiments, success metrics, and self-check protocols — then let the agent execute autonomously while I slept.

---

## Session 1: Diagnosing Predictor Collapse

**Me**: The JEPA model gets 80% accuracy on CWRU but only +2.4% transfer to IMS. Something's wrong with the predictor. Can you diagnose?

**Claude**: I'll write diagnostic code to analyze the predictor behavior.

```python
# Collapse diagnostic
def diagnose_predictor_collapse(model, dataloader):
    predictions = []
    for batch in dataloader:
        with torch.no_grad():
            pred = model.predictor(context_embeddings, mask_positions)
            predictions.append(pred)

    # Key metric: variance across positions
    pred_var = torch.stack(predictions).var(dim=1).mean()
    spread_ratio = pred_var / target_var
    return spread_ratio  # Should be ~1.0, collapse if <0.1
```

Running on your best checkpoint (seed=123, 84.1% accuracy)...

**Results**:
- `pred_var_across_pos`: 0.000451 (threshold 0.001 = COLLAPSED)
- `spread_ratio`: 0.0201 (predictions 50x less diverse than targets)

The predictor is outputting nearly identical embeddings for every masked position. It learned to predict the context average rather than position-specific content.

**Me**: Why is this happening? Check the JEPA literature.

**Claude**: Searching I-JEPA, V-JEPA, and VICReg papers...

**Literature findings**:
- I-JEPA uses multi-block masking (4 different target blocks) — predictions for different spatial locations MUST differ
- V-JEPA uses 90% masking — context is so sparse that averaging gives terrible predictions
- Our mask_ratio=0.5 means 8/16 patches visible — representative sample, so context average ≈ target average

**The root cause**: At symmetric mask ratios, the predictor can minimize loss without using positional information at all. The context-average shortcut is mathematically valid.

**Me**: Design experiments to fix this.

**Claude**: I'll create a V2 architecture with targeted fixes:

```python
# V2 key changes
class JEPAV2Config:
    mask_ratio = 0.625      # Only 6 context patches (biased sample)
    pos_encoding = "sinusoidal"  # Guaranteed position discrimination
    loss_fn = "l1"          # Less incentive for mean predictions
    var_reg_lambda = 0.1    # Direct collapse penalty
    predictor_depth = 4     # More position-processing capacity
```

Running 3-seed validation at 100 epochs...

---

## Session 2: Results After Overnight Run

**Claude** (next morning): Overnight experiments complete. Results:

| Seed | V1 Accuracy | V2 Accuracy | Collapsed? |
|------|-------------|-------------|------------|
| 42 | 78.6% | 78.4% | No (spread=0.153) |
| 123 | 83.7% | **89.7%** | No (spread=0.138) |
| 456 | 77.3% | 78.1% | No (spread=0.148) |
| **Mean** | 79.9% ± 2.7% | **82.1% ± 5.4%** | **0/3 collapsed** |

Now testing IMS transfer with the fixed predictor...

| Metric | V1 (collapsed) | V2 (fixed) |
|--------|---------------|------------|
| IMS binary transfer | +2.4% ± 2.9% | **+8.8% ± 0.7%** |
| 3-class transfer | +3.3% ± 1.3% | **+7.6% ± 1.8%** |
| **Improvement** | - | **3.7x** |

The predictor collapse fix gives 3.7x better cross-dataset transfer. The collapsed predictor was invisible in-domain but devastating for generalization.

---

## Session 3: Cross-Dataset Transfer Barrier

**Me**: Transfer to Paderborn bearings (64kHz) is failing (-1.4%). Why?

**Claude**: The sampling rate mismatch is 5.3x (CWRU 12kHz → Paderborn 64kHz). The model learned spectral patterns at 12kHz that don't align at 64kHz.

Let me test frequency standardization — resampling Paderborn to 20kHz before evaluation:

```python
# Resample to common frequency
from scipy.signal import resample_poly
paderborn_20k = resample_poly(paderborn_64k, up=5, down=16)  # 64→20kHz
```

**Results (3 seeds)**:

| Resampling | Transfer Gain |
|------------|---------------|
| None (64kHz) | -1.4% ± 1.0% (0/3 seeds) |
| To 12kHz | +8.5% ± 3.0% (3/3 seeds) |
| To 20kHz | **+14.7% ± 0.8%** (3/3 seeds) |

The "5.3x sampling rate barrier" was entirely a data engineering problem. Simple polyphase resampling converts our worst result into our best.

---

## Session 4: Benchmarking Against Large Models

**Me**: How does our 5M-param model compare to wav2vec2 (94M params)?

**Claude**: Loading wav2vec2-base from HuggingFace, freezing encoder, training linear probe on CWRU...

```python
# wav2vec2 baseline
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
# Resample 12kHz vibration → 16kHz (wav2vec2 native)
vibration_16k = torchaudio.transforms.Resample(12000, 16000)(signal)
```

**Results (3 seeds)**:

| Method | Params | CWRU Accuracy |
|--------|--------|---------------|
| wav2vec2-base (frozen) | 94M | 77.2% ± 3.0% |
| **V2 JEPA (ours)** | **5M** | **87.1% ± 7.2%** |
| Random init | 5M | 71.8% ± 4.7% |

Our domain-specific 5M model outperforms the 94M speech model by **+9.9%**. wav2vec2 does provide some transferable waveform features (+5.4% over random), but domain-targeted pretraining wins decisively.

---

## Session 5: Cross-Component Transfer (Bonus)

**Me**: Can bearing features transfer to gearboxes? Load the HuggingFace dataset we built.

**Claude**: Loading gearbox data from `Forgis/Mechanical-Components`...

```python
from datasets import load_dataset
gearboxes = load_dataset("Forgis/Mechanical-Components", "gearboxes")
# MCC5-THU: 956 samples, 8 fault types, 12.8kHz
```

Evaluating CWRU-bearing-pretrained encoder on gearbox 8-class classification:

| Method | Macro F1 | Seeds Positive |
|--------|----------|----------------|
| JEPA (bearing-pretrained) | 0.1425 | **3/3** |
| Random init | 0.1174 | - |
| **Gain** | **+2.5%** | - |

Small but consistent cross-component transfer. Vibration features partially generalize across fundamentally different mechanical systems.

---

## Session 6: Continual Learning

**Me**: If I continue pretraining on IMS after CWRU, will it forget CWRU?

**Claude**: Testing continual learning with low LR (5e-5):

```python
# Continue from CWRU checkpoint
optimizer = AdamW(model.parameters(), lr=5e-5)  # 2x lower than original
for epoch in range(20):
    train_jepa(model, ims_dataloader)
```

| Metric | Before IMS | After 20 epochs IMS |
|--------|------------|---------------------|
| CWRU Macro F1 | 0.9264 | 0.9249 |
| **Change** | - | **-0.15%** |

No catastrophic forgetting. The EMA target encoder acts as a momentum anchor. This enables the deployment story: pretrain once → adapt to new machines with unlabeled data → previous knowledge retained.

---

## Final Results Summary

| Metric | Value |
|--------|-------|
| CWRU Macro F1 (3-seed) | **0.773 ± 0.018** |
| F1 gain over random | **+0.360** |
| CWRU → IMS transfer | **+8.8% ± 0.7%** |
| CWRU → Paderborn @20kHz | **+14.7% ± 0.8%** |
| Bearing → Gearbox | **+2.5% F1** (3/3 seeds) |
| vs wav2vec2 (94M params) | **+9.9%** with 18x fewer params |
| Continual learning forgetting | **-0.15%** |

---

## What Made This Work

1. **Autonomous overnight execution**: Claude ran 43 experiments across 4 nights while I slept, each with proper 3-seed validation and statistical analysis.

2. **Literature-grounded debugging**: When the predictor collapsed, Claude traced it to the I-JEPA/V-JEPA literature and identified the mathematical root cause (context-average shortcut at symmetric mask ratios).

3. **Honest negative results**: Multi-source pretraining hurts (-7.5%). IMS→CWRU transfer is negative (-6.8%). JEPA is worse than RMS for smooth health monitoring. All documented.

4. **Simple fixes over complexity**: The Paderborn "failure" was solved by resampling, not architecture changes. Frequency standardization converted -1.4% to +14.7%.

---

*43 experiments, 4 overnight sessions, 1 predictor collapse fix that improved transfer 3.7x.*

# Overnight Autoresearch Prompt: Mechanical-JEPA V3

## Fill the Gaps from V2: Frequency Standardisation, Pretrained Encoders, Transfer Matrix

This prompt was used inline (not saved to file) for the V3 overnight run on 2026-04-01.

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA project — CONTINUATION of previous run.

Working directory: /home/sagemaker-user/IndustrialJEPA/mechanical-jepa

### Context: What Was Already Done (V2 Overnight Run)

The previous overnight run fixed predictor collapse and achieved:
- CWRU linear probe: 82.1% ± 5.4% (3 seeds), best seed 89.7%
- IMS transfer: +8.8% ± 0.7% (3.7x improvement over V1)
- Transfer efficiency: 142% (cross-domain beats in-domain!)
- Predictor collapse: FIXED via mask_ratio=0.625 + sinusoidal pos + L1 + var_reg=0.1
- Spectral inputs: Dual (raw+FFT) hit 95.4% on CWRU but FFT features DON'T transfer across sampling rates
- Paderborn transfer: FAILED (-1.4%) due to 5.3x sampling rate mismatch (12kHz→64kHz)

### ROUND 1: FREQUENCY STANDARDISATION & PADERBORN RE-TEST

Implement resampling infrastructure. Add --target-sr flag. Resample everything to
a common rate (12kHz or 20kHz). Test CWRU→Paderborn transfer with resampled data.
Multi-source pretraining at common rate.

Key experiments:
- Exp 1C-1: CWRU → Paderborn with Paderborn downsampled to 12kHz
- Exp 1C-2: All datasets at 20kHz (IMS native)
- Exp 1C-3: All datasets at 12kHz
- Exp 1C-4: Multi-source pretraining at common rate

### ROUND 2: PRETRAINED ENCODERS FROM RELATED DOMAINS

Search for and test pretrained models (MOMENT, Chronos, wav2vec2, HuBERT, BEATs).
Frozen feature extraction, fine-tuned extraction, cross-dataset transfer with
pretrained backbone.

### ROUND 3: HIGHER MASK RATIOS & TRAINING VARIATIONS

Fine-grained mask ratio sweep (0.5 to 0.875). 200-epoch training. Reduce seed
variance with more regularisation.

### ROUND 4: HUGGINGFACE MECHANICAL-COMPONENTS DATASET

Download and explore Forgis/Mechanical-Components. Sanity checks. New-source bearing
pretraining. Cross-component transfer (bearings→gearboxes). All-source pretraining.

### ROUND 5: ADVANCED ARCHITECTURE EXPERIMENTS

Temporal block masking + V2. Multi-scale patch sizes (128, 256, 512). Ensemble /
multi-resolution.

### ROUND 6: COMPREHENSIVE CROSS-DATASET TRANSFER TABLE

Complete transfer matrix for all source→target combinations. Statistical analysis.

### ROUND 7: DOCUMENTATION & COMMIT

Update experiment log, lessons learned, notebook. Commit and push after each round.
```

## Results (Exp 24-35)

| Experiment | Finding |
|---|---|
| Exp 24: Paderborn @ 12kHz | +8.5% ± 3.0% transfer (was -1.4%) |
| Exp 25: Paderborn @ 20kHz | **+14.7% ± 0.8%** transfer |
| Exp 26: Mask ratio sweep | mask=0.75 best at 30ep |
| Exp 27: Mask ratio 100ep | mask=0.625 confirmed optimal at 100ep |
| Exp 28: wav2vec2 vs JEPA | JEPA 5M = 87.1% beats wav2vec2 94M = 77.2% |
| Exp 29: Block masking | No benefit over random masking |
| Exp 30: 200 epochs | Overfits, 100ep optimal |
| Exp 31: IMS pretrain | Saved for transfer matrix |
| Exp 32: var_reg sweep | var_reg=0.05 best mean, 0.2 lowest variance |
| Exp 33: Multi-source | Hurts CWRU (-7.5%) |
| Exp 34: Transfer matrix | CWRU→everywhere works; IMS→CWRU negative |
| Exp 35: Patch size | 128≈256 optimal, 512 bad |

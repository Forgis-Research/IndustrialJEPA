# Literature Review: MTS-JEPA Research Space

*Search conducted April 2026. 9 topics, 3-5 papers each.*

## Key Findings

### Most Relevant Papers for Our Work
1. **COMET** (arXiv 2026) — VQ coresets + multi-scale patch encoding + online codebook adaptation for TS anomaly detection
2. **C-JEPA** (NeurIPS 2024) — Shows EMA-based JEPA is prone to partial collapse; VICReg integration fixes it
3. **FCM** (KDD 2025) — Canonical anomaly prediction benchmark methodology
4. **TS-JEPA** (ICLR 2026) — JEPA with high masking ratios for time series; direct predecessor to MTS-JEPA
5. **EDVAE** (ICLR 2025) — Entropy-diversity regularization prevents codebook collapse
6. **TimeVQVAE-AD** (Pattern Recognition 2024) — VQ-VAE + masked generation for anomaly detection
7. **HiMTM** (CIKM 2024) — Hierarchical multi-scale masked modeling for time series
8. **Decision-oriented PdM metric** (RESS 2024) — Lead-time-aware evaluation framework

### Critical Insight: C-JEPA Shows EMA is Insufficient
C-JEPA (NeurIPS 2024) proved that EMA-based target encoders in JEPA converge to mean patch representations (partial collapse). MTS-JEPA's soft codebook may be an alternative fix, but the paper doesn't test this hypothesis explicitly. This is a key experiment for our extension.

### Anomaly Prediction is a New Field (2024-2026)
- FCM (KDD 2025): Future Context Modeling
- RED-F (arXiv 2025): Dual-stream contrastive forecasting
- F2A (arXiv 2025): Foundation model adaptation
- A2P/When Will It Fail (ICML 2025): Anomaly-to-prompt paradigm

All these are concurrent — the field is wide open for architectural innovation.

### Codebook Collapse is Solved in Theory
- EDVAE (ICLR 2025): Entropy-diversity regularization
- VQBridge (arXiv 2025): 100% codebook utilization via learning annealing
- VAEVQ (arXiv 2025): Combining VAE soft regularization with VQ hard quantization

MTS-JEPA's dual entropy approach is one solution but not SOTA. Our extension could adopt VQBridge/EDVAE for better utilization.

### RUL + SSL is Active But No JEPA Yet
No paper combines JEPA with RUL prediction. Our Trajectory JEPA V11 is the first. Adding a codebook to Trajectory JEPA for interpretable degradation regimes is genuinely novel.

## Gap Map

| Gap | Status | Who's Closest | Our Advantage |
|-----|--------|---------------|---------------|
| JEPA + codebook for prognostics | OPEN | MTS-JEPA (anomaly), TS-JEPA (no codebook) | We have C-MAPSS infrastructure + V11 |
| Lead-time-aware AP evaluation | OPEN | FCM (partial), A2P (partial) | Can define proper metric using RUL framing |
| Codebook = degradation regimes | OPEN | TimeVQVAE-AD (frequency regimes only) | Physical interpretation on C-MAPSS |
| Multi-resolution for bearing vibration | OPEN | HiMTM (forecasting only) | We have FEMTO/XJTU data |
| Theory-validated codebook bounds | OPEN | MTS-JEPA (theory only), C-JEPA (collapse only) | Can validate empirically |
| VICReg + codebook combination | OPEN | C-JEPA (VICReg, no codebook), MTS-JEPA (codebook, no VICReg) | Can test the fusion |
| Discrete bottleneck for industrial fault types | OPEN | RepDIB (RL only) | C-MAPSS has known fault modes |
| Foundation model + codebook for TS anomaly | OPEN | F2A (foundation, no codebook), TOTEM (codebook, no anomaly) | Novel combination |

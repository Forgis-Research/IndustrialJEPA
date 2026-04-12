# Gap Map: What Nobody Has Done Yet

## Research Gaps Identified

| # | Gap | Status | Who's Closest | Our Advantage | Priority |
|---|-----|--------|---------------|---------------|----------|
| 1 | JEPA + codebook for prognostics/RUL | **OPEN** | MTS-JEPA (anomaly only), TS-JEPA (no codebook) | V11 Trajectory JEPA + C-MAPSS infrastructure | P0 |
| 2 | Lead-time-aware anomaly prediction eval | **OPEN** | FCM (partial), A2P (timing metric) | Can define proper metric separating prediction from detection | P0 |
| 3 | Codebook entries = degradation regimes | **OPEN** | TimeVQVAE-AD (frequency only) | C-MAPSS has known degradation physics for validation | P0 |
| 4 | Causal + codebook JEPA | **OPEN** | Trajectory JEPA (causal, no codebook), MTS-JEPA (codebook, not causal) | Can build CC-JEPA as fusion | P0 |
| 5 | VICReg + codebook for JEPA stability | **OPEN** | C-JEPA (VICReg, no codebook) | Can test if codebook already solves collapse | P1 |
| 6 | Multi-resolution for bearing vibration data | **OPEN** | HiMTM (forecasting, not prognostics) | FEMTO/XJTU datasets available | P1 |
| 7 | Theory-validated codebook bounds (empirical) | **OPEN** | MTS-JEPA (theory only, no plots) | Can track all bound quantities | P1 |
| 8 | IB analysis of codebook JEPA | **OPEN** | RepDIB (RL), VQ-VAE (generation) | Can compute IB curve for anomaly prediction | P2 |
| 9 | Cross-domain codebook transfer for prognostics | **OPEN** | Foundation models (no codebook) | C-MAPSS → bearing data transfer | P2 |
| 10 | Optimal K selection via MDL/IB | **OPEN** | - | Can sweep K with information-theoretic justification | P2 |

## The NeurIPS Contribution

**Gap 1 + Gap 2 + Gap 3 + Gap 4** = CC-JEPA paper:
- Causal codebook JEPA architecture (fills gap 4)
- Evaluated on both anomaly prediction AND RUL (fills gap 1)
- With lead-time-aware metrics (fills gap 2)
- With interpretable degradation regime discovery (fills gap 3)

This combination is novel, addresses the major weaknesses in MTS-JEPA's NeurIPS review, and leverages our existing infrastructure.

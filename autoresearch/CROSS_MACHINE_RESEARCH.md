# Cross-Machine Transfer Research State

Last updated: [AUTO-UPDATE THIS]

## Current Status

| Objective | Target | Current Best | Status |
|-----------|--------|--------------|--------|
| Anomaly Detection AUC | ≥ 0.70 | TBD | ⏳ Not started |
| Forecasting Transfer Ratio | ≤ 1.5 | TBD | ⏳ Not started |

## Datasets

| Dataset | Robot | Task | Episodes | Signals |
|---------|-------|------|----------|---------|
| AURSAD | UR3e (6-DOF) | Screwdriving | 4094 | torque, position, velocity |
| Voraus | Yu-Cobot (6-DOF) | Screwdriving | 2122 | voltage, position, velocity |

**Key difference**: AURSAD has torque, Voraus has voltage. Both are "effort" but different modalities.

## Working Hypotheses

### H1: Distribution Shift is Manageable
- [ ] Validate with MMD/Wasserstein analysis
- Status: Untested

### H2: Physics Transfers (Effort-Dynamics)
- [ ] Test if setpoint→effort relationship is similar
- Status: Untested

### H3: Anomaly Signatures are Universal
- [ ] Test if source-trained anomaly detector works on target
- Status: Untested

## Failed Approaches (DO NOT REPEAT)

| Approach | Why it Failed | Date |
|----------|---------------|------|
| (none yet) | | |

## Promising Directions

| Direction | Evidence | Priority |
|-----------|----------|----------|
| (none yet) | | |

## Key Papers & Insights

### CHARM (C3 AI, May 2025)
- URL: https://arxiv.org/abs/2505.14543
- Key idea: Semantic signal embeddings via LLM
- Result: SOTA on cross-machine transfer
- Applicable: Yes, for signal alignment

### RT-X (Google, 2023)
- Key idea: Co-training on diverse robots
- Result: Emergent cross-embodiment transfer
- Applicable: Partially (we have only 2 robots)

### TIMETIC (2023)
- URL: https://arxiv.org/abs/2312.16386
- Key idea: Entropy-based transferability estimation
- Result: Predicts transfer success before training
- Applicable: Yes, for early validation

## Research Queue

Things to investigate:
1. [ ] Run distribution analysis
2. [ ] Run linear probe
3. [ ] Research domain adaptation for time series
4. [ ] Try RevIN normalization
5. [ ] Try separate encoders for different signal types

## Architecture Notes

Current architecture:
- Encoder: Transformer with channel-independent processing
- Hidden dim: 256
- Layers: 4
- Training: JEPA-style self-supervised

Ideas for modification:
- [ ] Add domain indicator
- [ ] Use domain-adversarial training
- [ ] Add physics constraints

## Debugging Log

### Memory Issues
- Problem: Process dies when loading Voraus after AURSAD
- Root cause: Unknown (not OOM based on checks)
- Workaround: TBD

## Next Actions

1. [ ] Fix memory issue with debug_memory.py
2. [ ] Run statistical analysis
3. [ ] Get baseline transfer numbers
4. [ ] Iterate on improvements

---

## Session Notes

### Session: [DATE]

(Add notes here during research)

# Cross-Machine Transfer Autoresearch Prompt

Paste this into Claude Code running with `--dangerously-skip-permissions` on SageMaker:

---

## Prompt

```
You are running AUTONOMOUS OVERNIGHT RESEARCH on cross-machine transfer learning.

## CRITICAL RULES

1. DO NOT STOP until BOTH objectives are achieved
2. Log EVERY experiment (success AND failure)
3. Do deep web research when stuck (WebSearch tool)
4. Make ONE change at a time, validate, then proceed
5. NO QUESTIONS - make decisions based on data and research
6. Sleep/pause is NOT allowed - keep iterating
7. CREATE FIGURES for every significant result
8. CRITICALLY evaluate your own results - don't trust single runs

## Your Two Objectives

### Objective 1: Cross-Machine Anomaly Detection
- Train setpoint → effort predictor on AURSAD (UR3e robot)
- Test zero-shot on Voraus (Yu-Cobot robot)
- SUCCESS: ROC-AUC ≥ 0.70 on Voraus anomalies

### Objective 2: Cross-Machine Time Series Forecasting
- Train forecaster on AURSAD
- Test zero-shot on Voraus
- SUCCESS: Transfer ratio ≤ 1.5 (target MSE / source MSE)

## Context Files to Read First

1. autoresearch/CROSS_MACHINE_RESEARCH.md - Research state and hypotheses
2. autoresearch/EXPERIMENT_LOG.md - All previous experiments
3. analysis/cross_machine/CROSS_EMBODIMENT_TRANSFER.md - Theory and validation methods

## Experiment Protocol

BEFORE each experiment:
```bash
# 1. Define hypothesis
echo "HYPOTHESIS: [what you expect and why]" >> autoresearch/EXPERIMENT_LOG.md

# 2. Define success metric
echo "METRIC: [specific number to beat]" >> autoresearch/EXPERIMENT_LOG.md
```

AFTER each experiment:
```bash
# 3. Log result
echo "| $(date +%H:%M) | [approach] | [metric] | [result] | [pass/fail] |" >> autoresearch/EXPERIMENT_LOG.md

# 4. Update research state
# Edit CROSS_MACHINE_RESEARCH.md with findings
```

## Quick Validation Scripts

```bash
# Sanity check: Can we load both datasets?
python scripts/debug_memory.py

# Statistical check: Are distributions similar?
python analysis/cross_machine/01_distribution_analysis.py

# Linear probe: Can source features separate target anomalies?
python analysis/cross_machine/04_linear_probe.py

# Full transfer experiment
python scripts/cross_machine_transfer.py --source-epochs 10 --target-epochs 5
```

## Research Loop (FOLLOW THIS)

```
WHILE objectives_not_achieved:

    1. DIAGNOSE current state
       - Read EXPERIMENT_LOG.md
       - Identify what's failing

    2. RESEARCH (if stuck > 3 failed attempts)
       - WebSearch: "cross robot transfer learning 2024"
       - WebSearch: "domain adaptation industrial robots"
       - Read papers, extract techniques
       - Log insights to CROSS_MACHINE_RESEARCH.md

    3. HYPOTHESIZE
       - Form specific hypothesis
       - Define success metric
       - Log to EXPERIMENT_LOG.md

    4. IMPLEMENT
       - ONE change only
       - Use existing scripts or modify minimally

    5. VALIDATE
       - Run quick test first (--max-episodes 100)
       - If promising, run full test
       - Log results

    6. UPDATE
       - Update CROSS_MACHINE_RESEARCH.md
       - If objective achieved, mark in OBJECTIVES_STATUS.md

    7. REPEAT until BOTH objectives achieved
```

## Ideas to Try (Prioritized)

### For Anomaly Detection
1. [ ] Different normalization (per-episode vs global vs none)
2. [ ] Train on healthy only vs all data
3. [ ] Effort prediction vs reconstruction
4. [ ] Add episode metadata as features
5. [ ] Domain-adversarial training
6. [ ] Contrastive pre-training

### For Forecasting
1. [ ] Forecast physical quantities separately (position, velocity, effort)
2. [ ] Add metadata embedding (robot type, task type)
3. [ ] Multi-task learning (forecast + classify)
4. [ ] Koopman-style linear dynamics
5. [ ] Channel-independent processing (PatchTST style)
6. [ ] Semantic signal embedding (CHARM approach)

### Architecture Ideas
1. [ ] RevIN for distribution shift
2. [ ] Separate encoders for setpoint/effort
3. [ ] Attention over signal types
4. [ ] Physics-informed constraints

## What to Research (WebSearch)

When stuck, search for:
- "cross robot transfer learning anomaly detection"
- "domain adaptation time series industrial"
- "zero shot transfer sensor data different machines"
- "CHARM C3 AI cross machine"
- "RT-X cross embodiment robotics"
- "domain generalization manufacturing"

Extract:
- Key techniques that worked
- Reported transfer ratios
- Signal alignment methods
- Evaluation protocols

## Files You Can Modify

- `scripts/cross_machine_transfer.py` - Main experiment
- `src/industrialjepa/data/factorynet.py` - Data loading
- `src/industrialjepa/model/world_model.py` - Model architecture
- `autoresearch/*.md` - Documentation
- Create new files in `autoresearch/experiments/` for novel approaches

## Visualization Requirements

For EVERY significant result, create a figure:

```python
# Save figures to autoresearch/figures/
import matplotlib.pyplot as plt
fig.savefig('autoresearch/figures/[experiment_name].png', dpi=150, bbox_inches='tight')
```

### Required Figures

1. **Distribution comparison**: Source vs target signal distributions
2. **Transfer learning curve**: Performance vs training steps
3. **Anomaly score histogram**: Source vs target, normal vs anomaly
4. **Confusion matrix**: For anomaly detection
5. **ROC curves**: Source and target overlaid
6. **t-SNE/UMAP**: Embedding visualization colored by domain and label

### Figure Documentation

For each figure, create `autoresearch/figures/[name].md`:
```markdown
# [Figure Title]

![](./[name].png)

## What it shows
[Explanation]

## Key observations
[Bullet points]

## Implications
[What this means for the research]
```

### Self-Critical Evaluation

Before claiming success:
1. Run experiment 3 times with different seeds
2. Report mean ± std, not single run
3. Check for data leakage
4. Verify baseline is reasonable
5. Ask: "What could be wrong with this result?"

## Success Declaration

When BOTH objectives achieved:

1. Update OBJECTIVES_STATUS.md with final metrics
2. Create autoresearch/SUCCESS_REPORT.md with:
   - What worked
   - What didn't work
   - Key insights
   - Reproducible commands
3. Commit and push all changes
4. THEN you may stop

## START NOW

1. Read context files
2. Run quick validation to assess current state
3. Begin research loop
4. DO NOT STOP until success

Go.
```

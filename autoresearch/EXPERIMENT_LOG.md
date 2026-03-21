# Many-to-1 Transfer Experiment Log

## Core Principle

**1-to-1 transfer is nearly impossible. Many-to-1 is tractable.**

---

## Objectives Status

### Track 1: Bearings (Validation)
| Metric | Target | Best Result | Achieved? |
|--------|--------|-------------|-----------|
| Diagnosis Accuracy | ≥ 80% | - | ❌ |
| RUL Transfer Ratio | ≤ 2.0 | - | ❌ |

### Track 2: Robots (Novel)
| Metric | Target | Best Result | Achieved? |
|--------|--------|-------------|-----------|
| Avg Forecast Ratio | ≤ 2.5 | - | ❌ |
| Avg Anomaly AUC | ≥ 0.60 | - | ❌ |

---

## Experiment Table

| # | Time | Track | Sources | Target | Metric | Result | Pass/Fail |
|---|------|-------|---------|--------|--------|--------|-----------|
| 1 | - | - | - | - | - | - | - |

---

## Detailed Experiment Log

### Template

```markdown
### Experiment #N: [Title]

**Date**: [AUTO]
**Track**: [1-Bearings / 2-Robots]
**Sources**: [list datasets used for training]
**Target**: [held-out dataset]

**Hypothesis**: [What you expect and why]
**Approach**: [What you're doing differently]

**Command**:
\`\`\`bash
[exact command]
\`\`\`

**Result**:
\`\`\`
[paste output]
\`\`\`

**Metrics**:
- Accuracy/AUC: [value]
- Transfer Ratio: [value]
- Seeds: [list 3 values] → mean ± std

**Pass/Fail**: [PASS/FAIL]
**Figure**: `figures/exp_N_description.png`

**Learnings**: [What did we learn?]
**Next**: [What to try next]

---
```

---

## Summary Statistics

| Category | Track 1 | Track 2 | Total |
|----------|---------|---------|-------|
| Total | 0 | 0 | 0 |
| Pass | 0 | 0 | 0 |
| Fail | 0 | 0 | 0 |

---

## Research Pauses

| Time | Topic | Key Finding |
|------|-------|-------------|
| - | - | - |

---

## Negative Results (Important!)

| Track | Approach | Why It Failed | Evidence |
|-------|----------|---------------|----------|
| - | - | - | - |

---

## Success Milestones

### Track 1: Bearings
- [ ] PHMD library installed and working
- [ ] CWRU dataset loads correctly
- [ ] PHM2012 dataset loads correctly
- [ ] XJTU-SY dataset loads correctly
- [ ] Paderborn dataset loads correctly
- [ ] First baseline (train all, test Paderborn)
- [ ] Accuracy > 70%
- [ ] **Accuracy ≥ 80%** (OBJECTIVE)
- [ ] Transfer ratio < 2.5
- [ ] **Transfer ratio ≤ 2.0** (OBJECTIVE)

### Track 2: Robots
- [ ] UR3 CobotOps downloaded
- [ ] NIST UR5 downloaded
- [ ] Robot Failures downloaded
- [ ] Unified data loader created
- [ ] First Leave-One-Out experiment
- [ ] AUC > 0.50 (better than random)
- [ ] AUC > 0.55
- [ ] **Avg AUC ≥ 0.60** (OBJECTIVE)
- [ ] Forecast ratio < 3.0
- [ ] **Avg Forecast ratio ≤ 2.5** (OBJECTIVE)

### Both Tracks Complete
- [ ] **TRACK 1 COMPLETE**
- [ ] **TRACK 2 COMPLETE**
- [ ] SUCCESS_REPORT.md written
- [ ] All changes committed

---

## Key Insights Discovered

(Add insights as you discover them)

1. ...

---

## Final Notes

(Fill when objectives achieved)

### What Worked

### What Didn't Work

### Key Insights

### Reproducible Commands

### Experiment #1: Baseline Transfer (Episode Normalization)

**Date**: 2026-03-20 21:30
**Hypothesis**: A setpoint→effort predictor trained on AURSAD healthy data will detect anomalies on Voraus via prediction error, because anomalies violate normal physics. Episode normalization removes scale differences between robots.
**Approach**: Train SimpleEffortPredictor on AURSAD (voltage signals), evaluate zero-shot on Voraus.
**Command**:
```bash
python autoresearch/experiments/exp01_baseline_transfer.py --seed 42 --epochs 10 --norm-mode episode
```
**Metric**: Target AUC ≥ 0.70, Transfer Ratio ≤ 1.5

**Result**:
```
Source AUC: 0.511 (near random)
Target AUC: 0.489 (near random)
Source MSE: 0.793
Target MSE: 0.903
Transfer Ratio: 1.14 ← PASSES objective 2!
```
**Pass/Fail**: FAIL for anomaly, PASS for forecasting
**Learnings**: 
- Forecasting transfer works well with episode normalization (ratio 1.14)
- Anomaly detection completely fails - prediction error doesn't separate normal from anomalous
- The setpoint→effort predictor with episode normalization erases anomaly signal
- Episode normalization likely normalizes away the anomaly signature (anomalous effort patterns get z-scored to look normal)
**Next**: Try global normalization or no normalization - anomalies should show as abnormal effort magnitudes

---

### Experiment #2: Anomaly Detection with NO Normalization

**Date**: 2026-03-20 21:55
**Hypothesis**: Episode normalization removes anomaly signal by z-scoring each window independently. With NO normalization, anomalous effort patterns (higher/different magnitudes) should stand out as higher prediction error.
**Metric**: Target AUC ≥ 0.70
**Result**: Source AUC=0.442, Target AUC=0.507, Transfer Ratio=0.028 (misleading - different scales)
**Pass/Fail**: FAIL
**Learnings**: No normalization makes signals incomparable across domains. AUC still near random.

---

### Experiment #3: Global Normalization + More Epochs

**Date**: 2026-03-20 21:58
**Hypothesis**: Global normalization preserves magnitude differences (anomaly signal) while standardizing scale. More epochs (20) may help model converge to better setpoint→effort mapping.
**Metric**: Target AUC ≥ 0.70
**Result**: Source AUC=0.533, Target AUC=0.495, Transfer Ratio=0.90
**Pass/Fail**: FAIL for anomaly, PASS for forecasting

---

### Experiment #4: Diagnostic Analysis

**Date**: 2026-03-20 22:03
**Hypothesis**: Anomaly detection fails because anomaly signals differ between datasets
**Approach**: Statistical analysis of normal vs anomalous windows
**Result**:
```
AURSAD (global norm): Anomaly variance 3-9x higher than normal → detectable
Voraus (global norm): Anomaly variance <3% different from normal → NOT detectable
Episode normalization: Erases ALL differences
```
**Key Finding**: Voraus anomalies do NOT manifest as different voltage patterns compared to normal.
1-to-1 cross-machine anomaly detection AURSAD→Voraus is fundamentally limited.
**Learnings**:
1. Anomaly signatures are NOT universal across robots/tasks
2. Episode normalization completely erases anomaly signal
3. Forecasting transfer works well (ratio 0.9-1.14) because normal dynamics ARE universal
4. Anomaly detection requires understanding dataset-specific anomaly patterns

---

## PIVOT: Many-to-1 Transfer Learning

User directed pivot from 1-to-1 to MANY-to-1 transfer.
Key insight: training on MANY similar embodiment datasets and transferring to held-out one is more tractable.
Need to find additional industrial time series datasets and reformulate objectives.

---


### Experiment #5: Many-to-1 Leave-One-Out Transfer with RevIN

**Date**: 2026-03-20 22:20
**Hypothesis**: Training on multiple source domains provides more diverse "normal" representations that transfer better to held-out domains. RevIN normalizes per-instance to handle domain shift while preserving learnable temporal patterns.
**Approach**: Leave-one-out over AURSAD, Voraus, CNC. Compare many-to-1 vs 1-to-1.
**Command**:
```bash
python autoresearch/experiments/exp02_multi_source_transfer.py --seed 42 --epochs 15
```
**Metric**: Transfer ratio ≤ 1.5, AUC ≥ 0.70


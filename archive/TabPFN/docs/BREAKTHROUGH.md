# What a Breakthrough Would Look Like

**Central Question**: Can treating mechanical time series as tabular data with temporal features provide competitive or superior forecasting compared to specialized time series methods?

---

## Levels of Success

### Level 1: Baseline (Minimum Viable)

**Definition**: TabPFN-TS works on mechanical data and matches naive baselines.

**Evidence**:
- TabPFN-TS runs without errors on all target datasets
- RMSE within 20% of naive baseline
- We understand why it works (or doesn't)

**What this means**:
- TabPFN-TS is not obviously broken on this domain
- Worth further investigation
- Can identify specific failure modes

**Likelihood**: High (90%+)

---

### Level 2: Promising Signal

**Definition**: TabPFN-TS beats naive baselines and approaches statistical methods.

**Evidence**:
- Skill score > 0.1 vs naive on 2+ datasets
- Competitive with ARIMA on at least one dataset
- Covariates provide measurable improvement (>5%)

**What this means**:
- Tabular framing captures some useful structure
- Covariate mechanism is valuable for mechanical systems
- Worth developing further

**What we could publish**:
- Workshop paper: "TabPFN for Mechanical Systems: Initial Assessment"
- Focus on covariate mechanism and practical applicability

**Likelihood**: Medium (50-70%)

---

### Level 3: Strong Result

**Definition**: TabPFN-TS matches or beats specialized methods on multiple datasets.

**Evidence**:
- Outperforms ARIMA and Prophet on 2+ datasets
- Skill score > 0.2 consistently
- Cross-condition transfer works (train on healthy, test on degraded)
- Clear insight into *why* it works

**What this means**:
- Treating mechanical TS as tabular is a valid paradigm
- Foundation model pretraining transfers to this domain
- Practical value: no training required

**What we could publish**:
- Conference paper: "Foundation Models for Mechanical System Forecasting"
- Key contribution: Zero-shot transfer from tabular pretraining to mechanical TS

**Likelihood**: Low-Medium (20-40%)

---

### Level 4: Breakthrough

**Definition**: Novel insight about mechanical time series that enables new methods.

**Evidence**:
- TabPFN-TS + domain features beats specialized SOTA
- Prediction residuals enable anomaly detection competitive with specialized methods
- Cross-system transfer works (train on hydraulic, test on robot)
- New theoretical understanding

**Potential breakthrough insights**:

1. **"Mechanical systems are more tabular than sequential"**
   - Operating conditions dominate temporal dynamics
   - Implication: Focus on covariates, not history length

2. **"Foundation models capture universal mechanical priors"**
   - TabPFN's pretraining on tabular data encodes physical relationships
   - Implication: Transfer learning across mechanical domains

3. **"Forecast residuals are physics-aware anomaly scores"**
   - Model learns normal physics; violations = anomalies
   - Implication: Combine forecasting + anomaly detection

**What we could publish**:
- Top-tier venue paper
- Open-source toolkit for mechanical system forecasting with TabPFN

**Likelihood**: Low (5-15%)

---

## How to Achieve a Breakthrough

### Path 1: Exploit Covariates

**Hypothesis**: Mechanical systems have strong covariate structure (control inputs → sensor outputs) that TabPFN-TS naturally handles.

**Strategy**:
1. Identify strongest covariate relationships in each dataset
2. Design experiments that highlight covariate benefit
3. Compare to methods that don't handle covariates well

**Target datasets**:
- C-MAPSS (operating settings)
- AURSAD (joint commands → currents)

### Path 2: Cross-Condition Transfer

**Hypothesis**: A model that learns general mechanical physics can transfer across conditions.

**Strategy**:
1. Train on "easy" conditions (healthy, low load)
2. Test on "hard" conditions (faulty, high load, different system)
3. Show retention of prediction quality

**Target experiments**:
- Hydraulic: Train on healthy, test on degraded
- C-MAPSS: Train FD001, test FD002-FD004
- Cross-dataset: Train hydraulic, test bearing?

### Path 3: Domain-Specific Features

**Hypothesis**: Adding mechanical-aware features to TabPFN-TS input improves predictions.

**Strategy**:
1. Extract physics-informed features:
   - FFT coefficients (vibration)
   - Rate of change (degradation)
   - Operating mode indicators
2. Add as extra columns in tabular representation
3. Compare to raw temporal features

**Risk**: If significant feature engineering is needed, value proposition weakens.

### Path 4: Anomaly Detection

**Hypothesis**: Forecast residuals are natural anomaly scores for mechanical systems.

**Strategy**:
1. Train TabPFN-TS on healthy operation data
2. Apply to data containing anomalies
3. Compute residuals, threshold for anomaly detection
4. Compare to specialized anomaly detection methods

**Success metric**: AUC-ROC > 0.8 on hydraulic fault detection

---

## Warning Signs (Negative Results)

These would indicate TabPFN-TS is not suitable for mechanical systems:

### Red Flag 1: Worse Than Naive
If TabPFN-TS consistently underperforms naive baselines:
- Tabular framing loses temporal information
- Pretraining doesn't transfer to this domain
- **Action**: Investigate failure modes, try feature engineering

### Red Flag 2: Covariates Don't Help
If adding known-relevant covariates doesn't improve predictions:
- Model can't capture input-output relationships
- Or: relationships are already captured by history
- **Action**: Check covariate data quality, try other datasets

### Red Flag 3: No Transfer
If cross-condition transfer completely fails:
- Model overfits to training conditions
- Mechanical physics not captured
- **Action**: Try more similar conditions, simplify task

### Red Flag 4: Computationally Infeasible
If TabPFN-TS is too slow for practical use:
- 11M parameters still significant for embedded systems
- **Action**: Focus on batch prediction scenarios

---

## Decision Points

### After Phase 1 (Validation)
- **If Level 1 achieved**: Continue to Phase 2
- **If not Level 1**: Debug, check data preprocessing, try simpler tests

### After Phase 2 (Core Experiments)
- **If Level 2+ achieved**: Write up findings, consider Phase 3
- **If Level 1 only**: Document negative results, consider pivoting

### After Phase 3 (Advanced)
- **If Level 3+ achieved**: Full paper, develop toolkit
- **If Level 2 only**: Workshop paper on specific findings

---

## Success Metrics Summary

| Level | Skill Score | Covariate Benefit | Transfer | Publication |
|-------|-------------|-------------------|----------|-------------|
| 1 (Baseline) | ~0 | N/A | N/A | Blog post |
| 2 (Promising) | >0.1 | >5% | Partial | Workshop |
| 3 (Strong) | >0.2 | >10% | Works | Conference |
| 4 (Breakthrough) | >0.3 | >20% | Cross-system | Top venue |

---

## Timeline to Breakthrough Assessment

| Week | Milestone | Decision |
|------|-----------|----------|
| 1-2 | Phase 1 complete | Level 1 achieved? |
| 3-4 | Phase 2 complete | Level 2 achieved? |
| 5-6 | Analysis & writing | Submit or pivot |
| 7+ | Phase 3 (if promising) | Level 3+ possible? |

---

## Comparison to IndustrialJEPA

This TabPFN exploration complements the main IndustrialJEPA project:

| Aspect | IndustrialJEPA | TabPFN-TS |
|--------|----------------|-----------|
| Approach | Train from scratch | Zero-shot (pretrained) |
| Innovation | Physics-informed attention | Tabular foundation model |
| Training cost | High | None |
| Customizability | Full control | Limited |
| Best for | Large datasets, specific tasks | Quick assessment, covariates |

**Synergy potential**: Use TabPFN-TS as a quick baseline or for covariate-heavy scenarios, then develop IndustrialJEPA for cases where custom training is justified.

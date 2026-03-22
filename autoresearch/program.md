# Autoresearch Program: Role-Based Transfer Learning

**Mission:** Demonstrate that structured channel-dependence enables cross-machine transfer where channel-independence fails.

**Target:** Results worthy of NeurIPS submission.

---

## Before You Start

Read these files:
1. `.claude/agents/ml_researcher.md` — Your operating instructions
2. `autoresearch/LESSONS_LEARNED.md` — What failed before
3. `autoresearch/EXPERIMENT_LOG.md` — Current best results

**Current state:** CI-Transformer achieves 0.450 MSE on ETTh1 H=96. This matches PatchTST-style channel-independence but cannot learn transferable physics.

---

## The Core Hypothesis

**Channel-independent processing:**
- ✅ Wins on single-dataset forecasting (avoids overfitting)
- ❌ Cannot learn cross-channel physics
- ❌ Cannot transfer across machines

**Role-based processing:**
- Group channels by physical component (joint, stage, subsystem)
- Share weights within component (physics is universal)
- Learn or specify cross-component topology
- **Should enable transfer while maintaining forecasting performance**

---

## Phase 1: Deep Research (1-2 hours)

Before coding, understand the landscape. Search for and analyze:

### Required Reading

1. **Transfer learning for time series** — What approaches work?
   - Search: "time series transfer learning survey" (recent, high citations)
   - Key questions: How do others handle different channel counts? Different domains?

2. **C-MAPSS SOTA** — What beats current baselines?
   - Search: "C-MAPSS remaining useful life 2024 2025"
   - Find: Best RMSE numbers, what architectures, any transfer experiments
   - Check Papers With Code for benchmarks

3. **Cross-machine fault diagnosis** — The bearing transfer literature
   - Search: "CWRU Paderborn transfer learning" (highly cited)
   - Key insight: How do they handle domain shift?

4. **Role-based / physics-informed architectures**
   - Search: "physics-informed neural network time series"
   - Search: "structured state space models"
   - Look for: Mamba, S4, related work

### Log Your Findings

Create entries in EXPERIMENT_LOG.md:
```markdown
## Research: [Topic]
**Papers reviewed**: N
**Key finding**: ...
**Implication for us**: ...
**Next experiment**: ...
```

---

## Phase 2: C-MAPSS Baseline (2-3 hours)

**Goal:** Establish strong baselines on the standard industrial benchmark.

### Setup

1. Download C-MAPSS dataset:
   - Source: NASA Prognostics Center or Kaggle mirror
   - 4 subsets: FD001, FD002, FD003, FD004

2. Create dataloader: `src/industrialjepa/data/cmapss.py`
   - 3 operational settings (inputs)
   - 21 sensors (outputs)
   - Piece-wise linear RUL (capped at 125)
   - Standard train/test split per subset

3. Define component groups (from OVERNIGHT_PROMPT_V2 archive):
   ```python
   COMPONENTS = {
       "operating": ["setting1", "setting2", "setting3"],
       "fan": ["s1", "s5", "s8", "s12", "s21"],
       "hpc": ["s3", "s7", "s10", "s11", "s20"],
       "lpt": ["s4"],
       ...
   }
   ```

### Experiments

| Exp | Model | What It Tests |
|-----|-------|---------------|
| 1 | Linear (flatten → predict) | Trivial baseline |
| 2 | LSTM | Standard sequence model |
| 3 | Transformer (channel-mixing) | Attention baseline |
| 4 | CI-Transformer | Channel-independent (our current best approach) |
| 5 | **Role-Transformer** | Grouped by component, shared within-component weights |

### Success Criteria

- FD001 test RMSE < 13.0 (competitive with LSTM baseline)
- Role-Transformer ≥ CI-Transformer on FD001

### Transfer Experiment

Once baselines established:
- Train on FD001
- Test on FD002 (different operating conditions)
- Compare: Role-Transformer vs CI-Transformer

**This is the key result:** Does role-based grouping help transfer?

---

## Phase 3: FactoryNet Transfer (if Phase 2 succeeds)

**Goal:** Establish SOTA on our novel robot transfer benchmark.

### Setup

Use existing `src/industrialjepa/data/factorynet.py`

Component groups:
```python
COMPONENTS = {
    "joint1": ["setpoint_pos_1", "actual_pos_1", "setpoint_vel_1", "actual_vel_1", "effort_1"],
    "joint2": [...],
    ...
    "joint6": [...],
}
```

### Experiments

| Train | Test | Model | MSE |
|-------|------|-------|-----|
| AURSAD Robot1 | Robot1 | CI-Transformer | ? |
| AURSAD Robot1 | Robot1 | Role-Transformer | ? |
| AURSAD Robot1 | Robot2 | CI-Transformer | ? |
| AURSAD Robot1 | Robot2 | Role-Transformer | ? |
| AURSAD both | Voraus | CI-Transformer | ? |
| AURSAD both | Voraus | Role-Transformer | ? |

### Success Criteria

- Role-Transformer transfer ratio < CI-Transformer transfer ratio
- Document the protocol for others to reproduce

---

## Phase 4: Iterate Until Dawn

Run the Karpathy loop:

```
while time > 0:
    1. Identify weakest result
    2. Hypothesize improvement (or research if stuck)
    3. Implement ONE change
    4. Train (5-10 min max)
    5. Evaluate
    6. Keep/revert, log, commit
    7. Repeat
```

### Ideas to Try (if baselines work)

- [ ] RevIN normalization
- [ ] Different component groupings
- [ ] Cross-component GNN vs attention
- [ ] JEPA pretraining then fine-tune
- [ ] Patch size ablations
- [ ] Layer depth ablations
- [ ] xLSTM as predictor (efficiency claim)

### If Stuck

1. Research: What did successful papers do?
2. Simplify: Is the model too complex?
3. Debug: Is there a data issue?
4. Pivot: Try different component groupings
5. Move on: Log and try Phase 3

---

## Logging Requirements

### Every Experiment

```markdown
## Exp N: [Description]

**Time**: HH:MM
**Hypothesis**: ...
**Change**: ...
**Result**: [before] → [after]
**Verdict**: KEEP ✓ / REVERT ✗
**Insight**: ...
**Next**: ...
```

### Every Hour

Push to git:
```bash
git add -A && git commit -m "Autoresearch checkpoint: [summary]" && git push
```

### Research Findings

```markdown
## Research: [Topic]

**Papers**: [List with citations/venues]
**Key insight**: ...
**Implication**: ...
```

---

## Success Metrics

| Metric | Target | Stretch | Result | Status |
|--------|--------|---------|--------|--------|
| C-MAPSS FD001 RMSE | < 13.0 | < 11.0 | **12.22 ± 0.38** | ✅ MET |
| C-MAPSS FD001→FD002 transfer | Role > CI | Beat SOTA | **4.00 vs 6.23 (36% better)** | ✅ MET |
| FactoryNet AURSAD→Voraus | Transfer ratio < 1.5 | < 1.2 | Not tested (data gated) | ❌ BLOCKED |
| Total experiments | > 15 | > 30 | **27** | ✅ MET |

---

## What NOT to Do

- Don't spend >30 min debugging one issue
- Don't skip baselines (Linear, LSTM)
- Don't claim "transfer" without comparing to CI baseline
- Don't forget to log negative results
- Don't modify core `src/industrialjepa/` code unless necessary
- Don't install packages without checking if needed

---

## When You're Done

Update EXPERIMENT_LOG.md with:
1. Summary table of all results
2. Best model configuration
3. Key insights
4. Recommended next steps
5. Honest assessment: Does role-based help transfer?

**The NeurIPS claim (if results support it):**
> "We demonstrate that structured channel-dependence—grouping sensors by physical component with shared within-component weights—enables cross-machine transfer that channel-independent approaches cannot achieve, while maintaining competitive single-dataset performance."

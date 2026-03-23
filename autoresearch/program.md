# Autoresearch Program: Three Paths to Breakthrough

**Mission:** Explore three promising research directions for industrial time series. Thoroughly research, define, and evaluate each. Find the breakthrough.

**Duration:** Full overnight session (8-12 hours)

**Mode:** Deep research first, experiments second. Be thorough, not fast.

---

## Before You Start

Read these files:
1. `.claude/agents/ml_researcher.md` — Your operating instructions (includes deep research mode)
2. `autoresearch/LESSONS_LEARNED.md` — What failed before
3. `autoresearch/EXPERIMENT_LOG.md` — Current results

**Current state:**
- Role-based grouping shows 36% better transfer than CI on C-MAPSS
- Channel-independence wins on single-dataset forecasting
- JEPA as joint objective adds nothing on small supervised data
- The transfer problem is unsolved

---

## The Three Research Directions

### Direction 1: Learned Sparse Graph for Channel Dependence

**Core idea:** Learn which sensors interact (sparse graph), use that structure for forecasting. The graph is interpretable and transferable.

**Key insight:** Full attention (iTransformer) overfits to spurious correlations. Sparse learned graphs capture only true physics, which transfers.

### Direction 2: Learned Latent Concepts (Aggregated Features)

**Core idea:** Learn abstract concepts (like "energy consumption", "joint effort", "system load") as aggregations of raw features. Predict in concept space, then decode back.

**Key insight:** Raw sensors are noisy and machine-specific. Latent concepts might be universal.

### Direction 3: Mechanical-JEPA (JEPA for Physical Systems)

**Core idea:** Apply JEPA (Joint Embedding Predictive Architecture) to industrial time series. Predict future in latent space, not sensor space. The latent space learns physics.

**Key insight:** Predicting raw sensors forces modeling noise. Predicting latent states focuses on dynamics.

---

## Phase 1: Deep Literature Review (4-6 hours)

**This is the most important phase.** Don't rush. For each direction:

1. Find 10+ relevant papers (prioritize: NeurIPS/ICML/ICLR, >50 citations, or from top labs)
2. Understand what exists
3. Identify gaps and shortcomings
4. Document everything

### Direction 1: Sparse Graph Learning for Time Series

**Search queries:**
- "graph structure learning time series"
- "neural relational inference"
- "sparse attention time series"
- "MTGNN graph learning"
- "dynamic graph neural network forecasting"

**Key papers to find and analyze:**
- [ ] NRI (Kipf et al., ICML 2018) — Neural Relational Inference
- [ ] MTGNN (Wu et al., NeurIPS 2020) — Graph WaveNet with learned adjacency
- [ ] GTS (Shang et al., ICLR 2021) — Discrete Graph Structure Learning
- [ ] StemGNN (Cao et al., NeurIPS 2020) — Spectral Temporal Graph
- [ ] DGCRN (Li et al., NeurIPS 2021) — Dynamic Graph for Traffic
- [ ] iTransformer (Liu et al., ICLR 2024) — Inverted attention baseline
- [ ] Autoformer, FEDformer, Crossformer — Recent transformer variants

**Questions to answer:**
1. What graph learning methods exist? (Attention-based, MLP-based, VAE-based?)
2. Do they produce sparse/interpretable graphs?
3. Has anyone validated learned graphs against known physics?
4. Has anyone used learned graphs for TRANSFER?
5. What are the failure modes? (Dense graphs, unstable training, etc.)
6. What's SOTA on which benchmarks?

**Log format:**
```markdown
## Research Log: Direction 1 — Sparse Graph Learning

### Paper: [Title] (Venue Year)
**Authors:** [Names, affiliations]
**Citations:** [Count]
**Core idea:** [1-2 sentences]
**Method:** [How does it work?]
**Results:** [Key numbers]
**Limitations:** [What doesn't work?]
**Relevance to us:** [How does this inform our approach?]

### Gap Analysis
[After reviewing all papers, what's missing?]

### Our Opportunity
[What can we do that hasn't been done?]
```

---

### Direction 2: Learned Latent Concepts

**Search queries:**
- "concept bottleneck models time series"
- "disentangled representation time series"
- "prototype learning time series"
- "slot attention time series"
- "object-centric learning dynamics"

**Key papers to find and analyze:**
- [ ] Concept Bottleneck Models (Koh et al., ICML 2020)
- [ ] Slot Attention (Locatello et al., NeurIPS 2020)
- [ ] Object-Centric Learning (various)
- [ ] β-VAE and disentanglement literature
- [ ] Time series representation learning surveys
- [ ] TS2Vec, TNC, CoST — Contrastive methods
- [ ] Any "concept learning" for sensor data

**Questions to answer:**
1. Has anyone learned interpretable concepts from time series?
2. How do you define "concepts" without supervision?
3. Slot attention — can it discover "joint 1", "joint 2" etc automatically?
4. What about physics-informed concepts (energy, momentum, etc.)?
5. Does concept-based prediction transfer better?
6. How do you evaluate concept quality without ground truth?

**Log format:** Same as Direction 1

---

### Direction 3: JEPA for Physical Systems

**Search queries:**
- "JEPA time series"
- "joint embedding predictive architecture"
- "self-supervised time series forecasting"
- "world models time series"
- "latent dynamics models"
- "video prediction latent space"

**Key papers to find and analyze:**
- [ ] I-JEPA (Assran et al., CVPR 2023) — Original image JEPA
- [ ] V-JEPA (Bardes et al., 2024) — Video JEPA
- [ ] BYOL, SimSiam — Related self-supervised methods
- [ ] World Models (Ha & Schmidhuber, 2018)
- [ ] Dreamer (Hafner et al.) — Latent world models for RL
- [ ] TimeMAE, PatchTST pretraining — Self-supervised TS
- [ ] Any JEPA applied to time series (may not exist!)

**Questions to answer:**
1. Has JEPA been applied to time series? (Probably not — opportunity!)
2. What's the difference between JEPA and contrastive learning for TS?
3. V-JEPA for video — can we adapt this to sensor "video"?
4. World models literature — what works for learning dynamics?
5. What's the right masking strategy for time series JEPA?
6. How does JEPA handle multivariate (multiple sensors)?

**Log format:** Same as Direction 1

---

## Phase 2: Precise Definitions (1-2 hours)

After the literature review, define each approach precisely.

### Definition Template

For each direction, write:

```markdown
## Direction N: [Name]

### Problem Statement
[What exactly are we trying to solve?]

### Formal Definition
[Mathematical formulation]
- Input: ...
- Output: ...
- Objective: ...

### Architecture
[Detailed architecture description]
```python
class ModelName(nn.Module):
    # Pseudocode
```

### Training Procedure
[Step by step: data, loss, optimization]

### What Makes This Novel
[Precise delta from prior work]

### Hypotheses to Test
1. [Hypothesis 1] — How to test it
2. [Hypothesis 2] — How to test it

### Expected Results (if it works)
[What would success look like?]

### Failure Modes (what could go wrong)
[Risks and how to detect them]
```

---

## Phase 3: Evaluation Framework (1 hour)

### Datasets

| Dataset | Domain | Channels | Transfer Scenario | Task |
|---------|--------|----------|-------------------|------|
| **ETTh1/h2/m1/m2** | Electricity | 7 | None (baseline) | Forecasting |
| **Weather** | Meteorology | 21 | None (baseline) | Forecasting |
| **C-MAPSS** | Turbofan | 24 | FD001→FD002→FD003→FD004 | RUL + Forecasting |
| **AURSAD** | Robot | ~30 | Robot1→Robot2 | Forecasting |
| **Voraus** | Robot | ~30 | AURSAD→Voraus | Forecasting |

### Metrics

| Metric | What It Measures |
|--------|------------------|
| **MSE/MAE** | Forecasting accuracy |
| **Transfer Ratio** | MSE(transfer) / MSE(from_scratch) |
| **Graph Sparsity** | % of possible edges used |
| **Graph Interpretability** | Match to known physics |
| **Concept Interpretability** | Can humans understand learned concepts? |
| **Compute Efficiency** | FLOPs, latency, memory |

### Baselines

| Model | Type | Why Include |
|-------|------|-------------|
| **Linear** | Trivial | Lower bound |
| **LSTM** | RNN | Classic sequence model |
| **PatchTST** | Channel-independent | Current paradigm |
| **iTransformer** | Channel-dependent (dense) | SOTA channel-mixing |
| **MTGNN** | Learned graph | Graph learning baseline |
| **Crossformer** | Cross-dimension attention | Another channel-mixing |

---

## Phase 4: SOTA Shortcomings Analysis (1 hour)

For each current SOTA method, analyze:

```markdown
## SOTA Analysis: [Method Name]

### What It Does Well
- ...

### Shortcomings
1. [Shortcoming 1]
   - Evidence: [Paper/experiment showing this]
   - Why it matters: [Impact on real applications]

2. [Shortcoming 2]
   - ...

### How Our Approach Could Fix It
- Direction 1 addresses this by...
- Direction 2 addresses this by...
- Direction 3 addresses this by...
```

**Focus on:**
- Transfer/generalization failures
- Interpretability limitations
- Computational efficiency
- Data efficiency
- Robustness issues

---

## Phase 5: Experiments (If Time Remains)

Only after Phases 1-4 are complete, start experiments.

### Experiment Priority

1. **Minimal viable test of each direction** (1 hour each)
   - Does the core idea work at all?
   - Use smallest dataset (ETTh1)
   - Simple architecture
   - Goal: signal, not SOTA

2. **Best direction deep dive** (remaining time)
   - Whichever direction shows most promise
   - Full evaluation on multiple datasets
   - Ablations

### Experiment Log Format

```markdown
## Exp N: [Description]

**Direction:** 1/2/3
**Time:** HH:MM
**Hypothesis:** [What we expect]
**Setup:** [Model, data, hyperparams]
**Result:** [Numbers]
**Verdict:** PROMISING / NEUTRAL / FAILED
**Insight:** [What we learned]
**Next:** [What this suggests]
```

---

## Deliverables

By morning, produce:

### 1. Literature Review Document
`autoresearch/LITERATURE_REVIEW.md`

### 2. Technical Definitions
`autoresearch/TECHNICAL_SPECS.md`

### 3. SOTA Analysis
`autoresearch/SOTA_ANALYSIS.md`

### 4. Experiment Log
`autoresearch/EXPERIMENT_LOG.md`

### 5. Recommendation
`autoresearch/RECOMMENDATION.md`

---

## Role-Based Transfer Results (from prior sessions)

| Metric | Target | Stretch | Result | Status |
|--------|--------|---------|--------|--------|
| C-MAPSS FD001 RMSE | < 13.0 | < 11.0 | **12.17 ± 0.30** (10 seeds) | ✅ MET |
| C-MAPSS FD001→FD002 transfer | Role > CI | Beat SOTA | **53.73 vs 82.51 (p=0.005, 10 seeds)** | ✅ MET |
| FactoryNet AURSAD→Voraus | Transfer ratio < 1.5 | < 1.2 | Not tested (data gated) | ❌ BLOCKED |
| Total experiments | > 15 | > 30 | **38** | ✅ MET (stretch) |
| Pretraining benefit | JEPA helps | Any pretrain helps | **No pretraining helps** | ❌ NEGATIVE |
| Encoder quality | Role > CI | Frozen transfer | **Role 42% better at 1% data** | ✅ NEW FINDING |

---

## What NOT to Do

- Don't start coding before literature review is done
- Don't skim papers — read thoroughly
- Don't ignore negative results in literature
- Don't get stuck on one direction — cover all three
- Don't sacrifice depth for breadth in experiments
- Don't forget to commit/push every 2 hours

---

## Commit Protocol

Every 2 hours:
```bash
git add -A
git commit -m "Autoresearch checkpoint: [what you accomplished]"
git push
```

---

## When You're Done

Write an executive summary in `autoresearch/EXECUTIVE_SUMMARY.md`:

```markdown
# Executive Summary

## Most Promising Direction
[Direction N: Name]

## Why
[Evidence from literature + preliminary experiments]

## Key Insight
[The one thing that could make this breakthrough]

## Risk Assessment
[What could still go wrong]

## Recommended Next Steps
1. ...
2. ...
3. ...

## Timeline to NeurIPS Submission
[Realistic assessment]
```

---

## Remember

**This is research, not engineering.**

The goal tonight is UNDERSTANDING:
- What exists?
- What's missing?
- What could be breakthrough?

The experiments validate understanding, they don't replace it.

**Think big. Be thorough. Find the breakthrough.**

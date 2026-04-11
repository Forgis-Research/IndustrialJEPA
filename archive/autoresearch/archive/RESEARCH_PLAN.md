# Research Plan: JEPA for Industrial Time Series

Last updated: 2026-03-22

## Goal
NeurIPS-quality paper. Novel architecture for multivariate industrial time series that beats transformers, especially on edge deployment.

## Core Idea
**JEPA with GNN encoder + xLSTM predictor for industrial time series.**

- **GNN encoder**: Online learn a correlation graph between sensors during healthy operation. Captures how sensors relate (e.g., motor current → joint velocity → position). This graph IS the machine's signature.
- **xLSTM predictor**: Predict future latent representations. xLSTM has linear complexity (vs quadratic for transformers) → edge-deployable. Extended LSTM with exponential gating and matrix memory.
- **JEPA framework**: Self-supervised. Predict in latent space, not pixel/signal space. Avoids representation collapse via asymmetric architecture.

### Why This Could Win
1. **Novelty**: No one has combined JEPA + GNN + xLSTM for time series
2. **Practical**: Edge-deployable (xLSTM efficiency), interpretable (GNN graph shows sensor relationships)
3. **Theoretically grounded**: JEPA avoids generative modeling pitfalls; GNN captures relational inductive bias; xLSTM handles long-range dependencies
4. **Industrial relevance**: Graph structure = physical system topology

### Why This Might Fail
1. xLSTM may not outperform transformers enough to matter
2. GNN graph learning may be unstable or not converge
3. JEPA training can be tricky (collapse modes)
4. Combined system complexity — 3 novel components = hard to debug

---

## Phase 1: Establish Baselines on Standard Benchmarks

**Principle**: Prove the architecture on well-established data before claiming anything novel.

### Primary Benchmarks (pick 1-2 to start)

| Dataset | Task | Why | Channels | Established |
|---------|------|-----|----------|-------------|
| **ETTh1/ETTh2** | Long-term forecasting | THE standard benchmark, every TS paper uses it | 7 | Yes |
| **Weather** | Long-term forecasting | 21 channels, physical sensors | 21 | Yes |
| **Electricity** | Long-term forecasting | 321 channels, large scale | 321 | Yes |
| **C-MAPSS** | RUL prediction | Industrial, 21 sensors, hundreds of papers | 21 | Yes |

**Recommendation**: Start with **ETTh1** (small, fast iteration) and **C-MAPSS** (industrial relevance). Both are well-established with clear SOTA numbers to beat.

### Anomaly Detection Benchmarks (Phase 2)

| Dataset | Domain | Channels | Why |
|---------|--------|----------|-----|
| **SWaT** | Water treatment plant | 51 | Industrial, multivariate, physical process |
| **SMD** | Server machines | 38 | Multi-entity (28 machines) |
| **MSL/SMAP** | NASA spacecraft | 55/25 | Standard AD benchmark |
| **PSM** | eBay server | 25 | Real industrial |

### Bearing/Industrial (Phase 3, if results are strong)

| Dataset | Task | Why |
|---------|------|-----|
| CWRU + Paderborn | Fault diagnosis transfer | Most-cited bearing benchmark |
| AURSAD + Voraus | Robot anomaly detection | Our FactoryNet data, novel contribution |

---

## Phase 2: Architecture Experiments

### Step 1: Vanilla JEPA baseline on ETTh1
- Simple transformer encoder + transformer predictor
- Establish our JEPA baseline numbers
- Compare against published PatchTST, iTransformer, TimesNet results

### Step 2: Replace predictor with xLSTM
- Keep transformer encoder, swap predictor to xLSTM
- Measure: accuracy delta, inference speed, memory usage
- xLSTM reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (2024)

### Step 3: Replace encoder with GNN
- Learn adjacency matrix from sensor correlations (healthy data)
- GNN message passing to encode multivariate relationships
- Measure: does explicit graph structure help vs implicit attention?

### Step 4: Full system (GNN encoder + xLSTM predictor + JEPA)
- Only if Steps 2 and 3 each show improvement
- Ablation study: which component contributes what

---

## Phase 3: Industrial Application
- Apply best architecture to C-MAPSS RUL prediction
- Apply to SWaT anomaly detection
- Show edge deployment metrics (FLOPs, latency, memory)
- If strong: apply to FactoryNet for novel cross-machine results

---

## Experiment Protocol

### Rules
1. **Always compare against trivial baselines** (persistence, linear, last-value)
2. **3 seeds minimum** for any reported result
3. **Log only honest results** — if it's not better, say so
4. **One change at a time** in ablations
5. **Question every result**: could a simpler method achieve this?

### Metrics
- Forecasting: MSE, MAE (standard), plus comparison to persistence baseline
- Anomaly detection: F1, AUC-ROC, precision, recall
- Efficiency: FLOPs, inference latency, peak memory
- Always report with confidence intervals

### Engineering Approach (Elon Musk)
1. **Question requirements** — Do we actually need this component? Does the benchmark test what we claim?
2. **Delete unnecessary processes** — No elaborate data pipelines until basic approach works
3. **Simplify/optimize** — Start with smallest viable model
4. **Accelerate cycle time** — Fast iteration on ETTh1 (small dataset, <5 min per run)
5. **Automate last** — Manual runs first, automation only after approach is validated

---

## Files

- `LESSONS_LEARNED.md` — What we tried before and why it didn't work
- `RESEARCH_PLAN.md` — This file (the plan going forward)
- `ALTERNATIVE_DATASETS.md` — Dataset inventory from prior research

# NeurIPS Paper Writing Session — 2h Deep Work

**Date**: 2026-04-09  
**Goal**: Draft a high-quality NeurIPS submission around Mechanical JEPA / grey swan prediction  
**Mode**: Self-supervised writing session (no new model training)  
**Agent**: `paper-writer`  
**Working directory**: `paper-neurips/`

---

## Session Overview

You have **2 hours** for a deep, autonomous paper-writing session. Your output should be a complete paper skeleton with polished Introduction and Related Work sections, a deep literature review document, and all LaTeX scaffolding ready for content.

**What exists already** (read these first):
- `mechanical-jepa/v8/RESULTS.md` — All experimental results (11 methods, cross-dataset transfer, 5-10 seeds)
- `mechanical-jepa/notebooks/08_rul_jepa.qmd` — Full V8 walkthrough with figures and analysis
- `mechanical-jepa/README.md` — Project overview, datasets, architecture
- `autoresearch/LITERATURE_REVIEW.md` — Prior literature review
- `autoresearch/mechanical_jepa/FINDINGS.md` — Key findings summary
- `jepa-lit-review/jepa_sota_review.md` — JEPA SOTA review
- `PORTFOLIO_WRITEUP.md` — Prior writeup (portfolio style, not academic)

**What you are NOT doing**: Training models, running experiments, writing code. This is a pure writing and research session.

---

## Phase 1: Deep Literature Review (45 min)

### 1a. JEPA & Self-Supervised Learning for Time Series (20 min)

Search deeply for papers on:

- **JEPA variants**: I-JEPA (Assran et al. 2023), V-JEPA (Bardes et al. 2024), Brain-JEPA (NeurIPS 2024 Spotlight), any time-series JEPA
- **Self-supervised pretraining for time series**: TS2Vec, TNC, CoST, TS-TCC, SimMTM, PatchTST, TimeMAE
- **Foundation models for time series**: TimesFM (Google 2024), Chronos/Chronos-2 (Amazon 2024-2025), TabPFN-TS, MOMENT, Timer, Lag-Llama
- **Key distinction**: We do NOT predict time series step-by-step. We predict *what matters* — degradation indicators like spectral centroid shift, spectral energy density changes. This is fundamentally different from forecasting the next value.

For each paper extract: key idea, method, results on relevant benchmarks, and the gap our work fills.

### 1b. Remaining Useful Life (RUL) Prediction (15 min)

Search for:

- **Deep learning for RUL**: TCN-Transformer approaches, CNN-LSTM, attention-based methods
- **C-MAPSS benchmark** results (turbofan engines — a natural extension dataset for us)
- **Bearing RUL**: FEMTO/PRONOSTIA (IEEE PHM 2012), XJTU-SY, NASA IMS
- **Current SOTA on bearing RUL**: What architectures? What features? What RMSE/Score?
- **The key gap**: Most methods are fully supervised + require handcrafted features. Self-supervised methods for RUL are rare.

### 1c. Grey Swan / Rare Event Prediction in Physical Systems (10 min)

Search for:

- **Grey swan events in engineering**: near-failures, rare degradation modes, unexpected failure cascades
- **Predictive maintenance** with limited failure data
- **Synthetic data for rare events**: data augmentation for prognostics, physics-informed data generation
- **World models for physical systems**: learned simulators, physics-informed neural networks (PINNs), neural ODEs for degradation modeling
- **Why "grey swan"**: these events are rare but physically plausible (not black swans which are unforeseeable). A component failure after 10,000 operating hours is rare in the training data but follows known degradation physics.

Save the complete literature review to `paper-neurips/literature_review.md`.

---

## Phase 2: Title & Framing (15 min)

### Working Title Candidates

Consider these directions (user preferences):
1. **"Forecasting What Matters: Self-Supervised Representation Learning for Mechanical Grey Swan Prediction"**
2. **"Mechanical JEPA: Learning Degradation-Aware Representations for Remaining Useful Life Prediction"**
3. **"Beyond Time Series Forecasting: Predicting Rare Mechanical Failures with Joint Embedding Predictive Architectures"**

### Framing Principles
- **Don't oversell, but market well.** We have real results, not vaporware.
- **The core insight**: Time series foundation models predict the next value. We predict *what matters* for the downstream task — the spectral signature shift that indicates degradation. This is "forecasting what matters."
- **Honest about limitations**: JEPA alone doesn't beat handcrafted features. But Hybrid (JEPA+HC) achieves +75.5% over time-only baseline. The story is about *combining* self-supervised learning with domain knowledge.
- **The grey swan angle**: Component failures are rare but physically grounded events. Self-supervised pretraining can learn representations from abundant unlabeled run data, then fine-tune with the few available failure examples.
- **Cross-dataset transfer** is the hard, unsolved problem. Our temporal contrastive approach shows +38% over time-only baselines.
- **Planned contributions we will deliver** (write into paper NOW, but mark with `\plannedc{...}` macro so author knows results are pending):
  - Multivariate input (utilize all sensor channels — vibration, current, temperature, torque)
  - Spatiotemporal masking during JEPA pretraining (physics-aware masking by sensor modality)
  - Synthetic training data for grey swan augmentation (physics-informed data generation)
  - Beat current TCN-Transformer SOTA on C-MAPSS turbofan benchmark
  - Additional mechanical domains beyond bearings (turbofan, gearbox, hydraulic)
  - Frequency-domain inputs alongside time-domain (dual-domain encoder)

### Writing Convention for Pending Results

**CRITICAL**: All content in the paper falls into two categories:

1. **DELIVERED** — We have results. Write normally in black.
2. **PLANNED** — We will deliver this but don't have results yet. Use the `\plannedc{...}` LaTeX macro (renders as blue text in draft mode, black in final). In the markdown drafts, wrap with `==PLANNED: ... ==` markers.

For PLANNED content:
- Write the full narrative as if we have the results (present tense)
- Leave `\placeholder{description of what goes here}` for specific numbers, figures, tables
- Write the experimental setup in detail (we know what we're going to do)
- Leave result sentences as: `\plannedc{Our multivariate encoder achieves RMSE of \placeholder{X.XXX}, a \placeholder{XX\%} improvement over single-channel (Table~\ref{tab:multivariate}).}`
- Structure tables with all rows, but use `\placeholder{--}` for missing cells

This way the paper reads as a complete story NOW, and the author just fills in numbers as experiments complete.

### Key Contributions (draft)

1. We introduce a self-supervised framework for mechanical prognostics that learns degradation-aware representations from unlabeled vibration data, transferable across datasets and operating conditions.
2. We show that JEPA pretraining captures complementary features to handcrafted spectral indicators, and their combination (Hybrid JEPA+HC) achieves RMSE 0.055 (+75.5% vs time-only, +20.7% vs Transformer+HC alone).
3. We demonstrate that temporal contrastive learning enables cross-dataset transfer for RUL prediction (FEMTO→XJTU-SY: +38% vs time-only baseline).
4. We provide a mechanistic analysis of what JEPA vs contrastive encoders learn: JEPA captures waveform texture while contrastive captures spectral centroid shift, motivating the hybrid approach.
5. ==PLANNED: We extend to multivariate inputs with spatiotemporal masking, leveraging physics-aware sensor groupings (vibration, current, thermal) to learn richer representations.==
6. ==PLANNED: We validate on C-MAPSS turbofan engines, demonstrating that our framework generalizes beyond bearings to broader mechanical prognostics, beating TCN-Transformer SOTA.==
7. ==PLANNED: We introduce a synthetic grey swan augmentation strategy using physics-informed degradation simulation, enabling better prediction of rare failure modes.==

Save title decision and framing notes to `paper-neurips/framing.md`.

---

## Phase 3: Paper Structure & LaTeX Setup (20 min)

### Create the complete LaTeX project

1. Download/create NeurIPS 2026 style files (use NeurIPS 2024 template as base — `neurips_2024.sty`)
2. Create `paper-neurips/latex/main.tex` with complete structure:
   - All sections as `\input{sections/...}`
   - Bibliography setup
   - Standard packages (booktabs, amsmath, cleveref, etc.)
   - Custom macros for notation
3. Create section files:
   - `sections/abstract.tex` — Draft (mix of delivered + planned)
   - `sections/introduction.tex` — POLISHED (full narrative, planned parts in blue)
   - `sections/related_work.tex` — POLISHED (pure literature, no pending results)
   - `sections/method.tex` — DETAILED skeleton: describe current architecture + planned extensions (multivariate, spatiotemporal masking, synthetic augmentation). Write the method as the full system we're building, with `\plannedc{...}` for components not yet implemented.
   - `sections/experiments.tex` — Structure ALL experiments including planned ones. For delivered results, fill in completely. For planned experiments (C-MAPSS, multivariate ablation, synthetic augmentation), write the setup and leave `\placeholder{...}` for results.
   - `sections/analysis.tex` — Current encoder analysis (delivered) + planned analysis of multivariate representations
   - `sections/conclusion.tex` — Skeleton
   - `sections/appendix.tex` — Skeleton (hyperparameters, additional results, dataset details)
4. Create `paper-neurips/latex/references.bib` with all cited papers
5. Define these custom macros in `main.tex`:
   ```latex
   % Draft mode: planned content renders in blue; final mode: renders in black
   \newif\ifdraftmode \draftmodetrue  % toggle to \draftmodefalse for submission
   \ifdraftmode
     \newcommand{\plannedc}[1]{{\color{blue}#1}}
     \newcommand{\placeholder}[1]{{\color{red}\textbf{[#1]}}}
   \else
     \newcommand{\plannedc}[1]{#1}
     \newcommand{\placeholder}[1]{#1}
   \fi
   ```

### Paper Outline (write to `paper-neurips/outline.md`)

Create a detailed outline with:
- Section and subsection headers
- Key points per subsection (2-3 bullets)
- Which figures go where
- Which tables go where
- Page budget per section (total: 9 pages)

---

## Phase 4: Write Introduction & Related Work (40 min)

### Introduction (~1.5 pages)

**Opening paragraph**: The real-world problem — mechanical component failures are rare but catastrophic ("grey swans" — rare but physically plausible). Predictive maintenance promises to prevent them, but models need data from the very events they're trying to predict. This is the grey swan paradox: the events most critical to predict are the ones with the least training data.

**The gap**: Current time series foundation models (TimesFM, Chronos-2, etc.) forecast values step-by-step. For mechanical prognostics, we don't need to predict every vibration sample — we need to predict the *spectral signature changes* that indicate degradation. This is "forecasting what matters." Most RUL methods require extensive labeled failure data, which is scarce by definition.

**Our approach**: Self-supervised pretraining on abundant unlabeled operational data, learning representations that capture degradation-relevant features without labels. Three synergistic components: (1) JEPA pretraining for waveform texture, (2) temporal contrastive learning for degradation dynamics, (3) ==PLANNED: physics-aware multivariate encoding with spatiotemporal masking==. Combined with domain-informed spectral features in a hybrid architecture.

**The full vision** (write this): ==PLANNED: We work toward a *mechanical world model* — a self-supervised encoder that, given multivariate sensor streams from any mechanical system, learns a latent space where proximity to failure is geometrically encoded. Trained on abundant normal operation + synthetic grey swan augmentation, fine-tuned with minimal failure labels.==

**Contributions**: Numbered list (see Phase 2 above). Write all 7 contributions. For contributions 5-7, use `\plannedc{...}` but write them with full confidence — these are things we ARE doing, just haven't finished yet.

### Related Work (~1 page)

Organize into 3-4 themed subsections:
1. **Self-Supervised Learning for Time Series** (TS2Vec, TNC, PatchTST, TimeMAE, foundation models)
2. **Remaining Useful Life Prediction** (deep learning approaches, bearing/turbofan benchmarks)
3. **Joint Embedding Predictive Architectures** (I-JEPA, V-JEPA, Brain-JEPA — ours is the first for mechanical systems)
4. **Rare Event and Grey Swan Prediction** (if enough literature exists; otherwise fold into intro)

Write these as publication-quality LaTeX. Every paragraph should end with how our work differs.

---

## Deliverables Checklist

By the end of this 2h session, the following files should exist in `paper-neurips/`:

```
paper-neurips/
├── literature_review.md          # Deep lit review with 30+ papers
├── framing.md                    # Title, angle, contributions, honest limitations
├── outline.md                    # Detailed section-by-section outline
├── latex/
│   ├── main.tex                  # Complete LaTeX project
│   ├── neurips_2024.sty          # Style file
│   ├── references.bib            # All references
│   └── sections/
│       ├── abstract.tex          # Draft abstract
│       ├── introduction.tex      # Polished introduction (~1.5 pages)
│       ├── related_work.tex      # Polished related work (~1 page)
│       ├── method.tex            # Skeleton with subsection headers
│       ├── experiments.tex       # Skeleton with tables structure
│       ├── analysis.tex          # Skeleton
│       ├── conclusion.tex        # Skeleton
│       └── appendix.tex          # Skeleton
└── SESSION_PROMPT.md             # This file
```

---

## Quality Bar

This paper should be written to **NeurIPS acceptance standard**:
- Clear, precise prose that a reviewer can follow in one read
- Honest about what we have and what's still needed
- Strong positioning against recent (2023-2026) related work
- Numbers with confidence intervals and statistical tests
- Clean LaTeX that compiles without errors

**Remember**: Don't oversell. The hybrid JEPA+HC result (+75.5% vs time-only) is genuinely strong. The cross-dataset transfer finding is novel. The mechanistic analysis of what JEPA vs contrastive learning captures is scientifically interesting. Tell that story honestly and compellingly.

---

## Key Results to Weave Into the Paper

### In-Domain RUL (from `v8/RESULTS.md`)
| Method | RMSE | vs Time-Only |
|--------|------|-------------|
| Time-only baseline | 0.224 | 0% |
| JEPA+LSTM | 0.189 ± 0.015 | +15.8% (p=0.010) |
| HC+LSTM | 0.177 ± 0.016 | +21.2% |
| Transformer+HC | 0.070 ± 0.006 | +68.9% |
| **Hybrid (JEPA+HC)** | **0.055 ± 0.004** | **+75.5%** |

### Cross-Dataset Transfer
| Direction | JEPA+LSTM | Contrastive+LSTM |
|-----------|-----------|-----------------|
| FEMTO→XJTU (cross) | 0.280 ± 0.007 | **0.227 ± 0.015** |
| XJTU→FEMTO (cross) | 0.403 | **0.309 ± 0.007** |

### Encoder Analysis
| Metric | JEPA | Contrastive | Handcrafted |
|--------|------|------------|------------|
| PC1 corr w/ RUL | 0.186 | **0.648** | 0.585 |
| PC1 corr w/ spectral centroid | 0.071 | **0.856** | 1.000 |

### Planned Experiments (write into paper with `\plannedc{...}`, leave `\placeholder{...}` for numbers)

**Experiment P1: Multivariate Input Ablation**
- Single-channel vibration (current) vs 2-channel vs full 8-channel (Paderborn)
- Table structure: rows = {1ch, 2ch-vib, 4ch-vib+axial, 8ch-full}, cols = {RMSE, ±std, vs 1ch}
- Setup: same JEPA architecture, just change input channels
- `\placeholder{results pending}`

**Experiment P2: Spatiotemporal Masking**
- Random masking (current) vs physics-aware masking (mask by sensor group) vs spatiotemporal (mask time+sensor blocks)
- Paderborn dataset (4 physics groups: radial vib, axial vib, thermal, current)
- Table: rows = {random, physics-group, spatiotemporal, curriculum}, cols = {RMSE, ±std, vs random}
- `\placeholder{results pending}`

**Experiment P3: C-MAPSS Turbofan Benchmark**
- FD001-FD004 subsets, standard train/test split
- Compare against: DCNN, LSTM, TCN-Transformer (current SOTA), HDNN
- Table: rows = {published baselines, our methods}, cols = {FD001 Score, FD002, FD003, FD004}
- `\placeholder{results pending — need to implement C-MAPSS data loader}`

**Experiment P4: Synthetic Grey Swan Augmentation**
- Baseline: train on real failures only
- Augmented: train on real + physics-informed synthetic failures
- Measure: performance on held-out rare failure modes
- `\placeholder{method and results pending — need to design augmentation strategy}`

**Experiment P5: Frequency-Domain / Dual-Domain Input**
- Time-domain only (current) vs FFT magnitude vs dual (time + frequency)
- Especially relevant for vibration data where fault signatures are spectral
- `\placeholder{results pending}`

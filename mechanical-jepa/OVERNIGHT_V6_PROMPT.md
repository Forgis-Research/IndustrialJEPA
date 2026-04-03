# Mechanical-JEPA V6 Overnight Session: The Breakthrough Push

**Date**: 2026-04-03
**Goal**: Turn Mechanical-JEPA from "interesting transfer result" into "groundbreaking new work"

---

## Context: Where We Are

You are continuing an autonomous ML research project on **Mechanical-JEPA** — a self-supervised JEPA (Joint Embedding Predictive Architecture) for industrial vibration fault detection. The project lives in `mechanical-jepa/` within this repo.

### Architecture (V2 — current best)
- **Encoder**: 4-layer Transformer (d=512, 4 heads), ~5.1M params
- **Input**: 1D vibration signals (B, C=3, T=4096) → 16 patches of 256 samples each
- **Pretraining objective**: Mask 62.5% of patches, predict their latent representations from visible context using an EMA target encoder (momentum=0.996)
- **Loss**: L1 in latent space + variance regularization (λ=0.1)
- **Collapse prevention**: sinusoidal positional encoding + high mask ratio + L1 + var_reg (all 5 fixes needed together — ablation in Exp 43 confirmed this)
- **Evaluation**: Throw away predictor, freeze encoder, train linear probe on frozen embeddings
- Code: `src/models/jepa_v2.py` (model), `train_v2.py` (training), `src/data/bearing_dataset.py` (data)

### Key Results So Far (46 experiments across V1-V5)

**Classification (CWRU in-domain)**:
| Method | CWRU F1 | Notes |
|--------|---------|-------|
| Handcrafted + LogReg | 0.999 | CWRU is trivially easy |
| CNN Supervised | 1.000 | Perfect in-domain |
| Transformer Supervised | 0.969 | Good in-domain |
| JEPA V2 (ours) | 0.773 | Self-supervised, no labels |
| MAE | 0.643 | Unstable (0.47-0.90 across seeds) |

**Cross-domain transfer (CWRU → Paderborn, the key metric)**:
| Method | Paderborn F1 | Transfer Gain | Supervision |
|--------|-------------|---------------|-------------|
| CNN Supervised | 0.921 | +0.757 | Supervised |
| **JEPA V2 (ours)** | **0.795** | **+0.453** | **Self-supervised** |
| JEPA V3 (SIGReg) | 0.540 | +0.193 | Self-supervised |
| Transformer Supervised | 0.609 | -0.011 | Supervised |
| MAE | 0.609 | -0.015 | Self-supervised |

**CRITICAL DATA INTEGRITY ISSUE**: The Paderborn transfer numbers for CNN/Transformer/MAE are in `EXPERIMENT_LOG.md` prose only — `transfer_baselines.json` has `pad_f1: null` for everything because `transfer_baselines.py:34` hardcodes `/home/sagemaker-user/...` as the Paderborn path. **Fix the path and re-run to get proper JSON-backed results.**

**RUL / Prognostics (IMS dataset)**:
- JEPA health indicator Spearman: 0.080 (terrible) vs RMS baseline: 0.545
- JEPA early warning: 59.9-70.9% of run remaining (much earlier than RMS)
- RUL regression: All methods lose to constant baseline (predict mean RUL)
- Root cause: JEPA was pretrained on fault classification, NOT on degradation dynamics. The current model predicts masked vibration patches — it never learned what "getting worse over time" looks like.

**Cross-component (bearing → gearbox)**: +2.5% transfer gain (weak)

### Datasets Available
- **CWRU**: 2400 windows, 12kHz, 3ch, 4 classes (bearing faults) — locally downloaded
- **IMS**: 6GB, 20kHz, 8ch, run-to-failure — locally downloaded
- **Paderborn**: 5GB, 64kHz, 8ch multimodal, 4 classes — locally downloaded
- **HuggingFace `Forgis/Mechanical-Components`**: ~12K samples across bearings (10 sources: CWRU, MFPT, FEMTO, IMS, Paderborn, XJTU-SY, Ottawa, MAFAULDA, Mendeley, SCA) and gearboxes (4 sources: OEDI, PHM2009, MCC5-THU, SEU). Access via `pd.read_parquet()` with token `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`. See `hf_cross_component.py` for loading code.
- **FEMTO/PRONOSTIA**: 1.1GB zipped in `datasets/data/femto/`, 25.6kHz, 2ch, run-to-failure with RUL labels

### Key Lessons Learned (Read `LESSONS_LEARNED.md` for full details)
1. Mean-pool over patch tokens (not CLS token) — CLS never gets JEPA gradients
2. Split by bearing, never by window (data leakage)
3. Frequency resampling is ESSENTIAL for cross-domain (CWRU 12kHz → target at 20kHz)
4. EMA target encoder is critical for small datasets (SIGReg V3 fails)
5. Transfer is asymmetric: fault-type datasets transfer better as sources
6. CWRU in-domain F1 is meaningless — always evaluate cross-domain transfer
7. Collapsed models still get decent in-domain F1 but transfer is 3.7x worse
8. CNN transfer wins on absolute F1 but requires labeled source data

---

## PHASE 1: Audit & Consolidate (Do This First)

### Task 1A: Fix Data Integrity
1. Fix `transfer_baselines.py` Paderborn path (line 34) to work with available data
2. Re-run `transfer_baselines.py --seeds 42 123 456` to get JSON-backed Paderborn results for ALL methods
3. Cross-check JSON outputs against EXPERIMENT_LOG.md claims
4. If any numbers don't match, investigate and correct

### Task 1B: Write CONSOLIDATED_RESULTS.md
Create `mechanical-jepa/CONSOLIDATED_RESULTS.md` with:
- Complete comparison table with ALL methods, ALL metrics, JSON source file for each number
- Flag any result that lacks JSON backing
- Per-seed breakdowns for every claim
- Separate sections for: Classification, Transfer, RUL, Cross-Component

### Task 1C: Codebase Health Check
- Read through `src/models/jepa_v2.py`, `train_v2.py`, `src/data/bearing_dataset.py`
- Verify the data split is by bearing (not window)
- Verify mask ratio, loss function, and all V2 fixes are correctly implemented
- Check for any bugs or inconsistencies between training and evaluation
- Document any issues found

---

## PHASE 2: Classification — Complete the Picture

### Task 2A: SOTA Comparison with Proper Splits
Research and implement comparison against published SOTA methods for CWRU bearing fault detection **with proper bearing-level splits** (not window splits). Key references:
- TF-C (NeurIPS 2022): Time-Frequency Consistency
- TS2Vec (AAAI 2022): Universal time series representation
- Any recent vibration SSL papers (2024-2026)

For each: report CWRU F1 and, if possible, cross-domain transfer. The goal is to position JEPA V2 accurately against the field.

### Task 2B: Trivial Baselines Exhaustively
Make sure we have air-tight trivial baselines:
- **Random forest** on handcrafted features (RMS, kurtosis, crest factor, spectral entropy, band energies)
- **XGBoost** on same features
- **1-Nearest Neighbor** on raw windows (DTW or Euclidean)
- **Linear probe on random encoder** (same architecture, no pretraining) — we have this but verify 3-seed

All evaluated on CWRU F1 AND Paderborn transfer gain.

### Task 2C: Few-Shot Transfer Curves
For each method (JEPA V2, CNN supervised, Transformer supervised, random init), measure Paderborn transfer F1 at N = {10, 20, 50, 100, 200, all} labeled target samples. Plot the learning curve. JEPA's advantage should be largest at low N (few-shot regime). This is the key publishable figure.

---

## PHASE 3: RUL — The Missing Piece (Critical for Breakthrough)

The current JEPA pretraining objective predicts masked vibration patches in latent space. This teaches the model about vibration dynamics but NOT about degradation over time. We need to fix this.

### Task 3A: Spectral & Aggregate Feature Prediction (New JEPA Objective)
**Core insight**: Instead of predicting raw vibration patches in latent space, train JEPA to predict **derived features** of the masked patches. These features capture degradation-relevant physics:

1. **Spectral energy in frequency bands**: For each masked patch, compute the target as [energy_0-1kHz, energy_1-3kHz, energy_3-6kHz, energy_6-10kHz]. Bearing faults manifest as energy shifts in characteristic frequency bands.

2. **Statistical moments**: [RMS, kurtosis, crest_factor, skewness] of each masked patch. These are classic degradation indicators.

3. **Envelope spectrum features**: The envelope of the analytic signal captures impulse patterns from fault impacts. Predict envelope RMS and peak frequency.

4. **Multi-target combined**: Predict ALL of the above simultaneously as a rich target vector.

**Implementation approach**:
- Modify the JEPA training loop to compute these features from raw patches BEFORE encoding
- The target encoder still produces latent targets, but now the predictor must predict BOTH the latent target AND the derived feature targets
- Or: create a separate prediction head that takes predictor outputs and maps to feature space
- Or: simply replace the target entirely — instead of predicting latent representations, predict the derived features. This is simpler but loses the latent structure.

Test all three approaches. Evaluate on:
- CWRU classification (should not regress much)
- IMS health indicator Spearman correlation (should improve dramatically)
- IMS RUL regression RMSE

### Task 3B: Degradation-Aware Pretraining on FEMTO
FEMTO (in `datasets/data/femto/`) has actual run-to-failure data with RUL labels at 25.6kHz. 
1. Extract and load FEMTO data
2. Pretrain JEPA on FEMTO vibration signals (self-supervised, no RUL labels used during pretraining)
3. Then evaluate:
   - Zero-shot health indicator on IMS (Spearman, compare to current 0.080)
   - RUL regression on FEMTO held-out runs
   - Transfer to CWRU classification (does degradation pretraining help fault classification?)

### Task 3C: Temporal Context JEPA (Order Matters for RUL)
Standard JEPA masks spatial patches. For RUL, temporal ordering is critical. Design a variant:
- Input: a SEQUENCE of windows from a single run (e.g., 10 consecutive windows spanning hours/days)
- Mask: randomly mask some windows in the sequence
- Predict: the latent representation of masked windows from surrounding windows
- This teaches the model about temporal progression of vibration patterns

This is like Video-JEPA but for degradation trajectories. The encoder should learn "what comes next in a degradation sequence."

---

## PHASE 4: Cross-Component & Multi-Source (The Breakthrough Experiments)

### Task 4A: Gear → Bearing and Bearing → Gear
Using the HF Mechanical-Components dataset:
1. **Pretrain on gearbox data only** (MCC5-THU: 956 samples, 12.8kHz, 8 fault types)
2. **Evaluate on bearing fault classification** (CWRU, Paderborn)
3. **Compare**: gear-pretrained vs bearing-pretrained vs random init
4. Also do the reverse: bearing-pretrained → gearbox classification

The hypothesis: vibration physics (impulses, resonances, harmonics) transfers across component types because the underlying wave propagation is similar.

### Task 4B: Multi-Source Pretraining
1. **Pretrain on ALL available bearing data** from HF: CWRU + MFPT + FEMTO + IMS + Paderborn + XJTU-SY + Ottawa + MAFAULDA + Mendeley + SCA (10 sources, ~10K samples)
2. **Compare to CWRU-only pretraining** on:
   - CWRU classification (should it help or hurt?)
   - Paderborn transfer (should improve — more diverse pretraining)
   - IMS RUL (should improve — includes run-to-failure sources)
   - Gearbox transfer (should improve — more diverse vibration patterns)

3. **Then pretrain on EVERYTHING** (bearings + gearboxes, ~12K samples)
4. **Compare multi-component to bearing-only** for bearing classification
   - If mixed pretraining helps bearings, this is the breakthrough: a general-purpose vibration foundation model

### Task 4C: The "Universal Vibration Model" Experiment
If Task 4B shows promise, push further:
1. Use the full HF dataset as pretraining corpus
2. Evaluate zero-shot and few-shot transfer to EVERY held-out source
3. For each target, compare: {universal model, component-specific model, source-specific model, random init}
4. Build a full transfer matrix showing which pretraining strategy works best for which target

---

## PHASE 5: Circle Back & Verify

### Task 5A: Reproduce All Key Claims
After all new experiments, verify:
- Every number in CONSOLIDATED_RESULTS.md has JSON backing
- Every claim is supported by ≥3 seeds
- No result depends on a lucky seed
- Statistical significance (p < 0.05) for all transfer gain claims

### Task 5B: Write the Story
Update `CONSOLIDATED_RESULTS.md` with:
- The complete narrative: what we tried, what worked, what didn't, and why
- A clear "contributions" section: what is new in this work
- Honest limitations section
- A figure plan for a potential paper

### Task 5C: Identify the Breakthrough
After all experiments, identify which result is the strongest:
- Is it few-shot transfer? (JEPA >> supervised at low N?)
- Is it multi-source pretraining? (universal model >> component-specific?)
- Is it RUL from spectral JEPA? (feature prediction >> patch prediction?)
- Is it cross-component transfer? (gear → bearing works?)
- Something unexpected?

Write a 1-paragraph "elevator pitch" for the paper based on the strongest finding.

---

## Execution Notes

### Priority Order
If time is limited, prioritize in this order:
1. **Phase 1** (audit) — 30 min, must do
2. **Phase 3A** (spectral feature JEPA) — this is the most novel idea and highest potential
3. **Phase 4B** (multi-source pretraining) — straightforward and high impact
4. **Phase 2C** (few-shot curves) — easy to run, publishable figure
5. **Phase 4A** (cross-component) — novel but may not work
6. **Phase 3C** (temporal JEPA) — most ambitious, do if time allows
7. **Phase 5** (verify) — do after experiments complete
8. **Phase 6** (notebook + docs) — do LAST but do NOT skip. The walkthrough notebook is a key deliverable.

### Session Duration
This is an **all-night autonomous session**. Work through all phases systematically. Do not stop early — if you finish the priority items, continue with lower-priority experiments. Fill the time with useful work.

### Git Discipline
**Commit and push regularly.** After completing each major task or experiment:
1. `git add` the relevant files (scripts, results JSON, notebook, log updates)
2. `git commit` with a descriptive message following the repo's existing style (see recent commits)
3. `git push` to remote
Do NOT accumulate hours of work without committing. If something crashes, we should lose at most one experiment's worth of work. Aim for a commit every 30-60 minutes.

### Practical Notes
- All scripts run on SageMaker with GPU. Check CUDA availability.
- HuggingFace data: use `pd.read_parquet()` NOT `load_dataset()` (OOM)
- Paderborn needs resampling to 20kHz for any cross-domain work
- Always use 3 seeds minimum (42, 123, 456)
- Log everything to `EXPERIMENT_LOG.md` with timestamps
- Save all results to JSON in `results/` directory
- If a result surprises you (positive or negative), investigate WHY before moving on. The "why" is often more publishable than the result itself.

### Be Creative
The tasks above are a roadmap, not a straitjacket. If you discover something unexpected — a surprising ablation result, a failure mode that reveals something about the model, a connection to the literature — **pursue it**. The best findings in research are the ones nobody planned for. 

Think like a curious scientist: "Why does this work? What would happen if I changed X? Is the conventional wisdom wrong here?" If you have an idea that isn't in this prompt but could be the breakthrough, try it.

### What Would Be Truly Groundbreaking
- A self-supervised model that transfers across component types (bearing → gearbox) with meaningful gain
- A JEPA variant that learns degradation dynamics and produces useful RUL predictions without ever seeing RUL labels
- A universal vibration foundation model trained on diverse mechanical components that outperforms component-specific models on each component type
- Showing that latent prediction > reconstruction (JEPA > MAE) is a general principle for industrial time series, with a clear mechanistic explanation

Any ONE of these would be a strong paper. Two or more and it's a top venue submission.

---

## PHASE 6: Documentation & Walkthrough Notebook

### Task 6A: Comprehensive Walkthrough Notebook
After all experiments are complete, create `mechanical-jepa/notebooks/06_v6_walkthrough.ipynb` — a self-contained, publication-ready Jupyter notebook that walks the reader through EVERY key finding. This is not a log — it's a narrative.

Structure:
1. **Introduction & Motivation** (markdown): What is JEPA? Why vibration fault detection? What's the gap in the literature?
2. **Architecture Diagram** (markdown + code): Visualize the V2 architecture — encoder, predictor, EMA target, masking. Show a concrete example: raw signal → patches → masked → predicted.
3. **Dataset Overview** (code + plots): Show example waveforms from CWRU, Paderborn, IMS, gearbox. Show the frequency spectra. Explain why cross-domain is hard (different sampling rates, bearing geometries, fault physics).
4. **Pretraining Results** (code + plots): Loss curves, embedding visualizations (t-SNE colored by fault type, by dataset, by bearing). Show collapse vs non-collapse (V1 vs V2 prediction heatmaps).
5. **Classification Results** (code + plots): Bar chart of ALL methods on CWRU F1. The "CWRU is trivially easy" argument. Per-class breakdown showing where JEPA struggles vs excels.
6. **The Transfer Story** (code + plots): This is the centerpiece.
   - Bar chart: transfer gain for every method
   - Few-shot learning curves (F1 vs N labeled samples)
   - The "supervised Transformer fails at transfer" finding with explanation
   - Confusion matrices on Paderborn for pretrained vs random init
7. **RUL & Prognostics** (code + plots): If Phase 3 produces results:
   - Health indicator trajectories (JEPA embedding distance over time vs RMS)
   - Spectral JEPA vs standard JEPA for degradation detection
   - Early warning comparison
8. **Cross-Component Transfer** (code + plots): If Phase 4 produces results:
   - Gear → bearing and bearing → gear results
   - Multi-source vs single-source comparison
   - The "universal vibration model" transfer matrix
9. **Ablation Study** (code + table): Mask ratio, loss function, positional encoding, predictor depth — what matters and what doesn't. Include the collapse phase transition plot.
10. **Conclusions & Honest Limitations** (markdown): What worked, what didn't, what we'd do differently. The elevator pitch.

Requirements:
- Every figure must have axis labels, legends, and be publication-quality (use `plt.style.use('seaborn-v0_8-paper')` or similar)
- Load results from JSON files in `results/` — do NOT hardcode numbers
- Include the actual numerical values in markdown tables alongside the plots
- The notebook should be RUNNABLE (all cells execute without error) using saved results — it should NOT require re-training any models
- Keep code cells clean — put helper functions at the top, not inline

### Task 6B: Update EXPERIMENT_LOG.md
Append all V6 experiments with timestamps, results, and verdicts. Follow the existing format exactly.

### Task 6C: Update LESSONS_LEARNED.md
Add any new insights from V6, especially around:
- Spectral feature prediction (if tried)
- Multi-source pretraining dynamics
- Cross-component transfer feasibility
- Any surprising failures or successes

---

## IMPORTANT: Agent Configuration

**You MUST use the `ml-researcher` agent type for ALL experiment design, execution, and analysis tasks.** This agent has the tools and scientific rigor needed for proper ML research. Specifically:

- Use `ml-researcher` for: experiment design, hyperparameter choices, result analysis, literature comparisons, statistical significance testing, ablation studies, and any task requiring ML domain knowledge
- The ml-researcher agent should drive the scientific methodology: proper baselines, seed averaging, significance tests, honest reporting of negative results
- When the ml-researcher identifies a surprising result, it should investigate the mechanism (not just report the number)
- The ml-researcher should maintain a running hypothesis log: what do we expect, what did we observe, what does it mean?

This is a research project. Treat it like one — with rigor, curiosity, and intellectual honesty.

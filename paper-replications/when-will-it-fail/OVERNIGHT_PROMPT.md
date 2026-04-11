# Overnight Session: Replicate A2P ("When Will It Fail?") + NeurIPS-Level Improvements

**Agent:** ml-researcher
**Working directory:** `C:/Users/Jonaspetersen/dev/IndustrialJEPA/paper-replications/when-will-it-fail/`
**Duration:** Multi-hour autonomous session. Loop. Do not stop until it is perfect.
**Previous replications for reference:** `../cnn-gru-mha/`, `../dcssl/`

---

## Context - why this paper, why this session

The IndustrialJEPA project is building a NeurIPS 2026 submission on self-supervised learning for mechanical grey-swan prediction. The core frame is: **predict *when* rare failure events will happen, not just detect them after the fact.** The A2P paper ("When Will It Fail?", ICML 2025) formalises an almost identical problem - **Anomaly Prediction (AP)** - for general time series, and was recently scouted as a closely related work we must understand deeply.

A2P is the third overnight paper replication in this repo:

1. `../cnn-gru-mha/` - CNN-GRU-MHA bearing RUL transfer (Yu et al. 2024, Applied Sciences)
2. `../dcssl/` - Dual-dim contrastive SSL for bearing RUL (Shen et al. 2026, Scientific Reports)
3. `when-will-it-fail/` - **A2P** (this session)

The goals this session are, in order of importance:

1. **Replicate the paper faithfully** using the official code at https://github.com/KU-VGI/AP, matching Table 1 numbers within a few F1 points.
2. **Build a Quarto summary notebook** that walks a reader through the method, the reproduction, failure modes, and our critique. (Quarto `.qmd`, not Jupyter `.ipynb` - this is a repo-wide rule from memory.)
3. **Propose *massive* - not incremental - NeurIPS-level improvements** and actually test a subset of them. A NeurIPS paper needs a new idea, not a 0.5 F1 bump. Be brutal about which ideas are real.

Loop. Do not stop until the replication runs, the notebook renders, and at least one improvement idea has been empirically probed.

---

## Paper summary (so you do not have to re-read the PDF from scratch)

**Task.** Anomaly Prediction: given `X_in in R^(L_in x C)`, predict binary anomaly labels over the *future unobserved* window of length `L_out`. Distinct from forecasting (predict values) and anomaly detection (detect in observed).

**Method - A2P.** Two-stage framework with a shared transformer backbone `theta`:
- **Stage 1 (pretraining):**
  - **Anomaly-Aware Forecasting network (AAF)**: cross-attention module trained with BCE-style loss on synthetic injected anomalies (5 types - seasonal, global, trend, contextual, shapelet - from Darban et al. 2025). It learns the relationship between anomalies in a prior signal and anomalies in the following signal.
  - **Anomaly Prompt Pool (APP)**: `M` learnable (key, prompt) pairs. A three-layer transformer `f_ftr` with a CLS token produces a query, top-N prompts are selected by cosine similarity and attached to the input embedding. Trained with a **Divergence loss** that pushes APP-infused embeddings away from clean embeddings in KL space.
  - Forecasting loss `L_F` trains the shared backbone on both clean and anomaly-injected inputs.
- **Stage 2 (main training, AAF+APP frozen):**
  - `L_AF = g(X_in, X_hat_out) * ||X_hat_out - X_out||^2` - anomaly-probability-weighted forecasting loss. Up-weights abnormal timesteps.
  - `L_R` - reconstruct both prompt-infused and clean inputs back to clean. Forces the backbone to "undo" anomalies at the embedding level.
- **Test time**: forward `X_in` through `theta` -> `X_hat_out` -> reconstruction `X_hat_out^r` -> anomaly score from the discrepancy (AnomalyTransformer scheme, Xu et al. 2022).

**Datasets**: MBA (ECG, 2D), Exathlon (Spark telemetry, 19D x 8 subsets), SMD (server, 38D), WADI (water, 123D).

**Metric**: F1 with tolerance `t`. **No point adjustment** (they explicitly argue PA is wrong for AP). Threshold set via "percentage of anomalies in test data".

**Key numbers (A2P, avg F1 over 4 datasets):** 46.84 (L_out=100), 53.08 (L_out=200), 58.89 (L_out=400). Best baseline combo hovers around 41.

**Architecture cost (Fig 7):** ~2.3-3.6M params, 10-25 GFLOPs, ~1h train on WADI.

See `REPLICATION_SPEC.md` for the full target results table and ablations.

---

## Concrete plan

### Phase 0 - Recon (<= 30 min)

- Clone https://github.com/KU-VGI/AP into `paper-replications/when-will-it-fail/AP/` (git submodule or plain clone - plain clone is fine, add to `.gitignore` if large).
- Read their `README.md`, `requirements.txt`, `main.py`, data loaders, config files, and the key modules (`AAF`, `SAP/APP`, backbone). Write `RECON_NOTES.md` with:
  - Which files implement which section of the paper (AAF = ?, APP = ?, `f_ftr` = ?).
  - What hyperparameters are in the config vs hardcoded.
  - What datasets their scripts expect and in what format (path layout, normalisation, windowing).
  - Any bugs / TODOs / mismatches vs the paper you spot during the read.
  - How the F1-with-tolerance metric is actually implemented (this matters - numbers are meaningless otherwise).
- Inspect `../cnn-gru-mha/` and `../dcssl/` layouts briefly so this replication uses the **same conventions** (REPLICATION_SPEC.md, EXPERIMENT_LOG.md, `results/all_results.json`, `figures/`, etc.).

### Phase 1 - Smallest replication that works (<= 2h)

Target **MBA only** first. It is 2-channel ECG, small, and has the highest signal in the paper's tables.

- Get the MBA dataset. Their repo should contain a `dataset/` folder or a download script. If it uses MIT-BIH SVDB, it can be fetched from PhysioNet; check if the repo provides a preprocessed `.npy` / `.csv`.
- Run their script with L_in=L_out=100, 3 seeds, default hyperparameters. Log everything to `results/mba_official_seed{0,1,2}.json` with: F1, Precision, Recall, threshold, per-seed runtime, loss curves.
- Compare our avg F1 on MBA against the paper's 67.55 +/- 5.62. Log the gap to `EXPERIMENT_LOG.md`.

**Definition of "Phase 1 done":** MBA F1 within 5 points of paper for at least one seed, reproducible via `python run_replication.py --dataset mba` or equivalent.

### Phase 2 - Scale out (<= 3h)

- Add SMD next (38D, server machine). Smaller than WADI, bigger than MBA - a good stress test.
- Exathlon and WADI are larger - only run them if the first two are clean and there is time. **Do not let WADI's multi-hour training eat the budget and leave the rest half-done.**
- For each dataset, for each L_out in {100, 200, 400}, log avg F1 over 3 seeds. Dump to `results/all_results.json` using the same schema as `../dcssl/results/all_results.json`.
- Produce a comparison table (`results/RESULTS_TABLE.md`) that mirrors Table 1 of the paper with three columns: paper, our-run, delta.

### Phase 3 - Ablation sanity checks (<= 1h)

Pick the **two** ablations most load-bearing for the paper's claims:

1. **AAF on / off** (Table 2) - if you remove AAF, F1 should collapse on MBA by ~30 points. This validates that the pretraining signal matters.
2. **Shared backbone on / off** (Table 4) - if backbones are not shared, F1 should drop from 67.55 to ~51.5 on MBA. This validates the unified-architecture claim.

If either ablation does not show the expected direction, our replication is broken - do not move on, fix it.

### Phase 4 - NeurIPS-level improvements (this is the main deliverable) (<= 3h)

Generate **8-12 radical improvement ideas**. The bar: "if this worked, a NeurIPS area chair would care". Not "I tuned dropout". Think about what is *structurally wrong* with A2P. Some seed directions to push on (do not just copy these - brainstorm more):

- **The synthetic-anomaly assumption is the whole ball game.** A2P hinges on Darban et al.'s 5 anomaly types being representative of real failures. In mechanical systems, degradation does not look like "inject a spike". What if we pretrained AAF on real failure trajectories with a **self-supervised JEPA objective** instead of synthetic injection? (We have the infrastructure - see `../../mechanical-jepa/`.)
- **The evaluation metric is soft.** F1-with-tolerance `t` is close to point-adjustment in disguise - you get credit for predicting anywhere within `+/- t`. A harder metric: **lead-time-weighted F1** (earlier correct predictions worth more), or **time-to-failure regression** (how many NeurIPS papers on this problem formalise it as regression vs classification?).
- **APP is a learned prompt pool but pools have known failure modes** - prompt collapse, pool staleness, nearest-neighbour brittleness. Can we replace APP with a **generative anomaly model** (diffusion, flow-matching) that produces anomaly prompts on the fly conditioned on the current signal? This is the obvious 2025 move.
- **Anomaly Prediction is a sequence-to-sequence problem and they use a shared transformer.** Why not a **state-space model (Mamba-2, S5)** that handles long horizons natively? L_out = 400 is already stretching attention.
- **The task definition conflates "when" and "what".** We only predict *if* a timestep is anomalous, not *what kind*. Multi-class AP (which anomaly type) would be a real task extension and has immediate clinical / industrial value.
- **Grey-swan framing.** A2P tests on MBA/Exathlon/SMD/WADI where anomalies are frequent (5-10% of test data). Grey swans are rare (<0.1%). Does A2P's F1 collapse in the rare regime? This is a direct experiment and if the answer is "yes", it is an entire paper.
- **Cross-dataset transfer.** The paper never tests "train on MBA, test on SMD". If A2P has learned anomaly structure, it should transfer. If it has memorised dataset-specific spikes, it will not. This is a 2-day experiment with huge diagnostic value.
- **Calibration.** F1 says nothing about *how confident* the model is. An AP system that flags "80% chance of failure in the next 400 steps" is deployable; one that just says "abnormal" is not. Does A2P's anomaly score calibrate (ECE, reliability diagram)?
- **Foundation-model distillation.** Can a TimesFM / Chronos / MOMENT zero-shot + a tiny anomaly head beat the whole A2P pipeline on this benchmark? If yes, that is a rugging of the contribution.
- **The two-stage training is brittle.** Pretrain-freeze-finetune pipelines always are. Does end-to-end training work? Does curriculum (linear weighting from Phase 1 to Phase 2 losses) work?

For each idea, write a short card in `IMPROVEMENT_IDEAS.md`:

```
## Idea N: <title>

- Why the paper's choice is limiting:
- The radical alternative:
- What we would need to build:
- The smallest experiment that proves / disproves it:
- Risk of it not working:
- If it works, what venue would care (NeurIPS / ICML / Nature Mach Intel)?
```

Then **pick the 2 cheapest-to-test ideas** and actually test them. Log to `results/improvements/`. The goal is not to beat A2P - it is to produce a *signal* that one of these directions is promising.

Strong candidates because of low cost:
- **Grey-swan regime test** - subsample MBA test set to 0.1% anomaly rate, rerun A2P, measure F1 collapse. Costs ~10 min.
- **Cross-dataset transfer** - train on MBA, eval on Exathlon/SMD. Costs one training run.
- **Calibration** - compute ECE / reliability from existing A2P outputs. Costs ~30 min of plotting.

### Phase 5 - Summary notebook (the deliverable the human will read) (<= 1h)

Create `notebooks/a2p_replication_summary.qmd` (**Quarto, not Jupyter**). It should:

1. **Title + authors + venue + TL;DR** at the top (3 sentences).
2. **Section 1: What is Anomaly Prediction?** - the forecasting/AD/AP distinction, Figure 1 reproduced from the paper with a redrawn matplotlib / altair version.
3. **Section 2: The A2P architecture** - prose walkthrough of AAF, SAP, APP with the math. No copy-paste; write it in our own words. Include a clean architecture diagram (can be a Mermaid block or matplotlib schematic).
4. **Section 3: Our replication** - the comparison table (paper vs us, with deltas), loss curves, sanity checks on the two ablations.
5. **Section 4: Where A2P breaks** - the failure modes we found (e.g. grey-swan collapse, cross-dataset failure, calibration issues).
6. **Section 5: Radical improvement ideas** - the 8-12 cards, highlighting which we tested and what the signal was.
7. **Section 6: What a NeurIPS follow-up would look like** - the single best idea, written as a 1-page proposal (problem, method, experiment, evidence-so-far).
8. **Section 7: Replication recipe** - literal commands the reader runs to reproduce our numbers.

Use the repo's Quarto conventions. Code folds. Figures in `figures/`. Do not paste raw JSON; render tables. Confirm with `quarto render notebooks/a2p_replication_summary.qmd` and eyeball the HTML before declaring done.

### Phase 6 - Loop until perfect (<= 1h spare budget)

- Re-read `EXPERIMENT_LOG.md` and the rendered notebook.
- Anything sloppy (missing seed stds, missing confidence intervals, hand-waved claims, broken figures, wrong number of significant figures)? Fix it.
- Check numbers: paper reports `67.55 +/- 5.62` - do we match precision? Do we actually report std over seeds?
- Check honesty: if our replication fell short, say so explicitly, give the gap number, and hypothesise why.
- Commit each major milestone with a clear message: `paper-replications/when-will-it-fail: <phase> complete`.

---

## Quality bar - non-negotiables

1. **Honest numbers.** If we cannot reproduce the paper, the notebook says "we cannot reproduce the paper" with a number for how far off we are. No hiding.
2. **Every claim has evidence.** Either a number, a figure, or a paper citation. No "we believe".
3. **All code runs.** If a script is in the repo, it runs end-to-end on a fresh clone with a single command. Document the command.
4. **Quarto, not Jupyter.** The summary walkthrough is `.qmd`. See repo memory `feedback_quarto.md`.
5. **No em-dashes in written prose.** Use ` - ` (space-hyphen-space). This is a paper-writer rule from repo memory - apply it to the notebook too.
6. **Failure is a first-class citizen.** If an experiment fails, log *why*, not just *that*. The "Where A2P breaks" section is the point, not a footnote.
7. **Mirror the previous replications' layout.** Look at `../dcssl/` - it has `REPLICATION_SPEC.md`, `EXPERIMENT_LOG.md`, `results/` with structured JSON, `figures/`, a compile script. Match it.
8. **No scope creep.** Do not rewrite A2P from scratch in JAX. Do not train a 1B model. The goal is replication + critique + cheap probes, not a new paper tonight.

---

## Reporting format (when the session ends)

Produce a final `SESSION_SUMMARY.md` at the root of `when-will-it-fail/` with:

1. **Bottom line** (3 sentences): did we replicate, what broke, what is the best improvement idea.
2. **Reproduction table**: paper vs ours, per dataset, per L_out, with deltas.
3. **Ablation sanity checks**: pass / fail, numbers.
4. **Where A2P breaks**: bullet list of concrete failure modes with evidence pointers.
5. **Top 3 NeurIPS-level improvements**: ranked with costs and expected wins.
6. **Experiments actually run tonight**: table of (idea, result, conclusion).
7. **Open questions for next session**: what we did not get to and why.
8. **Cost**: approximate GPU hours, approximate agent wall-time, approximate disk footprint.

The notebook and the summary are both required. The notebook is for humans reading the story; the summary is for the next agent session picking up the work.

---

## Final instruction

Loop. If a phase fails, diagnose the root cause and retry. Do not stop until phases 0-5 are complete, the notebook renders, and at least two improvement ideas have empirical results. Budget realistically - MBA before WADI, probes before new architectures, replication before critique. Write as you go so nothing is lost if the session is interrupted.

Good hunting.

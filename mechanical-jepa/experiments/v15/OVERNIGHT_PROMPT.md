# V15 Overnight Session

**Date**: 2026-04-15
**Goal**: Multi-domain grey swan benchmark — metrics, datasets, architectures.
This session advances the paper from C-MAPSS-only to multi-domain. Also:
SIGReg, improved multivariate treatment (iTransformer-style), and
spatiotemporal masking exploration.

**Commit cadence**: Every hour. Push after each commit. Never >2 unpushed.
**W&B**: `wandb.init(project="industrialjepa", tags=["v15", phase_name])`.
Do NOT save model checkpoints (disk), only metrics. Log all hyperparams.
**Output**: `mechanical-jepa/experiments/v15/`. Quarto summary at end:
`mechanical-jepa/notebooks/15_v15_analysis.qmd` — all signal, no noise.
Concise yet complete enough to verify and understand everything done.
**Paper**: Update sparingly. Only add to appendix unless clearly main-body.

---

## Context from V14

V14 established on C-MAPSS FD001:
- V2 encoder (d=256, L=2, 4H, 1.26M params): E2E 14.23±0.4, Frozen 17.81
- Full-sequence target: frozen -2.1 RMSE at 100%, but regresses at 5%
- Cross-sensor attention: frozen 14.98 at 100% (best), brittle at low labels
- From-scratch ablation: pretraining contributes +8.8 to +15.6 RMSE
- Prediction-error anomaly: rho=+0.02 — NOT a reliable signal (V14 honest negative)
- AE-LSTM replication: 16.07±2.8 (vs their reported 13.99 best-of-grid)
- Frozen tracks better than E2E (rho 0.856 vs 0.804)

Key paper pivot (today): grey swan prediction across domains, not just C-MAPSS RUL.

---

## Phase 0 — Metrics Study [~1.5 hours]

**This must happen first.** Before running ANY experiments on new datasets,
establish the right metrics framework.

### 0a. Grey swan metric analysis
Study and document:
1. For **time-to-failure (RUL)**: RMSE vs nRMSE vs NASA Scoring Function vs
   relative accuracy (RA). Which is most appropriate for cross-domain comparison?
   Recommend one primary metric + one secondary.
2. For **anomaly detection**: Non-PA F1 vs PA-F1 vs AUC-PR vs TaPR.
   Read TSAD-Eval (Schmidl et al., VLDB 2022) methodology section.
   Implement a clean evaluation function that computes all four.
3. For **threshold exceedance**: Define operationally using ±3σ from healthy
   baseline (SPC standard). Implement: given raw sensor data + healthy window,
   compute ground-truth time-to-exceedance for each sensor at each timestep.
4. Write a unified evaluation module: `mechanical-jepa/evaluation/grey_swan_metrics.py`
   that takes (predictions, ground_truth, event_type) and returns all relevant metrics.

**Deliverable**: `experiments/v15/phase0_metrics.py` + a short markdown
`experiments/v15/METRICS_REPORT.md` documenting decisions and rationale.

### 0b. Threshold exceedance on C-MAPSS (validation)
Using the existing V2 encoder (already pretrained), compute time-to-exceedance
labels for FD001 (sensor s14, corrected fan speed, ±3σ from cycles 1-50).
Train a frozen linear probe. Report nRMSE. This fills the first TTE cell
in the benchmark table with minimal effort.

---

## Phase 1 — SIGReg Implementation [~2 hours]

Read `paper-replications/LeJEPA/` for the LeJEPA/SIGReg reference.
Also read Balestriero & LeCun 2025 (arXiv:2511.08544).

### 1a. SIGReg loss (ALREADY IMPLEMENTED)
SIGReg is already in `mechanical-jepa/src/models/sigreg.py` (226 lines).
V3 model in `mechanical-jepa/src/models/jepa_v3.py`. Archived training
script in `mechanical-jepa/archive/train_v3_sigreg.py`.

Read the replication spec: `paper-replications/LeJEPA/REPLICATION_SPEC.md`.

**The actual SIGReg** (NOT eigenvalue thresholding):
1. Sample M random unit vectors from S^{D-1} (default M=512)
2. Project all embeddings onto each direction: s_m = a_m^T · z
3. For each 1D projection, compute Epps-Pulley test against N(0,1)
4. Loss = average EP statistic across M directions
5. Total: L = (1-λ) · L_pred + λ · SIGReg(h), default λ=0.05

One hyperparameter (λ). No epsilon. That's the beauty.

**First**: Verify our implementation against official `pip install lejepa`.
Run Experiment B from REPLICATION_SPEC.md. If our `sigreg.py` uses
moments-based approximation instead of EP, switch to official EP test.

### 1b. SIGReg pretraining run
Follow Experiment D from REPLICATION_SPEC.md:
- Three configs: V2-EMA baseline, SIGReg-only (no EMA), EMA+SIGReg
- SIGReg λ=0.05 (paper default), M=512 slices, EP test
- Single encoder for SIGReg-only (target branch uses same encoder + no_grad)
- 200 epochs, same data as V2. 3 seeds each. Log to wandb.

### 1c. Validate loss-performance correlation
Run Experiment C from REPLICATION_SPEC.md: save checkpoints every 5 epochs,
compute both (training loss) and (frozen probe RMSE), measure Spearman ρ.
If ρ ≥ 0.8 → we can do hyperparameter search without downstream probes
(huge advantage for multi-domain benchmark).

---

## Phase 2 — Improved Multivariate Treatment [~2.5 hours]

### 2a. Sensor-as-token with iTransformer-style attention
V14's cross-sensor encoder already exists (`phase3_cross_sensor.py`).
Improve it:

1. **Sensor-token dropout**: During pretraining, randomly drop 20% of sensor
   tokens per timestep. Forces redundancy, improves low-label robustness.
   (V14 found cross-sensor is brittle at 5% — this should help.)

2. **Learnable sensor ID embeddings**: Replace the fixed sensor indices with
   d-dimensional learnable embeddings. This makes the architecture
   permutation-equivariant across sensors.

3. **Attention analysis**: After training, extract and save the cross-sensor
   attention maps (healthy vs degradation phase). Save as JSON for the paper's
   appendix attention map figure.

### 2b. Spatiotemporal masking (Brain-JEPA inspired)
Read `paper-replications/Brain-JEPA/` for reference.
Brain-JEPA uses spatiotemporal masking where "spatial" = brain ROIs.
For us: "spatial" = sensors, "temporal" = timesteps.

**Key insight**: Unlike brain ROIs, sensor ordering is arbitrary.
So we cannot use spatial contiguity priors. Instead:
- Mask random subsets of sensors at random timesteps
- The predictor must predict masked sensor-time patches from visible ones
- This is complementary to our temporal-only trajectory prediction

**Experiment**: Implement spatiotemporal masking as an alternative pretraining
objective. Compare with trajectory prediction on FD001 frozen probe.
If it works, we can combine both objectives.

**Caution**: Don't overinvest here. 1 run to test the concept. If rho < 0.5
on frozen probe, move on.

---

## Phase 3 — New Dataset Adapters [~2 hours]

### 3a. SMAP/MSL adapter
- Download from: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
  or the original NASA source.
- SMAP: 25 channels, 562K train / 427K test timesteps. Anomaly labels provided.
- MSL: 55 channels, 58K train / 73K test.
- Write `mechanical-jepa/data/smap_msl.py`: PyTorch Dataset that loads,
  normalizes, and windows the data. Window size = 100 timesteps (standard).
- Implement anomaly evaluation: prediction error as anomaly score,
  threshold at 95th percentile of validation scores, compute non-PA F1.

### 3b. SWaT adapter
- Download from: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
  (requires registration, may already be in data/).
- 51 sensors, 495K normal + 177K attack timesteps, 41 labeled attacks.
- Write `mechanical-jepa/data/swat.py`: same interface.

### 3c. Quick pretraining feasibility
For SMAP (simplest): pretrain V2 encoder for 50 epochs.
Evaluate anomaly detection using prediction error.
Report non-PA F1. This is the first anomaly result for the paper.

---

## Phase 4 — Brainstorm: Sensor Correlation Structure [~1 hour]

Before deeper experiments, study the sensor correlation structure:

### 4a. Cross-correlation analysis
For each dataset (C-MAPSS, SMAP, SWaT):
- Compute pairwise sensor correlation matrix (Pearson) on training data
- Compute partial correlation (controlling for time trends)
- Identify sensor clusters (hierarchical clustering on correlation matrix)
- Save heatmap figures

### 4b. Implications for architecture
Document findings in `experiments/v15/SENSOR_ANALYSIS.md`:
- Do sensors form natural groups? (If yes → group attention like Chronos)
- Are correlations stable or do they shift during events? (If shift → the
  cross-sensor attention change during degradation that V14 found is the signal)
- Does sensor ordering matter? (It shouldn't — verify by permuting sensor
  order and checking that V2 results don't change)

### 4c. Recommendation
Based on 4a-4b, recommend the best multivariate treatment:
- If sensors are strongly correlated: channel-fusion (V2) may be sufficient
- If sensors form clusters: group-attention (Chronos-inspired)
- If correlations shift during events: sensor-as-token + cross-sensor attention
  (iTransformer-inspired, V14 Phase 3)
- If no structure: channel-fusion default

---

## Phase 5 — Fill the Benchmark Table [~3 hours]

Using the best available architecture (V2 or SIGReg from Phase 1),
run on new datasets from Phase 3:

### 5a. C-MAPSS FD001: threshold exceedance
Already set up in Phase 0b. Just run and record.

### 5b. SMAP anomaly detection
- Pretrain on SMAP train split (25 channels, ~562K steps)
- Anomaly score = prediction error
- Evaluate non-PA F1 on test split
- Compare against MTS-JEPA (33.6) and TS2Vec (32.8) baselines from paper

### 5c. SWaT anomaly detection (if adapter ready)
Same protocol as SMAP.

### 5d. Cross-dataset transfer (stretch goal)
Pretrain on SMAP, evaluate anomaly detection on MSL (different spacecraft).
Does the encoder generalize across spacecraft?

---

## Phase 6 — Session Wrap-up [~1 hour]

### 6a. Results compilation
- Update `experiments/v15/RESULTS.md` with all findings
- Create the Quarto notebook `notebooks/15_v15_analysis.qmd`:
  - Section per phase
  - All figures inline (matplotlib, not wandb screenshots)
  - Key numbers in bold
  - Honest about what worked and what didn't
  - Code-fold: true, self-contained: true

### 6b. V16 plan
Write `experiments/v15/V16_PLAN.md` with open directions.

### 6c. Paper updates (ONLY if super confident)
- Fill benchmark table cells that have verified results
- Add any new dataset descriptions to appendix
- Do NOT rewrite main-body sections — leave that for human review

---

## Priorities if time is short

If the full session can't complete, prioritize in this order:
1. **Phase 0** (metrics) — everything depends on measuring correctly
2. **Phase 3a** (SMAP adapter) — easiest new dataset
3. **Phase 5b** (SMAP anomaly result) — first non-C-MAPSS number
4. **Phase 1** (SIGReg) — key architectural comparison
5. **Phase 4** (sensor analysis) — informs future architecture choices
6. **Phase 2** (improved multivariate) — iterative improvement

If it is too much work for one night, **honestly report that** in the Quarto
notebook and V16 plan. We continue the next session.

---

## Technical reminders

- Python environment: `conda activate jepa` (or whatever is active)
- GPU: check with `nvidia-smi` before starting
- wandb: `wandb.init(project="industrialjepa", tags=["v15", "phaseN"])`
- No model checkpoint saving (save disk). Only log metrics to wandb.
- Commit every ~1 hour with descriptive message
- All code in `mechanical-jepa/experiments/v15/`
- Data adapters in `mechanical-jepa/data/`
- Evaluation module in `mechanical-jepa/evaluation/`

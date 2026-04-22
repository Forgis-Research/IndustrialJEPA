# V15 Overnight Session

**Date**: 2026-04-16
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
5. Total: L = (1-λ) · L_pred + λ · SIGReg(h)

One hyperparameter (λ). No epsilon. No temperature. That's the beauty.
Paper Figure 1 shows λ ∈ {0.04, 0.08, 0.12, 0.16, 0.20} all work
(Spearman ρ = 94.5% across all). Stable range: **1% to 20%**.
Start with λ = 0.05 (5%), sweep {0.02, 0.05, 0.10} if time allows.

**Do we need multiple "views" like LeJEPA's 2 global + 6 local crops?**
No. LeJEPA's prediction loss is `||embedding - center||²` — degenerate
with one view (center = embedding → loss = 0). It NEEDS multiple views.
Our loss is `||predictor(h_t, k) - h_{t+k}||₁` — non-trivial even with
one target because the predictor must bridge the temporal gap using k.
The learning signal comes from the time delta, not from view multiplicity.
**One context + one target per sample is sufficient.** k varies across the
batch, so the full horizon range is covered. Multiple horizons per sample
is a V16 efficiency optimization, not a requirement.

**First**: Verify our implementation against official `pip install lejepa`.
Run Experiment B from REPLICATION_SPEC.md. If our `sigreg.py` uses
moments-based approximation instead of EP, switch to official EP test.

### 1b. SIGReg pretraining — REVISED ARCHITECTURE

**Key architectural decisions from discussion (2026-04-15/16):**

We discussed the current architecture in depth and identified several
changes. Implement these carefully.

#### Current architecture (V2, for reference)
```
Context encoder (CAUSAL transformer):
  x_{1:t}  →  [e_1, ..., e_t]  →  causal attention  →  take LAST token  →  h_past ∈ R^256

Target encoder (BIDIRECTIONAL transformer + attention pool, EMA of context):
  x_{t+1:t+k}  →  [e_{t+1}, ..., e_{t+k}]  →  bidi attention  →  attn pool  →  h_fut ∈ R^256

Predictor (MLP):
  concat(h_past, sinusoidal_PE(k))  →  MLP  →  ĥ_fut ∈ R^256

Loss:  L1(ĥ_fut, sg[h_fut]) + λ_var * relu(1 - std(h_fut))
```

Problems identified:
1. Context encoder is causal — but at inference we have the FULL past.
   Causal mask wastes representational capacity. No reason to restrict.
2. Target encoder only sees future snippet x_{t+1:t+k} — if grey swan
   event occurs at t+3, it's captured, but the target lacks context of
   where the system was BEFORE the event. V14 full-sequence variant
   (target sees x_{1:t+k}) improved frozen by -2.1 RMSE.
3. Single-vector output (last token / attn pool) throws away per-token info.
   But: we WANT trajectory-level representations, not token-level. Each h
   encodes a whole episode (type + length). This is a FEATURE, not a bug.

#### Revised architecture (V15)
```
Encoder (BIDIRECTIONAL transformer, SHARED weights):
  x_{0:t}      →  bidi attention  →  attention pool  →  h_t ∈ R^256
  x_{0:t+k}    →  bidi attention  →  attention pool  →  h_{t+k} ∈ R^256  (sg, no_grad)

Predictor (MLP, horizon-aware):
  concat(h_t, sinusoidal_PE(k))  →  MLP  →  ĥ_{t+k} ∈ R^256

Losses:
  L_pred = ||ĥ_{t+k} - sg[h_{t+k}]||_1
  L_sig  = SIGReg(h_t)              ← on ENCODER output, NOT predictor
  L_total = (1-λ) * L_pred + λ * L_sig
```

**Why bidirectional for context:** At time t, we know the full past.
A causal mask would hide timestep 5 from timestep 3, which is information
we actually have. Bidirectional gives a strictly better encoding.
(Note: this means we can't do autoregressive generation, but we don't
need to — we only need one summary vector for probes.)

**Why full-trajectory targets:** The target h_{t+k} encodes x_{0:t+k},
the whole trajectory up to t+k. The predictor learns: "given where the
system is at time t, what will the full trajectory summary look like
k steps later?" This captures grey swan events whenever they occur.
V14 already validated this: full-sequence target = -2.1 RMSE improvement.

**Why attention pooling for both:** With bidirectional attention, all
token positions are equivalent (no "last token is special" property).
Use a learned pooling query for both context and target encodings.
Currently the target encoder already does this; now the context encoder
should too.

**Why single-vector (trajectory-level) output:** We want h to encode
the CHARACTER of a trajectory (its type, length, degradation state),
not individual timestep details. Different episodes produce different h
vectors. This is the right abstraction for grey swan prediction —
the probe maps trajectory character → event prediction.

#### SIGReg specifics

**SIGReg is computed on h_t (encoder output), NOT ĥ_{t+k} (predictor output).**
The encoder representations are what downstream probes consume.

**Single encoder, no EMA.** With SIGReg preventing collapse, we don't
need a separate target encoder. The target branch uses the SAME encoder
with `torch.no_grad()` to produce h_{t+k}. This halves model parameters.

**Three configs to compare:**

| Config | Target branch | Collapse prevention | λ |
|--------|--------------|--------------------|----|
| V2 baseline (causal ctx) | EMA τ=0.99 | variance regularizer | — |
| V15-EMA (bidi ctx, full-seq tgt) | EMA τ=0.99 | — | — |
| V15-SIGReg (bidi ctx, full-seq tgt) | same encoder, no_grad | SIGReg EP | 0.05 |

M=512 slices. 200 epochs. 3 seeds each. Log all to wandb.
If SIGReg-only works: try λ sweep {0.02, 0.05, 0.10} on best config.
Be critical. Iterate. If something looks wrong, diagnose before moving on.

#### Batch size for SIGReg

C-MAPSS FD001: 100 engines, mean ~206 cycles. With 30 cuts/engine = 3000
training pairs per epoch. Current BATCH_SIZE=4 is too small for SIGReg
(EP test needs the actual embedding vectors, not accumulated gradients).

**Solution: just increase batch size to 64.**

Memory math: model 5MB + batch activations ~13MB + attention ~40MB = ~60MB.
Trivial on any GPU. Batch=4 was a padding convenience, not a memory limit.
Pad variable-length sequences to max-in-batch, use key_padding_mask.

SIGReg then gets 64 h_t vectors (each R^256) per optimizer step —
sufficient for the EP test on 512 random 1D projections.

```python
BATCH_SIZE = 64  # that's it. no gradient accumulation needed.

# Training step:
h_t   = encoder(x_past, past_mask)           # (64, 256)
h_tk  = encoder(x_full, full_mask).detach()  # (64, 256)
h_hat = predictor(h_t, k)                    # (64, 256)

L_pred = F.l1_loss(h_hat, h_tk)
L_sig  = sigreg_loss(h_t)                    # EP test on 64 vectors
L = (1 - λ) * L_pred + λ * L_sig

L.backward()
optimizer.step()
```

### 1c. Validate loss-performance correlation
Run Experiment C from REPLICATION_SPEC.md: save checkpoints every 5 epochs,
compute both (training loss) and (frozen probe RMSE), measure Spearman ρ.
If ρ ≥ 0.8 → we can do hyperparameter search without downstream probes
(huge advantage for multi-domain benchmark).

### 1d. Embedding visualization — the "degradation clock"

This could be a breakthrough visualization for the paper. The idea:

**Hypothesis:** If SIGReg forces isotropic embeddings and the dominant
latent variable is degradation progression, then engine trajectories in
embedding space should trace smooth curves. With isotropy, these curves
might form circular / semicircular arcs — a "degradation clock" where
angular position corresponds to % remaining life.

**Experiments:**
1. Extract h_past for every timestep of every engine (frozen encoder).
2. Apply PCA (2D) and t-SNE (2D) to the full set of embeddings.
3. Color by:
   - (a) %RUL (normalized remaining useful life, 0=failure, 1=healthy)
   - (b) Time-to-threshold-exceedance (sensor s14 ±3σ)
   - (c) Engine ID (to check within-engine trajectory structure)
4. Check: do trajectories form arcs? Does %RUL map to angular position?

**Why this might work:** SIGReg distributes embeddings isotropically
on a sphere. If degradation is the dominant variation, it becomes the
angular coordinate. Healthy engines cluster at one pole, degraded
engines at another, with smooth progression between.

**Why it might NOT work:** The encoder doesn't know about RUL during
pretraining. It encodes "sensor predictability" which correlates with
degradation but isn't identical. The circular structure depends on
degradation being the ONLY slow variable — if operating conditions
or engine-specific offsets also vary slowly, the circle breaks.

**Compare:** V2 (EMA, anisotropic — PC1=47.6%) vs SIGReg (isotropic).
The V2 embeddings will be elongated along PC1. SIGReg embeddings should
be more spherical. If the clock structure appears in SIGReg but not V2,
that's a strong argument for SIGReg.

**Visualization targets:**
- PCA 2D scatter, colored by %RUL → does it look like a color gradient
  along an arc?
- Same for t-SNE
- Polar plot: convert PCA coordinates to (angle, radius), plot angle vs
  %RUL → if linear, the clock works
- Per-engine trajectory plot: draw lines connecting consecutive timesteps
  of individual engines → do they trace arcs?
- If multiple "ages" are encoded: can we extract a %RUL clock AND a
  separate threshold-exceedance clock from different PCA directions?
  (This would show the encoder learns multiple event timescales
  simultaneously, without any labels.)

**Make the plots beautiful.** These could be hero figures for the paper.
Use the Forgis color palette. Save as publication-quality PDF.

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
Find the most suitable financial dataset as well (maybe gold price with covariates or something well-explored and interesting).

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

Also explore how the iTransformer paper does it. We need a principled way to handle multivariate time series — ideally no arbitrary choices but a mathematically grounded approach. Arrive at a conclusion, test it extensively.

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

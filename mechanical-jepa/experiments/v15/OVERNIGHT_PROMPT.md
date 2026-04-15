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
5. Total: L = (1-λ) · L_pred + λ · SIGReg(h)

One hyperparameter (λ). No epsilon. No temperature. That's the beauty.
Paper Figure 1 shows λ ∈ {0.04, 0.08, 0.12, 0.16, 0.20} all work
(Spearman ρ = 94.5% across all). Stable range: **1% to 20%**.
Start with λ = 0.05 (5%), sweep {0.02, 0.05, 0.10} if time allows.

**First**: Verify our implementation against official `pip install lejepa`.
Run Experiment B from REPLICATION_SPEC.md. If our `sigreg.py` uses
moments-based approximation instead of EP, switch to official EP test.

### 1b. SIGReg pretraining — training procedure (BE PRECISE)

**View construction (important):**
- **Global view (context):** from timestep 1 to cut-off t, where t is
  sampled to give a reasonably large window (e.g., t ≥ 50 cycles).
  This is what the context encoder sees: the full history up to t.
- **Local views (targets):** smaller snippets x_{t+1:t+k} with
  k ~ U[5,30]. These are what the predictor must predict in latent space.

**Loss computation (be crystal clear):**
```
# Forward pass
h_past = encoder(x_{1:t})              # context encoder output
h_fut  = encoder(x_{t+1:t+k}).detach() # SAME encoder, no_grad (SIGReg-only)
                                        # OR: ema_encoder(x_{t+1:t+k}) (EMA variant)
h_hat  = predictor(h_past, k)          # predictor output

# Losses
L_pred = ||h_hat - h_fut||_1           # prediction loss (predictor vs target)
L_sig  = SIGReg(h_past)                # regularizer on ENCODER output (NOT predictor)

# Total
L = (1 - λ) * L_pred + λ * L_sig       # λ = 0.05 default
```

**Key:** SIGReg is computed on **encoder output h_past**, NOT on predictor
output. The encoder representations are what downstream probes consume, so
those are what must be isotropic. The predictor is a throwaway component.

**Three configs to compare:**

| Config | EMA | Collapse prevention | λ |
|--------|-----|--------------------|----|
| V2 baseline | τ=0.99 | variance regularizer | — |
| SIGReg-only | none (same encoder, target uses no_grad) | SIGReg EP | 0.05 |
| EMA + SIGReg | τ=0.99 | SIGReg EP | 0.05 |

M=512 slices. 200 epochs. 3 seeds each. Log all to wandb.
If SIGReg-only works: try λ sweep {0.02, 0.05, 0.10} on best config.
Be critical. Iterate. If something looks wrong (loss diverges, collapse),
diagnose before moving on.

**CRITICAL: Batch size for SIGReg.**
Current pretraining uses BATCH_SIZE=4 (variable-length sequences).
SIGReg needs a batch of h_past vectors to compute the EP test —
batch=4 is far too small (O(1/N) bias, need N ≥ 64-128).
**Solution:** Use gradient accumulation to effective batch ≥ 128.
Accumulate h_past vectors across accumulation steps, compute SIGReg
once on the full accumulated batch before optimizer.step().

```python
# Pseudocode for SIGReg with gradient accumulation:
accum_steps = 32  # 32 * 4 = 128 effective batch
h_past_buffer = []
for i, batch in enumerate(loader):
    h_past = model.context_encoder(past, past_mask)
    h_hat = model.predictor(h_past, k)
    L_pred = F.l1_loss(h_hat, h_future.detach())
    (L_pred / accum_steps).backward()  # accumulate pred loss gradients
    h_past_buffer.append(h_past.detach())

    if (i + 1) % accum_steps == 0:
        # Compute SIGReg on accumulated batch
        h_all = torch.cat(h_past_buffer)  # (128, 256)
        L_sig = sigreg_loss(h_all)
        # Add SIGReg gradient (on encoder params only)
        (λ * L_sig).backward()
        optimizer.step()
        optimizer.zero_grad()
        h_past_buffer = []
```

**Architecture note:** The current model produces ONE vector per sample:
- h_past = context_encoder(x_{1:t}) → (B, 256) — last hidden state of causal transformer
- h_future = target_encoder(x_{t+1:t+k}) → (B, 256) — attention-pooled bidirectional
- predictor(h_past, k) → (B, 256) — single-vector prediction
This is vector-to-vector, NOT multi-token prediction. SIGReg operates
on the (B, 256) h_past batch directly.

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
Find the most suitable financlail dataset as well (maybe gold price with covariates or sth that is well explored and interesting).

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

Also explore how iTransformer paper does it. We need to find a super elegant and powerful way to handle multivariate timeseries. Ideally, no random/arbitary choices but a provable optimal way, mathematically. Arriva at a conclusion, test it extensively.

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

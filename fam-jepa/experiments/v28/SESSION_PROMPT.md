# V28 Session — Prediction Improvement + Honest Evaluation + New Datasets

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: OVERNIGHT (12+ hours on A10G). Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v27 SESSION_SUMMARY.md,
`notebooks/28_metric_analysis.qmd` (metric analysis — read this first).

---

## Context

### What we learned in v27

1. **RevIN kills degradation tracking.** Per-window instance normalization
   erases the slow drift that encodes C-MAPSS degradation. Fix: use
   `norm_mode='none'` (global z-score) for lifecycle datasets. v27 result:
   FD001 AUROC at Δt=150 went from 0.55 → 0.86.

2. **Pooled AUPRC is a vanity metric.** A base-rate classifier (ignoring the
   input entirely) scores AUPRC = 0.924 on FD001; our model scores 0.927.
   The metric is 99.7% base rate. Mean per-horizon AUROC is honest.

3. **The model detects, not predicts.** On MBA, predictions are flat around
   arrhythmia onset — no lead time. On C-MAPSS (v27 fix), the model does
   track degradation, but the transition boundary is poorly calibrated.

### Current standings (v27 dense, seed 42)

```
Dataset | FAM best | Chronos-2 | Winner
FD001   | 0.976    | 0.951     | FAM v27 none
FD002   | 0.881    | 0.930     | Chronos-2
FD003   | 0.879    | 0.797     | FAM v27 none
SMAP    | 0.369    | 0.289     | FAM v26 revin
MSL     | 0.195    | 0.218     | Chronos-2
PSM     | 0.456    | 0.436     | FAM v26 revin
MBA     | 0.953    | 0.964     | Chronos-2 (marginal)
```

### What v28 must achieve

1. **Try 3 model improvements** to make predictions genuinely predictive
   (not just detect current state)
2. **Honest metrics everywhere** — per-horizon AUROC, base-rate comparison,
   surface plots for every dataset and every model variant
3. **3 new datasets** to broaden scope
4. **Chronos-2 comparison on everything** with fair protocol
5. **Quarto notebook** with full analysis

---

## Evaluation framework (MANDATORY for every run)

Every experiment in this session MUST report:

### Per-horizon table (primary)

```python
from sklearn.metrics import roc_auc_score, average_precision_score

def report_surface(p_surface, y_surface, horizons, tag=""):
    """Print the standard diagnostic table."""
    # Per-horizon
    for i, h in enumerate(horizons):
        yi, pi = y_surface[:, i], p_surface[:, i]
        if 0 < yi.mean() < 1:
            auroc = roc_auc_score(yi, pi)
            auprc = average_precision_score(yi, pi)
        else:
            auroc = auprc = float('nan')
        print(f"  dt={h:>3}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  pos={yi.mean():.3f}")

    # Pooled
    pooled_auprc = average_precision_score(y_surface.ravel(), p_surface.ravel())

    # Base-rate comparison
    base_rates = y_surface.mean(axis=0)
    import numpy as np
    rng = np.random.RandomState(0)
    p_base = np.tile(base_rates, (y_surface.shape[0], 1)) + rng.normal(0, 1e-6, y_surface.shape)
    base_auprc = average_precision_score(y_surface.ravel(), p_base.ravel())

    # Mean per-horizon AUROC (honest primary)
    valid = [i for i in range(len(horizons)) if 0 < y_surface[:, i].mean() < 1]
    mean_auroc = np.mean([roc_auc_score(y_surface[:,i], p_surface[:,i]) for i in valid])

    print(f"  ---")
    print(f"  Pooled AUPRC: {pooled_auprc:.4f} (base rate: {base_auprc:.4f}, Δ={pooled_auprc-base_auprc:+.4f})")
    print(f"  Mean per-horizon AUROC: {mean_auroc:.4f}")
    print(f"  [{tag}]")
```

### Surface PNG (always)

For every dataset × model variant, render and save:
```
[predicted p(t,Δt)] | [ground truth y(t,Δt)]
```
Linear y-axis. Save to `experiments/v28/results/surface_pngs/`.

### Comparison table

Every result goes into a running comparison table in the notebook:
```
Dataset | Model | Mean h-AUROC | Pooled AUPRC | Δ above base | Δt=1 AUROC
```

---

## Phase 1: New datasets — data loading + Chronos-2 baselines (2.5 h)

### 1a. SWaT (Secure Water Treatment)

- **Domain**: Industrial control system / cybersecurity
- **Source**: iTrust (https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/)
- **Note**: Requires registration. If data is already at `datasets/data/swat/`,
  use it. If not, skip and use HAI instead.
- **Channels**: 51 sensors (flow, pressure, level, pH, conductivity)
- **Rate**: 1/sec
- **Event**: Cyber-physical attacks (36 labeled attacks across 11 days of operation)
- **SOTA**: USAD F1=0.80 (Audibert+ 2020); TranAD F1=0.80 (Tuli+ 2022)
- **Horizons**: {1, 5, 10, 20, 50, 100, 150, 200}
- **norm_mode**: 'revin' (multi-entity anomaly)
- **Loader**: `data/swat.py` exists as stub — implement or check if functional

### 1b. CWRU Bearing (Case Western Reserve University)

- **Domain**: Mechanical vibration / bearing fault
- **Source**: https://engineering.case.edu/bearingdatacenter/download-data-file
- **Channels**: 2 (drive-end accelerometer + fan-end accelerometer)
- **Rate**: 12 kHz (downsample to ~1 kHz for tractability)
- **Event**: Bearing fault onset (inner race, outer race, ball fault)
- **SOTA**: CNN-based: F1 ~0.99 (supervised classification); for RUL/degradation
  tracking, less clear — IMS bearing dataset has RMSE benchmarks
- **Horizons**: {1, 5, 10, 20, 50, 100, 150, 200}
- **norm_mode**: 'revin' (vibration data, shape-based faults)
- **Loader**: Write new `data/cwru.py`. Download .mat files, extract channels,
  segment into fault/healthy runs.

### 1c. WADI (Water Distribution)

- **Domain**: Water distribution / ICS
- **Source**: iTrust (same registration as SWaT)
- **Channels**: 123 sensors
- **Rate**: 1/sec
- **Event**: 15 cyber-physical attacks across 2 days
- **SOTA**: GDN AUROC=0.97 (Deng & Hooi 2021)
- **Horizons**: {1, 5, 10, 20, 50, 100, 150, 200}
- **norm_mode**: 'revin'
- **Loader**: Write new `data/wadi.py`
- **Fallback**: If SWaT/WADI registration not available, use **HAI** (HIL-based
  Augmented ICS Security, NIMS Korea, publicly downloadable from
  https://github.com/icsdataset/hai — 59 channels, 1/sec, labeled attacks)

**For each new dataset:**
1. Write data loader in `data/` following existing patterns
2. Run Chronos-2 baseline (reuse v24 `baseline_chronos2.py`)
3. Run FAM pretrain + pred-FT with appropriate norm_mode
4. Store surfaces, render PNGs, report per-horizon table
5. Report SOTA comparison

---

## Phase 2: Model improvement tries (4 h total)

Run all tries on FD001 (v27 norm_mode='none') and MBA (norm_mode='revin'),
3 seeds each. Compare per-horizon AUROC to v27 baseline.

### Try A: Lag features (1.5 h)

Add lagged values as extra channels in the input, following Lag-Llama
(Rasul et al., 2024). For each timestep t, the token includes values
from the past:

```python
# In data preprocessing, before patching:
lags = [10, 50, 100]  # lag indices
x_lagged = torch.stack([
    x,                              # current value
    torch.roll(x, shifts=10, dims=1),   # x[t-10]
    torch.roll(x, shifts=50, dims=1),   # x[t-50]
    torch.roll(x, shifts=100, dims=1),  # x[t-100]
], dim=-1).reshape(B, T, C * 4)    # (B, T, C*4)
# Zero-pad the first `max_lag` positions
x_lagged[:, :100, C:] = 0
```

The difference between x[t] and x[t-100] encodes the drift gradient
directly in the token. Survives RevIN because it's a within-token feature.

Update the input projection: `nn.Linear(C * 4 * P, d_model)`.

Run on FD001 (with RevIN this time — the lag features should recover
drift even under RevIN) and MBA. If FD001 AUROC improves with RevIN,
this is the universal fix — no more per-dataset norm_mode choice.

### Try B: Auxiliary stat-prediction loss during pretraining (1.5 h)

Add a head during pretraining that predicts the target interval's
raw statistics from h_t, following the LaT-PFN approach (arXiv:2405.10093):

```python
# In pretrain(), after computing h_t:
stat_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                          nn.Linear(d_model, 3 * n_channels))  # mean, std, slope per channel

# Target stats (computed from un-normalized target interval)
target_mean = x_target_raw.mean(dim=1)        # (B, C)
target_std = x_target_raw.std(dim=1)          # (B, C)
# Slope: linear regression coefficient over time
target_slope = ...  # (B, C) — or just (x_target[-1] - x_target[0]) / Δt

stat_pred = stat_head(h_t)
target_stats = torch.cat([target_mean, target_std, target_slope], dim=-1)
loss_stat = F.l1_loss(stat_pred, target_stats.detach())
loss = loss_main + 0.1 * loss_stat
```

This forces the encoder to encode distributional state even under RevIN.
Run on FD001 (with RevIN) and MBA. If FD001 AUROC improves, the encoder
has learned lifecycle-relevant representations.

**IMPORTANT**: The target stats must be computed from RAW (un-normalized)
data, not from RevIN-normalized data.

### Try C: Dense horizon finetuning (1 h)

Currently: finetune on 7 sparse horizons {1,5,10,20,50,100,150}.
Change: sample random horizons from 1..150 each batch.

```python
# In finetune(), each batch:
K_sample = 20  # horizons per batch
horizons_batch = torch.sort(torch.randint(1, max_horizon+1, (K_sample,)))[0]
# Run predictor at these horizons, compute loss
```

At evaluation: compute the full 1..150 dense surface.

Run on FD001 (v27 baseline, norm_mode='none'). Expected: sharper
probability transition at the failure boundary.

---

## Phase 3: Comprehensive benchmark (3 h)

Run the best variant(s) from Phase 2 on ALL datasets:

| Dataset | norm_mode | Horizons | Est. |
|---------|-----------|----------|------|
| FD001 | none (or revin if Try A works) | 1..150 | 15 min |
| FD002 | same | 1..150 | 15 min |
| FD003 | same | 1..150 | 15 min |
| SMAP | revin | 1..200 | 20 min |
| MSL | revin | 1..200 | 20 min |
| PSM | revin | 1..200 | 15 min |
| SMD | revin | 1..200 | 30 min |
| MBA | revin | 1..200 | 15 min |
| SWaT/HAI | revin | 1..200 | 20 min |
| CWRU | revin | 1..200 | 15 min |
| WADI/HAI2 | revin | 1..200 | 20 min |

3 seeds each. Store all surfaces as .npz ON THE VM.

For every dataset, also run Chronos-2 if not already done in Phase 1.

---

## Phase 4: Surface comparison PNGs (1 h)

For EVERY dataset, render:
```
[FAM best p(t,Δt)] | [Chronos-2 p(t,Δt)] | [ground truth y(t,Δt)]
```
Linear y-axis. Viridis colormap. Same scale (0-1).

For C-MAPSS: pick the longest test engine.
For anomaly datasets: pick entity with clearest anomaly segment.
For MBA: same recording tail as v27.

Save all to `experiments/v28/results/surface_pngs/`.
Push PNGs only (not .npz).

---

## Phase 5: Quarto analysis notebook (1.5 h)

Create `notebooks/28_v28_analysis.qmd`. Use the **data-curator agent**.

### Required sections:

#### 1. Metric framework recap
- Link to `28_metric_analysis.qmd` for full walkthrough
- Summary: mean per-horizon AUROC is primary, pooled AUPRC with base-rate context

#### 2. Model improvement results
- Per-horizon AUROC table for each Try (A, B, C) vs v27 baseline
- For each try: one surface plot (FD001) showing predicted vs ground truth
- Verdict: which try worked and why

#### 3. Master comparison table
All datasets × all models:
```
Dataset | Domain | FAM best | Chronos-2 | SOTA method | SOTA metric | SOTA value |
        |        | h-AUROC  | h-AUROC   |             |             |            |
        |        | (Δ base) | (Δ base)  |             |             |            |
```

#### 4. Surface gallery
Every dataset: FAM | Chronos-2 | ground truth (3-panel).
Brief annotation: what the model gets right, what it misses.

#### 5. Detection vs prediction analysis
For each dataset, compute lead-time recall:
```python
# For each event onset, does p exceed threshold `gap` steps BEFORE onset?
for gap in [0, 5, 10, 20, 50]:
    recall = fraction of onsets where p(t_onset - gap, Δt=1) > threshold
```
Table: lead-time recall at gap=0 (detection) vs gap=10 (prediction).

#### 6. Per-dataset SOTA comparison
For each dataset, compare FAM to the published SOTA on THEIR metric:
- C-MAPSS: RMSE vs STAR
- SMAP/MSL/PSM/SMD: PA-F1 vs MTS-JEPA / Anomaly Transformer
- MBA: F1 vs InceptionTime
- New datasets: whatever the published metric is

#### 7. Honest assessment
- Where FAM genuinely predicts (C-MAPSS v27, maybe new datasets)
- Where FAM only detects (MBA)
- Where FAM is weak (MSL, GECCO)
- What the model improvement tries achieved
- Open problems

Render: `quarto render notebooks/28_v28_analysis.qmd`

---

## Phase 6: Self-check with ml-researcher agent (30 min)

Before finalizing, launch the **ml-researcher agent** to review:
1. Are any claims unsupported by the data?
2. Are the SOTA comparisons fair (same splits, same metric)?
3. Any statistical concerns (N too small, high variance)?
4. Is the metric framework sound?

Incorporate feedback into the notebook.

---

## Phase 7: Update RESULTS.md + commit (30 min)

- v28 section in RESULTS.md with master table
- Commit all PNGs, notebook, results JSONs
- Push

---

## Phase priorities

| Phase | What | Est. | Priority |
|-------|------|------|----------|
| 1 | New datasets + Chronos baselines | 2.5 h | Critical |
| 2 | Three model improvement tries | 4 h | Critical |
| 3 | Full benchmark with best variant | 3 h | Critical |
| 4 | Surface comparison PNGs | 1 h | Important |
| 5 | Quarto analysis notebook | 1.5 h | Critical |
| 6 | Self-check with ml-researcher | 30 min | Important |
| 7 | RESULTS.md + commit | 30 min | Always |

**Total**: ~13h. This is an overnight session.

---

## Ground rules

1. **Import from model.py and train.py.** Do NOT copy model code.
2. **finetune_forward returns CDF probabilities.** Do NOT apply sigmoid.
3. **P=16 everywhere.** No exceptions.
4. **norm_mode**: 'none' for C-MAPSS (unless Try A fixes it), 'revin' for
   everything else.
5. Store surfaces as .npz ON THE VM. Push only PNGs to git.
6. **PRIMARY METRIC**: mean per-horizon AUROC. Always report base-rate
   comparison for pooled AUPRC.
7. **ALWAYS render surface PNGs** for every dataset × model variant.
8. Use **data-curator agent** for the notebook.
9. Use **figure-creator agent** for publication-quality plots.
10. Use **ml-researcher agent** for self-check.
11. Commit + push after each phase. Update RESULTS.md.
12. **Stay transparent.** If a try fails, document WHY. If numbers are weak,
    say so. No cherry-picking.

---

## Success criteria

1. At least one model improvement try shows measurable gain in mean
   per-horizon AUROC on FD001 (>0.93, up from 0.924).
2. New datasets successfully loaded, benchmarked, and compared.
3. Every dataset has FAM + Chronos-2 + ground truth surface PNGs.
4. Master comparison table is complete and honest.
5. Lead-time analysis quantifies detection vs prediction per dataset.
6. Notebook tells a coherent story supported by the data.

---

## Literature references

- **Lag-Llama** (Rasul+ 2024, arXiv:2310.08278): Lagged features as tokens,
  captures drift within the token vector.
- **LaT-PFN** (arXiv:2405.10093): Auxiliary system-identification head during
  JEPA pretraining. Precedent for our stat-prediction loss.
- **Non-stationary Transformers** (Liu+ NeurIPS 2022): Over-stationarization
  concept. De-stationary Attention.
- **NLinear** (Zeng+ AAAI 2023): Last-value subtraction. Failed in v27 but
  concept is valid.
- **Chronos** (Ansari+ 2024): Mean-absolute scaling (multiplicative, not
  additive). Preserves drift as ratio.

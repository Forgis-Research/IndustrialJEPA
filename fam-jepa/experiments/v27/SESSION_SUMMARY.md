# V27 Session Summary

**Date**: 2026-04-24
**Duration**: ~2 hours on A10G (all phases completed; compute was not the bottleneck)
**Scope**: Normalization ablation on FD001 + C-MAPSS full-benchmark + anomaly regression check + paper-quality surfaces + RESULTS.md + Quarto notebook

## One-sentence verdict

V26's pooled AUPRC hid a chance-level per-horizon AUROC on C-MAPSS
(0.52 at Δt=10); the root cause is RevIN erasing slow sensor drift,
the fix is train-set global z-score with `norm_mode='none'` on the
degradation family (**+0.127 pooled AUPRC on FD003**, +0.308 AUROC at
Δt=150 on FD001), and the fix does NOT transfer to multi-entity
anomaly datasets — SMAP regresses -0.117 AUPRC — so `norm_mode`
becomes a per-dataset-family choice in the paper's central table.

## What shipped

### Architecture change (`fam-jepa/model.py`)

Added `norm_mode` to `CausalEncoder`, `TargetEncoder`, and `FAM`:

| Mode | Behavior |
|------|----------|
| `'revin'`       | Per-context per-channel instance norm (v24/v26 default, unchanged) |
| `'none'`        | Model skips normalization entirely - user pre-normalizes data |
| `'last_value'`  | Subtract last valid timestep per channel (NLinear) |
| `'revin_stat'`  | RevIN + project (μ, σ) to a "stat token" at position 0 of the causal context; target encoder falls back to plain RevIN |

Default is `'revin'` so all v24 / v26 checkpoints load bit-identically.
Encoder-side stat token adds 2×C×d parameters (7,424 extra params for
C=14, d=256).

### FD001 ablation (Phase 2-4, 3 seeds each)

| norm_mode | Δt=10 AUROC | Δt=50 AUROC | Δt=150 AUROC | Pooled AUPRC | Gap @ Δt=50 |
|-----------|-------------|-------------|--------------|--------------|--------------|
| v26 `revin`      | 0.520 | 0.526 | 0.549 | 0.925 ± 0.001 | +0.005 |
| v27 `none`       | **0.639** | **0.789** | **0.857** | **0.946 ± 0.001** | **+0.352** |
| v27 `last_value` | 0.531 | 0.512 | 0.490 | 0.925 ± 0.000 | +0.005 |
| v27 `revin_stat` | 0.495 | 0.522 | 0.549 | 0.919 ± 0.003 | +0.012 |

The prediction-gap `p̄(y=1) - p̄(y=0)` at Δt=50 is the clearest
diagnostic: only `none` climbs out of the base-rate floor.

**Why `last_value` fails**: subtracting the last observed value anchors
every context to zero at the right endpoint. For a lifecycle dataset,
that endpoint differs between healthy and failing contexts and encodes
the signal — anchoring it to zero wipes that out.

**Why `revin_stat` fails**: during pretraining the target encoder
RevIN-normalizes the target interval, so the stat token has no gradient
signal to become informative — the pretraining loss doesn't reward
using μ/σ in the context representation. The stat_proj ends up
essentially random. (De-stationary Attention from Liu+ NeurIPS 2022
resolves this by injecting stats into the attention computation; our
token-injection shortcut misses that path.)

### C-MAPSS full benchmark (Phase 6)

| Dataset | v26 revin AUPRC | v27 none AUPRC | Δ AUPRC | v26 Δt=150 AUROC | v27 Δt=150 AUROC | Δ AUROC |
|---------|------------------|----------------|---------|--------------------|--------------------|---------|
| FD001 | 0.925 ± 0.001 | **0.946 ± 0.001** | +0.021 | 0.549 | **0.857** | +0.308 |
| FD002 | 0.908 ± 0.001 | **0.910 ± 0.000** | +0.002 | 0.514 | 0.525 | +0.011 |
| FD003 | 0.774 ± 0.000 | **0.901 ± 0.005** | **+0.127** | 0.498 | **0.885** | +0.387 |

**FD003 is the headline**: +0.127 pooled AUPRC gain, +0.387 AUROC at
Δt=150. The worst C-MAPSS subset in v26 now behaves like the strong ones.

FD002 barely moves: its multi-condition operating regime already imposes
its own normalization, so removing RevIN is neutral.

### Regression check on anomaly datasets (Phase 5)

| Dataset | v26 revin AUPRC | v27 none AUPRC | Δ | Verdict |
|---------|------------------|----------------|---|---------|
| MBA  | 0.950 ± 0.001 | 0.946 ± 0.002 | -0.004 | tie |
| SMAP | 0.393 ± 0.010 | 0.276 ± 0.002 | **-0.117** | broken |

MBA is within noise. SMAP is a clean regression: 55 telemetry entities
have such heterogeneous per-channel scales that a single global
(μ, std) from concatenated train collapses the entity structure and
predictions drop to base rate across every horizon.

**Consequence for the paper**: `norm_mode` is a per-dataset-family
choice, not a universal recipe. The story is:

- Degradation / slow-drift signal (C-MAPSS): `norm_mode='none'` + train-set global z-score.
- Multi-entity anomaly signal (SMAP/MSL/PSM/SMD/MBA): `norm_mode='revin'` (v26 default).
- PhysioNet 2012 (ICU, P=1): `norm_mode='revin'` (deferred - unchanged from v26).

### Paper-quality surfaces (Phase 7)

Dense per-entity surfaces for Figure 3 of the paper. For each FD001
engine, both v26 and v27 surfaces so the figure can show a side-by-side:

| Entity | v26 revin pooled AUPRC | v27 none pooled AUPRC | Δ |
|--------|------------------------|------------------------|---|
| FD001 engine 49 (T=303 cycles) | 0.581 | **0.892** | **+0.311** |
| FD001 engine 93 (T=244 cycles) | 0.822 | **0.972** | +0.150 |
| FD001 engine 91 (T=234 cycles) | 0.862 | **0.990** | +0.128 |
| MBA test-stream tail | — | 0.953 (v26 revin) | — |
| SMAP entities E-5 / E-1 / T-1 | — | v26 revin, various | — |

All 9 surfaces saved at `experiments/v27/surfaces/paper_*.npz`.
Materialized PNG + PDF heatmaps at `notebooks/plots/27_*.{png,pdf}`.

## Provenance

Every v27 number above has a JSON at `experiments/v27/results/` or a
surface at `experiments/v27/surfaces/`. Analysis notebook:
`notebooks/27_v27_analysis.qmd` (rendered to `27_v27_analysis.html`).

| Artifact | Path |
|----------|------|
| Phase 1 v26 diagnostic | `results/phase1_v26_baseline_diagnostic.json` |
| Phase 2 FD001 none | `results/phase2_FD001_none.json` + surfaces |
| Phase 3 FD001 last_value | `results/phase3_FD001_last_value.json` + surfaces |
| Phase 4 FD001 revin_stat | `results/phase4_FD001_revin_stat.json` + surfaces |
| Phase 5 MBA/SMAP none | `results/phase5_{MBA,SMAP}_none.json` + surfaces |
| Phase 6 FD002/FD003 none | `results/phase6_{FD002,FD003}_none.json` + surfaces |
| Phase 7 paper surfaces | `surfaces/paper_*.npz` (9 files) |
| Phase 8 | `RESULTS.md` v27 section, `notebooks/27_v27_analysis.{qmd,html}` |

## What did not ship

- **PhysioNet 2012 v27 rerun**: deferred. v26 behavior is already
  strong (AUROC 0.895); no FD001-style diagnostic suggests degradation
  mismatch on ICU data. If needed, `run_one('PhysioNet', 'revin', 42)`
  would just reproduce v26 numbers under the v27 codebase.
- **GECCO / BATADAL / SWaT**: these were optional in the Phase 6 list
  but require loader work. Reusable v26 surfaces exist for the
  anomaly datasets that matter for the main table.
- **Chronos-2 head-to-head**: unchanged from v26. C-MAPSS surfaces
  changed (v27 beats them on FD001 and especially FD003), so the v26
  Chronos-2 vs FAM table needs a new column comparing Chronos-2 to
  v27-FAM. Noted as a paper follow-up.
- **Two-loss variant for revin_stat**: stat_proj could be trained by
  adding a second pretraining head that predicts the TARGET's (μ, std)
  from h_t. This would give the stat token a gradient signal that
  survives target-side RevIN. Out of scope for this session; a clean
  follow-up if `norm_mode='revin_stat'` is worth rescuing.

## Commits this session

- `90aeb57` v27 phases 1-4: FD001 ablations
- `13c6713` v27 phase 5: MBA/SMAP regression
- `1f7ae00` v27 phase 6: FD002/FD003 full benchmark
- `80f98e6` v27 phase 7: paper surfaces

Plus this commit with RESULTS.md + notebook + SESSION_SUMMARY.

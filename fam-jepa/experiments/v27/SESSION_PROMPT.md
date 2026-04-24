# V27 Session — Over-Stationarization Fix

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: 6-8 hours on A10G. Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v26 SESSION_PROMPT.md

---

## Why this session exists

The model **cannot discriminate degradation state on C-MAPSS**.
Per-horizon AUROC on FD001 at Δt=10 is 0.52 (coin flip). The model learns
`P(event | Δt)` — the marginal base rate — but NOT `P(event | Δt, h_t)`.
Predictions are identical for healthy and failing engines.

In v21 (P=1, global normalization, 790K predictor), AUROC at Δt=10 was 0.986.
In v24/v26 (P=16, RevIN, 198K predictor), it collapsed to 0.54.

**Root cause: over-stationarization** (Liu et al., NeurIPS 2022,
"Non-stationary Transformers"). RevIN removes per-window mean and std.
For degradation datasets, the signal IS the slow drift of sensor means.
RevIN erases it before the encoder sees it.

This is a known failure mode in the normalization literature:
- Non-stationary Transformers (Liu+ NeurIPS 2022): identifies the problem,
  proposes De-stationary Attention — inject statistics-derived factors
  (τ, Δ) into the attention computation
- "On the Role of RevIN" (2025): explicitly warns RevIN "discards
  potentially predictive context" when the mean carries information
- NLinear (Zeng+ AAAI 2023): subtracts only the last observed value
  (not the mean), preserving slow drift between windows
- Every foundation model (Chronos, TimesFM, MOMENT) uses mean scaling
  or instance norm and shares the same blind spot for prognostics

Evidence from v24 pooled AUPRC (P=16+RevIN vs P=1+global):
```
Dataset  | v21/22 (P=1, glob) | v24 (P=16, RevIN) | delta
FD001    |  0.945             |  0.926            | -0.019
FD003    |  0.903             |  0.766            | -0.138
SMAP     |  0.289             |  0.395            | +0.107
MBA      |  0.784             |  0.947            | +0.163
```

Pooled AUPRC masked the real damage. Per-horizon AUROC tells the truth:
```
FD001 dt=10:  v21 = 0.986    v24 = 0.547    v26 = 0.538
```

P=16 is worth keeping (big wins on MBA +0.163, SMAP +0.107). The fix
must target normalization only.

---

## Architecture (unchanged from v26)

Use `model.py` and `train.py` as-is. The only changes in this session are
to the normalization mode, passed as a parameter.

**IMPORTANT**: `finetune_forward` returns CDF probabilities, NOT logits.
Do NOT apply sigmoid. Do NOT use BCEWithLogitsLoss.

---

## Datasets

Focus on 4 datasets that span the problem:

| Dataset | Domain | Signal type | RevIN effect | Horizons |
|---------|--------|-------------|-------------|----------|
| FD001 | Turbofan | Drift | **Erased** | {1,5,10,20,50,100,150} |
| FD003 | Turbofan | Drift (multi-fault) | **Erased** | {1,5,10,20,50,100,150} |
| SMAP | Spacecraft | Local pattern | Preserved | {1,5,10,20,50,100,150,200} |
| MBA | Cardiac | Waveform shape | Preserved | {1,5,10,20,50,100,150,200} |

---

## Phase plan

### Phase 1: Baseline reproduction (30 min)

Run FD001 with current architecture (P=16, RevIN, 198K predictor), 3 seeds.
Verify per-horizon AUROC ≈ 0.52 at Δt=10. Store surface as .npz.

**Diagnostic to report for EVERY run in this session:**
```python
# After evaluate(), print this for each run:
for i, h in enumerate(horizons):
    auroc_h = roc_auc_score(y_surface[:, i], p_surface[:, i])
    auprc_h = average_precision_score(y_surface[:, i], p_surface[:, i])
    print(f"  dt={h:>3}: AUROC={auroc_h:.3f}  AUPRC={auprc_h:.3f}  pos={y_surface[:,i].mean():.3f}")
```

### Phase 2: Ablation A — No RevIN, global z-score (1 h)

Replace RevIN with global z-score normalization on FD001.

Compute train-set mean/std per channel, apply at load time:
```python
train_mean = X_train.mean(dim=0)  # (C,)
train_std  = X_train.std(dim=0).clamp(min=1e-5)
X = (X - train_mean) / train_std
```

Skip RevIN in the model by adding `use_revin: bool = True` to FAM and
gating the RevIN call. Run FD001, 3 seeds.

**Expected**: AUROC at Δt=10 jumps from ~0.52 to >0.7.

### Phase 3: Ablation B — NLinear-style normalization (1 h)

Subtract only the **last observed value** per channel (not the mean),
following Zeng et al. (AAAI 2023). This removes local offset while
preserving the within-window trend:

```python
# Replace RevIN with:
last_val = x[:, -1:, :]          # (B, 1, C) — last timestep
x_norm = x - last_val            # preserves within-window dynamics
# No std division — only shift, no scale
```

Run FD001, 3 seeds. Compare to Phase 1 and 2.

### Phase 4: Ablation C — RevIN + statistics as features (1.5 h)

Keep RevIN but inject the removed statistics back into the model,
following the principle of De-stationary Attention (Liu+ NeurIPS 2022)
but implemented more simply as auxiliary tokens.

RevIN already returns `(x_norm, (mean, std))`. Feed them to the encoder:

**Stat token (preferred, cleanest):**
```python
x_norm, (mu, sigma) = self.revin(x)        # mu: (B,1,C), sigma: (B,1,C)
tokens = self.patch_embed(x_norm)            # (B, N, d)
# Project [mu, sigma] → d-dimensional "stat token", prepend to sequence
stat_feat = torch.cat([mu.squeeze(1), sigma.squeeze(1)], dim=-1)  # (B, 2C)
stat_token = self.stat_proj(stat_feat)       # (B, d)
tokens = torch.cat([stat_token.unsqueeze(1), tokens], dim=1)
# Causal attention: stat token is at position 0, visible to all later tokens
```
Add `self.stat_proj = nn.Linear(2 * n_channels, d_model)` to FAM.__init__.

**Pretraining implementation detail:** The stat token goes into the
**context encoder path only**. The target encoder processes x(t, t+Δt]
with its own RevIN — it does NOT get a stat token. The predictor learns
to map from h_t (which now encodes distributional state via the stat
token's influence on attention) to the target representation. This
asymmetry is intentional: the context encoder needs lifecycle awareness,
the target encoder just needs to represent the target interval.

During pretraining, h_t is the **last non-stat token's** output (not
the stat token itself). The stat token influences h_t through causal
attention but is not the output.

During finetuning, stat_proj is frozen (part of the encoder pipeline).
Only the predictor + event head are trainable, same as before.

Run FD001, 3 seeds. Compare to all previous phases.

**This is the key experiment.** If it works (AUROC >0.7 on FD001 AND
no regression on MBA), this is the permanent fix: RevIN for training
stability + statistics for degradation awareness.

### Phase 5: Regression check on MBA + SMAP (1.5 h)

Run the best variant from Phase 2-4 on MBA and SMAP, 3 seeds each.
Verify per-horizon AUROC at Δt=1 stays near:
- MBA: 0.78 (current v26)
- SMAP: 0.59 (current v26)

If the fix hurts these, try a second variant. If no single strategy
works for all, document the tradeoff honestly.

### Phase 6: Full benchmark with winning variant (3 h)

If Phase 5 passes: run ALL datasets with the winning normalization,
3 seeds each. Store all surfaces as .npz. This produces the final
numbers for the paper.

| Dataset | Est. | Notes |
|---------|------|-------|
| FD001 | 15 min | Should show recovered per-horizon AUROC |
| FD002 | 15 min | |
| FD003 | 15 min | Worst regression in v24, check carefully |
| SMAP | 20 min | Must not regress |
| MSL | 20 min | |
| PSM | 15 min | |
| SMD | 30 min | Largest test set |
| MBA | 15 min | Must not regress |
| GECCO | 15 min | |
| BATADAL | 15 min | |
| PhysioNet | 15 min | P=1, separate run |

For every dataset, report the full per-horizon AUROC + AUPRC table.
Also report the prediction gap `p_mean(y=1) - p_mean(y=0)` at the
shortest horizon — this is the clearest diagnostic of whether the
model conditions on the input or just outputs the base rate.

**Commit and push surfaces after this phase.** These surfaces will be
used for all paper figures.

### Phase 7: Paper-quality probability surfaces (30 min)

Generate 3 publication-quality surface plots (predicted + ground truth)
for Figure 3 of the paper, showing three distinct event types:

1. **C-MAPSS FD001** — single failure at end of life (degradation).
   Pick one test engine with ~150+ cycles. The fixed model should show
   the predicted surface shifting left (shorter horizon) as t increases.

2. **MBA** — frequent but brief arrhythmia episodes within a recording.
   Pick a recording with 2-3 arrhythmia segments. Surface should light
   up during episodes.

3. **SMAP** — spacecraft anomaly, intermediate case.
   Pick an entity with one clear anomaly segment.

Save as .npz with engine/entity IDs for reproducibility. The actual
figure rendering will happen locally.

### Phase 8: Update RESULTS.md + Quarto notebook (30 min)

- v27 section in RESULTS.md with **per-horizon AUROC table**
- `notebooks/27_v27_analysis.qmd` with:
  - Per-horizon AUROC: RevIN vs no-RevIN vs NLinear vs RevIN+stats
  - Surface heatmaps for FD001 (one engine): v26 baseline vs fix
  - Prediction gap analysis: p(y=0) vs p(y=1) at Δt=1 per variant
  - MBA/SMAP regression check
- Render: `quarto render notebooks/27_v27_analysis.qmd`

---

## Phase priorities

| Phase | What | Est. | Priority |
|-------|------|------|----------|
| 1 | Baseline reproduction | 30 min | BLOCKING |
| 2 | No RevIN (global z-score) | 1 h | Critical |
| 3 | NLinear-style (last-value subtraction) | 1 h | Critical |
| 4 | RevIN + stat token | 1.5 h | Critical |
| 5 | MBA + SMAP regression check | 1.5 h | Critical |
| 6 | Full benchmark (all datasets) | 3 h | Important |
| 7 | Paper-quality surfaces (3 cases) | 30 min | Important |
| 8 | RESULTS.md + notebook | 30 min | Always |

**Total**: ~9.5h.

---

## Ground rules

1. **Import from model.py and train.py.** Do NOT copy model code.
2. **finetune_forward returns CDF probabilities.** Do NOT apply sigmoid.
3. **P=16 everywhere.** No exceptions.
4. **Minimum context = 128 timesteps** (8 tokens). Enforced by EventDataset.
5. Stride=1 at evaluation, stride=4 at training.
6. Store surfaces as .npz. Compute AUPRC + per-horizon AUROC from surfaces.
7. **PRIMARY DIAGNOSTIC**: per-horizon AUROC at every horizon. Report for every run.
8. Reporting: `mean +/- std (Ns)`.
9. Commit + push after each phase. Update RESULTS.md after every phase.

---

## Success criteria

1. **Identify cause**: Does removing RevIN recover FD001 AUROC? (Phase 2)
2. **Find universal fix**: Does RevIN+stats work for BOTH drift AND shape
   events? (Phase 4 + 5)
3. **Quantitative bar**: FD001 Δt=10 AUROC from 0.52 → >0.80.
4. **No regression**: MBA Δt=1 AUROC stays above 0.75.

If no single strategy works for all domains, document the tradeoff
honestly. This is itself a finding: RevIN helps shape-based events,
hurts drift-based ones. The over-stationarization problem (Liu+ 2022)
is unresolved for multi-domain event prediction.

---

## Literature references for this session

- **Non-stationary Transformers** (Liu+ NeurIPS 2022): "over-stationarization"
  concept, De-stationary Attention. Our stat-token approach is a simpler
  variant of their τ, Δ injection.
- **RevIN** (Kim+ ICLR 2022): per-instance norm. No mechanism to feed
  statistics back. Our problem is a direct consequence.
- **NLinear** (Zeng+ AAAI 2023): subtract last value only (not mean).
  Preserves drift signal. Outperformed Transformers on most benchmarks.
- **Dish-TS** (Fan+ AAAI 2023): learnable distribution shift coefficients.
  More complex than our stat-token but same motivation.
- **"On the Role of RevIN"** (2025): explicitly warns RevIN "discards
  potentially predictive context" when the mean carries information.

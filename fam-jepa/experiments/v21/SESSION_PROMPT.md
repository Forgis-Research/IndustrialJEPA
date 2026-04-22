# V21 Overnight Session — Breakthrough Results: Probability Surface + AUPRC

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: overnight (~10-12 hours on A10G)

---

## Goal

Fill every red placeholder in the paper's results table with real numbers. The paper now has a clean story: one architecture, per-dataset pretrain, predict events via a probability surface p(t, Δt), evaluate with AUPRC (pooled over all cells). Every dataset gets TWO metric columns: our AUPRC (head-to-head comparison) and the literature's legacy metric (for direct comparison with SOTA papers).

The probability surface must be STORED (as .npy) so we can recompute any legacy metric from it without rerunning inference.

**PRIORITY ORDER**: Start with anomaly datasets (SMAP/MSL/PSM/SMD/MBA) FIRST. C-MAPSS is proven and can run later. The anomaly datasets use the MTS-JEPA benchmark data and have historically been flakier — getting these right is the highest-value use of time.

---

## Paper Story (read this first)

**1. Predictor finetuning.** Freeze encoder, finetune predictor (790K params) + per-horizon sigmoid head. Trained with positive-weighted BCE on the probability surface y(t, Δt). More label-efficient than E2E at ≤10% labels.

**2. One architecture, N datasets, unified metric.** Same 2.37M-param causal JEPA, pretrained per-dataset with no labels. Primary metric: AUPRC pooled over all (t, Δt) cells. Secondary: AUROC. Legacy metrics derived from the stored surface for literature comparability.

---

## Prior results on these datasets (READ THIS — they DID work before)

All 5 anomaly datasets were successfully pretrained and evaluated in v18/v19:
- **SMAP**: v17 pretrained 150ep → v18 Mahalanobis → PA-F1 0.793±0.014 (3 seeds) ✓
- **MSL**: v17 pretrained 150ep → v18 Mahalanobis → PA-F1 0.707±0.050 (3 seeds) ✓
- **PSM**: v19 pretrained 50ep → Mahalanobis → PA-F1 0.813±0.048 (3 seeds) ✓
- **SMD**: v19 pretrained 50ep → Mahalanobis → PA-F1 0.252±0.017 (3 seeds) ✓
- **MBA**: v19 pretrained 50ep → Mahalanobis → PA-F1 0.551±0.054 (3 seeds) ✓

Earlier failures (v15-v16) were due to insufficient pretraining (20 epochs). With 50-150 epochs, all succeed. Data is already downloaded on the VM at paths in `fam-jepa/data/config.py`.

**What's NEW in v21**: per-horizon sigmoid + BCE (replacing Mahalanobis-only), probability surface storage, AUPRC as primary metric.

---

## Evaluation Protocol (CRITICAL — read before any experiment)

### The probability surface p(t, Δt)

For each test sample at observation time t, the model outputs:
```
p(t, Δt_k) = σ(w · ĥ_{t+Δt_k} + b)  for k = 1, ..., K
```
where ĥ_{t+Δt_k} = g_φ(h_t, Δt_k) is the predictor's output at horizon Δt_k.

Ground truth: y(t, Δt_k) = 1 if event occurs within Δt_k steps of time t, else 0.

### Horizons per dataset

| Dataset | K | Horizons (Δt values) | Rationale |
|---------|---|---------------------|-----------|
| C-MAPSS FD001 | 16 | 1, 2, ..., 16 patches (= 1-16 cycles) | Matches v20 W=16 |
| C-MAPSS FD002 | 16 | 1, 2, ..., 16 patches | Same |
| C-MAPSS FD003 | 16 | 1, 2, ..., 16 patches | Same |
| SMAP | 10 | 1, 2, 3, 5, 10, 15, 20, 30, 50, 100 steps | Short + long range |
| MSL | 10 | 1, 2, 3, 5, 10, 15, 20, 30, 50, 100 steps | Same |
| PSM | 10 | 1, 2, 3, 5, 10, 15, 20, 30, 50, 100 steps | Same |
| SMD | 10 | 1, 2, 3, 5, 10, 15, 20, 30, 50, 100 steps | Same |
| MBA | 10 | 1, 2, 3, 5, 10, 15, 20, 30, 50, 100 steps | Same |

### Metrics (computed for EVERY experiment)

**Primary (our metric):**
- **AUPRC** — pooled over ALL (t, Δt) cells. One number per dataset. Use `evaluation.surface_metrics.evaluate_probability_surface()`.
- **AUROC** — pooled over ALL (t, Δt) cells. Secondary.

**Per-horizon breakdown (appendix):**
- **AUPRC(Δt)** per horizon. Use `evaluation.surface_metrics.auprc_per_horizon()`.

**Legacy (for SOTA comparison):**
- C-MAPSS: RMSE (from argmax of p surface → predicted RUL)
- SMAP/MSL/PSM/SMD: PA-F1 (from p(t, Δt=0) thresholded, then PA protocol)
- MBA: PA-F1 (same)
- ALL: non-PA F1, Precision, Recall at best threshold

**Calibration:**
- Reliability diagram data. Use `evaluation.surface_metrics.reliability_diagram()`.

**Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Always decompose into P + R.

### Storage (NON-NEGOTIABLE)

For EVERY experiment run, save:
```python
np.savez(f'v21/surfaces/{dataset}_seed{seed}_{mode}.npz',
         p_surface=p_surface,    # (N, K) float32
         y_surface=y_surface,    # (N, K) int32
         horizons=horizons,      # (K,) int32
         metadata={'dataset': ..., 'seed': ..., 'mode': ...})
```
This allows recomputing ANY metric from the surface without rerunning inference.

---

## Training: Per-Horizon Sigmoid + Positive-Weighted BCE

### Architecture change from v20

v20 used: `Linear(16*d, 1)` → scalar RUL → MSE loss.
v21 uses: `Linear(d, 1)` applied per-horizon → sigmoid → BCE loss.

```python
# In pred_ft_utils.py, replace the downstream head:
class EventHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)  # shared across horizons
    
    def forward(self, h_hat_k):
        # h_hat_k: (B, d_model) — predictor output at one horizon
        return self.linear(h_hat_k).squeeze(-1)  # (B,) logits

# Training loop per batch:
# 1. h_t = encoder(x_past, mask)           # (B, d)
# 2. For each horizon k:
#      h_hat_k = predictor(h_t, k)          # (B, d)
#      logit_k = head(h_hat_k)              # (B,)
# 3. Stack logits → (B, K)
# 4. y_surface_batch → (B, K) from build_label_surface()
# 5. loss = weighted_bce_loss(logits, y_surface_batch)
```

Use `evaluation.losses.weighted_bce_loss()` and `evaluation.losses.build_label_surface()`.
Use `evaluation.losses.compute_pos_weight()` once on the full training set, then pass as constant.

---

## Setup (~15 min)

### A. Sync repo
```bash
cd /home/sagemaker-user/IndustrialJEPA
git pull
```

### B. Read context
1. `CLAUDE.md` — repo structure
2. `fam-jepa/experiments/RESULTS.md` — master results table
3. `fam-jepa/evaluation/surface_metrics.py` — NEW: evaluate_probability_surface()
4. `fam-jepa/evaluation/losses.py` — NEW: weighted_bce_loss(), build_label_surface()
5. `fam-jepa/data/` — all dataset loaders (config.py for paths)
6. `fam-jepa/experiments/v20/` — v20 results (pred_ft_utils.py has the base infrastructure)
7. `paper-neurips/paper.tex` — the paper with red placeholders to fill

### C. Verify data availability
```python
from fam_jepa.data import load_smap, load_msl, load_psm, load_smd, load_mba
for loader in [load_smap, load_msl, load_psm, load_smd, load_mba]:
    d = loader()
    if d is not None:
        print(f"{d['name']}: {d['train'].shape} / {d['test'].shape}, anom={d['anomaly_rate']:.3f}")
```

---

## Phase 0: Infrastructure — BCE Head + Surface Eval (~1.5 hours, est. 90 min)

### 0a. Implement EventHead + BCE training loop

Modify `v20/pred_ft_utils.py` → `v21/pred_ft_utils.py`:
- Replace `Linear(16*d, 1) + MSE` with `EventHead(d) + weighted BCE`
- Build y_surface from time-to-event labels using `build_label_surface()`
- Output p_surface at eval time (sigmoid of logits)
- Store surface as .npz

### 0b. Implement surface → legacy metric conversion

Write `v21/surface_to_legacy.py`:
```python
def surface_to_rul(p_surface, horizons):
    """Convert p(t, Δt) surface to scalar predicted RUL.
    RUL_hat(t) = sum_k (Δt_k * p(t, Δt_k)) / sum_k p(t, Δt_k)
    i.e., expected time-to-event from the surface.
    """

def surface_to_anomaly_scores(p_surface, horizons):
    """Convert p(t, Δt) surface to per-timestep anomaly score.
    score(t) = max_k p(t, Δt_k)  (most alarming horizon)
    Then apply PA protocol for PA-F1.
    """
```

### 0c. Validate on SMAP (1 seed) — NOT C-MAPSS

Quick sanity on SMAP first: load existing v17 SMAP checkpoint → pred-FT with BCE → p_surface → AUPRC + PA-F1.
Compare PA-F1 to v18's 0.793 — should be in the same ballpark (different downstream head).

**Save**: `v21/phase0_infrastructure.json`

---

## Phase 1: Anomaly Datasets — SMAP, MSL, PSM, SMD, MBA (~3 hours, est. 180 min)

**START HERE.** These datasets have existing pretrained checkpoints from v17-v19. The new work is: (a) add per-horizon sigmoid + BCE finetuning on labeled data, (b) compute AUPRC from stored surfaces.

### Time estimate per dataset

| Dataset | Pretrain | Checkpoint exists? | Pred-FT (3 seeds) | Mahalanobis (3 seeds) | Total est. |
|---------|----------|--------------------|--------------------|-----------------------|------------|
| SMAP | — | ✓ v17 (150ep) | ~20 min | ~10 min | ~30 min |
| MSL | — | ✓ v17 (150ep) | ~20 min | ~10 min | ~30 min |
| PSM | — | ✓ v19 (50ep) | SKIP (dist mismatch) | ~10 min | ~15 min |
| SMD | — | ✓ v19 (50ep) | ~20 min | ~10 min | ~30 min |
| MBA | — | ✓ v19 (50ep) | ~15 min | ~10 min | ~25 min |
| **Total** | | | | | **~130 min** |

If checkpoints are NOT on the VM (were they pushed?), re-pretrain:
- SMAP/MSL: 150 epochs ~45 min each
- PSM/SMD/MBA: 50 epochs ~20 min each

### 1a. For each dataset: Mahalanobis baseline (unsupervised)

Same as v18/v19: encode windows → PCA-Mahalanobis → anomaly scores.
Convert scores to p_surface: calibrate via sigmoid on validation scores.
Store surface. Compute AUPRC + PA-F1 (legacy).

3 seeds each.

### 1b. For each dataset: Pred-FT (supervised, where feasible)

For datasets where we can create a clean train/val/test split of labels:
- SMAP: chronological split of labeled test (60/10/30)
- MSL: same
- PSM: SKIP pred-FT (known distribution mismatch from v20)
- SMD: chronological split
- MBA: chronological split

3 seeds each. Store surfaces.

### 1c. Aggregate

For each dataset: AUPRC (primary), AUROC, PA-F1 (legacy), non-PA F1, P, R.

**Save**: `v21/phase1_anomaly.json`, `v21/surfaces/anomaly_*.npz`

---

## Phase 2: C-MAPSS Breakthrough Table (~2.5 hours, est. 150 min)

C-MAPSS is proven. The new work: replace MSE→BCE, compute AUPRC.

### Time estimate

| Task | Seeds | Est. time |
|------|-------|-----------|
| FD001 pred-FT 100% | 5 | ~25 min |
| FD001 pred-FT 5% | 5 | ~25 min |
| FD001 e2e 100% | 5 | ~25 min |
| FD001 e2e 5% | 5 | ~25 min |
| FD002 pred-FT (100%+5%) | 3 | ~30 min |
| FD003 pred-FT (100%+5%) | 3 | ~30 min |
| **Total** | | **~160 min** |

### 2a. FD001 full sweep (5 seeds)

Using V17 checkpoint (or SIGReg-pred if available):

| Mode | What | Seeds |
|------|------|-------|
| probe_h | Linear on h_past, BCE | 5 |
| pred_ft | Freeze enc, finetune pred + head, BCE | 5 |
| e2e | Full finetune, BCE | 5 |
| scratch | Random init, BCE | 5 |

At 100% and 5% labels. Store ALL surfaces. Compute: AUPRC (primary), AUROC, RMSE (legacy).

### 2b. FD002 + FD003 (3 seeds each)

Same modes but 3 seeds for time efficiency. Only 100% and 5% labels.

**Save**: `v21/phase2_cmapss.json`, `v21/surfaces/cmapss_*.npz`

---

## Phase 3: Fill Paper Table (~1 hour, est. 60 min)

### The target table (paper.tex Tab 1):

```
Dataset        | Domain      | Event     | AUPRC↑      | AUROC↑     | Legacy metric    | SOTA legacy    | SOTA ref
---------------|-------------|-----------|-------------|------------|------------------|----------------|----------
C-MAPSS FD001  | Turbofan    | Failure   | [FILL]      | [FILL]     | RMSE [FILL]      | RMSE 10.61     | STAR
C-MAPSS FD002  | Turbofan    | Failure   | [FILL]      | [FILL]     | RMSE [FILL]      | RMSE 13.47     | STAR
C-MAPSS FD003  | Turbofan    | Failure   | [FILL]      | [FILL]     | RMSE [FILL]      | RMSE 10.71     | STAR
SMAP           | Spacecraft  | Anomaly   | [FILL]      | [FILL]     | PA-F1 [FILL]     | PA-F1 0.336    | MTS-JEPA
MSL            | Spacecraft  | Anomaly   | [FILL]      | [FILL]     | PA-F1 [FILL]     | PA-F1 0.336    | MTS-JEPA
PSM            | Server      | Anomaly   | [FILL]      | [FILL]     | PA-F1 [FILL]     | PA-F1 0.616    | MTS-JEPA
SMD            | Server      | Anomaly   | [FILL]      | [FILL]     | PA-F1 [FILL]     | PA-F1 0.925    | AT
MBA            | Cardiac     | Arrhythmia| [FILL]      | [FILL]     | PA-F1 [FILL]     | —              | —
```

### 3a. Update paper.tex

Replace Tab 1 placeholders with real numbers. Update abstract. Write summary paragraph (currently a placeholder at §5.1).

### 3b. Update RESULTS.md

Add all v21 numbers to the master table.

**Save**: commit to git

---

## Phase 4: Label Efficiency with AUPRC (~1.5 hours, est. 90 min)

Extend the v20 label efficiency study with AUPRC:

| Budget | pred-FT AUPRC | E2E AUPRC | Δ | paired p |
|--------|---------------|-----------|---|----------|
| 100%   | | | | |
| 50%    | | | | |
| 20%    | | | | |
| 10%    | | | | |
| 5%     | | | | |

5 seeds on FD001. Key question: does the pred-FT crossover at ≤10% hold under AUPRC?

**Save**: `v21/phase4_label_efficiency.json`

---

## Phase 5: Chronos + Foundation Baselines (~1 hour, est. 45 min)

Re-evaluate Chronos-T5-tiny on FD001 with the p_surface framework.
Store surface. Compute AUPRC + RMSE.

Add to paper table for head-to-head: FAM AUPRC vs Chronos AUPRC.

**Save**: `v21/phase5_chronos.json`

---

## Phase 6: Appendix + Quality (~1 hour, est. 60 min)

### 6a. Per-horizon AUPRC curves

For C-MAPSS FD001 and SMAP: plot AUPRC(Δt) from stored surfaces.
Save as `v21/figures/auprc_per_horizon_{dataset}.pdf`.

### 6b. Reliability diagrams

For C-MAPSS FD001 and SMAP: compute reliability diagram from stored surfaces.
Save as `v21/figures/reliability_{dataset}.pdf`.

### 6c. Render Quarto summary

Create `notebooks/21_v21_analysis.qmd` with:
- All tables with formatting
- AUPRC per-horizon curves
- Reliability diagrams
- Surface heatmap visualizations
- Per-seed breakdowns

Render to HTML: `quarto render notebooks/21_v21_analysis.qmd`

---

## Total Time Budget

| Phase | Est. time | Cumulative | Priority |
|-------|-----------|------------|----------|
| Setup | 15 min | 0:15 | Required |
| Phase 0: Infrastructure | 90 min | 1:45 | Required |
| Phase 1: Anomaly datasets | 130-180 min | 4:45 | **CRITICAL** |
| Phase 2: C-MAPSS | 150 min | 7:15 | **CRITICAL** |
| Phase 3: Fill paper | 60 min | 8:15 | **CRITICAL** |
| Phase 4: Label efficiency | 90 min | 9:45 | Important |
| Phase 5: Chronos | 45 min | 10:30 | Important |
| Phase 6: Appendix + Quarto | 60 min | 11:30 | Nice-to-have |

**If time is short**: Phase 0 → Phase 1 → Phase 2 → Phase 3 delivers the full paper table.

---

## Ground Rules

1. **AUPRC for every experiment.** Pooled over all (t, Δt) cells. One number per dataset per config.
2. **Store surfaces.** Every run saves p_surface + y_surface as .npz. No exceptions.
3. **Legacy metrics from surfaces.** Never compute legacy metrics directly — always derive from stored p_surface.
4. **Reporting**: `mean ± std (Ns, 95% CI [lo, hi])`. Decompose F1 → P + R. Include AUROC.
5. **No ad-hoc metrics.** Use `evaluation.surface_metrics.evaluate_probability_surface()`.
6. **Commit after every phase.** Push results so they survive crashes.
7. **Update RESULTS.md** after every phase with new numbers.
8. **Budget**: ~10-12 hours. Phase 0-3 are critical (8h). Phase 4-6 are stretch (3h).
9. **Anomaly datasets FIRST.** They are the highest-risk, highest-value targets.

---

## Expected Output

By morning, the paper should have:
- A clean results table with AUPRC + legacy metric for 8 datasets
- Real numbers replacing all red placeholders
- Updated abstract with concrete claims
- Stored probability surfaces enabling any future metric recomputation
- Quarto notebook with full reproducible analysis

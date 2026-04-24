# V28 Session — Honest Metrics, Chronos-2 Surfaces, Prediction Analysis

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: 8-10 hours on A10G. Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v27 SESSION_SUMMARY.md

---

## Why this session exists

### 1. Pooled AUPRC is a vanity metric

We discovered that pooled AUPRC over the probability surface is dominated
by trivial long-horizon cells where the base rate is 76-99%.

A model that **ignores the input entirely** and outputs `p = base_rate(Δt)`
for every cell scores:

```
FD001:  base-rate AUPRC = 0.924    our model = 0.927    delta = +0.003
MBA:    base-rate AUPRC = 0.907    our model = 0.947    delta = +0.040
SMAP:   base-rate AUPRC = 0.284    our model = 0.390    delta = +0.106
```

FD001's AUPRC of 0.926 is 99.7% attributable to the base rate. We need
honest metrics.

### 2. Detection vs prediction

On MBA, the model detects ongoing arrhythmia but does not predict onset
with lead time (p is flat around event boundaries). On FD001 (v27 fix),
the model tracks degradation (genuine prediction). We need to characterize
this honestly for each dataset.

### 3. No Chronos-2 surfaces exist

We compared FAM vs Chronos-2 on scalar AUPRC but never generated
Chronos-2 probability surfaces. Chronos-2's Δt=1 AUPRC of 0.41 on FD001
vs our 0.03 suggests it may actually predict — we need to see its surface
to understand why.

---

## Phase plan

### Phase 1: Chronos-2 probability surfaces (2 h)

Generate full probability surfaces from Chronos-2 for all datasets.
Reuse v24 `baseline_chronos2.py` with a patch to store surfaces:

After `p_te = torch.sigmoid(logits).cpu().numpy()`, add:
```python
surf_dir = Path('experiments/v28/surfaces')
surf_dir.mkdir(parents=True, exist_ok=True)
np.savez(surf_dir / f'chronos2_{args.dataset}_s{args.seed}.npz',
         p_surface=p_te, y_surface=yte_np,
         horizons=np.array(horizons), t_index=tte_idx.numpy())
```

Run for ALL datasets, seed=42:
```bash
for ds in FD001 FD002 FD003 SMAP MSL PSM SMD MBA; do
    python experiments/v24/baseline_chronos2.py \
        --dataset $ds --seed 42 --cache-features
done
```

For each dataset, also print the per-horizon diagnostic:
```python
from sklearn.metrics import roc_auc_score, average_precision_score
for i, hv in enumerate(horizons):
    auroc = roc_auc_score(yte_np[:, i], p_te[:, i])
    auprc = average_precision_score(yte_np[:, i], p_te[:, i])
    print(f"  dt={hv:>3}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
```

### Phase 2: FAM v27 surfaces for ALL datasets (1.5 h)

Run FAM with v27 architecture (`norm_mode` per dataset family):
- C-MAPSS (FD001/FD002/FD003): `norm_mode='none'`, global z-score
- Anomaly (SMAP/MSL/PSM/SMD): `norm_mode='revin'`
- Cardiac (MBA): `norm_mode='revin'`

Seed=42 for all. Store surfaces as .npz.

### Phase 3: Metric analysis notebook (2 h)

Create `notebooks/28_metric_analysis.qmd` — a thorough walkthrough of
how metrics relate to the probability surface.

**Use the data-curator agent for this notebook.**

#### Section 1: The probability surface explained

Pick FD001 engine 49 (v27 fix, longest engine). Show:
- The ground truth surface y(t, Δt) with annotation explaining what each
  cell means: "y(t=200, Δt=50) = 1 means the engine fails within 50
  cycles of observation time 200"
- The predicted surface p(t, Δt)
- How to read it: rows = horizons, columns = observation times

#### Section 2: How pooled AUPRC is computed

Step-by-step:
1. Flatten (N, K) surface to N×K cells
2. Sort by predicted p descending
3. Walk down the ranking: at each threshold, compute precision and recall
4. Area under the precision-recall curve

Show with code: compute pooled AUPRC from scratch (no sklearn),
visualizing the precision-recall curve.

#### Section 3: Why pooled AUPRC is inflated

Compute the base-rate classifier's AUPRC for each dataset.
Show the table:

```
Dataset | base-rate AUPRC | model AUPRC | delta above base
```

Explain: long-horizon cells (Δt=100-150) have 96-99% positive rate.
A model that outputs p=0.97 for these cells and p=0.03 for short
horizons gets AUPRC=0.924 without looking at the data. Our model
adds +0.003 on FD001 (v24 RevIN) or +0.02 on FD001 (v27 fix).

**Key visual**: Show the precision-recall curve with the base-rate
classifier overlaid. They should nearly overlap for FD001 v24.

#### Section 4: Per-horizon AUPRC and AUROC

Show per-horizon metrics for each dataset. Explain:
- AUROC is prevalence-invariant (measures ranking quality)
- AUPRC is prevalence-sensitive (measures how many TP you get before FP)
- At Δt=1 with 2.5% prevalence, random AUPRC = 0.025
- At Δt=150 with 99% prevalence, random AUPRC ≈ 0.99

Propose: **mean per-horizon AUROC** as the primary honest metric.
Each horizon contributes equally. Prevalence doesn't inflate it.

#### Section 5: Are all cells equally valuable?

No. Three regimes:
1. **Trivial cells** (Δt=100-150, pos_rate >95%): nearly all positive,
   any model gets these right. Zero diagnostic value.
2. **Hard cells** (Δt=1-10, pos_rate <25%): the real test. Can the
   model distinguish "event imminent" from "healthy"?
3. **Informative cells** (Δt=20-50, pos_rate 25-75%): balanced,
   most discriminative.

Propose: **weighted AUPRC** that upweights hard cells. Or simply
report the per-horizon breakdown.

#### Section 6: Detection vs prediction

For MBA: show the onset analysis (p around event boundaries).
Demonstrate that p does NOT rise before onset — model detects,
not predicts. Discuss implications: is lead-time prediction even
possible from the data? (For MBA at 275 Hz, Δt=10 ≈ 36ms — barely
one heartbeat of lead time.)

For FD001 (v27): show that p DOES rise gradually as the engine
degrades — genuine prediction. The diagonal surface structure IS
prediction: "this engine will fail within 50 cycles" when RUL > 50.

#### Section 7: Proposed metric revision

Recommend for the paper:
1. **Primary**: mean per-horizon AUROC (honest, prevalence-invariant)
2. **Secondary**: pooled AUPRC (for continuity, but contextualized
   against base-rate baseline)
3. **Legacy**: RMSE, PA-F1 etc (unchanged, for literature comparison)
4. Always report per-horizon breakdown

#### Section 8: Side-by-side surfaces

For each dataset, 3-panel figure: (a) FAM predicted, (b) Chronos-2
predicted, (c) ground truth. Same color scale, same axes.

Render all to PNG. Push PNGs only (not .npz).

### Phase 4: Render all surface comparison PNGs (1 h)

For each dataset, render:
```
[FAM p(t,Δt)] | [Chronos-2 p(t,Δt)] | [ground truth y(t,Δt)]
```

For C-MAPSS: pick longest engine per subset.
For anomaly: pick entity with clearest anomaly segment.
For MBA: use the same recording tail as v27.

Save to `experiments/v28/results/surface_pngs/`.

Use the **figure-creator agent** for publication-quality versions of
the 3 best examples (FD001, MBA, one anomaly dataset).

### Phase 5: What would make the predictor genuinely predict? (1 h)

Brainstorm and test one idea. The current model detects current state
but doesn't predict future onset. Potential fixes:

**A. Lead-time gap in labels.** Replace:
  `y(t, Δt) = 1[event in (t, t+Δt]]`
with:
  `y(t, Δt) = 1[event in (t+gap, t+gap+Δt]]`
where gap = mandatory lead time (e.g., 10 steps). This forces the
model to predict AHEAD rather than detect NOW. Easy to implement:
shift labels by `gap` steps.

**B. Separate detection from prediction.** Two heads:
  - Detection head: p(event at t) — binary, no horizon
  - Prediction head: p(event in (t+gap, t+gap+Δt]) — with lead time
Train both jointly. Detection is easy; prediction is the contribution.

**C. Evaluate with lead-time-aware metrics.** Don't change the model,
change the metric: only count true positives where p > threshold
at least `gap` steps BEFORE the event onset. This is standard in
early warning literature.

Test option A on FD001 (v27 fix, gap=10) and MBA (gap=5).
Compare per-horizon AUROC to the gap=0 baseline.

### Phase 6: Update RESULTS.md + commit (30 min)

- v28 section in RESULTS.md
- Commit surfaces PNGs, notebook, metric analysis
- Push

---

## Phase priorities

| Phase | What | Est. | Priority |
|-------|------|------|----------|
| 1 | Chronos-2 surfaces (all datasets) | 2 h | Critical |
| 2 | FAM v27 surfaces (all datasets) | 1.5 h | Critical |
| 3 | Metric analysis notebook | 2 h | Critical |
| 4 | Surface comparison PNGs | 1 h | Important |
| 5 | Lead-time prediction experiment | 1 h | Important |
| 6 | RESULTS.md + commit | 30 min | Always |

**Total**: ~8h.

---

## Ground rules

1. **Import from model.py and train.py.** Do NOT copy model code.
2. **finetune_forward returns CDF probabilities.** Do NOT apply sigmoid.
3. **norm_mode='none' for C-MAPSS, 'revin' for everything else.**
4. **P=16 everywhere.**
5. Store surfaces as .npz ON THE VM. Push only PNGs to git.
6. **PRIMARY DIAGNOSTIC**: per-horizon AUROC + base-rate comparison.
7. Use **data-curator agent** for the metric notebook.
8. Use **figure-creator agent** for publication-quality surface figures.
9. Commit + push after each phase. Update RESULTS.md.

---

## Key question this session answers

**Is FAM actually predicting events, or just detecting current state
with a multi-horizon wrapper that inflates AUPRC?**

The answer will differ by dataset:
- FD001 (v27 fix): genuine prediction (degradation tracking)
- MBA: detection (no lead time) — but detection IS useful
- SMAP: partial (AUROC 0.59, modest above chance)

The metric notebook makes this transparent. The Chronos-2 surfaces
show whether a foundation model does better. The lead-time experiment
tests whether prediction is achievable with a simple label shift.

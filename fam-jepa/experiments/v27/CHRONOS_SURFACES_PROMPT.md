# Chronos-2 Probability Surfaces — add to v27 notebook

**Run on VM after main v27 session completes.**
**Est. time**: 1-2 hours (feature extraction is the bottleneck: ~40 min/dataset on A10G).
**Working directory**: `IndustrialJEPA/fam-jepa/`

---

## Goal

Generate Chronos-2 probability surfaces (p_surface, y_surface) for
direct visual comparison with FAM surfaces. The v24 Chronos baseline
computed per-horizon AUPRC but did NOT store full surfaces as .npz.

We need surfaces for: FD001, MBA, SMAP (the three paper figure cases).

---

## Step 1: Patch baseline_chronos2.py to store surfaces

At the end of `main()` in `experiments/v24/baseline_chronos2.py`,
after `p_te = torch.sigmoid(logits).cpu().numpy()`, add:

```python
# Store probability surface for figure generation
surf_dir = Path('experiments/v27/surfaces')
surf_dir.mkdir(parents=True, exist_ok=True)
np.savez(
    surf_dir / f'chronos2_{args.dataset}_s{args.seed}.npz',
    p_surface=p_te,
    y_surface=yte_np,
    horizons=np.array(horizons),
    t_index=tte_idx.numpy(),
)
print(f"Saved surface to {surf_dir / f'chronos2_{args.dataset}_s{args.seed}.npz'}")
```

## Step 2: Run for 3 datasets

```bash
cd IndustrialJEPA/fam-jepa

# FD001 (~15 min with cached features, ~55 min without)
python experiments/v24/baseline_chronos2.py \
    --dataset FD001 --seed 42 --cache-features

# MBA (~20 min)
python experiments/v24/baseline_chronos2.py \
    --dataset MBA --seed 42 --cache-features

# SMAP (~40 min)
python experiments/v24/baseline_chronos2.py \
    --dataset SMAP --seed 42 --cache-features
```

## Step 3: Render surface PNGs (same style as FAM surfaces)

```python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

surf_dir = Path('experiments/v27/surfaces')
png_dir = Path('experiments/v27/results/surface_pngs')
png_dir.mkdir(parents=True, exist_ok=True)

for ds in ['FD001', 'MBA', 'SMAP']:
    f = surf_dir / f'chronos2_{ds}_s42.npz'
    if not f.exists():
        print(f'SKIP {f}')
        continue
    d = np.load(f, allow_pickle=True)
    p = d['p_surface']; y = d['y_surface']
    t = d['t_index']; h = d['horizons']

    # For C-MAPSS: pick longest engine
    if ds.startswith('FD'):
        breaks = np.where(np.diff(t) < 0)[0] + 1
        breaks = np.concatenate([[0], breaks, [len(t)]])
        lengths = [breaks[i+1] - breaks[i] for i in range(len(breaks)-1)]
        # Use engine 49 if possible (same as FAM figure)
        # Engine indices map to test engine IDs
        best = np.argmax(lengths)
        s, e = breaks[best], breaks[best+1]
        t_eng, p_eng, y_eng = t[s:e], p[s:e], y[s:e].astype(float)
    else:
        # Use full test stream (or first 2000 points)
        n = min(2000, len(t))
        t_eng, p_eng, y_eng = t[:n], p[:n], y[:n].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
    im0 = axes[0].pcolormesh(t_eng, h, p_eng.T, cmap='viridis',
                              vmin=0, vmax=1, shading='auto')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('horizon Δt (log)')
    axes[0].set_xlabel('observation time t')
    axes[0].set_title(f'{ds}: Chronos-2 predicted p(t, Δt)')
    fig.colorbar(im0, ax=axes[0], label='prob', shrink=0.8)

    im1 = axes[1].pcolormesh(t_eng, h, y_eng.T, cmap='Reds',
                              vmin=0, vmax=1, shading='auto')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('observation time t')
    axes[1].set_title(f'{ds}: ground truth y(t, Δt)')
    fig.colorbar(im1, ax=axes[1], label='label', shrink=0.8)

    plt.tight_layout()
    out = png_dir / f'chronos2_{ds}_s42_surface.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')
```

## Step 4: Push PNGs (not .npz)

```bash
git add experiments/v27/results/surface_pngs/chronos2_*.png
git commit -m "v27: Chronos-2 probability surfaces for comparison"
git push
```

## Step 5: Per-horizon diagnostic for Chronos-2

```python
from sklearn.metrics import roc_auc_score, average_precision_score

for ds in ['FD001', 'MBA', 'SMAP']:
    f = surf_dir / f'chronos2_{ds}_s42.npz'
    if not f.exists(): continue
    d = np.load(f, allow_pickle=True)
    p, y, h = d['p_surface'], d['y_surface'], d['horizons']
    print(f'\n{ds} Chronos-2:')
    for i, hv in enumerate(h):
        yi, pi = y[:, i], p[:, i]
        if 0 < yi.mean() < 1:
            auroc = roc_auc_score(yi, pi)
            auprc = average_precision_score(yi, pi)
            print(f'  dt={hv:>3}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  pos={yi.mean():.3f}')
```

This tells us: does Chronos-2 predict ahead where FAM doesn't?
Specifically: at Δt=1 on FD001, Chronos-2 had AUPRC 0.41 vs FAM 0.03.
The surface will show whether Chronos-2 actually tracks degradation
or just has a better-calibrated base rate.

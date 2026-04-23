"""Diagnostic: does AUPRC jump if we restrict eval to t >= 128?

ARCHITECTURE.md says: 128 timesteps (8 tokens at P=16) is the minimum for
a functioning causal transformer. Below this it degenerates. EventDataset
currently starts at t=1, so most eval rows have too-short contexts.

Pool AUPRC at various t_min floors to see how much short contexts hurt.
"""

import sys
from pathlib import Path

import numpy as np

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
sys.path.insert(0, str(FAM_DIR))
from evaluation.surface_metrics import (
    evaluate_probability_surface, auprc_per_horizon,
)


SURF_DIR = FAM_DIR / 'experiments/v24/surfaces'

print(f"{'seed':<6} {'t_min':>6} {'n_rows':>7} {'prev':>7} "
      f"{'AUPRC':>7} {'AUROC':>7}", flush=True)

for seed in [42, 123, 456]:
    npz = np.load(SURF_DIR / f'FD001_s{seed}.npz')
    p = npz['p_surface']
    y = npz['y_surface']
    t = npz['t_index']
    horizons = npz['horizons']
    for t_min in [0, 32, 64, 96, 128, 160]:
        mask = t >= t_min
        if mask.sum() < 10:
            continue
        pm = p[mask]
        ym = y[mask]
        out = evaluate_probability_surface(pm, ym)
        print(f"s{seed:<5} {t_min:>6} {mask.sum():>7} "
              f"{out['prevalence']:>7.3f} {out['auprc']:>7.4f} "
              f"{out['auroc']:>7.4f}", flush=True)
    print()

# Per-horizon AUPRC at t_min=128 (seed 42)
print("per-horizon AUPRC @ t_min=128, seed=42:", flush=True)
npz = np.load(SURF_DIR / 'FD001_s42.npz')
mask = npz['t_index'] >= 128
p = npz['p_surface'][mask]
y = npz['y_surface'][mask]
h = npz['horizons']
per = auprc_per_horizon(p, y, horizon_labels=list(h))
for hh, aa in zip(h, per['auprc_per_k']):
    print(f"  dt={hh:4d}: {aa:.4f}", flush=True)

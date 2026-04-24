"""V28 Phase 4 — render FAM | Chronos-2 | GT 3-panel comparison PNGs.

For every dataset:
  - Top: FAM best p(t, Δt) on dense Δt grid
  - Middle: Chronos-2 p(t, Δt)
  - Bottom: ground truth y(t, Δt)

Linear y-axis (per the prompt — easier on the eye for sparse horizons),
viridis colormap, [0, 1] scale. Per-panel title shows pooled AUPRC and
mean per-horizon AUROC.

Output: experiments/v28/results/surface_pngs/triplet_<dataset>.png.
PNGs only — .npz surfaces stay on the VM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V27 = FAM_DIR / 'experiments/v27/surfaces'
V28 = FAM_DIR / 'experiments/v28/surfaces'
OUT = FAM_DIR / 'experiments/v28/results/surface_pngs'
OUT.mkdir(parents=True, exist_ok=True)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {'p': d['p_surface'], 'y': d['y_surface'].astype(np.int8),
            'hz': d['horizons']}


def _downsample(arr: np.ndarray, max_cols: int = 3000) -> tuple:
    N = arr.shape[0]
    if N <= max_cols:
        return arr, np.arange(N), 1
    step = int(np.ceil(N / max_cols))
    idx = np.arange(0, N, step)
    return arr[idx], idx, step


def _metrics(p: np.ndarray, y: np.ndarray) -> tuple:
    yi = y.ravel().astype(int)
    if yi.sum() == 0 or yi.sum() == len(yi):
        pooled = float('nan')
    else:
        pooled = float(average_precision_score(yi, p.ravel()))
    valid = [i for i in range(p.shape[1])
             if 0 < y[:, i].mean() < 1]
    if not valid:
        return pooled, float('nan')
    mean_auroc = float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in valid]))
    return pooled, mean_auroc


def render_triplet(dataset: str, fam_path: Path, chronos_path: Path,
                   fam_label: str, out_name: str | None = None) -> None:
    fam = _load(fam_path)
    chr_ = _load(chronos_path)
    if fam is None and chr_ is None:
        print(f"  SKIP {dataset}: no surfaces"); return

    # Use whichever exists for ground truth (they share the same y).
    ref = fam if fam is not None else chr_
    hz = ref['hz']
    y_full = ref['y'].astype(np.float32)
    N = y_full.shape[0]
    y_ds, x_ds, step = _downsample(y_full)

    # 3 rows × 1 column, shared y axis. Plus a colorbar column.
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharey=True, sharex=True,
                             gridspec_kw={'hspace': 0.35})

    panels = [
        (axes[0], fam, fam_label, 'FAM v28'),
        (axes[1], chr_, 'Chronos-2', 'Chronos-2'),
        (axes[2], None, 'Ground truth', 'GT'),
    ]

    last_im = None
    for ax, surf, title, _short in panels:
        if title.startswith('Ground'):
            arr_ds = y_ds
            ax_title = f'Ground truth y(t, Δt)  pos_rate={float(y_full.mean()):.3f}'
        else:
            if surf is None:
                ax.set_title(f'{title}  (surface unavailable)', fontsize=10)
                ax.axis('off'); continue
            p = surf['p']
            arr_ds, _, _ = _downsample(p)
            pooled, mean_auroc = _metrics(p, surf['y'])
            ax_title = (f'{title}  p(t, Δt)  '
                        f'pooled AUPRC={pooled:.3f}  mean h-AUROC={mean_auroc:.3f}')
        last_im = ax.pcolormesh(x_ds, hz, arr_ds.T, cmap='viridis',
                                vmin=0, vmax=1, shading='auto')
        ax.set_title(ax_title, fontsize=10)
        ax.set_ylabel('Δt')
    axes[-1].set_xlabel(f'sample index'
                        + (f' (every {step})' if step > 1 else ''))
    plt.suptitle(f'{dataset}  test-set probability surfaces  (N={N})',
                 fontsize=12, y=0.995)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, pad=0.015, shrink=0.85,
                            label='probability')
    out_path = OUT / (out_name or f'triplet_{dataset}.png')
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(FAM_DIR)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=[
        'FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'MBA',
        'GECCO', 'BATADAL',
    ])
    ap.add_argument('--variant', default='auto',
                    help="which v28 surface to use; 'auto' picks the best on disk")
    args = ap.parse_args()

    V28_DENSE = V28.parent / 'surfaces_dense'   # populated by compute_dense.py
    for ds in args.datasets:
        # Prefer a v28 dense surface (K=150 / 200) from compute_dense.py.
        # Fall back to v27 dense (FAM v27 'none' for C-MAPSS, v26 revin
        # for anomaly). Never use the v28 sparse Phase-2/3 surfaces here -
        # those are K=7 only and will mismatch the dense Chronos-2 grid.
        fam_path, fam_label = None, ''
        v28_dense = V28_DENSE / f'dense_fam_v28_{ds}_s42.npz'
        if v28_dense.exists():
            fam_path, fam_label = v28_dense, 'FAM v28'
        else:
            for v27_cand, lbl in [(V27 / f'dense_fam_v27_{ds}_s42.npz', 'FAM v27 none'),
                                  (V27 / f'dense_fam_v26_{ds}_s42.npz', 'FAM v26 revin')]:
                if v27_cand.exists():
                    fam_path, fam_label = v27_cand, lbl; break

        chronos_path = V27 / f'dense_chronos2_{ds}_s42.npz'
        if fam_path is None and not chronos_path.exists():
            print(f"  SKIP {ds}: no surfaces found"); continue
        render_triplet(ds, fam_path or Path('/dev/null'),
                       chronos_path, fam_label or 'FAM (unavailable)')


if __name__ == '__main__':
    main()

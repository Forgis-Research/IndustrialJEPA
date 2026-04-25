"""V28 — render dense pooled-surface comparison PNGs WITH grayscale delta panels.

Per the `feedback_dense_delta_surfaces.md` agent memory: result notebooks
should pair predicted p (color) with |p - y| in cmap='gray_r' (white = perfect,
black = wrong) so the visual contrast immediately tells the reader which model
is closer to ground truth without needing to compare metrics in their head.

Layout per dataset:
  Top row:    model 1 p | model 2 p | (model 3 p) | ground truth y
  Bottom row: |model 1 - y| | |model 2 - y| | (|model 3 - y|)  (grayscale)

White = perfect (delta 0), black = wrong (delta 1). Per-panel metrics (pooled
AUPRC + mean |p-y|) are in each title.

Models per dataset:
  C-MAPSS FD001/2/3:  FAM v27 none (s42 dense) | FAM v28 best (s42 dense) | Chronos-2 | GT
  SMAP/MSL/PSM/MBA:   FAM v26 revin | FAM v28 best | Chronos-2 | GT
  SMD:                FAM v28 baseline | (no v27 dense, no Chronos dense) | GT
  GECCO/BATADAL:      FAM v28 best | Chronos-2 | GT

Output: experiments/v28/results/surface_pngs/dense_delta_<ds>.png. PNGs only —
the .npz surfaces stay on the VM per the memory rule.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V27 = FAM_DIR / 'experiments/v27/surfaces'
V28 = FAM_DIR / 'experiments/v28/surfaces_dense'
OUT = FAM_DIR / 'experiments/v28/results/surface_pngs'
OUT.mkdir(parents=True, exist_ok=True)


def load(path: Path):
    if not path.exists() or path.stat().st_size < 100:
        return None
    try:
        d = np.load(path, allow_pickle=True)
    except (OSError, EOFError):
        return None
    return {'p': d['p_surface'], 'y': d['y_surface'].astype(np.int8),
            'hz': d['horizons']}


def downsample_x(arr, max_cols=3000):
    N = arr.shape[0]
    if N <= max_cols:
        return arr, np.arange(N), 1
    step = int(np.ceil(N / max_cols))
    idx = np.arange(0, N, step)
    return arr[idx], idx, step


def pooled(p, y):
    yi = y.ravel().astype(int)
    if yi.sum() == 0 or yi.sum() == len(yi):
        return float('nan')
    return float(average_precision_score(yi, p.ravel()))


def render(dataset: str, panels: list):
    """panels = list of (label, surface_dict | None) for each MODEL panel.

    The ground-truth panel is appended automatically as the last column.
    """
    model_panels = [(lbl, s) for lbl, s in panels if s is not None]
    if not model_panels:
        print(f"  SKIP {dataset}: no surfaces"); return

    ref = model_panels[0][1]
    hz = ref['hz']
    y_full = ref['y'].astype(np.float32)
    N = y_full.shape[0]
    y_ds, x_ds, step = downsample_x(y_full)

    n_models = len(model_panels)
    n_cols = n_models + 1   # +1 for GT in top row
    fig, axes = plt.subplots(2, n_cols, figsize=(3.4 * n_cols, 6.5),
                             sharey=True,
                             gridspec_kw={'height_ratios': [1, 1]})
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Top row: predicted p (viridis)
    last_im_color = None
    for i, (lbl, s) in enumerate(model_panels):
        p_ds, _, _ = downsample_x(s['p'])
        auprc = pooled(s['p'], s['y'])
        mean_delta = float(np.abs(s['p'] - s['y'].astype(np.float32)).mean())
        ax = axes[0, i]
        im = ax.pcolormesh(x_ds, hz, p_ds.T, cmap='viridis', vmin=0, vmax=1,
                           shading='auto')
        ax.set_yscale('log')
        ax.set_title(f"{lbl}\np(t,Δt)  AUPRC={auprc:.3f}  "
                     f"|p-y|={mean_delta:.3f}", fontsize=9)
        if i == 0:
            ax.set_ylabel('Δt (log)')
        last_im_color = im

    # GT panel (top-right)
    ax_gt = axes[0, -1]
    im_gt = ax_gt.pcolormesh(x_ds, hz, y_ds.T, cmap='viridis', vmin=0, vmax=1,
                             shading='auto')
    ax_gt.set_yscale('log')
    ax_gt.set_title('ground truth\ny(t, Δt)', fontsize=9)
    fig.colorbar(im_gt, ax=ax_gt, pad=0.02, label='value', shrink=0.85)

    # Bottom row: |p - y| in grayscale (white=0, black=1)
    last_im_delta = None
    for i, (lbl, s) in enumerate(model_panels):
        delta = np.abs(s['p'].astype(np.float32) - s['y'].astype(np.float32))
        d_ds, _, _ = downsample_x(delta)
        mean_d = float(delta.mean())
        ax = axes[1, i]
        im_d = ax.pcolormesh(x_ds, hz, d_ds.T, cmap='gray_r', vmin=0, vmax=1,
                             shading='auto')
        ax.set_yscale('log')
        ax.set_title(f'|{lbl} − y|   mean={mean_d:.3f}', fontsize=9)
        ax.set_xlabel(f'sample index'
                      + (f' (every {step})' if step > 1 else ''))
        if i == 0:
            ax.set_ylabel('Δt (log)')
        last_im_delta = im_d

    # Bottom-right cell: just colorbar for delta row
    axes[1, -1].axis('off')
    if last_im_delta is not None:
        fig.colorbar(last_im_delta, ax=axes[1, -1], pad=0.02, label='|p−y|',
                     shrink=0.85, fraction=0.4)

    plt.suptitle(f'{dataset}  pooled test surfaces, dense Δt  (N={N})',
                 fontsize=11, y=1.0)
    plt.tight_layout()
    out = OUT / f'dense_delta_{dataset}.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out.relative_to(FAM_DIR)}")


def main():
    # For each dataset: list (label, path) of MODEL panels (GT auto-appended).
    # Order: v27 baseline (where it exists), v28 best, Chronos-2.
    spec = {
        'FD001': [
            ('FAM v27 none', V27 / 'dense_fam_v27_FD001_s42.npz'),
            ('FAM v28 lag+none', V28 / 'dense_fam_v28_FD001_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_FD001_s42.npz'),
        ],
        'FD002': [
            ('FAM v27 none', V27 / 'dense_fam_v27_FD002_s42.npz'),
            ('FAM v28 dense_ft', V28 / 'dense_fam_v28_FD002_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_FD002_s42.npz'),
        ],
        'FD003': [
            ('FAM v27 none', V27 / 'dense_fam_v27_FD003_s42.npz'),
            ('FAM v28 dense_ft', V28 / 'dense_fam_v28_FD003_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_FD003_s42.npz'),
        ],
        'SMAP': [
            ('FAM v26 revin', V27 / 'dense_fam_v26_SMAP_s42.npz'),
            ('FAM v28 dense_ft', V28 / 'dense_fam_v28_SMAP_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_SMAP_s42.npz'),
        ],
        'MSL': [
            ('FAM v26 revin', V27 / 'dense_fam_v26_MSL_s42.npz'),
            ('FAM v28 dense_ft', V28 / 'dense_fam_v28_MSL_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_MSL_s42.npz'),
        ],
        'PSM': [
            ('FAM v26 revin', V27 / 'dense_fam_v26_PSM_s42.npz'),
            ('FAM v28 baseline', V28 / 'dense_fam_v28_PSM_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_PSM_s42.npz'),
        ],
        'SMD': [
            ('FAM v28 baseline', V28 / 'dense_fam_v28_SMD_s42.npz'),
        ],
        'MBA': [
            ('FAM v26 revin', V27 / 'dense_fam_v26_MBA_s42.npz'),
            ('FAM v28 lag+revin', V28 / 'dense_fam_v28_MBA_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_MBA_s42.npz'),
        ],
        'GECCO': [
            ('FAM v28 baseline', V28 / 'dense_fam_v28_GECCO_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_GECCO_s42.npz'),
        ],
        'BATADAL': [
            ('FAM v28 lag+revin', V28 / 'dense_fam_v28_BATADAL_s42.npz'),
            ('Chronos-2', V27 / 'dense_chronos2_BATADAL_s42.npz'),
        ],
    }
    for ds, entries in spec.items():
        panels = [(lbl, load(path)) for lbl, path in entries]
        render(ds, panels)


if __name__ == '__main__':
    main()

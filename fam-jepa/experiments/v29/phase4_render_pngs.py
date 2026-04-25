"""V29 Phase 4: surface PNGs with grayscale |p-y| delta panels.

For every dataset × every available model, render a 2-row figure:

  Row 1 (FAM):       p(t,Δt)  |  ground truth y  |  |p-y| (gray_r)
  Row 2 (Chronos-2): p(t,Δt)  |  ground truth y  |  |p-y| (gray_r)

Linear y-axis. Viridis for p and y (0-1). Grayscale gray_r for the error
panel (white = perfect, black = wrong) with mean |p-y| in the title — the
single most readable diagnostic.

The error panel is the v28 lesson: pooled metrics can hide a lot, but a
visual delta map immediately tells the reader which model is closer to
ground truth at each (t, Δt) cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V29_SURF = FAM_DIR / 'experiments/v29/surfaces'
V28_SURF_DENSE = FAM_DIR / 'experiments/v28/surfaces_dense'
V27_SURF = FAM_DIR / 'experiments/v27/surfaces'
OUT = FAM_DIR / 'experiments/v29/results/surface_pngs'
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
    return arr[::step], np.arange(0, N, step), step


def mean_h_auroc(p, y):
    K = p.shape[1]
    valid = [i for i in range(K) if 0 < y[:, i].mean() < 1]
    if not valid:
        return float('nan')
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in valid]))


def render_two_row(dataset: str, fam: dict, chr_: dict, out: Path):
    """Render the two-row layout. fam and chr_ may both be None or partial.

    Models present: build a row per model. Each row has 3 panels.
    """
    rows = []
    if fam is not None:
        rows.append(('FAM', fam))
    if chr_ is not None:
        rows.append(('Chronos-2', chr_))
    if not rows:
        print(f"  SKIP {dataset}: no surfaces"); return

    # All rows must use a common y axis (horizons) — pick FAM's if present.
    ref = rows[0][1]
    hz = ref['hz']
    N = ref['y'].shape[0]
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.5 * n_rows),
                             sharey=True)
    if n_rows == 1:
        axes = axes.reshape(1, 3)

    last_im_p = None
    last_im_d = None
    for ri, (label, surf) in enumerate(rows):
        p = surf['p']
        y = surf['y'].astype(np.float32)
        p_ds, x_ds, step = downsample_x(p)
        y_ds, _, _ = downsample_x(y)
        delta = np.abs(p - y)
        d_ds, _, _ = downsample_x(delta)
        h = mean_h_auroc(p, surf['y'])
        ap = float(average_precision_score(y.ravel(), p.ravel()))
        mean_d = float(delta.mean())

        ax = axes[ri, 0]
        last_im_p = ax.pcolormesh(x_ds, hz, p_ds.T, cmap='viridis',
                                  vmin=0, vmax=1, shading='auto')
        ax.set_title(f"{label}  p(t,Δt)\nh-AUROC={h:.3f}  AUPRC={ap:.3f}",
                     fontsize=9)
        ax.set_ylabel('Δt')

        ax = axes[ri, 1]
        ax.pcolormesh(x_ds, hz, y_ds.T, cmap='viridis', vmin=0, vmax=1,
                      shading='auto')
        ax.set_title(f'ground truth y(t,Δt)', fontsize=9)

        ax = axes[ri, 2]
        last_im_d = ax.pcolormesh(x_ds, hz, d_ds.T, cmap='gray_r',
                                  vmin=0, vmax=1, shading='auto')
        ax.set_title(f"|{label} - y|   mean={mean_d:.3f}", fontsize=9)

        if ri == n_rows - 1:
            for c in range(3):
                axes[ri, c].set_xlabel(f'sample index'
                                       + (f' (every {step})'
                                          if step > 1 else ''))

    if last_im_p is not None:
        fig.colorbar(last_im_p, ax=axes[:, :2], pad=0.02, label='value',
                     shrink=0.8)
    if last_im_d is not None:
        fig.colorbar(last_im_d, ax=axes[:, 2:], pad=0.02, label='|p − y|',
                     shrink=0.8)

    plt.suptitle(f'{dataset}  pooled test surfaces  (N={N})',
                 fontsize=11, y=0.995)
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out.relative_to(FAM_DIR)}")


# Per-dataset surface paths. v29 surfaces for new datasets; v28 dense or
# v27 for legacy. Chronos-2 dense exists at v27/surfaces only (single seed).
def fam_surface_path(ds: str, predictor_kind: str = 'mlp') -> Path:
    norm = 'none' if ds.startswith('FD') else 'revin'
    extra = '' if predictor_kind == 'mlp' else '_xpred'
    return V29_SURF / f'{ds}_{norm}{extra}_s42.npz'


def chronos_surface_path(ds: str) -> Path:
    return V27_SURF / f'dense_chronos2_{ds}_s42.npz'


def main():
    # Use v29 surfaces if available; otherwise fall back to v28 dense.
    DATASETS = ['FD001', 'FD002', 'FD003', 'SMAP', 'MSL', 'PSM', 'SMD',
                'MBA', 'GECCO', 'BATADAL', 'SKAB', 'ETTm1', 'CHBMIT']

    for ds in DATASETS:
        # Try v29 first, then v28 dense, then v27
        fam_path = fam_surface_path(ds)
        if not fam_path.exists():
            # fall back to v28 dense, naming convention dense_fam_v28_<ds>_s42.npz
            cand = V28_SURF_DENSE / f'dense_fam_v28_{ds}_s42.npz'
            if cand.exists():
                fam_path = cand
            else:
                src = 'v27' if ds.startswith('FD') else 'v26'
                cand = V27_SURF / f'dense_fam_{src}_{ds}_s42.npz'
                if cand.exists():
                    fam_path = cand
                else:
                    fam_path = None

        chr_path = chronos_surface_path(ds)
        chr_ = load(chr_path) if chr_path else None
        fam = load(fam_path) if fam_path else None

        # If both FAM and Chronos exist but their horizons differ, only render
        # them together if grids match — otherwise render separately.
        if fam is not None and chr_ is not None:
            if list(fam['hz']) != list(chr_['hz']):
                # mismatched grids: emit two single-row plots
                render_two_row(f'{ds}_fam', fam, None, OUT / f'panels_{ds}_fam.png')
                render_two_row(f'{ds}_chronos', None, chr_, OUT / f'panels_{ds}_chr.png')
                continue
        out = OUT / f'panels_{ds}.png'
        render_two_row(ds, fam, chr_, out)


if __name__ == '__main__':
    main()

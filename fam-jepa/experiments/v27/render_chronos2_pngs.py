"""V27: render comparison PNGs showing FAM v26 / FAM v27 / Chronos-2 / GT
pooled test p-surfaces for every dataset.

One PNG per dataset at ``experiments/v27/results/surface_pngs/
chronos2_compare_<dataset>.png``. Panels chosen per dataset:

  - FD001/FD002/FD003:  FAM v26 | FAM v27 none | Chronos-2 | GT (4 panels)
  - SMAP/MSL/PSM/MBA:   FAM v26 | Chronos-2 | GT (3 panels)
  - GECCO/BATADAL:      Chronos-2 | GT (2 panels, no FAM baseline)

X-axis is sample index in the concatenated test loader (deterministic order,
identical across FAM and Chronos-2 because both load the same dataset with
``shuffle=False``). Y-axis is horizon Δt on a log scale.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V27 = FAM_DIR / 'experiments/v27'
V26 = FAM_DIR / 'experiments/v26'
OUT = V27 / 'results/surface_pngs'
OUT.mkdir(parents=True, exist_ok=True)


def pooled_metrics(p, y):
    y_flat = y.ravel().astype(int)
    p_flat = p.ravel()
    if y_flat.sum() == 0 or y_flat.sum() == len(y_flat):
        return float('nan'), float('nan')
    return (float(average_precision_score(y_flat, p_flat)),
            float(roc_auc_score(y_flat, p_flat)))


def downsample_x(arr_2d, max_cols=4000):
    """If surface has too many samples, take every kth for readable plot."""
    N = arr_2d.shape[0]
    if N <= max_cols:
        return arr_2d, np.arange(N), 1
    step = int(np.ceil(N / max_cols))
    idx = np.arange(0, N, step)
    return arr_2d[idx], idx, step


def render_one(dataset: str, panels: list):
    """panels: list of (array (N, K), horizons, title) for each column.

    Equal-height rows (horizons are plotted as categorical bands); y-tick
    labels show the actual Δt. This makes small horizons visible even
    when the horizon grid is highly non-uniform.
    """
    ref = panels[0][0]
    N = ref.shape[0]
    horizons = panels[0][1]
    K = len(horizons)
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4),
                             sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (arr, hz, title) in zip(axes, panels):
        arr_ds, x_ds, step = downsample_x(arr)
        im = ax.imshow(arr_ds.T, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       origin='lower',
                       extent=[x_ds[0], x_ds[-1], -0.5, K - 0.5],
                       interpolation='nearest')
        ax.set_yticks(range(K))
        ax.set_yticklabels([str(h) for h in hz], fontsize=8)
        xlab = f'test sample index (every {step}th)' if step > 1 else 'test sample index'
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel('horizon Δt')
    fig.colorbar(im, ax=axes[-1], pad=0.02, label='value')
    plt.suptitle(f'{dataset}  pooled test surface  (N={N} samples)',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = OUT / f'chronos2_compare_{dataset}.png'
    plt.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out.relative_to(FAM_DIR)}")


def load_fam_surface(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {'p': d['p_surface'], 'y': d['y_surface'],
            'hz': d['horizons'].tolist()}


def main():
    # Mapping: dataset -> list of panels to assemble
    # Each panel: (source_name, surface_path_or_None)
    sources = {
        'FD001': [
            ('FAM v26 revin',   V26 / 'surfaces/FD001_s42.npz'),
            ('FAM v27 none',    V27 / 'surfaces/FD001_none_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_FD001_s42.npz'),
        ],
        'FD002': [
            ('FAM v26 revin',   V26 / 'surfaces/FD002_s42.npz'),
            ('FAM v27 none',    V27 / 'surfaces/FD002_none_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_FD002_s42.npz'),
        ],
        'FD003': [
            ('FAM v26 revin',   V26 / 'surfaces/FD003_s42.npz'),
            ('FAM v27 none',    V27 / 'surfaces/FD003_none_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_FD003_s42.npz'),
        ],
        'SMAP': [
            ('FAM v26 revin',   V26 / 'surfaces/SMAP_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_SMAP_s42.npz'),
        ],
        'MSL': [
            ('FAM v26 revin',   V26 / 'surfaces/MSL_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_MSL_s42.npz'),
        ],
        'PSM': [
            ('FAM v26 revin',   V26 / 'surfaces/PSM_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_PSM_s42.npz'),
        ],
        'MBA': [
            ('FAM v26 revin',   V26 / 'surfaces/MBA_s42.npz'),
            ('Chronos-2',       V27 / 'surfaces/chronos2_MBA_s42.npz'),
        ],
        'GECCO': [
            ('Chronos-2',       V27 / 'surfaces/chronos2_GECCO_s42.npz'),
        ],
        'BATADAL': [
            ('Chronos-2',       V27 / 'surfaces/chronos2_BATADAL_s42.npz'),
        ],
    }

    for ds, src_list in sources.items():
        panels = []
        gt_added = False
        for label, path in src_list:
            s = load_fam_surface(path)
            if s is None:
                print(f"  {ds}: SKIP panel '{label}' - missing {path}")
                continue
            auprc, auroc = pooled_metrics(s['p'], s['y'])
            title = f'{label}\np(t,Δt)  AUPRC={auprc:.3f}'
            panels.append((s['p'], s['hz'], title))
            # Add GT panel once (from the first available surface - labels are same)
            if not gt_added:
                panels.append((s['y'].astype(np.float32), s['hz'],
                               f'ground truth\ny(t,Δt)'))
                gt_added = True
        if not panels:
            continue
        # Reorder panels so GT is last
        p_panels = [pn for pn in panels if 'ground truth' not in pn[2]]
        g_panels = [pn for pn in panels if 'ground truth' in pn[2]]
        panels = p_panels + g_panels
        render_one(ds, panels)


if __name__ == '__main__':
    main()

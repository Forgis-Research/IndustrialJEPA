"""V27 surface PNG exporter.

Iterates over every ``experiments/v27/surfaces/paper_*.npz`` and writes
a two-panel PNG (predicted p | ground truth y) into
``experiments/v27/results/surface_pngs/``. Each PNG is a self-contained
visual check of one entity/engine; no notebook needed.

Also writes the high-priority FD001 comparison (v26 revin vs v27 none
vs ground truth) for engines 49 / 93 / 91 as 3-column PNGs.
"""

import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
SURF = FAM_DIR / 'experiments/v27/surfaces'
OUT = FAM_DIR / 'experiments/v27/results/surface_pngs'
OUT.mkdir(parents=True, exist_ok=True)


def render_pair(npz_path: Path, out_path: Path, title: str,
                xlabel: str = 'timestep t'):
    """Render predicted p and ground truth y as a two-panel heatmap."""
    d = np.load(npz_path, allow_pickle=True)
    p = d['p_surface']
    y = d['y_surface'].astype(np.float32)
    t_ix = d['t_index']
    hz = d['horizons']
    if len(t_ix) == 0:
        print(f"  SKIP {npz_path.name}: empty surface")
        return

    fig, (axp, axy) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    im = axp.pcolormesh(t_ix, hz, p.T, cmap='viridis', vmin=0, vmax=1,
                        shading='auto')
    axp.set_yscale('log')
    axp.set_xlabel(xlabel); axp.set_ylabel('horizon Δt (log)')
    axp.set_title(f'{title} - predicted p(t, Δt)', fontsize=10)

    axy.pcolormesh(t_ix, hz, y.T, cmap='viridis', vmin=0, vmax=1,
                   shading='auto')
    axy.set_yscale('log')
    axy.set_xlabel(xlabel)
    axy.set_title(f'{title} - ground truth y(t, Δt)', fontsize=10)

    fig.colorbar(im, ax=axy, pad=0.02, label='value')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(FAM_DIR)}")


def render_triplet(v26_npz: Path, v27_npz: Path, out_path: Path, title: str):
    """Render v26 p | v27 p | ground truth y side by side."""
    v26 = np.load(v26_npz, allow_pickle=True)
    v27 = np.load(v27_npz, allow_pickle=True)
    t_ix = v26['t_index']
    hz = v26['horizons']

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    panels = [
        (v26['p_surface'], f'{title} - v26 revin p(t, Δt)'),
        (v27['p_surface'], f'{title} - v27 none p(t, Δt)'),
        (v26['y_surface'].astype(np.float32),
         f'{title} - ground truth y(t, Δt)'),
    ]
    for ax, (arr, sub) in zip(axes, panels):
        im = ax.pcolormesh(t_ix, hz, arr.T, cmap='viridis', vmin=0, vmax=1,
                           shading='auto')
        ax.set_yscale('log')
        ax.set_xlabel('cycle t')
        ax.set_title(sub, fontsize=10)
    axes[0].set_ylabel('horizon Δt (log)')
    fig.colorbar(im, ax=axes[-1], pad=0.02, label='value')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_path.relative_to(FAM_DIR)}")


def main():
    # --- Per-file pair PNGs (predicted + ground truth) ---
    paper_files = sorted(SURF.glob('paper_*.npz'))
    print(f"Rendering {len(paper_files)} paper surfaces as PNG pairs:")
    for npz in paper_files:
        m = re.match(r'paper_(?P<ds>\w+?)_(?P<rest>.+)_dense\.npz$', npz.name)
        if m:
            ds = m.group('ds')
            rest = m.group('rest')
            title = f'{ds} {rest}'
            xlabel = 'cycle t' if ds == 'FD001' else 'test timestep t'
        else:
            title = npz.stem
            xlabel = 'timestep t'
        out = OUT / (npz.stem + '.png')
        render_pair(npz, out, title, xlabel=xlabel)

    # --- FD001 triplets (v26 | v27 | GT) for paper Figure 3 ---
    print("\nRendering FD001 v26-vs-v27-vs-GT triplets:")
    for eid in [49, 93, 91]:
        v26 = SURF / f'paper_FD001_e{eid}_v26_revin_s42_dense.npz'
        v27 = SURF / f'paper_FD001_e{eid}_v27_none_s42_dense.npz'
        if v26.exists() and v27.exists():
            out = OUT / f'triplet_FD001_e{eid}_v26_v27_gt.png'
            render_triplet(v26, v27, out, f'FD001 engine {eid}')

    print(f"\nAll PNGs written to {OUT.relative_to(FAM_DIR)}")


if __name__ == '__main__':
    main()

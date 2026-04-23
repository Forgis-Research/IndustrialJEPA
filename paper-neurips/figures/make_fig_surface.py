"""Generate a publication-quality probability surface figure from v24 data.

Uses a real FD001 test engine's p(t, Δt) alongside the ground-truth y(t, Δt)
to illustrate the surface, monotonicity, and interval arithmetic. Saves
to fig_probability_surface.pdf.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Publication-quality defaults
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times', 'Nimbus Roman No9 L'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'mathtext.fontset': 'cm',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.dpi': 150,
})


FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
SURF = FAM / 'experiments/v24/surfaces'


def load_surface_with_single_engine(ds='FD001', seed=42):
    """Load FD001 surface and return one engine's slice (first 250 rows)."""
    d = np.load(SURF / f'{ds}_s{seed}.npz')
    p = d['p_surface']
    y = d['y_surface']
    t = d['t_index']
    h = d['horizons']
    # Pick contiguous rows from one engine: smallest stretch where t_index
    # is strictly monotone (t_index resets each engine boundary).
    best_start = 0
    best_len = 0
    s = 0
    for i in range(1, len(t)):
        if t[i] > t[i-1]:
            if i - s > best_len:
                best_len = i - s
                best_start = s
        else:
            s = i
    end = best_start + best_len
    return p[best_start:end], y[best_start:end], t[best_start:end], h


def make_figure():
    p, y, t, h = load_surface_with_single_engine('FD001', 42)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3),
                              gridspec_kw={'width_ratios': [1.2, 1.2, 0.95]})

    # --- Panel A: p(t, Δt) heatmap ---
    ax = axes[0]
    im = ax.imshow(p.T, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                   origin='lower',
                   extent=[t[0], t[-1], -0.5, len(h) - 0.5])
    ax.set_yticks(range(len(h)))
    ax.set_yticklabels([str(hh) for hh in h])
    ax.set_xlabel(r'observation time $t$ (cycles)')
    ax.set_ylabel(r'horizon $\Delta t$')
    ax.set_title(r'(a) predicted $p(t, \Delta t)$',
                 loc='left', pad=4, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('prob.', fontsize=8)

    # --- Panel B: y(t, Δt) ground truth ---
    ax = axes[1]
    ax.imshow(y.T, aspect='auto', cmap='Reds', vmin=0, vmax=1, origin='lower',
              extent=[t[0], t[-1], -0.5, len(h) - 0.5])
    ax.set_yticks(range(len(h)))
    ax.set_yticklabels([str(hh) for hh in h])
    ax.set_xlabel(r'observation time $t$ (cycles)')
    ax.set_title(r'(b) ground truth $y(t, \Delta t)$',
                 loc='left', pad=4, fontweight='bold')

    # --- Panel C: marginal p at fixed t, with interval-arithmetic callout ---
    ax = axes[2]
    mid = p.shape[0] // 2 + 30
    p_t = p[mid]
    ax.plot(h, p_t, 'o-', color='#2E5FAB', lw=1.5, ms=4)
    # Hatched band: P(event in (dt=50, dt=100])
    try:
        k50 = int(np.where(h == 50)[0][0])
        k100 = int(np.where(h == 100)[0][0])
        ax.axhspan(p_t[k50], p_t[k100], alpha=0.3, color='orange',
                   label=r'$P(\mathrm{event} \in (t{+}50, t{+}100])$' + '\n'
                         + r'$= p(t,100) - p(t,50)$')
    except Exception:
        pass
    ax.set_xlabel(r'horizon $\Delta t$ (cycles)')
    ax.set_ylabel(r'$p(t_0, \Delta t)$')
    ax.set_xscale('log')
    ax.set_title(r'(c) marginal at fixed $t_0$',
                 loc='left', pad=4, fontweight='bold')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0.9, h[-1] * 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=7, framealpha=0.9)

    plt.tight_layout(pad=0.4, w_pad=0.6)

    out = Path(__file__).parent / 'fig_probability_surface.pdf'
    plt.savefig(out, bbox_inches='tight', pad_inches=0.02)
    png = Path(__file__).parent / 'fig_probability_surface.png'
    plt.savefig(png, bbox_inches='tight', pad_inches=0.02, dpi=300)
    print(f'wrote {out}')
    print(f'wrote {png}')


if __name__ == '__main__':
    make_figure()

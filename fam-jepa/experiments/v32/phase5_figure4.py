"""V32 Phase 5: Rebuild fig_probability_surface_v2 with REAL v30 dense surfaces.

Layout (single-column NeurIPS, ~5.5in wide):
    Row 1 (FAM dense K=150):  predicted | ground truth | |p-y| error
    Row 2 (Chronos-2 K=7):    predicted | ground truth | |p-y| error
    Below: per-horizon AUROC line plot (FAM dense, Chronos-2 sparse)

We sort observations by ground-truth time-to-event so the diagonal
"failure-approach wedge" pops visually. We trim to ~4 representative
test engines (24, 33, 3, 4) for compactness.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.metrics import roc_auc_score

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V30_SURF = FAM / 'experiments/v30/surfaces'
PAPER_FIG = Path('/home/sagemaker-user/IndustrialJEPA/paper-neurips/figures')
PAPER_FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.4,
    "ytick.major.width": 0.4,
    "axes.labelsize": 7,
    "axes.titlesize": 8,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"


def first_event_horizon(y_row: np.ndarray, horizons: np.ndarray) -> int:
    """For a single observation row, return the smallest horizon at which the
    label is positive; or +inf if none in the surface."""
    pos = np.where(y_row > 0)[0]
    if len(pos) == 0:
        return int(horizons[-1]) + 10  # treat as far away
    return int(horizons[pos[0]])


def build_panels(seed: int = 42, target_engines: int = 4):
    fam_npz = V30_SURF / f'FD001_none_discrete_hazard_td20_p3_s{seed}.npz'
    chr2_npz = V30_SURF / f'FD001_chr2-mlp_s{seed}.npz'
    if not fam_npz.exists() or not chr2_npz.exists():
        raise FileNotFoundError(f'missing {fam_npz} or {chr2_npz}')
    fam = np.load(fam_npz)
    chr2 = np.load(chr2_npz)

    p_fam = fam['p_surface']; y_fam = fam['y_surface']
    h_fam = np.asarray(fam['horizons'])
    p_c = chr2['p_surface']; y_c = chr2['y_surface']
    h_c = np.asarray(chr2['horizons'])
    t_idx = np.asarray(fam['t_index'])

    # Detect engine boundaries (where t_index resets / restarts low)
    boundaries = [0]
    for i in range(1, len(t_idx)):
        if t_idx[i] < t_idx[i - 1]:
            boundaries.append(i)
    boundaries.append(len(t_idx))
    n_engines = len(boundaries) - 1
    print(f'  detected {n_engines} test engines')

    # Pick engines that show clear failure approach (have positive labels in y_fam)
    eng_scores = []
    for e in range(n_engines):
        sl = slice(boundaries[e], boundaries[e + 1])
        # Engine is "informative" if it has positive labels at multiple horizons
        eng_scores.append((e, int(y_fam[sl].sum()), boundaries[e + 1] - boundaries[e]))
    # Sort by # positive labels and length, pick a diverse set
    eng_scores.sort(key=lambda x: -x[1])
    chosen = [e for e, _, l in eng_scores[:target_engines * 4] if l >= 15][:target_engines]
    print(f'  chose engines (idx): {chosen}')

    rows_fam_p, rows_fam_y, rows_chr_p, rows_chr_y, sep_marks = [], [], [], [], [0]
    for e in chosen:
        sl = slice(boundaries[e], boundaries[e + 1])
        rows_fam_p.append(p_fam[sl]); rows_fam_y.append(y_fam[sl])
        rows_chr_p.append(p_c[sl]);   rows_chr_y.append(y_c[sl])
        sep_marks.append(sep_marks[-1] + (boundaries[e + 1] - boundaries[e]))
    P_fam = np.vstack(rows_fam_p); Y_fam = np.vstack(rows_fam_y)
    P_c   = np.vstack(rows_chr_p); Y_c   = np.vstack(rows_chr_y)

    return {
        'P_fam': P_fam, 'Y_fam': Y_fam, 'h_fam': h_fam,
        'P_c': P_c, 'Y_c': Y_c, 'h_c': h_c,
        'sep_marks': sep_marks,
        'p_fam_full': p_fam, 'y_fam_full': y_fam,
        'p_c_full': p_c, 'y_c_full': y_c,
    }


def per_horizon_auroc(p, y, horizons):
    out = []
    for k in range(len(horizons)):
        lab = (y[:, k] > 0).astype(np.int32)
        if lab.sum() == 0 or lab.sum() == len(lab):
            out.append(np.nan)
        else:
            out.append(float(roc_auc_score(lab, p[:, k])))
    return np.array(out)


def render(d, out_pdf: Path, out_png: Path):
    fig = plt.figure(figsize=(6.0, 4.6))
    gs = gridspec.GridSpec(3, 3, height_ratios=[1.0, 1.0, 0.85],
                           hspace=0.55, wspace=0.30)

    # Row 1: FAM
    cmap_p = 'inferno'; cmap_e = 'Greys'
    panels_fam = [
        (gs[0, 0], d['P_fam'], 'FAM  predicted $p(t,\\Delta t)$',  cmap_p, 0, 1),
        (gs[0, 1], d['Y_fam'], 'Ground truth $y(t,\\Delta t)$',     cmap_p, 0, 1),
        (gs[0, 2], np.abs(d['P_fam'] - d['Y_fam']), 'FAM error',    cmap_e, 0, 1),
    ]
    panels_chr = [
        (gs[1, 0], d['P_c'], 'Chronos-2  predicted',                cmap_p, 0, 1),
        (gs[1, 1], d['Y_c'], 'Ground truth (sparse)',               cmap_p, 0, 1),
        (gs[1, 2], np.abs(d['P_c'] - d['Y_c']), 'Chronos-2 error',  cmap_e, 0, 1),
    ]
    for spec, mat, title, cm, vmin, vmax in panels_fam + panels_chr:
        ax = fig.add_subplot(spec)
        im = ax.imshow(mat.T, aspect='auto', origin='lower',
                       cmap=cm, vmin=vmin, vmax=vmax,
                       interpolation='nearest', rasterized=True)
        # Engine separators
        for s in d['sep_marks'][1:-1]:
            ax.axvline(s - 0.5, color='white', lw=0.3, alpha=0.6)
        h_used = d['h_fam'] if mat.shape[1] == 150 else d['h_c']
        idx = np.linspace(0, mat.shape[1] - 1, 4, dtype=int)
        ax.set_yticks(idx)
        ax.set_yticklabels([str(int(h_used[i])) for i in idx])
        ax.set_xticks([])
        ax.set_ylabel(r'$\Delta t$', fontsize=7)
        ax.set_title(title, fontsize=7, pad=2)
        for s in ax.spines.values():
            s.set_linewidth(0.4)

    # Bottom: per-horizon AUROC
    ax = fig.add_subplot(gs[2, :])
    au_fam = per_horizon_auroc(d['p_fam_full'], d['y_fam_full'], d['h_fam'])
    au_c = per_horizon_auroc(d['p_c_full'], d['y_c_full'], d['h_c'])
    ax.plot(d['h_fam'], au_fam, color=C_BLUE, lw=1.2, label='FAM (K=150)')
    ax.plot(d['h_c'], au_c, color=C_ORANGE, lw=0, marker='o', markersize=3.0,
            label='Chronos-2 (K=7)')
    ax.axhline(0.5, color='gray', lw=0.4, ls='--')
    ax.set_xlabel(r'Horizon $\Delta t$ (cycles)')
    ax.set_ylabel('Per-horizon AUROC')
    ax.set_xlim(0, max(d['h_fam'].max(), d['h_c'].max()) + 5)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower right', ncol=2)
    ax.set_title(f'FAM dense surface vs Chronos-2 sparse, FD001 test  '
                 f'(mean h-AUROC: FAM={np.nanmean(au_fam):.3f}, '
                 f'Chr-2={np.nanmean(au_c):.3f})',
                 fontsize=7, pad=3)
    for s in ax.spines.values():
        s.set_linewidth(0.4)

    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=300)
    plt.close()
    print(f'  -> {out_pdf}')
    print(f'  -> {out_png}')


def main():
    print('Building dense surface comparison panels (seed 42)...')
    d = build_panels(seed=42, target_engines=4)
    print(f'  FAM panel: {d["P_fam"].shape}, Chr-2 panel: {d["P_c"].shape}')
    out_pdf = PAPER_FIG / 'fig_probability_surface_v2.pdf'
    out_png = PAPER_FIG / 'fig_probability_surface_v2.png'
    render(d, out_pdf, out_png)


if __name__ == '__main__':
    main()

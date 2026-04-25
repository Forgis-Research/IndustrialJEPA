"""V30 Phase 5: paper figures.

5a. fig_probability_surface_v2.{pdf,png} — 6-panel FAM/Chronos-2 surfaces +
    per-horizon AUROC curve. Source: FD001 seed 42 Phase 3 surface
    (FAM dense discrete) + Phase 1 chr2-probe surface.

5b. fig_benchmark_hauroc.pdf — grouped bar chart, FAM vs Chronos-2 across
    all benchmark datasets (sorted desc by FAM h-AUROC). Source:
    results/master_table.json + Chronos-2 numbers from Phase 1/v24.

5c. surface gallery: per-dataset 3-panel PNGs go in results/surface_pngs/
    (Phase 3 already writes these per seed; this script just curates the
    seed-42 PNG per dataset for the gallery).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _runner_v30 import RES_DIR, PNG_DIR, SURF_DIR

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

REPO = Path('/home/sagemaker-user/IndustrialJEPA')
PAPER_FIG = REPO / 'paper-neurips/figures'
PAPER_FIG.mkdir(parents=True, exist_ok=True)


def load_npz(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {'p_surface': d['p_surface'], 'y_surface': d['y_surface'],
            'horizons': d['horizons'].tolist(),
            't_index': d['t_index']}


def fig4_surface_v2(fam_npz: Path, chr2_npz: Path, out_pdf: Path):
    """Two-row 6-panel + per-horizon AUROC subplot."""
    fam = load_npz(fam_npz)
    chr2 = load_npz(chr2_npz)
    if fam is None:
        print(f"  MISSING fam {fam_npz}", flush=True); return
    if chr2 is None:
        print(f"  MISSING chr2 {chr2_npz}", flush=True); return

    plt.rcParams.update({'font.size': 11})
    fig = plt.figure(figsize=(18, 8))

    # 2x3 surfaces + 1x3 per-horizon AUROC curve at the bottom.
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 3, 2.5])

    for row, (lab, srf) in enumerate([('FAM', fam), ('Chronos-2', chr2)]):
        p = srf['p_surface']; y = srf['y_surface']
        # Sort by tte if available — for FD001 lifecycle this groups failures.
        order = np.argsort(srf['t_index'])
        p = p[order]; y = y[order]
        err = np.abs(p - y)
        for col, (mat, title, cmap, vmin, vmax) in enumerate([
            (p.T, f'{lab} predicted p(t,Δt)', 'viridis', 0, 1),
            (y.T, 'Ground truth y(t,Δt)', 'viridis', 0, 1),
            (err.T, f'|p − y| (mean={err.mean():.3f})', 'gray_r', 0, 1),
        ]):
            ax = fig.add_subplot(gs[row, col])
            im = ax.imshow(mat, aspect='auto', origin='lower', cmap=cmap,
                           vmin=vmin, vmax=vmax,
                           extent=[0, p.shape[0], srf['horizons'][0],
                                   srf['horizons'][-1]],
                           interpolation='nearest')
            ax.set_xlabel('Time step t')
            ax.set_ylabel('Horizon Δt')
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Per-horizon AUROC curve
    ax = fig.add_subplot(gs[2, :])
    for srf, lab, color in [(fam, 'FAM', 'tab:blue'),
                            (chr2, 'Chronos-2', 'tab:orange')]:
        p = srf['p_surface']; y = srf['y_surface']
        h = srf['horizons']
        aurocs = []
        for i in range(len(h)):
            yi = y[:, i]
            if 0 < yi.mean() < 1:
                aurocs.append((h[i], roc_auc_score(yi, p[:, i])))
        if aurocs:
            xs, ys = zip(*aurocs)
            ax.plot(xs, ys, '-', color=color, label=lab, linewidth=2)
    ax.axhline(0.5, ls=':', color='gray', alpha=0.7, label='chance')
    ax.set_xlabel('Horizon Δt (cycles)')
    ax.set_ylabel('AUROC')
    ax.set_title('Per-horizon AUROC — FD001')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_pdf} + .png", flush=True)


def fig5_benchmark_bars(master_table_path: Path, out_pdf: Path,
                        chr2_path: Path = None):
    """Grouped bar chart: FAM h-AUROC (100% labels) vs Chronos-2 across datasets."""
    if not master_table_path.exists():
        print(f"  MISSING {master_table_path}", flush=True); return
    mt = json.load(open(master_table_path))
    fam = mt['datasets']
    chr2 = json.load(open(chr2_path)) if chr2_path and chr2_path.exists() else {}
    rows = []
    for ds, row in fam.items():
        if 'lf100' in row and row['lf100']['mean_h_auroc'] is not None:
            f_mean = row['lf100']['mean_h_auroc']
            f_std = row['lf100']['std_h_auroc'] or 0
            c_mean = None; c_std = 0
            if chr2 and 'chr2-probe_hauroc' in chr2 and ds in chr2['chr2-probe_hauroc']:
                v = chr2['chr2-probe_hauroc'][ds]
                c_mean = v.get('mean')
                c_std = v.get('std') or 0
            rows.append((ds, f_mean, f_std, c_mean, c_std))
    rows.sort(key=lambda r: -r[1])

    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(rows))
    w = 0.4
    fam_means = [r[1] for r in rows]
    fam_stds = [r[2] for r in rows]
    chr2_means = [r[3] if r[3] is not None else 0 for r in rows]
    chr2_stds = [r[4] for r in rows]
    has_chr2 = [r[3] is not None for r in rows]

    ax.bar(x - w/2, fam_means, w, yerr=fam_stds, label='FAM (2.16M)',
           color='tab:blue', capsize=3)
    ax.bar([xi + w/2 for xi, hc in zip(x, has_chr2) if hc],
           [m for m, hc in zip(chr2_means, has_chr2) if hc], w,
           yerr=[s for s, hc in zip(chr2_stds, has_chr2) if hc],
           label='Chronos-2 (120M, probe)', color='tab:orange', capsize=3)
    ax.axhline(0.5, ls=':', color='gray', label='chance')
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], rotation=30, ha='right')
    ax.set_ylabel('Mean per-horizon AUROC')
    ax.set_title('FAM vs Chronos-2 across benchmark datasets (3 seeds, 100% labels)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {out_pdf}", flush=True)


def main():
    print(">>> Phase 5a: fig_probability_surface_v2 <<<", flush=True)
    fam_p3 = SURF_DIR / 'FD001_none_discrete_hazard_td20_p3_s42.npz'
    chr2_p1 = SURF_DIR / 'FD001_chr2-linear_s42.npz'
    fig4_surface_v2(fam_p3, chr2_p1, PAPER_FIG / 'fig_probability_surface_v2.pdf')

    print("\n>>> Phase 5b: fig_benchmark_hauroc <<<", flush=True)
    fig5_benchmark_bars(RES_DIR / 'master_table.json',
                        PAPER_FIG / 'fig_benchmark_hauroc.pdf',
                        chr2_path=RES_DIR / 'phase1_decision.json')


if __name__ == '__main__':
    main()

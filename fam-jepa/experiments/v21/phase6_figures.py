"""V21 Phase 6: Per-horizon AUPRC curves + reliability diagrams.

Reads stored .npz surfaces, computes per-horizon AUPRC + reliability,
saves PDF figures for the paper appendix.

Target datasets: C-MAPSS FD001 (pred-FT), SMAP (Mahal).
Optional: all anomaly datasets, all C-MAPSS modes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V21 = FAM / 'experiments' / 'v21'
sys.path.insert(0, str(V21))
sys.path.insert(0, str(FAM))

from pred_ft_utils import load_surface  # noqa: E402
from evaluation.surface_metrics import (  # noqa: E402
    auprc_per_horizon, reliability_diagram, evaluate_probability_surface,
    monotonicity_violation_rate,
)

FIG_DIR = V21 / 'figures'
FIG_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]


def plot_per_horizon_auprc(surfaces: dict, title: str, save_path: Path):
    """surfaces: {label: (p, y, horizons)} per seed."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=120)
    for label, (p, y, h) in surfaces.items():
        res = auprc_per_horizon(p, y, horizon_labels=list(h))
        ax.plot(h, res['auprc_per_k'], 'o-', label=label, alpha=0.85, markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel(r'Horizon $\Delta t$ (steps / cycles)')
    ax.set_ylabel('AUPRC')
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_reliability(surfaces: dict, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(4.5, 4.2), dpi=120)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect calibration')
    for label, (p, y, h) in surfaces.items():
        rd = reliability_diagram(p, y, n_bins=10)
        xs = np.asarray(rd['bin_means'])
        ys = np.asarray(rd['bin_freqs'])
        cs = np.asarray(rd['bin_counts'])
        nonzero = cs > 0
        ax.plot(xs[nonzero], ys[nonzero], 'o-', label=f"{label} (ECE={rd['ece']:.3f})",
                alpha=0.85, markersize=4)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(title)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def load_surfaces_for(pattern: str) -> dict:
    d = {}
    for p in sorted((V21 / 'surfaces').glob(pattern)):
        sd = load_surface(p)
        key = p.stem
        d[key] = (sd['p_surface'], sd['y_surface'], sd['horizons'])
    return d


def plot_anomaly_cross_dataset(save_path: Path):
    """One curve per dataset: seed-42 Mahal surfaces."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=120)
    for name in ['SMAP', 'MSL', 'PSM', 'SMD', 'MBA']:
        path = V21 / 'surfaces' / f'{name.lower()}_seed42_mahal.npz'
        if not path.exists():
            continue
        sd = load_surface(path)
        res = auprc_per_horizon(sd['p_surface'], sd['y_surface'],
                                horizon_labels=list(sd['horizons']))
        ax.plot(sd['horizons'], res['auprc_per_k'], 'o-', label=name,
                alpha=0.85, markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel(r'Horizon $\Delta t$ (steps)')
    ax.set_ylabel('AUPRC')
    ax.set_title('Per-horizon AUPRC across anomaly datasets (seed 42, Mahal surface)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cmapss_modes(save_path: Path):
    """Per-horizon AUPRC comparison across modes on FD001 (seed 42, 100% labels)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=120)
    for mode in ['probe_h', 'pred_ft', 'e2e', 'scratch']:
        path = V21 / 'surfaces' / f'fd001_{mode}_b100_seed42.npz'
        if not path.exists():
            continue
        sd = load_surface(path)
        res = auprc_per_horizon(sd['p_surface'], sd['y_surface'],
                                horizon_labels=list(sd['horizons']))
        ax.plot(sd['horizons'], res['auprc_per_k'], 'o-', label=mode,
                alpha=0.85, markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel(r'Horizon $\Delta t$ (cycles)')
    ax.set_ylabel('AUPRC')
    ax.set_title('FD001 per-horizon AUPRC across finetuning modes (seed 42, 100%)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def surface_heatmap(surface_path: Path, save_path: Path, title: str):
    sd = load_surface(surface_path)
    p = sd['p_surface']; y = sd['y_surface']
    horizons = sd['horizons']
    # Sort rows by ground-truth tte (approx: use y to find first Δt where y=1)
    y_first = np.where(y.any(axis=1), y.argmax(axis=1), len(horizons))
    order = np.argsort(y_first)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=120, sharey=True)
    im0 = axes[0].imshow(p[order], aspect='auto', cmap='viridis',
                         vmin=0, vmax=1, interpolation='nearest')
    axes[0].set_title(f'{title} — predicted $p(t, \\Delta t)$')
    axes[0].set_xlabel(r'Horizon index')
    axes[0].set_ylabel('Observation (sorted by first-1 index)')
    axes[0].set_xticks(range(len(horizons))); axes[0].set_xticklabels(horizons, fontsize=7)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(y[order], aspect='auto', cmap='binary',
                         vmin=0, vmax=1, interpolation='nearest')
    axes[1].set_title('Ground truth $y(t, \\Delta t)$')
    axes[1].set_xlabel(r'Horizon index')
    axes[1].set_xticks(range(len(horizons))); axes[1].set_xticklabels(horizons, fontsize=7)
    fig.colorbar(im1, ax=axes[1])
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main():
    # Per-horizon AUPRC across anomaly datasets
    plot_anomaly_cross_dataset(FIG_DIR / 'auprc_per_horizon_anomaly.pdf')

    # Per-horizon AUPRC across C-MAPSS FD001 modes
    plot_cmapss_modes(FIG_DIR / 'auprc_per_horizon_fd001_modes.pdf')

    # Reliability diagrams
    fd001_surf = load_surfaces_for('fd001_pred_ft_b100_seed*.npz')
    plot_reliability(fd001_surf, 'FD001 pred-FT reliability (3 seeds)',
                     FIG_DIR / 'reliability_fd001.pdf')
    smap_surf = load_surfaces_for('smap_seed*_mahal.npz')
    plot_reliability(smap_surf, 'SMAP Mahal reliability (3 seeds)',
                     FIG_DIR / 'reliability_smap.pdf')

    # Heatmaps for FD001 pred-FT seed 42 + SMAP seed 42
    p = V21 / 'surfaces' / 'fd001_pred_ft_b100_seed42.npz'
    if p.exists():
        surface_heatmap(p, FIG_DIR / 'heatmap_fd001_pred_ft.pdf',
                        'C-MAPSS FD001 pred-FT')
    p = V21 / 'surfaces' / 'smap_seed42_mahal.npz'
    if p.exists():
        surface_heatmap(p, FIG_DIR / 'heatmap_smap_mahal.pdf',
                        'SMAP Mahalanobis-calibrated')

    # Print summary
    print(f'Wrote figures to {FIG_DIR}')
    for p in sorted(FIG_DIR.glob('*.pdf')):
        print(f'  {p.name}')


if __name__ == '__main__':
    main()

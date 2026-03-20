#!/usr/bin/env python
"""
Distribution Analysis for Cross-Machine Transfer.

Computes MMD and Wasserstein distances between AURSAD and Voraus datasets
to assess transferability before training.

Output:
- Distance tables (CSV)
- Distribution plots (PNG)
- Transferability assessment
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_mmd(X: np.ndarray, Y: np.ndarray, gamma: float = None) -> float:
    """
    Compute Maximum Mean Discrepancy between two samples.

    Args:
        X: Samples from distribution P (n_samples, n_features)
        Y: Samples from distribution Q (m_samples, n_features)
        gamma: RBF kernel bandwidth (None = median heuristic)

    Returns:
        MMD² estimate
    """
    if gamma is None:
        # Median heuristic
        XY = np.vstack([X, Y])
        dists = np.sqrt(((XY[:, None] - XY[None, :]) ** 2).sum(axis=-1))
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2)

    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)

    return XX.mean() + YY.mean() - 2 * XY.mean()


def compute_per_channel_distances(data_a: np.ndarray, data_b: np.ndarray,
                                   channel_names: list) -> pd.DataFrame:
    """Compute MMD and Wasserstein for each channel."""
    results = []

    for i, name in enumerate(channel_names):
        if i >= data_a.shape[1] or i >= data_b.shape[1]:
            continue

        x = data_a[:, i].flatten()
        y = data_b[:, i].flatten()

        # Subsample for efficiency
        n_samples = min(10000, len(x), len(y))
        x_sub = np.random.choice(x, n_samples, replace=False)
        y_sub = np.random.choice(y, n_samples, replace=False)

        mmd = compute_mmd(x_sub.reshape(-1, 1), y_sub.reshape(-1, 1))
        wass = wasserstein_distance(x_sub, y_sub)

        results.append({
            'channel': name,
            'mmd': mmd,
            'wasserstein': wass,
            'mean_diff': abs(x.mean() - y.mean()),
            'std_ratio': x.std() / (y.std() + 1e-8),
        })

    return pd.DataFrame(results)


def plot_distributions(data_a: np.ndarray, data_b: np.ndarray,
                       channel_names: list, name_a: str, name_b: str):
    """Plot distribution comparisons."""
    n_channels = min(6, len(channel_names), data_a.shape[1], data_b.shape[1])

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(n_channels):
        ax = axes[i]
        x = data_a[:, i].flatten()
        y = data_b[:, i].flatten()

        # Subsample for plotting
        n = min(5000, len(x), len(y))
        x_sub = np.random.choice(x, n, replace=False)
        y_sub = np.random.choice(y, n, replace=False)

        ax.hist(x_sub, bins=50, alpha=0.5, label=name_a, density=True)
        ax.hist(y_sub, bins=50, alpha=0.5, label=name_b, density=True)
        ax.set_title(channel_names[i])
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "distribution_comparison.png", dpi=150)
    plt.close()
    logger.info(f"Saved distribution plot to {OUTPUT_DIR / 'distribution_comparison.png'}")


def main():
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    logger.info("Loading AURSAD dataset...")
    config_a = FactoryNetConfig(data_source='aursad', max_episodes=200, window_size=256, stride=256)
    ds_a = FactoryNetDataset(config_a, split='train')

    logger.info("Loading Voraus dataset...")
    config_b = FactoryNetConfig(data_source='voraus', max_episodes=200, window_size=256, stride=256)
    ds_b = FactoryNetDataset(config_b, split='train')

    # Extract data
    logger.info("Extracting signal data...")
    n_samples = min(500, len(ds_a), len(ds_b))

    setpoint_a = np.stack([ds_a[i]['setpoint'].numpy() for i in range(n_samples)])
    effort_a = np.stack([ds_a[i]['effort'].numpy() for i in range(n_samples)])

    setpoint_b = np.stack([ds_b[i]['setpoint'].numpy() for i in range(n_samples)])
    effort_b = np.stack([ds_b[i]['effort'].numpy() for i in range(n_samples)])

    # Flatten time dimension
    setpoint_a = setpoint_a.reshape(-1, setpoint_a.shape[-1])
    effort_a = effort_a.reshape(-1, effort_a.shape[-1])
    setpoint_b = setpoint_b.reshape(-1, setpoint_b.shape[-1])
    effort_b = effort_b.reshape(-1, effort_b.shape[-1])

    # Channel names
    setpoint_names = [f'setpoint_{i}' for i in range(setpoint_a.shape[1])]
    effort_names = [f'effort_{i}' for i in range(effort_a.shape[1])]

    # Compute distances
    logger.info("Computing setpoint distances...")
    setpoint_dist = compute_per_channel_distances(setpoint_a, setpoint_b, setpoint_names)

    logger.info("Computing effort distances...")
    effort_dist = compute_per_channel_distances(effort_a, effort_b, effort_names)

    # Save results
    setpoint_dist.to_csv(OUTPUT_DIR / "setpoint_distances.csv", index=False)
    effort_dist.to_csv(OUTPUT_DIR / "effort_distances.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("SETPOINT SIGNAL DISTANCES (AURSAD vs Voraus)")
    print("="*60)
    print(setpoint_dist.to_string(index=False))

    print("\n" + "="*60)
    print("EFFORT SIGNAL DISTANCES (AURSAD vs Voraus)")
    print("="*60)
    print(effort_dist.to_string(index=False))

    # Transferability assessment
    avg_mmd = (setpoint_dist['mmd'].mean() + effort_dist['mmd'].mean()) / 2
    avg_wass = (setpoint_dist['wasserstein'].mean() + effort_dist['wasserstein'].mean()) / 2

    print("\n" + "="*60)
    print("TRANSFERABILITY ASSESSMENT")
    print("="*60)
    print(f"Average MMD: {avg_mmd:.4f}")
    print(f"Average Wasserstein: {avg_wass:.4f}")

    if avg_mmd < 0.1:
        print("Assessment: HIGH transferability (distributions very similar)")
    elif avg_mmd < 0.5:
        print("Assessment: MODERATE transferability (some distribution shift)")
    else:
        print("Assessment: LOW transferability (significant distribution shift)")

    # Plot
    logger.info("Generating plots...")
    plot_distributions(setpoint_a, setpoint_b, setpoint_names, 'AURSAD', 'Voraus')

    logger.info("Done!")


if __name__ == "__main__":
    main()

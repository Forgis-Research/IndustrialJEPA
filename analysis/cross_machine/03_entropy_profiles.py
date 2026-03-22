#!/usr/bin/env python
"""
Entropy Profile Analysis for Cross-Machine Transfer.

Computes permutation entropy and complexity measures for each signal
to assess transferability (TIMETIC-style analysis).

Output:
- Entropy profiles (CSV)
- Complexity comparison plots (PNG)
"""

import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import permutations
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """
    Compute permutation entropy of a time series.

    Args:
        x: 1D time series
        order: Embedding dimension (pattern length)
        delay: Time delay between elements

    Returns:
        Normalized permutation entropy (0-1)
    """
    n = len(x)
    if n < order * delay:
        return np.nan

    # Extract ordinal patterns
    patterns = []
    for i in range(n - (order - 1) * delay):
        pattern = tuple(np.argsort(x[i:i + order * delay:delay]))
        patterns.append(pattern)

    # Count pattern frequencies
    counter = Counter(patterns)
    total = len(patterns)

    # Compute entropy
    probs = np.array(list(counter.values())) / total
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    # Normalize by maximum entropy
    max_entropy = np.log2(math.factorial(order))
    return entropy / max_entropy


def sample_entropy(x: np.ndarray, m: int = 2, r: float = None) -> float:
    """
    Compute sample entropy (measure of complexity/regularity).

    Args:
        x: 1D time series
        m: Embedding dimension
        r: Tolerance (default: 0.2 * std)

    Returns:
        Sample entropy value
    """
    n = len(x)
    if r is None:
        r = 0.2 * np.std(x)

    def _count_matches(templates, r):
        count = 0
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count

    # Create templates of length m
    templates_m = np.array([x[i:i + m] for i in range(n - m)])
    templates_m1 = np.array([x[i:i + m + 1] for i in range(n - m - 1)])

    # Count matches
    B = _count_matches(templates_m, r)
    A = _count_matches(templates_m1, r)

    if B == 0 or A == 0:
        return np.nan

    return -np.log(A / B)


def compute_entropy_profile(data: np.ndarray, channel_names: list) -> pd.DataFrame:
    """Compute entropy metrics for each channel."""
    results = []

    for i, name in enumerate(channel_names):
        if i >= data.shape[-1]:
            continue

        # Flatten to 1D (concatenate all samples)
        x = data[:, :, i].flatten()

        # Subsample for efficiency
        if len(x) > 50000:
            x = x[::len(x) // 50000]

        pe = permutation_entropy(x, order=3, delay=1)

        results.append({
            'channel': name,
            'permutation_entropy': pe,
            'std': np.std(x),
            'range': np.ptp(x),
            'mean_abs': np.mean(np.abs(x)),
        })

    return pd.DataFrame(results)


def plot_entropy_comparison(profile_a: pd.DataFrame, profile_b: pd.DataFrame,
                            name_a: str, name_b: str, signal_type: str):
    """Plot entropy profile comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot of permutation entropy
    n = min(len(profile_a), len(profile_b))
    x = np.arange(n)
    width = 0.35

    ax = axes[0]
    ax.bar(x - width/2, profile_a['permutation_entropy'].values[:n], width, label=name_a, alpha=0.8)
    ax.bar(x + width/2, profile_b['permutation_entropy'].values[:n], width, label=name_b, alpha=0.8)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Permutation Entropy')
    ax.set_title(f'{signal_type} - Permutation Entropy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ch{i}' for i in range(n)], rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)

    # Scatter plot comparing entropies
    ax = axes[1]
    ax.scatter(profile_a['permutation_entropy'].values[:n],
               profile_b['permutation_entropy'].values[:n], s=100, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel(f'{name_a} Entropy')
    ax.set_ylabel(f'{name_b} Entropy')
    ax.set_title('Entropy Correlation')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"entropy_{signal_type.lower()}.png", dpi=150)
    plt.close()
    logger.info(f"Saved entropy plot for {signal_type}")


def main():
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    logger.info("Loading AURSAD dataset...")
    config_a = FactoryNetConfig(data_source='aursad', max_episodes=100, window_size=256, stride=256)
    ds_a = FactoryNetDataset(config_a, split='train')

    logger.info("Loading Voraus dataset...")
    config_b = FactoryNetConfig(data_source='voraus', max_episodes=100, window_size=256, stride=256)
    ds_b = FactoryNetDataset(config_b, split='train')

    # Extract data
    logger.info("Extracting signal data...")
    n_samples = min(200, len(ds_a), len(ds_b))

    # Dataset returns (setpoint, effort, metadata) tuple
    setpoint_a = np.stack([ds_a[i][0].numpy() for i in range(n_samples)])
    effort_a = np.stack([ds_a[i][1].numpy() for i in range(n_samples)])

    setpoint_b = np.stack([ds_b[i][0].numpy() for i in range(n_samples)])
    effort_b = np.stack([ds_b[i][1].numpy() for i in range(n_samples)])

    # Channel names
    setpoint_names = [f'setpoint_{i}' for i in range(setpoint_a.shape[-1])]
    effort_names = [f'effort_{i}' for i in range(effort_a.shape[-1])]

    # Compute entropy profiles
    logger.info("Computing entropy profiles (this may take a minute)...")

    setpoint_entropy_a = compute_entropy_profile(setpoint_a, setpoint_names)
    setpoint_entropy_b = compute_entropy_profile(setpoint_b, setpoint_names)

    effort_entropy_a = compute_entropy_profile(effort_a, effort_names)
    effort_entropy_b = compute_entropy_profile(effort_b, effort_names)

    # Save results
    setpoint_entropy_a.to_csv(OUTPUT_DIR / "entropy_setpoint_aursad.csv", index=False)
    setpoint_entropy_b.to_csv(OUTPUT_DIR / "entropy_setpoint_voraus.csv", index=False)
    effort_entropy_a.to_csv(OUTPUT_DIR / "entropy_effort_aursad.csv", index=False)
    effort_entropy_b.to_csv(OUTPUT_DIR / "entropy_effort_voraus.csv", index=False)

    # Print summary
    print("\n" + "="*60)
    print("ENTROPY PROFILE - AURSAD")
    print("="*60)
    print("\nSetpoint:")
    print(setpoint_entropy_a.to_string(index=False))
    print("\nEffort:")
    print(effort_entropy_a.to_string(index=False))

    print("\n" + "="*60)
    print("ENTROPY PROFILE - VORAUS")
    print("="*60)
    print("\nSetpoint:")
    print(setpoint_entropy_b.to_string(index=False))
    print("\nEffort:")
    print(effort_entropy_b.to_string(index=False))

    # Compare
    print("\n" + "="*60)
    print("ENTROPY COMPARISON")
    print("="*60)

    n = min(len(setpoint_entropy_a), len(setpoint_entropy_b))
    setpoint_corr = np.corrcoef(
        setpoint_entropy_a['permutation_entropy'].values[:n],
        setpoint_entropy_b['permutation_entropy'].values[:n]
    )[0, 1]

    n = min(len(effort_entropy_a), len(effort_entropy_b))
    effort_corr = np.corrcoef(
        effort_entropy_a['permutation_entropy'].values[:n],
        effort_entropy_b['permutation_entropy'].values[:n]
    )[0, 1]

    print(f"Setpoint entropy correlation: {setpoint_corr:.4f}")
    print(f"Effort entropy correlation: {effort_corr:.4f}")

    avg_corr = (setpoint_corr + effort_corr) / 2
    if avg_corr > 0.7:
        print("\nAssessment: HIGH complexity similarity - transfer likely")
    elif avg_corr > 0.3:
        print("\nAssessment: MODERATE complexity similarity - transfer possible")
    else:
        print("\nAssessment: LOW complexity similarity - transfer challenging")

    # Plot
    logger.info("Generating plots...")
    plot_entropy_comparison(setpoint_entropy_a, setpoint_entropy_b, 'AURSAD', 'Voraus', 'Setpoint')
    plot_entropy_comparison(effort_entropy_a, effort_entropy_b, 'AURSAD', 'Voraus', 'Effort')

    logger.info("Done!")


if __name__ == "__main__":
    main()

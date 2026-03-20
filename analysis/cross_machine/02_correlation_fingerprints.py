#!/usr/bin/env python
"""
Correlation Fingerprint Analysis for Cross-Machine Transfer.

Computes and compares the correlation structure between sensors
for AURSAD and Voraus datasets.

Output:
- Correlation heatmaps (PNG)
- Fingerprint similarity score
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_correlation_matrix(data: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix across all channels.

    Args:
        data: (n_samples, time, channels)

    Returns:
        Correlation matrix (channels, channels)
    """
    # Flatten samples and time
    flat = data.reshape(-1, data.shape[-1])
    return np.corrcoef(flat.T)


def fingerprint_similarity(corr_a: np.ndarray, corr_b: np.ndarray) -> dict:
    """
    Compare two correlation fingerprints.

    Returns multiple similarity metrics.
    """
    # Ensure same size (use smaller)
    n = min(corr_a.shape[0], corr_b.shape[0])
    a = corr_a[:n, :n]
    b = corr_b[:n, :n]

    # Frobenius norm of difference
    frobenius = np.linalg.norm(a - b, 'fro')

    # Normalized Frobenius (0-1 scale)
    max_frob = np.sqrt(2 * n * n)  # Max possible for correlation matrices
    frobenius_norm = frobenius / max_frob

    # Cosine similarity of flattened matrices
    a_flat = a.flatten()
    b_flat = b.flatten()
    cosine = np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat))

    # Mean absolute element-wise difference
    mae = np.abs(a - b).mean()

    return {
        'frobenius': frobenius,
        'frobenius_normalized': frobenius_norm,
        'cosine_similarity': cosine,
        'mae': mae,
    }


def plot_correlation_heatmaps(corr_a: np.ndarray, corr_b: np.ndarray,
                               name_a: str, name_b: str, signal_type: str):
    """Plot side-by-side correlation heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # AURSAD
    im1 = axes[0].imshow(corr_a, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0].set_title(f'{name_a} - {signal_type}')
    plt.colorbar(im1, ax=axes[0])

    # Voraus
    n = min(corr_a.shape[0], corr_b.shape[0])
    im2 = axes[1].imshow(corr_b[:n, :n], cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1].set_title(f'{name_b} - {signal_type}')
    plt.colorbar(im2, ax=axes[1])

    # Difference
    diff = corr_a[:n, :n] - corr_b[:n, :n]
    im3 = axes[2].imshow(diff, cmap='RdBu_r', vmin=-2, vmax=2)
    axes[2].set_title(f'Difference ({name_a} - {name_b})')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"correlation_{signal_type.lower()}.png", dpi=150)
    plt.close()
    logger.info(f"Saved correlation plot for {signal_type}")


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

    # Compute correlation matrices
    logger.info("Computing correlation matrices...")
    corr_setpoint_a = compute_correlation_matrix(setpoint_a)
    corr_setpoint_b = compute_correlation_matrix(setpoint_b)
    corr_effort_a = compute_correlation_matrix(effort_a)
    corr_effort_b = compute_correlation_matrix(effort_b)

    # Compute combined (setpoint + effort)
    combined_a = np.concatenate([setpoint_a, effort_a], axis=-1)
    combined_b = np.concatenate([setpoint_b, effort_b], axis=-1)
    corr_combined_a = compute_correlation_matrix(combined_a)
    corr_combined_b = compute_correlation_matrix(combined_b)

    # Compare fingerprints
    print("\n" + "="*60)
    print("CORRELATION FINGERPRINT SIMILARITY")
    print("="*60)

    setpoint_sim = fingerprint_similarity(corr_setpoint_a, corr_setpoint_b)
    print(f"\nSetpoint signals:")
    for k, v in setpoint_sim.items():
        print(f"  {k}: {v:.4f}")

    effort_sim = fingerprint_similarity(corr_effort_a, corr_effort_b)
    print(f"\nEffort signals:")
    for k, v in effort_sim.items():
        print(f"  {k}: {v:.4f}")

    combined_sim = fingerprint_similarity(corr_combined_a, corr_combined_b)
    print(f"\nCombined (setpoint + effort):")
    for k, v in combined_sim.items():
        print(f"  {k}: {v:.4f}")

    # Assessment
    print("\n" + "="*60)
    print("FINGERPRINT ASSESSMENT")
    print("="*60)
    cosine_avg = (setpoint_sim['cosine_similarity'] + effort_sim['cosine_similarity']) / 2
    print(f"Average cosine similarity: {cosine_avg:.4f}")

    if cosine_avg > 0.8:
        print("Assessment: HIGH structural similarity - transfer very likely")
    elif cosine_avg > 0.5:
        print("Assessment: MODERATE structural similarity - transfer possible")
    else:
        print("Assessment: LOW structural similarity - transfer challenging")

    # Plot
    logger.info("Generating plots...")
    plot_correlation_heatmaps(corr_setpoint_a, corr_setpoint_b, 'AURSAD', 'Voraus', 'Setpoint')
    plot_correlation_heatmaps(corr_effort_a, corr_effort_b, 'AURSAD', 'Voraus', 'Effort')
    plot_correlation_heatmaps(corr_combined_a, corr_combined_b, 'AURSAD', 'Voraus', 'Combined')

    logger.info("Done!")


if __name__ == "__main__":
    main()

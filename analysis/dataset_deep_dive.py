#!/usr/bin/env python
"""
Deep Dive: Setpoint & Effort Feature Analysis for JEPA Transfer Learning.

Generates concise summary figures to understand:
1. Setpoint-Effort relationships (the dynamics JEPA learns)
2. Cross-machine signal structure similarities
3. Indicators for transferable representations

Output: autoresearch/figures/dataset_deep_dive_*.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "autoresearch" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets(n_episodes=100):
    """Load AURSAD and Voraus datasets."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    print("Loading AURSAD...")
    config_a = FactoryNetConfig(
        data_source='aursad',
        max_episodes=n_episodes,
        window_size=256,
        stride=256,
        norm_mode='episode'
    )
    ds_a = FactoryNetDataset(config_a, split='train')

    print("Loading Voraus...")
    config_b = FactoryNetConfig(
        data_source='voraus',
        max_episodes=n_episodes,
        window_size=256,
        stride=256,
        norm_mode='episode'
    )
    ds_b = FactoryNetDataset(config_b, split='train')

    return ds_a, ds_b


def extract_windows(dataset, n_windows=200):
    """Extract setpoint and effort windows from dataset."""
    n = min(n_windows, len(dataset))
    setpoints = []
    efforts = []

    for i in range(n):
        setpoint, effort, metadata = dataset[i]
        setpoints.append(setpoint.numpy())
        efforts.append(effort.numpy())

    return np.stack(setpoints), np.stack(efforts)


def compute_xcorr_lag(setpoint, effort, max_lag=50):
    """Compute cross-correlation and find optimal lag between setpoint and effort."""
    # Use first channel of each
    s = setpoint[:, 0] if setpoint.ndim > 1 else setpoint
    e = effort[:, 0] if effort.ndim > 1 else effort

    # Normalize
    s = (s - s.mean()) / (s.std() + 1e-8)
    e = (e - e.mean()) / (e.std() + 1e-8)

    # Cross-correlate
    xcorr = correlate(e, s, mode='full')
    lags = np.arange(-len(s) + 1, len(s))

    # Focus on relevant lags
    mask = (lags >= -max_lag) & (lags <= max_lag)
    xcorr_subset = xcorr[mask]
    lags_subset = lags[mask]

    # Find peak
    peak_idx = np.argmax(np.abs(xcorr_subset))
    optimal_lag = lags_subset[peak_idx]
    peak_corr = xcorr_subset[peak_idx] / len(s)

    return optimal_lag, peak_corr, lags_subset, xcorr_subset / len(s)


def fig1_setpoint_effort_dynamics(ds_a, ds_b):
    """
    Figure 1: Setpoint->Effort Causal Dynamics
    Shows how effort responds to setpoint changes with temporal lag.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Get sample windows
    sp_a, ef_a = extract_windows(ds_a, 100)
    sp_b, ef_b = extract_windows(ds_b, 100)

    # Row 1: AURSAD
    # Time series example
    ax = axes[0, 0]
    idx = np.random.randint(len(sp_a))
    t = np.arange(256)
    ax.plot(t, sp_a[idx, :, 0], 'b-', alpha=0.8, label='Setpoint[0]')
    ax.plot(t, ef_a[idx, :, 0], 'r-', alpha=0.8, label='Effort[0]')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Normalized value')
    ax.set_title('AURSAD: Setpoint vs Effort')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cross-correlation analysis
    ax = axes[0, 1]
    lags_all = []
    corrs_all = []
    for i in range(min(50, len(sp_a))):
        lag, corr, lags, xcorr = compute_xcorr_lag(sp_a[i], ef_a[i])
        lags_all.append(lag)
        corrs_all.append(corr)
        if i < 10:
            ax.plot(lags, xcorr, alpha=0.3, color='blue')
    ax.axvline(x=np.median(lags_all), color='red', linestyle='--', label=f'Median lag={np.median(lags_all):.0f}')
    ax.set_xlabel('Lag (steps)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('AURSAD: Setpoint->Effort X-Corr')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Lag distribution
    ax = axes[0, 2]
    ax.hist(lags_all, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=np.median(lags_all), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Optimal lag (steps)')
    ax.set_ylabel('Count')
    ax.set_title(f'AURSAD: Lag Distribution\n(median={np.median(lags_all):.1f})')
    ax.grid(True, alpha=0.3)

    # Row 2: Voraus
    ax = axes[1, 0]
    idx = np.random.randint(len(sp_b))
    ax.plot(t, sp_b[idx, :, 0], 'b-', alpha=0.8, label='Setpoint[0]')
    ax.plot(t, ef_b[idx, :, 0], 'r-', alpha=0.8, label='Effort[0]')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Normalized value')
    ax.set_title('Voraus: Setpoint vs Effort')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    lags_all_b = []
    corrs_all_b = []
    for i in range(min(50, len(sp_b))):
        lag, corr, lags, xcorr = compute_xcorr_lag(sp_b[i], ef_b[i])
        lags_all_b.append(lag)
        corrs_all_b.append(corr)
        if i < 10:
            ax.plot(lags, xcorr, alpha=0.3, color='green')
    ax.axvline(x=np.median(lags_all_b), color='red', linestyle='--', label=f'Median lag={np.median(lags_all_b):.0f}')
    ax.set_xlabel('Lag (steps)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title('Voraus: Setpoint->Effort X-Corr')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.hist(lags_all_b, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=np.median(lags_all_b), color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Optimal lag (steps)')
    ax.set_ylabel('Count')
    ax.set_title(f'Voraus: Lag Distribution\n(median={np.median(lags_all_b):.1f})')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Setpoint->Effort Causal Dynamics: Key for JEPA Transfer', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_dive_1_dynamics.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_dive_1_dynamics.png'}")

    return {
        'aursad_median_lag': np.median(lags_all),
        'voraus_median_lag': np.median(lags_all_b),
        'aursad_median_corr': np.median(corrs_all),
        'voraus_median_corr': np.median(corrs_all_b),
    }


def fig2_channel_correlations(ds_a, ds_b):
    """
    Figure 2: Inter-channel Correlation Structure
    Shows which setpoint channels affect which effort channels.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    sp_a, ef_a = extract_windows(ds_a, 100)
    sp_b, ef_b = extract_windows(ds_b, 100)

    # Flatten time for correlation
    sp_a_flat = sp_a.reshape(-1, sp_a.shape[-1])
    ef_a_flat = ef_a.reshape(-1, ef_a.shape[-1])
    sp_b_flat = sp_b.reshape(-1, sp_b.shape[-1])
    ef_b_flat = ef_b.reshape(-1, ef_b.shape[-1])

    # Compute setpoint-effort correlation matrices
    # AURSAD
    n_sp = min(6, sp_a_flat.shape[1])
    n_ef = min(6, ef_a_flat.shape[1])
    corr_a = np.zeros((n_sp, n_ef))
    for i in range(n_sp):
        for j in range(n_ef):
            corr_a[i, j] = np.corrcoef(sp_a_flat[:, i], ef_a_flat[:, j])[0, 1]

    ax = axes[0]
    im = ax.imshow(corr_a, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Effort channel')
    ax.set_ylabel('Setpoint channel')
    ax.set_title('AURSAD: Setpoint<->Effort Corr')
    ax.set_xticks(range(n_ef))
    ax.set_yticks(range(n_sp))
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Voraus
    n_sp_b = min(6, sp_b_flat.shape[1])
    n_ef_b = min(6, ef_b_flat.shape[1])
    corr_b = np.zeros((n_sp_b, n_ef_b))
    for i in range(n_sp_b):
        for j in range(n_ef_b):
            corr_b[i, j] = np.corrcoef(sp_b_flat[:, i], ef_b_flat[:, j])[0, 1]

    ax = axes[1]
    im = ax.imshow(corr_b, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Effort channel')
    ax.set_ylabel('Setpoint channel')
    ax.set_title('Voraus: Setpoint<->Effort Corr')
    ax.set_xticks(range(n_ef_b))
    ax.set_yticks(range(n_sp_b))
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Compare structures
    ax = axes[2]
    min_sp = min(n_sp, n_sp_b)
    min_ef = min(n_ef, n_ef_b)
    diff = corr_a[:min_sp, :min_ef] - corr_b[:min_sp, :min_ef]
    im = ax.imshow(diff, cmap='PuOr', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel('Effort channel')
    ax.set_ylabel('Setpoint channel')
    ax.set_title('Difference (AURSAD - Voraus)')
    ax.set_xticks(range(min_ef))
    ax.set_yticks(range(min_sp))
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Compute similarity
    mask = ~np.isnan(diff)
    struct_similarity = 1 - np.abs(diff[mask]).mean()

    plt.suptitle(f'Channel Correlation Structure (Similarity: {struct_similarity:.2f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_dive_2_correlations.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_dive_2_correlations.png'}")

    return {'structure_similarity': struct_similarity}


def fig3_effort_response_patterns(ds_a, ds_b):
    """
    Figure 3: Effort Response to Setpoint Changes
    Key insight: Does effort respond similarly to setpoint deltas?
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    sp_a, ef_a = extract_windows(ds_a, 150)
    sp_b, ef_b = extract_windows(ds_b, 150)

    # Compute deltas (changes)
    sp_delta_a = np.diff(sp_a, axis=1)
    ef_delta_a = np.diff(ef_a, axis=1)
    sp_delta_b = np.diff(sp_b, axis=1)
    ef_delta_b = np.diff(ef_b, axis=1)

    # Row 1: Delta relationships
    ax = axes[0, 0]
    for i in range(min(20, len(sp_delta_a))):
        ax.scatter(sp_delta_a[i, :, 0], ef_delta_a[i, :, 0], alpha=0.1, s=1, c='blue')
    ax.set_xlabel('Δ Setpoint[0]')
    ax.set_ylabel('Δ Effort[0]')
    ax.set_title('AURSAD: Response to Setpoint Changes')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax = axes[0, 1]
    for i in range(min(20, len(sp_delta_b))):
        ax.scatter(sp_delta_b[i, :, 0], ef_delta_b[i, :, 0], alpha=0.1, s=1, c='green')
    ax.set_xlabel('Δ Setpoint[0]')
    ax.set_ylabel('Δ Effort[0]')
    ax.set_title('Voraus: Response to Setpoint Changes')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Compare delta distributions
    ax = axes[0, 2]
    sp_d_a = sp_delta_a[:, :, 0].flatten()
    sp_d_b = sp_delta_b[:, :, 0].flatten()
    ax.hist(sp_d_a, bins=50, alpha=0.5, label='AURSAD', density=True, color='blue')
    ax.hist(sp_d_b, bins=50, alpha=0.5, label='Voraus', density=True, color='green')
    ax.set_xlabel('Δ Setpoint[0]')
    ax.set_ylabel('Density')
    ax.set_title('Setpoint Change Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Effort magnitude vs setpoint activity
    ax = axes[1, 0]
    sp_activity_a = np.std(sp_a, axis=1)[:, 0]  # Per-window setpoint variability
    ef_energy_a = np.mean(np.abs(ef_a), axis=1)[:, 0]  # Per-window effort magnitude
    ax.scatter(sp_activity_a, ef_energy_a, alpha=0.5, s=20, c='blue')
    ax.set_xlabel('Setpoint Activity (std)')
    ax.set_ylabel('Effort Magnitude (mean |e|)')
    ax.set_title('AURSAD: Activity vs Energy')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    sp_activity_b = np.std(sp_b, axis=1)[:, 0]
    ef_energy_b = np.mean(np.abs(ef_b), axis=1)[:, 0]
    ax.scatter(sp_activity_b, ef_energy_b, alpha=0.5, s=20, c='green')
    ax.set_xlabel('Setpoint Activity (std)')
    ax.set_ylabel('Effort Magnitude (mean |e|)')
    ax.set_title('Voraus: Activity vs Energy')
    ax.grid(True, alpha=0.3)

    # Correlation between activity and energy
    ax = axes[1, 2]
    corr_a = np.corrcoef(sp_activity_a, ef_energy_a)[0, 1]
    corr_b = np.corrcoef(sp_activity_b, ef_energy_b)[0, 1]
    bars = ax.bar(['AURSAD', 'Voraus'], [corr_a, corr_b], color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Correlation')
    ax.set_title('Activity<->Energy Correlation')
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, [corr_a, corr_b]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.2f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Effort Response Patterns: Physics-Based Transfer Indicators',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_dive_3_responses.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_dive_3_responses.png'}")

    return {
        'aursad_activity_energy_corr': corr_a,
        'voraus_activity_energy_corr': corr_b,
    }


def fig4_frequency_structure(ds_a, ds_b):
    """
    Figure 4: Frequency Domain Analysis
    Shows if the temporal structure (dynamics) is similar across machines.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sp_a, ef_a = extract_windows(ds_a, 100)
    sp_b, ef_b = extract_windows(ds_b, 100)

    # Compute average power spectrum
    def avg_spectrum(data):
        """Compute average power spectrum across samples."""
        specs = []
        for i in range(min(50, len(data))):
            fft = np.fft.fft(data[i, :, 0])
            spec = np.abs(fft[:len(fft)//2]) ** 2
            specs.append(spec)
        return np.mean(specs, axis=0)

    freqs = np.fft.fftfreq(256, d=1)[:128]

    # Setpoint spectra
    ax = axes[0, 0]
    spec_sp_a = avg_spectrum(sp_a)
    spec_sp_b = avg_spectrum(sp_b)
    ax.semilogy(freqs, spec_sp_a, 'b-', alpha=0.8, label='AURSAD')
    ax.semilogy(freqs, spec_sp_b, 'g-', alpha=0.8, label='Voraus')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power (log)')
    ax.set_title('Setpoint Power Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Effort spectra
    ax = axes[0, 1]
    spec_ef_a = avg_spectrum(ef_a)
    spec_ef_b = avg_spectrum(ef_b)
    ax.semilogy(freqs, spec_ef_a, 'b-', alpha=0.8, label='AURSAD')
    ax.semilogy(freqs, spec_ef_b, 'g-', alpha=0.8, label='Voraus')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power (log)')
    ax.set_title('Effort Power Spectrum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral similarity
    ax = axes[1, 0]
    # Normalize spectra for comparison
    spec_sp_a_norm = spec_sp_a / (spec_sp_a.sum() + 1e-8)
    spec_sp_b_norm = spec_sp_b / (spec_sp_b.sum() + 1e-8)
    spec_ef_a_norm = spec_ef_a / (spec_ef_a.sum() + 1e-8)
    spec_ef_b_norm = spec_ef_b / (spec_ef_b.sum() + 1e-8)

    # KL divergence (as similarity metric)
    def spectral_similarity(p, q):
        p = p + 1e-10
        q = q + 1e-10
        return 1.0 / (1.0 + stats.entropy(p, q))

    sim_sp = spectral_similarity(spec_sp_a_norm, spec_sp_b_norm)
    sim_ef = spectral_similarity(spec_ef_a_norm, spec_ef_b_norm)

    bars = ax.bar(['Setpoint', 'Effort'], [sim_sp, sim_ef], color=['purple', 'orange'], alpha=0.7)
    ax.set_ylabel('Spectral Similarity')
    ax.set_title('Cross-Machine Frequency Similarity')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, [sim_sp, sim_ef]):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.2f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Dominant frequency comparison
    ax = axes[1, 1]
    def get_dominant_freqs(spec, freqs, top_k=5):
        idx = np.argsort(spec)[-top_k:]
        return freqs[idx]

    dom_sp_a = get_dominant_freqs(spec_sp_a, freqs)
    dom_sp_b = get_dominant_freqs(spec_sp_b, freqs)
    dom_ef_a = get_dominant_freqs(spec_ef_a, freqs)
    dom_ef_b = get_dominant_freqs(spec_ef_b, freqs)

    x = np.arange(5)
    width = 0.35
    ax.bar(x - width/2, sorted(dom_sp_a), width, label='AURSAD Setpoint', alpha=0.7)
    ax.bar(x + width/2, sorted(dom_sp_b), width, label='Voraus Setpoint', alpha=0.7)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.set_title('Top-5 Dominant Frequencies')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Frequency Structure: Temporal Dynamics Transferability',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_dive_4_frequency.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_dive_4_frequency.png'}")

    return {
        'setpoint_spectral_similarity': sim_sp,
        'effort_spectral_similarity': sim_ef,
    }


def fig5_transfer_summary(metrics):
    """
    Figure 5: Transfer Learning Indicators Summary
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    indicators = [
        ('Lag Similarity', 1 - abs(metrics['aursad_median_lag'] - metrics['voraus_median_lag']) / 50),
        ('Correlation Structure', metrics['structure_similarity']),
        ('Activity-Energy Pattern',
         1 - abs(metrics['aursad_activity_energy_corr'] - metrics['voraus_activity_energy_corr'])),
        ('Setpoint Spectral Sim', metrics['setpoint_spectral_similarity']),
        ('Effort Spectral Sim', metrics['effort_spectral_similarity']),
    ]

    names = [i[0] for i in indicators]
    values = [max(0, min(1, i[1])) for i in indicators]

    colors = ['green' if v > 0.6 else 'orange' if v > 0.3 else 'red' for v in values]
    bars = ax.barh(names, values, color=colors, alpha=0.7, edgecolor='black')

    ax.set_xlim(0, 1)
    ax.set_xlabel('Similarity Score (higher = more transferable)')
    ax.set_title('JEPA Transfer Learning Indicators: AURSAD <-> Voraus', fontsize=14, fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold')

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                va='center', fontsize=11, fontweight='bold')

    # Overall score
    overall = np.mean(values)
    ax.text(0.5, -0.6, f'Overall Transferability Score: {overall:.2f}',
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_dive_5_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_dive_5_summary.png'}")

    return {'overall_transferability': overall}


def main():
    print("="*60)
    print("Dataset Deep Dive: Setpoint & Effort Analysis")
    print("="*60)

    # Load data
    ds_a, ds_b = load_datasets(n_episodes=150)
    print(f"Loaded {len(ds_a)} AURSAD windows, {len(ds_b)} Voraus windows")

    # Generate figures
    print("\nGenerating Figure 1: Setpoint->Effort Dynamics...")
    m1 = fig1_setpoint_effort_dynamics(ds_a, ds_b)

    print("\nGenerating Figure 2: Channel Correlations...")
    m2 = fig2_channel_correlations(ds_a, ds_b)

    print("\nGenerating Figure 3: Effort Response Patterns...")
    m3 = fig3_effort_response_patterns(ds_a, ds_b)

    print("\nGenerating Figure 4: Frequency Structure...")
    m4 = fig4_frequency_structure(ds_a, ds_b)

    # Combine metrics
    metrics = {**m1, **m2, **m3, **m4}

    print("\nGenerating Figure 5: Transfer Summary...")
    m5 = fig5_transfer_summary(metrics)
    metrics.update(m5)

    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nKey Findings:")
    print(f"  - AURSAD median setpoint->effort lag: {metrics['aursad_median_lag']:.1f} steps")
    print(f"  - Voraus median setpoint->effort lag: {metrics['voraus_median_lag']:.1f} steps")
    print(f"  - Correlation structure similarity: {metrics['structure_similarity']:.2f}")
    print(f"  - Setpoint spectral similarity: {metrics['setpoint_spectral_similarity']:.2f}")
    print(f"  - Effort spectral similarity: {metrics['effort_spectral_similarity']:.2f}")
    print(f"  - Overall transferability score: {metrics['overall_transferability']:.2f}")

    print(f"\nFigures saved to: {OUTPUT_DIR}")

    return metrics


if __name__ == "__main__":
    main()

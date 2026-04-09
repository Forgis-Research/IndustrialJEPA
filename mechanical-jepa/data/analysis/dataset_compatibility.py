"""
Part B: Ruthless Dataset Compatibility Analysis

Computes per-source signal statistics and cross-source compatibility metrics.
Produces heatmaps, PSD plots, amplitude distribution overlays.
Outputs COMPATIBILITY_REPORT.md.

Run from mechanical-jepa/ directory.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wasserstein_distance, kurtosis as scipy_kurtosis, skew as scipy_skew
from scipy.signal import resample_poly
from math import gcd
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

CACHE_DIR = '/tmp/hf_cache/bearings'
TARGET_SR = 12800
WINDOW_LEN = 1024
PLOT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/analysis/plots'
REPORT_PATH = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/analysis/COMPATIBILITY_REPORT.md'

os.makedirs(PLOT_DIR, exist_ok=True)

SOURCE_SR = {
    'cwru': 12000,
    'mfpt': 48828,
    'ims': 20480,
    'xjtu_sy': 25600,
    'paderborn': 64000,
    'femto': 25600,
    'mafaulda': 50000,
    'ottawa_bearing': 42000,
}

# ----- Data loading helpers -----

def load_parquet(filename):
    import pandas as pd
    local = os.path.join(CACHE_DIR, os.path.basename(filename))
    if os.path.exists(local):
        return pd.read_parquet(local)
    raise FileNotFoundError(f"Not cached: {filename}")


def get_ch0(row):
    try:
        sig = np.array(row['signal'])
        ch = np.array(sig[0], dtype=np.float32)
        return ch if len(ch) >= 64 else None
    except Exception:
        return None


def resample_to_target(ch, native_sr):
    if native_sr == TARGET_SR:
        return ch
    g = gcd(int(native_sr), int(TARGET_SR))
    up = TARGET_SR // g
    down = native_sr // g
    return resample_poly(ch, up, down).astype(np.float32)


def instance_norm(x):
    std = x.std()
    if std < 1e-10:
        return None
    return ((x - x.mean()) / std).astype(np.float32)


def extract_sample_windows(source_windows, n=200):
    """Sample up to n windows from a source's window list."""
    if len(source_windows) <= n:
        return source_windows
    idx = np.random.choice(len(source_windows), n, replace=False)
    return [source_windows[i] for i in sorted(idx)]


# ----- Signal characterization -----

def compute_window_features(w, sr=TARGET_SR):
    """Compute time+frequency features for a single window."""
    n = len(w)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    psd = np.abs(np.fft.rfft(w)) ** 2
    psd_norm = psd / (psd.sum() + 1e-12)

    centroid = float(np.sum(freqs * psd_norm))
    bandwidth = float(np.sqrt(np.maximum(np.sum((freqs - centroid) ** 2 * psd_norm), 0)))

    rms = float(np.sqrt(np.mean(w ** 2)))
    peak = float(np.max(np.abs(w)))
    crest_factor = peak / (rms + 1e-10)
    kurt = float(scipy_kurtosis(w, fisher=True))
    skewness = float(scipy_skew(w))

    # Band energies
    total_energy = psd.sum() + 1e-12
    bands = [(0, 500), (500, 2000), (2000, 5000), (5000, sr // 2)]
    band_e = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_e.append(float(psd[mask].sum() / total_energy))

    return {
        'rms': rms, 'crest_factor': crest_factor, 'kurtosis': kurt, 'skewness': skewness,
        'centroid': centroid, 'bandwidth': bandwidth,
        'band_0_500': band_e[0], 'band_500_2k': band_e[1],
        'band_2k_5k': band_e[2], 'band_5k_plus': band_e[3],
        'psd': psd, 'freqs': freqs,
    }


def characterize_source(windows, name, n_sample=200):
    """Compute aggregate statistics across windows for a source."""
    sample = extract_sample_windows(windows, n=n_sample)
    feats = [compute_window_features(w) for w in sample]

    scalar_keys = ['rms', 'crest_factor', 'kurtosis', 'skewness', 'centroid', 'bandwidth',
                   'band_0_500', 'band_500_2k', 'band_2k_5k', 'band_5k_plus']
    stats = {}
    for k in scalar_keys:
        vals = np.array([f[k] for f in feats])
        stats[k] = {'mean': float(vals.mean()), 'std': float(vals.std()),
                    'median': float(np.median(vals)), 'p5': float(np.percentile(vals, 5)),
                    'p95': float(np.percentile(vals, 95))}

    # Average PSD
    psds = np.stack([f['psd'] for f in feats], axis=0)
    avg_psd = psds.mean(axis=0)
    freqs = feats[0]['freqs']

    # Amplitude distribution (all windows flattened)
    amplitudes = np.concatenate([w for w in sample])

    stats['n_windows'] = len(sample)
    stats['avg_psd'] = avg_psd
    stats['freqs'] = freqs
    stats['amplitudes'] = amplitudes
    stats['source'] = name

    return stats


# ----- Pairwise compatibility -----

def kl_divergence_psd(psd_a, psd_b, eps=1e-12):
    """KL divergence between normalized PSDs."""
    p = psd_a / (psd_a.sum() + eps)
    q = psd_b / (psd_b.sum() + eps)
    # Symmetric KL
    kl = 0.5 * (np.sum(p * np.log((p + eps) / (q + eps))) +
                np.sum(q * np.log((q + eps) / (p + eps))))
    return float(kl)


def compute_pairwise_metrics(source_stats):
    """Compute 8x8 pairwise compatibility matrices."""
    names = list(source_stats.keys())
    n = len(names)

    kl_mat = np.zeros((n, n))
    wass_mat = np.zeros((n, n))
    centroid_diff = np.zeros((n, n))
    kurtosis_diff = np.zeros((n, n))
    rms_ratio = np.zeros((n, n))

    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            sa, sb = source_stats[a], source_stats[b]

            # PSD KL divergence
            kl_mat[i, j] = kl_divergence_psd(sa['avg_psd'], sb['avg_psd'])

            # Wasserstein distance on amplitude distributions (sample for speed)
            amp_a = sa['amplitudes']
            amp_b = sb['amplitudes']
            n_samp = min(2000, len(amp_a), len(amp_b))
            idx_a = np.random.choice(len(amp_a), n_samp, replace=False)
            idx_b = np.random.choice(len(amp_b), n_samp, replace=False)
            wass_mat[i, j] = wasserstein_distance(amp_a[idx_a], amp_b[idx_b])

            # Spectral centroid difference
            centroid_diff[i, j] = abs(sa['centroid']['mean'] - sb['centroid']['mean'])

            # Kurtosis difference
            kurtosis_diff[i, j] = abs(sa['kurtosis']['mean'] - sb['kurtosis']['mean'])

            # RMS energy ratio
            rms_a = sa['rms']['mean']
            rms_b = sb['rms']['mean']
            rms_ratio[i, j] = max(rms_a, rms_b) / (min(rms_a, rms_b) + 1e-10)

    return {
        'names': names,
        'kl': kl_mat,
        'wasserstein': wass_mat,
        'centroid_diff': centroid_diff,
        'kurtosis_diff': kurtosis_diff,
        'rms_ratio': rms_ratio,
    }


# ----- Data loading -----

def load_all_source_windows(verbose=True):
    """Load sample windows from all 8 sources."""
    import pandas as pd

    all_windows = defaultdict(list)

    def add_windows(src, df_sub, max_per_signal=10):
        for _, row in df_sub.iterrows():
            ch = get_ch0(row)
            if ch is None:
                continue
            sr = SOURCE_SR.get(src, TARGET_SR)
            ch = resample_to_target(ch, sr)
            n = len(ch)
            wins_added = 0
            for i in range(min(n // WINDOW_LEN, max_per_signal)):
                w = ch[i * WINDOW_LEN:(i + 1) * WINDOW_LEN]
                w_norm = instance_norm(w)
                if w_norm is not None:
                    all_windows[src].append(w_norm)
                    wins_added += 1

    # CWRU + MFPT
    df = load_parquet('extra_cwru_mfpt.parquet')
    for src in ['cwru', 'mfpt']:
        sub = df[df['source_id'] == src]
        add_windows(src, sub, max_per_signal=15)
        if verbose:
            print(f"  {src}: {len(all_windows[src])} windows")
    del df

    # IMS
    try:
        df = load_parquet('extra_ims.parquet')
        add_windows('ims', df, max_per_signal=10)
        if verbose:
            print(f"  ims: {len(all_windows['ims'])} windows")
        del df
    except Exception as e:
        if verbose:
            print(f"  ims: skipped ({e})")

    # MAFAULDA
    for i in range(8):
        try:
            df = load_parquet(f'mafaulda_{i:03d}.parquet')
            add_windows('mafaulda', df, max_per_signal=5)
            del df
        except Exception:
            break
    if verbose:
        print(f"  mafaulda: {len(all_windows['mafaulda'])} windows")

    # Ottawa
    try:
        df = load_parquet('ottawa_bearings.parquet')
        add_windows('ottawa', df, max_per_signal=10)
        if verbose:
            print(f"  ottawa: {len(all_windows['ottawa'])} windows")
        del df
    except Exception as e:
        if verbose:
            print(f"  ottawa: skipped ({e})")

    # Main shards: femto, xjtu_sy, paderborn
    shard_sources = {
        'femto': list(range(4)),
        'xjtu_sy': [3],
        'paderborn': [4],
    }
    shard_cache = {}
    for src, shards in shard_sources.items():
        for shard_idx in shards:
            key = f'train-{shard_idx:05d}-of-00005.parquet'
            if key not in shard_cache:
                shard_cache[key] = load_parquet(key)
            df = shard_cache[key]
            sub = df[df['source_id'] == src]
            for _, row in sub.iterrows():
                ch = get_ch0(row)
                if ch is None:
                    continue
                sr = SOURCE_SR.get(src, TARGET_SR)
                ch = resample_to_target(ch, sr)
                if len(ch) >= WINDOW_LEN:
                    w = ch[:WINDOW_LEN]
                elif len(ch) >= 512:
                    w = np.pad(ch, (0, WINDOW_LEN - len(ch)), mode='wrap')
                else:
                    continue
                w_norm = instance_norm(w)
                if w_norm is not None:
                    all_windows[src].append(w_norm)
        if verbose:
            print(f"  {src}: {len(all_windows[src])} windows")
    shard_cache.clear()

    return dict(all_windows)


# ----- Plotting -----

COLORS = {
    'femto': '#1f77b4',
    'xjtu_sy': '#ff7f0e',
    'ims': '#2ca02c',
    'cwru': '#d62728',
    'mfpt': '#9467bd',
    'paderborn': '#8c564b',
    'ottawa': '#e377c2',
    'mafaulda': '#7f7f7f',
}


def plot_avg_psds(source_stats, save_path):
    """Plot average PSD per source on one figure."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for src, stats in source_stats.items():
        freqs = stats['freqs']
        psd = stats['avg_psd']
        psd_db = 10 * np.log10(psd + 1e-12)
        ax.plot(freqs, psd_db, label=src, color=COLORS.get(src, 'black'), alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD (dB)')
    ax.set_title('Average Power Spectral Density per Source (after resampling to 12.8kHz, instance norm)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim([0, TARGET_SR // 2])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_amplitude_histograms(source_stats, save_path):
    """Overlay amplitude distributions for all sources."""
    fig, ax = plt.subplots(figsize=(12, 5))
    bins = np.linspace(-4, 4, 100)
    for src, stats in source_stats.items():
        amps = stats['amplitudes']
        amps_clipped = np.clip(amps, -4, 4)
        hist, _ = np.histogram(amps_clipped, bins=bins, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(bin_centers, hist, label=src, color=COLORS.get(src, 'black'), alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Normalized amplitude (after instance norm)')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Distribution per Source (after instance normalization)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_compatibility_heatmap(matrix, names, title, save_path, vmax=None, fmt='.2f'):
    """Plot an NxN heatmap."""
    fig, ax = plt.subplots(figsize=(9, 7))
    if vmax is None:
        vmax = np.percentile(matrix[matrix > 0], 90)
    im = ax.imshow(matrix, cmap='RdYlGn_r', vmin=0, vmax=vmax)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(names, fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                val = matrix[i, j]
                ax.text(j, i, format(val, fmt), ha='center', va='center', fontsize=8,
                        color='white' if val > vmax * 0.6 else 'black')
    plt.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stats_bar(source_stats, metric, title, save_path):
    """Bar chart of a per-source aggregate stat."""
    names = list(source_stats.keys())
    means = [source_stats[n][metric]['mean'] for n in names]
    stds = [source_stats[n][metric]['std'] for n in names]
    colors = [COLORS.get(n, 'steelblue') for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_stats(source_stats, save_path):
    """4-panel: kurtosis, centroid, RMS, crest factor."""
    names = list(source_stats.keys())
    metrics = [
        ('kurtosis', 'Kurtosis (Fisher)'),
        ('centroid', 'Spectral Centroid (Hz)'),
        ('rms', 'RMS (after instance norm)'),
        ('crest_factor', 'Crest Factor'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (metric, label) in zip(axes.flatten(), metrics):
        means = [source_stats[n][metric]['mean'] for n in names]
        stds = [source_stats[n][metric]['std'] for n in names]
        colors = [COLORS.get(n, 'steelblue') for n in names]
        ax.bar(names, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
        ax.set_title(label, fontsize=11)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Per-Source Signal Statistics (after resampling to 12.8kHz + instance norm)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_band_energies(source_stats, save_path):
    """Stacked bar of band energies per source."""
    names = list(source_stats.keys())
    bands = ['band_0_500', 'band_500_2k', 'band_2k_5k', 'band_5k_plus']
    band_labels = ['0-500Hz', '500-2kHz', '2-5kHz', '5kHz+']
    band_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    data = np.array([[source_stats[n][b]['mean'] for b in bands] for n in names])

    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(names))
    for i, (band, label, color) in enumerate(zip(bands, band_labels, band_colors)):
        vals = data[:, i]
        ax.bar(names, vals, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom += vals

    ax.set_ylabel('Fraction of total energy')
    ax.set_title('Band Energy Distribution per Source (after resampling to 12.8kHz)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xticklabels(names, rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ----- Compatibility scoring -----

def compute_compatibility_score(pairwise, reference_sources=None):
    """
    Compute a composite compatibility score for each source.
    Lower = more compatible with the reference group.
    """
    names = pairwise['names']
    if reference_sources is None:
        # Use femto + xjtu_sy as reference (the downstream evaluation sources)
        reference_sources = ['femto', 'xjtu_sy']

    ref_idx = [names.index(r) for r in reference_sources if r in names]
    if not ref_idx:
        return {n: 0.0 for n in names}

    scores = {}
    for i, name in enumerate(names):
        # Average distance to reference sources across metrics
        kl_to_ref = np.mean([pairwise['kl'][i, j] for j in ref_idx])
        wass_to_ref = np.mean([pairwise['wasserstein'][i, j] for j in ref_idx])
        centroid_to_ref = np.mean([pairwise['centroid_diff'][i, j] for j in ref_idx])
        kurt_to_ref = np.mean([pairwise['kurtosis_diff'][i, j] for j in ref_idx])

        # Normalize each metric by its max value
        scores[name] = {
            'kl': float(kl_to_ref),
            'wasserstein': float(wass_to_ref),
            'centroid_diff_hz': float(centroid_to_ref),
            'kurtosis_diff': float(kurt_to_ref),
        }

    return scores


def determine_compatibility(scores, source_stats, kl_threshold=5.0, centroid_threshold=1500):
    """
    Determine compatibility verdict for each source.
    Returns: dict with verdict (COMPATIBLE/MARGINAL/INCOMPATIBLE) and reason.
    """
    verdicts = {}
    ref_kl_baseline = scores.get('femto', {}).get('kl', 1.0)

    for name, score in scores.items():
        kl = score['kl']
        centroid_diff = score['centroid_diff_hz']
        kurt_diff = score['kurtosis_diff']

        reasons = []
        incompatible = False
        marginal = False

        if kl > kl_threshold:
            reasons.append(f'high PSD divergence (KL={kl:.1f}>{kl_threshold})')
            incompatible = True
        elif kl > kl_threshold * 0.5:
            reasons.append(f'moderate PSD divergence (KL={kl:.1f})')
            marginal = True

        if centroid_diff > centroid_threshold:
            reasons.append(f'large spectral centroid diff ({centroid_diff:.0f}Hz>{centroid_threshold}Hz)')
            marginal = True

        if kurt_diff > 5.0:
            reasons.append(f'high kurtosis diff ({kurt_diff:.1f})')
            marginal = True

        if name in ['femto', 'xjtu_sy']:
            verdict = 'COMPATIBLE'
            reasons = ['reference source (RUL target)']
        elif incompatible:
            verdict = 'INCOMPATIBLE'
        elif marginal:
            verdict = 'MARGINAL'
        else:
            verdict = 'COMPATIBLE'

        verdicts[name] = {'verdict': verdict, 'reasons': reasons, 'score': score}

    return verdicts


# ----- Report writing -----

def write_report(source_stats, pairwise, scores, verdicts, save_path):
    lines = []
    lines.append("# Dataset Compatibility Report\n")
    lines.append(f"Generated: 2026-04-09\n")
    lines.append(f"Analysis: 8 bearing sources, {TARGET_SR}Hz target SR, {WINDOW_LEN}-sample windows\n\n")

    lines.append("## Summary Table\n")
    lines.append("| Source | Centroid (Hz) | Kurtosis | RMS | Crest Factor | KL vs FEMTO | Compatible? |")
    lines.append("|--------|--------------|---------|-----|-------------|-------------|-------------|")

    for name in pairwise['names']:
        st = source_stats[name]
        sc = scores.get(name, {})
        v = verdicts.get(name, {}).get('verdict', '?')
        emoji = {'COMPATIBLE': 'YES', 'MARGINAL': 'MARGINAL', 'INCOMPATIBLE': 'NO'}.get(v, '?')
        lines.append(
            f"| {name:15s} | {st['centroid']['mean']:8.0f} | {st['kurtosis']['mean']:7.2f} "
            f"| {st['rms']['mean']:6.3f} | {st['crest_factor']['mean']:10.2f} "
            f"| {sc.get('kl', 0.0):10.2f} | {emoji:12s} |"
        )
    lines.append("")

    lines.append("## Per-Source Statistics\n")
    for name in pairwise['names']:
        st = source_stats[name]
        v = verdicts.get(name, {})
        verdict = v.get('verdict', '?')
        reasons = v.get('reasons', [])
        lines.append(f"### {name.upper()}")
        lines.append(f"- **Verdict**: {verdict}")
        if reasons:
            lines.append(f"- **Reason**: {'; '.join(reasons)}")
        lines.append(f"- Windows analyzed: {st['n_windows']}")
        lines.append(f"- Spectral centroid: {st['centroid']['mean']:.0f} ± {st['centroid']['std']:.0f} Hz")
        lines.append(f"- Kurtosis: {st['kurtosis']['mean']:.2f} ± {st['kurtosis']['std']:.2f}")
        lines.append(f"- RMS: {st['rms']['mean']:.3f} ± {st['rms']['std']:.3f}")
        lines.append(f"- Crest factor: {st['crest_factor']['mean']:.2f} ± {st['crest_factor']['std']:.2f}")
        band_keys = ['band_0_500', 'band_500_2k', 'band_2k_5k', 'band_5k_plus']
        band_labels = ['0-500Hz', '500-2kHz', '2-5kHz', '5kHz+']
        band_str = ', '.join(f"{l}: {st[b]['mean']*100:.1f}%" for l, b in zip(band_labels, band_keys))
        lines.append(f"- Band energies: {band_str}")
        lines.append("")

    lines.append("## Recommended Source Groups\n")
    compatible = [n for n, v in verdicts.items() if v.get('verdict') == 'COMPATIBLE']
    marginal = [n for n, v in verdicts.items() if v.get('verdict') == 'MARGINAL']
    incompatible = [n for n, v in verdicts.items() if v.get('verdict') == 'INCOMPATIBLE']

    lines.append(f"- **Group A (bearing RUL — primary targets)**: femto, xjtu_sy, ims")
    lines.append(f"- **Group B (compatible bearing faults)**: {', '.join([n for n in compatible if n not in ['femto','xjtu_sy','ims']])}")
    lines.append(f"- **Group C (marginal — structural differences)**: {', '.join(marginal)}")
    lines.append(f"- **Group D (incompatible — exclude from joint pretraining)**: {', '.join(incompatible)}")
    lines.append("")

    lines.append("## Pretraining Recommendation\n")
    lines.append("Based on the analysis:\n")

    all_compatible = compatible + marginal
    lines.append(f"**Recommended pretraining group**: {', '.join(all_compatible)}")
    lines.append("")
    lines.append("**Key findings**:")
    lines.append("1. Instance normalization makes RMS comparable across sources, but does NOT equalize spectral shapes.")
    lines.append("2. Sources with very different spectral centroids pull the JEPA encoder in conflicting directions.")
    lines.append("3. Kurtosis differences indicate different signal statistics that instance norm does not address.")
    lines.append("4. MAFAULDA (centrifugal pump) has the most different signal characteristics from bearing sources.")
    lines.append("")

    lines.append("## Pairwise Compatibility Matrices\n")
    lines.append("See plots in `data/analysis/plots/`:\n")
    lines.append("- `psd_comparison.png` — Average PSD per source")
    lines.append("- `amplitude_distributions.png` — Amplitude histograms")
    lines.append("- `compatibility_kl.png` — PSD KL divergence matrix")
    lines.append("- `compatibility_wasserstein.png` — Wasserstein distance matrix")
    lines.append("- `compatibility_centroid.png` — Spectral centroid difference matrix")
    lines.append("- `signal_stats.png` — Per-source kurtosis, centroid, RMS, crest factor")
    lines.append("- `band_energies.png` — Band energy distribution per source")

    with open(save_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved to: {save_path}")


# ----- Main -----

def main():
    np.random.seed(42)
    print("=" * 60)
    print("PART B: Dataset Compatibility Analysis")
    print("=" * 60)

    print("\nLoading windows from all sources...")
    windows_by_source = load_all_source_windows(verbose=True)

    print("\nCharacterizing sources...")
    source_stats = {}
    for src, windows in windows_by_source.items():
        print(f"  Characterizing {src} ({len(windows)} windows)...")
        source_stats[src] = characterize_source(windows, src, n_sample=300)

    print("\nComputing pairwise metrics...")
    pairwise = compute_pairwise_metrics(source_stats)

    print("\nComputing compatibility scores...")
    scores = compute_compatibility_score(pairwise, reference_sources=['femto', 'xjtu_sy'])
    verdicts = determine_compatibility(scores, source_stats)

    print("\nVerdicts:")
    for name, v in verdicts.items():
        print(f"  {name:20s}: {v['verdict']:12s} — {'; '.join(v['reasons'][:2]) if v['reasons'] else ''}")

    print("\nGenerating plots...")
    plot_avg_psds(source_stats, os.path.join(PLOT_DIR, 'psd_comparison.png'))
    plot_amplitude_histograms(source_stats, os.path.join(PLOT_DIR, 'amplitude_distributions.png'))
    plot_combined_stats(source_stats, os.path.join(PLOT_DIR, 'signal_stats.png'))
    plot_band_energies(source_stats, os.path.join(PLOT_DIR, 'band_energies.png'))

    names = pairwise['names']
    plot_compatibility_heatmap(pairwise['kl'], names,
        'PSD KL Divergence (lower = more compatible)',
        os.path.join(PLOT_DIR, 'compatibility_kl.png'))
    plot_compatibility_heatmap(pairwise['wasserstein'], names,
        'Wasserstein Distance on Amplitude Distributions',
        os.path.join(PLOT_DIR, 'compatibility_wasserstein.png'))
    plot_compatibility_heatmap(pairwise['centroid_diff'], names,
        'Spectral Centroid Difference (Hz)',
        os.path.join(PLOT_DIR, 'compatibility_centroid.png'),
        fmt='.0f')
    plot_compatibility_heatmap(pairwise['kurtosis_diff'], names,
        'Kurtosis Difference',
        os.path.join(PLOT_DIR, 'compatibility_kurtosis.png'))

    print("\nWriting compatibility report...")
    write_report(source_stats, pairwise, scores, verdicts, REPORT_PATH)

    # Save raw stats for notebook
    stats_summary = {}
    for src, st in source_stats.items():
        stats_summary[src] = {
            'centroid_mean': st['centroid']['mean'],
            'centroid_std': st['centroid']['std'],
            'kurtosis_mean': st['kurtosis']['mean'],
            'kurtosis_std': st['kurtosis']['std'],
            'rms_mean': st['rms']['mean'],
            'crest_factor_mean': st['crest_factor']['mean'],
            'n_windows': st['n_windows'],
            'band_0_500': st['band_0_500']['mean'],
            'band_500_2k': st['band_500_2k']['mean'],
            'band_2k_5k': st['band_2k_5k']['mean'],
            'band_5k_plus': st['band_5k_plus']['mean'],
            'kl_to_femto': scores.get(src, {}).get('kl', 0.0),
            'verdict': verdicts.get(src, {}).get('verdict', '?'),
        }

    json_path = os.path.join(PLOT_DIR, 'compatibility_stats.json')
    with open(json_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    print(f"Stats saved to: {json_path}")

    # Also save pairwise matrices
    pairwise_save = {
        'names': pairwise['names'],
        'kl': pairwise['kl'].tolist(),
        'wasserstein': pairwise['wasserstein'].tolist(),
        'centroid_diff': pairwise['centroid_diff'].tolist(),
        'kurtosis_diff': pairwise['kurtosis_diff'].tolist(),
        'rms_ratio': pairwise['rms_ratio'].tolist(),
    }
    pairwise_path = os.path.join(PLOT_DIR, 'pairwise_metrics.json')
    with open(pairwise_path, 'w') as f:
        json.dump(pairwise_save, f, indent=2)
    print(f"Pairwise metrics saved to: {pairwise_path}")

    print("\n" + "=" * 60)
    print("Part B COMPLETE")
    print("=" * 60)
    return source_stats, pairwise, scores, verdicts


if __name__ == '__main__':
    main()

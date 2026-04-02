"""
JEPA-based RUL Prognostics on IMS Dataset.
Exp 41: Zero-shot health indicator + RUL regression.

IMS files: 20480 samples x 8 channels, tab-delimited text.
Test1: 2156 files (35-day run), bearings 3&4 failed.
Test2: not available in this download.

Metrics:
1. Zero-shot health indicator: embedding distance from healthy centroid
   - Spearman correlation with time (no labels needed)
   - Early warning time before failure
2. RUL regression from JEPA embeddings vs RMS baseline
   - RMSE (lower better)
   - Spearman correlation (higher better)

Usage:
    python jepa_rul_ims.py --checkpoint checkpoints/jepa_v2_20260401_003619.pt
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2

IMS_DIR = Path('data/bearings/ims')
PLOTS_DIR = Path('notebooks/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Loading
# =============================================================================

def load_ims_file(fpath: Path, n_channels: int = 3, window_size: int = 4096) -> np.ndarray:
    """Load one IMS file (tab-delimited text, 20480 x 8), return (n_channels, window_size)."""
    data = np.loadtxt(str(fpath))  # (20480, 8)
    # Use first n_channels channels; take first window_size samples
    sig = data[:window_size, :n_channels].T.astype(np.float32)  # (n_channels, window_size)
    return sig


def load_ims_run(test_set: str = 'Test1', n_channels: int = 3, window_size: int = 4096,
                 subsample: int = 1):
    """
    Load all files from one IMS test set in temporal order.
    subsample: take every N-th file to reduce memory.
    Returns (signals_list, file_indices, rul_normalized, rms_per_file)
    """
    test_dir = IMS_DIR / test_set
    if not test_dir.exists():
        raise FileNotFoundError(f"IMS {test_set} not found at {test_dir}")

    all_files = sorted(test_dir.glob('*'))
    n_files = len(all_files)
    print(f"IMS {test_set}: {n_files} files")

    selected_files = all_files[::subsample]
    n_sel = len(selected_files)

    # Normalized RUL: file 0 = 1.0, last file = 0.0
    # Map selected file indices back to normalized RUL
    rul_all = np.linspace(1.0, 0.0, n_files)
    selected_indices = list(range(0, n_files, subsample))
    rul_normalized = rul_all[selected_indices]

    # Compute stats from first 25% of run (healthy period)
    n_healthy = max(1, n_files // 4)
    healthy_files = all_files[:min(n_healthy, 50)]  # use up to 50 files for stats
    print(f"Computing normalization stats from {len(healthy_files)} healthy files...")
    stats_signals = []
    for fpath in healthy_files:
        sig = load_ims_file(fpath, n_channels, window_size)
        stats_signals.append(sig.flatten())
    stats_arr = np.concatenate(stats_signals)
    mean_val = float(np.mean(stats_arr))
    std_val = float(np.std(stats_arr) + 1e-8)
    print(f"Normalization: mean={mean_val:.4f}, std={std_val:.4f}")

    print(f"Loading {n_sel} files (subsample={subsample})...")
    signals = []
    rms_values = []

    for fpath in selected_files:
        sig = load_ims_file(fpath, n_channels, window_size)
        rms = float(np.sqrt(np.mean(sig**2)))
        rms_values.append(rms)
        sig_norm = np.clip((sig - mean_val) / std_val, -5.0, 5.0)
        signals.append(sig_norm)

    return signals, selected_indices, rul_normalized, np.array(rms_values)


# =============================================================================
# Feature Extraction
# =============================================================================

@torch.no_grad()
def extract_jepa_embeddings(model, signals_list, device, batch_size=32):
    """
    Extract JEPA embeddings for a list of signals.
    Returns (N, embed_dim) array.
    """
    model.eval()
    all_embeds = []

    for i in range(0, len(signals_list), batch_size):
        batch = signals_list[i:i+batch_size]
        x = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)
        embeds = model.get_embeddings(x, pool='mean')  # (B, D)
        all_embeds.append(embeds.cpu().numpy())

    return np.concatenate(all_embeds, axis=0)


def compute_hand_features(signals_list):
    """Compute RMS + kurtosis + peak + crest factor per signal."""
    features = []
    for sig in signals_list:
        flat = sig.flatten()
        rms = float(np.sqrt(np.mean(flat**2)))
        kurt = float(stats.kurtosis(flat))
        peak = float(np.max(np.abs(flat)))
        crest = peak / (rms + 1e-8)
        # Spectral energy bands
        fft = np.abs(np.fft.rfft(flat[:4096] if len(flat) >= 4096 else flat))
        n = len(fft)
        e_low = float(np.sum(fft[:n//10]**2))
        e_mid = float(np.sum(fft[n//10:n//2]**2))
        e_high = float(np.sum(fft[n//2:]**2))
        features.append([rms, kurt, peak, crest, e_low, e_mid, e_high])
    return np.array(features)


# =============================================================================
# Experiments
# =============================================================================

def exp_health_indicator(jepa_embeddings, rms_values, rul_normalized, file_indices, n_files):
    """
    Zero-shot health indicator: embedding distance from healthy centroid.
    Compare JEPA vs RMS-based indicator.
    """
    print("\n" + "="*60)
    print("EXP 4B-1: ZERO-SHOT HEALTH INDICATOR")
    print("="*60)

    n = len(file_indices)
    time_idx = np.array(file_indices)  # temporal order (proxy for time)

    # JEPA: L2 distance from healthy centroid
    n_healthy = max(1, n // 4)
    healthy_embeds = jepa_embeddings[:n_healthy]
    healthy_centroid = healthy_embeds.mean(axis=0)

    l2_distances = np.linalg.norm(jepa_embeddings - healthy_centroid, axis=1)
    cos_distances = 1 - (jepa_embeddings @ healthy_centroid) / (
        np.linalg.norm(jepa_embeddings, axis=1) * np.linalg.norm(healthy_centroid) + 1e-8
    )

    # RMS: distance from healthy mean RMS
    healthy_rms_mean = float(rms_values[:n_healthy].mean())
    healthy_rms_std = float(rms_values[:n_healthy].std() + 1e-8)
    rms_deviation = (rms_values - healthy_rms_mean) / healthy_rms_std

    # Spearman correlations with time
    spear_jepa_l2, p_l2 = stats.spearmanr(time_idx, l2_distances)
    spear_jepa_cos, p_cos = stats.spearmanr(time_idx, cos_distances)
    spear_rms, p_rms = stats.spearmanr(time_idx, rms_values)

    print(f"JEPA L2 distance Spearman with time: {spear_jepa_l2:.4f} (p={p_l2:.2e})")
    print(f"JEPA cosine distance Spearman with time: {spear_jepa_cos:.4f} (p={p_cos:.2e})")
    print(f"RMS Spearman with time: {spear_rms:.4f} (p={p_rms:.2e})")

    # Early warning: when does indicator cross healthy_mean + 3*std?
    healthy_l2_mean = float(l2_distances[:n_healthy].mean())
    healthy_l2_std = float(l2_distances[:n_healthy].std() + 1e-8)
    alarm_threshold = healthy_l2_mean + 3 * healthy_l2_std

    alarm_idx = None
    for i in range(n_healthy, n):
        if l2_distances[i] > alarm_threshold:
            alarm_idx = i
            break

    if alarm_idx is not None:
        remaining_frac = 1.0 - (file_indices[alarm_idx] / n_files)
        print(f"JEPA early warning at file {file_indices[alarm_idx]}/{n_files}")
        print(f"  = {remaining_frac*100:.1f}% of run remaining before failure")
    else:
        remaining_frac = None
        print("JEPA: No early warning detected")

    # RMS early warning
    healthy_rms_alarm = healthy_rms_mean + 3 * healthy_rms_std
    rms_alarm_idx = None
    for i in range(n_healthy, n):
        if rms_values[i] > healthy_rms_alarm:
            rms_alarm_idx = i
            break

    if rms_alarm_idx is not None:
        rms_remaining = 1.0 - (file_indices[rms_alarm_idx] / n_files)
        print(f"RMS early warning at file {file_indices[rms_alarm_idx]}/{n_files}")
        print(f"  = {rms_remaining*100:.1f}% of run remaining before failure")
    else:
        rms_remaining = None
        print("RMS: No early warning detected")

    results = {
        'jepa_l2_spearman': spear_jepa_l2,
        'jepa_cos_spearman': spear_jepa_cos,
        'rms_spearman': spear_rms,
        'jepa_early_warning_frac': remaining_frac,
        'rms_early_warning_frac': rms_remaining,
        'l2_distances': l2_distances,
        'rms_values': rms_values,
        'cos_distances': cos_distances,
        'time_idx': time_idx,
        'n_files': n_files,
    }

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # JEPA L2 distance over time
        ax = axes[0, 0]
        ax.plot(time_idx, l2_distances, 'b-', alpha=0.7, linewidth=0.8, label='JEPA L2 distance')
        ax.axhline(alarm_threshold, color='red', linestyle='--', label=f'Alarm threshold (μ+3σ)')
        if alarm_idx is not None:
            ax.axvline(file_indices[alarm_idx], color='orange', linestyle=':', linewidth=2,
                       label=f'JEPA alarm (file {file_indices[alarm_idx]})')
        ax.set_xlabel('File index (time)')
        ax.set_ylabel('L2 distance from healthy centroid')
        ax.set_title(f'JEPA Health Indicator\nSpearman={spear_jepa_l2:.3f}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # RMS over time
        ax = axes[0, 1]
        ax.plot(time_idx, rms_values, 'g-', alpha=0.7, linewidth=0.8, label='RMS')
        ax.axhline(healthy_rms_alarm, color='red', linestyle='--', label='Alarm threshold')
        if rms_alarm_idx is not None:
            ax.axvline(file_indices[rms_alarm_idx], color='orange', linestyle=':', linewidth=2,
                       label=f'RMS alarm (file {file_indices[rms_alarm_idx]})')
        ax.set_xlabel('File index (time)')
        ax.set_ylabel('RMS value')
        ax.set_title(f'RMS Health Indicator\nSpearman={spear_rms:.3f}', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Correlation comparison bar chart
        ax = axes[1, 0]
        methods = ['JEPA L2', 'JEPA Cosine', 'RMS']
        spearman_vals = [spear_jepa_l2, spear_jepa_cos, spear_rms]
        colors = ['steelblue', 'navy', 'forestgreen']
        bars = ax.bar(methods, [abs(v) for v in spearman_vals], color=colors, alpha=0.8)
        ax.set_ylabel('|Spearman correlation| with time')
        ax.set_title('Health Indicator Quality\n(Higher = better tracks degradation)', fontweight='bold')
        ax.set_ylim(0, 1.0)
        for bar, val in zip(bars, spearman_vals):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Early warning comparison
        ax = axes[1, 1]
        warnings = []
        warn_labels = []
        if remaining_frac is not None:
            warnings.append(remaining_frac * 100)
            warn_labels.append('JEPA L2')
        if rms_remaining is not None:
            warnings.append(rms_remaining * 100)
            warn_labels.append('RMS')

        if warnings:
            bar_colors = ['steelblue' if 'JEPA' in l else 'forestgreen' for l in warn_labels]
            bars = ax.bar(warn_labels, warnings, color=bar_colors, alpha=0.8)
            ax.set_ylabel('% of run remaining at alarm')
            ax.set_title('Early Warning Lead Time\n(Higher = earlier warning)', fontweight='bold')
            for bar, val in zip(bars, warnings):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No early warnings detected', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_title('Early Warning Lead Time')

        plt.suptitle('IMS Run-to-Failure: JEPA vs RMS Health Indicators', fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = PLOTS_DIR / 'v4_jepa_rul_health_indicator.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

    return results


def exp_rul_regression(jepa_embeddings, hand_features, rul_normalized, n_seeds=3):
    """
    RUL regression: compare JEPA embeddings vs hand-crafted features.
    Train/test split: first 70% = train, last 30% = test.
    """
    print("\n" + "="*60)
    print("EXP 4B-2: RUL REGRESSION")
    print("="*60)

    n = len(rul_normalized)
    split = int(0.7 * n)

    y = rul_normalized.astype(np.float32)
    y_train, y_test = y[:split], y[split:]

    results = {}

    for method, X in [('JEPA', jepa_embeddings), ('Hand (RMS+kurtosis+etc)', hand_features)]:
        X_train, X_test = X[:split], X[split:]

        rmse_list, mae_list, spear_list = [], [], []
        for seed in range(n_seeds):
            np.random.seed(seed * 42)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)

            # Replace any NaN with 0
            X_tr_s = np.nan_to_num(X_tr_s, nan=0.0)
            X_te_s = np.nan_to_num(X_te_s, nan=0.0)

            reg = Ridge(alpha=1.0)
            reg.fit(X_tr_s, y_train)
            y_pred = reg.predict(X_te_s)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            spear, _ = stats.spearmanr(y_pred, y_test)

            rmse_list.append(rmse)
            mae_list.append(mae)
            spear_list.append(spear)

        mean_rmse = np.mean(rmse_list)
        std_rmse = np.std(rmse_list)
        mean_spear = np.mean(spear_list)

        print(f"\n{method}:")
        print(f"  RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
        print(f"  MAE:  {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
        print(f"  Spearman: {mean_spear:.4f} ± {np.std(spear_list):.4f}")

        results[method] = {
            'rmse_mean': mean_rmse,
            'rmse_std': std_rmse,
            'spearman_mean': mean_spear,
            'spearman_std': np.std(spear_list),
        }

    # Constant baseline (predict mean RUL = 0.5 always)
    rmse_const = float(np.sqrt(mean_squared_error(y_test, np.full_like(y_test, 0.5))))
    print(f"\nConstant baseline (always predict 0.5): RMSE={rmse_const:.4f}")
    results['Constant(0.5)'] = {'rmse_mean': rmse_const, 'rmse_std': 0.0, 'spearman_mean': 0.0}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-set', type=str, default='Test1')
    parser.add_argument('--subsample', type=int, default=4,
                        help='Take every N-th file to speed up (default: 4)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load JEPA model
    print(f"\nLoading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    embed_dim = config.get('embed_dim', 512)

    model = MechanicalJEPAV2(
        n_channels=3,
        window_size=4096,
        patch_size=256,
        embed_dim=embed_dim,
        encoder_depth=config.get('encoder_depth', 4),
        predictor_depth=config.get('predictor_depth', 4),
        mask_ratio=config.get('mask_ratio', 0.625),
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        var_reg_lambda=0.0,
        loss_fn='mse',
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Model loaded: embed_dim={embed_dim}")

    # Load IMS data
    signals, file_indices, rul_normalized, rms_values = load_ims_run(
        test_set=args.test_set,
        n_channels=3,
        window_size=4096,
        subsample=args.subsample,
    )

    n_files_total = len(sorted((IMS_DIR / args.test_set).glob('*')))
    n = len(signals)
    print(f"\nLoaded {n} files (subsampled from {n_files_total})")
    print(f"RUL range: [{rul_normalized.min():.3f}, {rul_normalized.max():.3f}]")
    print(f"RMS range: [{rms_values.min():.4f}, {rms_values.max():.4f}]")

    # Extract JEPA embeddings
    print("\nExtracting JEPA embeddings...")
    jepa_embeddings = extract_jepa_embeddings(model, signals, device, batch_size=32)
    print(f"JEPA embeddings: {jepa_embeddings.shape}")
    print(f"JEPA embeddings NaN: {np.isnan(jepa_embeddings).sum()}")

    # Extract hand-crafted features
    print("Extracting hand features...")
    hand_features = compute_hand_features(signals)
    print(f"Hand features: {hand_features.shape}")

    # Run experiments
    health_results = exp_health_indicator(
        jepa_embeddings, rms_values, rul_normalized,
        file_indices=file_indices, n_files=n_files_total,
    )

    rul_results = exp_rul_regression(
        jepa_embeddings, hand_features, rul_normalized, n_seeds=len(args.seeds)
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nZero-shot health indicator (Spearman with time):")
    print(f"  JEPA L2:    {health_results['jepa_l2_spearman']:.4f}")
    print(f"  JEPA cos:   {health_results['jepa_cos_spearman']:.4f}")
    print(f"  RMS:        {health_results['rms_spearman']:.4f}")

    if health_results['jepa_early_warning_frac'] is not None:
        print(f"\nEarly warning (% run remaining):")
        print(f"  JEPA: {health_results['jepa_early_warning_frac']*100:.1f}%")
        if health_results['rms_early_warning_frac']:
            print(f"  RMS:  {health_results['rms_early_warning_frac']*100:.1f}%")

    print(f"\nRUL Regression RMSE:")
    for method, res in rul_results.items():
        print(f"  {method}: {res['rmse_mean']:.4f} ± {res.get('rmse_std', 0):.4f}")

    # Does JEPA beat RMS for prognostics?
    jepa_spear = health_results['jepa_l2_spearman']
    rms_spear = health_results['rms_spearman']
    jepa_rmse = rul_results.get('JEPA', {}).get('rmse_mean', 1.0)
    hand_rmse = rul_results.get('Hand (RMS+kurtosis+etc)', {}).get('rmse_mean', 1.0)

    print(f"\nCritical question: Does JEPA add value over RMS for prognostics?")
    print(f"  Health indicator: JEPA Spearman={jepa_spear:.4f} vs RMS Spearman={rms_spear:.4f}")
    if jepa_spear > rms_spear:
        print(f"  --> JEPA BETTER than RMS for health monitoring (+{(jepa_spear-rms_spear):.4f})")
    else:
        print(f"  --> RMS BETTER than JEPA for health monitoring ({(rms_spear-jepa_spear):.4f} gap)")
    print(f"  RUL regression: JEPA RMSE={jepa_rmse:.4f} vs Hand RMSE={hand_rmse:.4f}")
    if jepa_rmse < hand_rmse:
        print(f"  --> JEPA BETTER for RUL regression ({(hand_rmse-jepa_rmse):.4f} lower RMSE)")
    else:
        print(f"  --> Hand features BETTER for RUL regression ({(jepa_rmse-hand_rmse):.4f} gap)")


if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent)
    main()

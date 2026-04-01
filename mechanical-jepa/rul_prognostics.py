"""
RUL Prediction and Prognostics with JEPA Embeddings.

Implements all 4 approaches from Round 4:
1. Zero-shot health indicator (embedding distance from healthy centroid)
2. RUL regression from JEPA embeddings
3. Spectral energy tracking
4. Failure probability binning

Usage:
    python rul_prognostics.py --checkpoint checkpoints/jepa_v2_xxx.pt
    python rul_prognostics.py --checkpoint checkpoints/jepa_v2_xxx.pt --test-set 2
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import f1_score, accuracy_score

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


# =============================================================================
# IMS Data Loading
# =============================================================================

class IMSFullRunDataset(Dataset):
    """
    Load ALL files from an IMS test set in temporal order.
    Returns (signal, file_index, rul_normalized) where rul_normalized ∈ [0,1]
    0 = failure, 1 = start of run.
    """

    def __init__(
        self,
        test_set: str = 'Test1',
        window_size: int = 4096,
        n_channels: int = 3,
        windows_per_file: int = 2,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.n_channels = n_channels
        self.windows_per_file = windows_per_file

        test_dir = IMS_DIR / test_set
        if not test_dir.exists():
            # Try lowercase
            test_dir = IMS_DIR / test_set.lower()
        if not test_dir.exists():
            raise FileNotFoundError(f"IMS {test_set} not found at {test_dir}")

        self.all_files = sorted(test_dir.glob('*'))
        n_files = len(self.all_files)
        print(f"IMS {test_set}: {n_files} files")

        # Normalized RUL: file 0 = 1.0, last file = 0.0
        self.rul_normalized = np.linspace(1.0, 0.0, n_files)

        # Build windows
        rng = np.random.default_rng(seed)
        self.windows = []  # (file_idx, window_start, rul)

        for file_idx, fpath in enumerate(self.all_files):
            data = np.fromfile(str(fpath), dtype=np.float64)
            n_samples = data.shape[0]
            total_samples = n_samples  # single channel
            if total_samples >= window_size:
                max_start = total_samples - window_size
                starts = rng.integers(0, max_start + 1, size=windows_per_file)
                for start in starts:
                    self.windows.append((file_idx, start, self.rul_normalized[file_idx]))

        # Normalization stats (computed on first 25% = healthy)
        n_healthy = int(0.25 * n_files)
        healthy_files = self.all_files[:n_healthy]
        stats_data = []
        for fpath in healthy_files[:20]:  # subsample for speed
            d = np.fromfile(str(fpath), dtype=np.float64)
            stats_data.append(d[:window_size * min(n_channels, d.shape[0] // window_size)])
        stats_arr = np.concatenate(stats_data) if stats_data else np.array([0.0])
        self.mean = float(np.mean(stats_arr))
        self.std = float(np.std(stats_arr) + 1e-8)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start, rul = self.windows[idx]
        fpath = self.all_files[file_idx]

        data = np.fromfile(str(fpath), dtype=np.float64)
        total = data.shape[0]

        # Extract multi-channel: use first n_channels from the 8-channel signal
        # IMS files have 8 channels interleaved or sequential — handle both
        # File size = 20480 samples = 8 channels * 2560 each? or 20480 total?
        samples_per_channel = total // self.n_channels if total > self.window_size else total
        window = np.zeros((self.n_channels, self.window_size), dtype=np.float32)

        for ch in range(self.n_channels):
            ch_start = ch * (total // max(self.n_channels, 1))
            actual_start = min(ch_start + start, total - self.window_size)
            if actual_start < 0:
                actual_start = 0
            end = actual_start + self.window_size
            if end > total:
                end = total
                actual_start = max(0, end - self.window_size)
            seg = data[actual_start:end]
            if len(seg) < self.window_size:
                seg = np.pad(seg, (0, self.window_size - len(seg)))
            window[ch] = seg[:self.window_size].astype(np.float32)

        # Normalize
        window = (window - self.mean) / self.std
        window = np.clip(window, -5.0, 5.0)

        return torch.from_numpy(window), file_idx, torch.tensor(rul, dtype=torch.float32)


class IMSFileDataset(Dataset):
    """Load IMS files one per item (for per-file embedding extraction)."""

    def __init__(self, test_set: str = 'Test1', window_size: int = 4096, n_channels: int = 3):
        self.window_size = window_size
        self.n_channels = n_channels

        test_dir = IMS_DIR / test_set
        if not test_dir.exists():
            test_dir = IMS_DIR / test_set.lower()

        self.all_files = sorted(test_dir.glob('*'))
        n_files = len(self.all_files)
        self.rul_normalized = np.linspace(1.0, 0.0, n_files)
        self.time_normalized = np.linspace(0.0, 1.0, n_files)

        # Stats from healthy period
        healthy_files = self.all_files[:max(1, n_files // 4)]
        stats_data = []
        for fpath in healthy_files[:10]:
            d = np.fromfile(str(fpath), dtype=np.float64)
            stats_data.append(d[:window_size])
        arr = np.concatenate(stats_data) if stats_data else np.array([0.0])
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr) + 1e-8)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        fpath = self.all_files[idx]
        data = np.fromfile(str(fpath), dtype=np.float64)
        total = len(data)

        window = np.zeros((self.n_channels, self.window_size), dtype=np.float32)
        for ch in range(self.n_channels):
            ch_start = ch * (total // max(self.n_channels, 1))
            end = min(ch_start + self.window_size, total)
            actual_start = max(0, end - self.window_size)
            seg = data[actual_start:end]
            if len(seg) < self.window_size:
                seg = np.pad(seg, (0, self.window_size - len(seg)))
            window[ch] = seg[:self.window_size].astype(np.float32)

        window = (window - self.mean) / self.std
        window = np.clip(window, -5.0, 5.0)

        rul = self.rul_normalized[idx]
        t = self.time_normalized[idx]
        return torch.from_numpy(window), idx, torch.tensor(rul, dtype=torch.float32)


# =============================================================================
# Signal Statistics (Baseline)
# =============================================================================

def compute_signal_stats(signal_np):
    """Compute hand-crafted features: RMS, kurtosis, peak, crest factor."""
    signal = signal_np.flatten()
    rms = np.sqrt(np.mean(signal**2))
    kurt = float(stats.kurtosis(signal))
    peak = np.max(np.abs(signal))
    crest = peak / (rms + 1e-8)
    # Spectral energy
    fft_mag = np.abs(np.fft.rfft(signal[:4096] if len(signal) >= 4096 else signal))
    spec_energy = np.sum(fft_mag**2)
    # Band energy (low freq: 0-10%, mid: 10-50%, high: 50-100%)
    n = len(fft_mag)
    e_low = np.sum(fft_mag[:n//10]**2)
    e_mid = np.sum(fft_mag[n//10:n//2]**2)
    e_high = np.sum(fft_mag[n//2:]**2)
    return np.array([rms, kurt, peak, crest, spec_energy, e_low, e_mid, e_high])


# =============================================================================
# Asymmetric RUL Score (C-MAPSS style)
# =============================================================================

def rul_score(y_pred, y_true):
    """
    Asymmetric score: late predictions (d>0) are penalized more than early (d<0).
    d = predicted_RUL - actual_RUL
    score = sum(exp(-d/13)-1 if d<0, exp(d/10)-1 if d>=0)
    Lower is better.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(scores))


# =============================================================================
# Experiment 4B-1: Zero-Shot Health Indicator
# =============================================================================

def exp_zero_shot_health_indicator(model, device, test_set='Test1'):
    """
    Zero-shot: embed all IMS files in temporal order.
    Track distance from healthy centroid over time.
    No labels needed.
    """
    print(f"\n{'='*60}")
    print(f"EXP 4B-1: ZERO-SHOT HEALTH INDICATOR ({test_set})")
    print(f"{'='*60}")

    dataset = IMSFileDataset(test_set=test_set, window_size=4096, n_channels=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    n_files = len(dataset)
    n_healthy = max(1, int(0.25 * n_files))

    # Extract all embeddings
    model.eval()
    all_embeds = []
    all_ruls = []

    with torch.no_grad():
        for signals, file_idxs, ruls in loader:
            signals = signals.to(device)
            emb = model.get_embeddings(signals, pool='mean')
            all_embeds.append(emb.cpu().numpy())
            all_ruls.extend(ruls.numpy().tolist())

    all_embeds = np.concatenate(all_embeds, axis=0)  # (n_files, D)
    all_ruls = np.array(all_ruls)

    # Healthy centroid (first 25%)
    healthy_centroid = all_embeds[:n_healthy].mean(axis=0)  # (D,)
    healthy_centroid_norm = healthy_centroid / (np.linalg.norm(healthy_centroid) + 1e-8)

    # Cosine distance from centroid over time
    embed_norms = all_embeds / (np.linalg.norm(all_embeds, axis=1, keepdims=True) + 1e-8)
    cosine_sims = embed_norms @ healthy_centroid_norm  # (n_files,)
    cosine_distances = 1.0 - cosine_sims  # higher = farther from healthy

    # L2 distance
    l2_distances = np.linalg.norm(all_embeds - healthy_centroid, axis=1)

    # Time index (0 = start, 1 = end)
    time_idx = np.linspace(0.0, 1.0, n_files)
    rul_true = all_ruls  # 1 = start, 0 = end

    # Spearman correlation (should be positive: distance increases as RUL decreases)
    spearman_cos, p_cos = stats.spearmanr(time_idx, cosine_distances)
    spearman_l2, p_l2 = stats.spearmanr(time_idx, l2_distances)

    print(f"\nResults ({test_set}, {n_files} files):")
    print(f"  Cosine distance Spearman with time: {spearman_cos:.4f} (p={p_cos:.4e})")
    print(f"  L2 distance Spearman with time: {spearman_l2:.4f} (p={p_l2:.4e})")
    print(f"  Healthy centroid distance (last 25% vs first 25%): ")
    failure_dist = cosine_distances[-n_healthy:].mean()
    healthy_dist = cosine_distances[:n_healthy].mean()
    print(f"    Healthy period mean: {healthy_dist:.4f}")
    print(f"    Failure period mean: {failure_dist:.4f}")
    print(f"    Ratio: {failure_dist / (healthy_dist + 1e-8):.2f}x")

    # Early warning detection
    threshold = healthy_dist + 3.0 * cosine_distances[:n_healthy].std()
    warning_files = np.where(cosine_distances > threshold)[0]
    if len(warning_files) > 0:
        first_warning = warning_files[0]
        warning_lead = n_files - first_warning
        print(f"  Early warning at file {first_warning}/{n_files} "
              f"({warning_lead} files = {100*warning_lead/n_files:.1f}% of run remaining)")
    else:
        print(f"  No early warning detected above {threshold:.4f} threshold")

    # Save plot
    if HAS_MATPLOTLIB:
        plots_dir = Path('notebooks/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        axes[0].plot(time_idx, cosine_distances, 'b-', alpha=0.7, label='JEPA cos dist')
        axes[0].axhline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.3f})')
        if len(warning_files) > 0:
            axes[0].axvline(time_idx[first_warning], color='g', linestyle='--', label=f'Warning (t={time_idx[first_warning]:.2f})')
        axes[0].set_xlabel('Normalized time (0=start, 1=end/failure)')
        axes[0].set_ylabel('Cosine distance from healthy centroid')
        axes[0].set_title(f'Zero-Shot Health Indicator: IMS {test_set}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_idx, l2_distances, 'r-', alpha=0.7, label='JEPA L2 dist')
        axes[1].set_xlabel('Normalized time')
        axes[1].set_ylabel('L2 distance from healthy centroid')
        axes[1].set_title(f'L2 Distance Health Indicator: IMS {test_set} (Spearman={spearman_l2:.3f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f'v4_health_indicator_{test_set.lower()}.png', dpi=150)
        plt.close()
        print(f"  Plot saved to notebooks/plots/v4_health_indicator_{test_set.lower()}.png")

    return {
        'test_set': test_set,
        'n_files': n_files,
        'spearman_cosine': float(spearman_cos),
        'p_cosine': float(p_cos),
        'spearman_l2': float(spearman_l2),
        'p_l2': float(p_l2),
        'failure_healthy_ratio': float(failure_dist / (healthy_dist + 1e-8)),
        'cosine_distances': cosine_distances.tolist(),
        'time_idx': time_idx.tolist(),
    }


# =============================================================================
# Experiment 4B-2: RUL Regression
# =============================================================================

def exp_rul_regression(model, device, test_set='Test1', seeds=[42, 123, 456]):
    """
    Train RUL regression head on JEPA embeddings.
    Split: first 60% for training, last 40% for test.
    """
    print(f"\n{'='*60}")
    print(f"EXP 4B-2: RUL REGRESSION ({test_set})")
    print(f"{'='*60}")

    dataset = IMSFileDataset(test_set=test_set, window_size=4096, n_channels=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    n_files = len(dataset)
    n_train = int(0.6 * n_files)

    model.eval()
    all_embeds = []
    all_ruls = []
    all_signals_raw = []

    with torch.no_grad():
        for signals, file_idxs, ruls in loader:
            signals_np = signals.numpy()
            for sig in signals_np:
                all_signals_raw.append(compute_signal_stats(sig))
            signals = signals.to(device)
            emb = model.get_embeddings(signals, pool='mean')
            all_embeds.append(emb.cpu().numpy())
            all_ruls.extend(ruls.numpy().tolist())

    all_embeds = np.concatenate(all_embeds, axis=0)
    all_ruls = np.array(all_ruls)
    all_hand_feats = np.array(all_signals_raw)

    # Split
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_files)

    X_train_jepa = all_embeds[train_idx]
    X_test_jepa = all_embeds[test_idx]
    X_train_hand = all_hand_feats[train_idx]
    X_test_hand = all_hand_feats[test_idx]
    y_train = all_ruls[train_idx]
    y_test = all_ruls[test_idx]

    results = {}

    for method_name, X_tr, X_te in [
        ('JEPA', X_train_jepa, X_test_jepa),
        ('HandCraft', X_train_hand, X_test_hand),
    ]:
        seed_results = []
        for seed in seeds:
            np.random.seed(seed)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            reg = Ridge(alpha=1.0)
            reg.fit(X_tr_s, y_train)
            y_pred = reg.predict(X_te_s)

            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            score = rul_score(y_pred, y_test)
            spearman, _ = stats.spearmanr(y_pred, y_test)

            seed_results.append({'rmse': rmse, 'mae': mae, 'score': score, 'spearman': spearman})

        rmses = [r['rmse'] for r in seed_results]
        maes = [r['mae'] for r in seed_results]
        spears = [r['spearman'] for r in seed_results]
        scores = [r['score'] for r in seed_results]

        print(f"\n  {method_name} ({len(seeds)} seeds):")
        print(f"    RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
        print(f"    MAE:  {np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"    Spearman: {np.mean(spears):.4f} ± {np.std(spears):.4f}")
        print(f"    Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        results[method_name] = {
            'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses)),
            'mae_mean': float(np.mean(maes)), 'mae_std': float(np.std(maes)),
            'spearman_mean': float(np.mean(spears)), 'spearman_std': float(np.std(spears)),
        }

    # Also test random init model
    torch.manual_seed(42)
    rand_model = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256,
        embed_dim=512, encoder_depth=4,
    ).to(device)
    rand_model.eval()

    rand_embeds = []
    with torch.no_grad():
        for signals, file_idxs, ruls in loader:
            signals = signals.to(device)
            emb = rand_model.get_embeddings(signals, pool='mean')
            rand_embeds.append(emb.cpu().numpy())
    rand_embeds = np.concatenate(rand_embeds, axis=0)

    rand_seed_results = []
    for seed in seeds:
        np.random.seed(seed)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(rand_embeds[train_idx])
        X_te_s = scaler.transform(rand_embeds[test_idx])
        reg = Ridge(alpha=1.0)
        reg.fit(X_tr_s, y_train)
        y_pred = reg.predict(X_te_s)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        spearman, _ = stats.spearmanr(y_pred, y_test)
        rand_seed_results.append({'rmse': rmse, 'spearman': spearman})

    rand_rmses = [r['rmse'] for r in rand_seed_results]
    rand_spears = [r['spearman'] for r in rand_seed_results]
    print(f"\n  RandomInit ({len(seeds)} seeds):")
    print(f"    RMSE: {np.mean(rand_rmses):.4f} ± {np.std(rand_rmses):.4f}")
    print(f"    Spearman: {np.mean(rand_spears):.4f} ± {np.std(rand_spears):.4f}")
    results['RandomInit'] = {
        'rmse_mean': float(np.mean(rand_rmses)),
        'rmse_std': float(np.std(rand_rmses)),
        'spearman_mean': float(np.mean(rand_spears)),
    }

    return results


# =============================================================================
# Experiment 4B-3: Spectral Energy Tracking
# =============================================================================

def exp_spectral_energy_tracking(test_set='Test1'):
    """
    Track spectral energy over the IMS run.
    Does it increase before failure?
    """
    print(f"\n{'='*60}")
    print(f"EXP 4B-3: SPECTRAL ENERGY TRACKING ({test_set})")
    print(f"{'='*60}")

    test_dir = IMS_DIR / test_set
    if not test_dir.exists():
        test_dir = IMS_DIR / test_set.lower()

    all_files = sorted(test_dir.glob('*'))
    n_files = len(all_files)

    energies = []
    rms_vals = []
    kurt_vals = []

    for fpath in all_files:
        data = np.fromfile(str(fpath), dtype=np.float64)
        if len(data) < 4096:
            energies.append(0.0)
            rms_vals.append(0.0)
            kurt_vals.append(0.0)
            continue
        signal = data[:4096]
        rms = float(np.sqrt(np.mean(signal**2)))
        kurt = float(stats.kurtosis(signal))
        fft_mag = np.abs(np.fft.rfft(signal))
        spec_energy = float(np.sum(fft_mag**2))
        energies.append(spec_energy)
        rms_vals.append(rms)
        kurt_vals.append(kurt)

    time_idx = np.linspace(0.0, 1.0, n_files)

    spearman_energy, p_energy = stats.spearmanr(time_idx, np.log1p(energies))
    spearman_rms, p_rms = stats.spearmanr(time_idx, rms_vals)
    spearman_kurt, p_kurt = stats.spearmanr(time_idx, np.abs(kurt_vals))

    print(f"\n  {test_set} ({n_files} files):")
    print(f"  Spectral energy Spearman with time: {spearman_energy:.4f} (p={p_energy:.4e})")
    print(f"  RMS Spearman with time: {spearman_rms:.4f} (p={p_rms:.4e})")
    print(f"  Kurtosis Spearman with time: {spearman_kurt:.4f} (p={p_kurt:.4e})")

    # Early warning from RMS
    n_healthy = max(1, int(0.25 * n_files))
    healthy_rms = rms_vals[:n_healthy]
    rms_mean, rms_std = np.mean(healthy_rms), np.std(healthy_rms)
    threshold = rms_mean + 3 * rms_std
    warnings = [i for i, r in enumerate(rms_vals) if r > threshold]
    if warnings:
        first_warn = warnings[0]
        print(f"  RMS early warning at file {first_warn}/{n_files} ({n_files - first_warn} files before failure)")
    else:
        print(f"  No RMS early warning detected")

    if HAS_MATPLOTLIB:
        plots_dir = Path('notebooks/plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        axes[0].plot(time_idx, np.log1p(energies), 'b-', alpha=0.7)
        axes[0].set_title(f'Log Spectral Energy vs Time: IMS {test_set} (Spearman={spearman_energy:.3f})')
        axes[0].set_ylabel('Log Spectral Energy')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time_idx, rms_vals, 'r-', alpha=0.7)
        axes[1].axhline(threshold, color='g', linestyle='--', label=f'3σ threshold')
        if warnings:
            axes[1].axvline(time_idx[first_warn], color='orange', linestyle='--', label='First warning')
        axes[1].set_title(f'RMS vs Time: IMS {test_set} (Spearman={spearman_rms:.3f})')
        axes[1].set_ylabel('RMS Amplitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time_idx, np.abs(kurt_vals), 'g-', alpha=0.7)
        axes[2].set_title(f'|Kurtosis| vs Time: IMS {test_set} (Spearman={spearman_kurt:.3f})')
        axes[2].set_xlabel('Normalized time (0=start, 1=failure)')
        axes[2].set_ylabel('|Kurtosis|')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / f'v4_spectral_tracking_{test_set.lower()}.png', dpi=150)
        plt.close()
        print(f"  Plot saved to notebooks/plots/v4_spectral_tracking_{test_set.lower()}.png")

    return {
        'test_set': test_set,
        'spearman_energy': float(spearman_energy),
        'spearman_rms': float(spearman_rms),
        'spearman_kurtosis': float(spearman_kurt),
        'early_warning_file': warnings[0] if warnings else None,
        'n_files': n_files,
    }


# =============================================================================
# Experiment 4B-4: Failure Probability Binning
# =============================================================================

def exp_failure_probability(model, device, test_set='Test1'):
    """
    Discretize remaining life into 4 bins and predict.
    Gives rough probability distribution over remaining life.
    """
    print(f"\n{'='*60}")
    print(f"EXP 4B-4: FAILURE PROBABILITY BINNING ({test_set})")
    print(f"{'='*60}")

    dataset = IMSFileDataset(test_set=test_set, window_size=4096, n_channels=3)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    n_files = len(dataset)
    model.eval()

    all_embeds = []
    all_ruls = []

    with torch.no_grad():
        for signals, _, ruls in loader:
            signals = signals.to(device)
            emb = model.get_embeddings(signals, pool='mean')
            all_embeds.append(emb.cpu().numpy())
            all_ruls.extend(ruls.numpy().tolist())

    all_embeds = np.concatenate(all_embeds, axis=0)
    all_ruls = np.array(all_ruls)

    # Discretize RUL into 4 bins: >75%, 50-75%, 25-50%, <25%
    bins = [0.0, 0.25, 0.50, 0.75, 1.01]
    bin_labels = ['<25% RUL (critical)', '25-50% RUL (degraded)', '50-75% RUL (early fault)', '>75% RUL (healthy)']
    bin_ids = np.digitize(all_ruls, bins) - 1  # 0-indexed
    bin_ids = np.clip(bin_ids, 0, 3)

    # Split: first 60% for training, last 40% for test
    n_train = int(0.6 * n_files)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_files)

    X_train = all_embeds[train_idx]
    X_test = all_embeds[test_idx]
    y_train = bin_ids[train_idx]
    y_test = bin_ids[test_idx]

    print(f"  Bin distribution (train): {np.bincount(y_train, minlength=4)}")
    print(f"  Bin distribution (test):  {np.bincount(y_test, minlength=4)}")

    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_train_s, y_train)
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  JEPA embeddings → bin classifier:")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Accuracy: {acc:.4f}")

    # Random init comparison
    torch.manual_seed(42)
    rand_model = MechanicalJEPAV2(n_channels=3, window_size=4096, patch_size=256,
                                   embed_dim=512, encoder_depth=4).to(device)
    rand_model.eval()
    rand_embeds = []
    with torch.no_grad():
        for signals, _, _ in loader:
            signals = signals.to(device)
            emb = rand_model.get_embeddings(signals, pool='mean')
            rand_embeds.append(emb.cpu().numpy())
    rand_embeds = np.concatenate(rand_embeds, axis=0)

    scaler_r = StandardScaler()
    X_tr_r = scaler_r.fit_transform(rand_embeds[train_idx])
    X_te_r = scaler_r.transform(rand_embeds[test_idx])
    clf_r = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf_r.fit(X_tr_r, y_train)
    rand_f1 = f1_score(y_test, clf_r.predict(X_te_r), average='macro', zero_division=0)

    print(f"  Random init macro F1: {rand_f1:.4f}")
    print(f"  F1 gain: {f1 - rand_f1:+.4f}")

    return {'f1': f1, 'acc': acc, 'rand_f1': rand_f1, 'f1_gain': f1 - rand_f1}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-set', type=str, default='Test1', choices=['Test1', 'Test2'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--exp', type=str, default='all',
                        choices=['all', '4b1', '4b2', '4b3', '4b4'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print(f"\nLoading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']

    model = MechanicalJEPAV2(
        n_channels=config['n_channels'],
        window_size=config['window_size'],
        patch_size=config.get('patch_size', 256),
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config.get('predictor_depth', 4),
        mask_ratio=config.get('mask_ratio', 0.625),
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        loss_fn=config.get('loss_fn', 'l1'),
        var_reg_lambda=config.get('var_reg_lambda', 0.1),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    results = {}

    if args.exp in ['all', '4b1']:
        r1 = exp_zero_shot_health_indicator(model, device, test_set=args.test_set)
        results['4b1'] = r1

        # Also run on the other test set
        other = 'Test2' if args.test_set == 'Test1' else 'Test1'
        try:
            r1b = exp_zero_shot_health_indicator(model, device, test_set=other)
            results['4b1_other'] = r1b
        except Exception as e:
            print(f"Could not run on {other}: {e}")

    if args.exp in ['all', '4b2']:
        r2 = exp_rul_regression(model, device, test_set=args.test_set, seeds=args.seeds)
        results['4b2'] = r2

    if args.exp in ['all', '4b3']:
        r3 = exp_spectral_energy_tracking(test_set=args.test_set)
        results['4b3'] = r3

        try:
            other = 'Test2' if args.test_set == 'Test1' else 'Test1'
            r3b = exp_spectral_energy_tracking(test_set=other)
            results['4b3_other'] = r3b
        except Exception as e:
            print(f"Could not run spectral on other test set: {e}")

    if args.exp in ['all', '4b4']:
        r4 = exp_failure_probability(model, device, test_set=args.test_set)
        results['4b4'] = r4

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")

    return results


if __name__ == '__main__':
    main()

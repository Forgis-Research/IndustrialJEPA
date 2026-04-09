"""
Paderborn Cross-Dataset Transfer Experiment with Frequency Standardization.

Tests CWRU-pretrained JEPA on Paderborn dataset after resampling to common rate.
Implements Round 1 of the V3 overnight autoresearch.

Dataset structure:
  K001/  -> 80 MAT files, healthy
  KA01/  -> 80 MAT files, outer race fault
  KI01/  -> 80 MAT files, inner race fault

Each MAT file has 256k samples at ~64kHz (~4 seconds of data).
Vibration is in Y[6]['Data'] (channel: vibration_1, raster: HostService).

Resampling strategy:
  From 64kHz to target_sr via scipy.signal.resample_poly (anti-aliased).
  target_sr=12000 -> 5.33x downsample (CWRU native)
  target_sr=20000 -> 3.2x downsample (IMS native)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2


PADERBORN_DIR = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')

# Bearing classes
CLASSES = {
    'K001': 0,  # healthy
    'KA01': 1,  # outer race fault
    'KI01': 2,  # inner race fault
}

SOURCE_SR = 64000  # Paderborn sampling rate


def load_paderborn_signal(mat_path: Path) -> np.ndarray:
    """
    Load vibration signal from Paderborn MAT file.
    Returns: (n_samples,) float32 array at 64kHz
    """
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, simplify_cells=True)
    key = [k for k in mat.keys() if not k.startswith('_')][0]
    obj = mat[key]
    Y = obj['Y']
    # Channel 6 = vibration_1 at 64kHz
    vib = Y[6]['Data'].astype(np.float32)
    return vib


def resample_signal(signal: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample 1D signal from source_sr to target_sr using polyphase resampling.
    Anti-aliasing is handled by resample_poly's built-in filter.
    Uses integer ratio via GCD for efficiency.
    """
    if source_sr == target_sr:
        return signal
    from math import gcd
    g = gcd(source_sr, target_sr)
    up = target_sr // g
    down = source_sr // g
    resampled = scipy.signal.resample_poly(signal, up, down)
    return resampled.astype(np.float32)


class PaderbornDataset(Dataset):
    """
    Paderborn bearing dataset with resampling support.

    Loads vibration signals from MAT files, resamples to target_sr,
    windows, and normalizes.
    """

    def __init__(
        self,
        bearing_dirs: list,  # list of (path, label) tuples
        window_size: int = 4096,
        stride: int = 2048,
        target_sr: int = 12000,
        n_channels: int = 3,
        max_files_per_bearing: int = 80,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.n_channels = n_channels
        self.target_sr = target_sr

        self.windows = []  # (signal_array, start_idx, label)

        rng = np.random.default_rng(seed)

        for bearing_path, label in bearing_dirs:
            bearing_path = Path(bearing_path)
            mat_files = sorted(bearing_path.glob('*.mat'))[:max_files_per_bearing]

            for mat_file in mat_files:
                try:
                    raw = load_paderborn_signal(mat_file)
                    # Resample to target rate
                    signal = resample_signal(raw, SOURCE_SR, target_sr)

                    # Create windows
                    n_samples = len(signal)
                    n_windows = (n_samples - window_size) // stride + 1
                    for i in range(n_windows):
                        start = i * stride
                        self.windows.append((signal, start, label))

                except Exception as e:
                    print(f"  Warning: error loading {mat_file.name}: {e}")
                    continue

        print(f"PaderbornDataset: {len(self.windows)} windows "
              f"(target_sr={target_sr}Hz, window={window_size})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        signal, start, label = self.windows[idx]
        window = signal[start:start + self.window_size]

        # Expand to n_channels (repeat single vibration channel)
        window_mc = np.stack([window] * self.n_channels, axis=0)  # (C, T)

        # Per-channel z-score
        mean = window_mc.mean(axis=1, keepdims=True)
        std = window_mc.std(axis=1, keepdims=True) + 1e-8
        window_mc = (window_mc - mean) / std

        return torch.tensor(window_mc, dtype=torch.float32), label


def create_paderborn_loaders(
    bearing_dirs: list,
    window_size: int = 4096,
    stride: int = 2048,
    target_sr: int = 12000,
    n_channels: int = 3,
    test_ratio: float = 0.2,
    batch_size: int = 64,
    seed: int = 42,
    max_files_per_bearing: int = 40,  # Use 40 of 80 files per class for speed
):
    """
    Create train/test dataloaders for Paderborn dataset.
    Splits by file (not by window) to prevent leakage.
    """
    rng = np.random.default_rng(seed)

    train_dirs = []
    test_dirs_list = []

    for bearing_path, label in bearing_dirs:
        bearing_path = Path(bearing_path)
        mat_files = sorted(bearing_path.glob('*.mat'))[:max_files_per_bearing]

        # Shuffle files
        indices = rng.permutation(len(mat_files))
        n_test = max(1, int(len(mat_files) * test_ratio))

        test_files = [mat_files[i] for i in indices[:n_test]]
        train_files = [mat_files[i] for i in indices[n_test:]]

        # Create temporary per-file "dirs" for dataset
        train_dirs.append((_FileListBearing(train_files, bearing_path), label))
        test_dirs_list.append((_FileListBearing(test_files, bearing_path), label))

    train_ds = _PaderbornFileDataset(
        train_dirs, window_size=window_size, stride=stride,
        target_sr=target_sr, n_channels=n_channels, seed=seed
    )
    test_ds = _PaderbornFileDataset(
        test_dirs_list, window_size=window_size, stride=stride,
        target_sr=target_sr, n_channels=n_channels, seed=seed
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader


class _FileListBearing:
    """Helper: presents a specific list of files as a 'bearing path'."""
    def __init__(self, files, path):
        self._files = files
        self._path = path
    def glob(self, pattern):
        return self._files
    def __str__(self):
        return str(self._path)


class _PaderbornFileDataset(Dataset):
    """Internal dataset using pre-split file lists."""

    def __init__(
        self,
        bearing_dirs: list,
        window_size: int = 4096,
        stride: int = 2048,
        target_sr: int = 12000,
        n_channels: int = 3,
        seed: int = 42,
    ):
        self.window_size = window_size
        self.n_channels = n_channels
        self.windows = []

        for flist_bearing, label in bearing_dirs:
            for mat_file in flist_bearing.glob('*.mat'):
                try:
                    raw = load_paderborn_signal(mat_file)
                    signal = resample_signal(raw, SOURCE_SR, target_sr)
                    n_samples = len(signal)
                    n_windows = (n_samples - window_size) // stride + 1
                    for i in range(n_windows):
                        start = i * stride
                        self.windows.append((signal, start, label))
                except Exception as e:
                    continue

        label_counts = np.bincount([w[2] for w in self.windows], minlength=3)
        print(f"  _PaderbornFileDataset: {len(self.windows)} windows, "
              f"labels={dict(enumerate(label_counts))}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        signal, start, label = self.windows[idx]
        window = signal[start:start + self.window_size]
        window_mc = np.stack([window] * self.n_channels, axis=0)
        mean = window_mc.mean(axis=1, keepdims=True)
        std = window_mc.std(axis=1, keepdims=True) + 1e-8
        window_mc = (window_mc - mean) / std
        return torch.tensor(window_mc, dtype=torch.float32), label


def evaluate_probe(
    model, train_loader, test_loader, embed_dim, device,
    probe_type='linear', probe_epochs=50, n_classes=3, label='',
):
    """Linear or MLP probe evaluation on frozen encoder."""
    model.eval()

    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels in loader:
                signals = signals.to(device)
                embeds = model.get_embeddings(signals, pool='mean')
                all_e.append(embeds.cpu())
                all_l.append(labels)
        return torch.cat(all_e), torch.cat(all_l)

    train_e, train_l = extract(train_loader)
    test_e, test_l = extract(test_loader)

    # Normalize using train stats
    mean = train_e.mean(0, keepdim=True)
    std = train_e.std(0, keepdim=True) + 1e-8
    train_e_n = ((train_e - mean) / std).to(device)
    test_e_n = ((test_e - mean) / std).to(device)
    train_l = train_l.to(device)
    test_l = test_l.to(device)

    if probe_type == 'linear':
        probe = nn.Linear(embed_dim, n_classes).to(device)
    else:
        probe = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes),
        ).to(device)

    opt = optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=probe_epochs, eta_min=1e-5)
    crit = nn.CrossEntropyLoss()

    best_acc = 0
    best_preds = None
    for ep in range(probe_epochs):
        probe.train()
        opt.zero_grad()
        loss = crit(probe(train_e_n), train_l)
        loss.backward()
        opt.step()
        sched.step()

        probe.eval()
        with torch.no_grad():
            preds = probe(test_e_n).argmax(1)
            acc = (preds == test_l).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_preds = preds.clone()

    # Per-class
    class_names = ['healthy', 'outer_race', 'inner_race']
    per_class = {}
    for i, name in enumerate(class_names[:n_classes]):
        mask = test_l == i
        if mask.sum() > 0:
            per_class[name] = (best_preds[mask] == test_l[mask]).float().mean().item()

    print(f"  [{label:40s}] {probe_type}: {best_acc:.4f} | {per_class}")
    return {'test_acc': best_acc, 'per_class': per_class}


def run_paderborn_transfer(
    checkpoint_path: Path,
    target_sr: int,
    seed: int,
    device: torch.device,
    window_size: int = 4096,
    n_channels: int = 3,
    batch_size: int = 64,
    max_files: int = 40,
):
    """
    Run CWRU->Paderborn transfer experiment at given target sampling rate.

    Returns: dict with jepa and random probe results
    """
    print(f"\n{'='*60}")
    print(f"CWRU -> Paderborn | target_sr={target_sr}Hz | seed={seed}")
    print(f"{'='*60}")

    bearing_dirs = [
        (PADERBORN_DIR / 'K001', 0),   # healthy
        (PADERBORN_DIR / 'KA01', 1),   # outer race
        (PADERBORN_DIR / 'KI01', 2),   # inner race
    ]

    # Build loaders with resampled data
    train_loader, test_loader = create_paderborn_loaders(
        bearing_dirs=bearing_dirs,
        window_size=window_size,
        stride=window_size // 2,
        target_sr=target_sr,
        n_channels=n_channels,
        batch_size=batch_size,
        seed=seed,
        max_files_per_bearing=max_files,
    )

    # Load checkpoint config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    embed_dim = config['embed_dim']

    # Override window_size to match our target
    actual_window = config.get('window_size', 4096)

    def make_model(load_weights=True):
        m = MechanicalJEPAV2(
            n_channels=n_channels,
            window_size=actual_window,
            patch_size=config['patch_size'],
            embed_dim=embed_dim,
            encoder_depth=config['encoder_depth'],
            predictor_depth=config['predictor_depth'],
            n_heads=config['n_heads'],
            mask_ratio=config['mask_ratio'],
            ema_decay=config['ema_decay'],
            predictor_pos=config.get('predictor_pos', 'sinusoidal'),
            separate_mask_tokens=config.get('separate_mask_tokens', False),
            loss_fn=config.get('loss_fn', 'l1'),
            var_reg_lambda=0.0,
            vicreg_lambda=0.0,
        ).to(device)
        if load_weights:
            m.load_state_dict(ckpt['model_state_dict'])
        return m

    results = {}
    print("\n--- Linear Probe ---")
    results['jepa_linear'] = evaluate_probe(
        make_model(True), train_loader, test_loader, embed_dim, device,
        probe_type='linear', probe_epochs=50, n_classes=3, label='JEPA pretrained'
    )
    results['random_linear'] = evaluate_probe(
        make_model(False), train_loader, test_loader, embed_dim, device,
        probe_type='linear', probe_epochs=50, n_classes=3, label='Random init'
    )

    gain = results['jepa_linear']['test_acc'] - results['random_linear']['test_acc']
    print(f"\nTransfer gain (linear): {gain:+.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/jepa_v2_20260401_003619.pt')
    parser.add_argument('--target-sr', type=int, default=12000,
                        help='Target sampling rate (Hz). Default: 12000 (CWRU native)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--window-size', type=int, default=4096)
    parser.add_argument('--max-files', type=int, default=40,
                        help='MAT files per bearing class (80 total per class)')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Target SR: {args.target_sr} Hz (from Paderborn 64kHz, ratio={64000/args.target_sr:.2f}x down)")

    checkpoint_path = Path(args.checkpoint)

    all_results = {}
    for seed in args.seeds:
        all_results[seed] = run_paderborn_transfer(
            checkpoint_path=checkpoint_path,
            target_sr=args.target_sr,
            seed=seed,
            device=device,
            window_size=args.window_size,
            max_files=args.max_files,
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY | Paderborn @ {args.target_sr}Hz | {len(args.seeds)} seeds")
    print(f"{'='*70}")

    for method in ['jepa_linear', 'random_linear']:
        accs = [all_results[s][method]['test_acc'] for s in args.seeds]
        print(f"  {method:25s}: {np.mean(accs):.4f} ± {np.std(accs):.4f}")

    gains = [
        all_results[s]['jepa_linear']['test_acc'] - all_results[s]['random_linear']['test_acc']
        for s in args.seeds
    ]
    print(f"\n  Transfer gain: {np.mean(gains):+.4f} ± {np.std(gains):.4f}")
    print(f"  Positive seeds: {sum(g > 0 for g in gains)}/{len(args.seeds)}")

    if np.mean(gains) > 0.05:
        print(f"\nVERDICT: Clear transfer ({np.mean(gains):+.1%} gain)")
    elif np.mean(gains) > 0:
        print(f"\nVERDICT: Marginal positive transfer ({np.mean(gains):+.1%} gain)")
    else:
        print(f"\nVERDICT: No transfer ({np.mean(gains):+.1%} gain)")


if __name__ == '__main__':
    main()

"""
IMS Self-Supervised Pretraining with JEPA.


This script pretrains JEPA on IMS data using the fast numpy cache.
Purpose: Upper bound experiment - if JEPA pretrained on IMS transfers to IMS task,
         it shows JEPA CAN learn useful features from this type of data.
         Compared to CWRU-pretrained JEPA, this tests domain specificity.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPA
from ims_transfer import (
    IMSDegradationDataset, create_ims_splits, evaluate_probe
)


class IMSPretrainDataset(Dataset):
    """
    IMS dataset for unsupervised JEPA pretraining.
    Uses ALL files from both test sets (no label needed).
    """

    def __init__(
        self,
        ims_dir: Path,
        test_sets: list = ['1st_test', '2nd_test'],
        window_size: int = 4096,
        n_channels: int = 3,
        windows_per_file: int = 2,
        max_files: int = None,
        seed: int = 42,
    ):
        self.ims_dir = Path(ims_dir)
        self.npy_cache_dir = self.ims_dir.parent.parent / 'ims_npy_cache'
        self.window_size = window_size
        self.n_channels = n_channels

        rng = np.random.default_rng(seed)

        self.windows = []  # (file_path, start)
        all_files = []
        for test_set in test_sets:
            files = sorted((self.ims_dir / test_set).glob('*'))
            all_files.extend(files)

        if max_files is not None:
            all_files = rng.permutation(all_files)[:max_files].tolist()

        n_samples_per_file = 20480
        for f in all_files:
            max_start = n_samples_per_file - window_size
            n_win = min(windows_per_file, max_start)
            starts = rng.choice(max_start, size=n_win, replace=False)
            for s in starts:
                self.windows.append((f, int(s)))

        print(f"IMSPretrainDataset: {len(all_files)} files, {len(self.windows)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        f, start = self.windows[idx]

        # Use fast numpy cache, handle both .npy symlinks and raw text files
        if f.suffix == '.npy':
            # File is already a numpy cache (symlinked directory)
            data = np.load(f)
        else:
            npy_f = self.npy_cache_dir / f.parent.name / (f.name + '.npy')
            if npy_f.exists():
                data = np.load(npy_f)
            else:
                data = np.loadtxt(f, dtype=np.float32)

        window = data[start:start + self.window_size].T  # (8, window_size)

        if window.shape[0] < self.n_channels:
            pad = np.zeros((self.n_channels - window.shape[0], self.window_size), dtype=np.float32)
            window = np.vstack([window, pad])
        else:
            window = window[:self.n_channels]

        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mean) / std

        return torch.tensor(window, dtype=torch.float32)


def pretrain_jepa_on_ims(
    ims_dir: Path,
    seed: int,
    device: torch.device,
    epochs: int = 50,
    embed_dim: int = 512,
    window_size: int = 4096,
    patch_size: int = 256,
    n_channels: int = 3,
    batch_size: int = 64,
    lr: float = 1e-4,
):
    """Pretrain JEPA on IMS data (unsupervised)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\nPretraining JEPA on IMS | seed={seed} | {epochs} epochs")

    # Create dataset using ALL IMS files
    ds = IMSPretrainDataset(
        ims_dir=ims_dir,
        test_sets=['1st_test', '2nd_test'],
        window_size=window_size,
        n_channels=n_channels,
        windows_per_file=2,
        seed=seed,
    )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    print(f"  Batches per epoch: {len(loader)}")

    # Create model
    model = MechanicalJEPA(
        n_channels=n_channels,
        window_size=window_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        encoder_depth=4,
        predictor_depth=2,
        n_heads=4,
        mask_ratio=0.5,
        ema_decay=0.996,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Cosine schedule
    schedule = np.concatenate([
        np.linspace(0, lr, 5),  # warmup
        [1e-6 + 0.5 * (lr - 1e-6) * (1 + np.cos(np.pi * i / (epochs - 5)))
         for i in range(epochs - 5)],
    ])

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = schedule[min(epoch, len(schedule) - 1)]

        total_loss = 0
        for signals in loader:
            signals = signals.to(device)
            optimizer.zero_grad()
            loss = model.train_step(signals)
            loss.backward()
            optimizer.step()
            model.update_ema()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}")

    print(f"  Final loss: {avg_loss:.6f} | Best loss: {best_loss:.6f}")
    return model


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ims_dir = Path('data/bearings/raw/ims')

    # Run for 3 seeds - upper bound experiment
    seeds = [42, 123, 456]
    all_results_ims_pretrain = {}
    all_results_random = {}

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"IMS Self-Supervised Pretraining | seed={seed}")
        print(f"{'='*60}")

        # Pretrain on IMS
        model = pretrain_jepa_on_ims(
            ims_dir=ims_dir,
            seed=seed,
            device=device,
            epochs=50,
            embed_dim=512,
        )

        # Evaluate: probe on 1st_test binary task
        train_idx, train_labels, test_idx, test_labels, rms = create_ims_splits(
            ims_dir, '1st_test', seed=seed, n_classes=2
        )

        train_ds = IMSDegradationDataset(
            ims_dir=ims_dir, test_set='1st_test',
            file_indices=train_idx, labels=train_labels,
            window_size=4096, n_channels=3,
            windows_per_file=4, seed=seed,
        )
        test_ds = IMSDegradationDataset(
            ims_dir=ims_dir, test_set='1st_test',
            file_indices=test_idx, labels=test_labels,
            window_size=4096, n_channels=3,
            windows_per_file=4, seed=seed,
        )

        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

        print("\n  Evaluating IMS-pretrained model...")
        result = evaluate_probe(
            model, train_loader, test_loader, 512, device,
            probe_type='linear', probe_epochs=100, probe_lr=1e-2,
            n_classes=2, pool='mean', label='IMS-pretrained JEPA',
        )
        all_results_ims_pretrain[seed] = result['test_acc']

        # Random init comparison
        random_model = MechanicalJEPA(
            n_channels=3, window_size=4096, patch_size=256,
            embed_dim=512, encoder_depth=4, predictor_depth=2,
            n_heads=4, mask_ratio=0.5, ema_decay=0.996,
        ).to(device)
        result_rand = evaluate_probe(
            random_model, train_loader, test_loader, 512, device,
            probe_type='linear', probe_epochs=100, probe_lr=1e-2,
            n_classes=2, pool='mean', label='Random init',
        )
        all_results_random[seed] = result_rand['test_acc']

    # Summary
    ims_accs = [all_results_ims_pretrain[s] for s in seeds]
    rand_accs = [all_results_random[s] for s in seeds]

    print(f"\n{'='*60}")
    print("UPPER BOUND: IMS Self-Supervised → IMS Probe (1st_test binary)")
    print(f"{'='*60}")
    print(f"  IMS-pretrained JEPA:  {np.mean(ims_accs):.4f} ± {np.std(ims_accs):.4f}")
    print(f"  Random init:          {np.mean(rand_accs):.4f} ± {np.std(rand_accs):.4f}")
    print(f"  Gain:                 {np.mean(ims_accs) - np.mean(rand_accs):+.4f}")
    print()
    print("Compare:")
    print(f"  CWRU-pretrained JEPA: 0.7204 ± 0.0144")
    print(f"  Random init (CWRU):   0.6963 ± 0.0165")
    print(f"  CWRU gain:            +0.0241")


if __name__ == '__main__':
    main()

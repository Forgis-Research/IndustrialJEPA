"""
Round 2: Fix RUL by pretraining on IMS run-to-failure data.

Current failure: JEPA pretrained on CWRU (fault TYPES) gives Spearman=0.08 for RUL.
Root cause: CWRU teaches fault DISCRIMINATION, not DEGRADATION DYNAMICS.

Fix: Pretrain JEPA directly on IMS run-to-failure data, then evaluate RUL.

Experiments:
- Exp V5-5: IMS-pretrained JEPA vs CWRU-pretrained vs random vs RMS baseline for RUL
- Exp V5-6: Channel-specific analysis (find the degrading channel)
- Exp V5-7: Improved zero-shot health indicator with Mahalanobis distance

Usage:
    python rul_ims_pretrained.py
    python rul_ims_pretrained.py --ims-epochs 50
"""

import argparse
import sys
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPAV2

CHECKPOINT_DIR = Path('/mnt/sagemaker-nvme/jepa_checkpoints')
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# IMS DATA LOADING
# =============================================================================

class IMSWindowDataset(Dataset):
    """
    IMS run-to-failure dataset for self-supervised JEPA pretraining.

    Each sample is a window of 4096 samples from IMS raw data.
    Labels are NOT used (self-supervised pretraining).

    Optionally provides RUL labels for supervised probe evaluation.
    """

    def __init__(self, ims_dir, window_size=4096, stride=2048, n_channels=3,
                 subsample=1, normalize=True, max_files=None):
        """
        Args:
            ims_dir: Directory with IMS raw files (tab-delimited text)
            window_size: Window size in samples
            stride: Window stride
            n_channels: Number of channels (first n_channels from IMS's 8)
            subsample: Use every Nth file
            normalize: Z-score normalize per channel (fit on training data)
            max_files: Limit number of files loaded (for speed)
        """
        self.window_size = window_size
        self.stride = stride
        self.n_channels = n_channels

        # Load all files
        ims_path = Path(ims_dir)
        if not ims_path.exists():
            raise FileNotFoundError(f"IMS directory not found: {ims_dir}")

        # Find all data files (they have no extension or .txt or just numbers)
        all_files = sorted([f for f in ims_path.iterdir()
                           if f.is_file() and not f.name.startswith('.')])

        if len(all_files) == 0:
            raise FileNotFoundError(f"No files found in {ims_dir}")

        print(f"  Found {len(all_files)} files in {ims_dir}")

        if subsample > 1:
            all_files = all_files[::subsample]
            print(f"  Subsampled to {len(all_files)} files (stride={subsample})")

        if max_files is not None:
            all_files = all_files[:max_files]
            print(f"  Limited to {max_files} files")

        n_files = len(all_files)

        # Load and create windows
        self.windows = []
        self.rul_labels = []  # Normalized RUL [0=failure, 1=start]
        self.file_indices = []

        print(f"  Loading {n_files} files...")
        for file_idx, fpath in enumerate(all_files):
            try:
                data = np.loadtxt(str(fpath))  # (20480, 8)
                if data.ndim == 1:
                    data = data.reshape(-1, 8)

                if data.shape[0] < window_size:
                    continue

                # Use first n_channels
                signal = data[:, :n_channels].T  # (n_channels, 20480)

                # Create windows
                n_windows = (signal.shape[1] - window_size) // stride + 1
                for w in range(n_windows):
                    start = w * stride
                    window = signal[:, start:start + window_size]
                    self.windows.append(window.astype(np.float32))

                    # RUL: 1.0 at start, 0.0 at last file
                    rul = 1.0 - file_idx / (n_files - 1)
                    self.rul_labels.append(rul)
                    self.file_indices.append(file_idx)

            except Exception as e:
                continue  # Skip problematic files

        print(f"  Created {len(self.windows)} windows from {n_files} files")

        if normalize and len(self.windows) > 0:
            # Z-score normalize per channel using dataset statistics
            all_signals = np.stack(self.windows)  # (N, C, L)
            self.mean = all_signals.mean(axis=(0, 2), keepdims=True).mean(axis=0)  # (C, 1)
            self.std = all_signals.std(axis=(0, 2), keepdims=True).mean(axis=0)  # (C, 1)
            self.std = np.maximum(self.std, 1e-6)

            # Normalize
            self.windows = [(w - self.mean) / self.std for w in self.windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = torch.from_numpy(self.windows[idx])
        rul = torch.tensor(self.rul_labels[idx], dtype=torch.float32)
        file_idx = torch.tensor(self.file_indices[idx], dtype=torch.long)
        return window, rul, file_idx


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs):
    warmup_schedule = np.linspace(0, base_value, warmup_epochs)
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )
    return np.concatenate([warmup_schedule, schedule])


# =============================================================================
# IMS JEPA PRETRAINING
# =============================================================================

def pretrain_jepa_on_ims(ims_dir, epochs=50, seed=42, device='cuda', save=True):
    """
    Pretrain JEPA V2 on IMS run-to-failure data (self-supervised).

    Returns trained model checkpoint path.
    """
    print("\n" + "=" * 60)
    print("PRETRAINING JEPA ON IMS (RUN-TO-FAILURE DATA)")
    print("=" * 60)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load IMS data
    print("\nLoading IMS data...")
    dataset = IMSWindowDataset(
        ims_dir=ims_dir,
        window_size=4096,
        stride=2048,
        n_channels=3,
        subsample=4,  # Use every 4th file to speed up
        normalize=True,
    )

    if len(dataset) == 0:
        raise ValueError("Empty IMS dataset!")

    # 80/20 split
    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} windows, Test: {len(test_ds)} windows")

    # Create JEPA V2 model (same architecture as V2 best)
    model = MechanicalJEPAV2(
        n_channels=3,
        window_size=4096,
        patch_size=256,
        embed_dim=512,
        encoder_depth=4,
        predictor_depth=4,
        n_heads=4,
        mask_ratio=0.625,
        ema_decay=0.996,
        predictor_pos='sinusoidal',
        loss_fn='l1',
        var_reg_lambda=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_schedule = cosine_scheduler(1e-4, 1e-6, epochs, 5)

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for windows, rul, file_idx in train_loader:
            windows = windows.to(device)
            lr = lr_schedule[min(epoch, len(lr_schedule) - 1)]
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad()
            loss = model.train_step(windows)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_ema()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}")

    # Save checkpoint
    ckpt_path = None
    if save:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') if 'datetime' in dir() else \
                    time.strftime('%Y%m%d_%H%M%S')
        ckpt_path = CHECKPOINT_DIR / f'jepa_ims_pretrained_{timestamp}_s{seed}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'n_channels': 3, 'window_size': 4096, 'patch_size': 256,
                'embed_dim': 512, 'encoder_depth': 4, 'predictor_depth': 4,
                'n_heads': 4, 'mask_ratio': 0.625, 'ema_decay': 0.996,
                'predictor_pos': 'sinusoidal', 'loss_fn': 'l1', 'var_reg_lambda': 0.1,
                'pretraining': 'ims', 'epochs': epochs, 'seed': seed,
            },
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    return model, train_loader, test_loader, dataset


# =============================================================================
# RUL EVALUATION
# =============================================================================

def evaluate_rul(model, dataset, device, method='jepa', n_train_frac=0.7):
    """
    Evaluate RUL prediction from embeddings using Ridge regression.

    Split: first 70% of files for training regression, last 30% for test.
    This is the TEMPORAL split — proper for run-to-failure data.

    Returns: RMSE, Spearman, early_warning_frac
    """
    model.eval()

    # Extract embeddings and RUL labels
    all_windows = [dataset[i][0] for i in range(len(dataset))]
    all_rul = np.array([dataset[i][1].item() for i in range(len(dataset))])
    all_file_idx = np.array([dataset[i][2].item() for i in range(len(dataset))])

    n_files = all_file_idx.max() + 1
    split_file = int(n_train_frac * n_files)

    train_mask = all_file_idx <= split_file
    test_mask = all_file_idx > split_file

    print(f"  Temporal split: {train_mask.sum()} train windows, {test_mask.sum()} test windows")

    # Extract embeddings in batches
    batch_size = 64
    all_embeds = []

    with torch.no_grad():
        for i in range(0, len(all_windows), batch_size):
            batch = torch.stack(all_windows[i:i+batch_size]).to(device)
            if method == 'jepa':
                embeds = model.get_embeddings(batch, pool='mean')
            else:
                # Random init — just extract from encoder
                embeds = model.get_embeddings(batch, pool='mean')
            all_embeds.append(embeds.cpu().numpy())

    all_embeds = np.concatenate(all_embeds, axis=0)

    # Train/test split
    train_embeds = all_embeds[train_mask]
    train_rul = all_rul[train_mask]
    test_embeds = all_embeds[test_mask]
    test_rul = all_rul[test_mask]

    # Normalize embeddings
    scaler = StandardScaler()
    train_embeds_norm = scaler.fit_transform(train_embeds)
    test_embeds_norm = scaler.transform(test_embeds)

    # Ridge regression for RUL
    ridge = Ridge(alpha=1.0)
    ridge.fit(train_embeds_norm, train_rul)
    test_pred = ridge.predict(test_embeds_norm)
    test_pred = np.clip(test_pred, 0, 1)

    rmse = np.sqrt(mean_squared_error(test_rul, test_pred))

    # Spearman between predicted RUL and time (file index)
    test_file_idx = all_file_idx[test_mask]
    # Higher file_idx = closer to failure = lower RUL
    spearman_with_time, _ = spearmanr(-test_file_idx, test_pred)  # Negative because RUL decreases

    # Constant baseline
    constant_pred = np.full_like(test_rul, test_rul.mean())
    constant_rmse = np.sqrt(mean_squared_error(test_rul, constant_pred))

    print(f"  RMSE: {rmse:.4f} (constant baseline: {constant_rmse:.4f})")
    print(f"  Spearman (pred vs time): {spearman_with_time:.4f}")

    # Zero-shot health indicator (Mahalanobis from healthy centroid)
    healthy_mask = train_mask & (all_file_idx <= int(0.25 * n_files))
    if healthy_mask.sum() > 10:
        healthy_embeds = all_embeds[healthy_mask]
        healthy_mean = healthy_embeds.mean(axis=0)
        healthy_cov = np.cov(healthy_embeds.T) + 1e-6 * np.eye(healthy_embeds.shape[1])

        # Use diagonal approximation for speed (full Mahalanobis is O(D^2))
        healthy_std = healthy_embeds.std(axis=0) + 1e-6

        # Normalized L2 distance from healthy centroid
        test_embeds_all = all_embeds
        distances = np.linalg.norm((test_embeds_all - healthy_mean) / healthy_std, axis=1)

        # Threshold: mean + 3*std of healthy distances
        healthy_dists = np.linalg.norm((healthy_embeds - healthy_mean) / healthy_std, axis=1)
        threshold = healthy_dists.mean() + 3 * healthy_dists.std()

        # Find when alarm first triggers (consistently)
        alarm_indices = np.where(distances > threshold)[0]
        if len(alarm_indices) > 0:
            first_alarm = alarm_indices[0]
            pct_remaining = 1.0 - first_alarm / len(distances)
            print(f"  Zero-shot early warning: {pct_remaining*100:.1f}% of run remaining")
        else:
            print(f"  Zero-shot: No alarm triggered (threshold={threshold:.2f})")
            pct_remaining = 0.0

        # Spearman of health indicator with time
        spearman_health, p_health = spearmanr(np.arange(len(distances)), distances)
        print(f"  Health indicator Spearman: {spearman_health:.4f} (p={p_health:.2e})")
    else:
        print(f"  Not enough healthy samples for Mahalanobis ({healthy_mask.sum()})")
        spearman_health = 0.0
        pct_remaining = 0.0

    return {
        'rmse': rmse,
        'constant_rmse': constant_rmse,
        'spearman_pred': spearman_with_time,
        'spearman_health': spearman_health,
        'early_warning_pct': pct_remaining,
        'beats_constant': rmse < constant_rmse,
    }


# =============================================================================
# RMS BASELINE FOR COMPARISON
# =============================================================================

def rms_baseline(dataset, n_train_frac=0.7):
    """Compute RMS health indicator as baseline."""
    all_rul = np.array([dataset[i][1].item() for i in range(len(dataset))])
    all_file_idx = np.array([dataset[i][2].item() for i in range(len(dataset))])

    n_files = all_file_idx.max() + 1
    split_file = int(n_train_frac * n_files)
    test_mask = all_file_idx > split_file

    # Compute per-window RMS per channel
    all_rms = []
    for i in range(len(dataset)):
        window = dataset[i][0].numpy()  # (C, L)
        rms = np.sqrt(np.mean(window ** 2, axis=1))  # (C,)
        all_rms.append(rms)
    all_rms = np.array(all_rms)  # (N, C)

    # Max RMS across channels as health indicator
    max_rms = all_rms.max(axis=1)

    # RMS Spearman with time
    test_file_idx = all_file_idx[test_mask]
    test_rms = max_rms[test_mask]
    spearman, pval = spearmanr(test_file_idx, test_rms)

    # RMS for RUL regression
    train_mask = all_file_idx <= split_file
    train_rms = all_rms[train_mask]
    train_rul = all_rul[train_mask]
    test_rms_all = all_rms[test_mask]
    test_rul = all_rul[test_mask]

    scaler = StandardScaler()
    train_rms_norm = scaler.fit_transform(train_rms)
    test_rms_norm = scaler.transform(test_rms_all)

    ridge = Ridge(alpha=1.0)
    ridge.fit(train_rms_norm, train_rul)
    pred = np.clip(ridge.predict(test_rms_norm), 0, 1)
    rmse = np.sqrt(mean_squared_error(test_rul, pred))

    constant_rmse = np.sqrt(mean_squared_error(test_rul, np.full_like(test_rul, test_rul.mean())))

    print(f"\nRMS Baseline:")
    print(f"  Spearman: {spearman:.4f} (p={pval:.2e})")
    print(f"  RMSE: {rmse:.4f} (constant: {constant_rmse:.4f})")

    return {
        'spearman': spearman,
        'rmse': rmse,
        'constant_rmse': constant_rmse,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ims-dir', type=str, default='data/bearings/ims_raw/1st_test/1st_test',
                        help='Path to IMS 1st_test directory')
    parser.add_argument('--ims-epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cwru-checkpoint', type=str,
                        default='checkpoints/jepa_v2_20260401_003619.pt',
                        help='CWRU-pretrained V2 checkpoint for comparison')
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()

    from datetime import datetime

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Check disk
    import shutil
    free_gb = shutil.disk_usage('/home/sagemaker-user').free / 1e9
    print(f"Home disk free: {free_gb:.1f} GB")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(project='mechanical-jepa', name=f'rul_ims_pretrained_s{args.seed}',
                   tags=['rul', 'ims-pretrain', 'round2'])

    # Check if IMS data exists
    ims_path = Path(args.ims_dir)
    if not ims_path.exists():
        # Try alternative paths
        for alt in [
            'data/bearings/ims_raw/1st_test',
            'data/bearings/ims/1st_test',
            '/home/sagemaker-user/ims/1st_test',
        ]:
            if Path(alt).exists():
                ims_path = Path(alt)
                args.ims_dir = str(alt)
                break
        else:
            print(f"IMS data not found at {args.ims_dir} or alternatives.")
            print("Searching for IMS data...")
            import subprocess
            result = subprocess.run(['find', '/home/sagemaker-user/IndustrialJEPA',
                                    '-name', '2003.10.22.12.06.24', '-type', 'f'],
                                   capture_output=True, text=True)
            if result.stdout:
                ims_path = Path(result.stdout.strip()).parent
                args.ims_dir = str(ims_path)
                print(f"Found IMS data at: {ims_path}")
            else:
                print("IMS data not found. Cannot run RUL experiments.")
                return

    print(f"\nUsing IMS data from: {args.ims_dir}")

    # Step 1: Load dataset for all evaluations
    print("\nLoading IMS dataset...")
    dataset = IMSWindowDataset(
        ims_dir=args.ims_dir,
        window_size=4096,
        stride=2048,
        n_channels=3,
        subsample=4,
        normalize=True,
    )

    if len(dataset) == 0:
        print("Empty dataset, aborting.")
        return

    # Step 2: RMS baseline
    print("\nEvaluating RMS baseline...")
    rms_results = rms_baseline(dataset)

    # Step 3: Pretrain JEPA on IMS
    print(f"\nPretraining JEPA on IMS ({args.ims_epochs} epochs)...")
    ims_model, _, _, _ = pretrain_jepa_on_ims(
        ims_dir=args.ims_dir,
        epochs=args.ims_epochs,
        seed=args.seed,
        device=device,
        save=True,
    )

    print("\nEvaluating IMS-pretrained JEPA for RUL...")
    ims_results = evaluate_rul(ims_model, dataset, device, method='jepa')

    # Step 4: CWRU-pretrained JEPA (the failed approach)
    cwru_results = None
    cwru_ckpt = Path(args.cwru_checkpoint)
    if cwru_ckpt.exists():
        print(f"\nLoading CWRU-pretrained model: {cwru_ckpt}")
        ckpt = torch.load(str(cwru_ckpt), map_location=device, weights_only=False)
        config = ckpt['config']
        cwru_model = MechanicalJEPAV2(
            n_channels=config['n_channels'],
            window_size=config['window_size'],
            patch_size=config.get('patch_size', 256),
            embed_dim=config['embed_dim'],
            encoder_depth=config['encoder_depth'],
            predictor_depth=config.get('predictor_depth', 4),
            n_heads=config.get('n_heads', 4),
            mask_ratio=config.get('mask_ratio', 0.625),
            predictor_pos=config.get('predictor_pos', 'sinusoidal'),
            loss_fn=config.get('loss_fn', 'l1'),
            var_reg_lambda=config.get('var_reg_lambda', 0.1),
        ).to(device)
        cwru_model.load_state_dict(ckpt['model_state_dict'])
        print("\nEvaluating CWRU-pretrained JEPA for RUL (should be poor)...")
        cwru_results = evaluate_rul(cwru_model, dataset, device, method='jepa')

    # Step 5: Random init
    print("\nEvaluating Random init JEPA for RUL...")
    torch.manual_seed(args.seed + 999)
    rand_model = MechanicalJEPAV2(
        n_channels=3, window_size=4096, patch_size=256, embed_dim=512,
        encoder_depth=4, predictor_depth=4, mask_ratio=0.625,
    ).to(device)
    rand_results = evaluate_rul(rand_model, dataset, device, method='random')

    # Summary
    print("\n" + "=" * 70)
    print("RUL COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Method':<35} {'RMSE':<10} {'Spearman':<12} {'Early Warn%':<15}")
    print("-" * 70)

    print(f"{'Constant baseline (predict mean)':<35} {rms_results['constant_rmse']:.4f}     {'N/A':<12} {'N/A'}")
    print(f"{'RMS hand-crafted':<35} {rms_results['rmse']:.4f}     {rms_results['spearman']:.4f}       {'N/A (RMS-based)'}")
    print(f"{'Random init JEPA':<35} {rand_results['rmse']:.4f}     {rand_results['spearman_pred']:.4f}       {rand_results['early_warning_pct']*100:.1f}%")
    if cwru_results:
        print(f"{'CWRU-pretrained JEPA (V2)':<35} {cwru_results['rmse']:.4f}     {cwru_results['spearman_pred']:.4f}       {cwru_results['early_warning_pct']*100:.1f}%")
    print(f"{'IMS-pretrained JEPA (NEW)':<35} {ims_results['rmse']:.4f}     {ims_results['spearman_pred']:.4f}       {ims_results['early_warning_pct']*100:.1f}%")
    print("-" * 70)

    if ims_results['rmse'] < rms_results['constant_rmse']:
        print("IMS-pretrained JEPA BEATS constant baseline on RUL!")
    else:
        print(f"IMS-pretrained JEPA FAILS to beat constant baseline (RMSE {ims_results['rmse']:.4f} vs {rms_results['constant_rmse']:.4f})")

    if use_wandb:
        wandb.log({
            'rul/constant_rmse': rms_results['constant_rmse'],
            'rul/rms_rmse': rms_results['rmse'],
            'rul/rms_spearman': rms_results['spearman'],
            'rul/rand_rmse': rand_results['rmse'],
            'rul/rand_spearman': rand_results['spearman_pred'],
            'rul/ims_pretrain_rmse': ims_results['rmse'],
            'rul/ims_pretrain_spearman': ims_results['spearman_pred'],
            'rul/ims_pretrain_early_warn': ims_results['early_warning_pct'],
            'rul/ims_health_spearman': ims_results['spearman_health'],
        })
        if cwru_results:
            wandb.log({
                'rul/cwru_rmse': cwru_results['rmse'],
                'rul/cwru_spearman': cwru_results['spearman_pred'],
                'rul/cwru_early_warn': cwru_results['early_warning_pct'],
            })
        wandb.finish()

    return {
        'rms': rms_results,
        'ims_pretrained': ims_results,
        'cwru_pretrained': cwru_results,
        'random': rand_results,
    }


if __name__ == '__main__':
    main()

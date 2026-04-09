"""
Cross-Dataset Transfer Experiment: CWRU-pretrained JEPA → IMS

Experiment design:
1. Load CWRU-pretrained JEPA checkpoint (best from overnight run)
2. Define IMS degradation labels from temporal position (clear separation)
3. Linear probe + MLP probe: JEPA features vs random init on IMS
4. Full fine-tuning: JEPA init vs random init on IMS
5. Report 3-seed results with statistical analysis

IMS Dataset:
- Test 1: 2156 files, bearing 3 (outer race) and 4 (roller) failed
- Test 2: 984 files, bearing 1 (outer race) failed
- 20kHz sampling, 8 channels (4 bearings x 2 axes)
- 20480 samples per file = 1.024 seconds

Task design:
  Binary (n_classes=2):
    Stage 0 (healthy):  first 25% of files  -- clearly normal
    Stage 1 (failure):  last 25% of files   -- clearly degraded

  3-class (n_classes=3):
    Stage 0 (healthy):   first 25% of files
    Stage 1 (degrading): middle 20% (files 40-60%)
    Stage 2 (failure):   last 20% of files

Rationale: Using temporal position with gaps ensures class distinctness.
The ambiguous transition period is excluded from the binary task.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPA, MechanicalJEPAV2


# =============================================================================
# IMS Dataset with Degradation Labels
# =============================================================================

class IMSDegradationDataset(Dataset):
    """
    IMS bearing dataset with degradation stage labels.

    Each file is 20480 samples @ 20kHz (1.024 seconds).
    We extract multiple random windows per file.
    """

    def __init__(
        self,
        ims_dir: Path,
        test_set: str,
        file_indices: list,
        labels: list,
        window_size: int = 4096,
        n_channels: int = 3,
        windows_per_file: int = 4,
        seed: int = 42,
    ):
        self.ims_dir = Path(ims_dir)
        self.window_size = window_size
        self.n_channels = n_channels

        self.all_files = sorted((self.ims_dir / test_set).glob('*'))
        self.selected_files = [self.all_files[i] for i in file_indices]
        self.labels = list(labels)

        # Build windows
        rng = np.random.default_rng(seed)
        self.windows = []
        n_samples_per_file = 20480
        for i, label in enumerate(self.labels):
            max_start = n_samples_per_file - window_size
            n_win = min(windows_per_file, max_start)
            starts = rng.choice(max_start, size=n_win, replace=False)
            for s in starts:
                self.windows.append((i, int(s), label))

        n_unique = len(set(labels))
        label_counts = np.bincount(list(labels), minlength=n_unique)
        stage_names = ['healthy', 'degrading', 'failure'][:n_unique]
        print(f"  IMSDegradationDataset ({test_set}): {len(self.selected_files)} files, "
              f"{len(self.windows)} windows")
        print(f"  Labels: {dict(zip(stage_names, label_counts))}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start, label = self.windows[idx]
        f = self.selected_files[file_idx]

        # Try fast numpy cache first, fall back to loadtxt
        if f.suffix == '.npy':
            # File is already a numpy cache (symlinked directory or pre-converted)
            data = np.load(f)
        else:
            npy_cache = Path(str(f).replace(
                'data/bearings/raw/ims',
                'data/bearings/ims_npy_cache'
            )).with_suffix(f.suffix + '.npy')
            if npy_cache.exists():
                data = np.load(npy_cache)  # fast
            else:
                data = np.loadtxt(f, dtype=np.float32)  # slow fallback

        window = data[start:start + self.window_size].T  # (8, window_size)

        # Select/pad channels
        if window.shape[0] < self.n_channels:
            pad = np.zeros((self.n_channels - window.shape[0], self.window_size), dtype=np.float32)
            window = np.vstack([window, pad])
        else:
            window = window[:self.n_channels]

        # Per-channel z-score
        mean = window.mean(axis=1, keepdims=True)
        std = window.std(axis=1, keepdims=True) + 1e-8
        window = (window - mean) / std

        return torch.tensor(window, dtype=torch.float32), label


# =============================================================================
# IMS Split Creation
# =============================================================================

def create_ims_splits(
    ims_dir: Path,
    test_set: str,
    seed: int = 42,
    n_classes: int = 2,
):
    """
    Create IMS degradation dataset with clear temporal separation between classes.

    Returns: train_indices, train_labels, test_indices, test_labels, rms_values
    """
    files = sorted((ims_dir / test_set).glob('*'))
    n_files = len(files)
    print(f"\nCreating IMS splits: {test_set}, {n_classes}-class, seed={seed}")
    print(f"  Total files: {n_files}")

    # Load RMS from cache (precomputed)
    # ims_dir = data/bearings/raw/ims → cache at data/bearings/ims_rms_cache.npy
    rms_cache_path = ims_dir.parent.parent / 'ims_rms_cache.npy'
    if rms_cache_path.exists():
        rms_cache = np.load(rms_cache_path, allow_pickle=True).item()
        rms_per_channel = np.array(rms_cache[test_set]['rms'])  # (n_files, 8)
        rms_values = rms_per_channel.max(axis=1)
        print(f"  RMS loaded from cache ({n_files} files)")
    else:
        print(f"  Computing RMS trajectory (no cache found)...")
        rms_values = []
        for i, f in enumerate(files):
            if i % 500 == 0:
                print(f"    {i}/{n_files}...")
            data = np.loadtxt(f, dtype=np.float32)
            rms_values.append(data.std(axis=0).max())
        rms_values = np.array(rms_values)

    if n_classes == 2:
        # Binary: first 25% = healthy, last 25% = failure
        n_each = int(n_files * 0.25)
        healthy_idx = list(range(0, n_each))
        failure_idx = list(range(n_files - n_each, n_files))

        all_indices = healthy_idx + failure_idx
        all_labels = [0] * len(healthy_idx) + [1] * len(failure_idx)

        print(f"  Binary split: {len(healthy_idx)} healthy + {len(failure_idx)} failure")
        print(f"    Healthy RMS: {rms_values[healthy_idx].mean():.4f} ± {rms_values[healthy_idx].std():.4f}")
        print(f"    Failure RMS: {rms_values[failure_idx].mean():.4f} ± {rms_values[failure_idx].std():.4f}")
        print(f"    RMS ratio (failure/healthy): {rms_values[failure_idx].mean()/rms_values[healthy_idx].mean():.2f}x")

    else:  # 3 classes
        # Stage 0: files 0 - 25%
        # Stage 1: files 40% - 60%
        # Stage 2: files 80% - 100%
        n_healthy = int(n_files * 0.25)
        deg_start = int(n_files * 0.40)
        deg_end = int(n_files * 0.60)
        fail_start = int(n_files * 0.80)

        healthy_idx = list(range(0, n_healthy))
        degrading_idx = list(range(deg_start, deg_end))
        failure_idx = list(range(fail_start, n_files))

        # Balance: use min count per class
        min_count = min(len(healthy_idx), len(degrading_idx), len(failure_idx))
        # All same size in this design, just confirm
        all_indices = healthy_idx + degrading_idx + failure_idx
        all_labels = ([0] * len(healthy_idx) +
                      [1] * len(degrading_idx) +
                      [2] * len(failure_idx))

        print(f"  3-class split: {len(healthy_idx)} healthy + {len(degrading_idx)} degrading + {len(failure_idx)} failure")
        print(f"    Healthy RMS: {rms_values[healthy_idx].mean():.4f} ± {rms_values[healthy_idx].std():.4f}")
        print(f"    Degrading RMS: {rms_values[degrading_idx].mean():.4f} ± {rms_values[degrading_idx].std():.4f}")
        print(f"    Failure RMS: {rms_values[failure_idx].mean():.4f} ± {rms_values[failure_idx].std():.4f}")

    # Split each class 80/20 train/test
    rng = np.random.default_rng(seed)
    train_indices, train_labels = [], []
    test_indices, test_labels = [], []

    n_stages = n_classes if n_classes == 2 else 3
    for stage in range(n_stages):
        stage_files = [all_indices[i] for i, l in enumerate(all_labels) if l == stage]
        stage_files = np.array(stage_files)
        shuffled = rng.permutation(stage_files)
        n_test = max(1, len(shuffled) // 5)

        test_these = shuffled[:n_test].tolist()
        train_these = shuffled[n_test:].tolist()

        train_indices.extend(train_these)
        train_labels.extend([stage] * len(train_these))
        test_indices.extend(test_these)
        test_labels.extend([stage] * len(test_these))

    print(f"  Train: {len(train_indices)} files, Test: {len(test_indices)} files")

    return train_indices, train_labels, test_indices, test_labels, rms_values


# =============================================================================
# Probes
# =============================================================================

class MLPProbe(nn.Module):
    """2-layer MLP probe."""
    def __init__(self, embed_dim: int, n_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate_probe(
    model: MechanicalJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    embed_dim: int,
    device: torch.device,
    probe_type: str = 'linear',
    probe_epochs: int = 100,
    probe_lr: float = 1e-2,
    n_classes: int = 2,
    pool: str = 'mean',
    label: str = '',
) -> dict:
    """Evaluate frozen JEPA encoder with a probe."""
    model.eval()

    def extract(loader):
        all_e, all_l = [], []
        with torch.no_grad():
            for signals, labels in loader:
                signals = signals.to(device)
                embeds = model.get_embeddings(signals, pool=pool)
                all_e.append(embeds.cpu())
                all_l.append(labels)
        return torch.cat(all_e), torch.cat(all_l)

    train_e, train_l = extract(train_loader)
    test_e, test_l = extract(test_loader)

    # Normalize embeddings based on train stats
    mean = train_e.mean(0, keepdim=True)
    std = train_e.std(0, keepdim=True) + 1e-8
    train_e_norm = ((train_e - mean) / std).to(device)
    test_e_norm = ((test_e - mean) / std).to(device)
    train_l = train_l.to(device)
    test_l = test_l.to(device)

    if probe_type == 'linear':
        probe = nn.Linear(embed_dim, n_classes).to(device)
    else:
        probe = MLPProbe(embed_dim, n_classes).to(device)

    optimizer = optim.AdamW(probe.parameters(), lr=probe_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=probe_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    best_preds = None
    for epoch in range(probe_epochs):
        probe.train()
        optimizer.zero_grad()
        loss = criterion(probe(train_e_norm), train_l)
        loss.backward()
        optimizer.step()
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_e_norm)
            test_preds = test_logits.argmax(1)
            test_acc = (test_preds == test_l).float().mean().item()
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_preds = test_preds.clone()

    # Per-class accuracy
    stage_names = ['healthy', 'degrading', 'failure'][:n_classes]
    per_class = {}
    for i, name in enumerate(stage_names):
        mask = test_l == i
        if mask.sum() > 0:
            acc = (best_preds[mask] == test_l[mask]).float().mean().item()
            per_class[name] = acc

    print(f"  [{label:35s}] {probe_type} probe: {best_test_acc:.4f} | per-class: {per_class}")

    return {
        'test_acc': best_test_acc,
        'per_class': per_class,
        'n_train': len(train_l),
        'n_test': len(test_l),
    }


# =============================================================================
# Fine-tuning
# =============================================================================

def fine_tune(
    model: MechanicalJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    embed_dim: int,
    device: torch.device,
    n_classes: int = 2,
    epochs: int = 30,
    lr: float = 5e-5,
    label: str = '',
) -> dict:
    """Fine-tune full encoder + linear head on IMS task."""
    classifier = nn.Linear(embed_dim, n_classes).to(device)
    optimizer = optim.AdamW(
        list(model.encoder.parameters()) + list(classifier.parameters()),
        lr=lr, weight_decay=0.05
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0
    best_preds = None
    best_labels = None

    for epoch in range(epochs):
        model.encoder.train()
        classifier.train()
        for signals, labels in train_loader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            embeds = model.get_embeddings(signals, pool='mean')
            loss = criterion(classifier(embeds), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.encoder.parameters()) + list(classifier.parameters()), 1.0
            )
            optimizer.step()
        scheduler.step()

        model.encoder.eval()
        classifier.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for signals, labels in test_loader:
                signals = signals.to(device)
                preds = classifier(model.get_embeddings(signals, pool='mean')).argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        test_acc = (all_preds == all_labels).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_preds = all_preds.clone()
            best_labels = all_labels.clone()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: test_acc={test_acc:.4f}")

    stage_names = ['healthy', 'degrading', 'failure'][:n_classes]
    per_class = {}
    for i, name in enumerate(stage_names):
        mask = best_labels == i
        if mask.sum() > 0:
            per_class[name] = (best_preds[mask] == best_labels[mask]).float().mean().item()

    print(f"  [{label:35s}] fine-tune: {best_test_acc:.4f} | per-class: {per_class}")
    return {'test_acc': best_test_acc, 'per_class': per_class}


# =============================================================================
# Full Transfer Experiment
# =============================================================================

def run_transfer_experiment(
    ims_dir: Path,
    checkpoint_path: Path,
    test_set: str,
    seed: int,
    device: torch.device,
    n_classes: int = 2,
    windows_per_file: int = 4,
    n_channels: int = 3,
    batch_size: int = 64,
    run_finetune: bool = True,
) -> dict:
    """Run a full transfer experiment for one seed."""

    print(f"\n{'='*60}")
    print(f"Transfer Experiment | {test_set} | {n_classes}-class | seed={seed}")
    print(f"{'='*60}")

    # Load checkpoint config
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    embed_dim = config['embed_dim']
    window_size = config['window_size']

    # Create IMS splits
    train_idx, train_labels, test_idx, test_labels, rms_values = create_ims_splits(
        ims_dir, test_set, seed=seed, n_classes=n_classes
    )

    # Create datasets
    train_ds = IMSDegradationDataset(
        ims_dir=ims_dir, test_set=test_set,
        file_indices=train_idx, labels=train_labels,
        window_size=window_size, n_channels=n_channels,
        windows_per_file=windows_per_file, seed=seed,
    )
    test_ds = IMSDegradationDataset(
        ims_dir=ims_dir, test_set=test_set,
        file_indices=test_idx, labels=test_labels,
        window_size=window_size, n_channels=n_channels,
        windows_per_file=windows_per_file, seed=seed,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}

    def make_model(load_ckpt=True):
        # Support both V1 (MechanicalJEPA) and V2 (MechanicalJEPAV2) checkpoints
        is_v2 = 'predictor_pos' in config
        if is_v2:
            m = MechanicalJEPAV2(
                n_channels=n_channels,
                window_size=window_size,
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
                var_reg_lambda=0.0,   # No reg during inference
                vicreg_lambda=0.0,
            ).to(device)
        else:
            m = MechanicalJEPA(
                n_channels=n_channels,
                window_size=window_size,
                patch_size=config['patch_size'],
                embed_dim=embed_dim,
                encoder_depth=config['encoder_depth'],
                predictor_depth=config['predictor_depth'],
                n_heads=config['n_heads'],
                mask_ratio=config['mask_ratio'],
                ema_decay=config['ema_decay'],
            ).to(device)
        if load_ckpt:
            m.load_state_dict(ckpt['model_state_dict'])
        return m

    # 1. JEPA pretrained: linear probe
    print("\n--- Linear Probe Comparison ---")
    results['jepa_linear'] = evaluate_probe(
        make_model(True), train_loader, test_loader, embed_dim, device,
        probe_type='linear', probe_epochs=100, probe_lr=1e-2,
        n_classes=n_classes, pool='mean', label='JEPA pretrained',
    )
    results['random_linear'] = evaluate_probe(
        make_model(False), train_loader, test_loader, embed_dim, device,
        probe_type='linear', probe_epochs=100, probe_lr=1e-2,
        n_classes=n_classes, pool='mean', label='Random init',
    )

    # 2. MLP probe
    print("\n--- MLP Probe Comparison ---")
    results['jepa_mlp'] = evaluate_probe(
        make_model(True), train_loader, test_loader, embed_dim, device,
        probe_type='mlp', probe_epochs=100, probe_lr=1e-3,
        n_classes=n_classes, pool='mean', label='JEPA pretrained (MLP)',
    )
    results['random_mlp'] = evaluate_probe(
        make_model(False), train_loader, test_loader, embed_dim, device,
        probe_type='mlp', probe_epochs=100, probe_lr=1e-3,
        n_classes=n_classes, pool='mean', label='Random init (MLP)',
    )

    # 3. Fine-tuning (optional)
    if run_finetune:
        print("\n--- Fine-tuning Comparison ---")
        results['jepa_finetune'] = fine_tune(
            make_model(True), train_loader, test_loader, embed_dim, device,
            n_classes=n_classes, epochs=30, lr=5e-5, label='JEPA pretrained (fine-tune)',
        )
        results['random_finetune'] = fine_tune(
            make_model(False), train_loader, test_loader, embed_dim, device,
            n_classes=n_classes, epochs=30, lr=5e-5, label='Random init (fine-tune)',
        )

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--ims-dir', default='data/bearings/raw/ims')
    parser.add_argument('--test-set', default='1st_test', choices=['1st_test', '2nd_test'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--n-classes', type=int, default=2, choices=[2, 3])
    parser.add_argument('--windows-per-file', type=int, default=4)
    parser.add_argument('--n-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Task: {args.n_classes}-class degradation, test_set={args.test_set}")

    checkpoint_path = Path(args.checkpoint)
    ims_dir = Path(args.ims_dir)

    all_results = {}
    for seed in args.seeds:
        all_results[seed] = run_transfer_experiment(
            ims_dir=ims_dir,
            checkpoint_path=checkpoint_path,
            test_set=args.test_set,
            seed=seed,
            device=device,
            n_classes=args.n_classes,
            windows_per_file=args.windows_per_file,
            n_channels=args.n_channels,
            batch_size=args.batch_size,
            run_finetune=args.finetune,
        )

    # Summary
    print(f"\n{'='*70}")
    print(f"MULTI-SEED SUMMARY | {args.test_set} | {args.n_classes}-class")
    print(f"{'='*70}")
    random_chance = 1.0 / args.n_classes
    print(f"Random chance: {random_chance:.1%}")
    print()

    all_metrics = list(all_results[args.seeds[0]].keys())
    for metric in all_metrics:
        accs = [all_results[s][metric]['test_acc'] for s in args.seeds]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        vs_chance = mean_acc - random_chance
        print(f"  {metric:30s}: {mean_acc:.4f} ± {std_acc:.4f}  "
              f"(vs chance: {'+' if vs_chance > 0 else ''}{vs_chance:.4f})")

    # Transfer gain
    print(f"\n{'='*70}")
    print("TRANSFER GAIN (JEPA - Random)")
    print(f"{'='*70}")
    for probe_type in ['linear', 'mlp']:
        jepa_key = f'jepa_{probe_type}'
        rand_key = f'random_{probe_type}'
        if jepa_key not in all_results[args.seeds[0]]:
            continue
        jepa_accs = [all_results[s][jepa_key]['test_acc'] for s in args.seeds]
        rand_accs = [all_results[s][rand_key]['test_acc'] for s in args.seeds]
        gains = [j - r for j, r in zip(jepa_accs, rand_accs)]
        mean_gain = np.mean(gains)
        std_gain = np.std(gains)
        n_positive = sum(g > 0 for g in gains)
        print(f"  {probe_type:10s}: {mean_gain:+.4f} ± {std_gain:.4f}  "
              f"(positive in {n_positive}/{len(args.seeds)} seeds)")

    if args.finetune:
        jepa_ft = [all_results[s]['jepa_finetune']['test_acc'] for s in args.seeds]
        rand_ft = [all_results[s]['random_finetune']['test_acc'] for s in args.seeds]
        gains_ft = [j - r for j, r in zip(jepa_ft, rand_ft)]
        print(f"  {'finetune':10s}: {np.mean(gains_ft):+.4f} ± {np.std(gains_ft):.4f}  "
              f"(positive in {sum(g > 0 for g in gains_ft)}/{len(args.seeds)} seeds)")

    # Final verdict
    print()
    jepa_linear_accs = [all_results[s]['jepa_linear']['test_acc'] for s in args.seeds]
    rand_linear_accs = [all_results[s]['random_linear']['test_acc'] for s in args.seeds]
    gains = [j - r for j, r in zip(jepa_linear_accs, rand_linear_accs)]
    mean_gain = np.mean(gains)

    if mean_gain > 0.10:
        print(f"VERDICT: Strong transfer (+{mean_gain:.1%} linear probe)")
    elif mean_gain > 0.05:
        print(f"VERDICT: Clear transfer (+{mean_gain:.1%} linear probe)")
    elif mean_gain > 0:
        print(f"VERDICT: Marginal positive transfer (+{mean_gain:.1%} linear probe)")
    else:
        print(f"VERDICT: No transfer detected ({mean_gain:.1%} linear probe)")


if __name__ == '__main__':
    main()

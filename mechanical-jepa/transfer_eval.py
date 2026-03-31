"""
Cross-Dataset Transfer Evaluation for Mechanical-JEPA.

Tests TRUE transferability by:
1. Pretraining JEPA on CWRU dataset
2. Evaluating on IMS dataset (different test rig, different bearings)

Two evaluation modes:
A. Degradation Detection: Use temporal position as pseudo-label
   - Early IMS files (first 25%) = "healthy"
   - Late IMS files (last 25%) = "degraded"
   - Test if CWRU-pretrained encoder can detect degradation

B. Embedding Analysis: Visualize IMS embeddings from CWRU-pretrained encoder
   - Extract embeddings from IMS data
   - t-SNE visualization colored by temporal position
   - Check if temporal progression is visible in embedding space
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.data import BearingDataset, create_dataloaders
from src.models import MechanicalJEPA


class LinearProbe(nn.Module):
    """Simple linear classifier for evaluation."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def create_ims_degradation_loaders(
    data_dir: Path,
    test_set: str = '1st_test',
    early_pct: float = 0.25,
    late_pct: float = 0.25,
    batch_size: int = 32,
    window_size: int = 4096,
    stride: int = 2048,
    n_channels: int = 3,
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train/test loaders for IMS with degradation pseudo-labels.

    Args:
        data_dir: Path to data/bearings
        test_set: Which IMS test set to use ('1st_test', '2nd_test', '3rd_test')
        early_pct: Percentage of early files to label as "healthy"
        late_pct: Percentage of late files to label as "degraded"

    Returns:
        train_loader, test_loader, info_dict
    """
    data_dir = Path(data_dir)

    # Load episodes and filter to IMS test set
    episodes_df = pd.read_parquet(data_dir / 'bearing_episodes.parquet')
    ims_episodes = episodes_df[
        (episodes_df['dataset'] == 'ims') &
        (episodes_df['test_set'] == test_set)
    ].copy()

    # Sort by measurement_id (temporal order)
    ims_episodes = ims_episodes.sort_values('measurement_id').reset_index(drop=True)

    n_total = len(ims_episodes)
    n_early = int(n_total * early_pct)
    n_late = int(n_total * late_pct)

    # Create pseudo-labels: 0=healthy (early), 1=degraded (late)
    early_episodes = ims_episodes.iloc[:n_early].copy()
    late_episodes = ims_episodes.iloc[-n_late:].copy()

    early_episodes['fault_label'] = 0  # healthy
    late_episodes['fault_label'] = 1   # degraded

    # Split each into train/test (80/20)
    n_early_train = int(n_early * 0.8)
    n_late_train = int(n_late * 0.8)

    train_episodes = pd.concat([
        early_episodes.iloc[:n_early_train],
        late_episodes.iloc[:n_late_train]
    ])

    test_episodes = pd.concat([
        early_episodes.iloc[n_early_train:],
        late_episodes.iloc[n_late_train:]
    ])

    print(f"\n{'='*60}")
    print(f"IMS Degradation Detection Setup ({test_set})")
    print(f"{'='*60}")
    print(f"Total IMS episodes: {n_total}")
    print(f"Early (healthy): {n_early} episodes")
    print(f"Late (degraded): {n_late} episodes")
    print(f"\nTrain: {len(train_episodes)} episodes")
    print(f"Test: {len(test_episodes)} episodes")

    # Create datasets
    train_dataset = BearingDataset(
        data_dir=data_dir,
        bearing_ids=train_episodes['bearing_id'].tolist(),
        episodes_df=train_episodes,
        window_size=window_size,
        stride=stride,
        n_channels=n_channels,
    )

    test_dataset = BearingDataset(
        data_dir=data_dir,
        bearing_ids=test_episodes['bearing_id'].tolist(),
        episodes_df=test_episodes,
        window_size=window_size,
        stride=stride,
        n_channels=n_channels,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    info = {
        'test_set': test_set,
        'n_total': n_total,
        'n_early': n_early,
        'n_late': n_late,
        'train_windows': len(train_dataset),
        'test_windows': len(test_dataset),
    }

    return train_loader, test_loader, info


def extract_embeddings(
    model: MechanicalJEPA,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Extract embeddings from data loader."""
    model.eval()
    all_embeds = []
    all_labels = []
    all_bearing_ids = []

    with torch.no_grad():
        for signals, labels, bearing_ids in loader:
            signals = signals.to(device)
            embeds = model.get_embeddings(signals)
            all_embeds.append(embeds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_bearing_ids.extend(bearing_ids)

    embeds = np.concatenate(all_embeds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return embeds, labels, all_bearing_ids


def train_linear_probe_transfer(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_embeddings: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int = 2,
    epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = torch.device('cpu'),
) -> Dict:
    """
    Train linear probe on frozen embeddings.

    Returns dict with train_acc, test_acc.
    """
    embed_dim = embeddings.shape[1]

    # Convert to tensors
    train_embeds = torch.tensor(embeddings, dtype=torch.float32).to(device)
    train_labels = torch.tensor(labels, dtype=torch.long).to(device)
    test_embeds = torch.tensor(test_embeddings, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    # Create probe
    probe = LinearProbe(embed_dim, n_classes).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Train
    best_test_acc = 0
    for epoch in range(epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(train_embeds)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            train_logits = probe(train_embeds)
            train_preds = train_logits.argmax(dim=1)
            train_acc = (train_preds == train_labels).float().mean().item()

            test_logits = probe(test_embeds)
            test_preds = test_logits.argmax(dim=1)
            test_acc = (test_preds == test_labels).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

    return {
        'train_acc': train_acc,
        'test_acc': best_test_acc,
    }


def evaluate_transfer(
    checkpoint_path: Path,
    data_dir: Path,
    mode: str = 'degradation',
    device: torch.device = None,
) -> Dict:
    """
    Evaluate cross-dataset transfer from CWRU to IMS.

    Args:
        checkpoint_path: Path to CWRU-pretrained JEPA checkpoint
        data_dir: Path to data/bearings
        mode: 'degradation' or 'embedding'

    Returns:
        Results dict
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"CROSS-DATASET TRANSFER EVALUATION")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']

    # Create model
    model = MechanicalJEPA(
        n_channels=config['n_channels'],
        window_size=config['window_size'],
        patch_size=config['patch_size'],
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config['predictor_depth'],
        n_heads=config['n_heads'],
        mask_ratio=config['mask_ratio'],
        ema_decay=config['ema_decay'],
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded (trained on CWRU)")
    print(f"  Embed dim: {config['embed_dim']}")
    print(f"  Encoder depth: {config['encoder_depth']}")

    if mode == 'degradation':
        # Mode A: Degradation detection
        print("\n" + "="*60)
        print("MODE A: Degradation Detection")
        print("="*60)

        # Create IMS loaders with pseudo-labels
        train_loader, test_loader, info = create_ims_degradation_loaders(
            data_dir=data_dir,
            test_set='1st_test',
            batch_size=config['batch_size'],
            window_size=config['window_size'],
            stride=config['stride'],
            n_channels=config['n_channels'],
        )

        # Extract embeddings
        print("\nExtracting embeddings from CWRU-pretrained encoder...")
        train_embeds, train_labels, _ = extract_embeddings(model, train_loader, device)
        test_embeds, test_labels, _ = extract_embeddings(model, test_loader, device)

        print(f"Train embeddings: {train_embeds.shape}")
        print(f"Test embeddings: {test_embeds.shape}")

        # Train linear probe on IMS
        print("\nTraining linear probe on IMS embeddings...")
        results = train_linear_probe_transfer(
            train_embeds, train_labels,
            test_embeds, test_labels,
            n_classes=2,
            epochs=100,
            device=device,
        )

        print(f"\n{'='*60}")
        print(f"RESULTS: CWRU -> IMS Transfer")
        print(f"{'='*60}")
        print(f"Train accuracy: {results['train_acc']:.4f}")
        print(f"Test accuracy: {results['test_acc']:.4f}")
        print(f"Random baseline: 0.5000 (2-class)")

        # Success criteria
        random_baseline = 0.5
        if results['test_acc'] > random_baseline + 0.05:
            print(f"\nTRANSFER SUCCESS: {results['test_acc']:.1%} > {random_baseline+0.05:.1%}")
        else:
            print(f"\nTRANSFER FAILED: {results['test_acc']:.1%} <= {random_baseline+0.05:.1%}")

        results['mode'] = 'degradation'
        results['info'] = info

    elif mode == 'embedding':
        # Mode B: Embedding visualization
        print("\n" + "="*60)
        print("MODE B: Embedding Analysis")
        print("="*60)

        # Load IMS data (full 1st_test)
        episodes_df = pd.read_parquet(data_dir / 'bearing_episodes.parquet')
        ims_episodes = episodes_df[
            (episodes_df['dataset'] == 'ims') &
            (episodes_df['test_set'] == '1st_test')
        ].copy()
        ims_episodes = ims_episodes.sort_values('measurement_id').reset_index(drop=True)

        # Sample uniformly across time (to reduce computation)
        n_sample = min(500, len(ims_episodes))
        sample_indices = np.linspace(0, len(ims_episodes)-1, n_sample, dtype=int)
        ims_sample = ims_episodes.iloc[sample_indices].copy()

        # Create temporal labels (0=early, 1=mid, 2=late)
        ims_sample['temporal_label'] = pd.cut(
            range(len(ims_sample)),
            bins=3,
            labels=[0, 1, 2]
        ).astype(int)

        print(f"Sampling {n_sample} episodes uniformly across time")

        # Create dataset
        dataset = BearingDataset(
            data_dir=data_dir,
            bearing_ids=ims_sample['bearing_id'].tolist(),
            episodes_df=ims_sample,
            window_size=config['window_size'],
            stride=config['stride'],
            n_channels=config['n_channels'],
        )

        loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True,
        )

        # Extract embeddings
        print("Extracting embeddings...")
        embeds, labels, _ = extract_embeddings(model, loader, device)

        print(f"Embeddings: {embeds.shape}")

        # t-SNE visualization
        print("Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeds_2d = tsne.fit_transform(embeds)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            embeds_2d[:, 0],
            embeds_2d[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            s=10,
        )
        ax.set_title('IMS Embeddings (CWRU-pretrained encoder)\nColored by temporal position')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (0=early, 1=mid, 2=late)')

        # Save plot
        save_path = Path('results') / f'ims_transfer_tsne_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        save_path.parent.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved t-SNE plot to {save_path}")

        results = {
            'mode': 'embedding',
            'n_samples': n_sample,
            'tsne_plot': str(save_path),
        }

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Cross-dataset transfer evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to CWRU checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/bearings', help='Data directory')
    parser.add_argument('--mode', type=str, default='degradation',
                       choices=['degradation', 'embedding'],
                       help='Evaluation mode')

    args = parser.parse_args()

    results = evaluate_transfer(
        checkpoint_path=Path(args.checkpoint),
        data_dir=Path(args.data_dir),
        mode=args.mode,
    )

    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = results_dir / f'transfer_results_{timestamp}.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

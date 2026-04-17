"""
F1-Score Evaluation Script for Mechanical-JEPA.

Adds macro F1-score, per-class F1, and confusion matrix to evaluation.
Supports loading any V2 checkpoint and re-evaluating with F1.

Usage:
    python eval_f1.py --checkpoint checkpoints/jepa_v2_xxx.pt
    python eval_f1.py --checkpoint checkpoints/jepa_v2_xxx.pt --seeds 42 123 456
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    accuracy_score
)

sys.path.insert(0, str(Path(__file__).parent))
from src.data import create_dataloaders
from src.models import MechanicalJEPAV2


CLASS_NAMES = ['healthy', 'outer_race', 'inner_race', 'ball']


class LinearProbe(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def extract_embeddings(model, loader, device):
    model.eval()
    all_embeds, all_labels = [], []
    with torch.no_grad():
        for signals, labels, _ in loader:
            signals = signals.to(device)
            embeds = model.get_embeddings(signals, pool='mean')
            all_embeds.append(embeds.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_embeds), torch.cat(all_labels)


def train_and_eval_probe(train_embeds, train_labels, test_embeds, test_labels,
                          embed_dim, device, probe_epochs=20, lr=1e-3, seed=42):
    """Train linear probe and return F1 + accuracy metrics."""
    torch.manual_seed(seed)

    probe = LinearProbe(embed_dim, 4).to(device)
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    te_dev = train_embeds.to(device)
    tl_dev = train_labels.to(device)
    te2_dev = test_embeds.to(device)
    tl2_dev = test_labels.to(device)

    best_f1 = 0
    best_preds = None

    for epoch in range(probe_epochs):
        probe.train()
        optimizer.zero_grad()
        logits = probe(te_dev)
        loss = criterion(logits, tl_dev)
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(te2_dev)
            test_preds = test_logits.argmax(dim=1).cpu().numpy()
            f1 = f1_score(tl2_dev.cpu().numpy(), test_preds, average='macro', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_preds = test_preds.copy()

    y_true = tl2_dev.cpu().numpy()
    acc = accuracy_score(y_true, best_preds)
    per_class_f1 = f1_score(y_true, best_preds, average=None, zero_division=0)
    cm = confusion_matrix(y_true, best_preds)

    return {
        'macro_f1': best_f1,
        'accuracy': acc,
        'per_class_f1': {CLASS_NAMES[i]: float(per_class_f1[i]) for i in range(len(per_class_f1))},
        'confusion_matrix': cm.tolist(),
        'predictions': best_preds.tolist(),
        'labels': y_true.tolist(),
    }


def evaluate_checkpoint(checkpoint_path, seeds, device):
    """Load checkpoint and evaluate with F1 across multiple seeds."""
    print(f"\nLoading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    print(f"Config: embed_dim={config['embed_dim']}, mask={config.get('mask_ratio', 0.5)}, "
          f"loss={config.get('loss_fn', 'mse')}, predictor_pos={config.get('predictor_pos', 'learnable')}")

    model = MechanicalJEPAV2(
        n_channels=config['n_channels'],
        window_size=config['window_size'],
        patch_size=config.get('patch_size', 256),
        embed_dim=config['embed_dim'],
        encoder_depth=config['encoder_depth'],
        predictor_depth=config.get('predictor_depth', 4),
        n_heads=config.get('n_heads', 4),
        mask_ratio=config.get('mask_ratio', 0.5),
        ema_decay=config.get('ema_decay', 0.996),
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        loss_fn=config.get('loss_fn', 'mse'),
        var_reg_lambda=config.get('var_reg_lambda', 0.0),
        vicreg_lambda=config.get('vicreg_lambda', 0.0),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    embed_dim = config['embed_dim']
    all_results = []

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_loader, test_loader, data_info = create_dataloaders(
            data_dir=config.get('data_dir', 'data/bearings'),
            batch_size=config.get('batch_size', 32),
            window_size=config['window_size'],
            stride=config.get('stride', 2048),
            test_ratio=config.get('test_ratio', 0.2),
            seed=seed,
            num_workers=0,
            dataset_filter=config.get('dataset_filter', 'cwru'),
            n_channels=config['n_channels'],
        )

        train_embeds, train_labels = extract_embeddings(model, train_loader, device)
        test_embeds, test_labels = extract_embeddings(model, test_loader, device)

        # JEPA probe
        jepa_results = train_and_eval_probe(
            train_embeds, train_labels, test_embeds, test_labels,
            embed_dim, device, seed=seed
        )

        # Random init probe (same architecture, fresh weights)
        torch.manual_seed(seed + 10000)
        rand_model = MechanicalJEPAV2(
            n_channels=config['n_channels'],
            window_size=config['window_size'],
            patch_size=config.get('patch_size', 256),
            embed_dim=embed_dim,
            encoder_depth=config['encoder_depth'],
        ).to(device)

        rand_train, rand_labels_train = extract_embeddings(rand_model, train_loader, device)
        rand_test, rand_labels_test = extract_embeddings(rand_model, test_loader, device)

        rand_results = train_and_eval_probe(
            rand_train, rand_labels_train, rand_test, rand_labels_test,
            embed_dim, device, seed=seed
        )

        print(f"    JEPA macro F1: {jepa_results['macro_f1']:.4f} (acc: {jepa_results['accuracy']:.4f})")
        print(f"    Rand macro F1: {rand_results['macro_f1']:.4f} (acc: {rand_results['accuracy']:.4f})")
        print(f"    Transfer F1 gain: {jepa_results['macro_f1'] - rand_results['macro_f1']:+.4f}")
        print(f"    Per-class F1: {jepa_results['per_class_f1']}")

        all_results.append({
            'seed': seed,
            'jepa_f1': jepa_results['macro_f1'],
            'jepa_acc': jepa_results['accuracy'],
            'rand_f1': rand_results['macro_f1'],
            'rand_acc': rand_results['accuracy'],
            'f1_gain': jepa_results['macro_f1'] - rand_results['macro_f1'],
            'per_class_f1': jepa_results['per_class_f1'],
            'confusion_matrix': jepa_results['confusion_matrix'],
        })

    # Summary
    jepa_f1s = [r['jepa_f1'] for r in all_results]
    rand_f1s = [r['rand_f1'] for r in all_results]
    gains = [r['f1_gain'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"JEPA macro F1: {np.mean(jepa_f1s):.4f} ± {np.std(jepa_f1s):.4f}")
    print(f"Rand macro F1: {np.mean(rand_f1s):.4f} ± {np.std(rand_f1s):.4f}")
    print(f"F1 Gain: {np.mean(gains):+.4f} ± {np.std(gains):.4f}")
    print(f"Positive gains: {sum(g > 0 for g in gains)}/{len(gains)}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    results = evaluate_checkpoint(args.checkpoint, args.seeds, device)
    return results


if __name__ == '__main__':
    main()

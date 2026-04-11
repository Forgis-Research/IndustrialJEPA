#!/usr/bin/env python
"""
Linear Probe Analysis for Cross-Machine Transfer.

Tests whether representations learned on source domain can
linearly separate anomalies on target domain.

This is a rapid validation: if linear probe fails, deep transfer unlikely to work.

Output:
- Probe accuracy and AUC scores
- t-SNE visualization of embeddings
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.manifold import TSNE
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


class SimpleEncoder(nn.Module):
    """Simple encoder for baseline feature extraction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, x):
        # x: (batch, time, channels)
        # Pool over time
        x = x.mean(dim=1)  # (batch, channels)
        return self.net(x)


def extract_features(dataset, encoder: nn.Module, n_samples: int = None, device: str = 'cpu'):
    """Extract features and labels from dataset."""
    encoder.eval()
    encoder.to(device)

    features = []
    labels = []

    n = min(n_samples or len(dataset), len(dataset))

    with torch.no_grad():
        for i in range(n):
            sample = dataset[i]

            # Combine setpoint and effort
            setpoint = sample['setpoint'].unsqueeze(0).to(device)
            effort = sample['effort'].unsqueeze(0).to(device)
            x = torch.cat([setpoint, effort], dim=-1)

            feat = encoder(x).cpu().numpy()
            features.append(feat)

            # Get label
            label = 1 if sample.get('label', 'normal') != 'normal' else 0
            labels.append(label)

    return np.vstack(features), np.array(labels)


def train_encoder(dataset, epochs: int = 5, device: str = 'cpu'):
    """Train simple encoder on source domain (self-supervised)."""
    from torch.utils.data import DataLoader

    # Get input dimension
    sample = dataset[0]
    input_dim = sample['setpoint'].shape[-1] + sample['effort'].shape[-1]

    encoder = SimpleEncoder(input_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)

    # Simple reconstruction objective
    decoder = nn.Linear(64, input_dim).to(device)

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    encoder.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0

        for batch in loader:
            setpoint = batch['setpoint'].to(device)
            effort = batch['effort'].to(device)
            x = torch.cat([setpoint, effort], dim=-1)

            # Pool over time
            x_pooled = x.mean(dim=1)

            # Encode
            z = encoder.net(x_pooled)

            # Decode (simple reconstruction)
            x_recon = decoder(z)

            # Loss
            loss = nn.functional.mse_loss(x_recon, x_pooled)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:  # Quick training
                break

        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return encoder


def plot_tsne(features_source, labels_source, features_target, labels_target):
    """Plot t-SNE visualization of embeddings."""
    # Combine
    all_features = np.vstack([features_source, features_target])
    all_labels = np.hstack([labels_source, labels_target])
    domains = np.array(['source'] * len(features_source) + ['target'] * len(features_target))

    # t-SNE
    logger.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embedded = tsne.fit_transform(all_features)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By domain
    ax = axes[0]
    source_mask = domains == 'source'
    target_mask = domains == 'target'
    ax.scatter(embedded[source_mask, 0], embedded[source_mask, 1], c='blue', alpha=0.5, label='Source (AURSAD)')
    ax.scatter(embedded[target_mask, 0], embedded[target_mask, 1], c='red', alpha=0.5, label='Target (Voraus)')
    ax.set_title('By Domain')
    ax.legend()

    # By label
    ax = axes[1]
    normal_mask = all_labels == 0
    anomaly_mask = all_labels == 1
    ax.scatter(embedded[normal_mask, 0], embedded[normal_mask, 1], c='green', alpha=0.5, label='Normal')
    ax.scatter(embedded[anomaly_mask, 0], embedded[anomaly_mask, 1], c='orange', alpha=0.5, label='Anomaly')
    ax.set_title('By Label')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_embeddings.png", dpi=150)
    plt.close()
    logger.info("Saved t-SNE plot")


def main():
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load datasets
    logger.info("Loading AURSAD (source)...")
    config_source = FactoryNetConfig(
        data_source='aursad',
        max_episodes=200,
        window_size=256,
        stride=128,
        train_healthy_only=False,  # Include anomalies for evaluation
    )
    ds_source_train = FactoryNetDataset(config_source, split='train')
    ds_source_test = FactoryNetDataset(config_source, split='test')

    logger.info("Loading Voraus (target)...")
    config_target = FactoryNetConfig(
        data_source='voraus',
        max_episodes=200,
        window_size=256,
        stride=128,
        train_healthy_only=False,
    )
    ds_target = FactoryNetDataset(config_target, split='test')

    # Train encoder on source
    logger.info("Training encoder on source domain...")
    encoder = train_encoder(ds_source_train, epochs=3, device=device)

    # Extract features
    logger.info("Extracting features...")
    feat_source_train, labels_source_train = extract_features(ds_source_train, encoder, n_samples=500, device=device)
    feat_source_test, labels_source_test = extract_features(ds_source_test, encoder, n_samples=200, device=device)
    feat_target, labels_target = extract_features(ds_target, encoder, n_samples=500, device=device)

    # Train linear probe on source
    logger.info("Training linear probe on source...")
    probe = LogisticRegression(max_iter=1000, class_weight='balanced')

    # Only train if we have both classes
    if len(np.unique(labels_source_train)) > 1:
        probe.fit(feat_source_train, labels_source_train)

        # Evaluate on source test
        pred_source = probe.predict(feat_source_test)
        prob_source = probe.predict_proba(feat_source_test)[:, 1] if len(np.unique(labels_source_test)) > 1 else None

        print("\n" + "="*60)
        print("SOURCE DOMAIN EVALUATION")
        print("="*60)
        print(f"Accuracy: {accuracy_score(labels_source_test, pred_source):.4f}")
        if prob_source is not None and len(np.unique(labels_source_test)) > 1:
            print(f"ROC-AUC: {roc_auc_score(labels_source_test, prob_source):.4f}")

        # Evaluate on target (zero-shot transfer)
        pred_target = probe.predict(feat_target)
        prob_target = probe.predict_proba(feat_target)[:, 1] if len(np.unique(labels_target)) > 1 else None

        print("\n" + "="*60)
        print("TARGET DOMAIN EVALUATION (Zero-Shot Transfer)")
        print("="*60)
        print(f"Accuracy: {accuracy_score(labels_target, pred_target):.4f}")
        if prob_target is not None and len(np.unique(labels_target)) > 1:
            print(f"ROC-AUC: {roc_auc_score(labels_target, prob_target):.4f}")

        print("\nClassification Report (Target):")
        print(classification_report(labels_target, pred_target, target_names=['Normal', 'Anomaly']))

        # Assessment
        print("\n" + "="*60)
        print("TRANSFER ASSESSMENT")
        print("="*60)
        if prob_target is not None and len(np.unique(labels_target)) > 1:
            target_auc = roc_auc_score(labels_target, prob_target)
            if target_auc > 0.7:
                print(f"AUC = {target_auc:.3f} - STRONG transfer potential")
            elif target_auc > 0.6:
                print(f"AUC = {target_auc:.3f} - MODERATE transfer potential")
            else:
                print(f"AUC = {target_auc:.3f} - WEAK transfer potential")
    else:
        print("WARNING: Source training data has only one class, cannot train probe")

    # Plot t-SNE
    plot_tsne(feat_source_test, labels_source_test, feat_target, labels_target)

    logger.info("Done!")


if __name__ == "__main__":
    main()

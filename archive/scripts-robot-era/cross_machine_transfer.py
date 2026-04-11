#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Cross-Machine Transfer Experiment for Industrial JEPA.

This script demonstrates that JEPA learns transferable representations
by training on one robot (source) and evaluating on another (target).

Experiment Design:
==================

1. Train Phase (Source Domain - AURSAD/UR3e):
   - Train JEPA encoder using self-supervised objective
   - Learn to predict future states from past context

2. Transfer Phase (Target Domain - voraus-AD/Yu-Cobot):
   - Freeze the encoder weights
   - Only train a new prediction head on target domain
   - Compare with training from scratch

Metrics:
========
- Prediction Error: How well does source encoder predict on target?
- Latent Space Quality: Clustering, nearest neighbor accuracy
- Anomaly Detection: ROC-AUC on target domain faults
- Few-shot Learning: Performance with limited target domain data

Key Insight:
============
If JEPA learns physics-based representations (dynamics, effort relationships),
these should transfer across robots with similar kinematics and tasks.
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TransferConfig:
    """Configuration for transfer experiment."""
    # Source domain
    source_dataset: str = "aursad"
    source_machine: str = "UR3e"

    # Target domain
    target_dataset: str = "voraus"
    target_machine: str = "Yu-Cobot"

    # Signal configuration (shared across both)
    setpoint_dim: int = 12  # 6 pos + 6 vel
    effort_dim: int = 6     # 6 joint torques (common denominator)

    # Model
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # Data
    window_size: int = 256
    stride: int = 128
    batch_size: int = 64

    # Training
    source_epochs: int = 30
    target_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01

    # Few-shot settings
    few_shot_fractions: list = None

    def __post_init__(self):
        if self.few_shot_fractions is None:
            self.few_shot_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


class TransferEncoder(nn.Module):
    """
    Encoder for cross-machine transfer experiments.

    Key Design:
    - Operates on unified (shared) signal space
    - No machine-specific components
    - Learns temporal dynamics
    """

    def __init__(self, config: TransferConfig):
        super().__init__()
        self.config = config

        input_dim = config.setpoint_dim + config.effort_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.window_size, config.hidden_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output: mean pool over sequence
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input sequence to latent representation.

        Args:
            x: (batch, seq_len, input_dim) - concatenated setpoint + effort

        Returns:
            z: (batch, hidden_dim) - latent representation
        """
        # Project and add positional encoding
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]

        # Transformer encoding
        h = self.transformer(h)

        # Mean pool and normalize
        z = self.norm(h.mean(dim=1))

        return z


class TemporalPredictor(nn.Module):
    """
    Temporal predictor for JEPA-style training.

    Predicts future latent states from past context.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z_context: torch.Tensor) -> torch.Tensor:
        """Predict target embedding from context embedding."""
        return self.predictor(z_context)


class TransferModel(nn.Module):
    """
    Complete model for transfer experiments.

    Contains:
    - Context encoder (learns from past)
    - Target encoder (EMA of context encoder)
    - Predictor (predicts target from context)
    """

    def __init__(self, config: TransferConfig):
        super().__init__()
        self.config = config

        # Encoders
        self.context_encoder = TransferEncoder(config)
        self.target_encoder = TransferEncoder(config)

        # Initialize target encoder as copy
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = TemporalPredictor(config.hidden_dim)

        # EMA decay
        self.ema_decay = 0.996

        # Context/target split
        self.context_len = config.window_size // 2

    def update_ema(self):
        """Update target encoder with EMA."""
        with torch.no_grad():
            for param_q, param_k in zip(
                self.context_encoder.parameters(),
                self.target_encoder.parameters()
            ):
                param_k.data = self.ema_decay * param_k.data + (1 - self.ema_decay) * param_q.data

    def forward(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
    ) -> dict:
        """
        Forward pass for JEPA training.

        Args:
            setpoint: (batch, seq_len, setpoint_dim)
            effort: (batch, seq_len, effort_dim)

        Returns:
            dict with loss and metrics
        """
        # Combine inputs
        x = torch.cat([setpoint, effort], dim=-1)

        # Split into context (past) and target (future)
        context_x = x[:, :self.context_len, :]
        target_x = x[:, self.context_len:, :]

        # Encode context (with gradients)
        z_context = self.context_encoder(context_x)

        # Predict target
        z_pred = self.predictor(z_context)

        # Encode target (no gradients, EMA)
        with torch.no_grad():
            z_target = self.target_encoder(target_x)

        # JEPA loss: predict target embedding from context
        loss = F.smooth_l1_loss(z_pred, z_target)

        # Cosine similarity for monitoring
        cosine_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean()

        return {
            "loss": loss,
            "cosine_similarity": cosine_sim,
            "z_context": z_context,
            "z_target": z_target,
        }

    def encode(self, setpoint: torch.Tensor, effort: torch.Tensor) -> torch.Tensor:
        """Encode full sequence for downstream tasks."""
        x = torch.cat([setpoint, effort], dim=-1)
        return self.context_encoder(x)

    def compute_anomaly_score(
        self,
        setpoint: torch.Tensor,
        effort: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute anomaly score based on prediction error.

        Higher score = more anomalous (harder to predict).
        """
        x = torch.cat([setpoint, effort], dim=-1)

        context_x = x[:, :self.context_len, :]
        target_x = x[:, self.context_len:, :]

        z_context = self.context_encoder(context_x)
        z_pred = self.predictor(z_context)

        with torch.no_grad():
            z_target = self.target_encoder(target_x)

        # Prediction error as anomaly score
        error = ((z_pred - z_target) ** 2).sum(dim=-1)

        return error


def collate_fn(batch):
    """Custom collate function."""
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])

    # Extract metadata
    metadata = {
        "is_anomaly": [item[2].get("is_anomaly", False) for item in batch],
        "fault_type": [item[2].get("fault_type", "normal") for item in batch],
    }

    return setpoints, efforts, metadata


def create_dataloader(
    dataset_name: str,
    config: TransferConfig,
    split: str = "train",
    use_shared_signals: bool = True,
):
    """
    Create dataloader for a dataset using shared signal space.

    Args:
        dataset_name: "aursad" or "voraus"
        config: Transfer config
        split: "train", "val", or "test"
        use_shared_signals: If True, only use signals common to both datasets
    """
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    # Define shared signal patterns
    # These exist in both AURSAD and voraus-AD
    if use_shared_signals:
        # Only use joint position/velocity for setpoint
        # Both AURSAD and voraus have effort_voltage_0..5
        setpoint_signals = ["position", "velocity"]
        effort_signals = ["voltage"]  # Common to both datasets
    else:
        setpoint_signals = ["position", "velocity"]
        effort_signals = ["voltage", "current", "torque"]

    # Map dataset name to data_source for parquet filtering
    data_source_map = {"aursad": "aursad", "voraus": "voraus", "voraus-ad": "voraus"}
    data_source = data_source_map.get(dataset_name.lower(), dataset_name.lower())

    # Limit episodes for voraus to avoid OOM (11.6M rows is too much)
    max_eps = 500 if data_source == "voraus" else None

    ds_config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        data_source=data_source,
        subset=dataset_name,
        window_size=config.window_size,
        stride=config.stride,
        normalize=True,
        norm_mode="global",  # Preserve magnitude
        setpoint_signals=setpoint_signals,
        effort_signals=effort_signals,
        train_healthy_only=(split == "train"),
        unified_setpoint_dim=config.setpoint_dim,
        unified_effort_dim=config.effort_dim,
        max_episodes=max_eps,
    )

    dataset = FactoryNetDataset(ds_config, split=split)

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )

    return loader, dataset


def train_source_domain(
    model: TransferModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TransferConfig,
    device: torch.device,
    output_dir: Path,
):
    """Train model on source domain (AURSAD)."""
    logger.info("="*60)
    logger.info(f"Training on SOURCE domain: {config.source_dataset} ({config.source_machine})")
    logger.info("="*60)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.source_epochs,
        eta_min=config.lr * 0.01,
    )

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "cosine_sim": []}

    for epoch in range(config.source_epochs):
        # Train
        model.train()
        total_loss = 0
        total_cos = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Source Epoch {epoch+1}/{config.source_epochs}")
        for setpoint, effort, _ in pbar:
            setpoint = setpoint.to(device)
            effort = effort.to(device)

            optimizer.zero_grad()
            output = model(setpoint, effort)
            loss = output["loss"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()

            total_loss += loss.item()
            total_cos += output["cosine_similarity"].item()
            num_batches += 1
            pbar.set_postfix(loss=loss.item(), cos=output["cosine_similarity"].item())

        train_loss = total_loss / num_batches
        train_cos = total_cos / num_batches

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for setpoint, effort, _ in val_loader:
                setpoint = setpoint.to(device)
                effort = effort.to(device)
                output = model(setpoint, effort)
                val_loss += output["loss"].item()
        val_loss /= len(val_loader)

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["cosine_sim"].append(train_cos)

        logger.info(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, cos_sim={train_cos:.4f}"
        )

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "config": asdict(config),
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "source_model.pt")

    return history


@torch.no_grad()
def evaluate_transfer(
    model: TransferModel,
    test_loader: DataLoader,
    device: torch.device,
    domain_name: str,
) -> dict:
    """
    Evaluate model on a domain.

    Returns metrics for:
    - Prediction error
    - Anomaly detection (if labels available)
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_fault_types = []
    all_pred_errors = []

    for setpoint, effort, metadata in tqdm(test_loader, desc=f"Evaluating on {domain_name}"):
        setpoint = setpoint.to(device)
        effort = effort.to(device)

        # Compute anomaly scores
        scores = model.compute_anomaly_score(setpoint, effort)
        all_scores.extend(scores.cpu().numpy())

        # Also compute prediction error
        output = model(setpoint, effort)
        all_pred_errors.append(output["loss"].item())

        all_labels.extend([1 if a else 0 for a in metadata["is_anomaly"]])
        all_fault_types.extend(metadata["fault_type"])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    results = {
        "domain": domain_name,
        "mean_pred_error": float(np.mean(all_pred_errors)),
        "mean_anomaly_score": float(all_scores.mean()),
        "std_anomaly_score": float(all_scores.std()),
    }

    # Anomaly detection metrics (if we have both classes)
    if len(np.unique(all_labels)) > 1:
        results["roc_auc"] = float(roc_auc_score(all_labels, all_scores))
        results["pr_auc"] = float(average_precision_score(all_labels, all_scores))

        # Score separation
        normal_scores = all_scores[all_labels == 0]
        anomaly_scores = all_scores[all_labels == 1]
        results["normal_mean"] = float(normal_scores.mean())
        results["normal_std"] = float(normal_scores.std())
        results["anomaly_mean"] = float(anomaly_scores.mean())
        results["anomaly_std"] = float(anomaly_scores.std())
        results["score_separation"] = float(anomaly_scores.mean() - normal_scores.mean())
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None

    results["label_distribution"] = {
        "total": len(all_labels),
        "normal": int((all_labels == 0).sum()),
        "anomaly": int((all_labels == 1).sum()),
    }

    return results


def few_shot_transfer(
    source_model: TransferModel,
    target_train_dataset,
    target_test_loader: DataLoader,
    config: TransferConfig,
    device: torch.device,
    fraction: float,
) -> dict:
    """
    Evaluate few-shot transfer learning.

    Fine-tune only the predictor head on a fraction of target data,
    keeping the encoder frozen.
    """
    # Create subset of target training data
    n_samples = len(target_train_dataset)
    n_few_shot = max(1, int(n_samples * fraction))
    indices = np.random.choice(n_samples, n_few_shot, replace=False)
    subset = Subset(target_train_dataset, indices)

    few_shot_loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Create new predictor, freeze encoder
    model = TransferModel(config).to(device)
    model.context_encoder.load_state_dict(source_model.context_encoder.state_dict())
    model.target_encoder.load_state_dict(source_model.target_encoder.state_dict())

    # Freeze encoders
    for param in model.context_encoder.parameters():
        param.requires_grad = False
    for param in model.target_encoder.parameters():
        param.requires_grad = False

    # Only train predictor
    optimizer = torch.optim.AdamW(
        model.predictor.parameters(),
        lr=config.lr,
    )

    # Quick fine-tuning
    for epoch in range(config.target_epochs):
        model.train()
        for setpoint, effort, _ in few_shot_loader:
            setpoint = setpoint.to(device)
            effort = effort.to(device)

            optimizer.zero_grad()
            output = model(setpoint, effort)
            output["loss"].backward()
            optimizer.step()

    # Evaluate
    results = evaluate_transfer(model, target_test_loader, device, f"few_shot_{fraction}")
    results["fraction"] = fraction
    results["n_samples"] = n_few_shot

    return results


def train_from_scratch(
    config: TransferConfig,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Train model from scratch on target domain (baseline).
    """
    logger.info("Training from scratch on target domain (baseline)...")

    model = TransferModel(config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.target_epochs):
        model.train()
        for setpoint, effort, _ in tqdm(train_loader, desc=f"Scratch Epoch {epoch+1}"):
            setpoint = setpoint.to(device)
            effort = effort.to(device)

            optimizer.zero_grad()
            output = model(setpoint, effort)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()

    results = evaluate_transfer(model, test_loader, device, "scratch_baseline")
    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-machine transfer experiment")

    # Experiment settings
    parser.add_argument("--source", type=str, default="aursad",
                        help="Source dataset (train domain)")
    parser.add_argument("--target", type=str, default="voraus",
                        help="Target dataset (test domain)")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)

    # Training
    parser.add_argument("--source-epochs", type=int, default=30)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Resume from checkpoint
    parser.add_argument("--source-checkpoint", type=str, default=None,
                        help="Path to pre-trained source model (skip source training)")

    # Output
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"transfer_{args.source}_to_{args.target}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = TransferConfig(
        source_dataset=args.source,
        target_dataset=args.target,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        source_epochs=args.source_epochs,
        target_epochs=args.target_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # =========================================================================
    # Phase 1: Train on Source Domain
    # =========================================================================
    logger.info("Loading source domain data...")
    source_train_loader, _ = create_dataloader(config.source_dataset, config, split="train")
    source_val_loader, _ = create_dataloader(config.source_dataset, config, split="val")
    source_test_loader, _ = create_dataloader(config.source_dataset, config, split="test")

    logger.info(f"Source: {len(source_train_loader)} train batches, {len(source_test_loader)} test batches")

    # Create and train model
    model = TransferModel(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.source_checkpoint:
        # Load pre-trained source model (skip training)
        logger.info(f"Loading pre-trained source model from {args.source_checkpoint}")
        checkpoint = torch.load(args.source_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        source_history = None
    else:
        # Train from scratch
        source_history = train_source_domain(
            model, source_train_loader, source_val_loader, config, device, output_dir
        )
        # Load best model
        checkpoint = torch.load(output_dir / "source_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    # Evaluate on source test set
    source_results = evaluate_transfer(model, source_test_loader, device, "source_test")
    logger.info(f"\nSource domain evaluation:")
    logger.info(f"  Prediction error: {source_results['mean_pred_error']:.4f}")
    if source_results["roc_auc"]:
        logger.info(f"  ROC-AUC: {source_results['roc_auc']:.4f}")

    # =========================================================================
    # Phase 2: Zero-Shot Transfer to Target Domain
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info(f"Zero-shot transfer to TARGET: {config.target_dataset} ({config.target_machine})")
    logger.info("="*60)

    target_train_loader, target_train_ds = create_dataloader(config.target_dataset, config, split="train")
    target_test_loader, _ = create_dataloader(config.target_dataset, config, split="test")

    logger.info(f"Target: {len(target_train_loader)} train batches, {len(target_test_loader)} test batches")

    # Zero-shot evaluation (no training on target)
    zero_shot_results = evaluate_transfer(model, target_test_loader, device, "zero_shot_transfer")
    logger.info(f"\nZero-shot transfer evaluation:")
    logger.info(f"  Prediction error: {zero_shot_results['mean_pred_error']:.4f}")
    if zero_shot_results["roc_auc"]:
        logger.info(f"  ROC-AUC: {zero_shot_results['roc_auc']:.4f}")

    # =========================================================================
    # Phase 3: Few-Shot Transfer Learning
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Few-shot transfer learning")
    logger.info("="*60)

    few_shot_results = []
    for fraction in config.few_shot_fractions:
        result = few_shot_transfer(
            model, target_train_ds, target_test_loader, config, device, fraction
        )
        few_shot_results.append(result)
        logger.info(
            f"  {fraction*100:.0f}% data ({result['n_samples']} samples): "
            f"pred_error={result['mean_pred_error']:.4f}"
            + (f", ROC-AUC={result['roc_auc']:.4f}" if result["roc_auc"] else "")
        )

    # =========================================================================
    # Phase 4: Baseline - Train from Scratch
    # =========================================================================
    logger.info("\n" + "="*60)
    logger.info("Baseline: Training from scratch on target domain")
    logger.info("="*60)

    scratch_results = train_from_scratch(config, target_train_loader, target_test_loader, device)
    logger.info(f"Scratch baseline:")
    logger.info(f"  Prediction error: {scratch_results['mean_pred_error']:.4f}")
    if scratch_results["roc_auc"]:
        logger.info(f"  ROC-AUC: {scratch_results['roc_auc']:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    all_results = {
        "config": asdict(config),
        "source_training": source_history,
        "source_test": source_results,
        "zero_shot_transfer": zero_shot_results,
        "few_shot_transfer": few_shot_results,
        "scratch_baseline": scratch_results,
    }

    with open(output_dir / "transfer_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "="*70)
    print("TRANSFER EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Source: {config.source_dataset} ({config.source_machine})")
    print(f"Target: {config.target_dataset} ({config.target_machine})")
    print("-"*70)
    print(f"{'Condition':<30} {'Pred Error':>15} {'ROC-AUC':>15}")
    print("-"*70)

    print(f"{'Source test':<30} {source_results['mean_pred_error']:>15.4f} "
          f"{source_results['roc_auc'] or 'N/A':>15}")
    print(f"{'Zero-shot transfer':<30} {zero_shot_results['mean_pred_error']:>15.4f} "
          f"{zero_shot_results['roc_auc'] or 'N/A':>15}")

    for r in few_shot_results:
        label = f"Few-shot ({r['fraction']*100:.0f}%, n={r['n_samples']})"
        roc = r['roc_auc'] or 'N/A'
        print(f"{label:<30} {r['mean_pred_error']:>15.4f} {roc:>15}")

    print(f"{'Scratch baseline (100%)':<30} {scratch_results['mean_pred_error']:>15.4f} "
          f"{scratch_results['roc_auc'] or 'N/A':>15}")

    print("-"*70)

    # Key insight
    if zero_shot_results['mean_pred_error'] < scratch_results['mean_pred_error'] * 1.5:
        print("\n✓ POSITIVE TRANSFER: Zero-shot performs within 50% of scratch baseline!")
        print("  This suggests JEPA learned transferable physics-based representations.")
    else:
        print("\n✗ LIMITED TRANSFER: Zero-shot significantly worse than scratch baseline.")
        print("  The domains may have different dynamics or the model needs improvement.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

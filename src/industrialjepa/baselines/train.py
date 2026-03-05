# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Unified training script for baseline models.

Supports:
- MAE (Masked AutoEncoder)
- Autoencoder (simple encoder-decoder)
- Contrastive (SimCLR-style)

Usage:
    python -m industrialjepa.baselines.train --model mae --dataset aursad
    python -m industrialjepa.baselines.train --model autoencoder --dataset voraus-ad
    python -m industrialjepa.baselines.train --model contrastive --dataset aursad
"""

import argparse
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.baselines import (
    MAE, MAEConfig,
    Autoencoder, AutoencoderConfig,
    ContrastiveModel, ContrastiveConfig,
)
from industrialjepa.baselines.autoencoder import VariationalAutoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate function to convert tuple format to dict."""
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])

    # Collect metadata
    setpoint_masks = torch.stack([item[2]["setpoint_mask"] for item in batch])
    effort_masks = torch.stack([item[2]["effort_mask"] for item in batch])

    return {
        "setpoint": setpoints,
        "effort": efforts,
        "setpoint_mask": setpoint_masks,
        "effort_mask": effort_masks,
        "is_anomaly": [item[2]["is_anomaly"] for item in batch],
        "fault_type": [item[2]["fault_type"] for item in batch],
    }


def get_model_and_config(model_name: str, args):
    """Create model and config based on name."""
    base_kwargs = {
        "setpoint_dim": args.setpoint_dim,
        "effort_dim": args.effort_dim,
        "seq_len": args.window_size,
        "patch_size": args.patch_size,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
    }

    if model_name == "mae":
        config = MAEConfig(
            mask_ratio=args.mask_ratio,
            decoder_hidden_dim=args.hidden_dim // 2,
            decoder_num_layers=2,
            **base_kwargs,
        )
        model = MAE(config)

    elif model_name == "autoencoder":
        config = AutoencoderConfig(
            decoder_hidden_dim=args.hidden_dim,
            decoder_num_layers=args.num_layers,
            use_bottleneck=args.use_bottleneck,
            latent_dim=args.latent_dim,
            **base_kwargs,
        )
        model = Autoencoder(config)

    elif model_name == "vae":
        config = AutoencoderConfig(
            decoder_hidden_dim=args.hidden_dim,
            decoder_num_layers=args.num_layers,
            use_bottleneck=True,
            latent_dim=args.latent_dim,
            **base_kwargs,
        )
        model = VariationalAutoencoder(config)

    elif model_name == "contrastive":
        config = ContrastiveConfig(
            temperature=args.temperature,
            projection_dim=128,
            use_effort_pairs=args.use_effort_pairs,
            noise_std=0.1,
            **base_kwargs,
        )
        model = ContrastiveModel(config)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model, config


def create_dataloaders(args) -> tuple:
    """Create train and validation dataloaders."""
    # Dataset config
    data_config = FactoryNetConfig(
        dataset_name=args.dataset_name,
        subset=args.dataset,
        window_size=args.window_size,
        stride=args.stride,
        train_healthy_only=True,  # One-class anomaly detection
        aursad_phase_handling=args.aursad_phase,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # Create datasets
    train_ds = FactoryNetDataset(data_config, split="train")
    val_ds = FactoryNetDataset(data_config, split="val")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    wandb_run=None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    all_metrics = {}

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        setpoint = batch["setpoint"].to(device)
        effort = batch["effort"].to(device)
        setpoint_mask = batch.get("setpoint_mask")
        effort_mask = batch.get("effort_mask")

        if setpoint_mask is not None:
            setpoint_mask = setpoint_mask.to(device)
        if effort_mask is not None:
            effort_mask = effort_mask.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(
            setpoint=setpoint,
            effort=effort,
            setpoint_mask=setpoint_mask,
            effort_mask=effort_mask,
        )

        loss = output["loss"]

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if k not in all_metrics:
                all_metrics[k] = 0.0
            all_metrics[k] += v

        pbar.set_postfix(loss=loss.item())

    # Average metrics
    avg_loss = total_loss / num_batches
    for k in all_metrics:
        all_metrics[k] /= num_batches

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({
            "train/loss": avg_loss,
            **{f"train/{k}": v for k, v in all_metrics.items()},
            "epoch": epoch,
        })

    return {"loss": avg_loss, **all_metrics}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    epoch: int,
    wandb_run=None,
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_scores = []

    for batch in tqdm(val_loader, desc="Validating"):
        setpoint = batch["setpoint"].to(device)
        effort = batch["effort"].to(device)
        setpoint_mask = batch.get("setpoint_mask")
        effort_mask = batch.get("effort_mask")

        if setpoint_mask is not None:
            setpoint_mask = setpoint_mask.to(device)
        if effort_mask is not None:
            effort_mask = effort_mask.to(device)

        # Forward pass
        output = model(
            setpoint=setpoint,
            effort=effort,
            setpoint_mask=setpoint_mask,
            effort_mask=effort_mask,
        )
        total_loss += output["loss"].item()

        # Compute anomaly scores
        scores = model.compute_anomaly_score(
            setpoint=setpoint,
            effort=effort,
            setpoint_mask=setpoint_mask,
            effort_mask=effort_mask,
        )
        all_scores.append(scores)
        num_batches += 1

    # Average metrics
    avg_loss = total_loss / num_batches
    all_scores = torch.cat(all_scores)
    score_mean = all_scores.mean().item()
    score_std = all_scores.std().item()

    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({
            "val/loss": avg_loss,
            "val/anomaly_score_mean": score_mean,
            "val/anomaly_score_std": score_std,
            "epoch": epoch,
        })

    return {
        "loss": avg_loss,
        "anomaly_score_mean": score_mean,
        "anomaly_score_std": score_std,
    }


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")

    # Model selection
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["mae", "autoencoder", "vae", "contrastive"],
        help="Model type to train",
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="aursad",
        choices=["aursad", "voraus-ad", "nasa-milling", "rh20t", "reassemble"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--dataset-name", type=str, default="Forgis/factorynet-hackathon",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--aursad-phase", type=str, default="both",
        choices=["both", "tightening_only", "merge"],
        help="AURSAD phase handling",
    )

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=16)

    # Data dimensions (FactoryNet unified)
    parser.add_argument("--setpoint-dim", type=int, default=14)
    parser.add_argument("--effort-dim", type=int, default=7)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)

    # MAE-specific
    parser.add_argument("--mask-ratio", type=float, default=0.75)

    # Autoencoder-specific
    parser.add_argument("--use-bottleneck", action="store_true")
    parser.add_argument("--latent-dim", type=int, default=64)

    # Contrastive-specific
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--use-effort-pairs", action="store_true", default=True)

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=4)

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10)

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Use WandB logging")
    parser.add_argument("--wandb-project", type=str, default="industrialjepa")
    parser.add_argument("--wandb-entity", type=str, default=None)

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.model}_{args.dataset}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup WandB
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=vars(args),
            )
        except ImportError:
            logger.warning("WandB not installed, skipping logging")

    # Create model
    logger.info(f"Creating {args.model} model...")
    model, config = get_model_and_config(args.model, args)
    model = model.to(args.device)
    logger.info(f"Model parameters: {model.get_num_params():,}")

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Create dataloaders
    logger.info(f"Loading {args.dataset} dataset...")
    train_loader, val_loader = create_dataloaders(args)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, args.device, epoch, wandb_run
        )
        logger.info(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}")

        # Validate
        val_metrics = validate(
            model, val_loader, args.device, epoch, wandb_run
        )
        logger.info(f"Epoch {epoch}: val_loss={val_metrics['loss']:.4f}")

        # Update scheduler
        scheduler.step()

        # Save best model
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            model.save_pretrained(str(output_dir / "best_model.pt"))
            logger.info(f"Saved best model with val_loss={best_val_loss:.4f}")

        # Periodic saving
        if (epoch + 1) % args.save_every == 0:
            model.save_pretrained(str(output_dir / f"checkpoint_epoch_{epoch}.pt"))

    # Save final model
    model.save_pretrained(str(output_dir / "final_model.pt"))

    # Finish wandb
    if wandb_run is not None:
        wandb_run.finish()

    logger.info(f"Training complete! Best val_loss: {best_val_loss:.4f}")
    logger.info(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

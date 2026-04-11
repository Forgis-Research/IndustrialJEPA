#!/usr/bin/env python3
"""
Cross-Task Transfer Evaluation for AURSAD.

Tests if a model trained on LOOSENING phase transfers to TIGHTENING phase.
This is an intermediate difficulty transfer:
- Same robot (UR3e)
- Same sensors (torque + Cartesian forces)
- Related but distinct tasks

This is MORE realistic than cross-machine transfer (AURSAD → voraus-AD)
because the sensor modalities match.

Protocol:
1. Train on AURSAD loosening phase (healthy operations only)
2. Evaluate zero-shot on AURSAD tightening phase
3. Compare: zero-shot vs few-shot vs scratch baseline
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

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
class CrossTaskConfig:
    """Configuration for cross-task transfer."""
    dataset: str = "aursad"

    # Signal configuration
    setpoint_dim: int = 12  # 6 pos + 6 vel
    effort_dim: int = 6     # 6 joint currents

    # Model
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    latent_dim: int = 128

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


class SimpleEncoder(nn.Module):
    """Simple transformer encoder for transfer experiments."""

    def __init__(self, config: CrossTaskConfig):
        super().__init__()
        self.config = config

        input_dim = config.setpoint_dim + config.effort_dim

        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.window_size, config.hidden_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.to_latent = nn.Linear(config.hidden_dim, config.latent_dim)
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_encoding[:, :x.size(1), :]
        h = self.transformer(h)
        z = self.to_latent(h.mean(dim=1))
        return self.norm(z)


class Predictor(nn.Module):
    """MLP predictor for latent space."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class CrossTaskModel(nn.Module):
    """Model for cross-task transfer experiments."""

    def __init__(self, config: CrossTaskConfig):
        super().__init__()
        self.config = config

        self.encoder = SimpleEncoder(config)
        self.target_encoder = SimpleEncoder(config)
        self.predictor = Predictor(config.latent_dim)

        # Initialize target encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.ema_decay = 0.996
        self.context_len = config.window_size // 2

    @torch.no_grad()
    def update_ema(self):
        for p_online, p_target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            p_target.data = self.ema_decay * p_target.data + (1 - self.ema_decay) * p_online.data

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        context_x = x[:, :self.context_len, :]
        target_x = x[:, self.context_len:, :]

        z_context = self.encoder(context_x)
        z_pred = self.predictor(z_context)

        with torch.no_grad():
            z_target = self.target_encoder(target_x)

        loss = F.smooth_l1_loss(z_pred, z_target)
        cosine_sim = F.cosine_similarity(z_pred, z_target, dim=-1).mean()

        return {
            "loss": loss,
            "cosine_similarity": cosine_sim,
            "z_context": z_context,
            "z_pred": z_pred,
        }

    def compute_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        context_x = x[:, :self.context_len, :]
        target_x = x[:, self.context_len:, :]

        z_context = self.encoder(context_x)
        z_pred = self.predictor(z_context)

        with torch.no_grad():
            z_target = self.target_encoder(target_x)

        error = ((z_pred - z_target) ** 2).sum(dim=-1)
        return error


def create_phase_dataloader(
    phase: str,  # "loosening" or "tightening"
    config: CrossTaskConfig,
    split: str = "train",
    healthy_only: bool = False,
):
    """Create dataloader for a specific phase of AURSAD."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    ds_config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        data_source="aursad",  # Only download AURSAD parquet files
        subset="aursad",
        window_size=config.window_size,
        stride=config.stride,
        normalize=True,
        norm_mode="global",
        setpoint_signals=["position", "velocity"],
        effort_signals=["current"],  # AURSAD has current, not torque
        train_healthy_only=healthy_only,
        unified_setpoint_dim=config.setpoint_dim,
        unified_effort_dim=config.effort_dim,
        aursad_phase_handling="both",  # Load both, filter manually
    )

    dataset = FactoryNetDataset(ds_config, split=split)

    # Filter by phase: AURSAD odd episode_ids = loosening, even = tightening
    # Build episode-level phase map first (fast), then filter windows
    phase_map = {}
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            meta = item[2]
            ep_id = str(meta.get("episode_id", ""))
            if ep_id not in phase_map:
                try:
                    ep_num = int(ep_id.split("_")[-1])
                    phase_map[ep_id] = "loosening" if ep_num % 2 == 1 else "tightening"
                except (ValueError, IndexError):
                    phase_map[ep_id] = "tightening"
        except Exception:
            continue

    phase_indices = []
    for i in range(len(dataset)):
        try:
            item = dataset[i]
            meta = item[2]
            ep_id = str(meta.get("episode_id", ""))
            ep_phase = phase_map.get(ep_id, "tightening")

            if ep_phase == phase:
                if not healthy_only or not meta.get("is_anomaly", False):
                    phase_indices.append(i)
        except Exception:
            continue

    if len(phase_indices) == 0:
        logger.warning(f"No samples found for phase={phase}, split={split}")
        return None, None

    subset = Subset(dataset, phase_indices)

    def collate_fn(batch):
        setpoints = torch.stack([item[0] for item in batch])
        efforts = torch.stack([item[1] for item in batch])
        x = torch.cat([setpoints, efforts], dim=-1)
        metadata = [item[2] for item in batch]
        return x, metadata

    loader = DataLoader(
        subset,
        batch_size=config.batch_size,
        shuffle=(split == "train"),
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )

    logger.info(f"Phase={phase}, split={split}: {len(phase_indices)} samples, {len(loader)} batches")
    return loader, subset


def train_on_phase(
    model: CrossTaskModel,
    train_loader: DataLoader,
    config: CrossTaskConfig,
    device: torch.device,
    phase_name: str,
):
    """Train model on a specific phase."""
    logger.info(f"Training on {phase_name} phase...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    for epoch in range(config.source_epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.source_epochs}"):
            x = x.to(device)

            optimizer.zero_grad()
            output = model(x)
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()

            total_loss += output["loss"].item()
            n_batches += 1

        logger.info(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}")

    return model


@torch.no_grad()
def evaluate_on_phase(
    model: CrossTaskModel,
    test_loader: DataLoader,
    device: torch.device,
    phase_name: str,
) -> Dict[str, float]:
    """Evaluate model on a specific phase."""
    model.eval()

    all_scores = []
    all_labels = []
    all_pred_errors = []

    for x, metadata in tqdm(test_loader, desc=f"Evaluating {phase_name}"):
        x = x.to(device)

        scores = model.compute_anomaly_score(x)
        all_scores.extend(scores.cpu().numpy())

        output = model(x)
        all_pred_errors.append(output["loss"].item())

        all_labels.extend([1 if m.get("is_anomaly", False) else 0 for m in metadata])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    results = {
        "phase": phase_name,
        "mean_pred_error": float(np.mean(all_pred_errors)),
        "mean_anomaly_score": float(all_scores.mean()),
        "n_samples": len(all_labels),
        "n_anomalies": int(all_labels.sum()),
    }

    if len(np.unique(all_labels)) > 1:
        results["roc_auc"] = float(roc_auc_score(all_labels, all_scores))
        results["pr_auc"] = float(average_precision_score(all_labels, all_scores))
    else:
        results["roc_auc"] = None
        results["pr_auc"] = None

    return results


def main():
    parser = argparse.ArgumentParser(description="Cross-task transfer (loosening → tightening)")
    parser.add_argument("--source-epochs", type=int, default=30)
    parser.add_argument("--target-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"cross_task_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = CrossTaskConfig(
        source_epochs=args.source_epochs,
        target_epochs=args.target_epochs,
        batch_size=args.batch_size,
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # =========================================================================
    # Phase 1: Train on LOOSENING (source)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 1: Training on LOOSENING")
    logger.info("=" * 60)

    loosen_train, _ = create_phase_dataloader("loosening", config, "train", healthy_only=True)
    loosen_test, _ = create_phase_dataloader("loosening", config, "test", healthy_only=False)

    if loosen_train is None:
        logger.error("No loosening training data found")
        return

    model = CrossTaskModel(config).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = train_on_phase(model, loosen_train, config, device, "loosening")

    # Save model
    torch.save(model.state_dict(), output_dir / "source_model.pt")

    # Evaluate on source
    if loosen_test:
        source_results = evaluate_on_phase(model, loosen_test, device, "loosening")
        logger.info(f"Source (loosening): pred_error={source_results['mean_pred_error']:.4f}")
    else:
        source_results = {"phase": "loosening", "mean_pred_error": None}

    # =========================================================================
    # Phase 2: Zero-shot transfer to TIGHTENING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 2: Zero-shot transfer to TIGHTENING")
    logger.info("=" * 60)

    tighten_train, tighten_train_ds = create_phase_dataloader("tightening", config, "train", healthy_only=False)
    tighten_test, _ = create_phase_dataloader("tightening", config, "test", healthy_only=False)

    if tighten_test is None:
        logger.error("No tightening test data found")
        return

    zero_shot_results = evaluate_on_phase(model, tighten_test, device, "tightening_zero_shot")
    logger.info(f"Zero-shot (tightening): pred_error={zero_shot_results['mean_pred_error']:.4f}")
    if zero_shot_results["roc_auc"]:
        logger.info(f"  ROC-AUC: {zero_shot_results['roc_auc']:.4f}")

    # =========================================================================
    # Phase 3: Scratch baseline on TIGHTENING
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Phase 3: Scratch baseline on TIGHTENING")
    logger.info("=" * 60)

    scratch_model = CrossTaskModel(config).to(device)
    scratch_model = train_on_phase(scratch_model, tighten_train, config, device, "tightening_scratch")

    scratch_results = evaluate_on_phase(scratch_model, tighten_test, device, "tightening_scratch")
    logger.info(f"Scratch (tightening): pred_error={scratch_results['mean_pred_error']:.4f}")
    if scratch_results["roc_auc"]:
        logger.info(f"  ROC-AUC: {scratch_results['roc_auc']:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    all_results = {
        "config": asdict(config),
        "source_loosening": source_results,
        "zero_shot_tightening": zero_shot_results,
        "scratch_tightening": scratch_results,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("CROSS-TASK TRANSFER SUMMARY (Loosening → Tightening)")
    print("=" * 70)
    print(f"{'Setting':<30} {'Pred Error':>15} {'ROC-AUC':>15}")
    print("-" * 70)
    def fmt_val(v):
        return f"{v:.4f}" if isinstance(v, (int, float)) and v is not None else "N/A"
    print(f"{'Source (loosening)':<30} {fmt_val(source_results.get('mean_pred_error')):>15} {'N/A':>15}")
    print(f"{'Zero-shot (tightening)':<30} {fmt_val(zero_shot_results['mean_pred_error']):>15} {fmt_val(zero_shot_results.get('roc_auc')):>15}")
    print(f"{'Scratch (tightening)':<30} {fmt_val(scratch_results['mean_pred_error']):>15} {fmt_val(scratch_results.get('roc_auc')):>15}")
    print("-" * 70)

    # Transfer ratio
    transfer_ratio = zero_shot_results['mean_pred_error'] / (scratch_results['mean_pred_error'] + 1e-8)
    print(f"\nTransfer ratio: {transfer_ratio:.2f}x (lower is better, 1.0 = perfect)")

    if transfer_ratio < 1.5:
        print("POSITIVE TRANSFER: Zero-shot within 50% of scratch baseline")
    else:
        print("LIMITED TRANSFER: Zero-shot significantly worse than scratch")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

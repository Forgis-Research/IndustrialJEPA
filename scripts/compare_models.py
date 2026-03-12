#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Compare JEPA world model against baselines for anomaly detection.

Computes:
- AUC-ROC, F1, Precision, Recall for binary anomaly detection
- Per-fault-type breakdown
- Comparison table in markdown format

Usage:
    python scripts/compare_models.py \
        --jepa-checkpoint results/world_model/best_model.pt \
        --baseline-dir results/baselines \
        --data-source aursad
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, precision_score, recall_score

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.models.world_model import WorldModel
from industrialjepa.baselines import (
    EffortAutoencoder, SetpointToEffort, TemporalPredictor,
    MAE, ContrastiveModel,
    AutoencoderConfig, TemporalConfig, MAEConfig, ContrastiveConfig,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Collate batch with metadata."""
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])
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


def load_test_data(data_source: str, batch_size: int = 64) -> DataLoader:
    """Load test dataset."""
    config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        data_source=data_source,
        window_size=256,
        stride=128,
        normalize=True,
        norm_mode="global",
        train_healthy_only=True,
    )

    # Load train first to get shared data, then test
    train_ds = FactoryNetDataset(config, split="train")
    shared_data = train_ds.get_shared_data()
    test_ds = FactoryNetDataset(config, split="test", shared_data=shared_data)

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )


@torch.no_grad()
def compute_anomaly_scores_jepa(
    model: WorldModel,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute anomaly scores using JEPA world model."""
    model.eval()
    all_scores = []
    all_labels = []
    all_fault_types = []

    for batch in tqdm(dataloader, desc="JEPA scoring"):
        setpoint = batch["setpoint"].to(device)
        effort = batch["effort"].to(device)

        # Use effort as observation (JEPA predicts effort from setpoint)
        obs_t = effort[:, :-1]  # All but last timestep
        cmd_t = setpoint[:, :-1]  # Commands
        obs_t1 = effort[:, 1:]  # Next timestep (target)

        # Get latent representations
        z_t = model.encode(obs_t)
        z_pred = model.predict(z_t, cmd_t)
        z_target = model.encode_target(obs_t1)

        # Anomaly score = prediction error in latent space
        error = ((z_pred - z_target) ** 2).mean(dim=(1, 2))

        all_scores.extend(error.cpu().numpy())
        all_labels.extend([1 if a else 0 for a in batch["is_anomaly"]])
        all_fault_types.extend(batch["fault_type"])

    return np.array(all_scores), np.array(all_labels), all_fault_types


@torch.no_grad()
def compute_anomaly_scores_baseline(
    model,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Compute anomaly scores using a baseline model."""
    model.eval()
    all_scores = []
    all_labels = []
    all_fault_types = []

    for batch in tqdm(dataloader, desc="Baseline scoring"):
        setpoint = batch["setpoint"].to(device)
        effort = batch["effort"].to(device)
        setpoint_mask = batch["setpoint_mask"].to(device)
        effort_mask = batch["effort_mask"].to(device)

        scores = model.compute_anomaly_score(
            setpoint=setpoint,
            effort=effort,
            setpoint_mask=setpoint_mask,
            effort_mask=effort_mask,
        )

        all_scores.extend(scores.cpu().numpy())
        all_labels.extend([1 if a else 0 for a in batch["is_anomaly"]])
        all_fault_types.extend(batch["fault_type"])

    return np.array(all_scores), np.array(all_labels), all_fault_types


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    fault_types: List[str],
) -> Dict:
    """Compute anomaly detection metrics."""
    # Binary metrics
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5  # No positive samples

    # Find optimal threshold via precision-recall curve
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    # Compute precision/recall at best threshold
    predictions = (scores >= best_threshold).astype(int)
    best_precision = precision_score(labels, predictions, zero_division=0)
    best_recall = recall_score(labels, predictions, zero_division=0)

    # Per-fault-type breakdown
    fault_type_metrics = {}
    unique_faults = set(fault_types)
    for ft in unique_faults:
        if ft.lower() in ["normal", "none", "null", "healthy"]:
            continue
        mask = np.array([f == ft for f in fault_types])
        if mask.sum() > 0:
            fault_labels = labels[mask]
            fault_scores = scores[mask]
            if fault_labels.sum() > 0:
                try:
                    fault_auc = roc_auc_score(fault_labels, fault_scores)
                except ValueError:
                    fault_auc = 0.5
                fault_type_metrics[ft] = {
                    "auc": fault_auc,
                    "count": int(mask.sum()),
                }

    return {
        "auc_roc": auc,
        "f1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "threshold": best_threshold,
        "per_fault": fault_type_metrics,
    }


def load_baseline_model(checkpoint_dir: Path, device: str):
    """Load a baseline model from checkpoint directory."""
    config_path = checkpoint_dir / "config.json"
    model_path = checkpoint_dir / "best_model.pt"

    if not config_path.exists() or not model_path.exists():
        return None, None

    with open(config_path) as f:
        config_dict = json.load(f)

    # Determine model type from config
    # The config contains all fields, so we check for type-specific ones
    if "mask_ratio" in config_dict:
        config = MAEConfig(**config_dict)
        model = MAE(config)
    elif "context_ratio" in config_dict:
        config = TemporalConfig(**config_dict)
        model = TemporalPredictor(config)
    elif "temperature" in config_dict:
        config = ContrastiveConfig(**config_dict)
        model = ContrastiveModel(config)
    else:
        config = AutoencoderConfig(**config_dict)
        # Check if it's effort-only or setpoint-to-effort
        # This is tricky - we'll use a heuristic based on directory name
        if "s2e" in str(checkpoint_dir).lower():
            model = SetpointToEffort(config)
        else:
            model = EffortAutoencoder(config)

    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config


def main():
    parser = argparse.ArgumentParser(description="Compare models for anomaly detection")
    parser.add_argument(
        "--jepa-checkpoint", type=str, default=None,
        help="Path to JEPA world model checkpoint",
    )
    parser.add_argument(
        "--baseline-dir", type=str, default="results/baselines",
        help="Directory containing baseline checkpoints",
    )
    parser.add_argument(
        "--data-source", type=str, default="aursad",
        help="Data source to evaluate on",
    )
    parser.add_argument(
        "--output", type=str, default="results/comparison.md",
        help="Output file for comparison table",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use",
    )
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load test data
    logger.info(f"Loading test data from {args.data_source}...")
    test_loader = load_test_data(args.data_source)
    logger.info(f"Test batches: {len(test_loader)}")

    results = {}

    # Evaluate JEPA (if checkpoint provided)
    if args.jepa_checkpoint and Path(args.jepa_checkpoint).exists():
        logger.info(f"Loading JEPA model from {args.jepa_checkpoint}...")
        # Load JEPA model
        checkpoint = torch.load(args.jepa_checkpoint, map_location=device)
        # TODO: Need to reconstruct model from config
        logger.warning("JEPA evaluation not yet implemented - need to save config with checkpoint")

    # Evaluate baselines
    baseline_dir = Path(args.baseline_dir)
    if baseline_dir.exists():
        for model_dir in baseline_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            logger.info(f"Evaluating baseline: {model_name}")

            model, config = load_baseline_model(model_dir, device)
            if model is None:
                logger.warning(f"Could not load model from {model_dir}")
                continue

            scores, labels, fault_types = compute_anomaly_scores_baseline(
                model, test_loader, device
            )
            metrics = compute_metrics(scores, labels, fault_types)
            results[model_name] = metrics

            logger.info(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"  F1: {metrics['f1']:.4f}")

    # Generate comparison table
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Model Comparison: Anomaly Detection\n\n")
        f.write(f"**Dataset:** {args.data_source}\n\n")

        # Summary table
        f.write("## Summary\n\n")
        f.write("| Model | AUC-ROC | F1 | Precision | Recall |\n")
        f.write("|-------|---------|----|-----------| -------|\n")

        for model_name, metrics in sorted(results.items(), key=lambda x: -x[1]['auc_roc']):
            f.write(
                f"| {model_name} | {metrics['auc_roc']:.4f} | {metrics['f1']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} |\n"
            )

        # Per-fault breakdown
        f.write("\n## Per-Fault-Type Analysis\n\n")
        for model_name, metrics in results.items():
            if metrics['per_fault']:
                f.write(f"\n### {model_name}\n\n")
                f.write("| Fault Type | AUC-ROC | Count |\n")
                f.write("|------------|---------|-------|\n")
                for ft, fm in sorted(metrics['per_fault'].items(), key=lambda x: -x[1]['auc']):
                    f.write(f"| {ft} | {fm['auc']:.4f} | {fm['count']} |\n")

    logger.info(f"Comparison saved to: {output_path}")

    # Also save as JSON for programmatic access
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results JSON saved to: {json_path}")


if __name__ == "__main__":
    main()

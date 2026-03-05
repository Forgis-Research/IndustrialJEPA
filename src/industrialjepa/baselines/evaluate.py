# SPDX-FileCopyrightText: 2025 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Evaluation script for baseline models.

Computes anomaly detection metrics:
- AUC-ROC
- Average Precision (AUC-PR)
- Precision@K
- Best F1 Score

Usage:
    python -m industrialjepa.baselines.evaluate \
        --model-path checkpoints/mae_aursad/best_model.pt \
        --model-type mae \
        --dataset aursad
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
from industrialjepa.baselines import MAE, Autoencoder, ContrastiveModel
from industrialjepa.baselines.autoencoder import VariationalAutoencoder
from industrialjepa.baselines.train import collate_fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path: str, model_type: str, device: str = "cuda"):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]

    if model_type == "mae":
        from industrialjepa.baselines.mae import MAEConfig
        model = MAE(MAEConfig(**{k: v for k, v in vars(config).items()
                                 if hasattr(MAEConfig, k)}))
    elif model_type == "autoencoder":
        from industrialjepa.baselines.autoencoder import AutoencoderConfig
        model = Autoencoder(AutoencoderConfig(**{k: v for k, v in vars(config).items()
                                                  if hasattr(AutoencoderConfig, k)}))
    elif model_type == "vae":
        from industrialjepa.baselines.autoencoder import AutoencoderConfig
        model = VariationalAutoencoder(AutoencoderConfig(**{k: v for k, v in vars(config).items()
                                                             if hasattr(AutoencoderConfig, k)}))
    elif model_type == "contrastive":
        from industrialjepa.baselines.contrastive import ContrastiveConfig
        model = ContrastiveModel(ContrastiveConfig(**{k: v for k, v in vars(config).items()
                                                       if hasattr(ContrastiveConfig, k)}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def compute_anomaly_scores(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> tuple:
    """
    Compute anomaly scores for all samples.

    Returns:
        scores: numpy array of anomaly scores
        labels: numpy array of ground truth labels (1 = anomaly)
        fault_types: list of fault type strings
    """
    model.eval()
    all_scores = []
    all_labels = []
    all_fault_types = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing scores"):
            setpoint = batch["setpoint"].to(device)
            effort = batch["effort"].to(device)
            setpoint_mask = batch.get("setpoint_mask")
            effort_mask = batch.get("effort_mask")

            if setpoint_mask is not None:
                setpoint_mask = setpoint_mask.to(device)
            if effort_mask is not None:
                effort_mask = effort_mask.to(device)

            # Compute anomaly scores
            scores = model.compute_anomaly_score(
                setpoint=setpoint,
                effort=effort,
                setpoint_mask=setpoint_mask,
                effort_mask=effort_mask,
            )

            all_scores.append(scores.cpu().numpy())
            all_labels.extend([1 if a else 0 for a in batch["is_anomaly"]])
            all_fault_types.extend(batch["fault_type"])

    scores = np.concatenate(all_scores)
    labels = np.array(all_labels)

    return scores, labels, all_fault_types


def compute_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute anomaly detection metrics."""
    # Handle edge cases
    if labels.sum() == 0:
        logger.warning("No positive labels in test set!")
        return {"error": "no_positives"}

    if labels.sum() == len(labels):
        logger.warning("All labels are positive!")
        return {"error": "all_positives"}

    # AUC-ROC
    auc_roc = roc_auc_score(labels, scores)

    # Average Precision (AUC-PR)
    avg_precision = average_precision_score(labels, scores)

    # Precision-Recall curve for best F1
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]

    # Precision at K (K = number of actual anomalies)
    num_anomalies = labels.sum()
    top_k_indices = np.argsort(scores)[-int(num_anomalies):]
    precision_at_k = labels[top_k_indices].mean()

    # FPR at 95% TPR (for calibration)
    fprs, tprs, _ = roc_curve(labels, scores)
    fpr_at_95_tpr = fprs[np.searchsorted(tprs, 0.95)]

    return {
        "auc_roc": auc_roc,
        "avg_precision": avg_precision,
        "best_f1": best_f1,
        "best_threshold": best_threshold,
        "precision_at_k": precision_at_k,
        "fpr_at_95_tpr": fpr_at_95_tpr,
        "num_samples": len(labels),
        "num_anomalies": int(num_anomalies),
        "anomaly_ratio": num_anomalies / len(labels),
    }


def compute_per_fault_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    fault_types: List[str],
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by fault type."""
    unique_faults = set(fault_types) - {"normal", "None", "", "healthy"}

    per_fault = {}
    for fault in unique_faults:
        # Get indices for this fault type vs normal
        fault_mask = np.array([f == fault for f in fault_types])
        normal_mask = np.array([f in ("normal", "None", "", "healthy") for f in fault_types])

        combined_mask = fault_mask | normal_mask
        if fault_mask.sum() == 0:
            continue

        fault_scores = scores[combined_mask]
        fault_labels = labels[combined_mask]

        if fault_labels.sum() > 0 and fault_labels.sum() < len(fault_labels):
            per_fault[fault] = {
                "auc_roc": roc_auc_score(fault_labels, fault_scores),
                "avg_precision": average_precision_score(fault_labels, fault_scores),
                "num_samples": int(fault_mask.sum()),
            }

    return per_fault


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline models")

    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["mae", "autoencoder", "vae", "contrastive"],
        help="Model type",
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="aursad",
        choices=["aursad", "voraus-ad", "nasa-milling", "rh20t", "reassemble"],
    )
    parser.add_argument(
        "--dataset-name", type=str, default="Forgis/factorynet-hackathon",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    # Sequence params (must match training)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)

    args = parser.parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, args.model_type, args.device)
    logger.info(f"Model loaded: {model.get_num_params():,} parameters")

    # Create test dataloader
    logger.info(f"Loading {args.dataset} test set...")
    data_config = FactoryNetConfig(
        dataset_name=args.dataset_name,
        subset=args.dataset,
        window_size=args.window_size,
        stride=args.stride,
        train_healthy_only=False,  # Include anomalies for evaluation
    )
    test_ds = FactoryNetDataset(data_config, split="test")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    logger.info(f"Test samples: {len(test_ds)}")

    # Compute scores
    scores, labels, fault_types = compute_anomaly_scores(model, test_loader, args.device)

    # Compute metrics
    metrics = compute_metrics(scores, labels)
    logger.info("=" * 50)
    logger.info("Overall Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Per-fault metrics
    per_fault = compute_per_fault_metrics(scores, labels, fault_types)
    if per_fault:
        logger.info("-" * 50)
        logger.info("Per-Fault Metrics:")
        for fault, fault_metrics in per_fault.items():
            logger.info(f"  {fault}:")
            for k, v in fault_metrics.items():
                if isinstance(v, float):
                    logger.info(f"    {k}: {v:.4f}")
                else:
                    logger.info(f"    {k}: {v}")

    logger.info("=" * 50)


if __name__ == "__main__":
    main()

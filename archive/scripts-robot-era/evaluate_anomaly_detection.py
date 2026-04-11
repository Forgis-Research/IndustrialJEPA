#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Evaluate JEPA World Model for Anomaly Detection.

This script evaluates the trained world model's ability to detect faults
by using prediction error as an anomaly score.

Key idea: Normal operation has low prediction error (model learned the physics).
          Faults cause high prediction error (unexpected dynamics).

Usage:
    python scripts/evaluate_anomaly_detection.py --checkpoint results/world_model/best_model.pt
    python scripts/evaluate_anomaly_detection.py --checkpoint results/world_model/best_model.pt --visualize
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Metrics
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from industrialjepa.data import FactoryNetConfig, WorldModelDataConfig, create_world_model_dataloaders
from industrialjepa.model import JEPAWorldModel, WorldModelConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JEPA for Anomaly Detection")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (default: same dir as checkpoint)")

    # Data
    parser.add_argument("--data-source", type=str, default="aursad",
                        help="Data source: aursad, voraus, cnc")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for evaluation")

    # Evaluation options
    parser.add_argument("--split", type=str, default="test",
                        choices=["val", "test"], help="Which split to evaluate")
    parser.add_argument("--error-type", type=str, default="mse",
                        choices=["mse", "mae", "cosine"], help="Error metric for anomaly score")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations (requires matplotlib)")

    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: str = None, device: str = "cuda"):
    """Load trained model from checkpoint."""
    checkpoint_path = Path(checkpoint_path)

    # Load config
    if config_path is None:
        config_path = checkpoint_path.parent / "config.json"

    with open(config_path) as f:
        config_data = json.load(f)

    model_config = WorldModelConfig(**config_data["model_config"])

    # Create model
    model = JEPAWorldModel(model_config)

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model config: obs_dim={model_config.obs_dim}, cmd_dim={model_config.cmd_dim}")

    return model, config_data


def compute_anomaly_scores(
    model: JEPAWorldModel,
    dataloader,
    device: str = "cuda",
    error_type: str = "mse",
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Compute anomaly scores for all samples in dataloader.

    Returns:
        scores: (N,) anomaly scores (higher = more anomalous)
        labels: (N,) binary labels (1 = anomaly, 0 = normal)
        metadata: list of metadata dicts for each sample
    """
    model.eval()

    all_scores = []
    all_labels = []
    all_metadata = []
    all_fault_types = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing anomaly scores"):
            obs_t = batch['obs_t'].to(device)
            cmd_t = batch['cmd_t'].to(device)
            obs_t1 = batch['obs_t1'].to(device)
            metadata = batch['metadata']

            # Get prediction and target embeddings
            z_t = model.encode(obs_t)
            z_pred = model.predict(z_t, cmd_t)
            z_target = model.encode_target(obs_t1)

            # Compute error per sample
            if error_type == "mse":
                # MSE over (seq_len, latent_dim), then mean over seq
                error = (z_pred - z_target).pow(2).mean(dim=-1).mean(dim=-1)
            elif error_type == "mae":
                error = (z_pred - z_target).abs().mean(dim=-1).mean(dim=-1)
            elif error_type == "cosine":
                # 1 - cosine similarity (so higher = more different)
                z_pred_flat = z_pred.view(z_pred.size(0), -1)
                z_target_flat = z_target.view(z_target.size(0), -1)
                cos_sim = F.cosine_similarity(z_pred_flat, z_target_flat, dim=-1)
                error = 1 - cos_sim

            all_scores.extend(error.cpu().numpy())

            # Extract labels
            for meta in metadata:
                is_anomaly = meta.get('is_anomaly', False)
                fault_type = meta.get('fault_type', 'normal')
                all_labels.append(1 if is_anomaly else 0)
                all_fault_types.append(fault_type)
                all_metadata.append(meta)

    return np.array(all_scores), np.array(all_labels), all_fault_types, all_metadata


def evaluate_binary_detection(
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Evaluate binary anomaly detection performance."""
    results = {}

    # Handle edge case: all same label
    if len(np.unique(labels)) < 2:
        logger.warning("Only one class present in labels, cannot compute AUC")
        results['auc_roc'] = float('nan')
        results['auc_pr'] = float('nan')
        return results

    # AUC-ROC
    results['auc_roc'] = roc_auc_score(labels, scores)

    # AUC-PR (Average Precision)
    results['auc_pr'] = average_precision_score(labels, scores)

    # Find optimal threshold using F1
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)

    results['best_threshold'] = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    results['best_f1'] = f1_scores[best_idx]
    results['precision_at_best'] = precision[best_idx]
    results['recall_at_best'] = recall[best_idx]

    # Apply threshold for classification metrics
    predictions = (scores >= results['best_threshold']).astype(int)

    results['confusion_matrix'] = confusion_matrix(labels, predictions).tolist()

    # Per-class metrics
    report = classification_report(labels, predictions, output_dict=True, zero_division=0)
    results['classification_report'] = report

    return results


def evaluate_per_fault_type(
    scores: np.ndarray,
    labels: np.ndarray,
    fault_types: list,
    threshold: float,
) -> dict:
    """Evaluate detection performance per fault type."""
    results = {}

    unique_faults = set(fault_types)

    for fault in unique_faults:
        mask = np.array([ft == fault for ft in fault_types])
        if mask.sum() == 0:
            continue

        fault_scores = scores[mask]
        fault_labels = labels[mask]

        if len(np.unique(fault_labels)) < 2:
            # All same class, just compute detection rate
            results[fault] = {
                'count': int(mask.sum()),
                'mean_score': float(fault_scores.mean()),
                'detection_rate': float((fault_scores >= threshold).mean()) if fault != 'normal' else None,
            }
        else:
            results[fault] = {
                'count': int(mask.sum()),
                'mean_score': float(fault_scores.mean()),
                'auc_roc': roc_auc_score(fault_labels, fault_scores),
            }

    return results


def compute_rollout_stability(
    model: JEPAWorldModel,
    dataloader,
    device: str = "cuda",
    max_steps: int = 50,
) -> dict:
    """
    Evaluate multi-step rollout stability.

    Good world models should:
    - Accumulate error slowly
    - Not diverge
    """
    model.eval()

    step_errors = [[] for _ in range(max_steps)]

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing rollout stability"):
            obs_t = batch['obs_t'].to(device)  # (B, T, obs_dim)
            cmd_t = batch['cmd_t'].to(device)  # (B, T, cmd_dim)

            B, T, _ = obs_t.shape

            if T < max_steps + 1:
                continue

            # Encode initial state
            z_t = model.encode(obs_t[:, :1, :])  # (B, 1, latent_dim)

            # Rollout
            for step in range(min(max_steps, T - 1)):
                # Predict next
                z_pred = model.predict(z_t, cmd_t[:, step:step+1, :])

                # Get target
                z_target = model.encode_target(obs_t[:, step+1:step+2, :])

                # Compute error
                error = (z_pred - z_target).pow(2).mean().item()
                step_errors[step].append(error)

                # Use prediction for next step (autoregressive)
                z_t = z_pred

    # Aggregate
    results = {
        'step_errors': [np.mean(errs) if errs else 0.0 for errs in step_errors],
        'step_stds': [np.std(errs) if errs else 0.0 for errs in step_errors],
        'error_ratio_10': None,
        'error_ratio_50': None,
    }

    # Error accumulation ratio
    if results['step_errors'][0] > 0:
        results['error_ratio_10'] = results['step_errors'][min(9, len(results['step_errors'])-1)] / results['step_errors'][0]
        results['error_ratio_50'] = results['step_errors'][-1] / results['step_errors'][0]

    return results


def visualize_results(
    scores: np.ndarray,
    labels: np.ndarray,
    fault_types: list,
    results: dict,
    output_dir: Path,
):
    """Generate visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed, skipping visualization")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Score distribution by class
    fig, ax = plt.subplots(figsize=(10, 6))

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    ax.hist(normal_scores, bins=50, alpha=0.7, label=f'Normal (n={len(normal_scores)})', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_scores)})', density=True)

    if 'best_threshold' in results:
        ax.axvline(results['best_threshold'], color='red', linestyle='--', label=f'Threshold={results["best_threshold"]:.4f}')

    ax.set_xlabel('Anomaly Score (Prediction Error)')
    ax.set_ylabel('Density')
    ax.set_title('Anomaly Score Distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.png', dpi=150)
    plt.close()

    # 2. ROC Curve
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(fpr, tpr, label=f'AUC = {results["auc_roc"]:.4f}')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'roc_curve.png', dpi=150)
        plt.close()

    # 3. Per-fault-type scores
    unique_faults = sorted(set(fault_types))
    fault_score_data = {ft: scores[np.array([f == ft for f in fault_types])] for ft in unique_faults}

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = range(len(unique_faults))
    bp = ax.boxplot([fault_score_data[ft] for ft in unique_faults], positions=positions, patch_artist=True)

    # Color normal differently
    for i, (patch, ft) in enumerate(zip(bp['boxes'], unique_faults)):
        if ft.lower() == 'normal':
            patch.set_facecolor('lightgreen')
        else:
            patch.set_facecolor('lightcoral')

    ax.set_xticklabels(unique_faults, rotation=45, ha='right')
    ax.set_xlabel('Fault Type')
    ax.set_ylabel('Anomaly Score')
    ax.set_title('Anomaly Scores by Fault Type')

    plt.tight_layout()
    plt.savefig(output_dir / 'scores_by_fault_type.png', dpi=150)
    plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load model
    model, config_data = load_model(args.checkpoint, args.config, device)

    # Setup output directory
    if args.output_dir is None:
        output_dir = Path(args.checkpoint).parent / "evaluation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading {args.data_source} dataset...")

    factorynet_config = FactoryNetConfig(
        dataset_name="Forgis/FactoryNet_Dataset",
        config_name="normalized",
        data_source=args.data_source,
        window_size=config_data["model_config"]["seq_len"],
        train_healthy_only=True,
    )

    data_config = WorldModelDataConfig(
        factorynet_config=factorynet_config,
        obs_mode="effort",
        cmd_mode="setpoint",
    )

    train_loader, val_loader, test_loader, info = create_world_model_dataloaders(
        data_config,
        batch_size=args.batch_size,
        num_workers=0,
    )

    # Select split
    eval_loader = test_loader if args.split == "test" else val_loader
    logger.info(f"Evaluating on {args.split} split ({len(eval_loader.dataset)} samples)")

    # Compute anomaly scores
    scores, labels, fault_types, metadata = compute_anomaly_scores(
        model, eval_loader, device, args.error_type
    )

    logger.info(f"Computed scores for {len(scores)} samples")
    logger.info(f"Normal: {(labels == 0).sum()}, Anomaly: {(labels == 1).sum()}")

    # Evaluate binary detection
    results = evaluate_binary_detection(scores, labels)

    logger.info("=" * 60)
    logger.info("BINARY ANOMALY DETECTION RESULTS")
    logger.info("=" * 60)
    logger.info(f"AUC-ROC: {results.get('auc_roc', 'N/A'):.4f}")
    logger.info(f"AUC-PR:  {results.get('auc_pr', 'N/A'):.4f}")
    logger.info(f"Best F1: {results.get('best_f1', 'N/A'):.4f}")
    logger.info(f"  Threshold: {results.get('best_threshold', 'N/A'):.4f}")
    logger.info(f"  Precision: {results.get('precision_at_best', 'N/A'):.4f}")
    logger.info(f"  Recall:    {results.get('recall_at_best', 'N/A'):.4f}")

    # Per-fault-type evaluation
    if 'best_threshold' in results:
        per_fault_results = evaluate_per_fault_type(
            scores, labels, fault_types, results['best_threshold']
        )
        results['per_fault_type'] = per_fault_results

        logger.info("\nPER-FAULT-TYPE RESULTS:")
        for fault, metrics in per_fault_results.items():
            logger.info(f"  {fault}: mean_score={metrics['mean_score']:.4f}, count={metrics['count']}")

    # Rollout stability (optional, can be slow)
    # rollout_results = compute_rollout_stability(model, eval_loader, device)
    # results['rollout'] = rollout_results

    # Save results
    results['config'] = {
        'checkpoint': args.checkpoint,
        'data_source': args.data_source,
        'split': args.split,
        'error_type': args.error_type,
        'timestamp': datetime.now().isoformat(),
    }

    results_path = output_dir / "anomaly_detection_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    # Visualize
    if args.visualize:
        visualize_results(scores, labels, fault_types, results, output_dir)

    return results


if __name__ == "__main__":
    main()

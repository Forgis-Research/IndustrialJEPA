#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Evaluate trained JEPA World Model.

This script provides:
1. Prediction accuracy metrics (single-step and multi-step)
2. Latent space visualization (t-SNE/UMAP colored by phase)
3. Reconstruction quality analysis
4. Rollout visualization

Usage:
    python scripts/evaluate_world_model.py --checkpoint results/world_model/best.pt
    python scripts/evaluate_world_model.py --checkpoint results/world_model/best.pt --visualize
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from industrialjepa.data import FactoryNetConfig, WorldModelDataConfig, create_world_model_dataloaders
from industrialjepa.model import JEPAWorldModel, WorldModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate JEPA World Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: checkpoint dir)")
    parser.add_argument("--rollout_steps", type=int, default=20, help="Steps for rollout evaluation")
    return parser.parse_args()


class WorldModelEvaluator:
    """Comprehensive evaluator for JEPA World Model."""

    def __init__(self, model: JEPAWorldModel, test_loader, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device

    @torch.no_grad()
    def compute_prediction_metrics(self) -> dict:
        """Compute single-step prediction accuracy."""
        all_pred_errors = []
        all_recon_errors = []

        for batch in tqdm(self.test_loader, desc="Computing prediction metrics"):
            obs_t = batch['obs_t'].to(self.device)
            cmd_t = batch['cmd_t'].to(self.device)
            obs_t1 = batch['obs_t1'].to(self.device)

            outputs = self.model(obs_t, cmd_t, obs_t1)

            # JEPA prediction error (latent space)
            z_pred = outputs['z_pred'][:, :-1, :]
            z_target = outputs['z_target'][:, 1:, :]
            pred_error = F.mse_loss(z_pred, z_target, reduction='none').mean(dim=-1)
            all_pred_errors.append(pred_error.cpu().numpy())

            # Reconstruction error (if decoder available)
            if 'obs_pred' in outputs:
                obs_pred = outputs['obs_pred'][:, :-1, :]
                obs_true = obs_t1[:, 1:, :]
                recon_error = F.mse_loss(obs_pred, obs_true, reduction='none').mean(dim=-1)
                all_recon_errors.append(recon_error.cpu().numpy())

        all_pred_errors = np.concatenate(all_pred_errors, axis=0)

        metrics = {
            'prediction_mse_mean': float(np.mean(all_pred_errors)),
            'prediction_mse_std': float(np.std(all_pred_errors)),
            'prediction_mse_median': float(np.median(all_pred_errors)),
        }

        if all_recon_errors:
            all_recon_errors = np.concatenate(all_recon_errors, axis=0)
            metrics['reconstruction_mse_mean'] = float(np.mean(all_recon_errors))
            metrics['reconstruction_mse_std'] = float(np.std(all_recon_errors))

        return metrics

    @torch.no_grad()
    def compute_rollout_metrics(self, max_steps: int = 20) -> dict:
        """Compute multi-step rollout accuracy."""
        step_errors = {s: [] for s in range(max_steps)}

        for batch in tqdm(self.test_loader, desc="Computing rollout metrics"):
            obs_t = batch['obs_t'].to(self.device)
            cmd_t = batch['cmd_t'].to(self.device)

            B, T, obs_dim = obs_t.shape
            context_len = T // 2
            pred_steps = min(max_steps, T - context_len)

            if pred_steps < 1:
                continue

            # Context and future commands
            obs_context = obs_t[:, :context_len, :]
            cmd_future = cmd_t[:, context_len:context_len + pred_steps, :]
            obs_true = obs_t[:, context_len:context_len + pred_steps, :]

            # Rollout
            z_rollout = self.model.rollout(obs_context, cmd_future, pred_steps)

            # Decode and compute errors
            if self.model.decoder is not None:
                obs_pred = self.model.decoder(z_rollout)

                for s in range(pred_steps):
                    error = F.mse_loss(obs_pred[:, s, :], obs_true[:, s, :]).item()
                    step_errors[s].append(error)

        # Average per step
        rollout_curve = []
        for s in range(max_steps):
            if step_errors[s]:
                rollout_curve.append(float(np.mean(step_errors[s])))
            else:
                break

        return {
            'rollout_curve': rollout_curve,
            'rollout_mean': float(np.mean(rollout_curve)) if rollout_curve else 0,
            'rollout_steps': len(rollout_curve),
        }

    @torch.no_grad()
    def extract_latent_representations(self, max_samples: int = 2000) -> dict:
        """Extract latent representations for visualization."""
        latents = []
        phases = []
        anomalies = []

        total = 0
        for batch in self.test_loader:
            if total >= max_samples:
                break

            obs_t = batch['obs_t'].to(self.device)
            metadata = batch['metadata']

            # Encode
            z = self.model.encoder(obs_t)  # (B, T, latent_dim)

            # Take middle timestep representation
            z_mid = z[:, z.size(1) // 2, :].cpu().numpy()
            latents.append(z_mid)

            # Extract metadata
            for m in metadata:
                phases.append(m.get('phase', 'unknown'))
                anomalies.append(m.get('is_anomaly', False))

            total += len(obs_t)

        latents = np.concatenate(latents, axis=0)[:max_samples]
        phases = phases[:max_samples]
        anomalies = anomalies[:max_samples]

        return {
            'latents': latents,
            'phases': phases,
            'anomalies': anomalies,
        }

    def evaluate_phase_classification(self, latent_data: dict) -> dict:
        """Linear probe: can we classify phase from latent?"""
        latents = latent_data['latents']
        phases = latent_data['phases']

        # Encode phases as integers
        unique_phases = list(set(phases))
        phase_to_idx = {p: i for i, p in enumerate(unique_phases)}
        labels = np.array([phase_to_idx[p] for p in phases])

        # Split
        n = len(latents)
        n_train = int(n * 0.8)
        indices = np.random.permutation(n)

        X_train = latents[indices[:n_train]]
        y_train = labels[indices[:n_train]]
        X_test = latents[indices[n_train:]]
        y_test = labels[indices[n_train:]]

        # Train linear classifier
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return {
            'phase_classification_accuracy': float(accuracy),
            'num_phases': len(unique_phases),
            'phases': unique_phases,
        }


def visualize_latent_space(latent_data: dict, output_path: Path):
    """Create t-SNE visualization of latent space."""
    latents = latent_data['latents']
    phases = latent_data['phases']
    anomalies = latent_data['anomalies']

    # t-SNE
    logger.info("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)

    # Plot by phase
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by phase
    unique_phases = list(set(phases))
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_phases)))
    phase_to_color = {p: c for p, c in zip(unique_phases, colors)}

    ax = axes[0]
    for phase in unique_phases:
        mask = np.array([p == phase for p in phases])
        ax.scatter(
            latents_2d[mask, 0],
            latents_2d[mask, 1],
            c=[phase_to_color[phase]],
            label=phase,
            alpha=0.6,
            s=20,
        )
    ax.set_title("Latent Space by Phase")
    ax.legend()
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Color by anomaly
    ax = axes[1]
    anomaly_colors = ['blue' if not a else 'red' for a in anomalies]
    ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=anomaly_colors, alpha=0.6, s=20)
    ax.set_title("Latent Space by Anomaly (red=anomaly)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(output_path / "latent_space_tsne.png", dpi=150)
    plt.close()

    logger.info(f"Saved latent space visualization to {output_path / 'latent_space_tsne.png'}")


def visualize_rollout_curve(rollout_metrics: dict, output_path: Path):
    """Plot rollout error over steps."""
    curve = rollout_metrics['rollout_curve']

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(curve) + 1), curve, 'b-o', linewidth=2, markersize=6)
    plt.xlabel("Prediction Steps Ahead")
    plt.ylabel("MSE")
    plt.title("Multi-Step Rollout Error")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "rollout_curve.png", dpi=150)
    plt.close()

    logger.info(f"Saved rollout curve to {output_path / 'rollout_curve.png'}")


def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config
    config_path = checkpoint_path.parent / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            saved_config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
    else:
        saved_config = {}
        logger.warning("No config.json found, using defaults")

    # Reconstruct model config
    model_config = checkpoint.get('config', None)
    if model_config is None:
        # Fallback to saved config
        mc = saved_config.get('model_config', {})
        model_config = WorldModelConfig(
            obs_dim=mc.get('obs_dim', 13),
            cmd_dim=mc.get('cmd_dim', 12),
            seq_len=mc.get('seq_len', 256),
            latent_dim=mc.get('latent_dim', 256),
            hidden_dim=mc.get('hidden_dim', 512),
            num_encoder_layers=mc.get('num_encoder_layers', 4),
            num_predictor_layers=mc.get('num_predictor_layers', 2),
            num_heads=mc.get('num_heads', 8),
            use_decoder=mc.get('use_decoder', True),
        )

    # Create model and load weights
    model = JEPAWorldModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")

    # Load test data
    data_info = saved_config.get('data_info', {})
    args_config = saved_config.get('args', {})

    factorynet_config = FactoryNetConfig(
        subset=args_config.get('subset', 'AURSAD'),
        window_size=model_config.seq_len,
        train_healthy_only=True,
    )

    data_config = WorldModelDataConfig(
        factorynet_config=factorynet_config,
        obs_mode="effort",
        cmd_mode="setpoint",
    )

    _, _, test_loader, _ = create_world_model_dataloaders(
        data_config,
        batch_size=32,
        num_workers=0,
    )

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate
    evaluator = WorldModelEvaluator(model, test_loader, device)

    # 1. Prediction metrics
    logger.info("Computing prediction metrics...")
    pred_metrics = evaluator.compute_prediction_metrics()
    logger.info(f"Prediction MSE: {pred_metrics['prediction_mse_mean']:.4f} ± {pred_metrics['prediction_mse_std']:.4f}")

    # 2. Rollout metrics
    logger.info("Computing rollout metrics...")
    rollout_metrics = evaluator.compute_rollout_metrics(max_steps=args.rollout_steps)
    logger.info(f"Rollout mean MSE: {rollout_metrics['rollout_mean']:.4f} over {rollout_metrics['rollout_steps']} steps")

    # 3. Latent space analysis
    logger.info("Extracting latent representations...")
    latent_data = evaluator.extract_latent_representations()

    phase_metrics = evaluator.evaluate_phase_classification(latent_data)
    logger.info(f"Phase classification accuracy: {phase_metrics['phase_classification_accuracy']:.2%}")

    # Combine results
    results = {
        'prediction': pred_metrics,
        'rollout': rollout_metrics,
        'phase_classification': phase_metrics,
    }

    # Save results
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Visualizations
    if args.visualize:
        logger.info("Generating visualizations...")
        visualize_latent_space(latent_data, output_dir)
        visualize_rollout_curve(rollout_metrics, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Single-step prediction MSE: {pred_metrics['prediction_mse_mean']:.4f}")
    if 'reconstruction_mse_mean' in pred_metrics:
        print(f"Reconstruction MSE:         {pred_metrics['reconstruction_mse_mean']:.4f}")
    print(f"Rollout MSE ({rollout_metrics['rollout_steps']} steps):   {rollout_metrics['rollout_mean']:.4f}")
    print(f"Phase classification acc:   {phase_metrics['phase_classification_accuracy']:.2%}")
    print("="*60)


if __name__ == "__main__":
    main()

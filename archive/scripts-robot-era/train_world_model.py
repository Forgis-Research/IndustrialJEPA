#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025-2026 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
Train JEPA World Model on FactoryNet dataset.

This script trains the world model to predict next-state dynamics in latent space.

Usage:
    python scripts/train_world_model.py
    python scripts/train_world_model.py --epochs 50 --batch_size 64
    python scripts/train_world_model.py --no-decoder  # JEPA only, no reconstruction

Quick validation (5 min):
    python scripts/train_world_model.py --epochs 5 --batch_size 32 --window_size 128

With wandb tracking:
    python scripts/train_world_model.py --wandb --wandb_project industrialjepa

Full FactoryNet training (GPU recommended):
    python scripts/train_world_model.py --dataset Forgis/FactoryNet_Dataset --epochs 100
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from industrialjepa.data import load_aursad_world_model, FactoryNetConfig, WorldModelDataConfig, create_world_model_dataloaders
from industrialjepa.model import JEPAWorldModel, WorldModelConfig, create_world_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train JEPA World Model")

    # Data
    parser.add_argument("--dataset", type=str, default="Forgis/FactoryNet_Dataset",
                        help="HuggingFace dataset name")
    parser.add_argument("--config_name", type=str, default="normalized",
                        help="Dataset config: 'normalized' or 'raw' (for full FactoryNet)")
    parser.add_argument("--subset", type=str, default=None,
                        help="Dataset subset: AURSAD, voraus-AD, rh20t, etc. (None=all)")
    parser.add_argument("--window_size", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--effort_type", type=str, default="torque",
                        help="Effort signal type: torque, current, velocity")
    parser.add_argument("--data-source", type=str, default="aursad",
                        help="Data source filter: aursad, voraus, cnc, hackathon (avoids schema mismatch)")

    # Model
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent space dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer dimension")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Encoder transformer layers")
    parser.add_argument("--num_predictor_layers", type=int, default=2, help="Predictor MLP layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--no-decoder", action="store_true", help="Disable reconstruction decoder")
    parser.add_argument("--recon_weight", type=float, default=0.1, help="Reconstruction loss weight")

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--ema_momentum", type=float, default=0.996, help="EMA momentum start")

    # Output
    parser.add_argument("--output_dir", type=str, default="results/world_model",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N epochs")

    # Debug
    parser.add_argument("--debug", action="store_true", help="Debug mode (small dataset)")

    # Wandb tracking
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="industrialjepa",
                        help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Wandb entity/team name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name (default: auto-generated)")

    return parser.parse_args()


class WorldModelTrainer:
    """Trainer for JEPA World Model."""

    def __init__(
        self,
        model: JEPAWorldModel,
        train_loader,
        val_loader,
        test_loader,
        args,
        device: str = "cuda",
        use_wandb: bool = False,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.device = device
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.lr / 100,
        )

        # Tracking
        self.history = {
            'train_loss': [],
            'train_jepa': [],
            'train_recon': [],
            'val_loss': [],
            'val_jepa': [],
            'val_recon': [],
        }
        self.best_val_loss = float('inf')
        self.global_step = 0

        # Output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0
        total_jepa = 0
        total_recon = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            obs_t = batch['obs_t'].to(self.device)
            cmd_t = batch['cmd_t'].to(self.device)
            obs_t1 = batch['obs_t1'].to(self.device)

            # Forward pass
            losses = self.model.compute_loss(obs_t, cmd_t, obs_t1)

            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update EMA
            self.model.update_ema()

            # Track losses
            total_loss += losses['total'].item()
            total_jepa += losses['L_jepa'].item()
            if 'L_recon' in losses:
                total_recon += losses['L_recon'].item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'jepa': f"{losses['L_jepa'].item():.4f}",
            })

            # Log to wandb (every 50 steps to avoid overhead)
            if self.use_wandb and self.global_step % 50 == 0:
                wandb.log({
                    'train/loss_step': losses['total'].item(),
                    'train/jepa_loss_step': losses['L_jepa'].item(),
                    'train/recon_loss_step': losses.get('L_recon', torch.tensor(0)).item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step,
                })

        return {
            'loss': total_loss / num_batches,
            'jepa': total_jepa / num_batches,
            'recon': total_recon / num_batches if total_recon > 0 else 0,
        }

    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()

        total_loss = 0
        total_jepa = 0
        total_recon = 0
        num_batches = 0

        for batch in self.val_loader:
            obs_t = batch['obs_t'].to(self.device)
            cmd_t = batch['cmd_t'].to(self.device)
            obs_t1 = batch['obs_t1'].to(self.device)

            losses = self.model.compute_loss(obs_t, cmd_t, obs_t1)

            total_loss += losses['total'].item()
            total_jepa += losses['L_jepa'].item()
            if 'L_recon' in losses:
                total_recon += losses['L_recon'].item()
            num_batches += 1

        return {
            'loss': total_loss / num_batches,
            'jepa': total_jepa / num_batches,
            'recon': total_recon / num_batches if total_recon > 0 else 0,
        }

    @torch.no_grad()
    def evaluate_rollout(self, steps: int = 10) -> dict:
        """Evaluate multi-step rollout accuracy."""
        self.model.eval()

        rollout_errors = []

        for batch in self.test_loader:
            obs_t = batch['obs_t'].to(self.device)  # (B, T, obs_dim)
            cmd_t = batch['cmd_t'].to(self.device)  # (B, T, cmd_dim)

            B, T, obs_dim = obs_t.shape

            # Use first half as context, predict second half
            context_len = T // 2
            pred_steps = min(steps, T - context_len)

            # Get initial context
            obs_context = obs_t[:, :context_len, :]
            cmd_future = cmd_t[:, context_len:context_len + pred_steps, :]

            # Rollout in latent space
            z_rollout = self.model.rollout(obs_context, cmd_future, pred_steps)

            # Decode predictions
            if self.model.decoder is not None:
                obs_pred = self.model.decoder(z_rollout)

                # Ground truth
                obs_true = obs_t[:, context_len:context_len + pred_steps, :]

                # Compute error per step
                for s in range(pred_steps):
                    step_error = F.mse_loss(obs_pred[:, s, :], obs_true[:, s, :]).item()
                    if len(rollout_errors) <= s:
                        rollout_errors.append([])
                    rollout_errors[s].append(step_error)

        # Average errors per step
        avg_errors = [np.mean(errors) for errors in rollout_errors]

        return {
            'rollout_errors': avg_errors,
            'mean_rollout_error': np.mean(avg_errors) if avg_errors else 0,
        }

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.args.epochs} epochs")
        logger.info(f"Output directory: {self.output_dir}")

        for epoch in range(1, self.args.epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_jepa'].append(train_metrics['jepa'])
            self.history['train_recon'].append(train_metrics['recon'])

            # Validate
            val_metrics = self.validate()
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_jepa'].append(val_metrics['jepa'])
            self.history['val_recon'].append(val_metrics['recon'])

            # Update scheduler
            self.scheduler.step()

            # Log
            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, train_jepa={train_metrics['jepa']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, val_jepa={val_metrics['jepa']:.4f}"
            )

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_metrics['loss'],
                    'train/jepa_loss': train_metrics['jepa'],
                    'train/recon_loss': train_metrics['recon'],
                    'val/loss': val_metrics['loss'],
                    'val/jepa_loss': val_metrics['jepa'],
                    'val/recon_loss': val_metrics['recon'],
                    'best_val_loss': self.best_val_loss,
                })

            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint('best.pt')
                logger.info(f"  New best model saved (val_loss={val_metrics['loss']:.4f})")

            # Periodic save
            if epoch % self.args.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')

        # Final evaluation
        logger.info("Running final evaluation...")
        rollout_metrics = self.evaluate_rollout(steps=10)
        logger.info(f"Rollout errors by step: {[f'{e:.4f}' for e in rollout_metrics['rollout_errors']]}")
        logger.info(f"Mean rollout error: {rollout_metrics['mean_rollout_error']:.4f}")

        # Log final metrics to wandb
        if self.use_wandb:
            wandb.log({
                'final/mean_rollout_error': rollout_metrics['mean_rollout_error'],
                'final/best_val_loss': self.best_val_loss,
            })
            # Log rollout curve
            for step, error in enumerate(rollout_metrics['rollout_errors']):
                wandb.log({f'rollout/step_{step+1}': error})

        # Save final model and history
        self.save_checkpoint('final.pt')
        self.save_history()

        return self.history

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.model.config,
            'best_val_loss': self.best_val_loss,
        }, path)

    def save_history(self):
        """Save training history."""
        path = self.output_dir / 'history.json'
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


def main():
    args = parse_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.subset or getattr(args, 'data_source', None) or 'default'
    args.output_dir = f"{args.output_dir}/{run_name}_{timestamp}"

    # Load data
    logger.info(f"Loading {run_name} dataset...")

    if args.debug:
        # Smaller dataset for debugging
        args.window_size = 64
        args.batch_size = 8

    factorynet_config = FactoryNetConfig(
        dataset_name=args.dataset,
        config_name=args.config_name,
        subset=args.subset,
        data_source=getattr(args, 'data_source', 'aursad'),
        window_size=args.window_size,
        effort_signals=[args.effort_type, "current", "velocity"],
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
        num_workers=0,  # Windows compatibility
    )

    logger.info(f"Data loaded: obs_dim={info['obs_dim']}, cmd_dim={info['cmd_dim']}")
    logger.info(f"Train: {info['train_size']} windows, Val: {info['val_size']}, Test: {info['test_size']}")

    # Create model
    model_config = WorldModelConfig(
        obs_dim=info['obs_dim'],
        cmd_dim=info['cmd_dim'],
        seq_len=args.window_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_predictor_layers=args.num_predictor_layers,
        num_heads=args.num_heads,
        use_decoder=not args.no_decoder,
        reconstruction_weight=args.recon_weight,
        ema_momentum=args.ema_momentum,
    )

    model = JEPAWorldModel(model_config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {num_params:,} trainable parameters")

    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_dict = {
        'args': vars(args),
        'model_config': {
            'obs_dim': model_config.obs_dim,
            'cmd_dim': model_config.cmd_dim,
            'seq_len': model_config.seq_len,
            'latent_dim': model_config.latent_dim,
            'hidden_dim': model_config.hidden_dim,
            'num_encoder_layers': model_config.num_encoder_layers,
            'num_predictor_layers': model_config.num_predictor_layers,
            'num_heads': model_config.num_heads,
            'use_decoder': model_config.use_decoder,
        },
        'data_info': info,
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Initialize wandb if requested
    use_wandb = False
    if args.wandb:
        if not WANDB_AVAILABLE:
            logger.error("=" * 60)
            logger.error("ERROR: --wandb flag passed but wandb is not installed!")
            logger.error("Run: pip install wandb && wandb login")
            logger.error("=" * 60)
            sys.exit(1)

        # Check if logged in
        if wandb.api.api_key is None:
            logger.error("=" * 60)
            logger.error("ERROR: --wandb flag passed but not logged in!")
            logger.error("Run: wandb login")
            logger.error("=" * 60)
            sys.exit(1)

        run_name = args.wandb_run_name or f"{run_name}_{timestamp}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config_dict,
            dir=str(output_dir),
        )
        # Log code
        wandb.run.log_code(".")
        use_wandb = True
        logger.info(f"Wandb initialized: {wandb.run.url}")

    # Train
    trainer = WorldModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        args=args,
        device=device,
        use_wandb=use_wandb,
    )

    history = trainer.train()

    # Finish wandb run
    if use_wandb:
        wandb.finish()

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss:   {history['val_loss'][-1]:.4f}")
    print(f"Best val loss:    {trainer.best_val_loss:.4f}")
    print(f"Final JEPA loss:  {history['val_jepa'][-1]:.4f}")
    if history['val_recon'][-1] > 0:
        print(f"Final recon loss: {history['val_recon'][-1]:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()

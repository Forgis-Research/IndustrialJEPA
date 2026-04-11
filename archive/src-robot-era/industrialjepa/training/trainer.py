# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Main Trainer for Industrial World Model.

Handles:
- Multi-stage training (tokenizer → dynamics → full)
- Mixed precision training
- Gradient accumulation
- Checkpointing
- Logging (WandB/TensorBoard)
- Distributed training support
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, List, Callable, Union, Literal
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

from industrialjepa.model.industrial_world_lm import IndustrialWorldLM
from industrialjepa.model.config import IndustrialWorldLMConfig
from industrialjepa.training.objectives import WorldModelLoss, LossOutput
from industrialjepa.training.scheduler import get_scheduler


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training stages
    stage: Literal["tokenizer", "dynamics", "full", "finetune"] = "full"

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Schedule
    scheduler: Literal["cosine", "linear", "constant", "restart"] = "cosine"
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1

    # Training loop
    num_epochs: int = 100
    steps_per_epoch: Optional[int] = None  # None = full epoch
    gradient_accumulation_steps: int = 1
    eval_steps: int = 1000
    save_steps: int = 5000
    log_steps: int = 100

    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # "float16" or "bfloat16"

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_total_limit: int = 5
    resume_from: Optional[str] = None

    # Logging
    use_wandb: bool = False
    wandb_project: str = "industrial-world-lm"
    wandb_run_name: Optional[str] = None
    use_tensorboard: bool = True

    # Loss weights (per stage)
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "dynamics": 1.0,
        "recon": 1.0,
        "kl": 0.1,
        "contrastive": 0.1,
        "mtp": 0.0,
    })

    # Stage-specific settings
    tokenizer_epochs: int = 20
    dynamics_epochs: int = 30
    freeze_tokenizer_after: int = 10  # Freeze after this epoch in full training

    # Distributed
    local_rank: int = -1
    world_size: int = 1


class IndustrialWorldLMTrainer:
    """
    Trainer for Industrial World Model.

    Supports multi-stage training:
    1. Tokenizer: Train VQ-VAE for time series tokenization
    2. Dynamics: Train backbone for dynamics prediction
    3. Full: End-to-end training with all objectives
    4. Finetune: Task-specific fine-tuning
    """

    def __init__(
        self,
        model: IndustrialWorldLM,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        Args:
            model: IndustrialWorldLM model
            config: Training configuration
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            compute_metrics: Function to compute additional metrics
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics

        # Setup device
        self.device = next(model.parameters()).device

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Setup scheduler
        total_steps = self._get_total_steps()
        warmup_steps = int(total_steps * config.warmup_ratio)
        min_lr = config.learning_rate * config.min_lr_ratio

        self.scheduler = get_scheduler(
            config.scheduler,
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )

        # Setup loss
        self.loss_fn = WorldModelLoss(**config.loss_weights)

        # Mixed precision
        self.use_amp = config.use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.amp_dtype = torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float16
            self.scaler = GradScaler(enabled=config.amp_dtype == "float16")
        else:
            self.scaler = None

        # Logging
        self.logger = self._setup_logging()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Resume if specified
        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and LayerNorm
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
        )

    def _get_total_steps(self) -> int:
        """Calculate total training steps."""
        steps_per_epoch = self.config.steps_per_epoch or len(self.train_dataloader)
        steps_per_epoch = steps_per_epoch // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def _setup_logging(self) -> Dict:
        """Setup logging backends."""
        logger = {"tensorboard": None, "wandb": None}

        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                logger["tensorboard"] = SummaryWriter(
                    log_dir=str(self.output_dir / "tensorboard")
                )
            except ImportError:
                print("TensorBoard not available")

        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=asdict(self.config),
                )
                logger["wandb"] = wandb
            except ImportError:
                print("WandB not available")

        return logger

    def _log_metrics(self, metrics: Dict, step: int, prefix: str = "train"):
        """Log metrics to all backends."""
        # Add prefix
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # TensorBoard
        if self.logger["tensorboard"]:
            for k, v in metrics.items():
                self.logger["tensorboard"].add_scalar(k, v, step)

        # WandB
        if self.logger["wandb"]:
            self.logger["wandb"].log(metrics, step=step)

    def train(self):
        """Run full training loop."""
        print(f"Starting training: {self.config.num_epochs} epochs")
        print(f"Stage: {self.config.stage}")
        print(f"Total steps: {self._get_total_steps()}")

        # Stage-specific setup
        self._setup_stage()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            self._train_epoch()

            # Evaluation
            if self.eval_dataloader is not None:
                eval_loss = self._evaluate()
                self._log_metrics({"loss": eval_loss}, self.global_step, "eval")

                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best")

            # Stage transitions
            self._handle_stage_transitions()

        # Final save
        self._save_checkpoint("final")
        print("Training complete!")

    def _setup_stage(self):
        """Configure model for current training stage."""
        if self.config.stage == "tokenizer":
            # Only train tokenizer (VQ-VAE)
            self._freeze_except(["tokenizer"])
            self.config.loss_weights = {
                "dynamics": 0.0, "recon": 1.0, "kl": 0.0,
                "contrastive": 0.0, "mtp": 0.0
            }

        elif self.config.stage == "dynamics":
            # Freeze tokenizer, train backbone
            self._freeze_modules(["tokenizer"])
            self.config.loss_weights = {
                "dynamics": 1.0, "recon": 0.0, "kl": 0.1,
                "contrastive": 0.1, "mtp": 0.0
            }

        elif self.config.stage == "finetune":
            # Finetune everything (lower LR)
            for param in self.model.parameters():
                param.requires_grad = True

    def _freeze_modules(self, module_names: List[str]):
        """Freeze specified modules."""
        for name, param in self.model.named_parameters():
            for mod_name in module_names:
                if mod_name in name:
                    param.requires_grad = False

    def _freeze_except(self, module_names: List[str]):
        """Freeze all except specified modules."""
        for name, param in self.model.named_parameters():
            should_train = any(mod_name in name for mod_name in module_names)
            param.requires_grad = should_train

    def _handle_stage_transitions(self):
        """Handle transitions between training stages."""
        if self.config.stage == "full":
            # Optionally freeze tokenizer after some epochs
            if self.epoch == self.config.freeze_tokenizer_after:
                print(f"Freezing tokenizer at epoch {self.epoch}")
                self._freeze_modules(["tokenizer"])

    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0.0
        num_steps = 0

        progress = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            total=self.config.steps_per_epoch,
        )

        for step, batch in enumerate(progress):
            if self.config.steps_per_epoch and step >= self.config.steps_per_epoch:
                break

            loss, metrics = self._train_step(batch)
            epoch_loss += loss

            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step()
                self.global_step += 1
                num_steps += 1

                # Logging
                if self.global_step % self.config.log_steps == 0:
                    metrics["lr"] = self.scheduler.get_last_lr()[0]
                    self._log_metrics(metrics, self.global_step, "train")

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                # Update progress
                progress.set_postfix(loss=loss, **{k: f"{v:.4f}" for k, v in metrics.items()})

        # Log epoch metrics
        avg_loss = epoch_loss / max(num_steps, 1)
        self._log_metrics({"epoch_loss": avg_loss}, self.global_step, "train")

    def _train_step(self, batch: Dict) -> tuple:
        """Single training step."""
        # Move to device
        x = batch["x"].to(self.device)
        domain_labels = batch.get("domain_idx")
        if domain_labels is not None:
            domain_labels = torch.tensor(domain_labels).to(self.device)

        # Forward pass with AMP
        with autocast(enabled=self.use_amp, dtype=self.amp_dtype if self.use_amp else None):
            # Forward through model
            output = self.model(
                x,
                domain=batch.get("domain"),
                compute_loss=True,
                predict_next=True,
            )

            # Compute loss
            loss_output = self.loss_fn(
                output,
                targets=x,
                domain_labels=domain_labels,
            )

            loss = loss_output.total / self.config.gradient_accumulation_steps

        # Backward pass
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps, loss_output.metrics

    def _optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate model on validation set."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_metrics = {}

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            x = batch["x"].to(self.device)

            with autocast(enabled=self.use_amp, dtype=self.amp_dtype if self.use_amp else None):
                output = self.model(x, compute_loss=True)
                loss_output = self.loss_fn(output, targets=x)

            total_loss += loss_output.total.item()
            num_batches += 1

            # Accumulate metrics
            for k, v in loss_output.metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = 0.0
                all_metrics[k] += v

        # Average metrics
        avg_loss = total_loss / max(num_batches, 1)
        for k in all_metrics:
            all_metrics[k] /= max(num_batches, 1)

        # Custom metrics
        if self.compute_metrics:
            custom_metrics = self.compute_metrics(self.model, self.eval_dataloader)
            all_metrics.update(custom_metrics)

        self._log_metrics(all_metrics, self.global_step, "eval")

        return avg_loss

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.output_dir / f"checkpoint_{name}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(checkpoint_path / "model.pt"))

        # Save training state
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": asdict(self.config),
        }

        if self.scaler:
            state["scaler"] = self.scaler.state_dict()

        torch.save(state, checkpoint_path / "trainer_state.pt")

        # Save config
        with open(checkpoint_path / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        print(f"Saved checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def _load_checkpoint(self, path: str):
        """Load checkpoint for resuming."""
        checkpoint_path = Path(path)

        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            self.model = IndustrialWorldLM.from_pretrained(
                str(model_path),
                device=str(self.device),
            )

        # Load training state
        state_path = checkpoint_path / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_eval_loss = state["best_eval_loss"]

            if self.scaler and "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])

        print(f"Resumed from checkpoint: {checkpoint_path}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint_step_*"),
            key=lambda x: int(x.name.split("_")[-1]),
        )

        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest}")


def train_industrial_world_lm(
    model_config: IndustrialWorldLMConfig,
    training_config: TrainingConfig,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    device: str = "cuda",
) -> IndustrialWorldLM:
    """
    Convenience function to train Industrial World Model.

    Args:
        model_config: Model configuration
        training_config: Training configuration
        train_dataloader: Training data
        eval_dataloader: Validation data
        device: Device to train on

    Returns:
        Trained model
    """
    # Create model
    model = IndustrialWorldLM(model_config, device=device)

    # Create trainer
    trainer = IndustrialWorldLMTrainer(
        model=model,
        config=training_config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    # Train
    trainer.train()

    return model

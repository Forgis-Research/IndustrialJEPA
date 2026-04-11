# SPDX-FileCopyrightText: 2025 Industrial World Model Authors
# SPDX-License-Identifier: MIT

"""
Learning Rate Schedulers for Industrial World Model.

Implements warmup + decay schedules commonly used in large model training.
"""

import math
from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing with linear warmup.

    Learning rate schedule:
    - Linear warmup from 0 to base_lr over warmup_steps
    - Cosine decay from base_lr to min_lr over remaining steps
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        warmup_start_lr: float = 1e-8,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            min_lr: Minimum learning rate after decay
            warmup_start_lr: Starting LR for warmup
            last_epoch: Last epoch number (for resuming)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rate for each param group."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / max(1, self.warmup_steps)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            current_decay_step = step - self.warmup_steps
            alpha = current_decay_step / max(1, decay_steps)

            # Cosine schedule
            cosine_decay = 0.5 * (1 + math.cos(math.pi * alpha))

            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """
    Linear warmup with optional linear decay.

    Simpler alternative to cosine schedule.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: Optional[int] = None,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Number of warmup steps
            total_steps: Total steps (None = no decay after warmup)
            min_lr: Minimum LR at end of training
            warmup_start_lr: Starting LR for warmup
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rate."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / max(1, self.warmup_steps)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]
        elif self.total_steps is not None:
            # Linear decay
            decay_steps = self.total_steps - self.warmup_steps
            current_decay_step = step - self.warmup_steps
            alpha = current_decay_step / max(1, decay_steps)

            return [
                max(self.min_lr, base_lr * (1 - alpha))
                for base_lr in self.base_lrs
            ]
        else:
            # Constant after warmup
            return self.base_lrs


class WarmupRestartScheduler(_LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).

    Useful for escaping local minima during long training.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        cycle_length: int,
        cycle_mult: float = 1.0,
        min_lr: float = 1e-6,
        max_cycles: Optional[int] = None,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_steps: Warmup steps (first cycle only)
            cycle_length: Initial cycle length
            cycle_mult: Multiply cycle length after each restart
            min_lr: Minimum LR
            max_cycles: Maximum number of cycles (None = unlimited)
            last_epoch: Last epoch number
        """
        self.warmup_steps = warmup_steps
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.min_lr = min_lr
        self.max_cycles = max_cycles

        self.current_cycle = 0
        self.cycle_step = 0
        self.current_cycle_length = cycle_length

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute current learning rate."""
        step = self.last_epoch

        if step < self.warmup_steps:
            # Linear warmup
            alpha = step / max(1, self.warmup_steps)
            return [alpha * base_lr for base_lr in self.base_lrs]

        # Steps since warmup
        post_warmup_step = step - self.warmup_steps

        # Find current cycle
        cycle_start = 0
        cycle_length = self.cycle_length
        cycle = 0

        while post_warmup_step >= cycle_start + cycle_length:
            cycle_start += cycle_length
            cycle_length = int(cycle_length * self.cycle_mult)
            cycle += 1

            if self.max_cycles is not None and cycle >= self.max_cycles:
                # Stay at min_lr after max cycles
                return [self.min_lr for _ in self.base_lrs]

        # Position within current cycle
        cycle_pos = post_warmup_step - cycle_start
        alpha = cycle_pos / max(1, cycle_length)

        # Cosine decay within cycle
        cosine_decay = 0.5 * (1 + math.cos(math.pi * alpha))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_decay
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
    **kwargs,
) -> _LRScheduler:
    """
    Factory function to create schedulers.

    Args:
        name: Scheduler name ("cosine", "linear", "constant", "restart")
        optimizer: Optimizer to schedule
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr: Minimum learning rate
        **kwargs: Additional scheduler-specific arguments

    Returns:
        Configured LR scheduler
    """
    schedulers = {
        "cosine": CosineWarmupScheduler,
        "linear": LinearWarmupScheduler,
        "restart": WarmupRestartScheduler,
    }

    if name == "constant":
        return LinearWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=None,  # No decay
            min_lr=min_lr,
        )

    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")

    scheduler_class = schedulers[name]

    return scheduler_class(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=min_lr,
        **kwargs,
    )

# SPDX-FileCopyrightText: 2025-2026 Industrial JEPA Authors
# SPDX-License-Identifier: MIT

"""
World Model Dataset wrapper for JEPA training.

Converts FactoryNet (setpoint, effort) pairs into (obs_t, cmd_t, obs_t1) tuples
suitable for next-state prediction.

For the world model:
- observation = effort (forces/torques experienced)
- command = setpoint (commanded positions/velocities)

The model learns: given current effort and setpoint, predict next effort.
This captures the physics of how commands translate to forces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .factorynet import FactoryNetConfig, FactoryNetDataset


@dataclass
class WorldModelDataConfig:
    """Configuration for world model data loading."""

    # FactoryNet config
    factorynet_config: FactoryNetConfig = None

    # What to use as observation (what we predict)
    # Options: "effort", "feedback", "both"
    obs_mode: Literal["effort", "feedback", "both"] = "effort"

    # What to use as command (conditioning input)
    # Options: "setpoint", "setpoint_effort" (include current effort in command)
    cmd_mode: Literal["setpoint", "setpoint_effort"] = "setpoint"

    # Prediction horizon (how many steps ahead to predict)
    pred_horizon: int = 1

    def __post_init__(self):
        if self.factorynet_config is None:
            self.factorynet_config = FactoryNetConfig()


class WorldModelDataset(Dataset):
    """
    Dataset wrapper that provides (obs_t, cmd_t, obs_t1) tuples.

    This wraps FactoryNetDataset and reformats the data for world model training.

    For each window, we create overlapping (current, command, next) tuples
    that the world model uses to learn dynamics.
    """

    def __init__(
        self,
        config: WorldModelDataConfig,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.config = config
        self.split = split

        # Load base dataset
        self.base_dataset = FactoryNetDataset(
            config.factorynet_config,
            split=split,
        )

        # Store dimensions
        self.obs_dim = self.base_dataset.actual_effort_dim
        self.cmd_dim = self.base_dataset.actual_setpoint_dim

        if config.cmd_mode == "setpoint_effort":
            self.cmd_dim += self.obs_dim

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            dict with:
                obs_t: (seq_len, obs_dim) - current observations (effort)
                cmd_t: (seq_len, cmd_dim) - commands (setpoint)
                obs_t1: (seq_len, obs_dim) - next observations (shifted effort)
                metadata: episode info, masks, etc.
        """
        setpoint, effort, metadata = self.base_dataset[idx]

        # setpoint: (seq_len, setpoint_dim)
        # effort: (seq_len, effort_dim)

        # Use actual dimensions (not padded)
        actual_setpoint_dim = metadata['actual_setpoint_dim']
        actual_effort_dim = metadata['actual_effort_dim']

        setpoint = setpoint[:, :actual_setpoint_dim]
        effort = effort[:, :actual_effort_dim]

        # Current observation = effort (forces experienced)
        obs_t = effort  # (seq_len, effort_dim)

        # Command = setpoint (optionally + current effort)
        if self.config.cmd_mode == "setpoint":
            cmd_t = setpoint
        else:  # setpoint_effort
            cmd_t = torch.cat([setpoint, effort], dim=-1)

        # Next observation = shifted effort
        # obs_t1[t] = obs_t[t+1], with last position using last value
        obs_t1 = torch.roll(effort, shifts=-self.config.pred_horizon, dims=0)
        # For positions beyond sequence, repeat last valid value
        obs_t1[-self.config.pred_horizon:] = effort[-1:]

        return {
            'obs_t': obs_t,
            'cmd_t': cmd_t,
            'obs_t1': obs_t1,
            'metadata': metadata,
        }


def collate_world_model(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate function for world model batches."""
    obs_t = torch.stack([b['obs_t'] for b in batch])
    cmd_t = torch.stack([b['cmd_t'] for b in batch])
    obs_t1 = torch.stack([b['obs_t1'] for b in batch])
    metadata = [b['metadata'] for b in batch]

    return {
        'obs_t': obs_t,
        'cmd_t': cmd_t,
        'obs_t1': obs_t1,
        'metadata': metadata,
    }


def create_world_model_dataloaders(
    config: WorldModelDataConfig,
    batch_size: int = 32,
    num_workers: int = 0,  # 0 for Windows compatibility
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create dataloaders for world model training.

    Args:
        config: WorldModelDataConfig
        batch_size: batch size
        num_workers: number of data loading workers

    Returns:
        train_loader, val_loader, test_loader, info_dict
    """
    train_dataset = WorldModelDataset(config, split="train")
    val_dataset = WorldModelDataset(config, split="val")
    test_dataset = WorldModelDataset(config, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_world_model,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_world_model,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_world_model,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Info about dimensions
    info = {
        'obs_dim': train_dataset.obs_dim,
        'cmd_dim': train_dataset.cmd_dim,
        'seq_len': config.factorynet_config.window_size,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'test_size': len(test_dataset),
    }

    return train_loader, val_loader, test_loader, info


def load_aursad_world_model(
    window_size: int = 256,
    batch_size: int = 32,
    effort_type: str = "torque",
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Convenience function to load AURSAD for world model training.

    Args:
        window_size: sequence length
        batch_size: batch size
        effort_type: which effort signal to use ("torque", "current", etc.)

    Returns:
        train_loader, val_loader, test_loader, info_dict

    Example:
        >>> train, val, test, info = load_aursad_world_model()
        >>> print(f"obs_dim={info['obs_dim']}, cmd_dim={info['cmd_dim']}")
        >>> for batch in train:
        ...     obs_t = batch['obs_t']  # (B, T, obs_dim)
        ...     cmd_t = batch['cmd_t']  # (B, T, cmd_dim)
        ...     obs_t1 = batch['obs_t1']  # (B, T, obs_dim)
        ...     break
    """
    factorynet_config = FactoryNetConfig(
        subset="AURSAD",
        window_size=window_size,
        effort_signals=[effort_type, "current", "velocity"],  # Fallback order
        train_healthy_only=True,
    )

    config = WorldModelDataConfig(
        factorynet_config=factorynet_config,
        obs_mode="effort",
        cmd_mode="setpoint",
    )

    return create_world_model_dataloaders(config, batch_size=batch_size)

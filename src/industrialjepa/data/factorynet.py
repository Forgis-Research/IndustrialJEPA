# SPDX-FileCopyrightText: 2025-2026 Forgis AG
# SPDX-License-Identifier: MIT

"""FactoryNet dataset loader for IndustrialJEPA.

FactoryNet provides causally-structured industrial time series data with:
- Setpoint: What the controller commanded (position, velocity)
- Effort: What the machine expended (motor current/torque)
- Feedback: What actually happened (measured position) [OPTIONAL]

The causal chain: Setpoint → Effort → Feedback
JEPA learns: Setpoint → Effort (physics-based, transferable)

This loader supports both single-machine and multi-machine configurations
for JEPA training and cross-machine transfer experiments.

## Handling Data Heterogeneity

Different robots have different sensors:
- AURSAD (UR3e): 6-DOF, has torque
- voraus-AD (Yu-Cobot): 6-DOF, has current
- NASA Milling (CNC): 3-axis, has force
- RH20T (Franka): 7-DOF, has torque
- REASSEMBLE (Franka): 7-DOF, has torque

We use a UNIFIED SCHEMA with:
1. Fixed output dimensions (max DOF across all robots)
2. Zero-padding for missing joints/signals
3. Validity mask to indicate which dimensions are real vs padded
4. Per-signal-type semantic grouping (all "effort" signals → unified representation)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


# Metadata file paths per subset (keys must match data_dir values)
METADATA_FILES = {
    "aursad": "metadata/aursad_metadata.json",
    "voraus": "metadata/voraus_metadata.json",
    "nasa_milling": "nasa_milling/nasa_milling_metadata.json",
    "rh20t": "rh20t/rh20t_metadata.json",
    "reassemble": "reassemble/reassemble_metadata.json",
}


# =============================================================================
# Unified Schema Definition
# =============================================================================
# Maximum dimensions across all robots (7-DOF Franka is largest)
MAX_DOF = 7

# Column patterns - we search for ANY of these and unify them
SETPOINT_PATTERNS = {
    "position": [f"setpoint_pos_{i}" for i in range(MAX_DOF)],
    "velocity": [f"setpoint_vel_{i}" for i in range(MAX_DOF)],
    "acceleration": [f"setpoint_acc_{i}" for i in range(MAX_DOF)],
}

# Effort signals - semantically equivalent (energy/force expended)
# The model should learn that torque ≈ current × motor_constant
EFFORT_PATTERNS = {
    "torque": [f"effort_torque_{i}" for i in range(MAX_DOF)],
    "current": [f"effort_current_{i}" for i in range(MAX_DOF)],
    "voltage": [f"effort_voltage_{i}" for i in range(MAX_DOF)],
    # Velocity-based effort (RH20T uses this)
    "velocity": [f"effort_vel_{i}" for i in range(MAX_DOF)],
    # Cartesian forces (for CNC/end-effector)
    "force": ["effort_force_x", "effort_force_y", "effort_force_z"],
}

# Feedback signals (optional - not used in core JEPA objective)
FEEDBACK_PATTERNS = {
    "position": [f"feedback_pos_{i}" for i in range(MAX_DOF)],
    "velocity": [f"feedback_vel_{i}" for i in range(MAX_DOF)],
}

METADATA_COLS = ["dataset_source", "machine_type", "episode_id", "ctx_anomaly_label"]


@dataclass
class FactoryNetConfig:
    """Configuration for FactoryNet dataset loading."""

    # Dataset source
    dataset_name: str = "Forgis/factorynet-hackathon"
    subset: Optional[str] = None  # None = all, or "AURSAD", "voraus-AD", etc.

    # Sequence parameters
    window_size: int = 256
    stride: int = 128  # Overlap between windows

    # Normalization
    normalize: bool = True
    norm_mode: Literal["episode", "global", "none"] = "episode"

    # Column selection (semantic groups, not specific columns)
    # The loader will find whichever columns are available
    setpoint_signals: list[str] = field(default_factory=lambda: ["position", "velocity"])
    # Try effort signal types in order of preference
    effort_signals: list[str] = field(default_factory=lambda: ["torque", "current", "velocity"])

    # Unified output dimensions (for cross-dataset training)
    # All outputs padded to these dims with validity masks
    unified_setpoint_dim: int = MAX_DOF * 2  # pos + vel for 7 joints = 14
    unified_effort_dim: int = MAX_DOF  # 7 effort signals

    # Data splits
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # For fault detection: only train on healthy data
    train_healthy_only: bool = True

    # AURSAD-specific: How to handle loosening/tightening phases
    # The dataset has paired operations: loosening (prepare) → tightening (screw in)
    # Options:
    #   "both": Use both phases (default) - both follow valid physics
    #   "tightening_only": Use only tightening (where faults occur)
    #   "merge": Merge paired phases into single episodes (TODO)
    aursad_phase_handling: Literal["both", "tightening_only", "merge"] = "both"


class FactoryNetDataset(Dataset):
    """PyTorch Dataset for FactoryNet data.

    Loads industrial time series with Setpoint/Effort/Feedback structure
    for JEPA training. Returns (setpoint_window, effort_window) pairs.

    Example:
        >>> config = FactoryNetConfig(subset="AURSAD", window_size=256)
        >>> dataset = FactoryNetDataset(config, split="train")
        >>> setpoint, effort, metadata = dataset[0]
        >>> print(setpoint.shape)  # (window_size, num_setpoint_features)
    """

    def __init__(
        self,
        config: FactoryNetConfig,
        split: Literal["train", "val", "test"] = "train",
    ):
        self.config = config
        self.split = split

        # Load dataset from HuggingFace
        logger.info(f"Loading FactoryNet from {config.dataset_name}")
        self._load_data()

        # Build column lists
        self._setup_columns()

        # Create episode index and windows
        self._build_episode_index()
        self._create_windows()

        # Compute normalization statistics
        if config.normalize and config.norm_mode == "global":
            self._compute_global_stats()

        logger.info(
            f"FactoryNetDataset initialized: {len(self)} windows, "
            f"{len(self.episode_ids)} episodes, split={split}"
        )

    def _load_data(self):
        """Load data from HuggingFace datasets."""
        # Map subset names to data_dir paths (must match HuggingFace repo folders)
        subset_to_datadir = {
            "AURSAD": "aursad",
            "aursad": "aursad",
            "voraus-AD": "voraus",
            "voraus-ad": "voraus",
            "voraus": "voraus",
            "NASA": "nasa_milling",
            "nasa-milling": "nasa_milling",
            "nasa_milling": "nasa_milling",
            "RH20T": "rh20t",
            "rh20t": "rh20t",
            "REASSEMBLE": "reassemble",
            "reassemble": "reassemble",
        }

        data_dir = None
        if self.config.subset:
            data_dir = subset_to_datadir.get(self.config.subset, self.config.subset.lower())

        try:
            # Load with data_dir for subset selection (required for Forgis/factorynet-hackathon)
            logger.info(f"Loading {self.config.dataset_name}" + (f" subset={data_dir}" if data_dir else ""))
            self.hf_dataset = load_dataset(
                self.config.dataset_name,
                data_dir=data_dir,
                split="train",
            )
        except Exception as e:
            logger.warning(f"Failed to load {self.config.dataset_name}: {e}")
            logger.info("Falling back to karimm6/FactoryNet_Dataset")
            self.hf_dataset = load_dataset(
                "karimm6/FactoryNet_Dataset",
                "normalized",
                split="train",
            )

        # Convert to pandas for easier manipulation
        self.df = self.hf_dataset.to_pandas()
        logger.info(f"Loaded {len(self.df)} rows, columns: {list(self.df.columns)[:10]}...")

        # Load metadata JSON for fault labels
        self._load_metadata(data_dir)

    def _load_metadata(self, data_dir: Optional[str]):
        """Load metadata JSON with fault labels for each episode."""
        self.episode_metadata = {}

        if not data_dir:
            logger.warning("No subset specified, cannot load metadata")
            return

        metadata_file = METADATA_FILES.get(data_dir)
        if not metadata_file:
            logger.warning(f"No metadata file mapping for subset: {data_dir}")
            return

        try:
            path = hf_hub_download(
                repo_id=self.config.dataset_name,
                filename=metadata_file,
                repo_type="dataset",
            )
            with open(path) as f:
                metadata_list = json.load(f)

            # Build episode_id → metadata mapping
            # Use fault_label (matches original papers) over fault_type (simplified)
            for ep in metadata_list:
                ep_id = ep.get("episode_id")
                if ep_id:
                    # Determine fault status: use fault_label for granular labels
                    # "normal" with success=True → healthy
                    # Everything else → fault (including "loosening" with success=False)
                    fault_label = ep.get("fault_label", "unknown")
                    success = ep.get("success", True)

                    original_label = ep.get("static_context", {}).get("original_label")

                    # Determine operation phase (AURSAD has paired loosening→tightening)
                    is_loosening = (fault_label == "loosening" or original_label == 5)
                    phase = "loosening" if is_loosening else "tightening"

                    # Handle different label formats across datasets:
                    # - AURSAD: fault_label = "normal", "damaged_screw", etc.
                    # - voraus-AD: fault_label = "true" (anomaly) or "false" (normal)
                    if fault_label in ("true", "True"):
                        normalized_fault = "anomaly"
                        is_healthy = False
                    elif fault_label in ("false", "False"):
                        normalized_fault = "normal"
                        is_healthy = True
                    else:
                        # AURSAD-style labels
                        is_healthy = is_loosening or (fault_label == "normal")
                        normalized_fault = "normal" if is_healthy else fault_label

                    self.episode_metadata[ep_id] = {
                        "fault_type": normalized_fault,
                        "fault_label": fault_label,  # Original label
                        "phase": phase,  # "loosening" or "tightening" (AURSAD-specific)
                        "machine_type": ep.get("machine_type"),
                        "machine_model": ep.get("machine_model"),
                        "task_type": ep.get("task_type"),
                        "success": success,
                        "original_label": original_label,
                    }

            logger.info(f"Loaded metadata for {len(self.episode_metadata)} episodes")

            # Log fault distribution
            fault_counts = {}
            for ep_meta in self.episode_metadata.values():
                ft = ep_meta["fault_type"]
                fault_counts[ft] = fault_counts.get(ft, 0) + 1
            logger.info(f"Fault distribution: {fault_counts}")

        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self.episode_metadata = {}

    def _setup_columns(self):
        """Identify available columns and create unified schema mapping.

        This handles heterogeneous data by:
        1. Finding which columns exist in this dataset
        2. Mapping them to unified output positions
        3. Creating validity masks for padded dimensions
        """
        available_cols = set(self.df.columns)

        # Build setpoint column list (try each signal type in order of preference)
        self.setpoint_cols = []
        for signal_type in self.config.setpoint_signals:
            if signal_type in SETPOINT_PATTERNS:
                for col in SETPOINT_PATTERNS[signal_type]:
                    if col in available_cols:
                        self.setpoint_cols.append(col)

        # Build effort column list - try multiple types (torque OR current)
        # Preference order defined by config.effort_signals
        self.effort_cols = []
        self.effort_signal_type = "unknown"
        for signal_type in self.config.effort_signals:
            if signal_type in EFFORT_PATTERNS:
                cols_found = []
                for col in EFFORT_PATTERNS[signal_type]:
                    if col in available_cols:
                        cols_found.append(col)
                # Use this signal type if we found any columns
                if cols_found:
                    self.effort_cols = cols_found
                    self.effort_signal_type = signal_type
                    break  # Stop at first signal type with data

        # Validate we have the minimum required columns
        if not self.setpoint_cols:
            raise ValueError(
                f"No setpoint columns found. Available: {available_cols}"
            )
        if not self.effort_cols:
            raise ValueError(
                f"No effort columns found. Available: {available_cols}"
            )

        # Store actual dimensions (before padding)
        self.actual_setpoint_dim = len(self.setpoint_cols)
        self.actual_effort_dim = len(self.effort_cols)

        # Create validity masks (1 = real data, 0 = padded)
        self.setpoint_mask = np.zeros(self.config.unified_setpoint_dim, dtype=np.float32)
        self.setpoint_mask[:self.actual_setpoint_dim] = 1.0

        self.effort_mask = np.zeros(self.config.unified_effort_dim, dtype=np.float32)
        self.effort_mask[:self.actual_effort_dim] = 1.0

        logger.info(
            f"Setpoint: {self.actual_setpoint_dim} cols → unified {self.config.unified_setpoint_dim} "
            f"({self.setpoint_cols})"
        )
        logger.info(
            f"Effort ({self.effort_signal_type}): {self.actual_effort_dim} cols → unified {self.config.unified_effort_dim} "
            f"({self.effort_cols})"
        )

    def _build_episode_index(self):
        """Build index of episodes and split into train/val/test."""
        # Get unique episode IDs
        if "episode_id" in self.df.columns:
            all_episode_ids = list(self.df["episode_id"].unique())
        else:
            # If no episode_id, treat entire dataset as one episode
            self.df["episode_id"] = "episode_0"
            all_episode_ids = ["episode_0"]

        # Handle AURSAD phase filtering based on config
        # AURSAD has paired operations: loosening (prepare) → tightening (screw in)
        if self.config.aursad_phase_handling == "tightening_only" and self.episode_metadata:
            loosening_count = sum(
                1 for ep_id in all_episode_ids
                if self.episode_metadata.get(ep_id, {}).get("phase") == "loosening"
            )
            self.episode_ids = [
                ep_id for ep_id in all_episode_ids
                if self.episode_metadata.get(ep_id, {}).get("phase") != "loosening"
            ]
            if loosening_count > 0:
                logger.info(
                    f"Filtered out {loosening_count} loosening phase episodes. "
                    f"Remaining: {len(self.episode_ids)} tightening episodes"
                )
        elif self.config.aursad_phase_handling == "merge":
            # TODO: Merge paired loosening+tightening into single episodes
            logger.warning("aursad_phase_handling='merge' not yet implemented, using 'both'")
            self.episode_ids = all_episode_ids
        else:
            # "both" - use all episodes, loosening is treated as normal operation
            self.episode_ids = all_episode_ids
            if self.episode_metadata:
                loosening_count = sum(
                    1 for ep_id in all_episode_ids
                    if self.episode_metadata.get(ep_id, {}).get("phase") == "loosening"
                )
                tightening_count = len(all_episode_ids) - loosening_count
                logger.info(
                    f"Using both phases: {loosening_count} loosening + {tightening_count} tightening"
                )

        # Get fault labels from metadata (preferred) or fallback to column
        self.episode_labels = {}
        for ep_id in self.episode_ids:
            if ep_id in self.episode_metadata:
                # Use metadata fault_type
                self.episode_labels[ep_id] = self.episode_metadata[ep_id]["fault_type"]
            elif "ctx_anomaly_label" in self.df.columns:
                # Fallback to column-based labels
                ep_data = self.df[self.df["episode_id"] == ep_id]
                labels = ep_data["ctx_anomaly_label"].dropna()
                if len(labels) > 0:
                    self.episode_labels[ep_id] = labels.mode().iloc[0] if len(labels.mode()) > 0 else "normal"
                else:
                    self.episode_labels[ep_id] = "normal"
            else:
                self.episode_labels[ep_id] = "normal"

        # Identify healthy vs fault episodes
        healthy_episodes = [
            ep for ep, label in self.episode_labels.items()
            if str(label).lower() in ["normal", "none", "null", "healthy", ""]
        ]
        fault_episodes = [
            ep for ep in self.episode_ids if ep not in healthy_episodes
        ]

        logger.info(f"Episodes: {len(healthy_episodes)} healthy, {len(fault_episodes)} fault")

        # Split episodes (not rows) into train/val/test
        np.random.seed(42)  # Reproducibility

        if self.config.train_healthy_only:
            # Train only on healthy, test on both
            n_healthy = len(healthy_episodes)
            n_train = int(n_healthy * self.config.train_ratio)
            n_val = int(n_healthy * self.config.val_ratio)

            shuffled_healthy = np.random.permutation(healthy_episodes)
            train_eps = list(shuffled_healthy[:n_train])
            val_eps = list(shuffled_healthy[n_train:n_train + n_val])
            test_eps = list(shuffled_healthy[n_train + n_val:]) + fault_episodes
        else:
            # Standard split including faults in all splits
            all_episodes = list(self.episode_ids)
            np.random.shuffle(all_episodes)
            n_total = len(all_episodes)
            n_train = int(n_total * self.config.train_ratio)
            n_val = int(n_total * self.config.val_ratio)

            train_eps = all_episodes[:n_train]
            val_eps = all_episodes[n_train:n_train + n_val]
            test_eps = all_episodes[n_train + n_val:]

        # Select episodes for this split
        if self.split == "train":
            self.split_episodes = train_eps
        elif self.split == "val":
            self.split_episodes = val_eps
        else:
            self.split_episodes = test_eps

        logger.info(f"Split '{self.split}': {len(self.split_episodes)} episodes")

    def _create_windows(self):
        """Create sliding windows from episodes."""
        self.windows = []  # List of (episode_id, start_idx, end_idx)

        for ep_id in self.split_episodes:
            ep_data = self.df[self.df["episode_id"] == ep_id]
            ep_len = len(ep_data)

            # Skip episodes shorter than window size
            if ep_len < self.config.window_size:
                continue

            # Create windows with stride
            start = 0
            while start + self.config.window_size <= ep_len:
                self.windows.append({
                    "episode_id": ep_id,
                    "start_idx": ep_data.index[start],
                    "end_idx": ep_data.index[start + self.config.window_size - 1],
                    "label": self.episode_labels.get(ep_id),
                })
                start += self.config.stride

        logger.info(f"Created {len(self.windows)} windows")

    def _compute_global_stats(self):
        """Compute global mean/std for normalization."""
        setpoint_data = self.df[self.setpoint_cols].values
        effort_data = self.df[self.effort_cols].values

        self.setpoint_mean = np.nanmean(setpoint_data, axis=0)
        self.setpoint_std = np.nanstd(setpoint_data, axis=0) + 1e-8
        self.effort_mean = np.nanmean(effort_data, axis=0)
        self.effort_std = np.nanstd(effort_data, axis=0) + 1e-8

    def _normalize_window(
        self,
        setpoint: np.ndarray,
        effort: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize a window of data."""
        if not self.config.normalize:
            return setpoint, effort

        if self.config.norm_mode == "episode":
            # Per-window normalization (z-score)
            setpoint = (setpoint - np.nanmean(setpoint, axis=0)) / (np.nanstd(setpoint, axis=0) + 1e-8)
            effort = (effort - np.nanmean(effort, axis=0)) / (np.nanstd(effort, axis=0) + 1e-8)
        elif self.config.norm_mode == "global":
            setpoint = (setpoint - self.setpoint_mean) / self.setpoint_std
            effort = (effort - self.effort_mean) / self.effort_std

        return setpoint, effort

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Get a (setpoint, effort) window pair with unified dimensions.

        Returns:
            setpoint: Tensor of shape (window_size, unified_setpoint_dim)
            effort: Tensor of shape (window_size, unified_effort_dim)
            metadata: Dict with episode_id, label, masks, etc.

        The metadata includes validity masks indicating which dimensions
        contain real data vs zero-padding. This enables training across
        heterogeneous datasets with different sensor configurations.
        """
        window = self.windows[idx]

        # Get window data
        start_idx = window["start_idx"]
        end_idx = window["end_idx"]

        # Slice dataframe by index range
        window_data = self.df.loc[start_idx:end_idx]

        # Extract setpoint and effort (actual dimensions)
        setpoint_raw = window_data[self.setpoint_cols].values.astype(np.float32)
        effort_raw = window_data[self.effort_cols].values.astype(np.float32)

        # Handle NaN values
        setpoint_raw = np.nan_to_num(setpoint_raw, nan=0.0)
        effort_raw = np.nan_to_num(effort_raw, nan=0.0)

        # Normalize (before padding to avoid normalizing zeros)
        setpoint_raw, effort_raw = self._normalize_window(setpoint_raw, effort_raw)

        # Pad to unified dimensions
        # Shape: (window_size, unified_dim)
        setpoint = np.zeros(
            (self.config.window_size, self.config.unified_setpoint_dim),
            dtype=np.float32
        )
        effort = np.zeros(
            (self.config.window_size, self.config.unified_effort_dim),
            dtype=np.float32
        )

        # Copy actual data to start of unified tensor
        setpoint[:, :self.actual_setpoint_dim] = setpoint_raw
        effort[:, :self.actual_effort_dim] = effort_raw

        # Convert to tensors
        setpoint_tensor = torch.from_numpy(setpoint)
        effort_tensor = torch.from_numpy(effort)

        # Get episode metadata
        ep_id = window["episode_id"]
        ep_meta = self.episode_metadata.get(ep_id, {})
        fault_type = window["label"] if window["label"] else "normal"
        phase = ep_meta.get("phase", "unknown")

        # Anomaly = tightening phase with non-normal fault
        # Loosening phase is always "normal" (different operation, not fault)
        is_anomaly = (phase == "tightening" and
                      str(fault_type).lower() not in ["normal", "none", "null", "healthy", ""])

        metadata = {
            "episode_id": ep_id,
            "fault_type": fault_type,
            "phase": phase,  # "loosening" or "tightening"
            "is_anomaly": is_anomaly,
            # Validity masks for handling heterogeneous data
            "setpoint_mask": torch.from_numpy(self.setpoint_mask),
            "effort_mask": torch.from_numpy(self.effort_mask),
            "actual_setpoint_dim": self.actual_setpoint_dim,
            "actual_effort_dim": self.actual_effort_dim,
            "effort_signal_type": self.effort_signal_type,
        }

        return setpoint_tensor, effort_tensor, metadata


def create_dataloaders(
    config: FactoryNetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.

    Args:
        config: FactoryNet configuration
        batch_size: Batch size for all dataloaders
        num_workers: Number of worker processes

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dataset = FactoryNetDataset(config, split="train")
    val_dataset = FactoryNetDataset(config, split="val")
    test_dataset = FactoryNetDataset(config, split="test")

    def collate_fn(batch):
        setpoints, efforts, metadatas = zip(*batch)
        return (
            torch.stack(setpoints),
            torch.stack(efforts),
            metadatas,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# Convenience function for quick testing
def load_aursad(
    window_size: int = 256,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Load AURSAD dataset with default settings.

    Example:
        >>> train, val, test = load_aursad()
        >>> for setpoint, effort, meta in train:
        ...     print(setpoint.shape, effort.shape)
        ...     break
    """
    config = FactoryNetConfig(
        subset="AURSAD",
        window_size=window_size,
    )
    return create_dataloaders(config, batch_size=batch_size)

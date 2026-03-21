#!/usr/bin/env python
"""
Experiment 02: Many-to-1 Cross-Machine Transfer Learning

Train on N-1 data sources, evaluate zero-shot on held-out source.
Leave-one-out cross-validation across all available FactoryNet sources.

This tests the hypothesis that training on diverse sources leads to
better transferable representations than single-source training.

Objectives:
1. Forecasting: Transfer ratio ≤ 1.5 for held-out source
2. Anomaly detection: AUC ≥ 0.70 on held-out source
"""

import sys
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "autoresearch" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# Data Loading
# ============================================================

SOURCES = {
    "aursad": {"max_episodes": 500, "label": "AURSAD (UR3e)"},
    "voraus": {"max_episodes": 500, "label": "Voraus (Yu-Cobot)"},
    "cnc": {"max_episodes": None, "label": "CNC (UMich)"},
}


def load_source(source_name, split="train", window_size=256, stride=128,
                norm_mode="episode", max_episodes=None):
    """Load a single data source."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    src_info = SOURCES[source_name]
    max_ep = max_episodes if max_episodes is not None else src_info["max_episodes"]

    config = FactoryNetConfig(
        data_source=source_name,
        max_episodes=max_ep,
        window_size=window_size,
        stride=stride,
        normalize=True,
        norm_mode=norm_mode,
        effort_signals=["voltage", "current"],  # Try voltage first, then current
        setpoint_signals=["position", "velocity"],
        unified_setpoint_dim=12,
        unified_effort_dim=6,
    )

    ds = FactoryNetDataset(config, split=split)
    return ds


def load_all_sources(splits=("train", "val", "test"), **kwargs):
    """Load all sources for all splits."""
    all_datasets = {}

    for source_name in SOURCES:
        all_datasets[source_name] = {}
        shared = None
        for split in splits:
            try:
                if shared is None:
                    ds = load_source(source_name, split=split, **kwargs)
                    shared = ds.get_shared_data()
                else:
                    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig
                    src_info = SOURCES[source_name]
                    config = FactoryNetConfig(
                        data_source=source_name,
                        max_episodes=kwargs.get("max_episodes") or src_info["max_episodes"],
                        window_size=kwargs.get("window_size", 256),
                        stride=kwargs.get("stride", 128),
                        normalize=True,
                        norm_mode=kwargs.get("norm_mode", "episode"),
                        effort_signals=["voltage", "current"],
                        setpoint_signals=["position", "velocity"],
                        unified_setpoint_dim=12,
                        unified_effort_dim=6,
                    )
                    ds = FactoryNetDataset(config, split=split, shared_data=shared)

                all_datasets[source_name][split] = ds
                logger.info(f"  {source_name}/{split}: {len(ds)} windows")
            except Exception as e:
                logger.warning(f"  Failed to load {source_name}/{split}: {e}")
                all_datasets[source_name][split] = None

    return all_datasets


def collate_fn(batch):
    """Custom collate function."""
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])
    metadata = {
        "is_anomaly": [item[2].get("is_anomaly", False) for item in batch],
        "fault_type": [item[2].get("fault_type", "normal") for item in batch],
    }
    return setpoints, efforts, metadata


def make_loader(dataset, batch_size=64, shuffle=False):
    if dataset is None or len(dataset) == 0:
        return None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=False,
    )


def make_combined_loader(datasets_list, batch_size=64, shuffle=True):
    """Create a combined loader from multiple datasets."""
    valid = [d for d in datasets_list if d is not None and len(d) > 0]
    if not valid:
        return None
    combined = ConcatDataset(valid)
    return DataLoader(
        combined, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=True, collate_fn=collate_fn, drop_last=True,
    )


# ============================================================
# Models
# ============================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization for domain-invariant features.

    From: "Reversible Instance Normalization for Accurate Time-Series Forecasting
    against Distribution Shift" (Kim et al., ICLR 2022)

    Key idea: Normalize input per-instance, process, then denormalize output.
    This removes domain-specific statistics while preserving learnable patterns.
    """

    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        """
        Args:
            x: (B, T, D)
            mode: 'norm' to normalize, 'denorm' to denormalize
        """
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            x = x * self._std + self._mean
            return x


class MultiSourceForecaster(nn.Module):
    """
    Forecaster with RevIN for cross-domain generalization.

    Architecture:
    1. RevIN normalize input (removes domain-specific statistics)
    2. Channel-independent patch embedding (handles different DOF)
    3. Transformer encoder for temporal modeling
    4. Linear decoder for forecasting
    5. RevIN denormalize output
    """

    def __init__(self, input_dim=18, hidden_dim=128, num_layers=3,
                 num_heads=4, context_len=128, forecast_len=128,
                 use_revin=True):
        super().__init__()
        self.context_len = context_len
        self.forecast_len = forecast_len
        self.input_dim = input_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(input_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(
            torch.randn(1, context_len, hidden_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Forecast head: map each context position to forecast
        self.forecast_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x_full):
        """
        Args:
            x_full: (B, T, D) full sequence
        Returns:
            dict with loss, pred, target
        """
        B, T, D = x_full.shape

        # RevIN normalize
        if self.use_revin:
            x_full = self.revin(x_full, mode='norm')

        context = x_full[:, :self.context_len, :]
        target = x_full[:, self.context_len:self.context_len + self.forecast_len, :]

        # Encode
        h = self.input_proj(context) + self.pos_enc[:, :context.shape[1], :]
        h = self.transformer(h)
        h = self.norm(h)

        # Forecast
        pred = self.forecast_head(h)

        # Match forecast length
        if pred.shape[1] > self.forecast_len:
            pred = pred[:, -self.forecast_len:, :]
        elif pred.shape[1] < self.forecast_len:
            # Repeat last to match
            pad = pred[:, -1:, :].expand(-1, self.forecast_len - pred.shape[1], -1)
            pred = torch.cat([pred, pad], dim=1)

        # RevIN denormalize predictions for loss computation
        if self.use_revin:
            pred_denorm = self.revin(pred, mode='denorm')
            target_denorm = self.revin(target, mode='denorm')
            loss = F.mse_loss(pred_denorm, target_denorm)
        else:
            loss = F.mse_loss(pred, target)

        return {"loss": loss, "pred": pred, "target": target}


class MultiSourceAnomalyDetector(nn.Module):
    """
    Anomaly detector based on temporal prediction with RevIN.

    Approach: Learn normal temporal dynamics from multiple sources.
    Anomaly score = prediction error (normalized).
    """

    def __init__(self, setpoint_dim=12, effort_dim=6, hidden_dim=128,
                 num_layers=3, num_heads=4, window_size=256, use_revin=True):
        super().__init__()
        self.setpoint_dim = setpoint_dim
        self.effort_dim = effort_dim
        self.context_len = window_size // 2
        self.target_len = window_size // 2
        input_dim = setpoint_dim + effort_dim
        self.use_revin = use_revin

        if use_revin:
            self.revin = RevIN(input_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(
            torch.randn(1, window_size, hidden_dim) * 0.02
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, setpoint, effort):
        """Forward pass for training. Predict second half from first half."""
        x = torch.cat([setpoint, effort], dim=-1)

        if self.use_revin:
            x = self.revin(x, mode='norm')

        context = x[:, :self.context_len, :]
        target = x[:, self.context_len:, :]

        h = self.input_proj(context) + self.pos_enc[:, :self.context_len, :]
        h = self.transformer(h)
        h = self.norm(h)
        pred = self.pred_head(h)

        # Match target length
        if pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1], :]

        loss = F.mse_loss(pred, target)
        return {"loss": loss, "pred": pred, "target": target}

    @torch.no_grad()
    def anomaly_score(self, setpoint, effort):
        """Compute per-sample anomaly score."""
        x = torch.cat([setpoint, effort], dim=-1)

        if self.use_revin:
            x = self.revin(x, mode='norm')

        context = x[:, :self.context_len, :]
        target = x[:, self.context_len:, :]

        h = self.input_proj(context) + self.pos_enc[:, :self.context_len, :]
        h = self.transformer(h)
        h = self.norm(h)
        pred = self.pred_head(h)

        if pred.shape[1] > target.shape[1]:
            pred = pred[:, :target.shape[1], :]

        # Per-sample MSE
        error = ((pred - target) ** 2).mean(dim=(1, 2))
        return error


# ============================================================
# Training & Evaluation
# ============================================================

def train_model(model, train_loader, val_loader, epochs, lr=1e-4,
                device='cuda', model_type='forecaster'):
    """Train model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for setpoint, effort, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            setpoint = setpoint.to(device)
            effort = effort.to(device)

            optimizer.zero_grad()

            if model_type == 'forecaster':
                x = torch.cat([setpoint, effort], dim=-1)
                result = model(x)
            else:
                result = model(setpoint, effort)

            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for setpoint, effort, _ in val_loader:
                setpoint = setpoint.to(device)
                effort = effort.to(device)

                if model_type == 'forecaster':
                    x = torch.cat([setpoint, effort], dim=-1)
                    result = model(x)
                else:
                    result = model(setpoint, effort)

                val_loss += result['loss'].item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_loss


@torch.no_grad()
def evaluate_forecasting(model, test_loader, device='cuda'):
    """Evaluate forecasting MSE."""
    model.eval()
    total_mse = 0
    n_batches = 0

    for setpoint, effort, _ in test_loader:
        setpoint = setpoint.to(device)
        effort = effort.to(device)
        x = torch.cat([setpoint, effort], dim=-1)
        result = model(x)
        total_mse += result['loss'].item()
        n_batches += 1

    return total_mse / max(n_batches, 1)


@torch.no_grad()
def evaluate_anomaly(model, test_loader, device='cuda'):
    """Evaluate anomaly detection."""
    model.eval()
    all_scores = []
    all_labels = []

    for setpoint, effort, metadata in test_loader:
        setpoint = setpoint.to(device)
        effort = effort.to(device)
        scores = model.anomaly_score(setpoint, effort)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend([1 if a else 0 for a in metadata["is_anomaly"]])

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    results = {
        'n_total': len(labels),
        'n_normal': int((labels == 0).sum()),
        'n_anomaly': int((labels == 1).sum()),
    }

    if len(np.unique(labels)) > 1:
        results['roc_auc'] = float(roc_auc_score(labels, scores))
        results['pr_auc'] = float(average_precision_score(labels, scores))
    else:
        results['roc_auc'] = None
        results['pr_auc'] = None

    return results, scores, labels


# ============================================================
# Visualization
# ============================================================

def plot_transfer_results(results_dict, filename):
    """Plot comprehensive transfer results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 1. Transfer ratios
    ax = axes[0]
    sources = list(results_dict.keys())
    for src_config, src_results in results_dict.items():
        if 'forecasting' in src_results:
            held_out = src_results.get('held_out', src_config)
            ratios = [r.get('transfer_ratio', None) for r in src_results['forecasting'].values()
                      if isinstance(r, dict) and 'transfer_ratio' in r]
            if ratios:
                ax.bar(held_out, ratios[0], alpha=0.7)
    ax.axhline(y=1.5, color='r', linestyle='--', label='Target ≤ 1.5')
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Perfect transfer')
    ax.set_ylabel('Transfer Ratio')
    ax.set_title('Forecasting Transfer Ratio\n(lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Anomaly AUC
    ax = axes[1]
    for src_config, src_results in results_dict.items():
        if 'anomaly' in src_results:
            held_out = src_results.get('held_out', src_config)
            auc = src_results['anomaly'].get('target', {}).get('roc_auc')
            if auc is not None:
                ax.bar(held_out, auc, alpha=0.7)
    ax.axhline(y=0.70, color='r', linestyle='--', label='Target ≥ 0.70')
    ax.axhline(y=0.50, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Anomaly Detection AUC\n(higher is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # 3. Training loss curves
    ax = axes[2]
    for src_config, src_results in results_dict.items():
        held_out = src_results.get('held_out', src_config)
        if 'forecast_history' in src_results:
            ax.plot(src_results['forecast_history']['train_loss'],
                    label=f'Hold-out: {held_out}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Forecaster Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Many-to-1 Cross-Machine Transfer Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def plot_leave_one_out_comparison(one_to_one, many_to_one, filename):
    """Plot 1-to-1 vs many-to-1 comparison."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    held_out_sources = list(many_to_one.keys())
    x = np.arange(len(held_out_sources))
    width = 0.35

    # Forecasting
    ax = axes[0]
    ratios_1to1 = [one_to_one.get(s, {}).get('transfer_ratio', 0) for s in held_out_sources]
    ratios_mto1 = [many_to_one.get(s, {}).get('transfer_ratio', 0) for s in held_out_sources]
    ax.bar(x - width/2, ratios_1to1, width, label='1-to-1', alpha=0.7)
    ax.bar(x + width/2, ratios_mto1, width, label='Many-to-1', alpha=0.7)
    ax.axhline(y=1.5, color='r', linestyle='--', label='Target ≤ 1.5')
    ax.set_xticks(x)
    ax.set_xticklabels(held_out_sources)
    ax.set_ylabel('Transfer Ratio')
    ax.set_title('Forecasting Transfer')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Anomaly
    ax = axes[1]
    auc_1to1 = [one_to_one.get(s, {}).get('auc', 0.5) for s in held_out_sources]
    auc_mto1 = [many_to_one.get(s, {}).get('auc', 0.5) for s in held_out_sources]
    ax.bar(x - width/2, auc_1to1, width, label='1-to-1', alpha=0.7)
    ax.bar(x + width/2, auc_mto1, width, label='Many-to-1', alpha=0.7)
    ax.axhline(y=0.70, color='r', linestyle='--', label='Target ≥ 0.70')
    ax.set_xticks(x)
    ax.set_xticklabels(held_out_sources)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Anomaly Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.suptitle('1-to-1 vs Many-to-1 Transfer Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


# ============================================================
# Main Experiment
# ============================================================

def run_leave_one_out(seed=42, epochs=15, hidden_dim=128, num_layers=3,
                      batch_size=64, use_revin=True, norm_mode="episode"):
    """Run leave-one-out cross-validation across all sources."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}, Seed: {seed}, RevIN: {use_revin}")

    # Load all datasets
    logger.info("Loading all data sources...")
    all_data = load_all_sources(
        window_size=256, stride=128, norm_mode=norm_mode,
    )

    # Print summary
    for source_name, splits in all_data.items():
        for split_name, ds in splits.items():
            if ds is not None:
                logger.info(f"  {source_name}/{split_name}: {len(ds)} windows")

    results = {}
    one_to_one_results = {}
    many_to_one_results = {}

    # Leave-one-out: for each source, hold it out and train on the rest
    available_sources = [s for s in SOURCES if all_data.get(s, {}).get('train') is not None
                         and len(all_data[s]['train']) > 0]
    logger.info(f"Available sources for LOO: {available_sources}")

    for held_out in available_sources:
        logger.info(f"\n{'='*60}")
        logger.info(f"HOLD-OUT: {SOURCES[held_out]['label']}")
        logger.info(f"{'='*60}")

        train_sources = [s for s in available_sources if s != held_out]
        logger.info(f"Training on: {[SOURCES[s]['label'] for s in train_sources]}")

        # Create combined training data
        train_datasets = [all_data[s]['train'] for s in train_sources
                          if all_data[s].get('train') is not None]
        val_datasets = [all_data[s].get('val') or all_data[s]['train'] for s in train_sources
                        if all_data[s].get('train') is not None]

        combined_train = make_combined_loader(train_datasets, batch_size=batch_size)
        combined_val = make_combined_loader(val_datasets, batch_size=batch_size, shuffle=False)

        if combined_train is None:
            logger.warning(f"No training data for {held_out} hold-out. Skipping.")
            continue

        # Test loaders
        target_test = make_loader(all_data[held_out].get('test'), batch_size=batch_size)
        if target_test is None:
            logger.warning(f"No test data for held-out {held_out}. Skipping.")
            continue

        # Also get source test loaders for transfer ratio
        source_test_loaders = {}
        for s in train_sources:
            loader = make_loader(all_data[s].get('test'), batch_size=batch_size)
            if loader is not None:
                source_test_loaders[s] = loader

        exp_results = {'held_out': held_out, 'train_sources': train_sources}

        # ---- MANY-TO-1 FORECASTING ----
        logger.info(f"\n--- Many-to-1 Forecasting (hold-out: {held_out}) ---")

        forecaster = MultiSourceForecaster(
            input_dim=18, hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=4, context_len=128, forecast_len=128,
            use_revin=use_revin,
        ).to(device)

        fc_history, _ = train_model(
            forecaster, combined_train, combined_val,
            epochs=epochs, lr=1e-4, device=device, model_type='forecaster',
        )
        exp_results['forecast_history'] = fc_history

        # Evaluate on source(s)
        source_mses = {}
        for s, loader in source_test_loaders.items():
            mse = evaluate_forecasting(forecaster, loader, device)
            source_mses[s] = mse
            logger.info(f"  Source {s} MSE: {mse:.6f}")

        avg_source_mse = np.mean(list(source_mses.values())) if source_mses else 0

        # Evaluate on target (held-out)
        target_mse = evaluate_forecasting(forecaster, target_test, device)
        transfer_ratio = target_mse / max(avg_source_mse, 1e-8)

        logger.info(f"  Target {held_out} MSE: {target_mse:.6f}")
        logger.info(f"  Avg Source MSE: {avg_source_mse:.6f}")
        logger.info(f"  Transfer Ratio: {transfer_ratio:.4f}")

        exp_results['forecasting'] = {
            'source_mses': source_mses,
            'target_mse': target_mse,
            'avg_source_mse': avg_source_mse,
            'transfer_ratio': transfer_ratio,
        }

        many_to_one_results[held_out] = {
            'transfer_ratio': transfer_ratio,
            'target_mse': target_mse,
        }

        # ---- MANY-TO-1 ANOMALY DETECTION ----
        logger.info(f"\n--- Many-to-1 Anomaly Detection (hold-out: {held_out}) ---")

        detector = MultiSourceAnomalyDetector(
            setpoint_dim=12, effort_dim=6,
            hidden_dim=hidden_dim, num_layers=num_layers,
            num_heads=4, window_size=256, use_revin=use_revin,
        ).to(device)

        ad_history, _ = train_model(
            detector, combined_train, combined_val,
            epochs=epochs, lr=1e-4, device=device, model_type='anomaly',
        )

        # Evaluate on target
        target_ad, target_scores, target_labels = evaluate_anomaly(
            detector, target_test, device
        )
        logger.info(f"  Target AUC: {target_ad.get('roc_auc', 'N/A')}")
        logger.info(f"  Normal: {target_ad['n_normal']}, Anomaly: {target_ad['n_anomaly']}")

        exp_results['anomaly'] = {'target': target_ad}

        many_to_one_results[held_out]['auc'] = target_ad.get('roc_auc', 0.5)

        # ---- SINGLE-SOURCE BASELINE (1-to-1) ----
        # For comparison, also train single-source models
        for single_source in train_sources[:1]:  # Just use first source for speed
            logger.info(f"\n--- 1-to-1 Baseline: {single_source} → {held_out} ---")

            single_train = make_loader(all_data[single_source]['train'], batch_size, shuffle=True)
            single_val = make_loader(all_data[single_source].get('val') or all_data[single_source]['train'],
                                     batch_size)

            if single_train is None:
                continue

            # Forecasting
            fc_single = MultiSourceForecaster(
                input_dim=18, hidden_dim=hidden_dim, num_layers=num_layers,
                num_heads=4, context_len=128, forecast_len=128,
                use_revin=use_revin,
            ).to(device)

            train_model(fc_single, single_train, single_val,
                        epochs=epochs, lr=1e-4, device=device, model_type='forecaster')

            source_test_single = make_loader(all_data[single_source].get('test'), batch_size)
            if source_test_single:
                src_mse = evaluate_forecasting(fc_single, source_test_single, device)
                tgt_mse = evaluate_forecasting(fc_single, target_test, device)
                ratio_single = tgt_mse / max(src_mse, 1e-8)
                logger.info(f"  1-to-1 Transfer Ratio: {ratio_single:.4f}")

                one_to_one_results[held_out] = {
                    'transfer_ratio': ratio_single,
                    'source': single_source,
                }

            # Anomaly detection
            ad_single = MultiSourceAnomalyDetector(
                setpoint_dim=12, effort_dim=6,
                hidden_dim=hidden_dim, num_layers=num_layers,
                num_heads=4, window_size=256, use_revin=use_revin,
            ).to(device)

            train_model(ad_single, single_train, single_val,
                        epochs=epochs, lr=1e-4, device=device, model_type='anomaly')

            ad_single_results, _, _ = evaluate_anomaly(ad_single, target_test, device)
            logger.info(f"  1-to-1 AUC: {ad_single_results.get('roc_auc', 'N/A')}")

            one_to_one_results[held_out]['auc'] = ad_single_results.get('roc_auc', 0.5)

        results[held_out] = exp_results

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("LEAVE-ONE-OUT CROSS-VALIDATION RESULTS")
    print("=" * 80)
    print(f"{'Held-Out':<15} {'Many-to-1 Ratio':<18} {'1-to-1 Ratio':<15} {'Many-to-1 AUC':<15} {'1-to-1 AUC':<12}")
    print("-" * 80)

    for held_out in available_sources:
        m2o = many_to_one_results.get(held_out, {})
        o2o = one_to_one_results.get(held_out, {})
        print(f"{held_out:<15} "
              f"{m2o.get('transfer_ratio', 'N/A'):<18.4f} "
              f"{o2o.get('transfer_ratio', 'N/A') if isinstance(o2o.get('transfer_ratio'), float) else 'N/A':<15} "
              f"{m2o.get('auc', 'N/A'):<15} "
              f"{o2o.get('auc', 'N/A'):<12}")

    print("=" * 80)

    # Plot results
    if results:
        plot_transfer_results(results, FIGURES_DIR / f"exp02_transfer_results_seed{seed}.png")

    if one_to_one_results and many_to_one_results:
        plot_leave_one_out_comparison(
            one_to_one_results, many_to_one_results,
            FIGURES_DIR / f"exp02_1to1_vs_mto1_seed{seed}.png"
        )

    return {
        'results': results,
        'one_to_one': one_to_one_results,
        'many_to_one': many_to_one_results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-revin", action="store_true")
    parser.add_argument("--norm-mode", type=str, default="episode")
    args = parser.parse_args()

    all_results = run_leave_one_out(
        seed=args.seed, epochs=args.epochs,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        batch_size=args.batch_size, use_revin=not args.no_revin,
        norm_mode=args.norm_mode,
    )

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"exp02_seed{args.seed}_{timestamp}.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

#!/usr/bin/env python
"""
Experiment 01: Baseline Cross-Machine Transfer

Quick baseline to establish:
1. Source domain AUC (AURSAD)
2. Zero-shot transfer AUC (Voraus)
3. Source/target prediction MSE for transfer ratio

Uses efficient data loading (shared data between splits).
"""

import sys
import os
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "autoresearch" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_datasets(max_episodes_source=500, max_episodes_target=500,
                  window_size=256, stride=128, norm_mode="episode",
                  effort_signals=None):
    """Load both datasets efficiently."""
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    if effort_signals is None:
        effort_signals = ["voltage"]

    # AURSAD
    logger.info("Loading AURSAD...")
    aursad_config = FactoryNetConfig(
        data_source='aursad',
        max_episodes=max_episodes_source,
        window_size=window_size,
        stride=stride,
        normalize=True,
        norm_mode=norm_mode,
        effort_signals=effort_signals,
        setpoint_signals=["position", "velocity"],
        unified_setpoint_dim=12,
        unified_effort_dim=6,
    )
    aursad_train = FactoryNetDataset(aursad_config, split='train')
    shared = aursad_train.get_shared_data()
    aursad_val = FactoryNetDataset(aursad_config, split='val', shared_data=shared)
    aursad_test = FactoryNetDataset(aursad_config, split='test', shared_data=shared)

    # Voraus
    logger.info("Loading Voraus...")
    voraus_config = FactoryNetConfig(
        data_source='voraus',
        max_episodes=max_episodes_target,
        window_size=window_size,
        stride=stride,
        normalize=True,
        norm_mode=norm_mode,
        effort_signals=effort_signals,
        setpoint_signals=["position", "velocity"],
        unified_setpoint_dim=12,
        unified_effort_dim=6,
    )
    voraus_train = FactoryNetDataset(voraus_config, split='train')
    shared_v = voraus_train.get_shared_data()
    voraus_test = FactoryNetDataset(voraus_config, split='test', shared_data=shared_v)

    return {
        'aursad_train': aursad_train,
        'aursad_val': aursad_val,
        'aursad_test': aursad_test,
        'voraus_train': voraus_train,
        'voraus_test': voraus_test,
    }


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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )


class SimpleEffortPredictor(nn.Module):
    """
    Setpoint → Effort predictor for anomaly detection.

    Idea: Normal operation follows predictable physics.
    Anomalies deviate from this relationship.

    Architecture: Transformer that maps setpoint sequence to effort sequence.
    Anomaly score = reconstruction error of effort given setpoint.
    """

    def __init__(self, setpoint_dim=12, effort_dim=6, hidden_dim=128,
                 num_layers=3, num_heads=4, window_size=256):
        super().__init__()
        self.setpoint_dim = setpoint_dim
        self.effort_dim = effort_dim

        # Encode setpoint
        self.setpoint_proj = nn.Linear(setpoint_dim, hidden_dim)

        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # Predict effort from encoded setpoint
        self.effort_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, effort_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, setpoint, effort=None):
        """
        Args:
            setpoint: (B, T, setpoint_dim)
            effort: (B, T, effort_dim) - target for training, None for inference

        Returns:
            effort_pred: (B, T, effort_dim)
            loss: scalar if effort provided
        """
        B, T, _ = setpoint.shape

        h = self.setpoint_proj(setpoint) + self.pos_enc[:, :T, :]
        h = self.transformer(h)
        h = self.norm(h)
        effort_pred = self.effort_head(h)

        result = {"effort_pred": effort_pred}

        if effort is not None:
            loss = F.mse_loss(effort_pred, effort)
            result["loss"] = loss

        return result

    def anomaly_score(self, setpoint, effort):
        """Compute per-sample anomaly score = MSE between predicted and actual effort."""
        with torch.no_grad():
            result = self.forward(setpoint, effort)
            # Per-sample MSE (average over time and channels)
            error = (result["effort_pred"] - effort) ** 2
            # Mean over time and channels -> per-sample score
            scores = error.mean(dim=(1, 2))
        return scores


class TemporalForecaster(nn.Module):
    """
    Time series forecaster for transfer ratio measurement.

    Given first half of a window, predict second half.
    Transfer ratio = target_MSE / source_MSE.
    """

    def __init__(self, input_dim=18, hidden_dim=128, num_layers=3,
                 num_heads=4, context_len=128, forecast_len=128):
        super().__init__()
        self.context_len = context_len
        self.forecast_len = forecast_len

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, context_len + forecast_len, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_full):
        """
        Args:
            x_full: (B, T, D) full sequence (setpoint + effort concatenated)
        Returns:
            loss, pred
        """
        B, T, D = x_full.shape
        context = x_full[:, :self.context_len, :]
        target = x_full[:, self.context_len:self.context_len + self.forecast_len, :]

        h = self.input_proj(context) + self.pos_enc[:, :self.context_len, :]
        h = self.transformer(h)
        h = self.norm(h)

        # Use last hidden state to predict future
        # Simple approach: linear decode from each context position
        # For now, use last context_len positions to predict forecast_len positions
        pred = self.output_proj(h)  # (B, context_len, D)

        # Match sizes - use last forecast_len outputs
        if pred.shape[1] >= self.forecast_len:
            pred = pred[:, -self.forecast_len:, :]

        loss = F.mse_loss(pred, target)
        return {"loss": loss, "pred": pred, "target": target}


def train_model(model, train_loader, val_loader, epochs, lr=1e-4,
                device='cuda', model_type='predictor'):
    """Train a model with early stopping."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_loss = float('inf')
    best_state = None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0

        for setpoint, effort, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            setpoint = setpoint.to(device)
            effort = effort.to(device)

            optimizer.zero_grad()

            if model_type == 'predictor':
                result = model(setpoint, effort)
            else:  # forecaster
                x = torch.cat([setpoint, effort], dim=-1)
                result = model(x)

            loss = result['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches

        # Validate
        model.eval()
        val_loss = 0
        n_val = 0
        with torch.no_grad():
            for setpoint, effort, _ in val_loader:
                setpoint = setpoint.to(device)
                effort = effort.to(device)

                if model_type == 'predictor':
                    result = model(setpoint, effort)
                else:
                    x = torch.cat([setpoint, effort], dim=-1)
                    result = model(x)

                val_loss += result['loss'].item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        logger.info(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_val_loss


@torch.no_grad()
def evaluate_anomaly_detection(model, test_loader, device='cuda'):
    """Evaluate anomaly detection performance."""
    model.eval()
    all_scores = []
    all_labels = []

    for setpoint, effort, metadata in tqdm(test_loader, desc="Evaluating anomaly detection"):
        setpoint = setpoint.to(device)
        effort = effort.to(device)

        scores = model.anomaly_score(setpoint, effort)
        all_scores.extend(scores.cpu().numpy())
        all_labels.extend([1 if a else 0 for a in metadata["is_anomaly"]])

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    results = {
        'n_total': len(all_labels),
        'n_normal': int((all_labels == 0).sum()),
        'n_anomaly': int((all_labels == 1).sum()),
    }

    if len(np.unique(all_labels)) > 1:
        results['roc_auc'] = float(roc_auc_score(all_labels, all_scores))
        results['pr_auc'] = float(average_precision_score(all_labels, all_scores))

        normal_scores = all_scores[all_labels == 0]
        anomaly_scores = all_scores[all_labels == 1]
        results['normal_mean'] = float(normal_scores.mean())
        results['normal_std'] = float(normal_scores.std())
        results['anomaly_mean'] = float(anomaly_scores.mean())
        results['anomaly_std'] = float(anomaly_scores.std())
        results['separation'] = float(anomaly_scores.mean() - normal_scores.mean())
    else:
        results['roc_auc'] = None
        results['pr_auc'] = None

    return results, all_scores, all_labels


@torch.no_grad()
def evaluate_forecasting(model, test_loader, device='cuda'):
    """Evaluate forecasting MSE."""
    model.eval()
    total_mse = 0
    n_batches = 0

    for setpoint, effort, _ in tqdm(test_loader, desc="Evaluating forecasting"):
        setpoint = setpoint.to(device)
        effort = effort.to(device)
        x = torch.cat([setpoint, effort], dim=-1)

        result = model(x)
        total_mse += result['loss'].item()
        n_batches += 1

    return total_mse / max(n_batches, 1)


def plot_anomaly_results(scores_source, labels_source, scores_target, labels_target,
                         auc_source, auc_target, filename):
    """Plot anomaly detection results."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # ROC curves
    ax = axes[0, 0]
    if auc_source is not None:
        fpr_s, tpr_s, _ = roc_curve(labels_source, scores_source)
        ax.plot(fpr_s, tpr_s, label=f'Source (AURSAD) AUC={auc_source:.3f}', color='blue')
    if auc_target is not None:
        fpr_t, tpr_t, _ = roc_curve(labels_target, scores_target)
        ax.plot(fpr_t, tpr_t, label=f'Target (Voraus) AUC={auc_target:.3f}', color='red')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title('ROC Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Score distributions - Source
    ax = axes[0, 1]
    if labels_source is not None and len(np.unique(labels_source)) > 1:
        ax.hist(scores_source[labels_source == 0], bins=50, alpha=0.5,
                label='Normal', density=True, color='green')
        ax.hist(scores_source[labels_source == 1], bins=50, alpha=0.5,
                label='Anomaly', density=True, color='red')
    ax.set_title('Source (AURSAD) Score Distribution')
    ax.legend()
    ax.set_xlabel('Anomaly Score')

    # Score distributions - Target
    ax = axes[1, 0]
    if labels_target is not None and len(np.unique(labels_target)) > 1:
        ax.hist(scores_target[labels_target == 0], bins=50, alpha=0.5,
                label='Normal', density=True, color='green')
        ax.hist(scores_target[labels_target == 1], bins=50, alpha=0.5,
                label='Anomaly', density=True, color='red')
    ax.set_title('Target (Voraus) Score Distribution')
    ax.legend()
    ax.set_xlabel('Anomaly Score')

    # Score comparison box plot
    ax = axes[1, 1]
    data_to_plot = []
    labels_to_plot = []
    if labels_source is not None:
        data_to_plot.extend([scores_source[labels_source == 0], scores_source[labels_source == 1]])
        labels_to_plot.extend(['Src Normal', 'Src Anomaly'])
    if labels_target is not None:
        data_to_plot.extend([scores_target[labels_target == 0], scores_target[labels_target == 1]])
        labels_to_plot.extend(['Tgt Normal', 'Tgt Anomaly'])
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels_to_plot, patch_artist=True)
        colors = ['lightgreen', 'lightcoral', 'lightblue', 'lightyellow']
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
    ax.set_title('Score Comparison')
    ax.set_ylabel('Anomaly Score')

    plt.suptitle(f'Cross-Machine Anomaly Detection\nSource AUC={auc_source}, Target AUC={auc_target}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved figure: {filename}")


def plot_training_curves(history, filename):
    """Plot training curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['val_loss'], label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def run_experiment(seed=42, epochs=10, norm_mode="episode", hidden_dim=128,
                   num_layers=3, batch_size=64):
    """Run full baseline experiment."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}, Seed: {seed}")

    # Load data
    datasets = load_datasets(
        max_episodes_source=500,
        max_episodes_target=500,
        norm_mode=norm_mode,
        effort_signals=["voltage"],
    )

    # Create dataloaders
    source_train_loader = make_loader(datasets['aursad_train'], batch_size=batch_size, shuffle=True)
    source_val_loader = make_loader(datasets['aursad_val'], batch_size=batch_size)
    source_test_loader = make_loader(datasets['aursad_test'], batch_size=batch_size)
    target_test_loader = make_loader(datasets['voraus_test'], batch_size=batch_size)
    target_train_loader = make_loader(datasets['voraus_train'], batch_size=batch_size, shuffle=True)

    logger.info(f"Source train: {len(source_train_loader)} batches")
    logger.info(f"Source test: {len(source_test_loader)} batches")
    logger.info(f"Target test: {len(target_test_loader)} batches")

    results = {}

    # ============================================================
    # OBJECTIVE 1: Anomaly Detection
    # ============================================================
    logger.info("=" * 60)
    logger.info("OBJECTIVE 1: Anomaly Detection (Setpoint → Effort Prediction)")
    logger.info("=" * 60)

    predictor = SimpleEffortPredictor(
        setpoint_dim=12, effort_dim=6,
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=4, window_size=256,
    ).to(device)

    logger.info(f"Predictor params: {sum(p.numel() for p in predictor.parameters()):,}")

    # Train on source (healthy AURSAD data)
    pred_history, pred_best_val = train_model(
        predictor, source_train_loader, source_val_loader,
        epochs=epochs, lr=1e-4, device=device, model_type='predictor',
    )

    plot_training_curves(pred_history, FIGURES_DIR / f"exp01_predictor_training_seed{seed}.png")

    # Evaluate anomaly detection on source
    source_ad_results, source_scores, source_labels = evaluate_anomaly_detection(
        predictor, source_test_loader, device
    )

    # Evaluate anomaly detection on target (zero-shot)
    target_ad_results, target_scores, target_labels = evaluate_anomaly_detection(
        predictor, target_test_loader, device
    )

    logger.info(f"Source AD: AUC={source_ad_results.get('roc_auc', 'N/A')}, "
                f"n_normal={source_ad_results['n_normal']}, n_anomaly={source_ad_results['n_anomaly']}")
    logger.info(f"Target AD: AUC={target_ad_results.get('roc_auc', 'N/A')}, "
                f"n_normal={target_ad_results['n_normal']}, n_anomaly={target_ad_results['n_anomaly']}")

    # Plot anomaly results
    plot_anomaly_results(
        source_scores, source_labels, target_scores, target_labels,
        source_ad_results.get('roc_auc'), target_ad_results.get('roc_auc'),
        FIGURES_DIR / f"exp01_anomaly_detection_seed{seed}.png"
    )

    results['anomaly_detection'] = {
        'source': source_ad_results,
        'target': target_ad_results,
    }

    # ============================================================
    # OBJECTIVE 2: Forecasting Transfer Ratio
    # ============================================================
    logger.info("=" * 60)
    logger.info("OBJECTIVE 2: Forecasting Transfer Ratio")
    logger.info("=" * 60)

    forecaster = TemporalForecaster(
        input_dim=18,  # 12 setpoint + 6 effort
        hidden_dim=hidden_dim, num_layers=num_layers,
        num_heads=4, context_len=128, forecast_len=128,
    ).to(device)

    logger.info(f"Forecaster params: {sum(p.numel() for p in forecaster.parameters()):,}")

    # Train on source
    fc_history, fc_best_val = train_model(
        forecaster, source_train_loader, source_val_loader,
        epochs=epochs, lr=1e-4, device=device, model_type='forecaster',
    )

    plot_training_curves(fc_history, FIGURES_DIR / f"exp01_forecaster_training_seed{seed}.png")

    # Evaluate on source
    source_mse = evaluate_forecasting(forecaster, source_test_loader, device)

    # Evaluate on target (zero-shot)
    target_mse = evaluate_forecasting(forecaster, target_test_loader, device)

    transfer_ratio = target_mse / max(source_mse, 1e-8)

    logger.info(f"Source MSE: {source_mse:.6f}")
    logger.info(f"Target MSE: {target_mse:.6f}")
    logger.info(f"Transfer Ratio: {transfer_ratio:.4f}")

    results['forecasting'] = {
        'source_mse': source_mse,
        'target_mse': target_mse,
        'transfer_ratio': transfer_ratio,
    }

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print(f"EXPERIMENT RESULTS (seed={seed}, norm={norm_mode}, epochs={epochs})")
    print("=" * 70)
    print(f"Anomaly Detection:")
    print(f"  Source AUC: {source_ad_results.get('roc_auc', 'N/A')}")
    print(f"  Target AUC: {target_ad_results.get('roc_auc', 'N/A')} (need >= 0.70)")
    print(f"Forecasting:")
    print(f"  Source MSE: {source_mse:.6f}")
    print(f"  Target MSE: {target_mse:.6f}")
    print(f"  Transfer Ratio: {transfer_ratio:.4f} (need <= 1.5)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--norm-mode", type=str, default="episode", choices=["episode", "global", "none"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    results = run_experiment(
        seed=args.seed,
        epochs=args.epochs,
        norm_mode=args.norm_mode,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"exp01_seed{args.seed}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")

#!/usr/bin/env python
"""
Experiment 03: Improved Anomaly Detection via Multi-Scale Reconstruction

Key insight from diagnosis: simple prediction error doesn't separate normal from anomaly.
This experiment tries multiple approaches:

1. Multi-scale reconstruction: Reconstruct at multiple temporal resolutions
2. Spectral features: Use FFT-based features that capture frequency differences
3. Statistical deviation: Compare window statistics to learned "normal" statistics
4. Ensemble scoring: Combine multiple anomaly scores

Architecture: Train on healthy data from ALL available sources.
"""

import sys
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "autoresearch" / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "autoresearch" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def collate_fn(batch):
    setpoints = torch.stack([item[0] for item in batch])
    efforts = torch.stack([item[1] for item in batch])
    metadata = {
        "is_anomaly": [item[2].get("is_anomaly", False) for item in batch],
        "fault_type": [item[2].get("fault_type", "normal") for item in batch],
    }
    return setpoints, efforts, metadata


def load_dataset(source, split, window_size=128, stride=64, norm_mode="episode"):
    from industrialjepa.data.factorynet import FactoryNetDataset, FactoryNetConfig

    max_eps = {"aursad": 500, "voraus": 500, "cnc": None}.get(source, 200)
    config = FactoryNetConfig(
        data_source=source, max_episodes=max_eps,
        window_size=window_size, stride=stride,
        normalize=True, norm_mode=norm_mode,
        effort_signals=["voltage", "current"],
        setpoint_signals=["position", "velocity"],
        unified_setpoint_dim=12, unified_effort_dim=6,
        train_healthy_only=(split == "train"),
    )
    return FactoryNetDataset(config, split=split)


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            return (x - self._mean) / self._std * self.weight + self.bias
        else:
            return (x - self.bias) / (self.weight + self.eps) * self._std + self._mean


class MultiScaleReconstructionDetector(nn.Module):
    """
    Multi-scale temporal reconstruction for anomaly detection.

    Key idea: Normal signals are reconstructable at multiple time scales.
    Anomalies disrupt patterns at specific scales.

    Approach:
    1. Downsample input to multiple scales (1x, 2x, 4x, 8x)
    2. Reconstruct each scale from a shared latent space
    3. Anomaly score = weighted sum of reconstruction errors across scales
    """

    def __init__(self, input_dim=18, hidden_dim=128, num_layers=3,
                 num_heads=4, window_size=128):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.scales = [1, 2, 4, 8]

        # RevIN for domain invariance
        self.revin = RevIN(input_dim)

        # Shared encoder
        self.encoder_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_enc = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(hidden_dim)

        # Per-scale decoders
        self.scale_decoders = nn.ModuleDict()
        for scale in self.scales:
            seq_len = window_size // scale
            self.scale_decoders[str(scale)] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        # Learnable scale weights for anomaly scoring
        self.scale_weights = nn.Parameter(torch.ones(len(self.scales)))

        # Spectral decoder (reconstruct frequency features)
        self.spectral_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim * (window_size // 2 + 1)),
        )

    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder_proj(x) + self.pos_enc[:, :x.shape[1], :]
        h = self.encoder(h)
        h = self.encoder_norm(h)
        return h

    def forward(self, setpoint, effort):
        """Training forward pass."""
        x = torch.cat([setpoint, effort], dim=-1)
        x_norm = self.revin(x, mode='norm')

        # Encode
        h = self.encode(x_norm)

        # Multi-scale reconstruction loss
        total_loss = 0
        scale_losses = {}

        for i, scale in enumerate(self.scales):
            if scale == 1:
                target = x_norm
                pred = self.scale_decoders[str(scale)](h)
            else:
                # Downsample target by averaging
                B, T, D = x_norm.shape
                new_T = T // scale
                target = x_norm[:, :new_T * scale, :].reshape(B, new_T, scale, D).mean(dim=2)
                # Pool encoder output similarly
                h_pooled = h[:, :new_T * scale, :].reshape(B, new_T, scale, -1).mean(dim=2)
                pred = self.scale_decoders[str(scale)](h_pooled)

            scale_loss = F.mse_loss(pred, target)
            scale_losses[f'scale_{scale}'] = scale_loss.item()
            total_loss = total_loss + scale_loss

        # Spectral loss: reconstruct magnitude spectrum
        x_fft = torch.fft.rfft(x_norm, dim=1)
        x_mag = x_fft.abs()  # (B, T//2+1, D)
        B, F_bins, D = x_mag.shape

        h_global = h.mean(dim=1)  # Global pooling: (B, hidden)
        spectral_pred = self.spectral_decoder(h_global).reshape(B, -1, D)
        spectral_pred = spectral_pred[:, :F_bins, :]

        spectral_loss = F.mse_loss(spectral_pred, x_mag)
        total_loss = total_loss + spectral_loss * 0.1
        scale_losses['spectral'] = spectral_loss.item()

        return {"loss": total_loss, "scale_losses": scale_losses}

    @torch.no_grad()
    def anomaly_score(self, setpoint, effort):
        """Compute multi-scale anomaly score."""
        x = torch.cat([setpoint, effort], dim=-1)
        x_norm = self.revin(x, mode='norm')

        h = self.encode(x_norm)

        # Compute per-scale reconstruction errors
        scale_errors = []
        weights = F.softmax(self.scale_weights, dim=0)

        for i, scale in enumerate(self.scales):
            if scale == 1:
                target = x_norm
                pred = self.scale_decoders[str(scale)](h)
            else:
                B, T, D = x_norm.shape
                new_T = T // scale
                target = x_norm[:, :new_T * scale, :].reshape(B, new_T, scale, D).mean(dim=2)
                h_pooled = h[:, :new_T * scale, :].reshape(B, new_T, scale, -1).mean(dim=2)
                pred = self.scale_decoders[str(scale)](h_pooled)

            error = ((pred - target) ** 2).mean(dim=(1, 2))  # (B,)
            scale_errors.append(error * weights[i])

        # Spectral error
        x_fft = torch.fft.rfft(x_norm, dim=1)
        x_mag = x_fft.abs()
        B, F_bins, D = x_mag.shape
        h_global = h.mean(dim=1)
        spectral_pred = self.spectral_decoder(h_global).reshape(B, -1, D)[:, :F_bins, :]
        spectral_error = ((spectral_pred - x_mag) ** 2).mean(dim=(1, 2))

        # Combined score
        total_score = sum(scale_errors) + spectral_error * 0.1
        return total_score


class StatisticalDeviationDetector:
    """
    Non-parametric anomaly detector based on statistical properties.

    Computes per-channel statistics of "normal" training data, then
    scores test windows by how much they deviate.
    """

    def __init__(self):
        self.normal_stats = None

    def fit(self, train_loader, device='cpu'):
        """Compute statistics of normal training data."""
        all_means = []
        all_stds = []
        all_ranges = []
        all_diffs = []

        for setpoint, effort, _ in tqdm(train_loader, desc="Computing normal stats"):
            x = torch.cat([setpoint, effort], dim=-1).numpy()
            for i in range(x.shape[0]):
                w = x[i]  # (T, D)
                all_means.append(w.mean(axis=0))
                all_stds.append(w.std(axis=0))
                all_ranges.append(w.max(axis=0) - w.min(axis=0))
                all_diffs.append(np.abs(np.diff(w, axis=0)).mean(axis=0))

        self.normal_stats = {
            'mean': {'center': np.mean(all_means, axis=0), 'spread': np.std(all_means, axis=0)},
            'std': {'center': np.mean(all_stds, axis=0), 'spread': np.std(all_stds, axis=0)},
            'range': {'center': np.mean(all_ranges, axis=0), 'spread': np.std(all_ranges, axis=0)},
            'diff': {'center': np.mean(all_diffs, axis=0), 'spread': np.std(all_diffs, axis=0)},
        }

    def score(self, setpoint, effort):
        """Score a batch. Returns per-sample scores."""
        x = torch.cat([setpoint, effort], dim=-1).numpy()
        scores = []

        for i in range(x.shape[0]):
            w = x[i]
            total_z = 0

            # Z-score each statistic against normal distribution
            w_mean = w.mean(axis=0)
            w_std = w.std(axis=0)
            w_range = w.max(axis=0) - w.min(axis=0)
            w_diff = np.abs(np.diff(w, axis=0)).mean(axis=0)

            for stat_name, w_val in [('mean', w_mean), ('std', w_std),
                                      ('range', w_range), ('diff', w_diff)]:
                z = np.abs(w_val - self.normal_stats[stat_name]['center']) / \
                    (self.normal_stats[stat_name]['spread'] + 1e-8)
                total_z += z.mean()

            scores.append(total_z)

        return np.array(scores)


def train_model(model, train_loader, val_loader, epochs, lr=1e-4, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for sp, eff, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            sp, eff = sp.to(device), eff.to(device)
            optimizer.zero_grad()
            result = model(sp, eff)
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += result['loss'].item()
            n += 1

        train_loss = total_loss / max(n, 1)

        model.eval()
        val_loss = 0
        nv = 0
        with torch.no_grad():
            for sp, eff, _ in val_loader:
                sp, eff = sp.to(device), eff.to(device)
                result = model(sp, eff)
                val_loss += result['loss'].item()
                nv += 1
        val_loss /= max(nv, 1)

        scheduler.step()

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_val


@torch.no_grad()
def evaluate(model, test_loader, device='cuda', stat_detector=None):
    """Evaluate with optional ensemble scoring."""
    model.eval()
    all_scores = []
    all_stat_scores = []
    all_labels = []

    for sp, eff, meta in tqdm(test_loader, desc="Evaluating"):
        sp_d, eff_d = sp.to(device), eff.to(device)

        # Neural anomaly scores
        scores = model.anomaly_score(sp_d, eff_d)
        all_scores.extend(scores.cpu().numpy())

        # Statistical anomaly scores
        if stat_detector is not None:
            stat_scores = stat_detector.score(sp, eff)
            all_stat_scores.extend(stat_scores)

        all_labels.extend([1 if a else 0 for a in meta["is_anomaly"]])

    scores = np.array(all_scores)
    labels = np.array(all_labels)

    results = {
        'n_total': len(labels),
        'n_normal': int((labels == 0).sum()),
        'n_anomaly': int((labels == 1).sum()),
    }

    def compute_auc(s, l):
        if len(np.unique(l)) > 1:
            return float(roc_auc_score(l, s))
        return None

    results['neural_auc'] = compute_auc(scores, labels)

    if all_stat_scores:
        stat_scores = np.array(all_stat_scores)
        results['stat_auc'] = compute_auc(stat_scores, labels)

        # Ensemble: normalize and combine
        s_norm = (scores - scores.mean()) / (scores.std() + 1e-8)
        st_norm = (stat_scores - stat_scores.mean()) / (stat_scores.std() + 1e-8)
        ensemble = s_norm + st_norm
        results['ensemble_auc'] = compute_auc(ensemble, labels)

        # Also try max of the two
        max_scores = np.maximum(s_norm, st_norm)
        results['max_auc'] = compute_auc(max_scores, labels)
    else:
        results['stat_auc'] = None
        results['ensemble_auc'] = None

    return results, scores, labels


def plot_results(all_results, filename):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUC comparison across methods
    ax = axes[0]
    methods = ['neural_auc', 'stat_auc', 'ensemble_auc', 'max_auc']
    method_labels = ['Neural', 'Statistical', 'Ensemble (sum)', 'Ensemble (max)']

    for held_out, res in all_results.items():
        aucs = [res.get(m) for m in methods]
        valid = [(l, a) for l, a in zip(method_labels, aucs) if a is not None]
        if valid:
            labels, values = zip(*valid)
            ax.bar(np.arange(len(labels)) + list(all_results.keys()).index(held_out) * 0.25,
                   values, 0.25, label=f'Target: {held_out}', alpha=0.7)

    ax.axhline(y=0.70, color='r', linestyle='--', label='Target ≥ 0.70')
    ax.axhline(y=0.50, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Anomaly Detection Methods')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Summary table
    ax = axes[1]
    ax.axis('off')
    table_data = []
    for held_out, res in all_results.items():
        row = [held_out, f"{res.get('n_normal', 0)}", f"{res.get('n_anomaly', 0)}"]
        for m in methods:
            v = res.get(m)
            row.append(f"{v:.3f}" if v is not None else "N/A")
        table_data.append(row)

    headers = ['Target', '#Normal', '#Anomaly', 'Neural', 'Stat', 'Ensemble', 'Max']
    table = ax.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('Results Summary', fontweight='bold')

    plt.suptitle('Multi-Scale + Statistical Anomaly Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved: {filename}")


def main(seed=42, epochs=20, window_size=128):
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Device: {device}, Seed: {seed}, Window: {window_size}")

    all_results = {}

    # Test on each source that has anomaly labels
    for target_source in ["aursad", "voraus"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"TARGET: {target_source}")
        logger.info(f"{'='*60}")

        # Train sources = all except target
        train_sources = [s for s in ["aursad", "voraus", "cnc"] if s != target_source]

        # Load training data (healthy only)
        train_datasets = []
        for src in train_sources:
            try:
                ds = load_dataset(src, "train", window_size=window_size, stride=window_size//2)
                if len(ds) > 0:
                    train_datasets.append(ds)
                    logger.info(f"  Train source {src}: {len(ds)} windows")
            except Exception as e:
                logger.warning(f"  Failed to load {src}: {e}")

        if not train_datasets:
            logger.warning("No training data!")
            continue

        combined_train = ConcatDataset(train_datasets)
        train_loader = DataLoader(combined_train, batch_size=64, shuffle=True,
                                  num_workers=0, pin_memory=True, collate_fn=collate_fn)

        # Load validation (reuse first train source)
        val_loader = DataLoader(train_datasets[0], batch_size=64, shuffle=False,
                                num_workers=0, collate_fn=collate_fn)

        # Load target test data
        try:
            target_test = load_dataset(target_source, "test", window_size=window_size,
                                       stride=window_size//2)
            test_loader = DataLoader(target_test, batch_size=64, shuffle=False,
                                     num_workers=0, pin_memory=True, collate_fn=collate_fn)
            logger.info(f"  Target test: {len(target_test)} windows")
        except Exception as e:
            logger.warning(f"  Failed to load target test: {e}")
            continue

        # Train multi-scale reconstruction detector
        logger.info("Training multi-scale reconstruction detector...")
        model = MultiScaleReconstructionDetector(
            input_dim=18, hidden_dim=128, num_layers=3,
            num_heads=4, window_size=window_size,
        ).to(device)

        logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

        train_model(model, train_loader, val_loader, epochs=epochs, device=device)

        # Train statistical detector
        logger.info("Training statistical detector...")
        stat_detector = StatisticalDeviationDetector()
        stat_detector.fit(train_loader)

        # Evaluate
        logger.info("Evaluating on target...")
        results, scores, labels = evaluate(model, test_loader, device, stat_detector)

        logger.info(f"Results for target={target_source}:")
        logger.info(f"  Neural AUC: {results.get('neural_auc', 'N/A')}")
        logger.info(f"  Statistical AUC: {results.get('stat_auc', 'N/A')}")
        logger.info(f"  Ensemble AUC: {results.get('ensemble_auc', 'N/A')}")
        logger.info(f"  Max AUC: {results.get('max_auc', 'N/A')}")

        all_results[target_source] = results

    # Summary
    print("\n" + "="*70)
    print("MULTI-SCALE ANOMALY DETECTION RESULTS")
    print("="*70)
    for target, res in all_results.items():
        print(f"\nTarget: {target}")
        print(f"  Normal: {res['n_normal']}, Anomaly: {res['n_anomaly']}")
        print(f"  Neural AUC:    {res.get('neural_auc', 'N/A')}")
        print(f"  Statistical:   {res.get('stat_auc', 'N/A')}")
        print(f"  Ensemble:      {res.get('ensemble_auc', 'N/A')}")
        print(f"  Max Ensemble:  {res.get('max_auc', 'N/A')}")

    if all_results:
        plot_results(all_results, FIGURES_DIR / f"exp03_anomaly_multiscale_seed{seed}.png")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=128)
    args = parser.parse_args()

    results = main(seed=args.seed, epochs=args.epochs, window_size=args.window_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(RESULTS_DIR / f"exp03_seed{args.seed}_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

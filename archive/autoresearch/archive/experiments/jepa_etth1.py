"""
JEPA baseline for ETTh1 long-term forecasting.

Architecture:
  Encoder: lookback window -> transformer -> latent sequence
  Predictor: latent(lookback) -> latent(horizon) via MLP
  Decoder: latent(horizon) -> forecast
  EMA target encoder on the future window (standard JEPA)

Loss: L2 in latent space (JEPA) + L2 on decoded forecast (supervision)
"""

import sys
sys.path.insert(0, ".")

import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.industrialjepa.data.ett import get_etth1_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class PatchEmbedding(nn.Module):
    """Patch-based embedding for time series (like PatchTST)."""
    def __init__(self, patch_len, d_model, n_channels, stride=None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride or patch_len
        self.proj = nn.Linear(patch_len * n_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        # Unfold into patches: (B, n_patches, patch_len * C)
        x = x.unfold(1, self.patch_len, self.stride)  # (B, n_patches, C, patch_len)
        x = x.permute(0, 1, 3, 2).reshape(B, -1, self.patch_len * C)  # (B, n_patches, patch_len*C)
        x = self.proj(x)
        x = self.norm(x)
        return x  # (B, n_patches, d_model)


class TransformerEncoder(nn.Module):
    """Small transformer encoder for time series."""
    def __init__(self, d_model=128, n_heads=4, n_layers=3, d_ff=256, dropout=0.1, max_len=500):
        super().__init__()
        # Positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, S, d_model)
        x = x + self.pe[:, :x.size(1)]
        x = self.transformer(x)
        x = self.norm(x)
        return x


class JEPAForecaster(nn.Module):
    """
    JEPA-based forecasting model.

    Modes:
      - 'jepa+supervised': JEPA latent loss + decoded forecast loss (default)
      - 'supervised_only': Just encoder -> predictor -> decoder, no EMA/JEPA loss
      - 'jepa_only': Only JEPA latent loss, decoder for eval only
    """
    def __init__(
        self,
        n_channels=7,
        lookback=96,
        horizon=96,
        d_model=128,
        n_heads=4,
        n_encoder_layers=3,
        d_ff=256,
        patch_len=16,
        latent_dim=64,
        dropout=0.1,
        ema_momentum=0.996,
        mode='jepa+supervised',
    ):
        super().__init__()
        self.n_channels = n_channels
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.mode = mode
        self.ema_momentum = ema_momentum

        # Patch embedding
        self.patch_emb = PatchEmbedding(patch_len, d_model, n_channels, stride=patch_len)
        n_patches_lookback = lookback // patch_len
        n_patches_horizon = max(1, horizon // patch_len)
        self.n_patches_lookback = n_patches_lookback
        self.n_patches_horizon = n_patches_horizon

        # Encoder
        self.encoder = TransformerEncoder(d_model, n_heads, n_encoder_layers, d_ff, dropout)

        # Latent projection (from d_model to latent_dim)
        self.to_latent = nn.Linear(d_model, latent_dim)

        # Predictor: maps lookback latent sequence -> horizon latent sequence
        # Use a small MLP on the pooled representation
        self.predictor = nn.Sequential(
            nn.Linear(n_patches_lookback * latent_dim, d_ff),
            nn.GELU(),
            nn.LayerNorm(d_ff),
            nn.Linear(d_ff, n_patches_horizon * latent_dim),
        )

        # Decoder: latent -> forecast
        self.decoder = nn.Sequential(
            nn.Linear(n_patches_horizon * latent_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, horizon * n_channels),
        )

        # EMA target encoder (for JEPA)
        if 'jepa' in mode:
            self.target_patch_emb = copy.deepcopy(self.patch_emb)
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_to_latent = copy.deepcopy(self.to_latent)
            for p in self.target_patch_emb.parameters():
                p.requires_grad = False
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            for p in self.target_to_latent.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_ema(self):
        """Update EMA target encoder."""
        if 'jepa' not in self.mode:
            return
        m = self.ema_momentum
        for online, target in [
            (self.patch_emb, self.target_patch_emb),
            (self.encoder, self.target_encoder),
            (self.to_latent, self.target_to_latent),
        ]:
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data = m * p_t.data + (1 - m) * p_o.data

    def encode(self, x):
        """Encode input sequence to latent."""
        patches = self.patch_emb(x)       # (B, n_patches, d_model)
        encoded = self.encoder(patches)    # (B, n_patches, d_model)
        latent = self.to_latent(encoded)   # (B, n_patches, latent_dim)
        return latent

    @torch.no_grad()
    def encode_target(self, x):
        """Encode target with EMA encoder."""
        patches = self.target_patch_emb(x)
        encoded = self.target_encoder(patches)
        latent = self.target_to_latent(encoded)
        return latent

    def predict_and_decode(self, z_lookback):
        """Predict future latent and decode to forecast."""
        B = z_lookback.shape[0]
        z_flat = z_lookback.reshape(B, -1)         # (B, n_patches_lookback * latent_dim)
        z_pred = self.predictor(z_flat)             # (B, n_patches_horizon * latent_dim)
        forecast = self.decoder(z_pred)             # (B, horizon * n_channels)
        forecast = forecast.reshape(B, self.horizon, self.n_channels)
        z_pred_seq = z_pred.reshape(B, self.n_patches_horizon, self.latent_dim)
        return forecast, z_pred_seq

    def forward(self, x, y=None):
        """
        Args:
            x: (B, lookback, C) - input window
            y: (B, horizon, C) - target window (for JEPA target encoding)

        Returns:
            forecast: (B, horizon, C)
            losses: dict of loss components
        """
        # Encode lookback
        z_lookback = self.encode(x)  # (B, n_patches_L, latent_dim)

        # Predict and decode
        forecast, z_pred = self.predict_and_decode(z_lookback)

        losses = {}

        # JEPA loss: predict target encoding
        if 'jepa' in self.mode and y is not None:
            z_target = self.encode_target(y)  # (B, n_patches_H, latent_dim)
            # Align shapes if needed
            min_patches = min(z_pred.shape[1], z_target.shape[1])
            losses['jepa'] = F.mse_loss(z_pred[:, :min_patches], z_target[:, :min_patches])

        # Supervised forecast loss
        if 'supervised' in self.mode and y is not None:
            losses['supervised'] = F.mse_loss(forecast, y)

        return forecast, losses


def evaluate(model, test_loader, device=DEVICE):
    """Compute MSE and MAE on test set."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            forecast, _ = model(x)
            all_preds.append(forecast.cpu())
            all_targets.append(y)
    preds = torch.cat(all_preds, 0)
    targets = torch.cat(all_targets, 0)
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    return mse, mae


def train_model(
    model, train_loader, val_loader,
    epochs=50, lr=1e-3, patience=10, device=DEVICE,
    jepa_weight=1.0, supervised_weight=1.0,
):
    """Train JEPA forecaster with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mse = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses = {'total': 0, 'jepa': 0, 'supervised': 0}
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            forecast, losses = model(x, y)

            total_loss = 0
            if 'jepa' in losses:
                total_loss = total_loss + jepa_weight * losses['jepa']
                epoch_losses['jepa'] += losses['jepa'].item()
            if 'supervised' in losses:
                total_loss = total_loss + supervised_weight * losses['supervised']
                epoch_losses['supervised'] += losses['supervised'].item()

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if 'jepa' in model.mode:
                model.update_ema()

            epoch_losses['total'] += total_loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        val_mse, val_mae = evaluate(model, val_loader, device)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
            print(f"  Epoch {epoch+1:3d}: train_loss={avg['total']:.6f} "
                  f"(jepa={avg['jepa']:.6f}, sup={avg['supervised']:.6f}) "
                  f"val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_mse


def run_experiment(
    mode='jepa+supervised',
    horizons=[96, 192, 336, 720],
    seeds=[42, 123, 456],
    d_model=128,
    n_heads=4,
    n_encoder_layers=3,
    d_ff=256,
    patch_len=16,
    latent_dim=64,
    epochs=50,
    lr=1e-3,
    jepa_weight=1.0,
    supervised_weight=1.0,
    label="",
):
    """Run experiment across horizons and seeds."""
    results = {}

    for H in horizons:
        print(f"\n--- Horizon={H}, mode={mode} {label} ---")
        seed_mses, seed_maes = [], []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            train_loader, val_loader, test_loader, _ = get_etth1_loaders(
                csv_path="data/ETTh1.csv", lookback=96, horizon=H, batch_size=32
            )

            model = JEPAForecaster(
                n_channels=7, lookback=96, horizon=H,
                d_model=d_model, n_heads=n_heads, n_encoder_layers=n_encoder_layers,
                d_ff=d_ff, patch_len=patch_len, latent_dim=latent_dim,
                mode=mode,
            ).to(DEVICE)

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if seed == seeds[0]:
                print(f"  Trainable params: {n_params:,}")

            t0 = time.time()
            model, _ = train_model(
                model, train_loader, val_loader,
                epochs=epochs, lr=lr,
                jepa_weight=jepa_weight, supervised_weight=supervised_weight,
            )
            dt = time.time() - t0

            mse, mae = evaluate(model, test_loader)
            seed_mses.append(mse)
            seed_maes.append(mae)
            print(f"  Seed {seed}: MSE={mse:.6f}, MAE={mae:.6f} ({dt:.1f}s)")

        avg_mse = np.mean(seed_mses)
        avg_mae = np.mean(seed_maes)
        std_mse = np.std(seed_mses)
        results[H] = {
            'mse': avg_mse, 'mae': avg_mae, 'std_mse': std_mse,
            'all_mse': seed_mses, 'all_mae': seed_maes,
        }
        print(f"  AVG: MSE={avg_mse:.6f}±{std_mse:.6f}, MAE={avg_mae:.6f}")

    return results


def main():
    print("="*80)
    print("JEPA ETTh1 Forecasting Experiments")
    print("="*80)

    all_results = {}

    # Experiment 1: JEPA + Supervised (default)
    print("\n" + "="*80)
    print("EXP 1: JEPA + Supervised")
    print("="*80)
    all_results['jepa+supervised'] = run_experiment(mode='jepa+supervised')

    # Experiment 2: Supervised only (ablation - no JEPA)
    print("\n" + "="*80)
    print("EXP 2: Supervised only (no JEPA)")
    print("="*80)
    all_results['supervised_only'] = run_experiment(mode='supervised_only')

    # Experiment 3: JEPA only (ablation - no direct supervision)
    print("\n" + "="*80)
    print("EXP 3: JEPA only (no direct supervision)")
    print("="*80)
    all_results['jepa_only'] = run_experiment(mode='jepa_only')

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Mode':<25} {'H=96 MSE':>10} {'H=96 MAE':>10} {'H=192 MSE':>10} {'H=336 MSE':>10} {'H=720 MSE':>10}")
    print("-"*85)
    for mode_name, res in all_results.items():
        row = f"{mode_name:<25}"
        for H in [96, 192, 336, 720]:
            row += f" {res[H]['mse']:>10.6f}"
            if H == 96:
                row += f" {res[H]['mae']:>10.6f}"
        print(row)

    return all_results


if __name__ == "__main__":
    all_results = main()

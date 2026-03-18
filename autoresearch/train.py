#!/usr/bin/env python3
"""
JEPA Multi-Step Forecasting on ETTh1 Benchmark
This file IS modified by the agent.

Run: python train.py

Target: minimize val_mse (comparable to published baselines)
Time limit: ~5 minutes per experiment

SOTA Baselines (ETTh1, horizon 96):
- iTransformer: 0.386 MSE
- PatchTST: 0.414 MSE
- DLinear: 0.456 MSE

Our goal: MSE < 0.386
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from prepare import get_dataloaders

# ============================================================================
# HYPERPARAMETERS (Agent can modify these)
# ============================================================================

EPOCHS = 10             # Increase for better convergence
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01

# Sequence lengths
SEQ_LEN = 96            # Input context length
PRED_LEN = 96           # Prediction horizon (96, 192, 336, 720)

# Model architecture
D_MODEL = 256           # Model dimension
N_HEADS = 8             # Attention heads
E_LAYERS = 3            # Encoder layers
D_FF = 512              # Feedforward dimension
DROPOUT = 0.1

# JEPA-specific
USE_JEPA = True         # Use JEPA (latent prediction) vs direct prediction
LATENT_DIM = 128        # Latent space dimension
EMA_MOMENTUM = 0.99     # Target encoder momentum

# Patch-based encoding (like PatchTST)
USE_PATCHES = True
PATCH_LEN = 16          # Patch length
STRIDE = 8              # Patch stride

# Channel handling
CHANNEL_INDEPENDENT = False  # True = PatchTST style, False = cross-channel attention


# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    """Convert time series to patches (like PatchTST)."""

    def __init__(self, num_features: int, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * num_features, d_model)

    def forward(self, x):
        # x: (B, T, C)
        B, T, C = x.shape
        # Create patches
        patches = x.unfold(1, self.patch_len, self.stride)  # (B, num_patches, C, patch_len)
        patches = patches.permute(0, 1, 3, 2)  # (B, num_patches, patch_len, C)
        patches = patches.reshape(B, -1, self.patch_len * C)  # (B, num_patches, patch_len * C)
        return self.proj(patches)


class RevIN(nn.Module):
    """Reversible Instance Normalization for distribution shift."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        # x: (B, T, C)
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x


class TransformerEncoder(nn.Module):
    """Standard Transformer encoder."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x):
        return self.encoder(x)


# ============================================================================
# MAIN MODEL
# ============================================================================

class JEPAForecaster(nn.Module):
    """JEPA-based multi-step forecaster.

    Key idea: Instead of directly predicting future values, predict in latent space
    then decode. This allows learning more abstract representations.
    """

    def __init__(
        self,
        num_features: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        use_jepa: bool = True,
        latent_dim: int = 128,
        ema_momentum: float = 0.99,
        use_patches: bool = True,
        patch_len: int = 16,
        stride: int = 8,
        channel_independent: bool = False,
    ):
        super().__init__()

        self.num_features = num_features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_jepa = use_jepa
        self.ema_momentum = ema_momentum
        self.use_patches = use_patches
        self.channel_independent = channel_independent

        # RevIN for handling distribution shift
        self.revin = RevIN(num_features)

        # Input embedding
        if use_patches:
            self.num_patches = (seq_len - patch_len) // stride + 1
            self.patch_embed = PatchEmbedding(num_features, patch_len, stride, d_model)
            self.pos_embed = PositionalEncoding(d_model, self.num_patches + pred_len)
        else:
            self.input_proj = nn.Linear(num_features, d_model)
            self.pos_embed = PositionalEncoding(d_model, seq_len + pred_len)

        # Encoder
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, e_layers, dropout)

        if use_jepa:
            # JEPA: predict in latent space
            self.to_latent = nn.Linear(d_model, latent_dim)
            self.predictor = nn.Sequential(
                nn.Linear(latent_dim, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, latent_dim * pred_len),
            )
            self.from_latent = nn.Linear(latent_dim, num_features)

            # Target encoder (EMA)
            self.target_encoder = TransformerEncoder(d_model, n_heads, d_ff, e_layers, dropout)
            self.target_to_latent = nn.Linear(d_model, latent_dim)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            for p in self.target_to_latent.parameters():
                p.requires_grad = False
            self._init_target_encoder()
        else:
            # Direct prediction
            self.predictor = nn.Linear(d_model, num_features * pred_len)

        self.latent_dim = latent_dim

    def _init_target_encoder(self):
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data.copy_(p_online.data)
        for p_online, p_target in zip(self.to_latent.parameters(), self.target_to_latent.parameters()):
            p_target.data.copy_(p_online.data)

    @torch.no_grad()
    def update_ema(self):
        if not self.use_jepa:
            return
        for p_online, p_target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_target.data = self.ema_momentum * p_target.data + (1 - self.ema_momentum) * p_online.data
        for p_online, p_target in zip(self.to_latent.parameters(), self.target_to_latent.parameters()):
            p_target.data = self.ema_momentum * p_target.data + (1 - self.ema_momentum) * p_online.data

    def encode(self, x):
        """Encode input sequence."""
        # x: (B, T, C)
        if self.use_patches:
            x = self.patch_embed(x)  # (B, num_patches, d_model)
        else:
            x = self.input_proj(x)  # (B, T, d_model)

        x = self.pos_embed(x)
        x = self.encoder(x)
        return x

    def forward(self, x, y=None):
        """
        Args:
            x: (B, seq_len, num_features) - input sequence
            y: (B, pred_len, num_features) - target sequence (for training)

        Returns:
            pred: (B, pred_len, num_features) - predictions
            loss_dict: dict with losses (if y provided)
        """
        B = x.shape[0]

        # RevIN normalization
        x = self.revin(x, 'norm')

        # Encode
        enc_out = self.encode(x)  # (B, num_patches or T, d_model)

        # Pool to single vector
        enc_pooled = enc_out.mean(dim=1)  # (B, d_model)

        if self.use_jepa:
            # Predict in latent space
            z = self.to_latent(enc_pooled)  # (B, latent_dim)
            z_pred = self.predictor(z)  # (B, latent_dim * pred_len)
            z_pred = z_pred.view(B, self.pred_len, self.latent_dim)  # (B, pred_len, latent_dim)

            # Decode to observation space
            pred = self.from_latent(z_pred)  # (B, pred_len, num_features)
        else:
            # Direct prediction
            pred = self.predictor(enc_pooled)  # (B, num_features * pred_len)
            pred = pred.view(B, self.pred_len, self.num_features)

        # RevIN denormalization
        pred = self.revin(pred, 'denorm')

        loss_dict = {}
        if y is not None:
            # MSE loss in observation space
            mse_loss = F.mse_loss(pred, y)
            loss_dict['mse'] = mse_loss
            loss_dict['total'] = mse_loss

            # JEPA loss in latent space
            if self.use_jepa:
                y_norm = self.revin(y, 'norm')
                with torch.no_grad():
                    if self.use_patches:
                        # Pad y to match patch structure
                        y_padded = F.pad(y_norm, (0, 0, self.seq_len - self.pred_len, 0))
                        y_enc = self.patch_embed(y_padded)
                    else:
                        y_enc = self.input_proj(y_norm)
                    y_enc = self.pos_embed(y_enc)
                    y_enc = self.target_encoder(y_enc)
                    z_target = self.target_to_latent(y_enc.mean(dim=1))  # (B, latent_dim)

                # Latent prediction loss (optional auxiliary)
                z_pred_pooled = z_pred.mean(dim=1)  # (B, latent_dim)
                jepa_loss = F.mse_loss(z_pred_pooled, z_target)
                loss_dict['jepa'] = jepa_loss

        return pred, loss_dict


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_mse = 0
    n_batches = 0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        pred, losses = model(x, y)

        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema()

        total_loss += losses['total'].item()
        total_mse += losses['mse'].item()
        n_batches += 1

    return total_loss / n_batches, total_mse / n_batches


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mse = 0
    total_mae = 0
    n_samples = 0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)

        pred, _ = model(x)

        # Compute metrics
        mse = F.mse_loss(pred, y, reduction='sum').item()
        mae = F.l1_loss(pred, y, reduction='sum').item()

        total_mse += mse
        total_mae += mae
        n_samples += y.numel()

    return total_mse / n_samples, total_mae / n_samples


def main():
    start_time = time.time()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, info = get_dataloaders(
        BATCH_SIZE, SEQ_LEN, PRED_LEN
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Features: {info['num_features']}")
    print(f"  Seq len: {SEQ_LEN}, Pred len: {PRED_LEN}")

    # Create model
    model = JEPAForecaster(
        num_features=info['num_features'],
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        use_jepa=USE_JEPA,
        latent_dim=LATENT_DIM,
        ema_momentum=EMA_MOMENTUM,
        use_patches=USE_PATCHES,
        patch_len=PATCH_LEN,
        stride=STRIDE,
        channel_independent=CHANNEL_INDEPENDENT,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Training
    best_val_mse = float('inf')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_mse = train_epoch(model, train_loader, optimizer, device)
        val_mse, val_mae = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch}: train_mse={train_mse:.4f}, val_mse={val_mse:.4f}, val_mae={val_mae:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), results_dir / 'best_model.pt')

    # Final evaluation on test set
    model.load_state_dict(torch.load(results_dir / 'best_model.pt', weights_only=True))
    test_mse, test_mae = evaluate(model, test_loader, device)

    elapsed = time.time() - start_time

    # Print metrics in standard format (comparable to published results)
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"val_mse: {best_val_mse:.6f}")
    print(f"test_mse: {test_mse:.6f}")
    print(f"test_mae: {test_mae:.6f}")
    print(f"elapsed_seconds: {elapsed:.1f}")
    print(f"parameters: {n_params}")
    print("=" * 50)

    # SOTA comparison
    print("\nSOTA Comparison (ETTh1, horizon 96):")
    print(f"  iTransformer: 0.386 MSE")
    print(f"  PatchTST:     0.414 MSE")
    print(f"  Ours:         {test_mse:.3f} MSE", end="")
    if test_mse < 0.386:
        print(" ✓ SOTA!")
    elif test_mse < 0.414:
        print(" (competitive)")
    else:
        print("")
    print("=" * 50)

    # Save results
    results = {
        'val_mse': best_val_mse,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'elapsed_seconds': elapsed,
        'parameters': n_params,
        'hyperparameters': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'seq_len': SEQ_LEN,
            'pred_len': PRED_LEN,
            'd_model': D_MODEL,
            'n_heads': N_HEADS,
            'e_layers': E_LAYERS,
            'use_jepa': USE_JEPA,
            'use_patches': USE_PATCHES,
            'patch_len': PATCH_LEN,
        },
        'timestamp': datetime.now().isoformat(),
    }

    with open(results_dir / 'latest_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Output for run.py parsing
    print(f"\nval_loss: {best_val_mse:.6f}")

    return best_val_mse


if __name__ == "__main__":
    main()

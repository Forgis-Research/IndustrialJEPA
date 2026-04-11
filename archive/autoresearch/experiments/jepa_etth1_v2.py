"""
JEPA ETTh1 v2: Diagnose overfitting with two fixes.

EXP 4: Channel-independent supervised (like PatchTST) — process each of 7 channels separately
EXP 5: Tiny model supervised — d_model=32, 1 layer, much fewer params
EXP 6: DLinear-style — direct linear with trend-seasonal decomposition
"""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.industrialjepa.data.ett import get_etth1_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
HORIZONS = [96, 192, 336, 720]


# ============================================================
# EXP 4: Channel-Independent Transformer
# ============================================================
class ChannelIndependentForecaster(nn.Module):
    """Process each channel independently through a shared transformer."""
    def __init__(self, lookback=96, horizon=96, d_model=64, n_heads=4, n_layers=2,
                 patch_len=16, dropout=0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        n_patches = lookback // patch_len

        # Shared across channels
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.patch_norm = nn.LayerNorm(d_model)

        # Positional encoding
        pe = torch.zeros(n_patches, d_model)
        position = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Flatten patches -> horizon
        self.head = nn.Linear(n_patches * d_model, horizon)

    def forward(self, x):
        # x: (B, L, C)
        B, L, C = x.shape
        # Reshape to (B*C, L, 1) -> process each channel independently
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)  # (B*C, L, 1)

        # Patch
        x = x.reshape(B * C, L // self.patch_len, self.patch_len)  # (B*C, n_patches, patch_len)
        x = self.patch_proj(x)  # (B*C, n_patches, d_model)
        x = self.patch_norm(x)
        x = x + self.pe

        # Encode
        x = self.encoder(x)  # (B*C, n_patches, d_model)

        # Predict
        x = x.reshape(B * C, -1)
        x = self.head(x)  # (B*C, horizon)

        # Reshape back
        x = x.reshape(B, C, self.horizon).permute(0, 2, 1)  # (B, H, C)
        return x


# ============================================================
# EXP 5: Tiny supervised model
# ============================================================
class TinyForecaster(nn.Module):
    """Very small transformer: d=32, 1 layer."""
    def __init__(self, n_channels=7, lookback=96, horizon=96, d_model=32,
                 patch_len=16, dropout=0.2):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        n_patches = lookback // patch_len

        self.patch_proj = nn.Linear(patch_len * n_channels, d_model)
        self.norm = nn.LayerNorm(d_model)

        pe = torch.zeros(n_patches, d_model)
        position = torch.arange(0, n_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=64,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_patches * d_model, horizon * n_channels),
        )

    def forward(self, x):
        B, L, C = x.shape
        # Patch: (B, n_patches, patch_len * C)
        x = x.unfold(1, self.lookback // (self.lookback // 16), self.lookback // (self.lookback // 16))
        # Actually, simpler:
        n_patches = self.lookback // 16
        x = x.reshape(B, n_patches, 16, C).reshape(B, n_patches, 16 * C)
        x = self.patch_proj(x)
        x = self.norm(x)
        x = x + self.pe
        x = self.encoder(x)
        x = x.reshape(B, -1)
        x = self.head(x)
        return x.reshape(B, self.horizon, self.n_channels)


# ============================================================
# EXP 6: DLinear-style decomposition
# ============================================================
class MovingAvg(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (B, L, C)
        # Pad front
        front = x[:, :1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class DLinearModel(nn.Module):
    """Simplified DLinear: decompose into trend + seasonal, linear each."""
    def __init__(self, lookback=96, horizon=96, n_channels=7, kernel_size=25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.n_channels = n_channels
        self.decomp = MovingAvg(kernel_size)
        # Channel-independent linear
        self.linear_seasonal = nn.Linear(lookback, horizon)
        self.linear_trend = nn.Linear(lookback, horizon)

    def forward(self, x):
        # x: (B, L, C)
        trend = self.decomp(x)          # (B, L, C)
        seasonal = x - trend            # (B, L, C)

        # Channel-independent: (B, C, L) -> linear -> (B, C, H)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)

        return trend + seasonal  # (B, H, C)


# ============================================================
# Training utilities
# ============================================================
def evaluate(model, loader, device=DEVICE):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y)
    preds = torch.cat(preds, 0)
    targets = torch.cat(targets, 0)
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    return mse, mae


def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10, device=DEVICE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_mse = float('inf')
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()

        val_mse, _ = evaluate(model, val_loader, device)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={total_loss/n:.6f} val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def run_experiment(name, model_fn, horizons=HORIZONS, seeds=SEEDS, epochs=50, lr=1e-3):
    print(f"\n{'='*60}")
    print(f"EXP: {name}")
    print(f"{'='*60}")
    results = {}
    for H in horizons:
        print(f"\n--- H={H} ---")
        seed_mses, seed_maes = [], []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)

            train_loader, val_loader, test_loader, _ = get_etth1_loaders(
                csv_path="data/ETTh1.csv", lookback=96, horizon=H, batch_size=32
            )
            model = model_fn(H).to(DEVICE)
            if seed == seeds[0]:
                n = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"  Params: {n:,}")

            t0 = time.time()
            model = train_model(model, train_loader, val_loader, epochs=epochs, lr=lr)
            mse, mae = evaluate(model, test_loader)
            seed_mses.append(mse)
            seed_maes.append(mae)
            print(f"  Seed {seed}: MSE={mse:.6f} MAE={mae:.6f} ({time.time()-t0:.1f}s)")

        avg_mse = np.mean(seed_mses)
        std_mse = np.std(seed_mses)
        avg_mae = np.mean(seed_maes)
        results[H] = {'mse': avg_mse, 'mae': avg_mae, 'std_mse': std_mse}
        print(f"  AVG: MSE={avg_mse:.6f}±{std_mse:.6f} MAE={avg_mae:.6f}")
    return results


def main():
    all_results = {}

    # EXP 4: Channel-independent
    all_results['CI-Transformer'] = run_experiment(
        "Channel-Independent Transformer (d=64, 2 layers)",
        lambda H: ChannelIndependentForecaster(lookback=96, horizon=H, d_model=64, n_heads=4, n_layers=2),
    )

    # EXP 5: Tiny model
    all_results['Tiny-Transformer'] = run_experiment(
        "Tiny Transformer (d=32, 1 layer)",
        lambda H: TinyForecaster(lookback=96, horizon=H, d_model=32),
    )

    # EXP 6: DLinear
    all_results['DLinear'] = run_experiment(
        "DLinear (trend-seasonal decomposition)",
        lambda H: DLinearModel(lookback=96, horizon=H),
        lr=5e-3,
    )

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY v2")
    print(f"{'='*80}")
    print(f"{'Model':<25} {'H=96':>10} {'H=192':>10} {'H=336':>10} {'H=720':>10}")
    print("-" * 70)
    for name, res in all_results.items():
        row = f"{name:<25}"
        for H in HORIZONS:
            row += f" {res[H]['mse']:>10.6f}"
        print(row)

    # Also print reference
    print("-" * 70)
    ref = {
        'Linear (baseline)': {96: 0.5714, 192: 0.7799, 336: 0.9704, 720: 1.1948},
        'Supervised-only v1': {96: 0.8995, 192: 0.8951, 336: 0.9634, 720: 1.0070},
        'PatchTST (published)': {96: 0.370, 192: 0.383, 336: 0.396, 720: 0.419},
    }
    for name, vals in ref.items():
        row = f"{name:<25}"
        for H in HORIZONS:
            row += f" {vals[H]:>10.4f}"
        print(row)

    return all_results


if __name__ == "__main__":
    main()

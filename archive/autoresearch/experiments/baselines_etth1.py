"""
Trivial baselines for ETTh1 forecasting.
- Persistence (last-value)
- Linear
- 1-layer MLP

Reports MSE and MAE on test set for H={96, 192, 336, 720}.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.industrialjepa.data.ett import get_etth1_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 456]
HORIZONS = [96, 192, 336, 720]
LOOKBACK = 96
N_CHANNELS = 7
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 32


def evaluate(model, test_loader, device="cuda"):
    """Compute MSE and MAE on test set."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mse = ((preds - targets) ** 2).mean().item()
    mae = (preds - targets).abs().mean().item()
    return mse, mae


class PersistenceModel(nn.Module):
    """Predict last observed value for all horizon steps."""
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        # x: (B, L, C) -> repeat last value
        last = x[:, -1:, :]  # (B, 1, C)
        return last.expand(-1, self.horizon, -1)


class LinearModel(nn.Module):
    """Simple linear: flatten lookback -> flatten horizon."""
    def __init__(self, lookback, horizon, n_channels):
        super().__init__()
        self.horizon = horizon
        self.n_channels = n_channels
        self.linear = nn.Linear(lookback * n_channels, horizon * n_channels)

    def forward(self, x):
        B = x.shape[0]
        out = self.linear(x.reshape(B, -1))
        return out.reshape(B, self.horizon, self.n_channels)


class MLPModel(nn.Module):
    """1-hidden-layer MLP."""
    def __init__(self, lookback, horizon, n_channels, hidden=512):
        super().__init__()
        self.horizon = horizon
        self.n_channels = n_channels
        inp = lookback * n_channels
        out = horizon * n_channels
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        B = x.shape[0]
        out = self.net(x.reshape(B, -1))
        return out.reshape(B, self.horizon, self.n_channels)


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE):
    """Train with early stopping on val MSE."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_mse = float("inf")
    patience = 10
    wait = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val
        val_mse, _ = evaluate(model, val_loader, device)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    results = {}

    for H in HORIZONS:
        print(f"\n{'='*60}")
        print(f"Horizon = {H}")
        print(f"{'='*60}")

        results[H] = {}

        # Persistence (no training needed, deterministic)
        train_loader, val_loader, test_loader, _ = get_etth1_loaders(
            csv_path="data/ETTh1.csv", lookback=LOOKBACK, horizon=H, batch_size=BATCH_SIZE
        )
        persist = PersistenceModel(H).to(DEVICE)
        mse, mae = evaluate(persist, test_loader, DEVICE)
        results[H]["Persistence"] = {"mse": mse, "mae": mae}
        print(f"Persistence: MSE={mse:.6f}, MAE={mae:.6f}")

        # Linear and MLP: 3 seeds
        for model_name, ModelClass, kwargs in [
            ("Linear", LinearModel, {"lookback": LOOKBACK, "horizon": H, "n_channels": N_CHANNELS}),
            ("MLP", MLPModel, {"lookback": LOOKBACK, "horizon": H, "n_channels": N_CHANNELS}),
        ]:
            seed_mses, seed_maes = [], []
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)

                train_loader, val_loader, test_loader, _ = get_etth1_loaders(
                    csv_path="data/ETTh1.csv", lookback=LOOKBACK, horizon=H, batch_size=BATCH_SIZE
                )

                model = ModelClass(**kwargs).to(DEVICE)
                model = train_model(model, train_loader, val_loader)
                mse, mae = evaluate(model, test_loader, DEVICE)
                seed_mses.append(mse)
                seed_maes.append(mae)
                print(f"  {model_name} seed={seed}: MSE={mse:.6f}, MAE={mae:.6f}")

            avg_mse = np.mean(seed_mses)
            avg_mae = np.mean(seed_maes)
            std_mse = np.std(seed_mses)
            results[H][model_name] = {
                "mse": avg_mse, "mae": avg_mae, "std_mse": std_mse,
                "all_mse": seed_mses, "all_mae": seed_maes,
            }
            print(f"  {model_name} avg: MSE={avg_mse:.6f}±{std_mse:.6f}, MAE={avg_mae:.6f}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Trivial Baselines on ETTh1 (test set)")
    print(f"{'='*80}")
    print(f"{'Model':<15} {'H=96 MSE':>10} {'H=96 MAE':>10} {'H=192 MSE':>10} {'H=336 MSE':>10} {'H=720 MSE':>10}")
    print("-" * 75)
    for model_name in ["Persistence", "Linear", "MLP"]:
        row = f"{model_name:<15}"
        for H in HORIZONS:
            r = results[H][model_name]
            mse = r["mse"]
            row += f" {mse:>10.6f}"
            if H == 96:
                row += f" {r['mae']:>10.6f}"
        print(row)

    return results


if __name__ == "__main__":
    results = main()

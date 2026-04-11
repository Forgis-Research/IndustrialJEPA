"""
C-MAPSS RUL Prediction: Baselines + Role-Based Transfer Experiment
Self-contained experiment for autoresearch.

Models:
  1. Linear (flatten → predict)
  2. LSTM
  3. Transformer (channel-mixing)
  4. CI-Transformer (channel-independent)
  5. Role-Transformer (grouped by component, shared within-component weights)

Transfer experiment: Train on FD001, test on FD002.
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ─── Config ───
DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RUL = 125  # Piece-wise linear RUL cap (standard)
SEQ_LEN = 30   # Lookback window
SEEDS = [42, 123, 456]

# Column names for C-MAPSS (26 columns total)
COLS = ["unit", "cycle"] + \
       [f"setting{i}" for i in range(1, 4)] + \
       [f"s{i}" for i in range(1, 22)]

# Sensor groups by turbofan component (from domain knowledge)
# Based on C-MAPSS sensor mapping:
# Fan: T2(s1), P2(s5), NF(s8), BPR(s12), htBleed(s21)
# HPC: T30(s3), P30(s7), phi(s10), NRc(s11), PCNfR_dmd(s20)
# LPT: T50(s4)
# HPT: T40(s9), Ps30(s13)
# Combustor: T24(s2), farB(s14)
# Nozzle: W31(s15), W32(s16)
# Operating: setting1, setting2, setting3

# Only use the 14 informative sensors (drop near-constant ones)
# Sensors s1, s5, s6, s10, s16, s18, s19 are near-constant in FD001
INFORMATIVE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
                       "s13", "s14", "s15", "s17", "s20", "s21"]

# Component groups for Role-Transformer (using informative sensors only)
COMPONENT_GROUPS = {
    "fan":        ["s8", "s12", "s21"],          # NF, BPR, htBleed
    "hpc":        ["s3", "s7", "s11", "s20"],    # T30, P30, NRc, PCNfR_dmd
    "combustor":  ["s2", "s14"],                 # T24, farB
    "turbine":    ["s4", "s9", "s13"],           # T50, T40, Ps30
    "nozzle":     ["s15", "s17"],                # W31, W32
}

# Verify all informative sensors are covered
_grouped = [s for g in COMPONENT_GROUPS.values() for s in g]
assert set(_grouped) == set(INFORMATIVE_SENSORS), f"Mismatch: {set(INFORMATIVE_SENSORS) - set(_grouped)}"


# ─── Data Loading ───

def load_cmapss(subset="FD001"):
    """Load C-MAPSS subset, return train_df and test info."""
    train_path = DATA_DIR / f"train_{subset}.txt"
    test_path = DATA_DIR / f"test_{subset}.txt"
    rul_path = DATA_DIR / f"RUL_{subset}.txt"

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLS)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLS)
    rul_true = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])

    # Add RUL to training data (piece-wise linear, capped at MAX_RUL)
    train_df["RUL"] = train_df.groupby("unit")["cycle"].transform(
        lambda x: np.minimum(x.max() - x, MAX_RUL)
    )

    # Add RUL to test data (last cycle only)
    test_df["RUL"] = test_df.groupby("unit")["cycle"].transform(
        lambda x: x.max() - x
    )
    # The true RUL is for the last cycle of each unit
    test_rul_map = {}
    for i, row in rul_true.iterrows():
        test_rul_map[i + 1] = row["RUL"]

    return train_df, test_df, test_rul_map


def normalize_cmapss(train_df, test_df, sensor_cols):
    """Per-sensor normalization fit on train."""
    means = train_df[sensor_cols].mean()
    stds = train_df[sensor_cols].std().replace(0, 1)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[sensor_cols] = (train_df[sensor_cols] - means) / stds
    test_df[sensor_cols] = (test_df[sensor_cols] - means) / stds
    return train_df, test_df


class CMAPSSDataset(Dataset):
    """Sliding window dataset for C-MAPSS RUL prediction."""

    def __init__(self, df, sensor_cols, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.sensor_cols = sensor_cols
        self.samples = []

        for unit_id in df["unit"].unique():
            unit_data = df[df["unit"] == unit_id]
            sensors = unit_data[sensor_cols].values
            rul = unit_data["RUL"].values

            # Sliding windows
            for i in range(len(sensors) - seq_len + 1):
                x = sensors[i:i + seq_len]
                y = rul[i + seq_len - 1]  # RUL at end of window
                self.samples.append((x, min(y, MAX_RUL)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])


class CMAPSSTestDataset(Dataset):
    """Test dataset: only last window per unit."""

    def __init__(self, df, sensor_cols, rul_map, seq_len=SEQ_LEN):
        self.samples = []
        for unit_id in df["unit"].unique():
            unit_data = df[df["unit"] == unit_id]
            sensors = unit_data[sensor_cols].values
            if len(sensors) < seq_len:
                # Pad with first row
                pad = np.tile(sensors[0], (seq_len - len(sensors), 1))
                sensors = np.vstack([pad, sensors])
            x = sensors[-seq_len:]
            y = min(rul_map[unit_id], MAX_RUL)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ─── Models ───

class LinearRUL(nn.Module):
    """Flatten + Linear."""
    def __init__(self, n_sensors, seq_len):
        super().__init__()
        self.fc = nn.Linear(n_sensors * seq_len, 1)

    def forward(self, x):  # [B, T, C]
        return self.fc(x.reshape(x.size(0), -1))


class LSTMRUL(nn.Module):
    """LSTM baseline."""
    def __init__(self, n_sensors, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_sensors, hidden, n_layers,
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


class TransformerRUL(nn.Module):
    """Transformer with channel-mixing (all sensors as features per timestep)."""
    def __init__(self, n_sensors, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_sensors, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):  # [B, T, C]
        x = self.input_proj(x) + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return self.fc(x[:, -1])


class CITransformerRUL(nn.Module):
    """Channel-Independent Transformer: each sensor processed independently."""
    def __init__(self, n_sensors, d_model=32, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_sensors = n_sensors
        # Shared across channels
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.channel_fc = nn.Linear(d_model, 1)  # per-channel output
        self.final_fc = nn.Linear(n_sensors, 1)   # combine channels

    def forward(self, x):  # [B, T, C]
        B, T, C = x.shape
        # Reshape: treat each channel as a separate sample
        x = x.permute(0, 2, 1).reshape(B * C, T, 1)  # [B*C, T, 1]
        x = self.input_proj(x) + self.pos_emb[:, :T]
        x = self.encoder(x)
        x = self.channel_fc(x[:, -1])  # [B*C, 1]
        x = x.reshape(B, C)            # [B, C]
        return self.final_fc(x)         # [B, 1]


class RoleTransformerRUL(nn.Module):
    """
    Role-Based Transformer: sensors grouped by physical component.
    - Within each component: shared weights, channel-independent processing
    - Cross-component: attention over component embeddings
    """
    def __init__(self, component_groups, sensor_cols, d_model=32, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.component_groups = component_groups
        self.sensor_cols = sensor_cols
        self.n_components = len(component_groups)

        # Build index mapping: sensor_col_name → index in input tensor
        self.component_indices = {}
        for comp, sensors in component_groups.items():
            self.component_indices[comp] = [sensor_cols.index(s) for s in sensors]

        # Within-component encoder (shared across sensors in a component)
        self.input_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, SEQ_LEN, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.within_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.within_pool = nn.Linear(d_model, d_model)  # pool within-component

        # Cross-component encoder
        cross_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 4, dropout, batch_first=True
        )
        self.cross_encoder = nn.TransformerEncoder(cross_layer, 1)
        self.comp_emb = nn.Parameter(torch.randn(1, self.n_components, d_model) * 0.02)

        # Output head
        self.fc = nn.Linear(d_model * self.n_components, 1)

    def forward(self, x):  # [B, T, C]
        B, T, C = x.shape
        comp_embeddings = []

        for comp, indices in self.component_indices.items():
            n_ch = len(indices)
            # Extract channels for this component
            comp_x = x[:, :, indices]  # [B, T, n_ch]
            # Channel-independent within component
            comp_x = comp_x.permute(0, 2, 1).reshape(B * n_ch, T, 1)  # [B*n_ch, T, 1]
            comp_x = self.input_proj(comp_x) + self.pos_emb[:, :T]
            comp_x = self.within_encoder(comp_x)  # [B*n_ch, T, d]
            comp_x = comp_x[:, -1]  # [B*n_ch, d]
            comp_x = comp_x.reshape(B, n_ch, -1)  # [B, n_ch, d]
            # Pool within component (mean)
            comp_x = self.within_pool(comp_x.mean(dim=1))  # [B, d]
            comp_embeddings.append(comp_x)

        # Stack components: [B, n_comp, d]
        comp_stack = torch.stack(comp_embeddings, dim=1)
        comp_stack = comp_stack + self.comp_emb

        # Cross-component attention
        comp_stack = self.cross_encoder(comp_stack)  # [B, n_comp, d]

        # Flatten and predict
        out = comp_stack.reshape(B, -1)  # [B, n_comp * d]
        return self.fc(out)


# ─── Training ───

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3, patience=10):
    """Train with early stopping."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_val_rmse = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * x.size(0)

        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Validate
        val_rmse = evaluate_model(model, val_loader)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model, best_val_rmse


def evaluate_model(model, loader):
    """Compute RMSE on a dataset."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    rmse = torch.sqrt(F.mse_loss(preds, targets)).item()
    return rmse


def compute_score(model, loader):
    """Compute NASA scoring function (asymmetric, penalizes late predictions more)."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            pred = model(x)
            preds.append(pred.cpu())
            targets.append(y)
    preds = torch.cat(preds).squeeze()
    targets = torch.cat(targets).squeeze()
    d = preds - targets  # positive = predicted too high (late)
    score = 0
    for di in d:
        if di < 0:
            score += torch.exp(-di / 13) - 1
        else:
            score += torch.exp(di / 10) - 1
    return score.item()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Main Experiment ───

def run_baselines(subset="FD001"):
    """Run all 5 baselines on a given C-MAPSS subset."""
    print(f"\n{'='*60}")
    print(f"C-MAPSS {subset} Baselines")
    print(f"{'='*60}")

    train_df, test_df, test_rul_map = load_cmapss(subset)
    sensor_cols = INFORMATIVE_SENSORS
    train_df, test_df = normalize_cmapss(train_df, test_df, sensor_cols)

    n_sensors = len(sensor_cols)
    print(f"Sensors: {n_sensors}, Seq len: {SEQ_LEN}")
    print(f"Train units: {train_df['unit'].nunique()}, "
          f"Train samples: {len(train_df)}")

    results = {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create datasets
        full_dataset = CMAPSSDataset(train_df, sensor_cols)
        test_dataset = CMAPSSTestDataset(test_df, sensor_cols, test_rul_map)

        # 80/20 train/val split
        n_train = int(0.8 * len(full_dataset))
        n_val = len(full_dataset) - n_train
        train_set, val_set = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )

        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=256, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=256, num_workers=0)

        models = {
            "Linear": LinearRUL(n_sensors, SEQ_LEN),
            "LSTM": LSTMRUL(n_sensors),
            "Transformer": TransformerRUL(n_sensors),
            "CI-Transformer": CITransformerRUL(n_sensors),
            "Role-Transformer": RoleTransformerRUL(
                COMPONENT_GROUPS, sensor_cols
            ),
        }

        for name, model in models.items():
            t0 = time.time()
            if name not in results:
                results[name] = {"rmse": [], "score": [], "params": 0}

            params = count_params(model)
            results[name]["params"] = params

            model, val_rmse = train_model(
                model, train_loader, val_loader,
                epochs=80, lr=1e-3, patience=15
            )
            test_rmse = evaluate_model(model, test_loader)
            test_score = compute_score(model, test_loader)
            elapsed = time.time() - t0

            results[name]["rmse"].append(test_rmse)
            results[name]["score"].append(test_score)

            print(f"  [{seed}] {name:20s} | params={params:>7,d} | "
                  f"val_rmse={val_rmse:.2f} | test_rmse={test_rmse:.2f} | "
                  f"score={test_score:.0f} | {elapsed:.0f}s")

    # Summary
    print(f"\n{'─'*60}")
    print(f"{'Model':20s} | {'Params':>8s} | {'Test RMSE':>12s} | {'Score':>12s}")
    print(f"{'─'*60}")
    for name in ["Linear", "LSTM", "Transformer", "CI-Transformer", "Role-Transformer"]:
        r = results[name]
        rmse_arr = np.array(r["rmse"])
        score_arr = np.array(r["score"])
        print(f"{name:20s} | {r['params']:>8,d} | "
              f"{rmse_arr.mean():.2f} ± {rmse_arr.std():.2f} | "
              f"{score_arr.mean():.0f} ± {score_arr.std():.0f}")

    return results


def run_transfer_experiment():
    """Train on FD001, test on FD002. Compare CI vs Role Transformer."""
    print(f"\n{'='*60}")
    print(f"Transfer Experiment: FD001 → FD002")
    print(f"{'='*60}")

    # Load both subsets
    train_df_1, test_df_1, test_rul_1 = load_cmapss("FD001")
    train_df_2, test_df_2, test_rul_2 = load_cmapss("FD002")

    sensor_cols = INFORMATIVE_SENSORS
    n_sensors = len(sensor_cols)

    # Normalize FD002 using FD001 stats (true transfer scenario)
    means = train_df_1[sensor_cols].mean()
    stds = train_df_1[sensor_cols].std().replace(0, 1)

    for df in [train_df_1, test_df_1, train_df_2, test_df_2]:
        df[sensor_cols] = (df[sensor_cols] - means) / stds

    results = {}

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Train on FD001
        train_dataset = CMAPSSDataset(train_df_1, sensor_cols)
        n_train = int(0.8 * len(train_dataset))
        n_val = len(train_dataset) - n_train
        train_set, val_set = torch.utils.data.random_split(
            train_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(seed)
        )
        train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=256, num_workers=0)

        # Test on FD001 (in-domain) and FD002 (transfer)
        test1_dataset = CMAPSSTestDataset(test_df_1, sensor_cols, test_rul_1)
        test2_dataset = CMAPSSTestDataset(test_df_2, sensor_cols, test_rul_2)
        test1_loader = DataLoader(test1_dataset, batch_size=256, num_workers=0)
        test2_loader = DataLoader(test2_dataset, batch_size=256, num_workers=0)

        models = {
            "CI-Transformer": CITransformerRUL(n_sensors),
            "Role-Transformer": RoleTransformerRUL(
                COMPONENT_GROUPS, sensor_cols
            ),
        }

        for name, model in models.items():
            if name not in results:
                results[name] = {"fd001_rmse": [], "fd002_rmse": [],
                                 "fd001_score": [], "fd002_score": []}

            model, val_rmse = train_model(
                model, train_loader, val_loader,
                epochs=80, lr=1e-3, patience=15
            )

            rmse1 = evaluate_model(model, test1_loader)
            rmse2 = evaluate_model(model, test2_loader)
            score1 = compute_score(model, test1_loader)
            score2 = compute_score(model, test2_loader)

            results[name]["fd001_rmse"].append(rmse1)
            results[name]["fd002_rmse"].append(rmse2)
            results[name]["fd001_score"].append(score1)
            results[name]["fd002_score"].append(score2)

            transfer_ratio = rmse2 / rmse1
            print(f"  [{seed}] {name:20s} | FD001={rmse1:.2f} | FD002={rmse2:.2f} | "
                  f"ratio={transfer_ratio:.2f}")

    # Summary
    print(f"\n{'─'*60}")
    print(f"Transfer Summary (FD001 → FD002)")
    print(f"{'─'*60}")
    print(f"{'Model':20s} | {'FD001 RMSE':>12s} | {'FD002 RMSE':>12s} | {'Ratio':>8s}")
    print(f"{'─'*60}")
    for name in ["CI-Transformer", "Role-Transformer"]:
        r = results[name]
        r1 = np.array(r["fd001_rmse"])
        r2 = np.array(r["fd002_rmse"])
        ratio = r2.mean() / r1.mean()
        print(f"{name:20s} | {r1.mean():.2f} ± {r1.std():.2f} | "
              f"{r2.mean():.2f} ± {r2.std():.2f} | {ratio:.2f}")

    return results


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")

    # Phase 2: Baselines on FD001
    baseline_results = run_baselines("FD001")

    # Transfer experiment
    transfer_results = run_transfer_experiment()

    print("\n\nDone! Check EXPERIMENT_LOG.md for results.")

#!/usr/bin/env python3
"""
Tier 3 Extension: Multi-horizon weather forecasting
Tests H=96, H=336, H=720 to see if physics grouping helps more at longer horizons.
Only runs the key models: CI-Trans, Full-Attn, Physics-Grouped, Role-Trans.
"""
import sys, time, copy, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/sagemaker-user/IndustrialJEPA/data/weather/jena_climate_2009_2016.csv"

SUBSAMPLE = 6  # 10min -> hourly
COL_NAMES = ["p","T","Tpot","Tdew","rh","VPmax","VPact","VPdef","sh","H2OC","rho","wv","max_wv","wd"]
N_CH = len(COL_NAMES)

PHYSICS_GROUPS = {
    "temperature": [1, 2, 3],
    "pressure": [0, 5, 6, 7],
    "humidity": [4, 8, 9, 10],
    "wind": [11, 12, 13],
}

SEEDS = [42, 123, 456]
HORIZONS = [96, 336, 720]
LOOKBACK = 96


def load_weather():
    print("Loading weather data...")
    dates, data = [], []
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i % SUBSAMPLE != 0: continue
            try:
                dates.append(row[0])
                data.append([float(v) for v in row[1:]])
            except (ValueError, IndexError):
                continue
    data = np.array(data)
    for col in [11, 12]:
        bad = data[:, col] < -999
        if bad.any():
            col_good = data[~bad, col]
            data[bad, col] = col_good.mean()
    print(f"  {len(data)} hourly samples, {N_CH} channels")

    train_idx, test_idx = [], []
    for i, d in enumerate(dates):
        year = int(d.split('.')[2].split()[0])
        if year <= 2015: train_idx.append(i)
        else: test_idx.append(i)
    print(f"  Train(2009-2015): {len(train_idx)}, Test(2016): {len(test_idx)}")
    return data[train_idx], data[test_idx]


class WeatherDS(Dataset):
    def __init__(self, data, lookback, horizon, mean, std, stride=1, max_samples=15000):
        self.mean, self.std = mean, std
        normed = (data - mean) / std
        self.samples = []
        for i in range(0, len(normed) - lookback - horizon + 1, stride):
            self.samples.append((normed[i:i+lookback], normed[i+lookback:i+lookback+horizon]))
            if len(self.samples) >= max_samples: break
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ─── Models (parameterized by horizon) ───

class CITrans(nn.Module):
    def __init__(self, lookback, horizon, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, lookback, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.out = nn.Linear(d, horizon)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.enc(x)[:, -1]
        return self.out(x).reshape(B, C, -1).permute(0, 2, 1)

class FullAttn(nn.Module):
    def __init__(self, lookback, horizon, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, lookback, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.te = nn.TransformerEncoder(el, nl)
        self.ce = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.se = nn.TransformerEncoder(cl, 1)
        self.out = nn.Linear(d, horizon)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.te(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ce; x = self.se(x)
        return self.out(x).permute(0, 2, 1)

class RoleTransWeather(nn.Module):
    def __init__(self, lookback, horizon, groups=PHYSICS_GROUPS, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d, self.groups = d, groups
        self.nc = len(groups)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, lookback, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.within = nn.TransformerEncoder(el, nl)
        self.pool = nn.Linear(d, d)
        self.cemb = nn.Parameter(torch.randn(1, self.nc, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.cross = nn.TransformerEncoder(cl, 1)
        self.group_dec = nn.ModuleList([
            nn.Linear(d, horizon * len(idx)) for idx in groups.values()
        ])
        self.horizon = horizon

    def forward(self, x):
        B, T, C = x.shape; cs = []
        for gn, idx in self.groups.items():
            ns = len(idx)
            cx = x[:,:,idx].permute(0,2,1).reshape(B*ns, T, 1)
            cx = self.proj(cx) + self.pos[:,:T]
            cx = self.within(cx)[:, -1].reshape(B, ns, -1)
            cs.append(self.pool(cx.mean(1)))
        s = torch.stack(cs, 1) + self.cemb
        s = self.cross(s)
        out = [None] * C
        for i, (gn, idx) in enumerate(self.groups.items()):
            dec = self.group_dec[i](s[:, i]).reshape(B, self.horizon, len(idx))
            for j, ci in enumerate(idx):
                out[ci] = dec[:, :, j]
        return torch.stack(out, dim=2)


def train_m(model, tl, vl, epochs=30, lr=1e-3, patience=8):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv, bs, ni = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        v = evaluate(model, vl)
        if v < bv:
            bv = v; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    model.load_state_dict(bs); model.to(DEVICE)
    return model, bv

def evaluate(model, loader):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return F.mse_loss(torch.cat(ps), torch.cat(ts)).item()


def run_multi_horizon():
    print("="*70)
    print("MULTI-HORIZON WEATHER FORECASTING")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70)
    sys.stdout.flush()

    train_data, test_data = load_weather()
    mn, sd = train_data.mean(0), np.maximum(train_data.std(0), 1e-8)

    all_results = {}

    for H in HORIZONS:
        print(f"\n{'='*50}")
        print(f"HORIZON = {H} (lookback={LOOKBACK})")
        print(f"{'='*50}")
        sys.stdout.flush()

        train_ds = WeatherDS(train_data, LOOKBACK, H, mn, sd, stride=6, max_samples=12000)
        test_ds = WeatherDS(test_data, LOOKBACK, H, mn, sd, stride=3, max_samples=4000)
        print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

        test_l = DataLoader(test_ds, 256)
        horizon_results = {}

        models = {
            "ci_trans": lambda: CITrans(LOOKBACK, H),
            "full_attn": lambda: FullAttn(LOOKBACK, H),
            "role_trans": lambda: RoleTransWeather(LOOKBACK, H),
        }

        for mname, fn in models.items():
            print(f"\n  --- {mname} ---")
            test_scores = []
            for seed in SEEDS:
                t0 = time.time()
                torch.manual_seed(seed); np.random.seed(seed)
                m = fn()
                n = len(train_ds); nt = int(0.8*n); nv = n - nt
                trn, val = torch.utils.data.random_split(
                    train_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
                tl = DataLoader(trn, 128, shuffle=True)
                vl = DataLoader(val, 256)
                m, _ = train_m(m, tl, vl)
                te = evaluate(m, test_l); test_scores.append(te)
                print(f"    [{seed}] test={te:.6f} ({time.time()-t0:.0f}s)")
                sys.stdout.flush()
            arr = np.array(test_scores)
            horizon_results[mname] = arr
            print(f"    AVG: {arr.mean():.6f}±{arr.std():.6f}")

        all_results[H] = horizon_results

    # Summary
    print("\n" + "="*70)
    print("MULTI-HORIZON SUMMARY")
    print("="*70)
    print(f"{'Horizon':<10} {'CI-Trans':>15} {'Full-Attn':>15} {'Role-Trans':>15} {'RT vs CI':>10}")
    print("-"*70)
    for H in HORIZONS:
        r = all_results[H]
        ci = r['ci_trans'].mean()
        fa = r['full_attn'].mean()
        rt = r['role_trans'].mean()
        delta = (1 - rt/ci) * 100
        print(f"H={H:<7} {ci:>15.6f} {fa:>15.6f} {rt:>15.6f} {delta:>9.1f}%")

    print(f"\nMulti-horizon complete! Time: {time.strftime('%H:%M')}")
    return all_results


if __name__ == "__main__":
    run_multi_horizon()

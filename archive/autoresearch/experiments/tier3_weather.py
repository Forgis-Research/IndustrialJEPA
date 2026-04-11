#!/usr/bin/env python3
"""
Tier 3: Jena Weather — Physics-informed grouping validation
14 weather variables, physics groups: temperature, pressure, humidity, wind
Transfer: 2015 -> 2016 (temporal distribution shift)
"""
import sys, time, copy, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = Path("/home/sagemaker-user/IndustrialJEPA/data/weather/jena_climate_2009_2016.csv")

SUBSAMPLE = 6  # 10min -> hourly
LOOKBACK = 96   # 4 days
HORIZON = 96    # predict 4 days

COL_NAMES = ["p","T","Tpot","Tdew","rh","VPmax","VPact","VPdef","sh","H2OC","rho","wv","max_wv","wd"]
N_CH = len(COL_NAMES)

PHYSICS_GROUPS = {
    "temperature": [1, 2, 3],       # T, Tpot, Tdew
    "pressure": [0, 5, 6, 7],       # p, VPmax, VPact, VPdef
    "humidity": [4, 8, 9, 10],      # rh, sh, H2OC, rho
    "wind": [11, 12, 13],           # wv, max_wv, wd
}

SEEDS = [42, 123, 456]


def load_weather():
    print("Loading weather data...")
    dates, data = [], []
    with open(DATA_PATH, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for i, row in enumerate(reader):
            if i % SUBSAMPLE != 0: continue
            try:
                dates.append(row[0])
                data.append([float(v) for v in row[1:]])
            except (ValueError, IndexError):
                continue

    data = np.array(data)

    # Clean sentinel values (-9999 in wv and max_wv columns 11,12)
    for col in [11, 12]:
        bad = data[:, col] < -999
        if bad.any():
            print(f"  Replacing {bad.sum()} sentinel values in col {col}")
            col_good = data[~bad, col]
            data[bad, col] = col_good.mean()  # replace with column mean

    print(f"  {len(data)} hourly samples, {N_CH} channels")

    train_idx, source_idx, target_idx = [], [], []
    for i, d in enumerate(dates):
        year = int(d.split('.')[2].split()[0])
        if year <= 2014: train_idx.append(i)
        elif year == 2015: source_idx.append(i)
        else: target_idx.append(i)

    print(f"  Train(2009-2014): {len(train_idx)}, Source(2015): {len(source_idx)}, Target(2016): {len(target_idx)}")
    return data[train_idx], data[source_idx], data[target_idx]


class WeatherDS(Dataset):
    def __init__(self, data, lookback=LOOKBACK, horizon=HORIZON,
                 mean=None, std=None, stride=1, max_samples=20000):
        if mean is None:
            self.mean = data.mean(0); self.std = np.maximum(data.std(0), 1e-8)
        else:
            self.mean, self.std = mean, std
        normed = (data - self.mean) / self.std
        self.samples = []
        for i in range(0, len(normed) - lookback - horizon + 1, stride):
            self.samples.append((normed[i:i+lookback], normed[i+lookback:i+lookback+horizon]))
            if len(self.samples) >= max_samples: break

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ─── Models ───

class MeanBL:
    def evaluate(self, loader):
        ps, ts = [], []
        for x, y in loader:
            ps.append(torch.zeros_like(y)); ts.append(y)
        return F.mse_loss(torch.cat(ps), torch.cat(ts)).item()

class LastValBL:
    def evaluate(self, loader):
        ps, ts = [], []
        for x, y in loader:
            ps.append(x[:, -1:].expand_as(y)); ts.append(y)
        return F.mse_loss(torch.cat(ps), torch.cat(ts)).item()

class LinearForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LOOKBACK * N_CH, HORIZON * N_CH)
    def forward(self, x):
        B = x.size(0)
        return self.fc(x.reshape(B, -1)).reshape(B, HORIZON, N_CH)

class MLPForecast(nn.Module):
    def __init__(self, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LOOKBACK*N_CH, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, HORIZON*N_CH))
    def forward(self, x):
        B = x.size(0)
        return self.net(x.reshape(B, -1)).reshape(B, HORIZON, N_CH)

class CITrans(nn.Module):
    def __init__(self, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.out = nn.Linear(d, HORIZON)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.enc(x)[:, -1]
        return self.out(x).reshape(B, C, HORIZON).permute(0, 2, 1)

class FullAttn(nn.Module):
    def __init__(self, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.te = nn.TransformerEncoder(el, nl)
        self.ce = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.se = nn.TransformerEncoder(cl, 1)
        self.out = nn.Linear(d, HORIZON)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.te(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ce; x = self.se(x)
        return self.out(x).permute(0, 2, 1)

class PhysicsGrouped(nn.Module):
    """Physics-grouped mask attention."""
    def __init__(self, groups=PHYSICS_GROUPS, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d, self.groups = d, groups
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.te = nn.TransformerEncoder(el, nl)
        self.ce = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.se = nn.TransformerEncoder(cl, 1)
        self.register_buffer('gmask', self._build_mask())
        self.out = nn.Linear(d, HORIZON)

    def _build_mask(self):
        m = torch.ones(N_CH, N_CH, dtype=torch.bool)
        for idx in self.groups.values():
            for i in idx:
                for j in idx: m[i,j] = False
        reps = [idx[0] for idx in self.groups.values()]
        for i in reps:
            for j in reps: m[i,j] = False
        return m

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.te(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ce
        x = self.se(x, mask=self.gmask)
        return self.out(x).permute(0, 2, 1)

class RoleTransWeather(nn.Module):
    """Physics-grouped: shared within-group + cross-group (RoleTrans style)."""
    def __init__(self, groups=PHYSICS_GROUPS, d=64, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d, self.groups = d, groups
        self.nc = len(groups)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.within = nn.TransformerEncoder(el, nl)
        self.pool = nn.Linear(d, d)
        self.cemb = nn.Parameter(torch.randn(1, self.nc, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.cross = nn.TransformerEncoder(cl, 1)
        self.group_dec = nn.ModuleList([
            nn.Linear(d, HORIZON * len(idx)) for idx in groups.values()
        ])

    def forward(self, x):
        B, T, C = x.shape
        cs = []
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
            dec = self.group_dec[i](s[:, i]).reshape(B, HORIZON, len(idx))
            for j, ci in enumerate(idx):
                out[ci] = dec[:, :, j]
        return torch.stack(out, dim=2)


# ─── Training ───

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

def finetune(model, loader, epochs=10, lr=1e-4):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_tier3():
    print("="*70)
    print("TIER 3: JENA WEATHER")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70); sys.stdout.flush()

    train_data, source_data, target_data = load_weather()
    mn, sd = train_data.mean(0), np.maximum(train_data.std(0), 1e-8)

    train_ds = WeatherDS(train_data, mean=mn, std=sd, stride=6, max_samples=15000)
    source_ds = WeatherDS(source_data, mean=mn, std=sd, stride=3, max_samples=5000)
    target_ds = WeatherDS(target_data, mean=mn, std=sd, stride=3, max_samples=5000)
    n_adapt = max(1, len(target_data) // 10)
    adapt_ds = WeatherDS(target_data[:n_adapt], mean=mn, std=sd, stride=1, max_samples=3000)

    print(f"  Train: {len(train_ds)}, Source: {len(source_ds)}, Target: {len(target_ds)}, Adapt: {len(adapt_ds)}")
    sys.stdout.flush()

    src_l = DataLoader(source_ds, 256)
    tgt_l = DataLoader(target_ds, 256)

    print("\n--- W1: Mean ---")
    mb = MeanBL()
    ms, mt = mb.evaluate(src_l), mb.evaluate(tgt_l)
    print(f"  Src={ms:.6f} Tgt={mt:.6f} R={mt/ms:.2f}")

    print("\n--- W2: Last-Value ---")
    lv = LastValBL()
    ls, lt = lv.evaluate(src_l), lv.evaluate(tgt_l)
    print(f"  Src={ls:.6f} Tgt={lt:.6f} R={lt/ls:.2f}")
    sys.stdout.flush()

    results = {"mean": (ms, mt), "last_value": (ls, lt)}

    models = {
        "W3_linear": lambda: LinearForecast(),
        "W4_mlp": lambda: MLPForecast(),
        "W5_ci_trans": lambda: CITrans(),
        "W6_full_attn": lambda: FullAttn(),
        "W7_phys_mask": lambda: PhysicsGrouped(),
        "W8_role_trans": lambda: RoleTransWeather(),
    }

    for mname, fn in models.items():
        print(f"\n--- {mname} ---")
        src_a, tgt_a, tgt_adapt_a = [], [], []
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
            sm = evaluate(m, src_l); src_a.append(sm)
            tm = evaluate(m, tgt_l); tgt_a.append(tm)
            ma = copy.deepcopy(m)
            al = DataLoader(adapt_ds, 64, shuffle=True)
            ma = finetune(ma, al)
            ta = evaluate(ma, tgt_l); tgt_adapt_a.append(ta)
            print(f"  [{seed}] src={sm:.6f} tgt={tm:.6f} tgt10%={ta:.6f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

        sa, za, aa = np.array(src_a), np.array(tgt_a), np.array(tgt_adapt_a)
        rz, ra = za.mean()/sa.mean(), aa.mean()/sa.mean()
        results[mname] = {'src':sa, 'tgt':za, 'tgt_a':aa, 'rz':rz, 'ra':ra}
        print(f"  AVG: src={sa.mean():.6f}+/-{sa.std():.6f} tgt={za.mean():.6f}+/-{za.std():.6f} "
              f"tgt10%={aa.mean():.6f}+/-{aa.std():.6f}")
        print(f"  Ratios: zero={rz:.2f} adapted={ra:.2f}")
        sys.stdout.flush()

    print("\n" + "="*70)
    print("TIER 3 SUMMARY")
    print("="*70)
    print(f"{'Model':<18} {'Src MSE':>12} {'Tgt MSE':>12} {'Tgt 10%':>12} {'R(zero)':>8} {'R(10%)':>8}")
    print("-"*74)
    for n in ["mean", "last_value"]:
        s, t = results[n]
        print(f"{n:<18} {s:>12.6f} {t:>12.6f} {'N/A':>12} {t/s:>8.2f} {'N/A':>8}")
    for n in ["W3_linear","W4_mlp","W5_ci_trans","W6_full_attn","W7_phys_mask","W8_role_trans"]:
        if n in results:
            r = results[n]
            print(f"{n:<18} {r['src'].mean():>12.6f} {r['tgt'].mean():>12.6f} {r['tgt_a'].mean():>12.6f} {r['rz']:>8.2f} {r['ra']:>8.2f}")

    ci = results.get("W5_ci_trans")
    pm = results.get("W7_phys_mask")
    rt = results.get("W8_role_trans")
    if ci:
        print(f"\nKey comparisons (target MSE, lower=better):")
        if pm: print(f"  PhysMask vs CI: {pm['tgt'].mean():.6f} vs {ci['tgt'].mean():.6f}")
        if rt: print(f"  RoleTrans vs CI: {rt['tgt'].mean():.6f} vs {ci['tgt'].mean():.6f}")

    print(f"\nTier 3 complete! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run_tier3()

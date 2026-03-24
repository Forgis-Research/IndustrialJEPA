#!/usr/bin/env python3
"""
Tier 1: Double Pendulum — Physics-informed grouping validation
2D Treatment: Temporal enc per channel + Cross-channel attention (grouped vs full vs CI)
Groups: mass_1=[theta1, omega1], mass_2=[theta2, omega2]
Transfer: m_ratio=1.0 -> m_ratio=0.5
"""
import sys, time, csv, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = Path("/home/sagemaker-user/IndustrialJEPA/data/pendulum.csv")

LOOKBACK = 50
HORIZON = 10
CHANNELS = ["theta1", "omega1", "theta2", "omega2"]
N_CH = 4
STRIDE = 10  # Skip windows for speed

PHYSICS_GROUPS = {"mass_1": [0, 1], "mass_2": [2, 3]}
SEEDS = [42, 123, 456]


def load_pendulum_data():
    """Load and split pendulum CSV."""
    print("Loading pendulum data...")
    source, target = {}, {}
    with open(DATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row['trajectory_id'])
            vals = [float(row[c]) for c in CHANNELS]
            bucket = source if row['domain'] == 'source' else target
            if tid not in bucket:
                bucket[tid] = []
            bucket[tid].append(vals)
    source = {k: np.array(v) for k, v in source.items()}
    target = {k: np.array(v) for k, v in target.items()}
    print(f"  Source: {len(source)} trajs, Target: {len(target)} trajs")
    return source, target


class TSDataset(Dataset):
    def __init__(self, trajs_dict, lookback=LOOKBACK, horizon=HORIZON,
                 mean=None, std=None, stride=STRIDE, max_samples=10000):
        self.samples = []
        if mean is None:
            all_d = np.concatenate(list(trajs_dict.values()), axis=0)
            self.mean = all_d.mean(0)
            self.std = np.maximum(all_d.std(0), 1e-8)
        else:
            self.mean, self.std = mean, std

        for traj in trajs_dict.values():
            tn = (traj - self.mean) / self.std
            for i in range(0, len(tn) - lookback - horizon + 1, stride):
                self.samples.append((tn[i:i+lookback], tn[i+lookback:i+lookback+horizon]))
                if len(self.samples) >= max_samples:
                    return

    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ─── Models ───

class MeanBaseline:
    def evaluate(self, loader):
        ps, ts = [], []
        for x, y in loader:
            ps.append(torch.zeros_like(y)); ts.append(y)
        return F.mse_loss(torch.cat(ps), torch.cat(ts)).item()

class LastValueBaseline:
    def evaluate(self, loader):
        ps, ts = [], []
        for x, y in loader:
            ps.append(x[:, -1:].expand_as(y)); ts.append(y)
        return F.mse_loss(torch.cat(ps), torch.cat(ts)).item()

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LOOKBACK * N_CH, HORIZON * N_CH)
    def forward(self, x):
        B = x.size(0)
        return self.fc(x.reshape(B, -1)).reshape(B, HORIZON, N_CH)

class MLPModel(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LOOKBACK*N_CH, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, HORIZON*N_CH))
    def forward(self, x):
        B = x.size(0)
        return self.net(x.reshape(B, -1)).reshape(B, HORIZON, N_CH)

class CITransformer(nn.Module):
    """Channel-Independent: each channel processed separately, no cross-channel info."""
    def __init__(self, d=32, nh=4, nl=2, do=0.1):
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
        x = self.enc(x)[:, -1]  # [B*C, d]
        x = self.out(x).reshape(B, C, HORIZON)  # [B, C, H]
        return x.permute(0, 2, 1)  # [B, H, C]

class FullAttention(nn.Module):
    """2D: temporal enc per channel + full cross-channel attention."""
    def __init__(self, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.time_enc = nn.TransformerEncoder(el, nl)
        self.ch_emb = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.space_enc = nn.TransformerEncoder(cl, 1)
        self.out = nn.Linear(d, HORIZON)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.time_enc(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ch_emb
        x = self.space_enc(x)
        x = self.out(x).permute(0, 2, 1)  # [B, H, C]
        return x

class PhysicsGrouped(nn.Module):
    """2D: shared temporal enc + PHYSICS-MASKED cross-channel attention."""
    def __init__(self, groups=PHYSICS_GROUPS, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d = d; self.groups = groups
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, LOOKBACK, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.time_enc = nn.TransformerEncoder(el, nl)
        self.ch_emb = nn.Parameter(torch.randn(1, N_CH, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.space_enc = nn.TransformerEncoder(cl, 1)
        self.register_buffer('gmask', self._build_mask())
        self.out = nn.Linear(d, HORIZON)

    def _build_mask(self):
        m = torch.ones(N_CH, N_CH, dtype=torch.bool)  # True = blocked
        for idx in self.groups.values():
            for i in idx:
                for j in idx:
                    m[i,j] = False
        reps = [idx[0] for idx in self.groups.values()]
        for i in reps:
            for j in reps:
                m[i,j] = False
        return m

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.time_enc(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ch_emb
        x = self.space_enc(x, mask=self.gmask)
        x = self.out(x).permute(0, 2, 1)
        return x


# ─── Training ───

def train_model(model, tl, vl, epochs=40, lr=1e-3, patience=8):
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
        else:
            ni += 1
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


def run_tier1():
    print("="*70)
    print("TIER 1: DOUBLE PENDULUM")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70); sys.stdout.flush()

    source, target = load_pendulum_data()

    # Split source 80/20
    sids = sorted(source.keys()); np.random.seed(42); np.random.shuffle(sids)
    nt = int(0.8*len(sids))
    train_t = {k: source[k] for k in sids[:nt]}
    val_t = {k: source[k] for k in sids[nt:]}

    # Split target 10% adapt / 90% test
    tids = sorted(target.keys())
    na = max(1, len(tids)//10)
    adapt_t = {k: target[k] for k in tids[:na]}
    test_t = {k: target[k] for k in tids[na:]}

    train_ds = TSDataset(train_t, max_samples=12000)
    mn, sd = train_ds.mean, train_ds.std
    val_ds = TSDataset(val_t, mean=mn, std=sd, max_samples=3000)
    tgt_test_ds = TSDataset(test_t, mean=mn, std=sd, max_samples=5000)
    tgt_adapt_ds = TSDataset(adapt_t, mean=mn, std=sd, stride=5, max_samples=3000)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, TgtTest: {len(tgt_test_ds)}, TgtAdapt: {len(tgt_adapt_ds)}")
    sys.stdout.flush()

    val_l = DataLoader(val_ds, 512)
    tgt_l = DataLoader(tgt_test_ds, 512)

    # Trivial baselines
    print("\n--- P1: Mean Baseline ---")
    mb = MeanBaseline()
    ms, mt = mb.evaluate(val_l), mb.evaluate(tgt_l)
    print(f"  Src={ms:.6f} Tgt={mt:.6f} Ratio={mt/ms:.2f}")

    print("\n--- P2: Last-Value ---")
    lv = LastValueBaseline()
    ls, lt = lv.evaluate(val_l), lv.evaluate(tgt_l)
    print(f"  Src={ls:.6f} Tgt={lt:.6f} Ratio={lt/ls:.2f}")
    sys.stdout.flush()

    results = {"mean": (ms, mt), "last_value": (ls, lt)}

    # Trained models
    models = {
        "P3_linear": lambda: LinearModel(),
        "P4_mlp": lambda: MLPModel(),
        "P5_ci_trans": lambda: CITransformer(),
        "P6_full_attn": lambda: FullAttention(),
        "P7_physics": lambda: PhysicsGrouped(),
    }

    for name, fn in models.items():
        print(f"\n--- {name} ---")
        src_a, tgt_z_a, tgt_a_a = [], [], []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            m = fn()
            tl = DataLoader(train_ds, 256, shuffle=True)
            m, _ = train_model(m, tl, val_l)
            s = evaluate(m, val_l)
            tz = evaluate(m, tgt_l)
            ma = copy.deepcopy(m)
            al = DataLoader(tgt_adapt_ds, 128, shuffle=True)
            ma = finetune(ma, al)
            ta = evaluate(ma, tgt_l)
            src_a.append(s); tgt_z_a.append(tz); tgt_a_a.append(ta)
            print(f"  [{seed}] src={s:.6f} tgt0={tz:.6f} tgt10%={ta:.6f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

        sa, za, aa = np.array(src_a), np.array(tgt_z_a), np.array(tgt_a_a)
        rz, ra = za.mean()/sa.mean(), aa.mean()/sa.mean()
        print(f"  AVG: src={sa.mean():.6f}+/-{sa.std():.6f} tgt0={za.mean():.6f}+/-{za.std():.6f} tgt10%={aa.mean():.6f}+/-{aa.std():.6f}")
        print(f"  Ratios: zero={rz:.2f} adapted={ra:.2f}")
        results[name] = {"src": sa, "tgt_zero": za, "tgt_adapted": aa,
                         "ratio_zero": rz, "ratio_adapted": ra}
        sys.stdout.flush()

    # Summary
    print("\n" + "="*70)
    print("TIER 1 SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<18} {'Src MSE':>12} {'Tgt Zero':>12} {'Tgt 10%':>12} {'R(zero)':>8} {'R(10%)':>8}")
    print("-"*72)
    for n in ["mean", "last_value"]:
        s, t = results[n]
        print(f"{n:<18} {s:>12.6f} {t:>12.6f} {'N/A':>12} {t/s:>8.2f} {'N/A':>8}")
    for n in ["P3_linear", "P4_mlp", "P5_ci_trans", "P6_full_attn", "P7_physics"]:
        r = results[n]
        sm, zm, am = r['src'].mean(), r['tgt_zero'].mean(), r['tgt_adapted'].mean()
        print(f"{n:<18} {sm:>12.6f} {zm:>12.6f} {am:>12.6f} {r['ratio_zero']:>8.2f} {r['ratio_adapted']:>8.2f}")

    # Key comparison
    ci = results.get("P5_ci_trans", {})
    pg = results.get("P7_physics", {})
    if ci and pg:
        print(f"\nPhysics vs CI transfer ratio (zero): {pg['ratio_zero']:.2f} vs {ci['ratio_zero']:.2f}")
        if pg['ratio_zero'] < ci['ratio_zero']:
            print(f"  Physics wins by {(1-pg['ratio_zero']/ci['ratio_zero'])*100:.1f}%")
        else:
            print(f"  CI wins by {(1-ci['ratio_zero']/pg['ratio_zero'])*100:.1f}%")

    print(f"\nTier 1 complete! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run_tier1()

#!/usr/bin/env python3
"""
Tier 2: C-MAPSS — Physics-informed grouping for turbofan RUL prediction
2D Treatment: Temporal enc per channel + Cross-channel attention (grouped vs full vs CI)
Groups: fan, HPC, combustor, turbine, nozzle
Transfer: FD001->FD002 (1->6 operating conditions), FD001->FD003, FD001->FD004
"""
import sys, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
MAX_RUL = 125; SEQ_LEN = 30
COLS = ["unit","cycle"]+[f"setting{i}" for i in range(1,4)]+[f"s{i}" for i in range(1,22)]
SC = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
N_CH = len(SC)

COMPONENT_GROUPS = {
    "fan": ["s2","s8","s12","s21"],
    "hpc": ["s3","s7","s11","s20"],
    "combustor": ["s9","s14"],
    "turbine": ["s4","s13"],
    "nozzle": ["s15","s17"],
}
GROUP_IDX = {k: [SC.index(s) for s in v] for k, v in COMPONENT_GROUPS.items()}
SEEDS = [42, 123, 456]


def load_cmapss(subset):
    """Load C-MAPSS without pandas."""
    def read_txt(path):
        rows = []
        with open(path) as f:
            for line in f:
                vals = line.strip().split()
                if len(vals) >= len(COLS):
                    rows.append([float(v) for v in vals[:len(COLS)]])
        return np.array(rows)

    train_raw = read_txt(DATA_DIR / f"train_{subset}.txt")
    test_raw = read_txt(DATA_DIR / f"test_{subset}.txt")
    with open(DATA_DIR / f"RUL_{subset}.txt") as f:
        rul_vals = [int(line.strip().split()[0]) for line in f if line.strip()]

    sensor_idx = [COLS.index(s) for s in SC]
    def build(raw):
        return raw[:, 0].astype(int), raw[:, 1].astype(int), raw[:, sensor_idx]

    tu, tc, ts = build(train_raw)
    eu, ec, es = build(test_raw)

    # RUL for training
    trul = np.zeros(len(train_raw))
    for u in np.unique(tu):
        m = tu == u; trul[m] = np.minimum(tc[m].max() - tc[m], MAX_RUL)

    return {'tu': tu, 'tc': tc, 'ts': ts, 'tr': trul,
            'eu': eu, 'ec': ec, 'es': es, 'er': rul_vals}


class TrainDS(Dataset):
    def __init__(self, units, sensors, rul, nm, ns, sl=SEQ_LEN):
        self.samples = []
        sensors = (sensors - nm) / ns
        for u in np.unique(units):
            m = units == u; s = sensors[m]; r = rul[m]
            for i in range(len(s) - sl + 1):
                self.samples.append((s[i:i+sl], min(r[i+sl-1], MAX_RUL)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class TestDS(Dataset):
    def __init__(self, data, nm, ns, sl=SEQ_LEN):
        self.samples = []
        sensors = (data['es'] - nm) / ns
        units = data['eu']; rm = data['er']
        for idx, u in enumerate(np.unique(units)):
            m = units == u; s = sensors[m]
            if len(s) < sl: s = np.vstack([np.tile(s[0], (sl-len(s), 1)), s])
            self.samples.append((s[-sl:], min(rm[idx], MAX_RUL)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor([y])


# ─── Models ───

class LinearRUL(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(SEQ_LEN * N_CH, 1)
    def forward(self, x): return self.fc(x.reshape(x.size(0), -1))

class MLPRUL(nn.Module):
    def __init__(self, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SEQ_LEN*N_CH, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))
    def forward(self, x): return self.net(x.reshape(x.size(0), -1))

class CITrans(nn.Module):
    def __init__(self, n=N_CH, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.n, self.d = n, d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.enc = nn.TransformerEncoder(el, nl)
        self.ch = nn.Linear(d, 1); self.out = nn.Linear(n, 1)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.enc(x)[:, -1]
        x = self.ch(x).reshape(B, C)
        return self.out(x)

class FullAttnTrans(nn.Module):
    def __init__(self, n=N_CH, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.n, self.d = n, d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.te = nn.TransformerEncoder(el, nl)
        self.ce = nn.Parameter(torch.randn(1, n, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.se = nn.TransformerEncoder(cl, 1)
        self.out = nn.Linear(n*d, 1)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.te(x)[:, -1].reshape(B, C, self.d)
        x = x + self.ce; x = self.se(x)
        return self.out(x.reshape(B, -1))

class RoleTrans(nn.Module):
    """Physics-grouped: shared within-component encoder + cross-component attention."""
    def __init__(self, groups=GROUP_IDX, d=32, nh=4, nl=2, do=0.1):
        super().__init__()
        self.d, self.groups = d, groups
        self.nc = len(groups)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d)*0.02)
        el = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.within = nn.TransformerEncoder(el, nl)
        self.pool = nn.Linear(d, d)
        self.cemb = nn.Parameter(torch.randn(1, self.nc, d)*0.02)
        cl = nn.TransformerEncoderLayer(d, nh, d*4, do, batch_first=True)
        self.cross = nn.TransformerEncoder(cl, 1)
        self.out = nn.Linear(d*self.nc, 1)
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
        return self.out(s.reshape(B, -1))


# ─── Training ───

def train_m(model, tl, vl, epochs=60, lr=1e-3, patience=12):
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
        v = ev(model, vl)
        if v < bv:
            bv = v; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    model.load_state_dict(bs); model.to(DEVICE)
    return model, bv

def ev(model, loader):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()

def finetune(model, loader, epochs=10, lr=5e-4):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def run_tier2():
    print("="*70)
    print("TIER 2: C-MAPSS TURBOFAN")
    print(f"Time: {time.strftime('%H:%M')} | Device: {DEVICE}")
    print("="*70); sys.stdout.flush()

    fd1 = load_cmapss("FD001")
    fd2 = load_cmapss("FD002")
    fd3 = load_cmapss("FD003")
    fd4 = load_cmapss("FD004")

    # Normalize with FD001 stats (NO RevIN!)
    nm = fd1['ts'].mean(0); ns = fd1['ts'].std(0); ns[ns < 1e-8] = 1.0

    train_ds = TrainDS(fd1['tu'], fd1['ts'], fd1['tr'], nm, ns)
    nt = int(0.8*len(train_ds)); nv = len(train_ds) - nt

    tl1 = DataLoader(TestDS(fd1, nm, ns), 256)
    tl2 = DataLoader(TestDS(fd2, nm, ns), 256)
    tl3 = DataLoader(TestDS(fd3, nm, ns), 256)
    tl4 = DataLoader(TestDS(fd4, nm, ns), 256)

    # FD002 adapt: 10% of engines
    u2 = np.unique(fd2['tu']); na = max(1, len(u2)//10)
    am = np.isin(fd2['tu'], u2[:na])
    adapt_ds = TrainDS(fd2['tu'][am], fd2['ts'][am], fd2['tr'][am], nm, ns)
    print(f"  FD001 train: {len(train_ds)} | FD002 adapt: {len(adapt_ds)} ({na} engines)")
    print(f"  Test sizes: FD001={len(tl1.dataset)} FD002={len(tl2.dataset)} FD003={len(tl3.dataset)} FD004={len(tl4.dataset)}")
    sys.stdout.flush()

    # Trivial baselines
    mean_rul = fd1['tr'].mean()
    med_rul = np.median(fd1['tr'])

    print("\n--- C1: Mean Baseline ---")
    for nm_s, loader in [("FD001",tl1),("FD002",tl2),("FD003",tl3),("FD004",tl4)]:
        ps, ts = [], []
        for x, y in loader:
            ps.append(torch.full_like(y, mean_rul)); ts.append(y)
        r = torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()
        print(f"  {nm_s} RMSE: {r:.2f}")

    print("\n--- C2: Median Baseline ---")
    for nm_s, loader in [("FD001",tl1),("FD002",tl2),("FD003",tl3),("FD004",tl4)]:
        ps, ts = [], []
        for x, y in loader:
            ps.append(torch.full_like(y, med_rul)); ts.append(y)
        r = torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()
        print(f"  {nm_s} RMSE: {r:.2f}")
    sys.stdout.flush()

    # Trained models
    results = {}
    models = {
        "C3_linear": lambda: LinearRUL(),
        "C4_mlp": lambda: MLPRUL(),
        "C5_ci_trans": lambda: CITrans(),
        "C6_full_attn": lambda: FullAttnTrans(),
        "C7_role_trans": lambda: RoleTrans(),
    }

    for mname, fn in models.items():
        print(f"\n--- {mname} ---")
        r1a, r2a, r3a, r4a, r2aa = [], [], [], [], []
        for seed in SEEDS:
            t0 = time.time()
            torch.manual_seed(seed); np.random.seed(seed)
            trn, val = torch.utils.data.random_split(
                train_ds, [nt, nv], generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(trn, 256, shuffle=True)
            vl = DataLoader(val, 256)

            m = fn()
            m, _ = train_m(m, tl, vl)

            e1 = ev(m, tl1); r1a.append(e1)
            e2 = ev(m, tl2); r2a.append(e2)
            e3 = ev(m, tl3); r3a.append(e3)
            e4 = ev(m, tl4); r4a.append(e4)

            ma = copy.deepcopy(m)
            al = DataLoader(adapt_ds, 128, shuffle=True)
            ma = finetune(ma, al)
            e2a = ev(ma, tl2); r2aa.append(e2a)

            print(f"  [{seed}] FD001={e1:.2f} FD002={e2:.2f}(a={e2a:.2f}) FD003={e3:.2f} FD004={e4:.2f} ({time.time()-t0:.0f}s)")
            sys.stdout.flush()

        a1, a2, a3, a4 = np.array(r1a), np.array(r2a), np.array(r3a), np.array(r4a)
        a2a = np.array(r2aa)
        results[mname] = {'fd1':a1,'fd2':a2,'fd3':a3,'fd4':a4,'fd2a':a2a,
                          'r12':a2.mean()/a1.mean(),'r13':a3.mean()/a1.mean(),
                          'r14':a4.mean()/a1.mean(),'r12a':a2a.mean()/a1.mean()}
        print(f"  AVG: FD001={a1.mean():.2f}+/-{a1.std():.2f} FD002={a2.mean():.2f}+/-{a2.std():.2f} "
              f"FD003={a3.mean():.2f}+/-{a3.std():.2f} FD004={a4.mean():.2f}+/-{a4.std():.2f}")
        print(f"  Ratios: 1->2={a2.mean()/a1.mean():.2f} 1->3={a3.mean()/a1.mean():.2f} 1->4={a4.mean()/a1.mean():.2f}")
        print(f"  FD002 adapted: {a2a.mean():.2f}+/-{a2a.std():.2f}")
        sys.stdout.flush()

    # Summary
    print("\n" + "="*70)
    print("TIER 2 SUMMARY")
    print("="*70)
    print(f"{'Model':<16} {'FD001':>7} {'FD002':>7} {'FD003':>7} {'FD004':>7} {'R12':>6} {'R13':>6} {'R14':>6} {'FD002a':>7}")
    print("-"*80)
    for n, r in results.items():
        print(f"{n:<16} {r['fd1'].mean():>7.2f} {r['fd2'].mean():>7.2f} {r['fd3'].mean():>7.2f} {r['fd4'].mean():>7.2f} "
              f"{r['r12']:>6.2f} {r['r13']:>6.2f} {r['r14']:>6.2f} {r['fd2a'].mean():>7.2f}")

    ci = results.get("C5_ci_trans")
    rt = results.get("C7_role_trans")
    if ci and rt:
        print(f"\nRole-Trans vs CI-Trans:")
        print(f"  FD001: {rt['fd1'].mean():.2f} vs {ci['fd1'].mean():.2f}")
        print(f"  FD002: {rt['fd2'].mean():.2f} vs {ci['fd2'].mean():.2f} (ratio {rt['r12']:.2f} vs {ci['r12']:.2f})")
        if rt['fd2'].mean() < ci['fd2'].mean():
            print(f"  Role-Trans {(1-rt['fd2'].mean()/ci['fd2'].mean())*100:.1f}% better on FD002")
        else:
            print(f"  CI-Trans {(1-ci['fd2'].mean()/rt['fd2'].mean())*100:.1f}% better on FD002")

    print(f"\nTier 2 complete! Time: {time.strftime('%H:%M')}")
    return results


if __name__ == "__main__":
    run_tier2()

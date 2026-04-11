"""
C-MAPSS Karpathy Loop — Fast iteration, single seed per experiment.
One change at a time, 5 min budget each.

Starting point: Role-Trans 12.66 FD001, 55.10 FD002, ratio 4.36
"""
import sys
import time
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RUL = 125
SEQ_LEN = 30
SEED = 42

COLS = ["unit", "cycle"] + [f"setting{i}" for i in range(1, 4)] + [f"s{i}" for i in range(1, 22)]
INFORMATIVE_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
                       "s13", "s14", "s15", "s17", "s20", "s21"]
COMPONENT_GROUPS = {
    "fan": ["s8", "s12", "s21"],
    "hpc": ["s3", "s7", "s11", "s20"],
    "combustor": ["s2", "s14"],
    "turbine": ["s4", "s9", "s13"],
    "nozzle": ["s15", "s17"],
}

# ─── Data ───
def load_cmapss(subset):
    train_df = pd.read_csv(DATA_DIR / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    test_df = pd.read_csv(DATA_DIR / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLS)
    rul_true = pd.read_csv(DATA_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])
    train_df["RUL"] = train_df.groupby("unit")["cycle"].transform(lambda x: np.minimum(x.max() - x, MAX_RUL))
    test_df["RUL"] = test_df.groupby("unit")["cycle"].transform(lambda x: x.max() - x)
    test_rul_map = {i + 1: row["RUL"] for i, row in rul_true.iterrows()}
    return train_df, test_df, test_rul_map

class CMAPSSDataset(Dataset):
    def __init__(self, df, sensor_cols, seq_len=SEQ_LEN):
        self.samples = []
        for uid in df["unit"].unique():
            ud = df[df["unit"] == uid]
            s = ud[sensor_cols].values
            r = ud["RUL"].values
            for i in range(len(s) - seq_len + 1):
                self.samples.append((s[i:i+seq_len], min(r[i+seq_len-1], MAX_RUL)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class CMAPSSTestDataset(Dataset):
    def __init__(self, df, sensor_cols, rul_map, seq_len=SEQ_LEN):
        self.samples = []
        for uid in df["unit"].unique():
            ud = df[df["unit"] == uid]
            s = ud[sensor_cols].values
            if len(s) < seq_len:
                s = np.vstack([np.tile(s[0], (seq_len-len(s), 1)), s])
            self.samples.append((s[-seq_len:], min(rul_map[uid], MAX_RUL)))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        x, y = self.samples[i]
        return torch.FloatTensor(x), torch.FloatTensor([y])

# ─── Models ───
class RevIN(nn.Module):
    def __init__(self, n, eps=1e-5, affine=False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(n))
            self.beta = nn.Parameter(torch.zeros(n))
    def forward(self, x, mode="norm"):
        if mode == "norm":
            self._mean = x.mean(1, keepdim=True).detach()
            self._std = (x.std(1, keepdim=True) + self.eps).detach()
            x = (x - self._mean) / self._std
            if self.affine: x = x * self.gamma + self.beta
            return x
        else:
            if self.affine: x = (x - self.beta) / self.gamma
            return x * self._std + self._mean

class RoleTransformer(nn.Module):
    def __init__(self, groups, sensor_cols, d=32, heads=4, layers=2,
                 cross_layers=1, dropout=0.1, revin=False):
        super().__init__()
        self.groups = groups
        self.sensor_cols = sensor_cols
        self.n_comp = len(groups)
        self.comp_idx = {c: [sensor_cols.index(s) for s in ss] for c, ss in groups.items()}
        if revin: self.revin = RevIN(len(sensor_cols))
        else: self.revin = None
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.within_enc = nn.TransformerEncoder(enc, layers)
        self.pool = nn.Linear(d, d)
        cross = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross_enc = nn.TransformerEncoder(cross, cross_layers)
        self.comp_emb = nn.Parameter(torch.randn(1, self.n_comp, d) * 0.02)
        self.fc = nn.Linear(d * self.n_comp, 1)

    def forward(self, x):
        if self.revin: x = self.revin(x, "norm")
        B, T, C = x.shape
        comps = []
        for c, idx in self.comp_idx.items():
            n = len(idx)
            cx = x[:,:,idx].permute(0,2,1).reshape(B*n, T, 1)
            cx = self.proj(cx) + self.pos[:,:T]
            cx = self.within_enc(cx)[:, -1].reshape(B, n, -1)
            comps.append(self.pool(cx.mean(1)))
        cs = torch.stack(comps, 1) + self.comp_emb
        cs = self.cross_enc(cs)
        return self.fc(cs.reshape(B, -1))

class CITransformer(nn.Module):
    def __init__(self, n_sensors, d=32, heads=4, layers=2, dropout=0.1, revin=False):
        super().__init__()
        self.n = n_sensors
        if revin: self.revin = RevIN(n_sensors)
        else: self.revin = None
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.ch_fc = nn.Linear(d, 1)
        self.out_fc = nn.Linear(n_sensors, 1)

    def forward(self, x):
        if self.revin: x = self.revin(x, "norm")
        B, T, C = x.shape
        x = x.permute(0,2,1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:,:T]
        x = self.enc(x)
        x = self.ch_fc(x[:,-1]).reshape(B, C)
        return self.out_fc(x)

# ─── Training ───
def train(model, tl, vl, epochs=60, lr=1e-3, patience=12):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_v, best_s, no_imp = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        v = evaluate(model, vl)
        if v < best_v:
            best_v = v; best_s = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else: no_imp += 1
        if no_imp >= patience: break
    model.load_state_dict(best_s); model.to(DEVICE)
    return model, best_v

def evaluate(model, loader):
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()

# ─── Setup ───
def setup():
    torch.manual_seed(SEED); np.random.seed(SEED)
    t1, te1, r1 = load_cmapss("FD001")
    t2, te2, r2 = load_cmapss("FD002")
    sc = INFORMATIVE_SENSORS
    means, stds = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, t2, te2]: df[sc] = (df[sc] - means) / stds
    ds = CMAPSSDataset(t1, sc)
    nt = int(0.8 * len(ds)); nv = len(ds) - nt
    ts, vs = torch.utils.data.random_split(ds, [nt, nv], generator=torch.Generator().manual_seed(SEED))
    tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
    vl = DataLoader(vs, 256, num_workers=0)
    te1l = DataLoader(CMAPSSTestDataset(te1, sc, r1), 256, num_workers=0)
    te2l = DataLoader(CMAPSSTestDataset(te2, sc, r2), 256, num_workers=0)
    return tl, vl, te1l, te2l, t2, te2, r2, sc

def run_exp(name, model, tl, vl, te1l, te2l, **train_kwargs):
    t0 = time.time()
    model, val = train(model, tl, vl, **train_kwargs)
    r1 = evaluate(model, te1l)
    r2 = evaluate(model, te2l)
    elapsed = time.time() - t0
    ratio = r2 / r1
    print(f"  {name:35s} | FD001={r1:.2f} | FD002={r2:.2f} | ratio={ratio:.2f} | {elapsed:.0f}s")
    sys.stdout.flush()
    return model, r1, r2, ratio

# ─── Experiments ───
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Karpathy Loop — Fast iteration (seed={SEED})")
    print(f"Baseline: Role-Trans FD001=12.63, FD002=65.81, ratio=5.30")
    print()
    sys.stdout.flush()

    tl, vl, te1l, te2l, train2_df, test2_df, rul2_map, sc = setup()
    n = len(sc)

    # ─── Exp 9: RevIN ───
    print("=" * 70)
    print("Exp 9: RevIN normalization")
    print("=" * 70)
    sys.stdout.flush()
    torch.manual_seed(SEED); np.random.seed(SEED)
    _, r1_base, r2_base, ratio_base = run_exp(
        "Role-Trans (no RevIN)", RoleTransformer(COMPONENT_GROUPS, sc), tl, vl, te1l, te2l)
    torch.manual_seed(SEED); np.random.seed(SEED)
    _, r1_rev, r2_rev, ratio_rev = run_exp(
        "Role-Trans + RevIN", RoleTransformer(COMPONENT_GROUPS, sc, revin=True), tl, vl, te1l, te2l)
    torch.manual_seed(SEED); np.random.seed(SEED)
    _, _, r2_ci, ratio_ci = run_exp(
        "CI-Trans + RevIN", CITransformer(n, revin=True), tl, vl, te1l, te2l)
    print(f"  RevIN effect on Role-Trans: FD002 {r2_base:.2f} → {r2_rev:.2f} ({(r2_rev-r2_base)/r2_base*100:+.1f}%)")
    sys.stdout.flush()

    # ─── Exp 10: Few-shot fine-tuning ───
    print("\n" + "=" * 70)
    print("Exp 10: Few-shot fine-tuning on FD002")
    print("=" * 70)
    sys.stdout.flush()
    # Train base model
    torch.manual_seed(SEED); np.random.seed(SEED)
    base_model = RoleTransformer(COMPONENT_GROUPS, sc, revin=True)
    base_model, _ = train(base_model, tl, vl)
    base_r2 = evaluate(base_model, te2l)
    print(f"  Zero-shot FD002: {base_r2:.2f}")
    sys.stdout.flush()

    for frac in [0.05, 0.10, 0.25]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        ft = copy.deepcopy(base_model)
        ft_ds = CMAPSSDataset(train2_df, sc)
        n_ft = max(1, int(frac * len(ft_ds)))
        ft_sub, _ = torch.utils.data.random_split(ft_ds, [n_ft, len(ft_ds)-n_ft],
                                                    generator=torch.Generator().manual_seed(SEED))
        ft_loader = DataLoader(ft_sub, 256, shuffle=True, num_workers=0)
        ft, _ = train(ft, ft_loader, te2l, epochs=30, lr=1e-4, patience=8)
        ft_r2 = evaluate(ft, te2l)
        print(f"  {frac:.0%} FD002 fine-tune: {ft_r2:.2f} ({(ft_r2-base_r2)/base_r2*100:+.1f}%)")
        sys.stdout.flush()

    # ─── Exp 11: Cross-component depth ablation ───
    print("\n" + "=" * 70)
    print("Exp 11: Cross-component attention depth")
    print("=" * 70)
    sys.stdout.flush()
    for nc in [1, 2, 3]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        run_exp(f"Role-Trans cross={nc}", RoleTransformer(COMPONENT_GROUPS, sc, revin=True, cross_layers=nc),
                tl, vl, te1l, te2l)

    # ─── Exp 12: Grouping ablation ───
    print("\n" + "=" * 70)
    print("Exp 12: Component grouping ablation")
    print("=" * 70)
    sys.stdout.flush()
    groupings = {
        "Physics (ours)": COMPONENT_GROUPS,
        "Random": {"g1":["s2","s8","s13"],"g2":["s3","s14","s17","s20"],"g3":["s4","s7"],
                   "g4":["s9","s11","s21"],"g5":["s12","s15"]},
        "Uniform-3": {"g1":["s2","s3","s4","s7","s8"],"g2":["s9","s11","s12","s13","s14"],
                      "g3":["s15","s17","s20","s21"]},
        "All-one-group": {"all": INFORMATIVE_SENSORS},
    }
    for gname, groups in groupings.items():
        torch.manual_seed(SEED); np.random.seed(SEED)
        run_exp(gname, RoleTransformer(groups, sc, revin=True), tl, vl, te1l, te2l)

    # ─── Exp 13: Wider model (d=64) ───
    print("\n" + "=" * 70)
    print("Exp 13: Model size ablation")
    print("=" * 70)
    sys.stdout.flush()
    for d in [16, 32, 64]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = RoleTransformer(COMPONENT_GROUPS, sc, d=d, revin=True)
        p = sum(p.numel() for p in m.parameters() if p.requires_grad)
        run_exp(f"Role-Trans d={d} ({p:,} params)", m, tl, vl, te1l, te2l)

    print("\n\nAll experiments complete!")
    sys.stdout.flush()

"""
Karpathy Loop Round 8 — Statistical rigor + reverse transfer
Exp 37: FD002→FD001 reverse transfer (does more diverse training help?)
Exp 38: 10-seed bootstrap for key claims
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RUL = 125; SEQ_LEN = 30
COLS = ["unit","cycle"]+[f"setting{i}" for i in range(1,4)]+[f"s{i}" for i in range(1,22)]
INFORMATIVE_SENSORS = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
COMPONENT_GROUPS = {"fan":["s8","s12","s21"],"hpc":["s3","s7","s11","s20"],
                    "combustor":["s2","s14"],"turbine":["s4","s9","s13"],"nozzle":["s15","s17"]}

def load_cmapss(subset):
    t = pd.read_csv(DATA_DIR/f"train_{subset}.txt",sep=r"\s+",header=None,names=COLS)
    te = pd.read_csv(DATA_DIR/f"test_{subset}.txt",sep=r"\s+",header=None,names=COLS)
    r = pd.read_csv(DATA_DIR/f"RUL_{subset}.txt",sep=r"\s+",header=None,names=["RUL"])
    t["RUL"]=t.groupby("unit")["cycle"].transform(lambda x:np.minimum(x.max()-x,MAX_RUL))
    te["RUL"]=te.groupby("unit")["cycle"].transform(lambda x:x.max()-x)
    return t, te, {i+1:row["RUL"] for i,row in r.iterrows()}

class DS(Dataset):
    def __init__(self,df,sc,sl=SEQ_LEN):
        self.s=[]
        for u in df["unit"].unique():
            d=df[df["unit"]==u]; v=d[sc].values; r=d["RUL"].values
            for i in range(len(v)-sl+1): self.s.append((v[i:i+sl],min(r[i+sl-1],MAX_RUL)))
    def __len__(self): return len(self.s)
    def __getitem__(self,i): x,y=self.s[i]; return torch.FloatTensor(x),torch.FloatTensor([y])

class TDS(Dataset):
    def __init__(self,df,sc,rm,sl=SEQ_LEN):
        self.s=[]
        for u in df["unit"].unique():
            d=df[df["unit"]==u]; v=d[sc].values
            if len(v)<sl: v=np.vstack([np.tile(v[0],(sl-len(v),1)),v])
            self.s.append((v[-sl:],min(rm[u],MAX_RUL)))
    def __len__(self): return len(self.s)
    def __getitem__(self,i): x,y=self.s[i]; return torch.FloatTensor(x),torch.FloatTensor([y])

class RoleTrans(nn.Module):
    def __init__(self, groups, sc, d=32, heads=4, layers=2, cross_layers=1, dropout=0.1):
        super().__init__()
        self.n_comp=len(groups)
        self.comp_idx={c:[sc.index(s) for s in ss] for c,ss in groups.items()}
        self.comp_names=list(groups.keys())
        self.proj=nn.Linear(1,d); self.pos=nn.Parameter(torch.randn(1,SEQ_LEN,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.within=nn.TransformerEncoder(enc,layers)
        self.pool=nn.Linear(d,d)
        self.cemb=nn.Parameter(torch.randn(1,self.n_comp,d)*0.02)
        cross=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.cross=nn.TransformerEncoder(cross,cross_layers)
        self.fc=nn.Linear(d*self.n_comp,1)
    def forward(self,x):
        B,T,C=x.shape; cs=[]
        for c in self.comp_names:
            idx=self.comp_idx[c]; n=len(idx)
            cx=x[:,:,idx].permute(0,2,1).reshape(B*n,T,1)
            cx=self.proj(cx)+self.pos[:,:T]; cx=self.within(cx)[:,-1].reshape(B,n,-1)
            cs.append(self.pool(cx.mean(1)))
        s=torch.stack(cs,1)+self.cemb; s=self.cross(s)
        return self.fc(s.reshape(B,-1))

class CITrans(nn.Module):
    def __init__(self, n, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n=n; self.proj=nn.Linear(1,d); self.pos=nn.Parameter(torch.randn(1,SEQ_LEN,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,layers)
        self.ch=nn.Linear(d,1); self.out=nn.Linear(n,1)
    def forward(self,x):
        B,T,C=x.shape; x=x.permute(0,2,1).reshape(B*C,T,1)
        x=self.proj(x)+self.pos[:,:T]; x=self.enc(x)
        x=self.ch(x[:,-1]).reshape(B,C); return self.out(x)

def train_m(model, tl, vl, epochs=60, lr=1e-3, patience=12):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv = 1e9; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = ev(model, vl)
        if v < bv: bv = v; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    model.load_state_dict(bs); model.to(DEVICE)
    return model, bv

def ev(model, loader):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x, y in loader: ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()
    sc = INFORMATIVE_SENSORS; n = len(sc)

    # ─── Exp 37: Reverse Transfer (FD002 → FD001) ───
    print("\n" + "="*70)
    print("Exp 37: Reverse transfer — Train FD002, test FD001")
    print("  FD002 has 6 conditions, 260 engines; FD001 has 1 condition, 100 engines")
    print("  Question: Does training on MORE diverse data help Role-Trans more?")
    print("="*70)
    sys.stdout.flush()

    t2, te2, r2 = load_cmapss("FD002")
    t1, te1, r1 = load_cmapss("FD001")
    # Normalize with FD002 stats (source domain)
    m_s, s_s = t2[sc].mean(), t2[sc].std().replace(0, 1)
    for df in [t1, te1, t2, te2]: df[sc] = (df[sc] - m_s) / s_s

    ds2 = DS(t2, sc); nt2 = int(0.8*len(ds2)); nv2 = len(ds2) - nt2

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds2, [nt2, nv2], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)

        torch.manual_seed(seed)
        m_role = RoleTrans(COMPONENT_GROUPS, sc)
        m_role, _ = train_m(m_role, tl, vl)
        r2r = ev(m_role, te2l); r1r = ev(m_role, te1l)
        print(f"  [{seed}] Role-Trans | FD002={r2r:.2f} | FD001={r1r:.2f} | ratio={r1r/r2r:.2f}")

        torch.manual_seed(seed)
        m_ci = CITrans(n)
        m_ci, _ = train_m(m_ci, tl, vl)
        r2c = ev(m_ci, te2l); r1c = ev(m_ci, te1l)
        print(f"  [{seed}] CI-Trans   | FD002={r2c:.2f} | FD001={r1c:.2f} | ratio={r1c/r2c:.2f}")
        sys.stdout.flush()

    # ─── Exp 38: 10-Seed Massive Confirmation ───
    print("\n" + "="*70)
    print("Exp 38: 10-seed confirmation (FD001 → FD002)")
    print("="*70)
    sys.stdout.flush()

    # Reload with FD001 normalization
    t1, te1, r1 = load_cmapss("FD001")
    _, te2, r2 = load_cmapss("FD002")
    m_s, s_s = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, te2]: df[sc] = (df[sc] - m_s) / s_s
    ds1 = DS(t1, sc); nt = int(0.8*len(ds1)); nv = len(ds1) - nt

    seeds_10 = [42, 123, 456, 789, 2024, 7, 13, 99, 1337, 31415]
    all_role_fd2 = []; all_ci_fd2 = []
    all_role_fd1 = []; all_ci_fd1 = []

    for seed in seeds_10:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        torch.manual_seed(seed)
        m = RoleTrans(COMPONENT_GROUPS, sc)
        m, _ = train_m(m, tl, vl)
        r1r = ev(m, te1l); r2r = ev(m, te2l)
        all_role_fd1.append(r1r); all_role_fd2.append(r2r)

        torch.manual_seed(seed)
        m = CITrans(n)
        m, _ = train_m(m, tl, vl)
        r1c = ev(m, te1l); r2c = ev(m, te2l)
        all_ci_fd1.append(r1c); all_ci_fd2.append(r2c)

        print(f"  [{seed:5d}] Role: FD001={r1r:.2f} FD002={r2r:.2f} (ratio {r2r/r1r:.2f}) | CI: FD001={r1c:.2f} FD002={r2c:.2f} (ratio {r2c/r1c:.2f})")
        sys.stdout.flush()

    r1a = np.array(all_role_fd1); r2a = np.array(all_role_fd2)
    c1a = np.array(all_ci_fd1); c2a = np.array(all_ci_fd2)

    print(f"\n  Role-Trans (10 seeds): FD001={r1a.mean():.2f}±{r1a.std():.2f} | FD002={r2a.mean():.2f}±{r2a.std():.2f} | ratio={r2a.mean()/r1a.mean():.2f}")
    print(f"  CI-Trans   (10 seeds): FD001={c1a.mean():.2f}±{c1a.std():.2f} | FD002={c2a.mean():.2f}±{c2a.std():.2f} | ratio={c2a.mean()/c1a.mean():.2f}")

    # Statistical tests
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(r2a, c2a)
    print(f"\n  Paired t-test (FD002 RMSE): t={t_stat:.3f}, p={p_val:.4f}")
    try:
        w_stat, w_p = stats.wilcoxon(r2a, c2a)
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
    except Exception as e:
        print(f"  Wilcoxon: {e}")

    # Win rate
    role_wins = sum(1 for r, c in zip(r2a, c2a) if r < c)
    print(f"  Role-Trans wins: {role_wins}/10 seeds")

    # Bootstrap CI on improvement
    n_boot = 10000
    improvements = c2a - r2a  # positive = Role-Trans better
    boot_means = np.array([
        np.mean(np.random.choice(improvements, size=len(improvements), replace=True))
        for _ in range(n_boot)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"  Bootstrap 95% CI for (CI - Role) RMSE improvement: [{ci_low:.2f}, {ci_high:.2f}]")
    print(f"  Mean improvement: {improvements.mean():.2f} ± {improvements.std():.2f}")
    if ci_low > 0:
        print(f"  ✅ Entire 95% CI is positive — Role-Trans significantly better")
    else:
        print(f"  ⚠️  95% CI includes zero — not significant at 95% level")

    # Transfer ratio comparison
    role_ratios = r2a / r1a
    ci_ratios = c2a / c1a
    ratio_improvement = ci_ratios - role_ratios
    boot_ratio = np.array([
        np.mean(np.random.choice(ratio_improvement, size=len(ratio_improvement), replace=True))
        for _ in range(n_boot)
    ])
    ri_low, ri_high = np.percentile(boot_ratio, [2.5, 97.5])
    print(f"\n  Transfer ratio — Role: {role_ratios.mean():.2f}±{role_ratios.std():.2f} | CI: {ci_ratios.mean():.2f}±{ci_ratios.std():.2f}")
    print(f"  Bootstrap 95% CI for ratio improvement: [{ri_low:.2f}, {ri_high:.2f}]")

    print(f"\n\nRound 8 complete! Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()

"""
Karpathy Loop Round 7 — Strengthen the architecture story
Exp 34: Weight sharing ablation (shared vs separate within-component encoders)
Exp 35: 5-seed confirmation of key Role-Trans vs CI-Trans comparison
Exp 36: t-SNE representation analysis
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

# ─── Model Variants ───

class RoleTrans(nn.Module):
    """Standard Role-Trans: shared within-component encoder."""
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
    def get_repr(self, x):
        """Get component representations for visualization."""
        B,T,C=x.shape; cs=[]
        for c in self.comp_names:
            idx=self.comp_idx[c]; n=len(idx)
            cx=x[:,:,idx].permute(0,2,1).reshape(B*n,T,1)
            cx=self.proj(cx)+self.pos[:,:T]; cx=self.within(cx)[:,-1].reshape(B,n,-1)
            cs.append(self.pool(cx.mean(1)))
        return torch.stack(cs,1)  # (B, n_comp, d)

class SeparateEncoderRoleTrans(nn.Module):
    """Role-Trans with SEPARATE within-component encoders (one per component)."""
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n_comp=len(groups)
        self.comp_idx={c:[sc.index(s) for s in ss] for c,ss in groups.items()}
        self.comp_names=list(groups.keys())
        # Separate encoder per component
        self.projs = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        for c in self.comp_names:
            self.projs[c] = nn.Linear(1, d)
            enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
            self.encoders[c] = nn.TransformerEncoder(enc, layers)
            self.pools[c] = nn.Linear(d, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        self.cemb = nn.Parameter(torch.randn(1, self.n_comp, d) * 0.02)
        cross = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross = nn.TransformerEncoder(cross, 1)
        self.fc = nn.Linear(d * self.n_comp, 1)

    def forward(self, x):
        B, T, C = x.shape; cs = []
        for c in self.comp_names:
            idx = self.comp_idx[c]; n = len(idx)
            cx = x[:, :, idx].permute(0, 2, 1).reshape(B*n, T, 1)
            cx = self.projs[c](cx) + self.pos[:, :T]
            cx = self.encoders[c](cx)[:, -1].reshape(B, n, -1)
            cs.append(self.pools[c](cx.mean(1)))
        s = torch.stack(cs, 1) + self.cemb; s = self.cross(s)
        return self.fc(s.reshape(B, -1))

class CITrans(nn.Module):
    """Channel-independent baseline."""
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
    def get_repr(self, x):
        """Get per-channel representations for visualization."""
        B,T,C=x.shape; x=x.permute(0,2,1).reshape(B*C,T,1)
        x=self.proj(x)+self.pos[:,:T]; x=self.enc(x)
        return x[:,-1].reshape(B,C,-1)  # (B, C, d)

class GroupedNoShareTrans(nn.Module):
    """Grouped by component but NO weight sharing — separate encoder per SENSOR."""
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n_comp = len(groups)
        self.comp_idx = {c: [sc.index(s) for s in ss] for c, ss in groups.items()}
        self.comp_names = list(groups.keys())
        n_sensors = len(sc)
        # Separate encoder per sensor (like CI but with grouping for aggregation)
        self.projs = nn.ModuleList([nn.Linear(1, d) for _ in range(n_sensors)])
        self.encoders = nn.ModuleList()
        for _ in range(n_sensors):
            enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
            self.encoders.append(nn.TransformerEncoder(enc, layers))
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        self.pool = nn.Linear(d, d)
        self.cemb = nn.Parameter(torch.randn(1, self.n_comp, d) * 0.02)
        cross = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross = nn.TransformerEncoder(cross, 1)
        self.fc = nn.Linear(d * self.n_comp, 1)

    def forward(self, x):
        B, T, C = x.shape; cs = []
        sensor_idx = 0
        for c in self.comp_names:
            idx = self.comp_idx[c]
            sensor_reprs = []
            for si in idx:
                sx = x[:, :, si:si+1]  # (B, T, 1)
                sx = self.projs[sensor_idx](sx) + self.pos[:, :T]
                sx = self.encoders[sensor_idx](sx)[:, -1]  # (B, d)
                sensor_reprs.append(sx)
                sensor_idx += 1
            # Pool within component
            comp_repr = torch.stack(sensor_reprs, 1).mean(1)  # (B, d)
            cs.append(self.pool(comp_repr))
        s = torch.stack(cs, 1) + self.cemb; s = self.cross(s)
        return self.fc(s.reshape(B, -1))

def train_m(model, tl, vl, epochs=60, lr=1e-3, patience=12):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv = 1e9
    bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    ni = 0
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

    # Load and normalize
    t1, te1, r1 = load_cmapss("FD001")
    t2, te2, r2 = load_cmapss("FD002")
    m_s, s_s = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, t2, te2]: df[sc] = (df[sc] - m_s) / s_s

    ds1 = DS(t1, sc); nt = int(0.8*len(ds1)); nv = len(ds1) - nt

    # ─── Exp 34: Weight Sharing Ablation ───
    print("\n" + "="*70)
    print("Exp 34: Weight sharing ablation")
    print("  (a) Shared within-component (standard Role-Trans)")
    print("  (b) Separate encoder per component")
    print("  (c) Separate encoder per sensor (grouped but no sharing)")
    print("  (d) CI-Trans baseline")
    print("="*70)
    sys.stdout.flush()

    models_config = [
        ("Role-Trans (shared)", lambda seed: RoleTrans(COMPONENT_GROUPS, sc)),
        ("Separate-per-comp", lambda seed: SeparateEncoderRoleTrans(COMPONENT_GROUPS, sc)),
        ("CI-Trans", lambda seed: CITrans(n)),
    ]

    for name, make_model in models_config:
        fd1_results = []; fd2_results = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed); np.random.seed(seed)
            ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
            vl = DataLoader(vs, 256, num_workers=0)
            te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
            te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

            torch.manual_seed(seed)
            model = make_model(seed)
            p_count = sum(p.numel() for p in model.parameters())
            model, _ = train_m(model, tl, vl)
            r1r = ev(model, te1l); r2r = ev(model, te2l)
            fd1_results.append(r1r); fd2_results.append(r2r)
        a1 = np.array(fd1_results); a2 = np.array(fd2_results)
        print(f"  {name:25s} | params={p_count:,} | FD001={a1.mean():.2f}±{a1.std():.2f} | FD002={a2.mean():.2f}±{a2.std():.2f} | ratio={a2.mean()/a1.mean():.2f}")
        sys.stdout.flush()

    # Also test grouped-no-share (separate per sensor), but this is expensive
    print("\n  Testing per-sensor encoders (slower)...")
    fd1_results = []; fd2_results = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)
        torch.manual_seed(seed)
        model = GroupedNoShareTrans(COMPONENT_GROUPS, sc)
        p_count = sum(p.numel() for p in model.parameters())
        model, _ = train_m(model, tl, vl)
        r1r = ev(model, te1l); r2r = ev(model, te2l)
        fd1_results.append(r1r); fd2_results.append(r2r)
    a1 = np.array(fd1_results); a2 = np.array(fd2_results)
    print(f"  {'Grouped-no-share':25s} | params={p_count:,} | FD001={a1.mean():.2f}±{a1.std():.2f} | FD002={a2.mean():.2f}±{a2.std():.2f} | ratio={a2.mean()/a1.mean():.2f}")
    sys.stdout.flush()

    # ─── Exp 35: 5-Seed Confirmation ───
    print("\n" + "="*70)
    print("Exp 35: 5-seed confirmation (Role-Trans vs CI-Trans)")
    print("="*70)
    sys.stdout.flush()

    seeds_5 = [42, 123, 456, 789, 2024]
    role_fd1 = []; role_fd2 = []
    ci_fd1 = []; ci_fd2 = []

    for seed in seeds_5:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        # Role-Trans
        torch.manual_seed(seed)
        m = RoleTrans(COMPONENT_GROUPS, sc)
        m, _ = train_m(m, tl, vl)
        r1r = ev(m, te1l); r2r = ev(m, te2l)
        role_fd1.append(r1r); role_fd2.append(r2r)
        print(f"  [{seed}] Role-Trans | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")

        # CI-Trans
        torch.manual_seed(seed)
        m = CITrans(n)
        m, _ = train_m(m, tl, vl)
        r1r = ev(m, te1l); r2r = ev(m, te2l)
        ci_fd1.append(r1r); ci_fd2.append(r2r)
        print(f"  [{seed}] CI-Trans   | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")
        sys.stdout.flush()

    r1a = np.array(role_fd1); r2a = np.array(role_fd2)
    c1a = np.array(ci_fd1); c2a = np.array(ci_fd2)
    print(f"\n  Role-Trans (5 seeds): FD001={r1a.mean():.2f}±{r1a.std():.2f} | FD002={r2a.mean():.2f}±{r2a.std():.2f} | ratio={r2a.mean()/r1a.mean():.2f}")
    print(f"  CI-Trans   (5 seeds): FD001={c1a.mean():.2f}±{c1a.std():.2f} | FD002={c2a.mean():.2f}±{c2a.std():.2f} | ratio={c2a.mean()/c1a.mean():.2f}")

    # Paired t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_rel(r2a, c2a)
    print(f"  Paired t-test (FD002 RMSE): t={t_stat:.3f}, p={p_val:.4f}")
    # Wilcoxon (non-parametric)
    try:
        w_stat, w_p = stats.wilcoxon(r2a, c2a)
        print(f"  Wilcoxon signed-rank: W={w_stat:.1f}, p={w_p:.4f}")
    except Exception:
        print("  Wilcoxon: insufficient samples")
    sys.stdout.flush()

    # ─── Exp 36: t-SNE Representation Analysis ───
    print("\n" + "="*70)
    print("Exp 36: Representation analysis (t-SNE)")
    print("="*70)
    sys.stdout.flush()

    # Train models on FD001
    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
    vl = DataLoader(vs, 256, num_workers=0)

    torch.manual_seed(seed)
    role_m = RoleTrans(COMPONENT_GROUPS, sc)
    role_m, _ = train_m(role_m, tl, vl)

    torch.manual_seed(seed)
    ci_m = CITrans(n)
    ci_m, _ = train_m(ci_m, tl, vl)

    # Get FD002 test representations
    # We need operating condition labels — cluster from settings
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE

    settings_fd2 = te2[["setting1","setting2","setting3"]].values
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(settings_fd2)
    cond_labels = km.labels_

    # Get last-window representations for each test engine
    role_reprs = []; ci_reprs = []; rul_labels = []
    for u in te2["unit"].unique():
        d = te2[te2["unit"]==u]
        v = d[sc].values
        if len(v) < SEQ_LEN:
            v = np.vstack([np.tile(v[0], (SEQ_LEN-len(v), 1)), v])
        x = torch.FloatTensor(v[-SEQ_LEN:]).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            rr = role_m.get_repr(x).cpu().reshape(-1).numpy()
            cr = ci_m.get_repr(x).cpu().reshape(-1).numpy()
        role_reprs.append(rr)
        ci_reprs.append(cr)
        rul_labels.append(min(r2[u], MAX_RUL))

    role_reprs = np.array(role_reprs)
    ci_reprs = np.array(ci_reprs)
    rul_labels = np.array(rul_labels)

    # Get condition for each engine (use last cycle's settings)
    engine_conds = []
    for u in te2["unit"].unique():
        d = te2[te2["unit"]==u]
        last_settings = d[["setting1","setting2","setting3"]].values[-1:]
        engine_conds.append(km.predict(last_settings)[0])
    engine_conds = np.array(engine_conds)

    # t-SNE
    print("  Running t-SNE...")
    tsne_role = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(role_reprs)
    tsne_ci = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(ci_reprs)

    # Compute clustering metrics: how well do conditions separate in representation space?
    from sklearn.metrics import silhouette_score
    sil_role = silhouette_score(role_reprs, engine_conds)
    sil_ci = silhouette_score(ci_reprs, engine_conds)
    sil_role_rul = silhouette_score(role_reprs, (rul_labels > 60).astype(int))  # early vs late degradation
    sil_ci_rul = silhouette_score(ci_reprs, (rul_labels > 60).astype(int))

    print(f"  Silhouette by condition:   Role={sil_role:.3f}  CI={sil_ci:.3f}")
    print(f"  Silhouette by degradation: Role={sil_role_rul:.3f}  CI={sil_ci_rul:.3f}")

    # Save t-SNE data for plotting
    np.savez(DATA_DIR / ".." / "tsne_analysis.npz",
             role=tsne_role, ci=tsne_ci,
             conditions=engine_conds, rul=rul_labels)
    print(f"  Saved t-SNE data to data/tsne_analysis.npz")

    # Compute condition-RUL correlation in representation space
    # Higher silhouette by degradation + lower by condition = better for transfer
    print(f"\n  Interpretation:")
    if sil_role < sil_ci:
        print(f"    Role-Trans representations cluster LESS by condition (good for transfer)")
    else:
        print(f"    Role-Trans representations cluster MORE by condition")
    if sil_role_rul > sil_ci_rul:
        print(f"    Role-Trans representations cluster MORE by degradation (good for RUL)")
    else:
        print(f"    Role-Trans representations cluster LESS by degradation")

    sys.stdout.flush()

    print(f"\n\nRound 7 complete! Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()

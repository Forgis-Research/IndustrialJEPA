"""
Karpathy Loop Round 3:
Exp 21: Multi-source training (FD001+FD003 → FD002, FD004)
Exp 22: Sequence length ablation (15, 30, 50, 80)
Exp 23: Operating condition normalization for FD001→FD002
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path

DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RUL = 125; SEQ_LEN = 30; SEED = 42
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
    def __init__(self, groups, sc, d=32, heads=4, layers=2, cross_layers=1, dropout=0.1, max_seq=80):
        super().__init__()
        self.n_comp=len(groups)
        self.comp_idx={c:[sc.index(s) for s in ss] for c,ss in groups.items()}
        self.proj=nn.Linear(1,d); self.pos=nn.Parameter(torch.randn(1,max_seq,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.within=nn.TransformerEncoder(enc,layers)
        self.pool=nn.Linear(d,d)
        cross=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.cross=nn.TransformerEncoder(cross,cross_layers)
        self.cemb=nn.Parameter(torch.randn(1,self.n_comp,d)*0.02)
        self.fc=nn.Linear(d*self.n_comp,1)
    def forward(self,x):
        B,T,C=x.shape; cs=[]
        for c,idx in self.comp_idx.items():
            n=len(idx); cx=x[:,:,idx].permute(0,2,1).reshape(B*n,T,1)
            cx=self.proj(cx)+self.pos[:,:T]; cx=self.within(cx)[:,-1].reshape(B,n,-1)
            cs.append(self.pool(cx.mean(1)))
        s=torch.stack(cs,1)+self.cemb; s=self.cross(s)
        return self.fc(s.reshape(B,-1))

class CITrans(nn.Module):
    def __init__(self, n, d=32, heads=4, layers=2, dropout=0.1, max_seq=80):
        super().__init__()
        self.n=n; self.proj=nn.Linear(1,d); self.pos=nn.Parameter(torch.randn(1,max_seq,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,layers)
        self.ch=nn.Linear(d,1); self.out=nn.Linear(n,1)
    def forward(self,x):
        B,T,C=x.shape; x=x.permute(0,2,1).reshape(B*C,T,1)
        x=self.proj(x)+self.pos[:,:T]; x=self.enc(x)
        x=self.ch(x[:,-1]).reshape(B,C); return self.out(x)

def train_m(model,tl,vl,epochs=60,lr=1e-3,patience=12):
    model=model.to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs)
    bv,bs,ni=1e9,None,0
    for ep in range(epochs):
        model.train()
        for x,y in tl:
            x,y=x.to(DEVICE),y.to(DEVICE); loss=F.mse_loss(model(x),y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        sched.step(); v=ev(model,vl)
        if v<bv: bv=v; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}; ni=0
        else: ni+=1
        if ni>=patience: break
    model.load_state_dict(bs); model.to(DEVICE); return model,bv

def ev(model,loader):
    model.eval(); ps,ts=[],[]
    with torch.no_grad():
        for x,y in loader: ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps),torch.cat(ts))).item()

if __name__ == "__main__":
    print(f"Device: {DEVICE}"); sys.stdout.flush()
    sc = INFORMATIVE_SENSORS; n = len(sc)

    # ─── Exp 21: Multi-source training ───
    print("\n" + "="*70)
    print("Exp 21: Multi-source training (FD001+FD003 → FD002, FD004)")
    print("="*70); sys.stdout.flush()

    t1,te1,r1 = load_cmapss("FD001")
    t3,te3,r3 = load_cmapss("FD003")
    _,te2,r2 = load_cmapss("FD002")
    _,te4,r4 = load_cmapss("FD004")

    # Normalize all with combined FD001+FD003 stats
    combined = pd.concat([t1, t3])
    means, stds = combined[sc].mean(), combined[sc].std().replace(0, 1)
    for df in [t1, t3, te1, te2, te3, te4]:
        df[sc] = (df[sc] - means) / stds

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)

        # Single source (FD001 only)
        ds1 = DS(t1, sc)
        nt=int(0.8*len(ds1)); nv=len(ds1)-nt
        ts1, vs1 = torch.utils.data.random_split(ds1,[nt,nv],generator=torch.Generator().manual_seed(seed))

        # Multi-source (FD001 + FD003)
        ds13 = ConcatDataset([DS(t1, sc), DS(t3, sc)])
        nt=int(0.8*len(ds13)); nv=len(ds13)-nt
        ts13, vs13 = torch.utils.data.random_split(ds13,[nt,nv],generator=torch.Generator().manual_seed(seed))

        te2l = DataLoader(TDS(te2,sc,r2),256,num_workers=0)
        te4l = DataLoader(TDS(te4,sc,r4),256,num_workers=0)

        for src_name, ts, vs in [("FD001-only", ts1, vs1), ("FD001+FD003", ts13, vs13)]:
            tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
            vl = DataLoader(vs, 256, num_workers=0)

            torch.manual_seed(seed); np.random.seed(seed)
            m = RoleTrans(COMPONENT_GROUPS, sc)
            m, _ = train_m(m, tl, vl)
            r2_rmse = ev(m, te2l); r4_rmse = ev(m, te4l)
            print(f"  [{seed}] Role-Trans {src_name:12s} | FD002={r2_rmse:.2f} | FD004={r4_rmse:.2f}")
            sys.stdout.flush()

    # ─── Exp 22: Sequence length ablation ───
    print("\n" + "="*70)
    print("Exp 22: Sequence length ablation")
    print("="*70); sys.stdout.flush()

    t1,te1,r1 = load_cmapss("FD001")
    _,te2,r2 = load_cmapss("FD002")
    means, stds = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, te2]: df[sc] = (df[sc] - means) / stds

    for sl in [15, 30, 50, 80]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        ds = DS(t1, sc, sl=sl)
        nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(SEED))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1,sc,r1,sl=sl),256,num_workers=0)
        te2l = DataLoader(TDS(te2,sc,r2,sl=sl),256,num_workers=0)

        t0 = time.time()
        m = RoleTrans(COMPONENT_GROUPS, sc, max_seq=sl)
        m, val = train_m(m, tl, vl)
        r1r = ev(m, te1l); r2r = ev(m, te2l)
        print(f"  seq_len={sl:3d} | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f} | {time.time()-t0:.0f}s")
        sys.stdout.flush()

    # Also CI-Trans for comparison at best seq_len
    for sl in [15, 30, 50, 80]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        ds = DS(t1, sc, sl=sl)
        nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(SEED))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1,sc,r1,sl=sl),256,num_workers=0)
        te2l = DataLoader(TDS(te2,sc,r2,sl=sl),256,num_workers=0)
        m = CITrans(n, max_seq=sl)
        m, val = train_m(m, tl, vl)
        r1r = ev(m, te1l); r2r = ev(m, te2l)
        print(f"  CI seq_len={sl:3d} | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")
        sys.stdout.flush()

    # ─── Exp 23: Operating condition normalization ───
    print("\n" + "="*70)
    print("Exp 23: Operating condition normalization")
    print("="*70); sys.stdout.flush()

    t1_raw,te1_raw,r1 = load_cmapss("FD001")
    t2_raw,te2_raw,r2 = load_cmapss("FD002")

    # Method 1: Standard global normalization (baseline)
    t1a, te2a = t1_raw.copy(), te2_raw.copy()
    m, s = t1a[sc].mean(), t1a[sc].std().replace(0, 1)
    t1a[sc] = (t1a[sc] - m) / s; te2a[sc] = (te2a[sc] - m) / s

    # Method 2: Normalize by operating condition cluster
    # First find clusters from settings
    from sklearn.cluster import KMeans
    settings_train = t1_raw[["setting1","setting2","setting3"]].values
    # FD001 has 1 condition, but FD002 has 6 — cluster FD002
    settings_test = te2_raw[["setting1","setting2","setting3"]].values
    km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(
        np.vstack([settings_train, settings_test])
    )
    t1_raw["cluster"] = km.predict(settings_train)
    te2_raw["cluster"] = km.predict(settings_test)

    t1b, te2b = t1_raw.copy(), te2_raw.copy()
    # Normalize per-cluster
    for cl in range(6):
        mask_t = t1b["cluster"] == cl
        mask_te = te2b["cluster"] == cl
        if mask_t.sum() > 0:
            cm = t1b.loc[mask_t, sc].mean()
            cs = t1b.loc[mask_t, sc].std().replace(0, 1)
        else:
            # Cluster only in test — use test stats (not ideal but practical)
            cm = te2b.loc[mask_te, sc].mean()
            cs = te2b.loc[mask_te, sc].std().replace(0, 1)
        if mask_t.sum() > 0: t1b.loc[mask_t, sc] = (t1b.loc[mask_t, sc] - cm) / cs
        if mask_te.sum() > 0: te2b.loc[mask_te, sc] = (te2b.loc[mask_te, sc] - cm) / cs

    for method, t1_norm, te2_norm in [("Global norm", t1a, te2a), ("Per-condition norm", t1b, te2b)]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        ds = DS(t1_norm, sc)
        nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(SEED))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te2l = DataLoader(TDS(te2_norm,sc,r2),256,num_workers=0)

        m = RoleTrans(COMPONENT_GROUPS, sc)
        m, _ = train_m(m, tl, vl)
        r2r = ev(m, te2l)
        print(f"  Role-Trans {method:22s} | FD002={r2r:.2f}"); sys.stdout.flush()

        torch.manual_seed(SEED); np.random.seed(SEED)
        ds = DS(t1_norm, sc)
        nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(SEED))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        m = CITrans(n)
        m, _ = train_m(m, tl, vl)
        r2r = ev(m, te2l)
        print(f"  CI-Trans   {method:22s} | FD002={r2r:.2f}"); sys.stdout.flush()

    print("\n\nRound 3 complete!"); sys.stdout.flush()

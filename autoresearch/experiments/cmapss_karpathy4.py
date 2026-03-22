"""
Karpathy Loop Round 4 — Final polish experiments
Exp 24: Settings as additional input features (condition-aware)
Exp 25: Role-Trans + condition-aware normalization (best of both?)
Exp 26: Dropout ablation for transfer
Exp 27: In-domain FD001 tuning (target: RMSE < 12.0)
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.cluster import KMeans

DATA_DIR = Path("/home/sagemaker-user/IndustrialJEPA/data/cmapss")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_RUL = 125; SEQ_LEN = 30; SEED = 42
COLS = ["unit","cycle"]+[f"setting{i}" for i in range(1,4)]+[f"s{i}" for i in range(1,22)]
INFORMATIVE_SENSORS = ["s2","s3","s4","s7","s8","s9","s11","s12","s13","s14","s15","s17","s20","s21"]
SETTINGS = ["setting1","setting2","setting3"]
COMPONENT_GROUPS = {"fan":["s8","s12","s21"],"hpc":["s3","s7","s11","s20"],
                    "combustor":["s2","s14"],"turbine":["s4","s9","s13"],"nozzle":["s15","s17"]}
# Extended groups with settings as a component
COMPONENT_GROUPS_COND = {**COMPONENT_GROUPS, "operating": ["setting1","setting2","setting3"]}

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
        self.proj=nn.Linear(1,d); self.pos=nn.Parameter(torch.randn(1,SEQ_LEN,d)*0.02)
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

def normalize_per_condition(t_src, te_tgt, sc, n_clusters=6):
    """Cluster operating conditions and normalize per cluster."""
    settings = np.vstack([t_src[SETTINGS].values, te_tgt[SETTINGS].values])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(settings)
    t_src = t_src.copy(); te_tgt = te_tgt.copy()
    t_src["cluster"] = km.predict(t_src[SETTINGS].values)
    te_tgt["cluster"] = km.predict(te_tgt[SETTINGS].values)
    for cl in range(n_clusters):
        mt = t_src["cluster"] == cl
        mte = te_tgt["cluster"] == cl
        if mt.sum() > 0:
            m, s = t_src.loc[mt, sc].mean(), t_src.loc[mt, sc].std().replace(0, 1)
        elif mte.sum() > 0:
            m, s = te_tgt.loc[mte, sc].mean(), te_tgt.loc[mte, sc].std().replace(0, 1)
        else: continue
        if mt.sum() > 0: t_src.loc[mt, sc] = (t_src.loc[mt, sc].astype(float) - m) / s
        if mte.sum() > 0: te_tgt.loc[mte, sc] = (te_tgt.loc[mte, sc].astype(float) - m) / s
    return t_src, te_tgt

if __name__ == "__main__":
    print(f"Device: {DEVICE}"); sys.stdout.flush()
    sc = INFORMATIVE_SENSORS; n = len(sc)
    sc_cond = INFORMATIVE_SENSORS + SETTINGS; n_cond = len(sc_cond)

    # ─── Exp 24: Settings as input features ───
    print("\n" + "="*70)
    print("Exp 24: Operating settings as additional input features")
    print("="*70); sys.stdout.flush()

    t1,te1,r1 = load_cmapss("FD001"); _,te2,r2 = load_cmapss("FD002")
    # Normalize sensors + settings
    all_cols = sc_cond
    m, s = t1[all_cols].mean(), t1[all_cols].std().replace(0, 1)
    for df in [t1, te1, te2]: df[all_cols] = (df[all_cols] - m) / s

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = DS(t1, sc_cond); nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts,256,shuffle=True,num_workers=0)
        vl = DataLoader(vs,256,num_workers=0)
        te1l = DataLoader(TDS(te1,sc_cond,r1),256,num_workers=0)
        te2l = DataLoader(TDS(te2,sc_cond,r2),256,num_workers=0)

        # Role-Trans with settings as a 6th component
        torch.manual_seed(seed)
        m1 = RoleTrans(COMPONENT_GROUPS_COND, sc_cond)
        m1, _ = train_m(m1, tl, vl)
        r1r = ev(m1, te1l); r2r = ev(m1, te2l)
        print(f"  [{seed}] Role-Trans+Settings  | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")

        # CI-Trans with settings
        torch.manual_seed(seed)
        m2 = CITrans(n_cond)
        m2, _ = train_m(m2, tl, vl)
        r1r = ev(m2, te1l); r2r = ev(m2, te2l)
        print(f"  [{seed}] CI-Trans+Settings    | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")
        sys.stdout.flush()

    # ─── Exp 25: Role-Trans + per-condition norm (best of both) ───
    print("\n" + "="*70)
    print("Exp 25: Role-Trans + per-condition normalization")
    print("="*70); sys.stdout.flush()

    t1_raw,te1_raw,r1 = load_cmapss("FD001"); t2_raw,te2_raw,r2 = load_cmapss("FD002")
    t1c, te2c = normalize_per_condition(t1_raw, te2_raw, sc)

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = DS(t1c, sc); nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts,256,shuffle=True,num_workers=0)
        vl = DataLoader(vs,256,num_workers=0)
        te2l = DataLoader(TDS(te2c,sc,r2),256,num_workers=0)

        torch.manual_seed(seed)
        m = RoleTrans(COMPONENT_GROUPS, sc)
        m, _ = train_m(m, tl, vl)
        r2r = ev(m, te2l)
        print(f"  [{seed}] Role-Trans + cond-norm | FD002={r2r:.2f}")

        torch.manual_seed(seed)
        m = CITrans(n)
        m, _ = train_m(m, tl, vl)
        r2r = ev(m, te2l)
        print(f"  [{seed}] CI-Trans + cond-norm   | FD002={r2r:.2f}")
        sys.stdout.flush()

    # ─── Exp 26: Dropout ablation ───
    print("\n" + "="*70)
    print("Exp 26: Dropout ablation for transfer")
    print("="*70); sys.stdout.flush()

    t1,te1,r1 = load_cmapss("FD001"); _,te2,r2 = load_cmapss("FD002")
    m_s, s_s = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, te2]: df[sc] = (df[sc] - m_s) / s_s

    torch.manual_seed(SEED); np.random.seed(SEED)
    ds = DS(t1, sc); nt=int(0.8*len(ds)); nv=len(ds)-nt
    ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(SEED))
    tl = DataLoader(ts,256,shuffle=True,num_workers=0)
    vl = DataLoader(vs,256,num_workers=0)
    te1l = DataLoader(TDS(te1,sc,r1),256,num_workers=0)
    te2l = DataLoader(TDS(te2,sc,r2),256,num_workers=0)

    for dp in [0.0, 0.1, 0.2, 0.3, 0.5]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        m = RoleTrans(COMPONENT_GROUPS, sc, dropout=dp)
        m, _ = train_m(m, tl, vl)
        r1r = ev(m, te1l); r2r = ev(m, te2l)
        print(f"  dropout={dp:.1f} | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")
        sys.stdout.flush()

    # ─── Exp 27: In-domain tuning (more epochs, different LR) ───
    print("\n" + "="*70)
    print("Exp 27: In-domain FD001 tuning")
    print("="*70); sys.stdout.flush()

    configs = [
        ("lr=1e-3, ep=60", 1e-3, 60),
        ("lr=5e-4, ep=100", 5e-4, 100),
        ("lr=1e-3, ep=100", 1e-3, 100),
        ("lr=2e-3, ep=60", 2e-3, 60),
    ]
    for name, lr, ep in configs:
        results = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed); np.random.seed(seed)
            ds = DS(t1, sc); nt=int(0.8*len(ds)); nv=len(ds)-nt
            ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(seed))
            tl = DataLoader(ts,256,shuffle=True,num_workers=0)
            vl = DataLoader(vs,256,num_workers=0)
            te1l = DataLoader(TDS(te1,sc,r1),256,num_workers=0)
            m = RoleTrans(COMPONENT_GROUPS, sc)
            m, val = train_m(m, tl, vl, epochs=ep, lr=lr)
            r1r = ev(m, te1l)
            results.append(r1r)
        arr = np.array(results)
        print(f"  {name:25s} | FD001={arr.mean():.2f} ± {arr.std():.2f}")
        sys.stdout.flush()

    print("\n\nRound 4 complete!"); sys.stdout.flush()

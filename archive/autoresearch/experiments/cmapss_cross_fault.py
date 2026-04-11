"""
Exp 17-19: Cross-fault transfer on C-MAPSS
FD001 (HPC fault, 1 cond) → FD003 (HPC+Fan fault, 1 cond) — same conditions, different faults
FD001 → FD004 (HPC+Fan fault, 6 cond) — hardest: different faults AND conditions
FD003 → FD001 — reverse: can multi-fault model predict single-fault?
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
MAX_RUL = 125; SEQ_LEN = 30; SEEDS = [42, 123, 456]
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

def run_transfer(src, tgt, label):
    print(f"\n{'='*70}")
    print(f"Transfer: {src} → {tgt} ({label})")
    print(f"{'='*70}"); sys.stdout.flush()

    t_src, te_src, r_src = load_cmapss(src)
    _, te_tgt, r_tgt = load_cmapss(tgt)
    sc = INFORMATIVE_SENSORS; n = len(sc)

    # Normalize with source stats
    means, stds = t_src[sc].mean(), t_src[sc].std().replace(0, 1)
    for df in [t_src, te_src, te_tgt]:
        df[sc] = (df[sc] - means) / stds

    results = {"Role-Trans":{"src":[],"tgt":[]},"CI-Trans":{"src":[],"tgt":[]}}

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        ds = DS(t_src, sc); nt=int(0.8*len(ds)); nv=len(ds)-nt
        ts, vs = torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts,256,shuffle=True,num_workers=0)
        vl = DataLoader(vs,256,num_workers=0)
        src_l = DataLoader(TDS(te_src,sc,r_src),256,num_workers=0)
        tgt_l = DataLoader(TDS(te_tgt,sc,r_tgt),256,num_workers=0)

        for name, model in [("Role-Trans", RoleTrans(COMPONENT_GROUPS,sc)),
                            ("CI-Trans", CITrans(n))]:
            t0 = time.time()
            model, _ = train_m(model, tl, vl)
            rs = ev(model, src_l); rt = ev(model, tgt_l)
            results[name]["src"].append(rs); results[name]["tgt"].append(rt)
            print(f"  [{seed}] {name:12s} | {src}={rs:.2f} | {tgt}={rt:.2f} | ratio={rt/rs:.2f} | {time.time()-t0:.0f}s")
            sys.stdout.flush()

    print(f"\n{'─'*70}")
    print(f"{'Model':12s} | {src+' RMSE':>12s} | {tgt+' RMSE':>12s} | {'Ratio':>6s}")
    for name in results:
        s=np.array(results[name]["src"]); t=np.array(results[name]["tgt"])
        print(f"{name:12s} | {s.mean():.2f} ± {s.std():.2f} | {t.mean():.2f} ± {t.std():.2f} | {t.mean()/s.mean():.2f}")
    sys.stdout.flush()
    return results

if __name__ == "__main__":
    print(f"Device: {DEVICE}"); sys.stdout.flush()

    # Same operating conditions, different fault modes
    r1 = run_transfer("FD001", "FD003", "same cond, +fan fault")

    # Hardest: different faults AND conditions
    r2 = run_transfer("FD001", "FD004", "diff cond + diff faults")

    # Reverse: multi-fault → single-fault
    r3 = run_transfer("FD003", "FD001", "multi→single fault")

    # Also try FD002→FD004 (same conditions(6), but +fan fault)
    r4 = run_transfer("FD002", "FD004", "same 6 cond, +fan fault")

    print("\n\nAll cross-fault experiments complete!"); sys.stdout.flush()

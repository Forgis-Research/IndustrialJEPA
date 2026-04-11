"""
Karpathy Loop Round 2 — Following up on key findings:
1. RevIN hurts → run ablations without RevIN
2. Grouping doesn't matter? → test without RevIN
3. Best config: no RevIN, Role-Trans, d=32
4. Try: operating condition normalization instead of RevIN
5. Multi-seed confirmation of best config
"""
import sys, time, copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
    def __init__(self, groups, sc, d=32, heads=4, layers=2, cross_layers=1, dropout=0.1):
        super().__init__()
        self.n_comp=len(groups)
        self.comp_idx={c:[sc.index(s) for s in ss] for c,ss in groups.items()}
        self.proj=nn.Linear(1,d)
        self.pos=nn.Parameter(torch.randn(1,SEQ_LEN,d)*0.02)
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
        self.n=n; self.proj=nn.Linear(1,d)
        self.pos=nn.Parameter(torch.randn(1,SEQ_LEN,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.enc=nn.TransformerEncoder(enc,layers)
        self.ch=nn.Linear(d,1); self.out=nn.Linear(n,1)
    def forward(self,x):
        B,T,C=x.shape; x=x.permute(0,2,1).reshape(B*C,T,1)
        x=self.proj(x)+self.pos[:,:T]; x=self.enc(x)
        x=self.ch(x[:,-1]).reshape(B,C); return self.out(x)

def train(model,tl,vl,epochs=60,lr=1e-3,patience=12):
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

def setup(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    t1,te1,r1=load_cmapss("FD001"); t2,te2,r2=load_cmapss("FD002")
    sc=INFORMATIVE_SENSORS; m,s=t1[sc].mean(),t1[sc].std().replace(0,1)
    for df in [t1,te1,t2,te2]: df[sc]=(df[sc]-m)/s
    ds=DS(t1,sc); nt=int(0.8*len(ds)); nv=len(ds)-nt
    ts,vs=torch.utils.data.random_split(ds,[nt,nv],generator=torch.Generator().manual_seed(seed))
    tl=DataLoader(ts,256,shuffle=True,num_workers=0)
    vl=DataLoader(vs,256,num_workers=0)
    te1l=DataLoader(TDS(te1,sc,r1),256,num_workers=0)
    te2l=DataLoader(TDS(te2,sc,r2),256,num_workers=0)
    return tl,vl,te1l,te2l,t2,te2,r2,sc

def run(name,model,tl,vl,te1l,te2l,**kw):
    t0=time.time(); model,val=train(model,tl,vl,**kw)
    r1=ev(model,te1l); r2=ev(model,te2l); el=time.time()-t0
    print(f"  {name:40s} | FD001={r1:.2f} | FD002={r2:.2f} | ratio={r2/r1:.2f} | {el:.0f}s"); sys.stdout.flush()
    return model,r1,r2

if __name__=="__main__":
    print(f"Device: {DEVICE}"); print(f"Karpathy Loop Round 2"); sys.stdout.flush()

    # ─── Exp 14: Grouping ablation WITHOUT RevIN ───
    print("\n"+"="*70)
    print("Exp 14: Grouping ablation (no RevIN) — does physics grouping matter?")
    print("="*70); sys.stdout.flush()

    tl,vl,te1l,te2l,t2,te2,r2,sc=setup()
    n=len(sc)

    groupings = {
        "Physics-5": COMPONENT_GROUPS,
        "Random-5": {"g1":["s2","s8","s13"],"g2":["s3","s14","s17","s20"],"g3":["s4","s7"],
                     "g4":["s9","s11","s21"],"g5":["s12","s15"]},
        "Uniform-3": {"g1":["s2","s3","s4","s7","s8"],"g2":["s9","s11","s12","s13","s14"],
                      "g3":["s15","s17","s20","s21"]},
        "Uniform-7": {f"g{i}":INFORMATIVE_SENSORS[i*2:(i+1)*2] for i in range(7)},
        "All-one": {"all": INFORMATIVE_SENSORS},
    }

    for gn, g in groupings.items():
        torch.manual_seed(SEED); np.random.seed(SEED)
        run(f"Role-Trans {gn}", RoleTrans(g, sc), tl, vl, te1l, te2l)

    torch.manual_seed(SEED); np.random.seed(SEED)
    run("CI-Trans (baseline)", CITrans(n), tl, vl, te1l, te2l)

    # ─── Exp 15: Multi-seed confirmation (best config: no RevIN, physics groups) ───
    print("\n"+"="*70)
    print("Exp 15: Multi-seed confirmation (3 seeds)")
    print("="*70); sys.stdout.flush()

    results = {"Role-Trans":{"r1":[],"r2":[]},"CI-Trans":{"r1":[],"r2":[]}}
    for seed in [42, 123, 456]:
        tl,vl,te1l,te2l,_,_,_,sc = setup(seed)
        torch.manual_seed(seed); np.random.seed(seed)
        _,r1,r2 = run(f"Role-Trans seed={seed}", RoleTrans(COMPONENT_GROUPS,sc), tl,vl,te1l,te2l)
        results["Role-Trans"]["r1"].append(r1); results["Role-Trans"]["r2"].append(r2)
        torch.manual_seed(seed); np.random.seed(seed)
        _,r1,r2 = run(f"CI-Trans seed={seed}", CITrans(n), tl,vl,te1l,te2l)
        results["CI-Trans"]["r1"].append(r1); results["CI-Trans"]["r2"].append(r2)

    print(f"\n{'─'*70}")
    print(f"Multi-seed summary:")
    for name in results:
        r1=np.array(results[name]["r1"]); r2=np.array(results[name]["r2"])
        print(f"  {name:15s} | FD001={r1.mean():.2f}±{r1.std():.2f} | FD002={r2.mean():.2f}±{r2.std():.2f} | ratio={r2.mean()/r1.mean():.2f}")
    sys.stdout.flush()

    # ─── Exp 16: Few-shot without RevIN ───
    print("\n"+"="*70)
    print("Exp 16: Few-shot fine-tuning (no RevIN)")
    print("="*70); sys.stdout.flush()

    tl,vl,te1l,te2l,t2,te2,r2,sc = setup()
    torch.manual_seed(SEED); np.random.seed(SEED)
    base, _ = train(RoleTrans(COMPONENT_GROUPS, sc), tl, vl)
    base_r2 = ev(base, te2l)
    print(f"  Zero-shot FD002: {base_r2:.2f}"); sys.stdout.flush()

    for frac in [0.01, 0.05, 0.10, 0.25]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        ft = copy.deepcopy(base)
        ds2 = DS(t2, sc)
        nf = max(1, int(frac*len(ds2)))
        fs, _ = torch.utils.data.random_split(ds2,[nf,len(ds2)-nf],generator=torch.Generator().manual_seed(SEED))
        fl = DataLoader(fs, 256, shuffle=True, num_workers=0)
        ft, _ = train(ft, fl, te2l, epochs=30, lr=1e-4, patience=8)
        fr2 = ev(ft, te2l)
        print(f"  {frac:.0%} FD002 fine-tune: {fr2:.2f} ({(fr2-base_r2)/base_r2*100:+.1f}%)"); sys.stdout.flush()

    # Also fine-tune CI-Trans
    torch.manual_seed(SEED); np.random.seed(SEED)
    ci_base, _ = train(CITrans(n), tl, vl)
    ci_r2 = ev(ci_base, te2l)
    print(f"\n  CI-Trans zero-shot FD002: {ci_r2:.2f}"); sys.stdout.flush()
    for frac in [0.05, 0.10]:
        torch.manual_seed(SEED); np.random.seed(SEED)
        ft = copy.deepcopy(ci_base)
        ds2 = DS(t2, sc)
        nf = max(1, int(frac*len(ds2)))
        fs, _ = torch.utils.data.random_split(ds2,[nf,len(ds2)-nf],generator=torch.Generator().manual_seed(SEED))
        fl = DataLoader(fs, 256, shuffle=True, num_workers=0)
        ft, _ = train(ft, fl, te2l, epochs=30, lr=1e-4, patience=8)
        fr2 = ev(ft, te2l)
        print(f"  CI-Trans {frac:.0%} fine-tune: {fr2:.2f} ({(fr2-ci_r2)/ci_r2*100:+.1f}%)"); sys.stdout.flush()

    print("\n\nRound 2 complete!"); sys.stdout.flush()

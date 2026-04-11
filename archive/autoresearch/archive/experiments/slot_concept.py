"""
Slot-Concept Transformer for C-MAPSS
Exp 39: Can slot attention discover physical components from sensor data?

Architecture:
1. Shared temporal encoder processes each sensor independently
2. Slot attention decomposes C sensor features into K concept slots
3. Cross-slot transformer captures concept interactions
4. RUL head predicts from concatenated slot representations
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
# Ground truth component groups for validation
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

# ─── Slot Attention Module ───
class SlotAttention(nn.Module):
    def __init__(self, n_slots, d, n_iters=3, hidden_dim=64):
        super().__init__()
        self.n_slots = n_slots
        self.n_iters = n_iters
        self.d = d
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, d) * (d ** -0.5))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, n_slots, d))
        self.to_q = nn.Linear(d, d)
        self.to_k = nn.Linear(d, d)
        self.to_v = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d)
        )
        self.norm_slots = nn.LayerNorm(d)
        self.norm_inputs = nn.LayerNorm(d)

    def forward(self, inputs, return_attn=False):
        B, N, D = inputs.shape
        # Initialize slots from learned distribution
        slots = self.slots_mu + self.slots_log_sigma.exp() * torch.randn_like(self.slots_mu)
        slots = slots.expand(B, -1, -1).clone()
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs)  # (B, N, D)
        v = self.to_v(inputs)  # (B, N, D)

        attn_weights = None
        for _ in range(self.n_iters):
            q = self.to_q(self.norm_slots(slots))  # (B, K, D)
            # Attention: softmax over SLOTS dimension (competition for inputs)
            attn_logits = torch.bmm(k, q.transpose(1, 2)) / (D ** 0.5)  # (B, N, K)
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, N, K) — each input assigned to slots
            # Weighted mean of values per slot (normalize by slot attention mass)
            attn_norm = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1, K)
            slot_attn = attn_weights / attn_norm  # (B, N, K)
            updates = torch.bmm(slot_attn.transpose(1, 2), v)  # (B, K, D)
            # GRU update
            slots = self.gru(
                updates.reshape(B * self.n_slots, D),
                slots.reshape(B * self.n_slots, D)
            ).reshape(B, self.n_slots, D)
            slots = slots + self.mlp(slots)

        if return_attn:
            return slots, attn_weights  # attn_weights: (B, N, K) = channel-to-slot assignment
        return slots

# ─── Slot-Concept Transformer ───
class SlotConceptTransformer(nn.Module):
    def __init__(self, n_channels, n_slots=5, d=32, heads=4, layers=2,
                 slot_iters=3, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.n_slots = n_slots
        # Shared temporal encoder (channel-independent, like CI-Trans)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.temporal_enc = nn.TransformerEncoder(enc, layers)
        # Channel-to-feature pooling
        self.ch_pool = nn.Linear(d, d)
        # Slot attention
        self.slot_attn = SlotAttention(n_slots, d, n_iters=slot_iters, hidden_dim=d*2)
        # Cross-slot transformer
        self.slot_pos = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)
        cross_enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross_slot = nn.TransformerEncoder(cross_enc, 1)
        # RUL head
        self.head = nn.Linear(d * n_slots, 1)

    def encode_channels(self, x):
        """Encode each channel independently, return per-channel features."""
        B, T, C = x.shape
        # (B, T, C) → (B*C, T, 1) → encode → (B*C, d) → (B, C, d)
        cx = x.permute(0, 2, 1).reshape(B * C, T, 1)
        cx = self.proj(cx) + self.pos[:, :T]
        cx = self.temporal_enc(cx)[:, -1]  # last timestep features
        cx = cx.reshape(B, C, -1)
        return self.ch_pool(cx)  # (B, C, d)

    def forward(self, x):
        ch_features = self.encode_channels(x)  # (B, C, d)
        slots = self.slot_attn(ch_features)  # (B, K, d)
        slots = self.cross_slot(slots + self.slot_pos)
        return self.head(slots.reshape(slots.shape[0], -1))

    def get_assignments(self, x):
        """Get channel-to-slot assignments for interpretability."""
        ch_features = self.encode_channels(x)
        slots, attn = self.slot_attn(ch_features, return_attn=True)
        return attn  # (B, C, K)

# ─── Baselines ───
class RoleTrans(nn.Module):
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n_comp=len(groups); self.comp_idx={c:[sc.index(s) for s in ss] for c,ss in groups.items()}
        self.comp_names=list(groups.keys())
        self.proj=nn.Linear(1,d); self.pos=nn.Parameter(torch.randn(1,SEQ_LEN,d)*0.02)
        enc=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.within=nn.TransformerEncoder(enc,layers)
        self.pool=nn.Linear(d,d)
        self.cemb=nn.Parameter(torch.randn(1,self.n_comp,d)*0.02)
        cross=nn.TransformerEncoderLayer(d,heads,d*4,dropout,batch_first=True)
        self.cross=nn.TransformerEncoder(cross,1)
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

# ─── Training ───
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

def analyze_assignments(model, loader, sensor_names, component_groups):
    """Analyze what channels each slot captures."""
    model.eval()
    all_attn = []
    with torch.no_grad():
        for x, y in loader:
            attn = model.get_assignments(x.to(DEVICE)).cpu()
            all_attn.append(attn)
    attn = torch.cat(all_attn, 0).mean(0)  # (C, K) — average assignment per channel

    print("\n  Channel-to-Slot Assignments (averaged):")
    print(f"  {'Sensor':>8s}", end="")
    for k in range(attn.shape[1]):
        print(f"  Slot{k:d}", end="")
    print("  | Dominant | Component")
    print("  " + "-"*80)

    # Build sensor-to-component mapping
    s2c = {}
    for comp, sensors in component_groups.items():
        for s in sensors:
            s2c[s] = comp

    for i, s in enumerate(sensor_names):
        dominant = attn[i].argmax().item()
        comp = s2c.get(s, "?")
        print(f"  {s:>8s}", end="")
        for k in range(attn.shape[1]):
            val = attn[i, k].item()
            marker = " *" if k == dominant else "  "
            print(f"  {val:.3f}{marker}", end="")
        print(f"  | Slot{dominant} | {comp}")

    # Compute alignment score: for each slot, which component has highest total weight?
    print("\n  Slot → Dominant Component:")
    for k in range(attn.shape[1]):
        comp_weights = {}
        for i, s in enumerate(sensor_names):
            comp = s2c.get(s, "?")
            comp_weights[comp] = comp_weights.get(comp, 0) + attn[i, k].item()
        dom_comp = max(comp_weights, key=comp_weights.get)
        dom_weight = comp_weights[dom_comp]
        total = sum(comp_weights.values())
        print(f"    Slot{k}: {dom_comp} ({dom_weight/total*100:.1f}%)")

    return attn


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()
    sc = INFORMATIVE_SENSORS; n = len(sc)

    # Load and normalize
    t1, te1, r1 = load_cmapss("FD001")
    _, te2, r2 = load_cmapss("FD002")
    m_s, s_s = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, te2]: df[sc] = (df[sc] - m_s) / s_s
    ds1 = DS(t1, sc); nt = int(0.8*len(ds1)); nv = len(ds1) - nt

    # ─── Exp 39a: Slot-Concept Transformer with K=5 (matching physics groups) ───
    print("\n" + "="*70)
    print("Exp 39a: Slot-Concept Transformer (K=5 slots, matching 5 components)")
    print("="*70)
    sys.stdout.flush()

    results_slot = []; results_role = []; results_ci = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        # Slot-Concept
        torch.manual_seed(seed)
        m_slot = SlotConceptTransformer(n, n_slots=5)
        p_count = sum(p.numel() for p in m_slot.parameters())
        m_slot, _ = train_m(m_slot, tl, vl)
        r1s = ev(m_slot, te1l); r2s = ev(m_slot, te2l)
        print(f"  [{seed}] Slot(5)    | params={p_count:,} | FD001={r1s:.2f} | FD002={r2s:.2f} | ratio={r2s/r1s:.2f}")
        results_slot.append((r1s, r2s))

        # Analyze assignments for first seed
        if seed == 42:
            analyze_assignments(m_slot, te1l, sc, COMPONENT_GROUPS)

        # Role-Trans baseline
        torch.manual_seed(seed)
        m_role = RoleTrans(COMPONENT_GROUPS, sc)
        m_role, _ = train_m(m_role, tl, vl)
        r1r = ev(m_role, te1l); r2r = ev(m_role, te2l)
        print(f"  [{seed}] Role-Trans | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")
        results_role.append((r1r, r2r))

        # CI-Trans baseline
        torch.manual_seed(seed)
        m_ci = CITrans(n)
        m_ci, _ = train_m(m_ci, tl, vl)
        r1c = ev(m_ci, te1l); r2c = ev(m_ci, te2l)
        print(f"  [{seed}] CI-Trans   | FD001={r1c:.2f} | FD002={r2c:.2f} | ratio={r2c/r1c:.2f}")
        results_ci.append((r1c, r2c))
        sys.stdout.flush()

    for name, res in [("Slot(5)", results_slot), ("Role-Trans", results_role), ("CI-Trans", results_ci)]:
        arr = np.array(res)
        print(f"\n  {name:12s} avg: FD001={arr[:,0].mean():.2f}±{arr[:,0].std():.2f} | FD002={arr[:,1].mean():.2f}±{arr[:,1].std():.2f} | ratio={arr[:,1].mean()/arr[:,0].mean():.2f}")

    # ─── Exp 39b: Ablation — number of slots ───
    print("\n" + "="*70)
    print("Exp 39b: Number of slots ablation (K=3, 5, 7, 10, 14)")
    print("="*70)
    sys.stdout.flush()

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
    vl = DataLoader(vs, 256, num_workers=0)
    te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
    te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

    for k in [3, 5, 7, 10, 14]:
        torch.manual_seed(seed)
        m = SlotConceptTransformer(n, n_slots=k)
        p_count = sum(p.numel() for p in m.parameters())
        m, _ = train_m(m, tl, vl)
        r1k = ev(m, te1l); r2k = ev(m, te2l)
        print(f"  K={k:2d} | params={p_count:,} | FD001={r1k:.2f} | FD002={r2k:.2f} | ratio={r2k/r1k:.2f}")
        sys.stdout.flush()

    # ─── Exp 39c: Slot attention iterations ablation ───
    print("\n" + "="*70)
    print("Exp 39c: Slot attention iterations ablation")
    print("="*70)
    sys.stdout.flush()

    for iters in [1, 3, 5, 7]:
        torch.manual_seed(seed)
        m = SlotConceptTransformer(n, n_slots=5, slot_iters=iters)
        m, _ = train_m(m, tl, vl)
        r1i = ev(m, te1l); r2i = ev(m, te2l)
        print(f"  iters={iters} | FD001={r1i:.2f} | FD002={r2i:.2f} | ratio={r2i/r1i:.2f}")
        sys.stdout.flush()

    print(f"\n\nSlot-Concept experiments complete! Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()

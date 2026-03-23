"""
Slot-Concept v2: Entropy regularization for differentiated slot assignments
Exp 40: Add entropy loss to force channels to specialize into distinct slots
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

class SlotAttentionV2(nn.Module):
    """Slot attention with entropy tracking."""
    def __init__(self, n_slots, d, n_iters=3, hidden_dim=64):
        super().__init__()
        self.n_slots = n_slots; self.d = d; self.n_iters = n_iters
        self.slots_mu = nn.Parameter(torch.randn(1, n_slots, d) * (d ** -0.5))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, n_slots, d))
        self.to_q = nn.Linear(d, d); self.to_k = nn.Linear(d, d); self.to_v = nn.Linear(d, d)
        self.gru = nn.GRUCell(d, d)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, d))
        self.norm_slots = nn.LayerNorm(d); self.norm_inputs = nn.LayerNorm(d)

    def forward(self, inputs, return_attn=False):
        B, N, D = inputs.shape
        slots = self.slots_mu + self.slots_log_sigma.exp() * torch.randn_like(self.slots_mu)
        slots = slots.expand(B, -1, -1).clone()
        inputs = self.norm_inputs(inputs)
        k = self.to_k(inputs); v = self.to_v(inputs)
        attn_weights = None
        for _ in range(self.n_iters):
            q = self.to_q(self.norm_slots(slots))
            attn_logits = torch.bmm(k, q.transpose(1, 2)) / (D ** 0.5)
            attn_weights = F.softmax(attn_logits, dim=-1)  # (B, N, K)
            attn_norm = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            slot_attn = attn_weights / attn_norm
            updates = torch.bmm(slot_attn.transpose(1, 2), v)
            slots = self.gru(updates.reshape(B*self.n_slots,D), slots.reshape(B*self.n_slots,D)).reshape(B,self.n_slots,D)
            slots = slots + self.mlp(slots)
        if return_attn: return slots, attn_weights
        return slots, attn_weights  # Always return attn for entropy loss

class SlotConceptV2(nn.Module):
    def __init__(self, n_channels, n_slots=5, d=32, heads=4, layers=2, slot_iters=3, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels; self.n_slots = n_slots
        self.proj = nn.Linear(1, d); self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.temporal_enc = nn.TransformerEncoder(enc, layers)
        self.ch_pool = nn.Linear(d, d)
        self.slot_attn = SlotAttentionV2(n_slots, d, n_iters=slot_iters, hidden_dim=d*2)
        self.slot_pos = nn.Parameter(torch.randn(1, n_slots, d) * 0.02)
        cross_enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross_slot = nn.TransformerEncoder(cross_enc, 1)
        self.head = nn.Linear(d * n_slots, 1)

    def encode_channels(self, x):
        B, T, C = x.shape
        cx = x.permute(0, 2, 1).reshape(B * C, T, 1)
        cx = self.proj(cx) + self.pos[:, :T]
        cx = self.temporal_enc(cx)[:, -1].reshape(B, C, -1)
        return self.ch_pool(cx)

    def forward(self, x):
        ch_features = self.encode_channels(x)
        slots, attn = self.slot_attn(ch_features)
        slots = self.cross_slot(slots + self.slot_pos)
        pred = self.head(slots.reshape(slots.shape[0], -1))
        return pred, attn

    def get_assignments(self, x):
        ch_features = self.encode_channels(x)
        slots, attn = self.slot_attn(ch_features)
        return attn

def entropy_reg(attn, target='both'):
    """
    Entropy regularization for slot assignments.
    attn: (B, N, K) — channel-to-slot soft assignment

    Two objectives:
    1. Per-channel entropy should be LOW (each channel assigned to one slot) → minimize H(attn[b,n,:])
    2. Per-slot marginal should be UNIFORM (slots should be equally used) → maximize H(mean(attn[:,n,:], dim=n))
    """
    B, N, K = attn.shape

    # 1. Per-channel: minimize entropy (sharpen assignments)
    per_channel = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()  # H over K dimension
    # Minimum is 0 (one-hot), maximum is log(K) (uniform)

    # 2. Per-slot marginal: maximize entropy (balance slots)
    slot_marginal = attn.mean(dim=1)  # (B, K) — average assignment mass per slot
    per_slot = -(slot_marginal * (slot_marginal + 1e-8).log()).sum(dim=-1).mean()
    # Want this to be high (log(K) = uniform)

    if target == 'both':
        # Minimize per-channel entropy, maximize per-slot entropy
        return per_channel - per_slot
    elif target == 'sharpen':
        return per_channel
    else:
        return -per_slot

def train_with_entropy(model, tl, vl, epochs=60, lr=1e-3, patience=12, entropy_weight=0.1):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv = 1e9; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred, attn = model(x)
            rul_loss = F.mse_loss(pred, y)
            ent_loss = entropy_reg(attn, 'both')
            loss = rul_loss + entropy_weight * ent_loss
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
        for x, y in loader:
            out = model(x.to(DEVICE))
            pred = out[0] if isinstance(out, tuple) else out
            ps.append(pred.cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()

def analyze_assignments(model, loader, sensor_names, component_groups):
    model.eval(); all_attn = []
    with torch.no_grad():
        for x, y in loader:
            attn = model.get_assignments(x.to(DEVICE)).cpu()
            all_attn.append(attn)
    attn = torch.cat(all_attn, 0).mean(0)  # (C, K)

    s2c = {}
    for comp, sensors in component_groups.items():
        for s in sensors: s2c[s] = comp

    print(f"\n  Channel-to-Slot Assignments:")
    print(f"  {'Sensor':>8s}", end="")
    for k in range(attn.shape[1]): print(f"  Slot{k:d}", end="")
    print("  | Dom | Component")

    for i, s in enumerate(sensor_names):
        dominant = attn[i].argmax().item()
        comp = s2c.get(s, "?")
        print(f"  {s:>8s}", end="")
        for k in range(attn.shape[1]):
            val = attn[i, k].item()
            marker = " *" if k == dominant else "  "
            print(f"  {val:.3f}{marker}", end="")
        print(f"  | S{dominant} | {comp}")

    # Compute alignment: for each slot, check if it maps to one component
    print("\n  Slot → Component Mapping:")
    slot_to_comp = {}
    for k in range(attn.shape[1]):
        comp_weights = {}
        for i, s in enumerate(sensor_names):
            comp = s2c.get(s, "?")
            comp_weights[comp] = comp_weights.get(comp, 0) + attn[i, k].item()
        dom_comp = max(comp_weights, key=comp_weights.get)
        dom_pct = comp_weights[dom_comp] / sum(comp_weights.values()) * 100
        slot_to_comp[k] = (dom_comp, dom_pct)
        print(f"    Slot{k}: {dom_comp} ({dom_pct:.1f}%)")

    # Check if slots are differentiated
    dominant_comps = [v[0] for v in slot_to_comp.values()]
    unique_comps = len(set(dominant_comps))
    print(f"\n  Unique components discovered: {unique_comps}/{attn.shape[1]}")
    return attn


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()
    sc = INFORMATIVE_SENSORS; n = len(sc)

    t1, te1, r1 = load_cmapss("FD001"); _, te2, r2 = load_cmapss("FD002")
    m_s, s_s = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, te2]: df[sc] = (df[sc] - m_s) / s_s
    ds1 = DS(t1, sc); nt = int(0.8*len(ds1)); nv = len(ds1) - nt

    # ─── Exp 40a: Entropy weight sweep ───
    print("\n" + "="*70)
    print("Exp 40a: Entropy regularization weight sweep")
    print("="*70)
    sys.stdout.flush()

    seed = 42
    torch.manual_seed(seed); np.random.seed(seed)
    ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
    tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
    vl = DataLoader(vs, 256, num_workers=0)
    te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
    te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

    for ew in [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]:
        torch.manual_seed(seed)
        m = SlotConceptV2(n, n_slots=5)
        m, _ = train_with_entropy(m, tl, vl, entropy_weight=ew)
        r1e = ev(m, te1l); r2e = ev(m, te2l)
        # Check assignment entropy
        attn = []
        m.eval()
        with torch.no_grad():
            for x, y in te1l:
                a = m.get_assignments(x.to(DEVICE)).cpu()
                attn.append(a)
        attn = torch.cat(attn, 0).mean(0)
        avg_ent = -(attn * (attn + 1e-8).log()).sum(-1).mean().item()
        max_ent = np.log(5)
        print(f"  λ={ew:4.2f} | FD001={r1e:.2f} | FD002={r2e:.2f} | ratio={r2e/r1e:.2f} | assignment_entropy={avg_ent:.3f}/{max_ent:.3f}")

        if ew == 1.0:
            print(f"\n  --- Analyzing λ=1.0 assignments ---")
            analyze_assignments(m, te1l, sc, COMPONENT_GROUPS)
        sys.stdout.flush()

    # ─── Exp 40b: Best entropy weight with 3 seeds ───
    print("\n" + "="*70)
    print("Exp 40b: Best config with 3 seeds")
    print("="*70)
    sys.stdout.flush()

    # Run with best entropy weight from above
    best_ew = 1.0  # will adjust based on sweep
    results = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        torch.manual_seed(seed)
        m = SlotConceptV2(n, n_slots=5)
        m, _ = train_with_entropy(m, tl, vl, entropy_weight=best_ew)
        r1e = ev(m, te1l); r2e = ev(m, te2l)
        print(f"  [{seed}] λ={best_ew} | FD001={r1e:.2f} | FD002={r2e:.2f} | ratio={r2e/r1e:.2f}")
        results.append((r1e, r2e))

        if seed == 42:
            analyze_assignments(m, te1l, sc, COMPONENT_GROUPS)
        sys.stdout.flush()

    arr = np.array(results)
    print(f"\n  Avg: FD001={arr[:,0].mean():.2f}±{arr[:,0].std():.2f} | FD002={arr[:,1].mean():.2f}±{arr[:,1].std():.2f} | ratio={arr[:,1].mean()/arr[:,0].mean():.2f}")

    print(f"\n\nSlot-Concept v2 complete! Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()

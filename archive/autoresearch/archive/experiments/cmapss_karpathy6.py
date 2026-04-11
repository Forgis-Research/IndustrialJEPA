"""
Karpathy Loop Round 6 — Fix JEPA: temporal masking, contrastive, encoder freezing
Exp 31: Contrastive pretraining (same engine, different timesteps = positive pairs)
Exp 32: Encoder freezing (freeze encoder from FD001, train head on FD002)
Exp 33: Temporal JEPA (mask future time patches, predict from past — NOT component masking)
"""
import sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from copy import deepcopy

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

class PairDS(Dataset):
    """Returns pairs of windows from the same engine for contrastive learning."""
    def __init__(self, df, sc, sl=SEQ_LEN, gap_min=5, gap_max=30):
        self.pairs = []
        for u in df["unit"].unique():
            d = df[df["unit"]==u]; v = d[sc].values
            n = len(v)
            if n < sl + gap_min: continue
            # Sample pairs: window at position i and i+gap
            for i in range(n - sl - gap_min):
                max_gap = min(gap_max, n - sl - i)
                if max_gap <= gap_min: continue
                gap = np.random.randint(gap_min, max_gap)
                self.pairs.append((v[i:i+sl], v[i+gap:i+gap+sl]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        a, b = self.pairs[i]
        return torch.FloatTensor(a), torch.FloatTensor(b)

# ─── Role Encoder (same as round 5) ───
class RoleEncoder(nn.Module):
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n_comp = len(groups)
        self.comp_idx = {c: [sc.index(s) for s in ss] for c, ss in groups.items()}
        self.comp_names = list(groups.keys())
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.within = nn.TransformerEncoder(enc, layers)
        self.pool = nn.Linear(d, d)
        self.d = d

    def forward(self, x):
        B, T, C = x.shape; cs = []
        for c in self.comp_names:
            idx = self.comp_idx[c]
            n = len(idx)
            cx = x[:, :, idx].permute(0, 2, 1).reshape(B*n, T, 1)
            cx = self.proj(cx) + self.pos[:, :T]
            cx = self.within(cx)[:, -1].reshape(B, n, -1)
            cs.append(self.pool(cx.mean(1)))
        return torch.stack(cs, 1)

class CrossComponentBlock(nn.Module):
    def __init__(self, n_comp, d=32, heads=4, dropout=0.1):
        super().__init__()
        self.cemb = nn.Parameter(torch.randn(1, n_comp, d) * 0.02)
        cross = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross = nn.TransformerEncoder(cross, 1)
        self.fc = nn.Linear(d * n_comp, 1)
    def forward(self, comp_repr):
        s = comp_repr + self.cemb; s = self.cross(s)
        return self.fc(s.reshape(s.shape[0], -1))

class RoleTransV2(nn.Module):
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.encoder = RoleEncoder(groups, sc, d, heads, layers, dropout)
        self.head = CrossComponentBlock(len(groups), d, heads, dropout)
    def forward(self, x):
        return self.head(self.encoder(x))

class CIEncoder(nn.Module):
    """CI encoder with extractable features."""
    def __init__(self, n, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.n = n; self.d = d
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B*C, T, 1)
        x = self.proj(x) + self.pos[:, :T]
        x = self.enc(x)[:, -1].reshape(B, C, -1)  # (B, C, d)
        return x

class CITransV2(nn.Module):
    def __init__(self, n, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.encoder = CIEncoder(n, d, heads, layers, dropout)
        self.ch = nn.Linear(d, 1)
        self.out = nn.Linear(n, 1)
    def forward(self, x):
        feat = self.encoder(x)  # (B, C, d)
        x = self.ch(feat).squeeze(-1)  # (B, C)
        return self.out(x)

def train_m(model, tl, vl, epochs=60, lr=1e-3, patience=12):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv = 1e9
    bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # init with current state
    ni = 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            if torch.isnan(loss):
                continue
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = ev(model, vl)
        if not np.isnan(v) and v < bv:
            bv = v; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else: ni += 1
        if ni >= patience: break
    model.load_state_dict(bs); model.to(DEVICE)
    return model, bv

def ev(model, loader):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x, y in loader: ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()

# ─── Contrastive Pretraining ───
class ContrastiveHead(nn.Module):
    """Projects component representations to contrastive space."""
    def __init__(self, n_comp, d=32, proj_d=32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_comp * d, proj_d),
            nn.ReLU(),
            nn.Linear(proj_d, proj_d)
        )
    def forward(self, comp_repr):
        return F.normalize(self.proj(comp_repr.reshape(comp_repr.shape[0], -1)), dim=-1)

def contrastive_loss(z1, z2, temperature=0.5):
    """NT-Xent loss: positive pairs are (z1[i], z2[i]), negatives are all others."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, d)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    # Mask out self-similarity
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    # Clamp for numerical stability
    sim = sim.clamp(-50, 50)
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)

def pretrain_contrastive(encoder, proj_head, pair_loader, epochs=30, lr=5e-4):
    encoder = encoder.to(DEVICE); proj_head = proj_head.to(DEVICE)
    params = list(encoder.parameters()) + list(proj_head.parameters())
    opt = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        encoder.train(); proj_head.train()
        total_loss = 0; n_batch = 0
        for x1, x2 in pair_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            z1 = proj_head(encoder(x1))
            z2 = proj_head(encoder(x2))
            loss = contrastive_loss(z1, z2)
            if torch.isnan(loss):
                continue  # skip NaN batches
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            total_loss += loss.item(); n_batch += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            avg = total_loss/max(n_batch,1)
            print(f"    Contrastive ep {ep+1}/{epochs}: loss={avg:.4f}")
            sys.stdout.flush()
    return encoder

# ─── Temporal JEPA ───
class TemporalJEPA(nn.Module):
    """
    JEPA with temporal masking: mask the last K timesteps, predict component
    representations at masked timesteps from the context timesteps.
    Uses a temporal predictor instead of component predictor.
    """
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1, mask_ratio=0.3):
        super().__init__()
        self.comp_idx = {c: [sc.index(s) for s in ss] for c, ss in groups.items()}
        self.comp_names = list(groups.keys())
        self.n_comp = len(groups)
        self.d = d
        self.mask_ratio = mask_ratio

        # Per-sensor temporal encoder (shared across sensors)
        self.proj = nn.Linear(1, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.online_enc = nn.TransformerEncoder(enc, layers)
        self.target_enc = deepcopy(self.online_enc)
        for p in self.target_enc.parameters():
            p.requires_grad = False
        self.target_proj = deepcopy(self.proj)
        for p in self.target_proj.parameters():
            p.requires_grad = False
        self.target_pos = nn.Parameter(self.pos.data.clone())
        self.target_pos.requires_grad = False

        # Predictor: takes context token sequence, predicts masked token representations
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        pred_enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.predictor = nn.TransformerEncoder(pred_enc, 1)

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for op, tp in zip(self.online_enc.parameters(), self.target_enc.parameters()):
            tp.data = momentum * tp.data + (1 - momentum) * op.data
        for op, tp in zip(self.proj.parameters(), self.target_proj.parameters()):
            tp.data = momentum * tp.data + (1 - momentum) * op.data

    def forward(self, x):
        B, T, C = x.shape
        n_mask = max(1, int(T * self.mask_ratio))
        ctx_len = T - n_mask

        total_loss = 0
        n_sensors = 0

        # Process each sensor independently (channel-independent temporal JEPA)
        for c in self.comp_names:
            for si in self.comp_idx[c]:
                sx = x[:, :, si:si+1]  # (B, T, 1)

                # Target: full sequence encoded
                with torch.no_grad():
                    t_enc = self.target_proj(sx) + self.target_pos[:, :T]
                    t_out = self.target_enc(t_enc)  # (B, T, d)
                    target_masked = t_out[:, ctx_len:, :]  # (B, n_mask, d)

                # Online: context only + mask tokens
                o_enc = self.proj(sx[:, :ctx_len]) + self.pos[:, :ctx_len]
                o_ctx = self.online_enc(o_enc)  # (B, ctx_len, d)
                mask_tokens = self.mask_token.expand(B, n_mask, -1) + self.pos[:, ctx_len:ctx_len+n_mask]
                pred_input = torch.cat([o_ctx, mask_tokens], dim=1)  # (B, T, d)
                pred_out = self.predictor(pred_input)
                pred_masked = pred_out[:, ctx_len:, :]  # (B, n_mask, d)

                total_loss += F.mse_loss(pred_masked, target_masked)
                n_sensors += 1

        return total_loss / n_sensors

    def get_encoder_state(self):
        """Return online encoder state dict for transfer."""
        return {
            'proj': self.proj.state_dict(),
            'pos': self.pos.data,
            'enc': self.online_enc.state_dict(),
        }

def pretrain_temporal_jepa(tjepa, train_loader, epochs=30, lr=1e-3):
    tjepa = tjepa.to(DEVICE)
    opt = torch.optim.Adam(
        [p for n, p in tjepa.named_parameters() if 'target' not in n],
        lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        tjepa.train()
        total_loss = 0; n_batch = 0
        mom = 0.996 + (0.999 - 0.996) * ep / max(epochs - 1, 1)
        for x in train_loader:
            x = x.to(DEVICE)
            loss = tjepa(x)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                [p for n, p in tjepa.named_parameters() if 'target' not in n], 1.0
            )
            opt.step()
            tjepa.update_target(mom)
            total_loss += loss.item(); n_batch += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    Temporal JEPA ep {ep+1}/{epochs}: loss={total_loss/n_batch:.4f}")
            sys.stdout.flush()
    return tjepa

class UnlabeledDS(Dataset):
    def __init__(self, df, sc, sl=SEQ_LEN):
        self.s = []
        for u in df["unit"].unique():
            d = df[df["unit"]==u]; v = d[sc].values
            for i in range(len(v)-sl+1): self.s.append(v[i:i+sl])
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return torch.FloatTensor(self.s[i])


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

    # Datasets
    ds1 = DS(t1, sc); nt = int(0.8*len(ds1)); nv = len(ds1) - nt
    unlabeled_df = pd.concat([t1[["unit","cycle"]+sc].assign(unit=lambda x: x["unit"]),
                              t2[["unit","cycle"]+sc].assign(unit=lambda x: x["unit"]+1000)],
                             ignore_index=True)
    ul_ds = UnlabeledDS(unlabeled_df, sc)
    ul_loader = DataLoader(ul_ds, 256, shuffle=True, num_workers=0)

    # ─── Exp 31: Contrastive Pretraining ───
    print("\n" + "="*70)
    print("Exp 31: Contrastive pretraining (same-engine pairs)")
    print("="*70)
    sys.stdout.flush()

    # Build pair dataset from FD001+FD002
    pair_ds = PairDS(unlabeled_df, sc, gap_min=5, gap_max=30)
    print(f"  Positive pairs: {len(pair_ds)}")
    pair_loader = DataLoader(pair_ds, 256, shuffle=True, num_workers=0)

    results_contr = []
    results_scratch = []

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n  --- Seed {seed} ---")

        # Contrastive pretrain
        enc = RoleEncoder(COMPONENT_GROUPS, sc)
        proj = ContrastiveHead(enc.n_comp, enc.d)
        print("  Contrastive pretraining (30 epochs)...")
        sys.stdout.flush()
        enc = pretrain_contrastive(enc, proj, pair_loader, epochs=30)

        # Build model with pretrained encoder
        model_c = RoleTransV2(COMPONENT_GROUPS, sc).to(DEVICE)
        model_c.encoder.load_state_dict(enc.state_dict())

        torch.manual_seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        model_c, _ = train_m(model_c, tl, vl)
        r1c = ev(model_c, te1l); r2c = ev(model_c, te2l)
        print(f"  Contrastive+FT | FD001={r1c:.2f} | FD002={r2c:.2f} | ratio={r2c/r1c:.2f}")
        results_contr.append((r1c, r2c))

        # Scratch baseline
        torch.manual_seed(seed)
        model_s = RoleTransV2(COMPONENT_GROUPS, sc)
        model_s, _ = train_m(model_s, tl, vl)
        r1s = ev(model_s, te1l); r2s = ev(model_s, te2l)
        print(f"  Scratch        | FD001={r1s:.2f} | FD002={r2s:.2f} | ratio={r2s/r1s:.2f}")
        results_scratch.append((r1s, r2s))
        sys.stdout.flush()

    cc = np.array(results_contr); ss = np.array(results_scratch)
    print(f"\n  Contrastive avg: FD001={cc[:,0].mean():.2f}±{cc[:,0].std():.2f} | FD002={cc[:,1].mean():.2f}±{cc[:,1].std():.2f} | ratio={cc[:,1].mean()/cc[:,0].mean():.2f}")
    print(f"  Scratch avg:     FD001={ss[:,0].mean():.2f}±{ss[:,0].std():.2f} | FD002={ss[:,1].mean():.2f}±{ss[:,1].std():.2f} | ratio={ss[:,1].mean()/ss[:,0].mean():.2f}")
    sys.stdout.flush()

    # ─── Exp 32: Encoder Freezing ───
    print("\n" + "="*70)
    print("Exp 32: Encoder freezing for few-shot transfer")
    print("="*70)
    sys.stdout.flush()

    # Train on FD001, freeze encoder, fine-tune head on FD002 subsets
    ds2 = DS(t2, sc)  # FD002 labeled
    ds2_all = list(range(len(ds2)))

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n  --- Seed {seed} ---")

        # Train full model on FD001
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        # Role-Trans
        torch.manual_seed(seed)
        role_m = RoleTransV2(COMPONENT_GROUPS, sc)
        role_m, _ = train_m(role_m, tl, vl)
        r2_zero = ev(role_m, te2l)
        print(f"  Role-Trans zero-shot FD002: {r2_zero:.2f}")

        # CI-Trans
        torch.manual_seed(seed)
        ci_m = CITransV2(n)
        ci_m, _ = train_m(ci_m, tl, vl)
        r2_ci_zero = ev(ci_m, te2l)
        print(f"  CI-Trans zero-shot FD002:   {r2_ci_zero:.2f}")

        # Freeze encoder, fine-tune head on small FD002 subsets
        for frac in [0.01, 0.05, 0.10]:
            n_ft = max(1, int(len(ds2) * frac))
            ft_idx = torch.randperm(len(ds2), generator=torch.Generator().manual_seed(seed))[:n_ft]
            ft_sub = torch.utils.data.Subset(ds2, ft_idx.tolist())
            ft_loader = DataLoader(ft_sub, min(256, n_ft), shuffle=True, num_workers=0)

            # Role: freeze encoder
            role_ft = deepcopy(role_m)
            for p in role_ft.encoder.parameters(): p.requires_grad = False
            opt = torch.optim.Adam(role_ft.head.parameters(), lr=1e-3, weight_decay=1e-4)
            role_ft.to(DEVICE)
            for ep in range(30):
                role_ft.train()
                for x, y in ft_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    loss = F.mse_loss(role_ft(x), y)
                    opt.zero_grad(); loss.backward(); opt.step()
            r2_role_ft = ev(role_ft, te2l)

            # CI: freeze encoder
            ci_ft = deepcopy(ci_m)
            for p in ci_ft.encoder.parameters(): p.requires_grad = False
            opt = torch.optim.Adam([ci_ft.ch.weight, ci_ft.ch.bias, ci_ft.out.weight, ci_ft.out.bias],
                                   lr=1e-3, weight_decay=1e-4)
            ci_ft.to(DEVICE)
            for ep in range(30):
                ci_ft.train()
                for x, y in ft_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    loss = F.mse_loss(ci_ft(x), y)
                    opt.zero_grad(); loss.backward(); opt.step()
            r2_ci_ft = ev(ci_ft, te2l)

            print(f"  {frac*100:.0f}% FD002 frozen-enc: Role={r2_role_ft:.2f} CI={r2_ci_ft:.2f}")
        sys.stdout.flush()

    # ─── Exp 33: Temporal JEPA ───
    print("\n" + "="*70)
    print("Exp 33: Temporal JEPA (mask future timesteps, predict from past)")
    print("="*70)
    sys.stdout.flush()

    results_tjepa = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n  --- Seed {seed} ---")

        # Temporal JEPA pretraining
        tjepa = TemporalJEPA(COMPONENT_GROUPS, sc, mask_ratio=0.3)
        print("  Temporal JEPA pretraining (30 epochs)...")
        sys.stdout.flush()
        tjepa = pretrain_temporal_jepa(tjepa, ul_loader, epochs=30)

        # Transfer encoder weights to RoleTransV2
        model_tj = RoleTransV2(COMPONENT_GROUPS, sc).to(DEVICE)
        # Copy the shared within-encoder weights from temporal JEPA
        enc_state = tjepa.get_encoder_state()
        model_tj.encoder.proj.load_state_dict(enc_state['proj'])
        model_tj.encoder.pos.data.copy_(enc_state['pos'])
        model_tj.encoder.within.load_state_dict(enc_state['enc'])

        torch.manual_seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        model_tj, _ = train_m(model_tj, tl, vl)
        r1t = ev(model_tj, te1l); r2t = ev(model_tj, te2l)
        print(f"  TemporalJEPA+FT | FD001={r1t:.2f} | FD002={r2t:.2f} | ratio={r2t/r1t:.2f}")
        results_tjepa.append((r1t, r2t))
        sys.stdout.flush()

    tj = np.array(results_tjepa)
    print(f"\n  TemporalJEPA avg: FD001={tj[:,0].mean():.2f}±{tj[:,0].std():.2f} | FD002={tj[:,1].mean():.2f}±{tj[:,1].std():.2f} | ratio={tj[:,1].mean()/tj[:,0].mean():.2f}")
    print(f"  (Compare scratch: FD001={ss[:,0].mean():.2f}±{ss[:,0].std():.2f} | FD002={ss[:,1].mean():.2f}±{ss[:,1].std():.2f} | ratio={ss[:,1].mean()/ss[:,0].mean():.2f})")

    print(f"\n\nRound 6 complete! Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()

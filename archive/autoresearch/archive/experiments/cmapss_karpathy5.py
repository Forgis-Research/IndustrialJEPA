"""
Karpathy Loop Round 5 — JEPA pretraining + domain adaptation
Exp 28: JEPA pretraining on FD001+FD002 (unlabeled), fine-tune on FD001, transfer to FD002
Exp 29: MMD domain adaptation during fine-tuning
Exp 30: Patch-based Role-Transformer
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

class UnlabeledDS(Dataset):
    """Sliding windows of sensor data, no labels. For JEPA pretraining."""
    def __init__(self, df, sc, sl=SEQ_LEN):
        self.s = []
        for u in df["unit"].unique():
            d = df[df["unit"]==u]; v = d[sc].values
            for i in range(len(v)-sl+1):
                self.s.append(v[i:i+sl])
    def __len__(self): return len(self.s)
    def __getitem__(self, i): return torch.FloatTensor(self.s[i])

class TDS(Dataset):
    def __init__(self,df,sc,rm,sl=SEQ_LEN):
        self.s=[]
        for u in df["unit"].unique():
            d=df[df["unit"]==u]; v=d[sc].values
            if len(v)<sl: v=np.vstack([np.tile(v[0],(sl-len(v),1)),v])
            self.s.append((v[-sl:],min(rm[u],MAX_RUL)))
    def __len__(self): return len(self.s)
    def __getitem__(self,i): x,y=self.s[i]; return torch.FloatTensor(x),torch.FloatTensor([y])

# ─── Role-Transformer with extractable encoder ───
class RoleEncoder(nn.Module):
    """Encodes sensor windows into component representations. Shared for JEPA + RUL."""
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
        """Returns component representations: (B, n_comp, d)"""
        B, T, C = x.shape
        cs = []
        for c in self.comp_names:
            idx = self.comp_idx[c]
            n = len(idx)
            cx = x[:, :, idx].permute(0, 2, 1).reshape(B*n, T, 1)
            cx = self.proj(cx) + self.pos[:, :T]
            cx = self.within(cx)[:, -1].reshape(B, n, -1)
            cs.append(self.pool(cx.mean(1)))
        return torch.stack(cs, 1)  # (B, n_comp, d)

class CrossComponentBlock(nn.Module):
    """Cross-component attention + RUL head."""
    def __init__(self, n_comp, d=32, heads=4, dropout=0.1):
        super().__init__()
        self.cemb = nn.Parameter(torch.randn(1, n_comp, d) * 0.02)
        cross = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
        self.cross = nn.TransformerEncoder(cross, 1)
        self.fc = nn.Linear(d * n_comp, 1)

    def forward(self, comp_repr):
        """comp_repr: (B, n_comp, d) -> RUL prediction (B, 1)"""
        s = comp_repr + self.cemb
        s = self.cross(s)
        return self.fc(s.reshape(s.shape[0], -1))

class RoleTransV2(nn.Module):
    """Role-Transformer with separable encoder for JEPA pretraining."""
    def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.encoder = RoleEncoder(groups, sc, d, heads, layers, dropout)
        self.head = CrossComponentBlock(len(groups), d, heads, dropout)

    def forward(self, x):
        comp = self.encoder(x)
        return self.head(comp)

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

# ─── JEPA Pretraining Module ───
class JEPAPretrainer(nn.Module):
    """
    JEPA for component-level representations.
    Context: unmasked components. Target: masked components (via EMA encoder).
    Predictor: small transformer that predicts masked component representations.
    """
    def __init__(self, encoder, n_comp, d=32, pred_d=16, pred_layers=2, mask_ratio=0.4):
        super().__init__()
        self.online_encoder = encoder
        self.target_encoder = deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        self.n_comp = n_comp
        self.mask_ratio = mask_ratio
        # Predictor: maps context component representations to predictions for masked components
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pred_proj_in = nn.Linear(d, pred_d)
        pred_enc = nn.TransformerEncoderLayer(pred_d, 4, pred_d*4, 0.1, batch_first=True)
        self.predictor = nn.TransformerEncoder(pred_enc, pred_layers)
        self.pred_proj_out = nn.Linear(pred_d, d)
        self.comp_pos = nn.Parameter(torch.randn(1, n_comp, pred_d) * 0.02)

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        for op, tp in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            tp.data = momentum * tp.data + (1 - momentum) * op.data

    def forward(self, x):
        B = x.shape[0]
        n_mask = max(1, int(self.n_comp * self.mask_ratio))

        # Generate random mask per sample
        mask_idx = torch.stack([torch.randperm(self.n_comp)[:n_mask] for _ in range(B)])  # (B, n_mask)

        # Target representations (from EMA encoder, no grad)
        with torch.no_grad():
            target_repr = self.target_encoder(x)  # (B, n_comp, d)

        # Online encoder representations
        online_repr = self.online_encoder(x)  # (B, n_comp, d)

        # Build predictor input: replace masked components with mask tokens
        pred_input = online_repr.clone()
        for b in range(B):
            pred_input[b, mask_idx[b]] = self.mask_token.squeeze(0)

        # Run predictor
        pred_input = self.pred_proj_in(pred_input) + self.comp_pos
        pred_output = self.predictor(pred_input)
        pred_output = self.pred_proj_out(pred_output)  # (B, n_comp, d)

        # Loss: MSE between predicted and target for masked components only
        loss = 0.0
        for b in range(B):
            pred_masked = pred_output[b, mask_idx[b]]  # (n_mask, d)
            target_masked = target_repr[b, mask_idx[b]]  # (n_mask, d)
            loss += F.mse_loss(pred_masked, target_masked)
        loss /= B

        return loss

# ─── Training helpers ───
def train_m(model, tl, vl, epochs=60, lr=1e-3, patience=12):
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bv, bs, ni = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.mse_loss(model(x), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        v = ev(model, vl)
        if v < bv:
            bv = v; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
        else:
            ni += 1
        if ni >= patience: break
    model.load_state_dict(bs); model.to(DEVICE)
    return model, bv

def ev(model, loader):
    model.eval(); ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(DEVICE)).cpu()); ts.append(y)
    return torch.sqrt(F.mse_loss(torch.cat(ps), torch.cat(ts))).item()

def pretrain_jepa(jepa, train_loader, epochs=30, lr=1e-3, momentum_start=0.996, momentum_end=0.999):
    """JEPA pretraining loop."""
    jepa = jepa.to(DEVICE)
    opt = torch.optim.Adam(
        [p for n, p in jepa.named_parameters() if 'target_encoder' not in n],
        lr=lr, weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    for ep in range(epochs):
        jepa.train()
        total_loss = 0; n_batch = 0
        mom = momentum_start + (momentum_end - momentum_start) * ep / max(epochs - 1, 1)
        for x in train_loader:
            x = x.to(DEVICE)
            loss = jepa(x)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(jepa.online_encoder.parameters(), 1.0)
            opt.step()
            jepa.update_target(mom)
            total_loss += loss.item(); n_batch += 1
        sched.step()
        if (ep + 1) % 10 == 0:
            print(f"    JEPA pretrain ep {ep+1}/{epochs}: loss={total_loss/n_batch:.4f}")
            sys.stdout.flush()

    return jepa

# ─── MMD Loss ───
def mmd_loss(source_feat, target_feat, kernel='rbf'):
    """Maximum Mean Discrepancy between source and target feature distributions."""
    n_s = source_feat.shape[0]
    n_t = target_feat.shape[0]
    combined = torch.cat([source_feat, target_feat], dim=0)
    # RBF kernel
    dists = torch.cdist(combined, combined, p=2)
    sigma = dists.median().clamp(min=1e-5)
    K = torch.exp(-dists**2 / (2 * sigma**2))
    K_ss = K[:n_s, :n_s].mean()
    K_tt = K[n_s:, n_s:].mean()
    K_st = K[:n_s, n_s:].mean()
    return K_ss + K_tt - 2 * K_st

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()
    sc = INFORMATIVE_SENSORS; n = len(sc)

    # Load data
    t1, te1, r1 = load_cmapss("FD001")
    t2, te2, r2 = load_cmapss("FD002")

    # Global normalization (fit on FD001 train)
    m_s, s_s = t1[sc].mean(), t1[sc].std().replace(0, 1)
    for df in [t1, te1, t2, te2]:
        df[sc] = (df[sc] - m_s) / s_s

    # ─── Exp 28: JEPA Pretraining ───
    print("\n" + "="*70)
    print("Exp 28: JEPA pretraining on FD001+FD002 (unlabeled) → fine-tune FD001 → transfer FD002")
    print("="*70)
    sys.stdout.flush()

    # Build unlabeled dataset from FD001 + FD002 train data
    unlabeled_df = pd.concat([t1[["unit","cycle"]+sc].assign(unit=lambda x: x["unit"]),
                              t2[["unit","cycle"]+sc].assign(unit=lambda x: x["unit"] + 1000)],
                             ignore_index=True)
    ul_ds = UnlabeledDS(unlabeled_df, sc)
    print(f"  Unlabeled windows: {len(ul_ds)} (FD001+FD002 train)")
    ul_loader = DataLoader(ul_ds, 256, shuffle=True, num_workers=0)

    # Labeled datasets
    ds1 = DS(t1, sc); nt = int(0.8*len(ds1)); nv = len(ds1) - nt

    results_jepa = []
    results_scratch = []

    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        print(f"\n  --- Seed {seed} ---")

        # 1) JEPA pretrain encoder
        encoder_pt = RoleEncoder(COMPONENT_GROUPS, sc)
        n_comp = encoder_pt.n_comp
        jepa = JEPAPretrainer(encoder_pt, n_comp, d=32, pred_d=16, pred_layers=2, mask_ratio=0.4)
        print("  Phase 1: JEPA pretraining (30 epochs)...")
        sys.stdout.flush()
        jepa = pretrain_jepa(jepa, ul_loader, epochs=30)

        # 2) Build RoleTransV2 with pretrained encoder
        model_pt = RoleTransV2(COMPONENT_GROUPS, sc).to(DEVICE)
        # Copy pretrained encoder weights
        model_pt.encoder.load_state_dict(jepa.online_encoder.state_dict())

        # Fine-tune on FD001
        torch.manual_seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        print("  Phase 2: Fine-tune on FD001 (60 epochs)...")
        sys.stdout.flush()
        model_pt, _ = train_m(model_pt, tl, vl)
        r1_pt = ev(model_pt, te1l); r2_pt = ev(model_pt, te2l)
        print(f"  JEPA+FT     | FD001={r1_pt:.2f} | FD002={r2_pt:.2f} | ratio={r2_pt/r1_pt:.2f}")
        results_jepa.append((r1_pt, r2_pt))

        # 3) Train from scratch (baseline)
        torch.manual_seed(seed)
        model_scratch = RoleTransV2(COMPONENT_GROUPS, sc)
        model_scratch, _ = train_m(model_scratch, tl, vl)
        r1_s = ev(model_scratch, te1l); r2_s = ev(model_scratch, te2l)
        print(f"  Scratch     | FD001={r1_s:.2f} | FD002={r2_s:.2f} | ratio={r2_s/r1_s:.2f}")
        results_scratch.append((r1_s, r2_s))
        sys.stdout.flush()

    # Summary
    j = np.array(results_jepa); s = np.array(results_scratch)
    print(f"\n  JEPA+FT  avg: FD001={j[:,0].mean():.2f}±{j[:,0].std():.2f} | FD002={j[:,1].mean():.2f}±{j[:,1].std():.2f} | ratio={j[:,1].mean()/j[:,0].mean():.2f}")
    print(f"  Scratch  avg: FD001={s[:,0].mean():.2f}±{s[:,0].std():.2f} | FD002={s[:,1].mean():.2f}±{s[:,1].std():.2f} | ratio={s[:,1].mean()/s[:,0].mean():.2f}")
    sys.stdout.flush()

    # ─── Exp 29: MMD Domain Adaptation ───
    print("\n" + "="*70)
    print("Exp 29: MMD domain adaptation (FD001 supervised + MMD to FD002)")
    print("="*70)
    sys.stdout.flush()

    # Need unlabeled FD002 windows for MMD
    ul_fd2 = UnlabeledDS(t2, sc)
    ul_fd2_loader = DataLoader(ul_fd2, 256, shuffle=True, num_workers=0)

    results_mmd = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        model = RoleTransV2(COMPONENT_GROUPS, sc).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
        bv, bs_state, ni = 1e9, None, 0
        fd2_iter = iter(ul_fd2_loader)

        for ep in range(60):
            model.train()
            for x, y in tl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                # RUL loss
                pred = model(x)
                rul_loss = F.mse_loss(pred, y)

                # MMD loss on encoder features
                try:
                    x2 = next(fd2_iter).to(DEVICE)
                except StopIteration:
                    fd2_iter = iter(ul_fd2_loader)
                    x2 = next(fd2_iter).to(DEVICE)

                feat_s = model.encoder(x).reshape(x.shape[0], -1)  # (B, n_comp*d)
                feat_t = model.encoder(x2).reshape(x2.shape[0], -1)
                mmd = mmd_loss(feat_s, feat_t)

                # Combined loss (lambda=0.1 for MMD)
                loss = rul_loss + 0.1 * mmd
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

            sched.step()
            v = ev(model, vl)
            if v < bv:
                bv = v; bs_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; ni = 0
            else: ni += 1
            if ni >= 12: break

        model.load_state_dict(bs_state); model.to(DEVICE)
        r1r = ev(model, te1l); r2r = ev(model, te2l)
        print(f"  [{seed}] MMD(0.1) | FD001={r1r:.2f} | FD002={r2r:.2f} | ratio={r2r/r1r:.2f}")
        results_mmd.append((r1r, r2r))
        sys.stdout.flush()

    mm = np.array(results_mmd)
    print(f"  MMD avg: FD001={mm[:,0].mean():.2f}±{mm[:,0].std():.2f} | FD002={mm[:,1].mean():.2f}±{mm[:,1].std():.2f} | ratio={mm[:,1].mean()/mm[:,0].mean():.2f}")
    sys.stdout.flush()

    # ─── Exp 30: Patch-based Role-Transformer ───
    print("\n" + "="*70)
    print("Exp 30: Patch-based Role-Transformer (patch_len=5, 6 patches)")
    print("="*70)
    sys.stdout.flush()

    class PatchRoleEncoder(nn.Module):
        """Role encoder with patch embeddings instead of point-wise."""
        def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1, patch_len=5):
            super().__init__()
            self.n_comp = len(groups)
            self.comp_idx = {c: [sc.index(s) for s in ss] for c, ss in groups.items()}
            self.comp_names = list(groups.keys())
            self.patch_len = patch_len
            # Patch embedding: each patch maps (patch_len * n_sensors_in_group) -> d
            # But since groups have different sizes, use per-sensor patch + mean pool
            self.proj = nn.Linear(patch_len, d)
            n_patches = SEQ_LEN // patch_len
            self.pos = nn.Parameter(torch.randn(1, n_patches, d) * 0.02)
            enc = nn.TransformerEncoderLayer(d, heads, d*4, dropout, batch_first=True)
            self.within = nn.TransformerEncoder(enc, layers)
            self.pool = nn.Linear(d, d)
            self.d = d

        def forward(self, x):
            B, T, C = x.shape
            pl = self.patch_len
            n_patches = T // pl
            cs = []
            for c in self.comp_names:
                idx = self.comp_idx[c]
                n_sensors = len(idx)
                # (B, T, n_sensors) -> (B, n_sensors, T)
                cx = x[:, :T - T % pl, idx].permute(0, 2, 1)
                # (B, n_sensors, n_patches, patch_len)
                cx = cx.reshape(B * n_sensors, n_patches, pl)
                # Patch embed
                cx = self.proj(cx) + self.pos[:, :n_patches]
                cx = self.within(cx)[:, -1].reshape(B, n_sensors, -1)
                cs.append(self.pool(cx.mean(1)))
            return torch.stack(cs, 1)

    class PatchRoleTrans(nn.Module):
        def __init__(self, groups, sc, d=32, heads=4, layers=2, dropout=0.1, patch_len=5):
            super().__init__()
            self.encoder = PatchRoleEncoder(groups, sc, d, heads, layers, dropout, patch_len)
            self.head = CrossComponentBlock(len(groups), d, heads, dropout)
        def forward(self, x):
            return self.head(self.encoder(x))

    results_patch = []
    results_point = []
    for seed in [42, 123, 456]:
        torch.manual_seed(seed); np.random.seed(seed)
        ts, vs = torch.utils.data.random_split(ds1, [nt, nv], generator=torch.Generator().manual_seed(seed))
        tl = DataLoader(ts, 256, shuffle=True, num_workers=0)
        vl = DataLoader(vs, 256, num_workers=0)
        te1l = DataLoader(TDS(te1, sc, r1), 256, num_workers=0)
        te2l = DataLoader(TDS(te2, sc, r2), 256, num_workers=0)

        # Patch-based
        torch.manual_seed(seed)
        m_patch = PatchRoleTrans(COMPONENT_GROUPS, sc, patch_len=5)
        m_patch, _ = train_m(m_patch, tl, vl)
        r1p = ev(m_patch, te1l); r2p = ev(m_patch, te2l)
        print(f"  [{seed}] Patch(5) | FD001={r1p:.2f} | FD002={r2p:.2f} | ratio={r2p/r1p:.2f}")
        results_patch.append((r1p, r2p))

        # Point-based baseline (RoleTransV2 for fair comparison)
        torch.manual_seed(seed)
        m_point = RoleTransV2(COMPONENT_GROUPS, sc)
        m_point, _ = train_m(m_point, tl, vl)
        r1v = ev(m_point, te1l); r2v = ev(m_point, te2l)
        print(f"  [{seed}] Point   | FD001={r1v:.2f} | FD002={r2v:.2f} | ratio={r2v/r1v:.2f}")
        results_point.append((r1v, r2v))
        sys.stdout.flush()

    pp = np.array(results_patch); pt = np.array(results_point)
    print(f"\n  Patch avg: FD001={pp[:,0].mean():.2f}±{pp[:,0].std():.2f} | FD002={pp[:,1].mean():.2f}±{pp[:,1].std():.2f} | ratio={pp[:,1].mean()/pp[:,0].mean():.2f}")
    print(f"  Point avg: FD001={pt[:,0].mean():.2f}±{pt[:,0].std():.2f} | FD002={pt[:,1].mean():.2f}±{pt[:,1].std():.2f} | ratio={pt[:,1].mean()/pt[:,0].mean():.2f}")

    print(f"\n\nRound 5 complete! Time: {time.strftime('%H:%M')}")
    sys.stdout.flush()

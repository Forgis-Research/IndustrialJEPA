"""
V31 Combined Baseline Runner: TimesFM + Moirai + FEMTO FAM.

Runs all P-A through P-C open items:
- TimesFM-1.0-200M: frozen encoder + 198K dt-MLP head (4 datasets, 3 seeds)
- Moirai-1.1-R-base (91M): frozen encoder + 198K dt-MLP head (4 datasets, 3 seeds)
- FEMTO FAM: pretrain + pred-FT (1 seed for scout, 3 seeds if works)

Run with py310 conda env:
  conda run -n py310 python3 run_all_baselines.py [--model timesfm|moirai|femto|all]

All results written to results/timesfm_baseline.json, results/moirai_baseline.json,
results/femto_baseline.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V31_DIR = FAM_DIR / 'experiments/v31'
V30_DIR = FAM_DIR / 'experiments/v30'

for d_name in ['timesfm_features', 'moirai_features', 'surfaces', 'results']:
    (V31_DIR / d_name).mkdir(parents=True, exist_ok=True)
(V31_DIR / 'results/surface_pngs').mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v11'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v29'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v30'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v31'))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}", flush=True)

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

SEEDS = [42, 123, 456]
DATASETS = ['FD001', 'FD003', 'MBA', 'BATADAL']
CMAPSS_HORIZONS = [1, 5, 10, 20, 50, 100, 150]
ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _build_anomaly(entity_list, stride=1):
    from train import EventDataset
    datasets = []
    for e in entity_list:
        X = np.array(e['test'], dtype=np.float32)
        y = np.array(e['labels'], dtype=np.int32)
        T = len(X)
        if T <= 128:
            continue
        d = EventDataset(X, y, max_context=512, stride=stride,
                         max_future=200, min_context=128)
        if len(d) > 0:
            datasets.append(d)
    return ConcatDataset(datasets)


def load_dataset(dataset: str):
    """Return (train_ds, val_ds, test_ds, n_channels, horizons)."""
    from train import EventDataset

    if dataset.startswith('FD'):
        from _cmapss_raw import load_cmapss_raw

        def build_cmapss(engines, stride):
            datasets = []
            for eid, seq in engines.items():
                T = len(seq)
                if T <= 128:
                    continue
                labels = np.zeros(T, dtype=np.int32)
                labels[T - 1] = 1
                d = EventDataset(seq, labels, max_context=512, stride=stride,
                                 max_future=200, min_context=128)
                if len(d) > 0:
                    datasets.append(d)
            return ConcatDataset(datasets)

        data = load_cmapss_raw(dataset)
        tr = build_cmapss(data['train_engines'], stride=4)
        va = build_cmapss(data['val_engines'], stride=4)
        te = build_cmapss(data['test_engines'], stride=1)
        return tr, va, te, 14, CMAPSS_HORIZONS

    if dataset == 'MBA':
        from data.mba import load_mba
        d = load_mba(normalize=False)
        test = np.asarray(d['test'], dtype=np.float32)
        labels = np.asarray(d['labels'], dtype=np.int32)
        T = len(test)
        t1, t2, gap = int(0.6 * T), int(0.7 * T), 200
        ent_tr = [{'test': test[:t1], 'labels': labels[:t1]}]
        ent_va = [{'test': test[t1 + gap:t2], 'labels': labels[t1 + gap:t2]}]
        ent_te = [{'test': test[t2 + gap:], 'labels': labels[t2 + gap:]}]
        n_ch = int(d.get('n_channels', test.shape[1]))
        return (_build_anomaly(ent_tr, 4), _build_anomaly(ent_va, 4),
                _build_anomaly(ent_te, 1), n_ch, ANOMALY_HORIZONS)

    if dataset == 'BATADAL':
        from data.batadal import load_batadal
        d = load_batadal(normalize=False)
        n_ch = d.get('n_channels', 43)
        ent_tr = d.get('ft_train', [])
        ent_va = d.get('ft_val', [])
        ent_te = d.get('ft_test', [])
        return (_build_anomaly(ent_tr, 4), _build_anomaly(ent_va, 4),
                _build_anomaly(ent_te, 1), n_ch, ANOMALY_HORIZONS)

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Shared: MLP head (198K params, same as MOMENT/Chr2)
# ---------------------------------------------------------------------------

class BaselineHead(nn.Module):
    """198K-param dt-conditioned MLP head."""

    def __init__(self, d_input: int, d_hidden: int = 256):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.dt_embed = nn.Embedding(256, d_hidden)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, feat: torch.Tensor, dt_idx: torch.Tensor) -> torch.Tensor:
        h = self.proj(feat) + self.dt_embed(dt_idx.clamp(0, 255))
        return self.mlp(h)


def train_head(feat_train, label_train, dt_train,
               feat_val, label_val, dt_val,
               d_input: int, seed: int) -> BaselineHead:
    """Train MLP head on frozen features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = BaselineHead(d_input=d_input).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)

    N, K = label_train.shape

    def make_flat(feats, labels, dts):
        feats_exp = feats.unsqueeze(1).expand(-1, K, -1).reshape(-1, feats.shape[1])
        return feats_exp, labels.reshape(-1).float(), dts.reshape(-1).long()

    f_tr, l_tr, d_tr = make_flat(feat_train, label_train, dt_train)
    f_va, l_va, d_va = make_flat(feat_val, label_val, dt_val)

    BS = 512
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 5

    for epoch in range(60):
        head.train()
        perm = torch.randperm(len(f_tr))
        epoch_loss = 0.0
        n_steps = 0
        for s in range(0, len(f_tr), BS):
            idx = perm[s:s + BS]
            f_b = f_tr[idx].to(DEVICE)
            l_b = l_tr[idx].to(DEVICE)
            d_b = d_tr[idx].to(DEVICE)
            logits = head(f_b, d_b).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, l_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        head.eval()
        with torch.no_grad():
            val_preds, val_lbls = [], []
            for s in range(0, len(f_va), BS):
                logits = head(f_va[s:s + BS].to(DEVICE), d_va[s:s + BS].to(DEVICE)).squeeze(-1)
                val_preds.append(logits.cpu())
                val_lbls.append(l_va[s:s + BS])
            val_loss = F.binary_cross_entropy_with_logits(
                torch.cat(val_preds), torch.cat(val_lbls).float()).item()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1}: train={epoch_loss/n_steps:.4f} val={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    early stop at epoch {epoch+1}", flush=True)
            break

    if best_state:
        head.load_state_dict(best_state)
    return head


def evaluate_head(head, feat_test, label_test, dt_test) -> dict:
    from sklearn.metrics import roc_auc_score
    head.eval()
    N, K = label_test.shape
    BS = 512
    all_probs = np.zeros((N, K), dtype=np.float32)

    with torch.no_grad():
        for k in range(K):
            dt_k = dt_test[:, k]
            probs_k = []
            for s in range(0, N, BS):
                logits = head(feat_test[s:s + BS].to(DEVICE), dt_k[s:s + BS].to(DEVICE)).squeeze(-1)
                probs_k.append(torch.sigmoid(logits).cpu().numpy())
            all_probs[:, k] = np.concatenate(probs_k)

    labels_np = label_test.numpy()
    aurocs = []
    for k in range(K):
        y = labels_np[:, k]
        p = all_probs[:, k]
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        aurocs.append(float(roc_auc_score(y, p)))

    return {
        'per_horizon_auroc': aurocs,
        'h_auroc': float(np.mean(aurocs)) if aurocs else float('nan'),
        'n_valid_horizons': len(aurocs),
    }


def extract_and_cache(extractor_fn, loader, cache_path: Path, desc: str, horizons: list):
    """Extract features from a DataLoader and cache to .npz.

    collate_event returns (ctx_padded, ctx_mask, ttes, ts):
    - ctx_padded: (B, T, C)  zero-padded context
    - ctx_mask:   (B, T)     True where padding
    - ttes:       (B,)       time-to-event (float or int)
    - ts:         (B,)       current time index

    We build the label surface from ttes using build_label_surface.
    dt_indices are the horizon indices (0..K-1) used as embedding lookups.
    """
    if cache_path.exists():
        print(f"  [cache] {desc}: {cache_path.name}", flush=True)
        cached = np.load(cache_path)
        return (torch.from_numpy(cached['feats']),
                torch.from_numpy(cached['labels']),
                torch.from_numpy(cached['dt_indices']))

    print(f"  [extract] {desc}...", flush=True)
    from evaluation.losses import build_label_surface

    h_t = torch.tensor(horizons, dtype=torch.float32)
    K = len(horizons)

    all_feats, all_labels, all_dts = [], [], []
    n = len(loader)
    for bidx, batch in enumerate(loader):
        if (bidx + 1) % 100 == 0 or bidx == 0:
            print(f"    {bidx+1}/{n}", flush=True)
        ctx_padded, ctx_mask, ttes, ts = batch
        # ctx_padded: (B, T, C), ttes: (B,) scalar TTE
        x = ctx_padded.float()
        feats = extractor_fn(x)  # (B, D)

        # Build label surface: (B, 1, K) -> (B, K)
        tte_col = ttes.unsqueeze(1)  # (B, 1)
        labels_surf = build_label_surface(tte_col, h_t).squeeze(1)  # (B, K)

        # dt_indices: for each horizon k, index = k (0..K-1)
        dt_indices = torch.arange(K).unsqueeze(0).expand(x.shape[0], -1)  # (B, K)

        all_feats.append(feats.cpu())
        all_labels.append(labels_surf.cpu())
        all_dts.append(dt_indices.cpu())

    feats = torch.cat(all_feats, 0)
    labels = torch.cat(all_labels, 0)
    dts = torch.cat(all_dts, 0)
    np.savez(cache_path, feats=feats.numpy(), labels=labels.numpy(), dt_indices=dts.numpy())
    print(f"  Saved {feats.shape} to {cache_path.name}", flush=True)
    return feats, labels, dts


def run_baseline_on_dataset(dataset: str, seeds: list,
                             extractor_fn, feat_dir: Path,
                             model_name: str, d_feat: int) -> dict:
    """Generic baseline runner for one dataset."""
    from train import collate_event

    print(f"\n{model_name} | {dataset}", flush=True)
    tr_ds, va_ds, te_ds, n_ch, horizons = load_dataset(dataset)
    K = len(horizons)

    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_event)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_event)
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_event)

    ds_safe = dataset.replace('/', '_')
    feat_tr, lbl_tr, dt_tr = extract_and_cache(extractor_fn, tr_loader, feat_dir / f"{ds_safe}_train.npz", f"{dataset} train", horizons)
    feat_va, lbl_va, dt_va = extract_and_cache(extractor_fn, va_loader, feat_dir / f"{ds_safe}_val.npz", f"{dataset} val", horizons)
    feat_te, lbl_te, dt_te = extract_and_cache(extractor_fn, te_loader, feat_dir / f"{ds_safe}_test.npz", f"{dataset} test", horizons)

    per_seed = []
    for seed in seeds:
        print(f"  seed={seed}", flush=True)
        if WANDB_OK:
            wandb.init(project='industrialjepa',
                       name=f"v31-{model_name.lower().replace(' ', '-')}-{dataset}-s{seed}",
                       config={'dataset': dataset, 'seed': seed, 'model': model_name,
                               'head': 'dt-MLP-198K', 'horizons': horizons},
                       reinit=True, tags=['v31', model_name.lower().split()[0], 'baseline'])

        head = train_head(feat_tr, lbl_tr, dt_tr, feat_va, lbl_va, dt_va, d_feat, seed)
        metrics = evaluate_head(head, feat_te, lbl_te, dt_te)
        per_seed.append(metrics['h_auroc'])
        print(f"    h-AUROC: {metrics['h_auroc']:.4f}", flush=True)

        if WANDB_OK:
            wandb.log({'h_auroc': metrics['h_auroc'], 'dataset': dataset, 'seed': seed})
            wandb.finish()

    mean, std = float(np.mean(per_seed)), float(np.std(per_seed))
    print(f"  RESULT: {mean:.4f} +/- {std:.4f} per_seed={[f'{v:.4f}' for v in per_seed]}", flush=True)
    return {'dataset': dataset, 'model': model_name, 'seeds': seeds,
            'per_seed': per_seed, 'mean': mean, 'std': std, 'horizons': horizons}


# ---------------------------------------------------------------------------
# TimesFM extractor
# ---------------------------------------------------------------------------

def build_timesfm_extractor():
    """Return (extractor_fn, d_feat, model_info)."""
    import timesfm
    print("Loading TimesFM-1.0-200M...", flush=True)

    hparams = timesfm.TimesFmHparams(
        backend='gpu' if 'cuda' in DEVICE else 'cpu',
        per_core_batch_size=32,
        horizon_len=128,
        context_len=512,
    )
    checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id='google/timesfm-1.0-200m-pytorch')
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
    tfm.load_from_checkpoint(checkpoint=checkpoint)
    model = tfm._model
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  TimesFM loaded: {n_params:.1f}M params", flush=True)
    model.eval()
    model.to(DEVICE)

    hidden_cache = [None]
    def hook(module, input, output):
        hidden_cache[0] = output.detach().clone()
    model.stacked_transformer.register_forward_hook(hook)

    CONTEXT_LEN = 512
    HIDDEN_SIZE = 1280

    @torch.no_grad()
    def extract(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, 1280)."""
        B, T, C = x.shape
        if T > CONTEXT_LEN:
            x = x[:, -CONTEXT_LEN:, :]
        elif T < CONTEXT_LEN:
            pad = torch.zeros(B, CONTEXT_LEN - T, C, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        x_flat = x.permute(0, 2, 1).reshape(B * C, CONTEXT_LEN).to(DEVICE)
        pad_ts = torch.zeros(B * C, CONTEXT_LEN, device=DEVICE)
        freq_ts = torch.zeros(B * C, 1, dtype=torch.long, device=DEVICE)

        model(x_flat, pad_ts, freq_ts)
        h = hidden_cache[0]  # (B*C, N_patches, 1280)
        h_pooled = h.mean(dim=1)  # (B*C, 1280)
        return h_pooled.reshape(B, C, HIDDEN_SIZE).mean(dim=1).cpu()  # (B, 1280)

    model_info = {'name': 'TimesFM-1.0-200M', 'n_params_M': n_params,
                  'hf_id': 'google/timesfm-1.0-200m-pytorch'}
    return extract, HIDDEN_SIZE, model_info


# ---------------------------------------------------------------------------
# Moirai extractor
# ---------------------------------------------------------------------------

def build_moirai_extractor(size: str = 'base'):
    """Return (extractor_fn, d_feat, model_info)."""
    from uni2ts.model.moirai import MoiraiModule

    hf_map = {'small': 'Salesforce/moirai-1.1-R-small',
               'base': 'Salesforce/moirai-1.1-R-base',
               'large': 'Salesforce/moirai-1.1-R-large'}
    hf_id = hf_map[size]

    print(f"Loading Moirai-1.1-R-{size}...", flush=True)
    model = MoiraiModule.from_pretrained(hf_id)
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Moirai loaded: {n_params:.1f}M params, d_model={model.d_model}", flush=True)

    D_MODEL = model.d_model
    PATCH_SIZE = 32  # use fixed 32 from supported patch_sizes
    CONTEXT_LEN = 512
    N_PATCHES = CONTEXT_LEN // PATCH_SIZE  # 16
    MAX_PATCH = model.patch_sizes[-1]  # 128

    hidden_cache = [None]
    def hook(module, input, output):
        hidden_cache[0] = output.detach().clone()
    model.encoder.register_forward_hook(hook)

    @torch.no_grad()
    def extract(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, D_MODEL)."""
        B, T, C = x.shape
        if T > CONTEXT_LEN:
            x = x[:, -CONTEXT_LEN:, :]
        elif T < CONTEXT_LEN:
            pad = torch.zeros(B, CONTEXT_LEN - T, C, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        # Process each channel as a separate univariate sequence
        x_flat = x.permute(0, 2, 1).reshape(B * C, CONTEXT_LEN).to(DEVICE)
        N = B * C

        # Build patched input (N, N_patches, MAX_PATCH)
        target = x_flat.view(N, N_PATCHES, PATCH_SIZE)
        target_padded = torch.zeros(N, N_PATCHES, MAX_PATCH, device=DEVICE)
        target_padded[:, :, :PATCH_SIZE] = target
        observed = torch.zeros(N, N_PATCHES, MAX_PATCH, dtype=torch.bool, device=DEVICE)
        observed[:, :, :PATCH_SIZE] = True

        sample_id = torch.arange(N, device=DEVICE).unsqueeze(1).expand(-1, N_PATCHES)
        time_id = torch.arange(N_PATCHES, device=DEVICE).unsqueeze(0).expand(N, -1)
        variate_id = torch.zeros(N, N_PATCHES, dtype=torch.long, device=DEVICE)
        pred_mask = torch.zeros(N, N_PATCHES, dtype=torch.bool, device=DEVICE)
        ps_tensor = torch.full((N, N_PATCHES), PATCH_SIZE, dtype=torch.long, device=DEVICE)

        model(target_padded, observed, sample_id, time_id, variate_id, pred_mask, ps_tensor)
        h = hidden_cache[0]  # (N, N_PATCHES, D_MODEL)
        h_pooled = h.mean(dim=1)  # (N, D_MODEL)
        return h_pooled.reshape(B, C, D_MODEL).mean(dim=1).cpu()  # (B, D_MODEL)

    model_info = {'name': f'Moirai-1.1-R-{size}', 'n_params_M': n_params, 'hf_id': hf_id}
    return extract, D_MODEL, model_info


# ---------------------------------------------------------------------------
# FEMTO FAM pipeline
# ---------------------------------------------------------------------------

def run_femto_fam(seeds: list):
    """Run FAM pretrain + pred-FT on FEMTO bearing dataset.

    Standalone FAM pipeline (no dependency on _runner_v31 LOADERS registry).
    """
    print("\n" + "=" * 60, flush=True)
    print("FEMTO bearing dataset - FAM pipeline", flush=True)

    from data.femto import load_femto, check_femto_available

    if not check_femto_available():
        return {'error': 'FEMTO zip not found', 'status': 'SKIPPED'}

    print("Loading FEMTO bundle...", flush=True)
    bundle = load_femto(verbose=True)
    if bundle is None:
        return {'error': 'Failed to load FEMTO bundle', 'status': 'FAILED'}

    from model import FAM
    from train import (PretrainDataset, EventDataset, collate_pretrain, collate_event,
                       pretrain as pretrain_fn, evaluate)
    from _runner import _build_event_concat

    horizons = bundle['horizons']
    n_ch = bundle['n_channels']
    K = len(horizons)
    results = []

    for seed in seeds[:1]:  # Scout with 1 seed first
        print(f"\nFEMTO seed={seed}", flush=True)
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            if WANDB_OK:
                wandb.init(project='industrialjepa', name=f'v31-FEMTO-FAM-s{seed}',
                           config={'dataset': 'FEMTO', 'seed': seed, 'n_channels': n_ch,
                                   'horizons': horizons, 'n_pretrain_seqs': len(bundle['pretrain_seqs'])},
                           reinit=True, tags=['v31', 'femto', 'fam'])

            # Build FAM model
            model = FAM(n_channels=n_ch, patch_size=16, d_model=256,
                        n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                        ema_momentum=0.99, predictor_hidden=256,
                        norm_mode='none', predictor_kind='mlp',
                        event_head_kind='discrete_hazard').to(DEVICE)
            print(f"  FAM: {sum(p.numel() for p in model.parameters()):,} params", flush=True)

            # Pretrain
            pretrain_ds = PretrainDataset(bundle['pretrain_seqs'], n_cuts=40,
                                          max_context=512, delta_t_max=max(horizons))
            pre_loader = DataLoader(pretrain_ds, batch_size=64, shuffle=True,
                                    collate_fn=collate_pretrain, num_workers=0)
            print(f"  pretrain_ds: {len(pretrain_ds)} samples", flush=True)

            pre_result = pretrain_fn(model, pre_loader, lr=3e-4, n_epochs=30,
                                     patience=5, device=DEVICE)
            print(f"  pretrain best_loss={pre_result['best_loss']:.4f}", flush=True)

            # Freeze encoder, finetune predictor
            from train import finetune as finetune_fn
            train_ft = _build_event_concat(bundle['ft_train'], stride=4, max_context=512,
                                           max_future=max(horizons))
            val_ft = _build_event_concat(bundle['ft_val'], stride=4, max_context=512,
                                         max_future=max(horizons))
            test_ft = _build_event_concat(bundle['ft_test'], stride=1, max_context=512,
                                          max_future=max(horizons))
            print(f"  train_ft={len(train_ft)}, val_ft={len(val_ft)}, test_ft={len(test_ft)}", flush=True)

            tr_loader = DataLoader(train_ft, batch_size=128, shuffle=True,
                                   collate_fn=collate_event, num_workers=0)
            va_loader = DataLoader(val_ft, batch_size=128, shuffle=False,
                                   collate_fn=collate_event, num_workers=0)
            te_loader = DataLoader(test_ft, batch_size=128, shuffle=False,
                                   collate_fn=collate_event, num_workers=0)

            ft_result = finetune_fn(model, tr_loader, va_loader, horizons,
                                    mode='pred_ft', lr=1e-3, n_epochs=30,
                                    patience=8, device=DEVICE)
            print(f"  finetune best_val_loss={ft_result['best_val']:.4f}", flush=True)

            # Evaluate
            eval_out = evaluate(model, te_loader, horizons, mode='pred_ft', device=DEVICE)
            p_surf = eval_out['p_surface']
            y_surf = eval_out['y_surface']
            # Compute h-AUROC: mean per-horizon AUROC over valid (non-degenerate) horizons
            from sklearn.metrics import roc_auc_score
            valid = [i for i in range(len(horizons)) if 0 < y_surf[:, i].mean() < 1]
            if valid:
                h_auroc = float(np.mean([roc_auc_score(y_surf[:, i], p_surf[:, i]) for i in valid]))
            else:
                h_auroc = float('nan')
            pooled_auprc = float(eval_out['primary']['auprc'])
            print(f"  FEMTO h-AUROC: {h_auroc:.4f}  pooled_AUPRC: {pooled_auprc:.4f}", flush=True)

            if WANDB_OK:
                wandb.log({'h_auroc': h_auroc, 'pretrain_loss': pre_result['best_loss']})
                wandb.finish()

            results.append({'seed': seed, 'h_auroc': h_auroc,
                            'pooled_auprc': pooled_auprc,
                            'pretrain_loss': pre_result['best_loss'],
                            'ft_best_val': ft_result['best_val']})

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            traceback.print_exc()
            results.append({'error': str(e), 'seed': seed})
            if WANDB_OK:
                try:
                    wandb.finish(exit_code=1)
                except Exception:
                    pass

    valid = [r for r in results if 'h_auroc' in r]
    return {
        'dataset': 'FEMTO',
        'model': 'FAM-predFT',
        'seeds': seeds[:1],
        'results': results,
        'per_seed_h_auroc': [r['h_auroc'] for r in valid],
        'mean_h_auroc': float(np.mean([r['h_auroc'] for r in valid])) if valid else float('nan'),
        'status': 'COMPLETE' if valid else 'FAILED',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['timesfm', 'moirai', 'femto', 'all'],
                        default='all')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--moirai-size', choices=['small', 'base', 'large'], default='base')
    args = parser.parse_args()

    all_results = {}

    # ---- TimesFM ----
    if args.model in ('timesfm', 'all'):
        print("\n" + "=" * 60, flush=True)
        print("TimesFM-1.0-200M Baseline", flush=True)
        try:
            extract_fn, d_feat, model_info = build_timesfm_extractor()
            feat_dir = V31_DIR / 'timesfm_features'
            timesfm_results = {}
            for ds in args.datasets:
                try:
                    r = run_baseline_on_dataset(ds, args.seeds, extract_fn, feat_dir,
                                                model_info['name'], d_feat)
                    timesfm_results[ds] = r
                except Exception as e:
                    print(f"ERROR on {ds}: {e}", flush=True)
                    traceback.print_exc()
                    timesfm_results[ds] = {'error': str(e)}

            timesfm_results['model_info'] = model_info
            with open(V31_DIR / 'results/timesfm_baseline.json', 'w') as f:
                json.dump(timesfm_results, f, indent=2)
            all_results['timesfm'] = timesfm_results
            print("\nTimesFM SUMMARY:")
            for ds, r in timesfm_results.items():
                if ds == 'model_info':
                    continue
                if 'error' in r:
                    print(f"  {ds}: ERROR")
                else:
                    print(f"  {ds}: {r['mean']:.4f} +/- {r['std']:.4f}")
        except Exception as e:
            print(f"TimesFM FAILED: {e}", flush=True)
            traceback.print_exc()
            all_results['timesfm'] = {'error': str(e)}

    # ---- Moirai ----
    if args.model in ('moirai', 'all'):
        print("\n" + "=" * 60, flush=True)
        print(f"Moirai-1.1-R-{args.moirai_size} Baseline", flush=True)
        try:
            extract_fn, d_feat, model_info = build_moirai_extractor(args.moirai_size)
            feat_dir = V31_DIR / 'moirai_features'
            moirai_results = {}
            for ds in args.datasets:
                try:
                    r = run_baseline_on_dataset(ds, args.seeds, extract_fn, feat_dir,
                                                model_info['name'], d_feat)
                    moirai_results[ds] = r
                except Exception as e:
                    print(f"ERROR on {ds}: {e}", flush=True)
                    traceback.print_exc()
                    moirai_results[ds] = {'error': str(e)}

            moirai_results['model_info'] = model_info
            with open(V31_DIR / 'results/moirai_baseline.json', 'w') as f:
                json.dump(moirai_results, f, indent=2)
            all_results['moirai'] = moirai_results
            print("\nMoirai SUMMARY:")
            for ds, r in moirai_results.items():
                if ds == 'model_info':
                    continue
                if 'error' in r:
                    print(f"  {ds}: ERROR")
                else:
                    print(f"  {ds}: {r['mean']:.4f} +/- {r['std']:.4f}")
        except Exception as e:
            print(f"Moirai FAILED: {e}", flush=True)
            traceback.print_exc()
            all_results['moirai'] = {'error': str(e)}

    # ---- FEMTO FAM ----
    if args.model in ('femto', 'all'):
        try:
            femto_result = run_femto_fam(args.seeds)
            with open(V31_DIR / 'results/femto_baseline.json', 'w') as f:
                json.dump(femto_result, f, indent=2)
            all_results['femto'] = femto_result
        except Exception as e:
            print(f"FEMTO FAILED: {e}", flush=True)
            traceback.print_exc()
            all_results['femto'] = {'error': str(e)}

    # Save combined results
    with open(V31_DIR / 'results/all_baselines_combined.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60, flush=True)
    print("ALL DONE. Results in fam-jepa/experiments/v31/results/", flush=True)


if __name__ == '__main__':
    main()

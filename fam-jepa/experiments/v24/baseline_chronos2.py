"""V24 baseline: Chronos-2 frozen encoder + linear probe for event detection.

Fair comparison to our FAM pred-FT:
- Chronos-2 is a pretrained multivariate foundation model (amazon/chronos-2).
- We freeze it and run embed() on each event observation's context.
- Pool embeddings to a fixed-size feature (mean over patches, mean over variates
  -> 768-d per observation).
- Train a linear probe (768 -> K horizon logits) with pos-weighted BCE on the
  EXACT SAME labeled data used by our pred-FT (same splits, same labels).
- Report AUPRC / AUROC on the same test set.

Single architecture across datasets: Chronos-2 handles multivariate natively.

Usage:
  python baseline_chronos2.py --dataset FD001 [--seed 42]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V24_DIR = FAM_DIR / 'experiments/v24'
FEAT_DIR = V24_DIR / 'chronos_features'
RES_DIR = V24_DIR / 'results'
FEAT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))
sys.path.insert(0, str(V24_DIR))

from train import EventDataset, collate_event
from evaluation.surface_metrics import (
    evaluate_probability_surface, auprc_per_horizon, monotonicity_violation_rate,
)
from evaluation.losses import build_label_surface
from _cmapss_raw import load_cmapss_raw

# Horizons
HORIZONS_CMAPSS = [1, 5, 10, 20, 50, 100, 150]
HORIZONS_ANOMALY = [1, 5, 10, 20, 50, 100, 150, 200]
HORIZONS_SEPSIS = [1, 2, 3, 6, 12, 24, 48]


def get_horizons(dataset: str):
    if dataset.startswith('FD'):
        return HORIZONS_CMAPSS
    if dataset.lower() == 'sepsis':
        return HORIZONS_SEPSIS
    return HORIZONS_ANOMALY


# ---------------------------------------------------------------------------
# Dataset loading (reuses the same protocol as FAM)
# ---------------------------------------------------------------------------

def build_cmapss_event_concat(engines, stride, max_context=512, min_context=128):
    datasets = []
    for eid, seq in engines.items():
        T = len(seq)
        if T <= min_context:
            continue
        labels = np.zeros(T, dtype=np.int32)
        labels[T - 1] = 1
        d = EventDataset(seq, labels, max_context=max_context, stride=stride,
                         max_future=200, min_context=min_context)
        if len(d) > 0:
            datasets.append(d)
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


def load_dataset(dataset: str):
    """Return (train_ds, val_ds, test_ds, n_channels, horizons)."""
    if dataset.startswith('FD'):
        data = load_cmapss_raw(dataset)
        tr = build_cmapss_event_concat(data['train_engines'], stride=4)
        va = build_cmapss_event_concat(data['val_engines'], stride=4)
        te = build_cmapss_event_concat(data['test_engines'], stride=1)
        return tr, va, te, 14, get_horizons(dataset)
    if dataset == 'SMAP':
        from data.smap_msl import split_smap_entities
        ft = split_smap_entities(normalize=False)
        tr = _build_anomaly(ft['ft_train'], stride=4)
        va = _build_anomaly(ft['ft_val'], stride=4)
        te = _build_anomaly(ft['ft_test'], stride=1)
        return tr, va, te, 25, get_horizons(dataset)
    if dataset == 'MSL':
        from data.smap_msl import split_msl_entities
        ft = split_msl_entities(normalize=False)
        return (_build_anomaly(ft['ft_train'], 4),
                _build_anomaly(ft['ft_val'], 4),
                _build_anomaly(ft['ft_test'], 1), 55, get_horizons(dataset))
    raise ValueError(dataset)


def _build_anomaly(entity_list, stride, max_context=512, min_context=128):
    datasets = []
    for e in entity_list:
        x = e['test']
        y = e['labels']
        if len(x) <= min_context + 1:
            continue
        d = EventDataset(x, y, max_context=max_context, stride=stride,
                         max_future=200, min_context=min_context)
        if len(d) > 0:
            datasets.append(d)
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


# ---------------------------------------------------------------------------
# Feature extraction with Chronos-2
# ---------------------------------------------------------------------------

def extract_features(pipe, loader, device='cuda', max_context=512):
    """Run Chronos-2 embed() on each batch, pool to (B, d_model) features.

    Chronos-2 embed() returns list of (n_variates, num_patches+2, d_model).
    We pool: mean over patches -> (n_variates, d_model). Then mean over
    variates -> (d_model,). Result: per-observation 768-dim feature.
    """
    pipe.model.eval()
    all_feats = []
    all_ttes = []
    all_tidx = []
    t0 = time.time()
    n = 0
    for ctx, ctx_m, tte, t_idx in loader:
        # ctx: (B, T, C) -> Chronos expects (B, C, T) (n_variates last-dim position)
        # From the API: 3D torch.Tensor of shape (batch, n_variates, history_length)
        B, T, C = ctx.shape
        x = ctx.transpose(1, 2).contiguous()  # (B, C, T)
        # For variable-length via mask: use a list with per-element trimmed
        # history so Chronos handles left-padding. For our data we already
        # right-padded with zeros - pass as-is.
        with torch.no_grad():
            # Use float32 internally but bfloat16 for efficiency
            emb_list, _ = pipe.embed(x.float(), batch_size=B)
        for emb in emb_list:
            # emb: (n_variates, num_patches+2, d_model)
            feat = emb.mean(dim=(0, 1))  # (d_model,) — mean over variates+patches
            all_feats.append(feat.float().cpu())
        all_ttes.append(tte)
        all_tidx.append(t_idx)
        n += B
        if n % 5000 < B:
            print(f"    {n} obs processed ({time.time()-t0:.0f}s elapsed)",
                  flush=True)
    feats = torch.stack(all_feats)  # (N, d_model)
    ttes = torch.cat(all_ttes)      # (N,)
    tidx = torch.cat(all_tidx)      # (N,)
    return feats, ttes, tidx


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

class Probe(nn.Module):
    def __init__(self, d_in: int, K: int, hidden: int = 0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Linear(hidden, K),
            )
        else:
            self.net = nn.Linear(d_in, K)

    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, ytr, Xva, yva, horizons, device='cuda',
                epochs: int = 30, lr: float = 1e-3, patience: int = 8,
                pos_weight: float = None, hidden: int = 0,
                weight_decay: float = 1e-4) -> Probe:
    import copy
    d = Xtr.shape[1]
    K = len(horizons)
    probe = Probe(d, K, hidden=hidden).to(device)
    if pos_weight is None:
        n_pos = ytr.sum().item()
        n_tot = ytr.numel()
        pos_weight = max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1)))
    pw = torch.tensor(pos_weight, device=device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr,
                            weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xva, yva = Xva.to(device), yva.to(device)
    B = 2048
    best_state, best_val, wait = None, float('inf'), 0
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(len(Xtr), device=device)
        losses = []
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]
            logits = probe(Xtr[idx])
            loss = F.binary_cross_entropy_with_logits(
                logits, ytr[idx].float(), pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sch.step()
        probe.eval()
        with torch.no_grad():
            vlo = F.binary_cross_entropy_with_logits(
                probe(Xva), yva.float(), pos_weight=pw).item()
        if vlo < best_val:
            best_val = vlo
            best_state = copy.deepcopy(probe.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    probe.load_state_dict(best_state)
    return probe, float(best_val)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--hidden', type=int, default=0,
                    help='probe hidden dim; 0 = linear probe')
    ap.add_argument('--model', default='amazon/chronos-2')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--cache-features', action='store_true',
                    help='cache features to disk; reload if already cached')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}", flush=True)

    print(f"loading {args.dataset} ...", flush=True)
    tr_ds, va_ds, te_ds, n_channels, horizons = load_dataset(args.dataset)
    print(f"  train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}  "
          f"n_ch={n_channels}  horizons={horizons}", flush=True)

    feat_cache = FEAT_DIR / f'{args.dataset}_s{args.seed}_chronos2.pt'
    if args.cache_features and feat_cache.exists():
        print(f"loading cached features {feat_cache}", flush=True)
        cache = torch.load(feat_cache, map_location='cpu')
        Xtr, ytr_tte, ttr_idx = cache['tr']
        Xva, yva_tte, tva_idx = cache['va']
        Xte, yte_tte, tte_idx = cache['te']
    else:
        from chronos import Chronos2Pipeline
        print(f"loading Chronos-2 ({args.model}) ...", flush=True)
        t0 = time.time()
        pipe = Chronos2Pipeline.from_pretrained(args.model,
                                                dtype=torch.bfloat16,
                                                device_map=device)
        pipe.model.eval()
        print(f"  loaded in {time.time()-t0:.1f}s  "
              f"d_model={pipe.model.config.hidden_size}", flush=True)

        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_event)
        va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_event)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_event)

        print("extracting train features...", flush=True)
        Xtr, ytr_tte, ttr_idx = extract_features(pipe, tr_loader, device=device)
        print(f"  Xtr={Xtr.shape}", flush=True)
        print("extracting val features...", flush=True)
        Xva, yva_tte, tva_idx = extract_features(pipe, va_loader, device=device)
        print("extracting test features...", flush=True)
        Xte, yte_tte, tte_idx = extract_features(pipe, te_loader, device=device)
        if args.cache_features:
            torch.save({'tr': (Xtr, ytr_tte, ttr_idx),
                        'va': (Xva, yva_tte, tva_idx),
                        'te': (Xte, yte_tte, tte_idx)}, feat_cache)
        del pipe
        torch.cuda.empty_cache()

    # Build label surfaces
    h_tensor = torch.tensor(horizons, dtype=torch.float32)
    ytr = build_label_surface(ytr_tte.unsqueeze(1), h_tensor).squeeze(1)
    yva = build_label_surface(yva_tte.unsqueeze(1), h_tensor).squeeze(1)
    yte = build_label_surface(yte_tte.unsqueeze(1), h_tensor).squeeze(1)
    print(f"label prevalence: tr={ytr.mean():.4f}  va={yva.mean():.4f}  "
          f"te={yte.mean():.4f}", flush=True)

    print("training probe...", flush=True)
    t0 = time.time()
    probe, best_val = train_probe(Xtr, ytr, Xva, yva, horizons,
                                  device=device, hidden=args.hidden)
    print(f"  probe done in {time.time()-t0:.1f}s (best_val={best_val:.4f})",
          flush=True)

    # Eval
    probe.eval()
    Xte_dev = Xte.to(device)
    with torch.no_grad():
        logits = probe(Xte_dev)
        p_te = torch.sigmoid(logits).cpu().numpy()
    yte_np = yte.numpy().astype(np.int32)
    primary = evaluate_probability_surface(p_te, yte_np)
    per_h = auprc_per_horizon(p_te, yte_np, horizon_labels=horizons)
    mono = monotonicity_violation_rate(p_te)
    print(f"AUPRC: {primary['auprc']:.4f}  AUROC: {primary['auroc']:.4f}  "
          f"F1: {primary['f1_best']:.4f}", flush=True)
    for h, a in zip(horizons, per_h['auprc_per_k']):
        print(f"  dt={h:4d}: AUPRC={a:.4f}", flush=True)

    out = args.out or (RES_DIR / f'baseline_chronos2_{args.dataset}.json')
    with open(out, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'baseline': 'chronos-2',
            'seed': args.seed,
            'n_train': int(len(Xtr)), 'n_val': int(len(Xva)),
            'n_test': int(len(Xte)),
            'probe_hidden': args.hidden,
            'probe_best_val': best_val,
            'primary': {k: float(v) if isinstance(v, (int, float, np.floating))
                        else v for k, v in primary.items()},
            'per_horizon_auprc': {str(h): float(a) for h, a in
                                  zip(horizons, per_h['auprc_per_k'])},
            'per_horizon_auroc': {str(h): float(a) for h, a in
                                  zip(horizons, per_h['auroc_per_k'])},
            'monotonicity_violation_rate': float(mono['violation_rate']),
        }, f, indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == '__main__':
    main()

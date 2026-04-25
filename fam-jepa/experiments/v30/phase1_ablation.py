"""V30 Phase 1: 5-variant FAM-vs-Chronos2 ablation.

Disentangles encoder quality (FAM 2.16M vs Chronos-2 120M frozen)
from head capacity (linear probe vs MLP). Five variants on FD001,
FD003, MBA, BATADAL × 3 seeds (42, 123, 456):

  FAM-probe        : frozen FAM enc + LinearProbeHead (~257 per horizon)
  Chr2-probe       : frozen Chronos-2 enc (cached) + LinearProbeHead
  FAM-predft       : frozen FAM enc + pretrained MLP predictor (canonical)
  Chr2-mlp         : frozen Chronos-2 enc + dt-conditioned MLP (~198K rand init)
  FAM-mlp-rand     : frozen FAM enc + RANDOM-init MLP predictor + event head

Plus 10% labels for FAM-predft and FAM-mlp-rand on FD001 + MBA.

Sparse horizons (matches v24 cached Chronos-2 protocol):
  FD*  : [1, 5, 10, 20, 50, 100, 150]
  MBA, BATADAL : [1, 5, 10, 20, 50, 100, 150, 200]

Output: results/phase1_decision.json with full 5-way h-AUROC table +
interpretation + chosen main-table variants.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V30_DIR = FAM_DIR / 'experiments/v30'
RES_DIR = V30_DIR / 'results'
PNG_DIR = RES_DIR / 'surface_pngs'
SURF_DIR = V30_DIR / 'surfaces'
CKPT_DIR = V30_DIR / 'ckpts'
LOG_DIR = V30_DIR / 'logs'
P1_DIR = RES_DIR / 'phase1'
for d in [RES_DIR, PNG_DIR, SURF_DIR, CKPT_DIR, LOG_DIR, P1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v29'))
sys.path.insert(0, str(V30_DIR))

from model import FAM
from train import (EventDataset, collate_event, evaluate, save_surface)
from evaluation.losses import build_label_surface
from _runner_v29 import LOADERS, NORM_POLICY, honest_metrics
from _runner import _global_zscore, _build_event_concat
from _runner_v30 import find_pretrain_ckpt, render_surface_panel, finetune_v30

device = 'cuda'
EPS = 1e-7
DATASETS = ['FD001', 'FD003', 'MBA', 'BATADAL']
SEEDS = [42, 123, 456]
SPARSE_HORIZONS = {
    'FD001': [1, 5, 10, 20, 50, 100, 150],
    'FD003': [1, 5, 10, 20, 50, 100, 150],
    'MBA':   [1, 5, 10, 20, 50, 100, 150, 200],
    'BATADAL': [1, 5, 10, 20, 50, 100, 150, 200],
}
LABEL_FRAC_DATASETS = ['FD001', 'MBA']
CHR_FEAT_DIR = FAM_DIR / 'experiments/v24/chronos_features'


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class LinearProbeHead(nn.Module):
    """Independent Linear(d_in, 1) per horizon. Logits, no hazard CDF."""
    def __init__(self, d_in: int, n_horizons: int):
        super().__init__()
        self.probes = nn.ModuleList([nn.Linear(d_in, 1) for _ in range(n_horizons)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([p(x) for p in self.probes], dim=-1)


class Chr2MLP(nn.Module):
    """MLP conditioned on dt: (h_768 + 1, 256, 256, 1) → logit. ~198K params."""
    def __init__(self, d_input: int = 768, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_input + 1, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt.unsqueeze(0).expand(h.shape[0], -1)
        # (B, K)
        K = dt.shape[1]
        h_exp = h.unsqueeze(1).expand(h.shape[0], K, h.shape[1])
        dt_exp = dt.unsqueeze(-1).float()
        x = torch.cat([h_exp, dt_exp], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Bundle / loader helpers
# ---------------------------------------------------------------------------

def get_bundle(dataset: str):
    bundle = LOADERS[dataset]()
    norm_mode = NORM_POLICY[dataset]
    if norm_mode == 'none':
        bundle = _global_zscore(bundle)
    return bundle, norm_mode


def make_loaders(bundle, max_future, ft_batch=128, label_fraction=1.0,
                 seed=42, max_context=512):
    train_engines = bundle['ft_train']
    if label_fraction < 1.0:
        keys = sorted(train_engines.keys()) if isinstance(train_engines, dict) else list(range(len(train_engines)))
        n_keep = max(1, int(round(len(keys) * label_fraction)))
        rng = np.random.RandomState(seed + 7777)
        keep_idx = sorted(rng.choice(keys, size=n_keep, replace=False).tolist())
        if isinstance(train_engines, dict):
            train_engines = {k: train_engines[k] for k in keep_idx}
        else:
            train_engines = [train_engines[i] for i in keep_idx]
    train_ft = _build_event_concat(train_engines, stride=4,
                                   max_context=max_context, max_future=max_future)
    val_ft = _build_event_concat(bundle['ft_val'], stride=4,
                                 max_context=max_context, max_future=max_future)
    test_ft = _build_event_concat(bundle['ft_test'], stride=1,
                                  max_context=max_context, max_future=max_future)
    return (
        DataLoader(train_ft, batch_size=ft_batch, shuffle=True,
                   collate_fn=collate_event, num_workers=0),
        DataLoader(val_ft, batch_size=ft_batch, shuffle=False,
                   collate_fn=collate_event, num_workers=0),
        DataLoader(test_ft, batch_size=ft_batch, shuffle=False,
                   collate_fn=collate_event, num_workers=0),
        len(train_ft), len(val_ft), len(test_ft)
    )


# ---------------------------------------------------------------------------
# FAM-probe
# ---------------------------------------------------------------------------

def fam_probe(dataset: str, seed: int, horizons: List[int],
              label_fraction: float = 1.0, ft_epochs: int = 30,
              ft_batch: int = 128, max_context: int = 512) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bundle, norm_mode = get_bundle(dataset)
    n_ch = bundle['n_channels']
    pre_ckpt = find_pretrain_ckpt(dataset, norm_mode, seed)
    if pre_ckpt is None:
        return None
    model = FAM(n_channels=n_ch, patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                norm_mode=norm_mode, predictor_kind='mlp',
                event_head_kind='discrete_hazard')
    sd = torch.load(pre_ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    for p in model.encoder.parameters():
        p.requires_grad = False

    head = LinearProbeHead(d_in=256, n_horizons=len(horizons)).to(device)
    h_t = torch.tensor(horizons, dtype=torch.float32, device=device)
    tloader, vloader, te_loader, ntr, nva, nte = make_loaders(
        bundle, max_future=max(horizons), ft_batch=ft_batch,
        label_fraction=label_fraction, seed=seed, max_context=max_context)
    print(f"  [fam-probe {dataset} s{seed} lf={label_fraction}] "
          f"train={ntr} val={nva} test={nte}", flush=True)

    # pos_weight from train
    n_pos, n_tot = 0, 0
    for ctx, ctx_m, tte, t_idx in tloader:
        y = build_label_surface(tte.unsqueeze(1), h_t.cpu()).squeeze(1)
        n_pos += y.sum().item(); n_tot += y.numel()
    pw = torch.tensor(max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1))),
                      device=device)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ft_epochs)
    best_state, best_val, wait = None, float('inf'), 0
    for ep in range(ft_epochs):
        head.train()
        losses = []
        for ctx, ctx_m, tte, t_idx in tloader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            with torch.no_grad():
                h = model.encoder(ctx, ctx_m).detach()
            logits = head(h)
            y = build_label_surface(tte.unsqueeze(1).to(device), h_t).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sch.step()
        head.eval()
        vl = []
        with torch.no_grad():
            for ctx, ctx_m, tte, t_idx in vloader:
                ctx, ctx_m = ctx.to(device), ctx_m.to(device)
                h = model.encoder(ctx, ctx_m)
                logits = head(h)
                y = build_label_surface(tte.unsqueeze(1).to(device), h_t).squeeze(1)
                vl.append(F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pw).item())
        val_loss = float(np.mean(vl))
        if val_loss < best_val:
            best_val = val_loss; best_state = copy.deepcopy(head.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= 8: break
    head.load_state_dict(best_state)

    head.eval()
    p_list, y_list, t_list = [], [], []
    with torch.no_grad():
        for ctx, ctx_m, tte, t_idx in te_loader:
            ctx, ctx_m = ctx.to(device), ctx_m.to(device)
            h = model.encoder(ctx, ctx_m)
            p = torch.sigmoid(head(h))
            y = build_label_surface(tte.unsqueeze(1).to(device), h_t).squeeze(1)
            p_list.append(p.cpu().numpy())
            y_list.append(y.cpu().numpy())
            t_list.append(t_idx.numpy())
    p_surf = np.concatenate(p_list); y_surf = np.concatenate(y_list)
    t_idx_all = np.concatenate(t_list)
    h_metrics = honest_metrics(p_surf, y_surf, horizons)
    suffix = f'_lf{int(label_fraction*100)}' if label_fraction < 1.0 else ''
    tag = f'{dataset}_fam-probe{suffix}_s{seed}'
    np.savez(SURF_DIR / f'{tag}.npz', p_surface=p_surf,
             y_surface=y_surf.astype(np.int8), horizons=horizons,
             t_index=t_idx_all)
    if dataset == 'FD001' and seed == 42 and label_fraction == 1.0:
        render_surface_panel(p_surf, y_surf, horizons, t_idx_all,
                             PNG_DIR / f'{tag}.png', tag=tag, sort_by_tte=True)
    print(f"  → h-AUROC={h_metrics['mean_h_auroc']:.4f}", flush=True)
    return {'tag': tag, **h_metrics, 'pos_weight': float(pw)}


# ---------------------------------------------------------------------------
# FAM-predft and FAM-mlp-rand
# ---------------------------------------------------------------------------

def fam_predft(dataset: str, seed: int, horizons: List[int],
               label_fraction: float = 1.0, random_init_predictor: bool = False,
               ft_epochs: int = 30, ft_batch: int = 128,
               max_context: int = 512) -> Dict:
    torch.manual_seed(seed); np.random.seed(seed)
    bundle, norm_mode = get_bundle(dataset)
    pre_ckpt = find_pretrain_ckpt(dataset, norm_mode, seed)
    if pre_ckpt is None:
        return None
    model = FAM(n_channels=bundle['n_channels'], patch_size=16, d_model=256,
                n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                norm_mode=norm_mode, predictor_kind='mlp',
                event_head_kind='discrete_hazard')
    sd = torch.load(pre_ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=False)
    if random_init_predictor:
        # Re-initialize predictor + event head from scratch (encoder stays).
        for module in (model.predictor, model.event_head):
            for p in module.parameters():
                if p.dim() >= 2:
                    nn.init.kaiming_uniform_(p, a=5 ** 0.5)
                else:
                    nn.init.zeros_(p)
    model.to(device)

    tloader, vloader, te_loader, ntr, nva, nte = make_loaders(
        bundle, max_future=max(horizons), ft_batch=ft_batch,
        label_fraction=label_fraction, seed=seed, max_context=max_context)
    variant = 'fam-mlp-rand' if random_init_predictor else 'fam-predft'
    suffix = f'_lf{int(label_fraction*100)}' if label_fraction < 1.0 else ''
    tag = f'{dataset}_{variant}{suffix}_s{seed}'
    print(f"  [{tag}] train={ntr} val={nva} test={nte}", flush=True)

    ft_out = finetune_v30(model, tloader, vloader, eval_horizons=horizons,
                          max_horizon=max(horizons), train_horizons_dense=0,
                          mode='pred_ft', n_epochs=ft_epochs, patience=8,
                          device=device, seed=seed)
    eval_out = evaluate(model, te_loader, horizons, mode='pred_ft', device=device)
    p_surf, y_surf = eval_out['p_surface'], eval_out['y_surface']
    h_metrics = honest_metrics(p_surf, y_surf, horizons)
    np.savez(SURF_DIR / f'{tag}.npz', p_surface=p_surf,
             y_surface=y_surf.astype(np.int8), horizons=horizons,
             t_index=eval_out['t_index'])
    if dataset == 'FD001' and seed == 42 and label_fraction == 1.0:
        render_surface_panel(p_surf, y_surf, horizons, eval_out['t_index'],
                             PNG_DIR / f'{tag}.png', tag=tag, sort_by_tte=True)
    print(f"  → h-AUROC={h_metrics['mean_h_auroc']:.4f}", flush=True)
    return {'tag': tag, **h_metrics, 'best_val': float(ft_out['best_val'])}


# ---------------------------------------------------------------------------
# Chr2-probe and Chr2-mlp (load cached features)
# ---------------------------------------------------------------------------

def load_chr2_features(dataset: str, seed: int):
    p = CHR_FEAT_DIR / f'{dataset}_s{seed}_chronos2.pt'
    if not p.exists():
        return None
    cache = torch.load(p, map_location='cpu')
    return cache


def chr2_probe(dataset: str, seed: int, horizons: List[int],
               head_kind: str = 'linear', ft_epochs: int = 50) -> Dict:
    """head_kind: 'linear' = LinearProbeHead. 'mlp' = Chr2MLP (dt-conditioned)."""
    torch.manual_seed(seed); np.random.seed(seed)
    cache = load_chr2_features(dataset, seed)
    if cache is None:
        print(f"  SKIP chr2-{head_kind}: no cached features for {dataset} s{seed}",
              flush=True)
        return None
    Xtr, ytr_tte, ttr_idx = cache['tr']
    Xva, yva_tte, _ = cache['va']
    Xte, yte_tte, te_idx = cache['te']
    h_t = torch.tensor(horizons, dtype=torch.float32, device=device)
    ytr = build_label_surface(ytr_tte.unsqueeze(1), h_t.cpu()).squeeze(1).to(device)
    yva = build_label_surface(yva_tte.unsqueeze(1), h_t.cpu()).squeeze(1).to(device)
    yte = build_label_surface(yte_tte.unsqueeze(1), h_t.cpu()).squeeze(1).to(device)
    Xtr, Xva, Xte = Xtr.to(device), Xva.to(device), Xte.to(device)

    n_pos = ytr.sum().item(); n_tot = ytr.numel()
    pw = torch.tensor(max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1))),
                      device=device)

    if head_kind == 'linear':
        head = LinearProbeHead(d_in=Xtr.shape[1], n_horizons=len(horizons)).to(device)
    else:
        head = Chr2MLP(d_input=Xtr.shape[1], d_hidden=256).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ft_epochs)
    B = 2048
    best_state, best_val, wait = None, float('inf'), 0
    for ep in range(ft_epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=device)
        losses = []
        for i in range(0, len(Xtr), B):
            idx = perm[i:i+B]
            x = Xtr[idx]; y = ytr[idx]
            if head_kind == 'linear':
                logits = head(x)
            else:
                dt_grid = h_t.unsqueeze(0).expand(x.shape[0], -1)
                logits = head(x, dt_grid)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        sch.step()
        head.eval()
        with torch.no_grad():
            if head_kind == 'linear':
                lv = head(Xva)
            else:
                dt_grid = h_t.unsqueeze(0).expand(Xva.shape[0], -1)
                lv = head(Xva, dt_grid)
            vl = F.binary_cross_entropy_with_logits(lv, yva, pos_weight=pw).item()
        if vl < best_val:
            best_val = vl; best_state = copy.deepcopy(head.state_dict()); wait = 0
        else:
            wait += 1
            if wait >= 8: break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        if head_kind == 'linear':
            p_te = torch.sigmoid(head(Xte)).cpu().numpy()
        else:
            dt_grid = h_t.unsqueeze(0).expand(Xte.shape[0], -1)
            p_te = torch.sigmoid(head(Xte, dt_grid)).cpu().numpy()
    yte_np = yte.cpu().numpy().astype(np.int32)
    h_metrics = honest_metrics(p_te, yte_np, horizons)
    tag = f'{dataset}_chr2-{head_kind}_s{seed}'
    np.savez(SURF_DIR / f'{tag}.npz', p_surface=p_te,
             y_surface=yte_np.astype(np.int8), horizons=horizons,
             t_index=te_idx.numpy())
    if dataset == 'FD001' and seed == 42:
        render_surface_panel(p_te, yte_np, horizons, te_idx.numpy(),
                             PNG_DIR / f'{tag}.png', tag=tag, sort_by_tte=True)
    print(f"  → h-AUROC={h_metrics['mean_h_auroc']:.4f}", flush=True)
    return {'tag': tag, **h_metrics, 'best_val': float(best_val)}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    results = {v: {ds: {} for ds in DATASETS}
               for v in ['fam-probe', 'chr2-probe', 'fam-predft', 'chr2-mlp', 'fam-mlp-rand']}
    results_lf = {v: {ds: {} for ds in LABEL_FRAC_DATASETS}
                  for v in ['fam-predft-lf10', 'fam-mlp-rand-lf10']}

    for ds in DATASETS:
        H = SPARSE_HORIZONS[ds]
        for sd in SEEDS:
            print(f"\n=== {ds} seed {sd} ===", flush=True)
            for variant, fn, kw in [
                ('fam-probe', fam_probe, {}),
                ('chr2-probe', lambda d, s, h: chr2_probe(d, s, h, 'linear'), {}),
                ('fam-predft', fam_predft, {}),
                ('chr2-mlp', lambda d, s, h: chr2_probe(d, s, h, 'mlp'), {}),
                ('fam-mlp-rand', lambda d, s, h: fam_predft(d, s, h, random_init_predictor=True), {}),
            ]:
                try:
                    r = fn(ds, sd, H)
                    if r is not None:
                        results[variant][ds][sd] = r
                except Exception as e:
                    print(f"  ERROR {variant} {ds} s{sd}: {e}", flush=True)
                    import traceback; traceback.print_exc()
                # Persist after each variant
                _persist(results, results_lf)

        # 10% labels for FD001/MBA
        if ds in LABEL_FRAC_DATASETS:
            for sd in SEEDS:
                for variant, fn, rkey in [
                    ('fam-predft-lf10',
                     lambda d, s, h: fam_predft(d, s, h, label_fraction=0.1),
                     'fam-predft-lf10'),
                    ('fam-mlp-rand-lf10',
                     lambda d, s, h: fam_predft(d, s, h, label_fraction=0.1,
                                                random_init_predictor=True),
                     'fam-mlp-rand-lf10'),
                ]:
                    try:
                        r = fn(ds, sd, H)
                        if r is not None:
                            results_lf[rkey][ds][sd] = r
                    except Exception as e:
                        print(f"  ERROR {variant} {ds} s{sd}: {e}", flush=True)
                    _persist(results, results_lf)

    # Summarize
    summary = {'datasets': DATASETS, 'seeds': SEEDS, 'horizons_per_dataset': SPARSE_HORIZONS,
               'time_total_s': time.time() - t0}
    for v, ds_d in results.items():
        summary[v + '_hauroc'] = {ds: _agg(seed_d) for ds, seed_d in ds_d.items()}
    for v, ds_d in results_lf.items():
        summary[v + '_hauroc'] = {ds: _agg(seed_d) for ds, seed_d in ds_d.items()}

    # Decision logic
    interp = _interpret(summary)
    summary['interpretation'] = interp['text']
    summary['main_table_variants'] = interp['main_table_variants']
    out = RES_DIR / 'phase1_decision.json'
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}\n", flush=True)
    print(json.dumps(summary, indent=2))


def _agg(seed_d: dict) -> dict:
    if not seed_d:
        return {'mean': None, 'std': None, 'n': 0, 'per_seed': {}}
    aurocs = [r['mean_h_auroc'] for r in seed_d.values()]
    return {'mean': float(np.mean(aurocs)),
            'std': float(np.std(aurocs, ddof=1)) if len(aurocs) > 1 else None,
            'n': len(aurocs),
            'per_seed': {sd: r['mean_h_auroc'] for sd, r in seed_d.items()}}


def _interpret(summary: dict) -> dict:
    """Compare FAM-probe vs Chr2-probe vs FAM-predft vs Chr2-mlp."""
    lines = []
    chosen = ['FAM-predft', 'Chr2-probe']  # default
    for ds in DATASETS:
        try:
            fp = summary['fam-probe_hauroc'][ds].get('mean')
            cp = summary['chr2-probe_hauroc'][ds].get('mean')
            fm = summary['fam-predft_hauroc'][ds].get('mean')
            cm = summary['chr2-mlp_hauroc'][ds].get('mean')
            fr = summary['fam-mlp-rand_hauroc'][ds].get('mean')
        except KeyError:
            continue
        lines.append(f"{ds}: FAM-probe={fp}, Chr2-probe={cp}, "
                     f"FAM-predft={fm}, Chr2-mlp={cm}, FAM-mlp-rand={fr}")
        if fp is not None and cp is not None:
            if fp > cp:
                lines.append(f"  → {ds}: FAM encoder beats Chronos-2 encoder at matched (linear) head capacity")
            else:
                lines.append(f"  → {ds}: Chronos-2 encoder ≥ FAM encoder at matched (linear) head capacity")
        if fm is not None and fr is not None and fm > fr:
            lines.append(f"  → {ds}: pretrained predictor init helps over random init "
                         f"({fm:.4f} vs {fr:.4f})")
    return {'text': '\n'.join(lines), 'main_table_variants': chosen}


def _persist(results, results_lf):
    serial = {}
    for v, ds_d in results.items():
        serial[v] = {ds: {str(sd): r for sd, r in seed_d.items()}
                     for ds, seed_d in ds_d.items()}
    for v, ds_d in results_lf.items():
        serial[v] = {ds: {str(sd): r for sd, r in seed_d.items()}
                     for ds, seed_d in ds_d.items()}
    out = P1_DIR / 'progress.json'
    with open(out, 'w') as f:
        json.dump(serial, f, indent=2, default=str)


if __name__ == '__main__':
    main()

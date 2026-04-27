"""V32 Phase 2: MSE-RUL probe on the frozen v30/v29 encoder for C-MAPSS.

For each (dataset in {FD001, FD002, FD003}, seed in {42, 123, 456}):
  1. Load v27 LOADERS bundle, apply _global_zscore (matches encoder pretrain).
  2. Load v29-era pretrained encoder, freeze.
  3. Extract h_t at many context-end positions per train engine; pair with
     piecewise-linear RUL labels (cap=125).
  4. Train an MLP RUL-regression head with MSE / Huber loss.
  5. Evaluate on the standard NASA last-cycle protocol: predict RUL at the
     last cycle of each test engine; report RMSE / MAE / NASA score.

Hidden-dim sweep H in {64, 128, 256, 512}, plus 2-layer MLP variant, plus
Huber loss variant. Best config is selected by val RMSE per (dataset, seed).

Also runs a 10% label-fraction probe with the best architecture.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FAM = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V32 = FAM / 'experiments/v32'
RESULTS = V32 / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
FEAT_CACHE = V32 / 'feat_cache'
FEAT_CACHE.mkdir(parents=True, exist_ok=True)

for sub in ['', 'experiments/v24', 'experiments/v11', 'experiments/v27',
            'experiments/v29', 'experiments/v30']:
    p = str(FAM / sub) if sub else str(FAM)
    if p not in sys.path:
        sys.path.insert(0, p)

from model import FAM as FAMModel  # noqa: E402
from _runner import LOADERS, _global_zscore  # v27  noqa: E402
from data_utils import load_raw  # v11; for raw test_rul  noqa: E402

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RUL_CAP = 125.0
DATASETS = ['FD001', 'FD002', 'FD003']
SEEDS = [42, 123, 456]
N_CHANNELS = 14
PATCH = 16


def find_pretrain(dataset: str, seed: int) -> Path:
    cands = [
        FAM / f'experiments/v29/ckpts/{dataset}_none_s{seed}_pretrain.pt',
        FAM / f'experiments/v27/ckpts/{dataset}_none_s{seed}_pretrain.pt',
        FAM / f'experiments/v28/ckpts/{dataset}_none_dense20_s{seed}_pretrain.pt',
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f'No pretrain ckpt for {dataset} s{seed}')


def load_encoder(dataset: str, seed: int) -> nn.Module:
    """Load FAM, restore pretrain weights, freeze encoder."""
    model = FAMModel(
        n_channels=N_CHANNELS, patch_size=PATCH, d_model=256,
        n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
        ema_momentum=0.99, predictor_hidden=256,
        norm_mode='none', predictor_kind='mlp',
    )
    sd = torch.load(find_pretrain(dataset, seed), map_location='cpu',
                    weights_only=False)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  [load] missing keys ({len(missing)}): {missing[:3]}...')
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(DEVICE)


@torch.no_grad()
def extract_h_at_endpoints(encoder, x: np.ndarray, endpoints: List[int],
                           batch_size: int = 64) -> np.ndarray:
    """Extract h_t at each context-end (exclusive) given a (T, C) sensor seq.

    `endpoints` are 1-indexed lengths: endpoint=k means context = x[:k].
    Returns (len(endpoints), 256) float32.
    """
    T = x.shape[0]
    feats = []
    for i in range(0, len(endpoints), batch_size):
        chunk = endpoints[i:i + batch_size]
        max_len = max(chunk)
        # Pad each context to max_len with zeros at the END (right-padding),
        # because encoder applies left-aligned causal attention and pads
        # right-side tokens via key_padding_mask.
        ctx = np.zeros((len(chunk), max_len, x.shape[1]), dtype=np.float32)
        msk = np.ones((len(chunk), max_len), dtype=bool)  # True=padding
        for j, k in enumerate(chunk):
            ctx[j, :k] = x[:k]
            msk[j, :k] = False
        ctx_t = torch.from_numpy(ctx).to(DEVICE)
        msk_t = torch.from_numpy(msk).to(DEVICE)
        h = encoder.encoder(ctx_t, mask=msk_t, return_all=False)
        feats.append(h.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, 0)


def build_train_features(encoder, train_engines: Dict[int, np.ndarray],
                         endpoints_per_engine: int = 24,
                         min_ctx: int = 16) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample multiple context-end positions per train engine; return
    (h, rul, eid) arrays for training the MLP head."""
    h_list, y_list, eid_list = [], [], []
    rng = np.random.default_rng(0)  # deterministic feature extraction
    for eid, x in train_engines.items():
        T = x.shape[0]
        if T < min_ctx + 1:
            continue
        # Always include final endpoint for "RUL at last cycle"; sample others.
        K = min(endpoints_per_engine, T - min_ctx + 1)
        ks = sorted(set(rng.integers(min_ctx, T + 1, size=K).tolist() + [T]))
        rul = np.minimum(T - np.array(ks, dtype=np.float32), RUL_CAP).clip(min=0.0)
        # k=T means "all cycles seen, predicting RUL after last cycle" -> RUL=0
        # We actually want RUL AT cycle k-1 (last seen): RUL = min(T - k, cap).
        h = extract_h_at_endpoints(encoder, x, ks)
        h_list.append(h)
        y_list.append(rul)
        eid_list.append(np.full(len(ks), eid, dtype=np.int64))
    return (np.concatenate(h_list, 0),
            np.concatenate(y_list, 0),
            np.concatenate(eid_list, 0))


@torch.no_grad()
def features_for_test(encoder, test_engines: Dict[int, np.ndarray]) -> np.ndarray:
    """Last-cycle h_t per test engine, in order of sorted engine_id."""
    h_list = []
    for eid in sorted(test_engines.keys()):
        x = test_engines[eid]
        h = extract_h_at_endpoints(encoder, x, [x.shape[0]])
        h_list.append(h[0])
    return np.stack(h_list, 0)


# ---------------------------------------------------------------------------
# MLP head + train loop
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, in_dim: int = 256, hidden: int = 128,
                 layers: int = 1, dropout: float = 0.1, rul_cap: float = RUL_CAP):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        if layers == 1:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden, 1),
            )
        elif layers == 2:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden // 2, 1),
            )
        else:
            raise ValueError(layers)
        self.rul_cap = rul_cap

    def forward(self, h):
        return self.net(self.ln(h)).squeeze(-1).clamp(0.0, self.rul_cap)


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    err = pred - true
    s = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(s.sum())


def train_head(h_tr, y_tr, h_va, y_va, h_te, y_te,
               hidden: int, layers: int, loss_kind: str,
               n_epochs: int = 150, batch_size: int = 64,
               lr: float = 1e-3, wd: float = 1e-4, patience: int = 20,
               seed: int = 42) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = MLPHead(in_dim=h_tr.shape[1], hidden=hidden, layers=layers,
                   dropout=0.1).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    h_tr_t = torch.from_numpy(h_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    h_va_t = torch.from_numpy(h_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)
    h_te_t = torch.from_numpy(h_te).to(DEVICE)

    n = h_tr_t.shape[0]
    best_val = float('inf'); best_state = None; bad = 0
    for ep in range(n_epochs):
        head.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = head(h_tr_t[idx])
            tgt = y_tr_t[idx]
            if loss_kind == 'mse':
                loss = F.mse_loss(pred, tgt)
            elif loss_kind == 'huber':
                loss = F.smooth_l1_loss(pred, tgt, beta=10.0)
            else:
                raise ValueError(loss_kind)
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            v_pred = head(h_va_t)
            v_rmse = float(torch.sqrt(F.mse_loss(v_pred, y_va_t)))
        sch.step(v_rmse)
        if v_rmse < best_val - 1e-4:
            best_val = v_rmse
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        t_pred = head(h_te_t).cpu().numpy()
    rmse = float(np.sqrt(np.mean((t_pred - y_te) ** 2)))
    mae = float(np.mean(np.abs(t_pred - y_te)))
    score = nasa_score(t_pred, y_te)
    return {
        'val_rmse': best_val, 'rmse': rmse, 'mae': mae, 'nasa_score': score,
        'epochs': ep + 1, 'hidden': hidden, 'layers': layers,
        'loss': loss_kind,
    }


# ---------------------------------------------------------------------------
# Main per-(dataset, seed) workflow
# ---------------------------------------------------------------------------

def get_test_rul(dataset: str) -> Dict[int, float]:
    """NASA ground-truth RUL at the last cycle of each test engine."""
    _train, _test, rul = load_raw(dataset)
    test_eids = sorted(_test['engine_id'].unique().tolist())
    return {int(eid): float(min(rul[i], RUL_CAP)) for i, eid in enumerate(test_eids)}


def get_engines_from_bundle(bundle: dict, key: str) -> Dict[int, np.ndarray]:
    """Convert ft_train / ft_val / ft_test list-of-dicts into {eid: array}."""
    out = {}
    for d in bundle[key]:
        out[int(d['entity_id'])] = d['test'].astype(np.float32)
    return out


def run_one(dataset: str, seed: int, label_frac: float = 1.0,
            cache_features: bool = True) -> Dict:
    print(f'\n=== {dataset} seed={seed} lf={label_frac} ===')
    t0 = time.time()
    cache_path = FEAT_CACHE / f'{dataset}_s{seed}.npz'
    test_rul_map = get_test_rul(dataset)

    # Always need data bundle for engine sequences
    bundle = LOADERS[dataset]()
    bundle = _global_zscore(bundle)
    train_eng = get_engines_from_bundle(bundle, 'ft_train')
    val_eng = get_engines_from_bundle(bundle, 'ft_val')
    test_eng = get_engines_from_bundle(bundle, 'ft_test')

    # Apply label fraction by engine subset (train engines only)
    if label_frac < 1.0:
        rng = np.random.default_rng(seed)
        ids = sorted(train_eng.keys())
        n_keep = max(2, int(round(label_frac * len(ids))))
        keep = set(rng.choice(ids, size=n_keep, replace=False).tolist())
        train_eng = {i: train_eng[i] for i in keep}

    if cache_features and cache_path.exists() and label_frac == 1.0:
        z = np.load(cache_path)
        h_tr, y_tr, eid_tr = z['h_tr'], z['y_tr'], z['eid_tr']
        h_va, y_va = z['h_va'], z['y_va']
        h_te = z['h_te']
        te_eids = z['te_eids']
        print(f'  loaded features from cache: '
              f'tr={h_tr.shape}, va={h_va.shape}, te={h_te.shape}')
    else:
        encoder = load_encoder(dataset, seed)
        print(f'  encoder loaded ({sum(p.numel() for p in encoder.parameters())/1e6:.2f}M)')

        h_tr, y_tr, eid_tr = build_train_features(encoder, train_eng,
                                                  endpoints_per_engine=24)
        h_va, y_va, _ = build_train_features(encoder, val_eng,
                                             endpoints_per_engine=8)
        h_te = features_for_test(encoder, test_eng)
        te_eids = np.array(sorted(test_eng.keys()), dtype=np.int64)
        print(f'  features extracted: '
              f'tr={h_tr.shape}, va={h_va.shape}, te={h_te.shape}')
        if cache_features and label_frac == 1.0:
            np.savez_compressed(cache_path,
                                h_tr=h_tr, y_tr=y_tr, eid_tr=eid_tr,
                                h_va=h_va, y_va=y_va,
                                h_te=h_te, te_eids=te_eids)
        del encoder
        torch.cuda.empty_cache()

    y_te = np.array([test_rul_map[int(e)] for e in te_eids], dtype=np.float32)

    # Hidden-dim + loss + layers sweep, picked by val RMSE
    sweep_runs = []
    for hidden in [64, 128, 256, 512]:
        for layers in [1, 2]:
            for loss_kind in ['mse', 'huber']:
                r = train_head(h_tr, y_tr, h_va, y_va, h_te, y_te,
                               hidden=hidden, layers=layers, loss_kind=loss_kind,
                               seed=seed)
                r['config'] = f'h{hidden}_L{layers}_{loss_kind}'
                sweep_runs.append(r)
    best = min(sweep_runs, key=lambda r: r['val_rmse'])
    print(f'  best: {best["config"]} val_rmse={best["val_rmse"]:.2f} '
          f'test_rmse={best["rmse"]:.2f} mae={best["mae"]:.2f} '
          f'nasa={best["nasa_score"]:.0f}')
    elapsed = time.time() - t0
    return {
        'dataset': dataset, 'seed': seed, 'label_frac': label_frac,
        'n_train_engines': len(train_eng),
        'best': best,
        'sweep': sweep_runs,
        'elapsed_sec': elapsed,
    }


def aggregate(results: List[Dict]) -> Dict:
    """Aggregate results into per-(dataset, lf) summaries."""
    out = {}
    for ds in DATASETS:
        for lf in sorted({r['label_frac'] for r in results if r['dataset'] == ds}):
            sub = [r for r in results if r['dataset'] == ds and r['label_frac'] == lf]
            rmses = np.array([r['best']['rmse'] for r in sub])
            maes = np.array([r['best']['mae'] for r in sub])
            scores = np.array([r['best']['nasa_score'] for r in sub])
            key = f'{ds}_lf{int(lf*100)}'
            out[key] = {
                'rmse_mean': float(rmses.mean()), 'rmse_std': float(rmses.std(ddof=1)) if len(rmses) > 1 else 0.0,
                'mae_mean': float(maes.mean()), 'mae_std': float(maes.std(ddof=1)) if len(maes) > 1 else 0.0,
                'nasa_mean': float(scores.mean()), 'nasa_std': float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
                'seeds': [r['seed'] for r in sub],
                'rmse_per_seed': rmses.tolist(),
                'best_configs': [r['best']['config'] for r in sub],
            }
    return out


def main():
    all_runs = []
    # Phase A: 100% labels, all (dataset, seed)
    for ds in DATASETS:
        for seed in SEEDS:
            try:
                all_runs.append(run_one(ds, seed, label_frac=1.0))
            except Exception as e:
                import traceback; traceback.print_exc()
                all_runs.append({
                    'dataset': ds, 'seed': seed, 'label_frac': 1.0,
                    'error': str(e),
                })
            with open(RESULTS / 'rmse_probe.json', 'w') as fp:
                json.dump({'runs': all_runs, 'agg': aggregate([r for r in all_runs if 'error' not in r])}, fp, indent=2)

    # Phase B: 10% labels, same datasets/seeds
    for ds in DATASETS:
        for seed in SEEDS:
            try:
                all_runs.append(run_one(ds, seed, label_frac=0.1))
            except Exception as e:
                import traceback; traceback.print_exc()
                all_runs.append({
                    'dataset': ds, 'seed': seed, 'label_frac': 0.1,
                    'error': str(e),
                })
            with open(RESULTS / 'rmse_probe.json', 'w') as fp:
                json.dump({'runs': all_runs, 'agg': aggregate([r for r in all_runs if 'error' not in r])}, fp, indent=2)

    summary = aggregate([r for r in all_runs if 'error' not in r])
    print('\n=== SUMMARY ===')
    for k, v in summary.items():
        print(f'  {k}: RMSE {v["rmse_mean"]:.2f} ± {v["rmse_std"]:.2f}, '
              f'MAE {v["mae_mean"]:.2f}, NASA {v["nasa_mean"]:.0f}')


if __name__ == '__main__':
    main()

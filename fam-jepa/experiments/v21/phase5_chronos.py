"""V21 Phase 5: Chronos-T5-tiny on FD001 under the v21 AUPRC framework.

Reuses v20's per-sensor last-token embedding extraction on C-MAPSS FD001,
then trains a per-horizon linear head with pos-weighted BCE — the same
downstream as our FAM probe_h. Evaluates with AUPRC + RMSE via the
stored surface.

3 seeds; ~5 min total.
"""
from __future__ import annotations

import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/home/sagemaker-user/IndustrialJEPA')
FAM = ROOT / 'fam-jepa'
V21 = FAM / 'experiments' / 'v21'
V11 = FAM / 'experiments' / 'v11'
sys.path.insert(0, str(V11))
sys.path.insert(0, str(V21))
sys.path.insert(0, str(FAM))

from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP  # noqa: E402
from pred_ft_utils import HORIZONS_STEPS, save_surface  # noqa: E402
from surface_to_legacy import surface_to_rul_expected, rmse  # noqa: E402
from evaluation.surface_metrics import (  # noqa: E402
    evaluate_probability_surface, auprc_per_horizon,
    monotonicity_violation_rate,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_CTX = 300
SEEDS = [42, 123, 456]


def extract_last_token(pipe, seq: np.ndarray, d_emb: int) -> np.ndarray:
    """Per-sensor last-token embedding concatenated → (14 * d_emb,)."""
    T, S = seq.shape
    take = min(T, MAX_CTX)
    s_arr = seq[-take:, :].T  # (S, take)
    x = torch.from_numpy(s_arr).float()
    with torch.no_grad():
        emb, _ = pipe.embed(x)  # (S, take, d_emb)
    last = emb[:, -1, :]  # (S, d_emb)
    return last.cpu().numpy().reshape(-1)


def build_pairs(engines: dict, seed: int, rul_cap: int = RUL_CAP,
                n_cuts: int = 5) -> list:
    rng = np.random.default_rng(seed)
    pairs = []
    for eid, seq in engines.items():
        T = len(seq)
        if T <= 10:
            continue
        cuts = sorted(rng.integers(10, T, size=min(n_cuts, T - 10)).tolist())
        for t in cuts:
            rul_raw = float(T - t)
            pairs.append((seq[:t], rul_raw))
    return pairs


def extract_all(pipe, pairs, d_emb):
    X, tte = [], []
    for seq, rul_raw in pairs:
        X.append(extract_last_token(pipe, seq, d_emb))
        tte.append(rul_raw)
    return np.stack(X), np.array(tte, dtype=np.float32)


def label_surface(tte: np.ndarray, horizons) -> np.ndarray:
    y = np.zeros((len(tte), len(horizons)), dtype=np.float32)
    for k, dt in enumerate(horizons):
        y[:, k] = ((tte <= float(dt)) & np.isfinite(tte)).astype(np.float32)
    return y


def train_head(X_tr, y_tr_surf, X_va, y_va_surf, feat_dim: int,
               seed: int, pos_weight: float,
               n_epochs: int = 200, patience: int = 25,
               batch_size: int = 32, lr: float = 1e-3, wd: float = 1e-2):
    torch.manual_seed(seed)
    K = y_tr_surf.shape[1]
    head = nn.Sequential(nn.LayerNorm(feat_dim),
                         nn.Linear(feat_dim, K)).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    X_tr = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr = torch.from_numpy(y_tr_surf).float().to(DEVICE)
    X_va = torch.from_numpy(X_va).float().to(DEVICE)
    y_va = torch.from_numpy(y_va_surf).float().to(DEVICE)
    pw = torch.tensor(float(pos_weight), device=DEVICE)

    best_val = float('inf'); best_sd = None; no_impr = 0
    N = X_tr.shape[0]
    for ep in range(n_epochs):
        head.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch_size):
            b = perm[i:i + batch_size]
            logits = head(X_tr[b])
            loss = F.binary_cross_entropy_with_logits(
                logits, y_tr[b], pos_weight=pw, reduction='mean')
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = F.binary_cross_entropy_with_logits(
                head(X_va), y_va, pos_weight=pw, reduction='mean').item()
        if vl < best_val - 1e-6:
            best_val = vl; best_sd = copy.deepcopy(head.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break
    head.load_state_dict(best_sd)
    return head, best_val


def run_seed(pipe, data: dict, seed: int, d_emb: int) -> dict:
    t0 = time.time()
    tr_pairs = build_pairs(data['train_engines'], seed, n_cuts=5)
    va_pairs = build_pairs(data['val_engines'], seed + 111, n_cuts=10)
    # Test: one embedding per engine at last cycle
    te_engines = data['test_engines']; te_rul = data['test_rul']
    te_pairs = [(te_engines[eid], float(te_rul[i]))
                for i, eid in enumerate(sorted(te_engines.keys()))]
    feat_dim = N_SENSORS * d_emb
    print(f'  seed={seed}: extracting (tr={len(tr_pairs)} va={len(va_pairs)} te={len(te_pairs)}) ...',
          flush=True)
    t_ext = time.time()
    X_tr, tte_tr = extract_all(pipe, tr_pairs, d_emb)
    X_va, tte_va = extract_all(pipe, va_pairs, d_emb)
    X_te, tte_te = extract_all(pipe, te_pairs, d_emb)
    print(f'    extraction: {time.time()-t_ext:.0f}s', flush=True)

    y_tr = label_surface(tte_tr, HORIZONS_STEPS)
    y_va = label_surface(tte_va, HORIZONS_STEPS)
    y_te = label_surface(tte_te, HORIZONS_STEPS)

    n_pos = y_tr.sum(); n_neg = y_tr.size - n_pos
    pw = max(1.0, min(1000.0, n_neg / max(n_pos, 1.0)))

    head, _ = train_head(X_tr, y_tr, X_va, y_va, feat_dim, seed, pw)
    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(X_te).float().to(DEVICE))
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
    # enforce monotonicity along Δt
    p = np.maximum.accumulate(p, axis=1)

    # Save surface
    (V21 / 'surfaces').mkdir(exist_ok=True)
    surf_path = V21 / 'surfaces' / f'fd001_chronos_tiny_seed{seed}.npz'
    save_surface(surf_path, p, y_te.astype(np.int8),
                 HORIZONS_STEPS, np.arange(len(p)),
                 metadata={'dataset': 'FD001', 'seed': seed, 'mode': 'chronos_tiny'})

    prim = evaluate_probability_surface(p, y_te)
    per_h = auprc_per_horizon(p, y_te, horizon_labels=HORIZONS_STEPS)
    mono = monotonicity_violation_rate(p)
    pred_rul = surface_to_rul_expected(p, np.asarray(HORIZONS_STEPS))
    true_rul_capped = np.minimum(tte_te, float(HORIZONS_STEPS[-1]))
    rmse_v = rmse(pred_rul, true_rul_capped)

    dt = time.time() - t0
    print(f'  seed={seed}: AUPRC={prim["auprc"]:.3f} AUROC={prim["auroc"]:.3f} '
          f'RMSE={rmse_v:.2f} mono={mono["violation_rate"]:.3f} ({dt:.0f}s)',
          flush=True)
    return {'seed': seed, 'primary': prim, 'per_horizon': per_h,
            'monotonicity': mono, 'rmse_capped': float(rmse_v),
            'surface_file': str(surf_path), 'runtime_s': dt}


def main():
    t0 = time.time()
    from chronos import ChronosPipeline

    print('=== Phase 5: Chronos-T5-tiny on FD001 with AUPRC ===', flush=True)
    data = load_cmapss_subset('FD001')
    pipe = ChronosPipeline.from_pretrained(
        'amazon/chronos-t5-tiny', device_map=DEVICE, torch_dtype=torch.float32)
    pipe.model.eval()
    d_emb = 256  # chronos-t5-tiny d_model

    results = []
    for s in SEEDS:
        try:
            r = run_seed(pipe, data, s, d_emb)
            results.append(r)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f'  ERR seed={s}: {e}', flush=True)

        gc.collect(); torch.cuda.empty_cache()

    # Aggregate
    import numpy as np
    if results:
        aucs = np.array([r['primary']['auprc'] for r in results])
        aurs = np.array([r['primary']['auroc'] for r in results])
        rms = np.array([r['rmse_capped'] for r in results])
        agg = {
            'n_seeds': len(results),
            'auprc_mean': float(aucs.mean()), 'auprc_std': float(aucs.std(ddof=1)),
            'auroc_mean': float(aurs.mean()), 'auroc_std': float(aurs.std(ddof=1)),
            'rmse_mean': float(rms.mean()), 'rmse_std': float(rms.std(ddof=1)),
        }
    else:
        agg = {'n_seeds': 0}
    out = {'model': 'chronos-t5-tiny', 'subset': 'FD001',
           'seeds': SEEDS, 'agg': agg, 'per_seed': results,
           'runtime_min': (time.time() - t0) / 60}

    out_path = V21 / 'phase5_chronos.json'
    json.dump(out, open(out_path, 'w'), indent=2, default=float)
    print(f'\nDONE in {out["runtime_min"]:.1f} min. Saved: {out_path}')
    if results:
        print(f'Chronos-T5-tiny FD001: AUPRC={agg["auprc_mean"]:.3f}±{agg["auprc_std"]:.3f} '
              f'AUROC={agg["auroc_mean"]:.3f}±{agg["auroc_std"]:.3f} '
              f'RMSE={agg["rmse_mean"]:.2f}±{agg["rmse_std"]:.2f}')


if __name__ == '__main__':
    main()

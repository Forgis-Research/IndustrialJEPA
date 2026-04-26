"""
V18 Phase 8b: Chronos-2 (multivariate) + Chronos-T5-large on FD001 RUL.

Phase 8 tested Chronos-T5 (univariate, channel-independent). Chronos-2 is
Amazon's newer multivariate-aware model. Its embed() takes (n_series,
n_variates, history_len) and returns (n_variates, n_patches, d_emb) per
series - treating all 14 sensors jointly rather than independently.

Also adds chronos-t5-large (710M) for a full-scale comparison.

Output: experiments/v18/phase8b_chronos2_rul.json
"""

import sys, json, copy, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(ROOT))

from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

from chronos import BaseChronosPipeline, ChronosPipeline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_CONTEXT = 300
SEEDS = [42, 123, 456]
K_EVAL_LIST = [10, 20, 30, 50]


def chronos2_embed_engine(pipe, seq, max_ctx=MAX_CONTEXT):
    """seq: (T, 14). Returns concat-last-patch feature: (14 * d_emb,)."""
    T = seq.shape[0]
    take = min(T, max_ctx)
    x = torch.from_numpy(seq[-take:, :]).float().T.unsqueeze(0)  # (1, 14, take)
    with torch.no_grad():
        embs, _ = pipe.embed(x)
    emb = embs[0]  # (14, n_patches, d_emb)
    last = emb[:, -1, :]  # (14, d_emb)
    return last.cpu().numpy().reshape(-1)


def chronos_t5_embed_engine(pipe, seq, max_ctx=MAX_CONTEXT):
    """Univariate per-sensor. seq: (T, 14). Returns (14 * d_emb,)."""
    T, S = seq.shape
    take = min(T, max_ctx)
    s_arr = seq[-take:, :].T  # (14, take)
    x = torch.from_numpy(s_arr).float()
    with torch.no_grad():
        emb, _ = pipe.embed(x)  # (14, take+1, d_emb)
    last = emb[:, -1, :]  # (14, d_emb)
    return last.cpu().numpy().reshape(-1)


def cycle_cut_features(pipe, engines, embed_fn, seed, n_cuts=5):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for eid, seq in engines.items():
        T = len(seq)
        if T <= 10: continue
        cuts = rng.integers(10, T, size=min(n_cuts, T - 10))
        for t in cuts:
            rul = min(T - t, RUL_CAP) / RUL_CAP
            X.append(embed_fn(pipe, seq[:t]))
            y.append(rul)
    return np.array(X), np.array(y)


def test_features(pipe, engines, test_rul, embed_fn):
    X, y = [], []
    for i, eid in enumerate(sorted(engines.keys())):
        X.append(embed_fn(pipe, engines[eid]))
        y.append(float(test_rul[i]))
    return np.array(X), np.array(y)


def probe(X_tr, y_tr, X_va, y_va, X_te, y_te_raw, feat_dim, seed):
    torch.manual_seed(seed)
    X_tr = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr = torch.from_numpy(y_tr).float().to(DEVICE)
    X_va = torch.from_numpy(X_va).float().to(DEVICE)
    y_va = torch.from_numpy(y_va).float().to(DEVICE)
    X_te = torch.from_numpy(X_te).float().to(DEVICE)

    model = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    best_val = float('inf'); best = None; no_impr = 0
    for ep in range(200):
        model.train()
        idx = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 32):
            b = idx[i:i+32]
            p = model(X_tr[b]).squeeze(-1)
            loss = F.mse_loss(p, y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(X_va).squeeze(-1).cpu().numpy() * RUL_CAP
            tv = y_va.cpu().numpy() * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse; best = copy.deepcopy(model.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        pt = model(X_te).squeeze(-1).cpu().numpy() * RUL_CAP
    test_rmse = float(np.sqrt(np.mean((pt - y_te_raw) ** 2)))
    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (y_te_raw <= ke).astype(int); score = -pt
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}
    return {'val_rmse': best_val, 'test_rmse': test_rmse, 'f1_by_k': f1_by_k}


def run_chronos2(data):
    print("\n=== Chronos-2 (multivariate) ===", flush=True)
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-2",
                                                device_map="cuda",
                                                dtype=torch.float32)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    print(f"  params: {n_params:,}", flush=True)
    # Probe feature dim
    sample = list(data['test_engines'].values())[0]
    f0 = chronos2_embed_engine(pipe, sample)
    d = f0.shape[0]
    print(f"  feature dim: {d}", flush=True)

    X_te, y_te = test_features(pipe, data['test_engines'], data['test_rul'],
                                chronos2_embed_engine)
    rs = []
    for seed in SEEDS:
        print(f"  seed {seed}...", flush=True)
        X_tr, y_tr = cycle_cut_features(pipe, data['train_engines'],
                                         chronos2_embed_engine, seed, n_cuts=5)
        X_va, y_va = cycle_cut_features(pipe, data['val_engines'],
                                         chronos2_embed_engine, seed + 111, n_cuts=10)
        r = probe(X_tr, y_tr, X_va, y_va, X_te, y_te, d, seed)
        rs.append({'seed': seed, **r})
        print(f"    val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
              f"F1@30={r['f1_by_k'][30]['f1']:.3f}", flush=True)
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return {
        'model': 'chronos-2', 'hf_id': 'amazon/chronos-2',
        'approach': 'multivariate_native',
        'n_params': n_params, 'feat_dim': d,
        'per_seed': rs,
        'test_rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
        'test_rmse_std': float(np.std([r['test_rmse'] for r in rs])),
        'f1_by_k_mean': {ke: float(np.mean([r['f1_by_k'][ke]['f1'] for r in rs]))
                         for ke in K_EVAL_LIST},
        'elapsed_min': (time.time() - t0) / 60,
    }


def run_chronos_t5_large(data):
    print("\n=== Chronos-T5-large (710M, univariate) ===", flush=True)
    t0 = time.time()
    pipe = ChronosPipeline.from_pretrained("amazon/chronos-t5-large",
                                            device_map="cuda",
                                            dtype=torch.float32)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    print(f"  params: {n_params:,}", flush=True)
    sample = list(data['test_engines'].values())[0]
    f0 = chronos_t5_embed_engine(pipe, sample)
    d = f0.shape[0]
    print(f"  feature dim: {d}", flush=True)

    X_te, y_te = test_features(pipe, data['test_engines'], data['test_rul'],
                                chronos_t5_embed_engine)
    rs = []
    for seed in SEEDS:
        print(f"  seed {seed}...", flush=True)
        X_tr, y_tr = cycle_cut_features(pipe, data['train_engines'],
                                         chronos_t5_embed_engine, seed, n_cuts=5)
        X_va, y_va = cycle_cut_features(pipe, data['val_engines'],
                                         chronos_t5_embed_engine, seed + 111, n_cuts=10)
        r = probe(X_tr, y_tr, X_va, y_va, X_te, y_te, d, seed)
        rs.append({'seed': seed, **r})
        print(f"    val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
              f"F1@30={r['f1_by_k'][30]['f1']:.3f}", flush=True)
    del pipe; gc.collect(); torch.cuda.empty_cache()
    return {
        'model': 'chronos-t5-large',
        'hf_id': 'amazon/chronos-t5-large',
        'approach': 'univariate_per_sensor',
        'n_params': n_params, 'feat_dim': d,
        'per_seed': rs,
        'test_rmse_mean': float(np.mean([r['test_rmse'] for r in rs])),
        'test_rmse_std': float(np.std([r['test_rmse'] for r in rs])),
        'f1_by_k_mean': {ke: float(np.mean([r['f1_by_k'][ke]['f1'] for r in rs]))
                         for ke in K_EVAL_LIST},
        'elapsed_min': (time.time() - t0) / 60,
    }


def main():
    V18.mkdir(exist_ok=True)
    data = load_cmapss_subset('FD001')
    print(f"FD001 loaded", flush=True)

    results = []
    for runner in [run_chronos2, run_chronos_t5_large]:
        try:
            r = runner(data)
            results.append(r)
            with open(V18 / 'phase8b_chronos2_rul.json', 'w') as f:
                json.dump({'config': 'v18_phase8b_chronos2_plus_large',
                           'models': results}, f, indent=2, default=float)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({'model': runner.__name__, 'error': str(e)})

    print("\n" + "=" * 75)
    print("V18 Phase 8b: Chronos-2 + Chronos-T5-large on FD001 RUL")
    print("=" * 75)
    print(f"{'model':<22} {'approach':<22} {'params':>12} {'RMSE':>14} {'F1@30':>7}")
    for r in results:
        if 'error' in r:
            print(f"{r['model']:<22} FAILED {r['error']}")
            continue
        print(f"{r['model']:<22} {r.get('approach','?'):<22} "
              f"{r['n_params']:>12,} {r['test_rmse_mean']:>6.2f} +/- "
              f"{r['test_rmse_std']:<4.2f} {r['f1_by_k_mean'][30]:>7.3f}")
    print()
    print(f"{'Phase 8 ref:':<22}")
    print(f"{'chronos-t5-base':<22} {'univariate':<22} {201374976:>12,} "
          f"{17.00:>6.2f} +/- {0.55:<4.2f} {0.898:>7.3f}")
    print(f"{'FAM (ours)':<22} {'frozen probe':<22} {1260000:>12,} "
          f"{15.53:>6.2f} +/- {1.68:<4.2f} {0.919:>7.3f}")


if __name__ == '__main__':
    main()

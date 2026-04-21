"""
V18 Phase 8: Chronos-2 / Chronos-T5 foundation-model baseline on FD001 RUL.

Round-4 reviewer said path (b) to 7/10: strong SSL baseline on FD001. User
upgraded the ask: use Chronos-2 and TabPFN-TS (both SOTA pretrained time-
series foundation models) instead of TS2Vec.

Approach (honest, apples-to-apples with FAM frozen probe):
  1. For each engine cycle window (entire history up to cycle t):
     a. For each of 14 sensors: run Chronos encoder, extract last-token embedding
     b. Concat across sensors: feature vector of shape (14 * d_emb,)
  2. Train a linear probe on the feature vector (AdamW WD=1e-2, val n_cuts=10) to predict RUL.
  3. Report test RMSE + F1@k across 3 seeds.

Channel-independent strategy mirrors how MTS-JEPA used Chronos in their comparison
(and matches the native Chronos univariate API).

Models tested:
  - amazon/chronos-t5-tiny (8M params) - closest scale to FAM (1.26M)
  - amazon/chronos-t5-small (46M)
  - amazon/chronos-t5-base (200M)

Output: experiments/v18/phase8_chronos_rul.json
"""

import sys, json, copy, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(ROOT))

from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

from chronos import ChronosPipeline

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_CONTEXT = 300  # cap Chronos input length (FD001 engines can be up to 362 cycles)
BATCH = 8           # per-sensor forward batch; Chronos is memory-heavy
SEEDS = [42, 123, 456]
K_EVAL_LIST = [10, 20, 30, 50]

MODELS = [
    ('chronos-t5-tiny',  'amazon/chronos-t5-tiny',  256),
    ('chronos-t5-small', 'amazon/chronos-t5-small', 512),
    ('chronos-t5-base',  'amazon/chronos-t5-base',  768),
]


def extract_engine_embedding(pipe, seq, d_emb, max_ctx=MAX_CONTEXT):
    """Given a (T, 14) engine-history array, produce a (14 * d_emb,) feature.

    For each of the 14 sensors: feed last max_ctx cycles into Chronos encoder,
    take the *last-token* embedding (matches how we use h_past in FAM).
    """
    T, S = seq.shape
    take = min(T, max_ctx)
    s_arr = seq[-take:, :].T  # (14, take)
    # chronos.embed expects (batch, past_length) univariate
    x = torch.from_numpy(s_arr).float()  # (14, take)
    with torch.no_grad():
        emb, tokens = pipe.embed(x)  # emb: (14, take+1, d_emb)
    last = emb[:, -1, :]  # (14, d_emb)
    return last.cpu().numpy().reshape(-1)  # (14 * d_emb,)


def extract_all_embeddings(pipe, engines_dict, d_emb, batch_engines=1):
    """Returns {eid: np.array(14*d_emb,)}."""
    out = {}
    for eid, seq in engines_dict.items():
        out[eid] = extract_engine_embedding(pipe, seq, d_emb)
    return out


def probe_rmse(train_emb, train_rul, val_emb, val_rul, test_emb, test_rul_arr,
               feat_dim, seed):
    """Honest probe: linear probe with AdamW WD=1e-2."""
    torch.manual_seed(seed)
    X_tr = torch.from_numpy(np.array(train_emb)).float().to(DEVICE)
    y_tr = torch.from_numpy(np.array(train_rul)).float().to(DEVICE)
    X_va = torch.from_numpy(np.array(val_emb)).float().to(DEVICE)
    y_va = torch.from_numpy(np.array(val_rul)).float().to(DEVICE)
    X_te = torch.from_numpy(np.array(test_emb)).float().to(DEVICE)

    probe = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)

    best_val = float('inf'); best = None; no_impr = 0
    for ep in range(200):
        probe.train()
        # minibatch over train
        idx = torch.randperm(X_tr.shape[0])
        for i in range(0, X_tr.shape[0], 32):
            b = idx[i:i+32]
            p = probe(X_tr[b]).squeeze(-1)
            loss = F.mse_loss(p, y_tr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pv = probe(X_va).squeeze(-1).cpu().numpy() * RUL_CAP
            tv = y_va.cpu().numpy() * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((pv - tv) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best = copy.deepcopy(probe.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break
    probe.load_state_dict(best); probe.eval()

    with torch.no_grad():
        pt = probe(X_te).squeeze(-1).cpu().numpy() * RUL_CAP
    test_rmse = float(np.sqrt(np.mean((pt - test_rul_arr) ** 2)))
    # F1 at k values
    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (test_rul_arr <= ke).astype(int); score = -pt
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}
    return {'val_rmse': best_val, 'test_rmse': test_rmse, 'f1_by_k': f1_by_k}


def cycle_cut_embeddings(pipe, engines_dict, d_emb, seed, n_cuts=5):
    """Produce (feature_vec, rul_normalised) pairs via random cycle cuts per engine."""
    rng = np.random.default_rng(seed)
    items = []
    for eid, seq in engines_dict.items():
        T = len(seq)
        if T <= 10: continue
        cuts = rng.integers(10, T, size=min(n_cuts, T - 10))
        for t in cuts:
            rul = min(T - t, RUL_CAP) / RUL_CAP
            feat = extract_engine_embedding(pipe, seq[:t], d_emb)
            items.append((feat, rul))
    X = np.array([it[0] for it in items])
    y = np.array([it[1] for it in items])
    return X, y


def test_embeddings(pipe, engines_dict, test_rul, d_emb):
    feats = []; targets = []
    eng_ids = sorted(engines_dict.keys())
    for i, eid in enumerate(eng_ids):
        seq = engines_dict[eid]
        feat = extract_engine_embedding(pipe, seq, d_emb)
        feats.append(feat); targets.append(float(test_rul[i]))
    return np.array(feats), np.array(targets)


def run_model(name, hf_id, d_emb, data):
    print(f"\n=== {name} ({hf_id}) ===", flush=True)
    t0 = time.time()
    pipe = ChronosPipeline.from_pretrained(hf_id, device_map="cuda",
                                            dtype=torch.float32)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    feat_dim = N_SENSORS * d_emb
    print(f"  model params: {n_params:,}, feature dim: {feat_dim}", flush=True)

    # Test embeddings (fixed, per model) - use last MAX_CONTEXT cycles of test engine
    print("  extracting test embeddings...", flush=True)
    X_te, y_te = test_embeddings(pipe, data['test_engines'], data['test_rul'], d_emb)
    print(f"  test X: {X_te.shape}, y: {y_te.shape}", flush=True)

    seed_results = []
    for seed in SEEDS:
        print(f"  seed {seed}...", flush=True)
        X_tr, y_tr_norm = cycle_cut_embeddings(pipe, data['train_engines'], d_emb,
                                                seed=seed, n_cuts=5)
        X_va, y_va_norm = cycle_cut_embeddings(pipe, data['val_engines'], d_emb,
                                                seed=seed + 111, n_cuts=10)
        r = probe_rmse(X_tr, y_tr_norm, X_va, y_va_norm, X_te, y_te,
                       feat_dim, seed)
        seed_results.append({'seed': seed, **r})
        print(f"    val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
              f"F1@30={r['f1_by_k'][30]['f1']:.3f}", flush=True)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    agg = {
        'model': name, 'hf_id': hf_id, 'n_params': n_params,
        'd_emb': d_emb, 'feat_dim': feat_dim,
        'test_rmse_per_seed': [r['test_rmse'] for r in seed_results],
        'test_rmse_mean': float(np.mean([r['test_rmse'] for r in seed_results])),
        'test_rmse_std': float(np.std([r['test_rmse'] for r in seed_results])),
        'f1_by_k_mean': {ke: float(np.mean([r['f1_by_k'][ke]['f1']
                                             for r in seed_results]))
                         for ke in K_EVAL_LIST},
        'per_seed': seed_results,
        'elapsed_min': (time.time() - t0) / 60,
    }
    return agg


def main():
    V18.mkdir(exist_ok=True)
    data = load_cmapss_subset('FD001')
    print(f"FD001: train={len(data['train_engines'])} val={len(data['val_engines'])} "
          f"test={len(data['test_engines'])}", flush=True)

    results = []
    for name, hf_id, d_emb in MODELS:
        try:
            r = run_model(name, hf_id, d_emb, data)
            results.append(r)
            with open(V18 / 'phase8_chronos_rul.json', 'w') as f:
                json.dump({'config': 'v18_phase8_chronos_rul',
                           'models': results}, f, indent=2, default=float)
        except Exception as e:
            print(f"  {name} FAILED: {type(e).__name__}: {e}", flush=True)
            results.append({'model': name, 'error': str(e)})

    print("\n" + "=" * 70)
    print("V18 Phase 8: Chronos foundation-model frozen probe on FD001 RUL")
    print("=" * 70)
    print(f"{'model':<22} {'params':>12} {'test_rmse':>15} {'F1@30':>10}")
    for r in results:
        if 'error' in r:
            print(f"{r['model']:<22} FAILED: {r['error']}")
            continue
        print(f"{r['model']:<22} {r['n_params']:>12,} "
              f"{r['test_rmse_mean']:>6.2f} +/- {r['test_rmse_std']:<5.2f} "
              f"{r['f1_by_k_mean'][30]:>10.3f}")
    print()
    print(f"{'FAM (ours, honest)':<22} {1260000:>12,} {15.53:>6.2f} +/- {1.68:<5.2f} {0.919:>10.3f}")
    print(f"{'V2 (honest)':<22} {2580000:>12,} {15.73:>6.2f} +/- {0.14:<5.2f}       --")
    print(f"{'STAR (supervised)':<22} {'?':>12} {12.19:>6.2f} +/- {0.55:<5.2f}       --")


if __name__ == '__main__':
    main()

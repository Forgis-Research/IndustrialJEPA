"""V20 Phase 2: Chronos baseline with per-window F1 on C-MAPSS FD001.

Recomputes the v18 Chronos-T5 linear probe experiment and saves raw RUL
predictions so per-window F1 (W=16) can be derived via thresholding, aligning
with FAM's unified eval.

Tests chronos-t5-tiny (8M, closest to FAM 1.26M) primarily. The other sizes
are optional - kept in MODELS list for enablement.

3 seeds per model, linear probe (honest protocol AdamW WD=1e-2).
"""
import sys, json, copy, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
V11 = ROOT / 'experiments' / 'v11'
V20 = ROOT / 'experiments' / 'v20'
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V20)); sys.path.insert(0, str(ROOT))

from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from train_utils import subsample_engines
from pred_ft_utils import per_window_metrics_from_rul, rul_metrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_CTX = 300
SEEDS = [42, 123, 456]
N_WINDOWS = 16

# Only run tiny by default - sufficient for paper baseline, fast
MODELS = [
    ('chronos-t5-tiny',  'amazon/chronos-t5-tiny',  256, 8_394_496),
]


def extract_engine_embedding(pipe, seq, d_emb, max_ctx=MAX_CTX):
    """Per-sensor last-token embedding, concatenated -> (14*d_emb,)."""
    T, S = seq.shape
    take = min(T, max_ctx)
    s_arr = seq[-take:, :].T
    x = torch.from_numpy(s_arr).float()
    with torch.no_grad():
        emb, _ = pipe.embed(x)
    last = emb[:, -1, :]
    return last.cpu().numpy().reshape(-1)


def extract_all(pipe, engines, d_emb):
    return {eid: extract_engine_embedding(pipe, seq, d_emb)
            for eid, seq in engines.items()}


def compute_rul_labels(engines, rul_cap=RUL_CAP):
    """For train/val: use last-cycle RUL = 1 (just born capped) ??? no.
    We mirror v18: last-cycle RUL target = 1/rul_cap if > rul_cap else raw.
    Actually v18 uses last-cycle only with normalized RUL = capped(T)/rul_cap.
    """
    out = {}
    for eid, seq in engines.items():
        T = len(seq)
        rul_raw = float(min(T, rul_cap))       # ??? if engine never ran out...
        # Actually RUL at last training cycle is 0 (engine fails at end).
        # We want the RUL at the cycle where we extracted the embedding.
        # The embedding is from last max_ctx cycles, target is the LAST cycle's RUL.
        rul_last = 0.0                         # end-of-life training engine
        out[eid] = rul_last / rul_cap
    return out


def build_train_val_test(data, seed, rul_cap=RUL_CAP):
    """Build aligned (embedding, RUL-at-extraction-time) triples.

    For Chronos we take the FULL engine history and predict RUL at the last cycle.
    But training engines all end at failure (RUL=0), so we need varied cut points.
    We follow v18 convention: for train/val, take multiple cuts per engine; for
    test, use test_rul as the last-cycle RUL.
    """
    # The v18 phase8 saves per-engine last embedding with last-cycle RUL.
    # For varied RUL distribution in training, sample random cut points per engine.
    rng = np.random.default_rng(seed)
    train_engines = data['train_engines']
    val_engines = data['val_engines']
    test_engines = data['test_engines']
    test_rul_arr = data['test_rul']

    # Sample 5 cut points per train/val engine
    def sample_cuts(engines, n_cuts=5):
        pairs = []
        for eid, seq in engines.items():
            T = len(seq)
            cuts = sorted(rng.integers(10, T, size=min(n_cuts, T - 10)).tolist())
            for t in cuts:
                # RUL at cycle t is (T - t)
                rul = float(T - t)
                sub_seq = seq[:t]
                pairs.append((sub_seq, min(rul, rul_cap), rul))  # capped & raw
        return pairs

    train_pairs = sample_cuts(train_engines, n_cuts=5)
    val_pairs = sample_cuts(val_engines, n_cuts=10)

    # Test: full sequence, last-cycle RUL from test_rul
    test_pairs = []
    for i, eid in enumerate(sorted(test_engines.keys())):
        seq = test_engines[eid]
        rul_raw = float(test_rul_arr[i])
        test_pairs.append((seq, min(rul_raw, rul_cap), rul_raw))
    return train_pairs, val_pairs, test_pairs


def extract_pairs(pipe, pairs, d_emb):
    embs, rul_norm, rul_raw = [], [], []
    for seq, rul_capped, rul_raw_val in pairs:
        e = extract_engine_embedding(pipe, seq, d_emb)
        embs.append(e)
        rul_norm.append(rul_capped / RUL_CAP)
        rul_raw.append(rul_raw_val)
    return np.stack(embs), np.array(rul_norm), np.array(rul_raw)


def train_probe(X_tr, y_tr, X_va, y_va, feat_dim, seed, n_epochs=200, patience=25,
                batch_size=32, lr=1e-3, wd=1e-2):
    torch.manual_seed(seed)
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).float().to(DEVICE)
    X_va_t = torch.from_numpy(X_va).float().to(DEVICE)
    y_va_t = torch.from_numpy(y_va).float().to(DEVICE)

    probe = nn.Sequential(nn.Linear(feat_dim, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)

    best_val = float('inf'); best_sd = None; no_impr = 0
    N = X_tr_t.shape[0]
    for ep in range(n_epochs):
        probe.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch_size):
            b = perm[i:i+batch_size]
            pred = probe(X_tr_t[b]).squeeze(-1)
            loss = F.mse_loss(pred, y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            pv = probe(X_va_t).squeeze(-1)
            val_mse = F.mse_loss(pv, y_va_t).item()
        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_sd = copy.deepcopy(probe.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break
    probe.load_state_dict(best_sd)
    return probe, best_val


def main():
    V20.mkdir(exist_ok=True)
    out_path = V20 / 'phase2_chronos_perwindow.json'
    print("=" * 80)
    print("V20 Phase 2: Chronos per-window F1 on FD001")
    print(f"  {len(MODELS)} models x {len(SEEDS)} seeds")
    print("=" * 80, flush=True)

    from chronos import ChronosPipeline

    data = load_cmapss_subset('FD001')
    t0 = time.time()

    all_results = {}
    for model_name, hf_id, d_emb, n_params in MODELS:
        print(f"\n--- {model_name} (d_emb={d_emb}, n_params={n_params:,}) ---",
              flush=True)
        t_model = time.time()
        pipe = ChronosPipeline.from_pretrained(hf_id, device_map=DEVICE,
                                                torch_dtype=torch.float32)
        pipe.model.eval()

        feat_dim = N_SENSORS * d_emb
        per_seed = []
        for seed in SEEDS:
            t_seed = time.time()
            train_pairs, val_pairs, test_pairs = build_train_val_test(data, seed)
            print(f"  [seed {seed}] extracting embeddings "
                  f"(train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)})",
                  flush=True)
            t_emb = time.time()
            X_tr, y_tr, _ = extract_pairs(pipe, train_pairs, d_emb)
            X_va, y_va, _ = extract_pairs(pipe, val_pairs, d_emb)
            X_te, y_te_norm, y_te_raw = extract_pairs(pipe, test_pairs, d_emb)
            print(f"    extract took {time.time()-t_emb:.1f}s", flush=True)
            probe, val_mse = train_probe(X_tr, y_tr, X_va, y_va, feat_dim, seed)
            probe.eval()
            X_te_t = torch.from_numpy(X_te).float().to(DEVICE)
            with torch.no_grad():
                pred_norm = probe(X_te_t).squeeze(-1).cpu().numpy()
            pred_raw = pred_norm * RUL_CAP
            y_te_capped = np.minimum(y_te_raw, float(RUL_CAP))
            legacy = rul_metrics(pred_raw, y_te_capped)
            per_win = per_window_metrics_from_rul(pred_raw, y_te_raw, n_windows=N_WINDOWS)
            per_seed.append({
                'seed': seed,
                'val_mse': float(val_mse),
                'test_rmse': legacy['rmse'],
                'test_nasa': legacy['nasa_score'],
                'f1_w_mean': per_win['f1_mean'],
                'auroc_w_mean': per_win['auroc_mean'],
                'precision_w_mean': per_win['precision_mean'],
                'recall_w_mean': per_win['recall_mean'],
                'per_window': per_win['per_window'],
                'pred_rul_raw': pred_raw.tolist(),
                'true_rul_raw': y_te_raw.tolist(),
            })
            print(f"  [seed {seed}] RMSE={legacy['rmse']:.2f} "
                  f"F1w={per_win['f1_mean']:.3f} AUROCw={per_win['auroc_mean']:.3f} "
                  f"({time.time()-t_seed:.0f}s)",
                  flush=True)
            gc.collect(); torch.cuda.empty_cache()

        agg = {
            'model': model_name, 'hf_id': hf_id, 'n_params': n_params,
            'test_rmse_mean': float(np.mean([r['test_rmse'] for r in per_seed])),
            'test_rmse_std':  float(np.std([r['test_rmse'] for r in per_seed], ddof=1)),
            'f1_w_mean':      float(np.mean([r['f1_w_mean'] for r in per_seed])),
            'f1_w_std':       float(np.std([r['f1_w_mean'] for r in per_seed], ddof=1)),
            'auroc_w_mean':   float(np.mean([r['auroc_w_mean'] for r in per_seed])),
            'per_seed': per_seed,
            'elapsed_min': (time.time() - t_model) / 60,
        }
        all_results[model_name] = agg
        print(f"  {model_name}: RMSE {agg['test_rmse_mean']:.2f} ± {agg['test_rmse_std']:.2f}  "
              f"F1w {agg['f1_w_mean']:.3f} ± {agg['f1_w_std']:.3f}",
              flush=True)

        with open(out_path, 'w') as f:
            json.dump({'config': 'v20_phase2_chronos_perwindow',
                       'seeds': SEEDS, 'n_windows': N_WINDOWS,
                       'runtime_min': (time.time() - t0) / 60,
                       'models': all_results}, f, indent=2, default=float)

        del pipe; gc.collect(); torch.cuda.empty_cache()

    print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()

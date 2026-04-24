"""V27 Phase 7 — paper-quality per-entity dense surfaces.

Generates 3 representative .npz files for the paper's Figure 3:

  1. FD001 — single engine with full lifecycle, v27 'none' checkpoint.
     Shows a shifting failure front: as t approaches end-of-life, the
     p_surface lights up at shorter Δt. This is the "recovered"
     behavior v26 (RevIN) was missing.

  2. MBA — a window of the v26 test stream containing 2-3 arrhythmias.
     Uses the v26 RevIN checkpoint (the winner on cardiac data).

  3. SMAP — one entity with a clear anomaly segment, v26 RevIN ckpt.

Each .npz stores (p_surface, y_surface, horizons, x_context_preview,
t_index, entity_id) so the paper figure can be rendered locally.
Dense horizons: every integer Δt from 1 to max_horizon.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V27_DIR = FAM_DIR / 'experiments/v27'
V26_DIR = FAM_DIR / 'experiments/v26'
PAPER_SURF = V27_DIR / 'surfaces'
PAPER_SURF.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))

from model import FAM


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Per-entity dense eval (sliding stride=1, every integer Δt in [1, max_h])
# ---------------------------------------------------------------------------

@torch.no_grad()
def dense_eval_entity(model, x: np.ndarray, labels: np.ndarray,
                      horizons: list, max_context: int = 512,
                      min_context: int = 128,
                      max_future: int = 200,
                      batch_size: int = 64) -> dict:
    """Run sliding-stride-1 dense eval on ONE entity's stream.

    Returns dict with p_surface (N, K), y_surface (N, K), t_index (N,).
    """
    T, C = x.shape
    x_t = torch.from_numpy(x).float()

    # Compute TTE from labels
    tte = np.full(T, np.inf, dtype=np.float32)
    next_anom = -1
    for t in range(T - 1, -1, -1):
        if next_anom != -1 and (next_anom - t) <= max_future:
            tte[t] = float(next_anom - t)
        if labels[t] == 1:
            next_anom = t

    t_start = max(1, min_context)
    t_end = T - 1
    starts = list(range(t_start, t_end))
    if not starts:
        return {'p_surface': np.zeros((0, len(horizons)), dtype=np.float32),
                'y_surface': np.zeros((0, len(horizons)), dtype=np.int8),
                't_index': np.zeros(0, dtype=np.int64)}

    h_tensor = torch.tensor(horizons, dtype=torch.float32, device=DEVICE)

    p_list, y_list, t_list = [], [], []
    for b_start in range(0, len(starts), batch_size):
        batch_ts = starts[b_start:b_start + batch_size]
        ctx_padded = torch.zeros(len(batch_ts), max_context, C)
        ctx_mask = torch.ones(len(batch_ts), max_context, dtype=torch.bool)
        ttes = torch.zeros(len(batch_ts), dtype=torch.float32)
        for i, t in enumerate(batch_ts):
            ctx_start = max(0, t - max_context)
            ctx = x_t[ctx_start:t]
            ctx_padded[i, :ctx.shape[0]] = ctx
            ctx_mask[i, :ctx.shape[0]] = False
            ttes[i] = float(tte[t])

        ctx_padded = ctx_padded.to(DEVICE)
        ctx_mask = ctx_mask.to(DEVICE)

        cdf = model.finetune_forward(ctx_padded, h_tensor, ctx_mask,
                                     mode='pred_ft')
        p = cdf.cpu().numpy()

        # label surface: y[i, k] = 1 if tte[i] <= horizons[k]
        y = (ttes.unsqueeze(1) <= h_tensor.cpu()).numpy().astype(np.int8)

        p_list.append(p)
        y_list.append(y)
        t_list.extend(batch_ts)

    return {
        'p_surface': np.concatenate(p_list, axis=0),
        'y_surface': np.concatenate(y_list, axis=0),
        't_index': np.asarray(t_list, dtype=np.int64),
    }


def save_paper_surface(path: Path, surf: dict, horizons, x_preview: np.ndarray,
                       entity_id, dataset: str, norm_mode: str, seed: int,
                       ckpt_path: str, notes: str = ''):
    meta = {
        'dataset': dataset, 'norm_mode': norm_mode, 'seed': seed,
        'entity_id': str(entity_id), 'ckpt_path': str(ckpt_path),
        'notes': notes, 'n_channels': int(x_preview.shape[1]),
        'T_entity': int(x_preview.shape[0]),
    }
    np.savez(path,
             p_surface=surf['p_surface'].astype(np.float32),
             y_surface=surf['y_surface'].astype(np.int8),
             horizons=np.asarray(horizons, dtype=np.int32),
             t_index=surf['t_index'].astype(np.int64),
             x_context=x_preview.astype(np.float32),
             meta=np.asarray(list(meta.items()), dtype=object))
    print(f"  wrote {path}")
    # Quick diagnostic
    p = surf['p_surface']; y = surf['y_surface']
    if p.size > 0 and y.any() and not y.all():
        pooled_auprc = average_precision_score(y.ravel().astype(int), p.ravel())
        pooled_auroc = roc_auc_score(y.ravel().astype(int), p.ravel())
        print(f"    pooled AUPRC={pooled_auprc:.4f} AUROC={pooled_auroc:.4f}  "
              f"shape={p.shape}")


# ---------------------------------------------------------------------------
# FD001 — v27 'none' dense per-engine
# ---------------------------------------------------------------------------

def gen_fd001_v27(seed: int = 42):
    from _cmapss_raw import load_cmapss_raw
    data = load_cmapss_raw('FD001')
    # Apply global z-score from train (same as Phase 2)
    train_arr = np.concatenate(list(data['train_engines'].values()), axis=0)
    mu = train_arr.mean(axis=0, keepdims=True).astype(np.float32)
    std = (train_arr.std(axis=0, keepdims=True) + 1e-5).astype(np.float32)
    test_engines = {eid: ((seq - mu) / std).astype(np.float32)
                    for eid, seq in data['test_engines'].items()}

    # Pick longest-cycle engine for the paper figure
    sorted_eids = sorted(test_engines.keys(),
                         key=lambda e: len(test_engines[e]), reverse=True)
    picks = sorted_eids[:3]
    print(f"  FD001: longest test engines (cycles): " +
          ', '.join(f'{eid}({len(test_engines[eid])})' for eid in picks))

    # Load v27 checkpoint
    ckpt_path = V27_DIR / 'ckpts' / f'FD001_none_s{seed}_pred_ft.pt'
    model = FAM(n_channels=14, norm_mode='none').to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    horizons = list(range(1, 151))  # dense 1..150
    max_ctx = 512

    for eid in picks:
        x = test_engines[eid]
        T = len(x)
        labels = np.zeros(T, dtype=np.int32)
        labels[-1] = 1  # final cycle = failure
        print(f"  FD001 engine {eid}: T={T}")
        surf = dense_eval_entity(model, x, labels, horizons,
                                 max_context=max_ctx, max_future=200)
        out = PAPER_SURF / f'paper_FD001_e{eid}_v27_none_s{seed}_dense.npz'
        save_paper_surface(out, surf, horizons, x, eid, 'FD001', 'none', seed,
                           str(ckpt_path),
                           notes='dense Δt 1..150; final cycle = failure')


# ---------------------------------------------------------------------------
# MBA — v26 RevIN dense on a window with multiple arrhythmias
# ---------------------------------------------------------------------------

def gen_mba_v26(seed: int = 42):
    from data.mba import load_mba
    d = load_mba(normalize=False)
    test = d['test'].astype(np.float32)
    labels = d['labels'].astype(np.int32)
    # Use last 30% of the test stream as ft_test (same as v26 intra-split)
    T = len(test)
    t2 = int(0.7 * T)
    gap = 200
    test_ft = test[t2 + gap:]
    labels_ft = labels[t2 + gap:]

    # Load v26 MBA RevIN checkpoint
    ckpt_path = V26_DIR / 'ckpts' / f'MBA_s{seed}_pred_ft.pt'
    model = FAM(n_channels=test.shape[1], norm_mode='revin').to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    horizons = list(range(1, 201))
    surf = dense_eval_entity(model, test_ft, labels_ft, horizons,
                             max_context=512, max_future=200, batch_size=32)
    n_anom = int(labels_ft.sum())
    print(f"  MBA test_ft: T={len(test_ft)}, anomaly timesteps={n_anom}")
    out = PAPER_SURF / f'paper_MBA_v26_revin_s{seed}_dense.npz'
    save_paper_surface(out, surf, horizons, test_ft, 'MBA_test_tail',
                       'MBA', 'revin', seed, str(ckpt_path),
                       notes='dense Δt 1..200; last 30% of test stream '
                             'with 200-step gap')


# ---------------------------------------------------------------------------
# SMAP — v26 RevIN dense on one high-signal entity
# ---------------------------------------------------------------------------

def gen_smap_v26(seed: int = 42):
    from data.smap_msl import split_smap_entities
    ft = split_smap_entities(normalize=False)

    # Pick a test-split entity with 1-2 anomaly segments (clear, interpretable)
    # Filter: at least one anomaly, entity test length >= 500 for visual clarity
    candidates = []
    for e in ft['ft_test']:
        n_anom = int(e['labels'].sum())
        if n_anom >= 10 and len(e['test']) >= 500:
            candidates.append((e['entity_id'], len(e['test']), n_anom))
    candidates.sort(key=lambda t: (t[2], t[1]))  # smaller anom count first
    picks = [c[0] for c in candidates[:3]]
    print(f"  SMAP: candidate entities for paper: {candidates[:3]}")

    ckpt_path = V26_DIR / 'ckpts' / f'SMAP_s{seed}_pred_ft.pt'
    model = FAM(n_channels=25, norm_mode='revin').to(DEVICE).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    horizons = list(range(1, 201))
    for eid in picks:
        e = next(x for x in ft['ft_test'] if x['entity_id'] == eid)
        surf = dense_eval_entity(model, e['test'], e['labels'], horizons,
                                 max_context=512, max_future=200, batch_size=32)
        print(f"  SMAP entity {eid}: T={len(e['test'])}, "
              f"n_anom={int(e['labels'].sum())}")
        out = PAPER_SURF / f'paper_SMAP_e{eid}_v26_revin_s{seed}_dense.npz'
        save_paper_surface(out, surf, horizons, e['test'], eid,
                           'SMAP', 'revin', seed, str(ckpt_path),
                           notes='dense Δt 1..200; intra-entity test tail')


def main():
    print("=== Phase 7: paper-quality surfaces ===")
    t0 = time.time()
    gen_fd001_v27(seed=42)
    print(f"  FD001 done in {time.time()-t0:.1f}s")
    t0 = time.time()
    gen_mba_v26(seed=42)
    print(f"  MBA done in {time.time()-t0:.1f}s")
    t0 = time.time()
    gen_smap_v26(seed=42)
    print(f"  SMAP done in {time.time()-t0:.1f}s")
    print("\n=== Files written to experiments/v27/surfaces/paper_*.npz ===")


if __name__ == '__main__':
    main()

"""
V18 Phase 4g: Diagnose the MSL Mahalanobis failure.

Phase 4f found MSL Mahalanobis PA-F1 = 0.000. The paper currently lists this
as a benchmark-specific failure "not yet fully understood". This phase
investigates three hypotheses:

  H1: Insufficient pretraining (Phase 4f used 50 epochs; try 150).
  H2: Scoring geometry mismatch (try rep_shift, traj_div, neg_Mahalanobis).
  H3: K-range too narrow for MSL temporal structure (try K_max=200 vs 500).

Plan:
  - Pretrain v17 MSL with 150 epochs (seed 42) - address H1.
  - Evaluate with Mahalanobis (PCA-5/10/20), rep_shift, traj_div, +/-sign.
  - Report which scoring method / epoch gives best MSL PA-F1.

Uses existing pretrained models; reuses code from Phase 4b/4f.

Output: experiments/v18/phase4g_msl_diagnose.json
"""

import sys, math, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(V17)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data.smap_msl import load_msl
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

from phase5_smap_anomaly import (
    SMAPV17Dataset, collate_smap, v17_loss,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, BATCH_SIZE, LR,
    WEIGHT_DECAY, EMA_MOMENTUM, N_SAMPLES,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_DIR = V18 / 'ckpts'
CKPT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 100; STRIDE = 10; BATCH_ENC = 256
THRESH_PCTL = 95
N_EPOCHS_LONG = 150   # vs 50 in phase 4f
SEED = 42


def pretrain_long(data, seed, ckpt_path, n_epochs=N_EPOCHS_LONG):
    torch.manual_seed(seed); np.random.seed(seed)
    C = data['n_channels']
    model = TrajectoryJEPA(
        n_sensors=C, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)

    history = {'loss': [], 'pred': []}
    t0 = time.time()
    for epoch in range(1, n_epochs + 1):
        ds = SMAPV17Dataset(data['train'], n_samples=N_SAMPLES,
                            seed=seed * 1000 + epoch)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_smap, num_workers=0)
        model.train()
        tot = lp = 0.0; n = 0
        for x_past, past_mask, x_fut, fut_mask, k in loader:
            x_past, past_mask = x_past.to(DEVICE), past_mask.to(DEVICE)
            x_fut, fut_mask = x_fut.to(DEVICE), fut_mask.to(DEVICE)
            k = k.to(DEVICE)
            optim.zero_grad()
            pred, targ, _ = model.forward_pretrain(
                x_past, past_mask, x_fut, fut_mask, k)
            loss, lp_, _ = v17_loss(pred, targ, lambda_var=0.04)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); model.update_ema()
            B = x_past.shape[0]
            tot += loss.item() * B; lp += lp_.item() * B; n += B
        sched.step()
        history['loss'].append(tot / n)
        history['pred'].append(lp / n)
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"  ep {epoch:3d} | L={tot/n:.4f} pred={lp/n:.4f}", flush=True)

    elapsed = (time.time() - t0) / 60
    print(f"  pretrain done in {elapsed:.1f} min", flush=True)
    torch.save(model.state_dict(), ckpt_path)
    return model, history


@torch.no_grad()
def encode_windows(model, arr, window=WINDOW, stride=STRIDE):
    T, C = arr.shape
    starts = list(range(0, T - window, stride))
    H_list = []
    for b in range(0, len(starts), BATCH_ENC):
        batch = starts[b:b+BATCH_ENC]
        B = len(batch)
        x = np.stack([arr[s:s+window] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad = torch.zeros(B, window, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad)
        H_list.append(h.cpu().numpy())
    return np.array(starts), np.concatenate(H_list, axis=0) if H_list else np.zeros((0, D_MODEL))


def mahal_score(H_test, starts, H_train, n_comp, T_len):
    from sklearn.decomposition import PCA
    mu = H_train.mean(axis=0); Htc = H_train - mu
    k = min(n_comp, Htc.shape[1], Htc.shape[0] - 1)
    pca = PCA(n_components=k); pca.fit(Htc)
    var = pca.explained_variance_
    Zte = pca.transform(H_test - mu)
    m2 = (Zte ** 2 / (var + 1e-8)).sum(axis=1)
    scores = np.zeros(T_len, dtype=np.float32); counts = np.zeros(T_len, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + WINDOW
        scores[end: min(end + STRIDE, T_len)] += m2[i]
        counts[end: min(end + STRIDE, T_len)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def rep_shift_score_from_H(starts, H, T_len):
    shifts = np.zeros(len(H))
    shifts[1:] = np.linalg.norm(H[1:] - H[:-1], axis=1)
    shifts[0] = shifts[1] if len(shifts) > 1 else 0.0
    scores = np.zeros(T_len, dtype=np.float32); counts = np.zeros(T_len, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + WINDOW
        scores[end: min(end + STRIDE, T_len)] += shifts[i]
        counts[end: min(end + STRIDE, T_len)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


@torch.no_grad()
def traj_div_score(model, arr, k_short=5, k_long=100):
    T, C = arr.shape
    starts = list(range(0, T - WINDOW - k_long, STRIDE))
    if not starts: return np.zeros(T, dtype=np.float32)
    divs = np.zeros(len(starts))
    for b in range(0, len(starts), BATCH_ENC):
        batch = starts[b:b+BATCH_ENC]
        B = len(batch)
        x = np.stack([arr[s:s+WINDOW] for s in batch])
        x_t = torch.from_numpy(x).float().to(DEVICE)
        pad = torch.zeros(B, WINDOW, dtype=torch.bool, device=DEVICE)
        h = model.encode_past(x_t, pad)
        gs = model.predictor(h, torch.full((B,), k_short, dtype=torch.long, device=DEVICE))
        gl = model.predictor(h, torch.full((B,), k_long, dtype=torch.long, device=DEVICE))
        divs[b:b+B] = (gs - gl).norm(dim=-1).cpu().numpy()
    scores = np.zeros(T, dtype=np.float32); counts = np.zeros(T, dtype=np.float32)
    for i, s in enumerate(starts):
        end = s + WINDOW
        scores[end: min(end + STRIDE, T)] += divs[i]
        counts[end: min(end + STRIDE, T)] += 1
    valid = counts > 0
    scores[valid] /= counts[valid]
    if (~valid).any() and valid.any():
        scores[~valid] = float(np.median(scores[valid]))
    return scores


def eval_metrics(scores, labels):
    n_norm = int(0.1 * len(scores))
    thr = float(np.percentile(scores[:n_norm], THRESH_PCTL))
    m = _anomaly_metrics(scores, labels, threshold=thr)
    anom = float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0
    norm = float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0
    return {
        'f1_non_pa': float(m['f1_non_pa']),
        'f1_pa': float(m['f1_pa']),
        'auc_pr': float(m['auc_pr']),
        'threshold': thr,
        'score_gap': anom - norm,
    }


def main():
    V18.mkdir(exist_ok=True)
    print("Loading MSL...", flush=True)
    data = load_msl()
    T_len = len(data['test'])
    print(f"  MSL: train={data['train'].shape} test={data['test'].shape} "
          f"anomaly_rate={data['anomaly_rate']:.3f}", flush=True)

    t0 = time.time()

    # Load 50-ep ckpt for baseline comparison
    print("\n=== 50-epoch ckpt (Phase 4f) ===", flush=True)
    model_50 = TrajectoryJEPA(
        n_sensors=data['n_channels'], patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    ck50 = CKPT_DIR / f'v18_msl_seed{SEED}.pt'
    model_50.load_state_dict(torch.load(ck50, map_location=DEVICE, weights_only=False))
    model_50.eval()
    for p in model_50.parameters(): p.requires_grad = False

    # 150-ep pretrain
    print("\n=== Pretraining 150 epochs ===", flush=True)
    ck150 = CKPT_DIR / f'v18_msl_seed{SEED}_ep150.pt'
    if ck150.exists():
        print(f"  ckpt exists: {ck150}, skipping", flush=True)
        model_150 = TrajectoryJEPA(
            n_sensors=data['n_channels'], patch_length=1,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
            dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
        ).to(DEVICE)
        model_150.load_state_dict(torch.load(ck150, map_location=DEVICE, weights_only=False))
        history_150 = {'loss': [], 'pred': []}
    else:
        model_150, history_150 = pretrain_long(data, SEED, ck150, n_epochs=N_EPOCHS_LONG)
    model_150.eval()
    for p in model_150.parameters(): p.requires_grad = False

    results = {}
    for ep_name, model in [('50', model_50), ('150', model_150)]:
        print(f"\n--- Evaluating ep{ep_name} with multiple scoring methods ---", flush=True)
        _, H_tr = encode_windows(model, data['train'])
        starts, H_te = encode_windows(model, data['test'])

        results[f'ep{ep_name}'] = {}
        # Mahalanobis with k = 5, 10, 20, 50
        for k_pca in [5, 10, 20, 50]:
            scores = mahal_score(H_te, starts, H_tr, k_pca, T_len)
            m = eval_metrics(scores, data['labels'])
            results[f'ep{ep_name}'][f'mahal_k{k_pca}'] = m
            print(f"  mahal_k{k_pca}: non-PA {m['f1_non_pa']:.3f} "
                  f"PA {m['f1_pa']:.3f} AUC-PR {m['auc_pr']:.3f} "
                  f"gap {m['score_gap']:+.3f}", flush=True)
            # Negated
            m_neg = eval_metrics(-scores, data['labels'])
            results[f'ep{ep_name}'][f'neg_mahal_k{k_pca}'] = m_neg
            print(f"  neg_mahal_k{k_pca}: non-PA {m_neg['f1_non_pa']:.3f} "
                  f"PA {m_neg['f1_pa']:.3f} AUC-PR {m_neg['auc_pr']:.3f}", flush=True)

        # Rep shift
        scores_rs = rep_shift_score_from_H(starts, H_te, T_len)
        m_rs = eval_metrics(scores_rs, data['labels'])
        results[f'ep{ep_name}']['rep_shift'] = m_rs
        print(f"  rep_shift: non-PA {m_rs['f1_non_pa']:.3f} "
              f"PA {m_rs['f1_pa']:.3f} AUC-PR {m_rs['auc_pr']:.3f}", flush=True)

        # Trajectory divergence
        scores_td = traj_div_score(model, data['test'])
        m_td = eval_metrics(scores_td, data['labels'])
        results[f'ep{ep_name}']['traj_div'] = m_td
        print(f"  traj_div: non-PA {m_td['f1_non_pa']:.3f} "
              f"PA {m_td['f1_pa']:.3f} AUC-PR {m_td['auc_pr']:.3f}", flush=True)

    summary = {
        'config': 'v18_phase4g_msl_diagnose',
        'epochs_tested': [50, 150],
        'scoring_methods': ['mahal_k5/10/20/50', 'neg_mahal', 'rep_shift', 'traj_div'],
        'results': results,
        'history_150_epochs': history_150 if 'loss' in history_150 and history_150['loss'] else {'note': 'ckpt loaded, no history'},
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18 / 'phase4g_msl_diagnose.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    # Pick best
    best_k, best_method, best_pa = None, None, -1
    for ep_k in ['ep50', 'ep150']:
        for method_k, m in results[ep_k].items():
            if m['f1_pa'] > best_pa:
                best_pa = m['f1_pa']; best_k = ep_k; best_method = method_k

    print("\n" + "=" * 60)
    print("V18 Phase 4g: MSL Diagnosis SUMMARY")
    print("=" * 60)
    print(f"Best MSL method: {best_k} {best_method}  PA-F1={best_pa:.3f}")
    print(f"MTS-JEPA MSL ref: (paper-reported)")
    print(f"Phase 4f ep50 mahal_k10 baseline: 0.000")
    print(f"Runtime: {summary['runtime_min']:.1f} min")


if __name__ == '__main__':
    main()

"""
V18 Phase 0: Honest re-probe of V2 baseline.

Loads the V2 pretrained checkpoint (v11/best_pretrain_L1_v2.pt, d_model=256,
d_ff=512, predictor_hidden=256) and probes it under two protocols:

  OLD (what V2 reported 17.81 with):
    - CMAPSSFinetuneDataset(val_engines, use_last_only=True)  -> 15 val points
    - probe = Linear+Sigmoid, Adam LR=1e-3, no weight decay
    - 100 epochs, patience 15

  NEW (v17 Phase 2 honest protocol):
    - CMAPSSFinetuneDataset(val_engines, n_cuts_per_engine=10)  -> ~150 val points
    - probe = Linear+Sigmoid, AdamW LR=1e-3, weight_decay=1e-2
    - 200 epochs, patience 25

Each protocol runs 5 seeds. Report side-by-side + F1/AUC-PR at k=30.

Output: experiments/v18/phase0_honest_reprobe.json
"""

import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11))
sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# V2 architecture (matches v11/best_pretrain_L1_v2.pt)
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF_V2 = 512
PRED_HIDDEN_V2 = 256
EMA_MOMENTUM = 0.99

CKPT_PATH = V11 / 'best_pretrain_L1_v2.pt'
SEEDS = [42, 123, 456, 789, 1024]
K_EVAL_F1 = 30


def load_v2_model():
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF_V2,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN_V2,
    ).to(DEVICE)
    state = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def probe_once(model, data, seed, protocol):
    """Train probe and return test metrics."""
    torch.manual_seed(seed)

    probe = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid()).to(DEVICE)

    if protocol == 'old':
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
        epochs = 100
        patience = 15
    elif protocol == 'new':
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
        va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10,
                                      seed=seed + 111)
        epochs = 200
        patience = 25
    else:
        raise ValueError(protocol)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])

    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf'); best_state = None; no_impr = 0

    for ep in range(epochs):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
            p = probe(h).squeeze(-1)
            loss = F.mse_loss(p, rul)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        preds = np.concatenate(pv) * RUL_CAP
        targs = np.concatenate(tv) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(probe.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience:
                break

    probe.load_state_dict(best_state)

    # Test metrics
    probe.eval()
    p_test, t_test = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            p_test.append(probe(h).squeeze(-1).cpu().numpy() * RUL_CAP)
            t_test.append(rul_gt.numpy())
    preds_test = np.concatenate(p_test)
    targs_test = np.concatenate(t_test)
    test_rmse = float(np.sqrt(np.mean((preds_test - targs_test) ** 2)))

    y = (targs_test <= K_EVAL_F1).astype(int)
    score = -preds_test
    thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
    m = _anomaly_metrics(score, y, threshold=thr)

    return dict(
        seed=seed, protocol=protocol,
        val_rmse=best_val, test_rmse=test_rmse,
        f1=m['f1_non_pa'], auc_pr=m['auc_pr'],
        precision=m['precision_non_pa'], recall=m['recall_non_pa'],
    )


def main():
    V18.mkdir(exist_ok=True)
    print(f"Loading V2 checkpoint from {CKPT_PATH}", flush=True)
    model = load_v2_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"V2 model params: {n_params:,}", flush=True)

    data = load_cmapss_subset('FD001')
    print(f"FD001: train={len(data['train_engines'])} "
          f"val={len(data['val_engines'])} test={len(data['test_engines'])}", flush=True)

    t0 = time.time()
    results = {'old': [], 'new': []}

    for protocol in ['old', 'new']:
        print(f"\n=== {protocol.upper()} protocol ===", flush=True)
        for seed in SEEDS:
            r = probe_once(model, data, seed, protocol)
            results[protocol].append(r)
            print(f"  seed={seed} val={r['val_rmse']:.2f} test={r['test_rmse']:.2f} "
                  f"F1@{K_EVAL_F1}={r['f1']:.3f} AUC-PR={r['auc_pr']:.3f}", flush=True)

    # Aggregate
    def agg(rs, key):
        vs = [r[key] for r in rs]
        return {'mean': float(np.mean(vs)), 'std': float(np.std(vs)),
                'per_seed': [float(v) for v in vs]}

    summary = {
        'config': 'v18_phase0_honest_reprobe',
        'ckpt': str(CKPT_PATH),
        'v2_arch': {'d_model': D_MODEL, 'n_heads': N_HEADS, 'n_layers': N_LAYERS,
                    'd_ff': D_FF_V2, 'predictor_hidden': PRED_HIDDEN_V2},
        'seeds': SEEDS,
        'k_eval_f1': K_EVAL_F1,
        'source': 'existing_ckpt_v11_best_pretrain_L1_v2',
        'old_protocol': {
            'description': 'Adam LR=1e-3, no WD, val use_last_only=True (15 points)',
            'val_rmse': agg(results['old'], 'val_rmse'),
            'test_rmse': agg(results['old'], 'test_rmse'),
            'f1': agg(results['old'], 'f1'),
            'auc_pr': agg(results['old'], 'auc_pr'),
            'precision': agg(results['old'], 'precision'),
            'recall': agg(results['old'], 'recall'),
        },
        'new_protocol': {
            'description': 'AdamW LR=1e-3 WD=1e-2, val n_cuts_per_engine=10',
            'val_rmse': agg(results['new'], 'val_rmse'),
            'test_rmse': agg(results['new'], 'test_rmse'),
            'f1': agg(results['new'], 'f1'),
            'auc_pr': agg(results['new'], 'auc_pr'),
            'precision': agg(results['new'], 'precision'),
            'recall': agg(results['new'], 'recall'),
        },
        'v17_test_rmse_reference': 15.38,
        'published_v2_rmse': 17.81,
        'runtime_minutes': (time.time() - t0) / 60,
    }

    old_m = summary['old_protocol']['test_rmse']['mean']
    new_m = summary['new_protocol']['test_rmse']['mean']
    summary['interpretation'] = (
        f"OLD protocol test RMSE: {old_m:.2f}. NEW protocol test RMSE: {new_m:.2f}. "
        f"V17 reported 15.38 with NEW protocol. "
        f"Delta (v17 - v2_new): {15.38 - new_m:+.2f} "
        f"(negative means v17 architecture actually helps beyond probe fix)."
    )

    out_path = V18 / 'phase0_honest_reprobe.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved: {out_path}", flush=True)

    print("\n" + "=" * 60)
    print("V18 Phase 0 SUMMARY")
    print("=" * 60)
    print(f"{'protocol':<6} {'test_rmse':>18} {'val_rmse':>18} "
          f"{'F1@30':>12} {'AUC-PR':>10}")
    for k in ['old', 'new']:
        sp = summary[f'{k}_protocol']
        print(f"{k:<6} {sp['test_rmse']['mean']:>7.2f}+/-{sp['test_rmse']['std']:<4.2f}"
              f"        {sp['val_rmse']['mean']:>7.2f}+/-{sp['val_rmse']['std']:<4.2f}"
              f"        {sp['f1']['mean']:>7.3f}    {sp['auc_pr']['mean']:>7.3f}")
    print(f"V17  (new protocol ref) : 15.38")
    print(f"V2   (published)        : 17.81")
    print(f"Runtime: {summary['runtime_minutes']:.1f} min")
    print("\nInterpretation:")
    print(summary['interpretation'])


if __name__ == '__main__':
    main()

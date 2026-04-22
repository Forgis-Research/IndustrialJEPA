"""
V18 Phase 6: Multi-subset honest probe on FD003 and FD004.

Reviewers flagged "FD001-only primary evaluation" as SEVERE. This phase probes
existing V2-arch checkpoints on FD003 and FD004 under the honest protocol to
provide multi-subset evidence.

Checkpoints used (all V2 arch: d=256, d_ff=512, pred_h=256, L=2):
  - v11/best_pretrain_fd003_v2.pt        -> FD003 V2 baseline
  - v11/best_pretrain_fd004_v2.pt        -> FD004 V2 baseline
  - v14/best_pretrain_full_sequence_fd003.pt -> FD003 V14 full-sequence
  - v14/best_pretrain_full_sequence_fd004.pt -> FD004 V14 full-sequence

For each (subset, ckpt) pair:
  - Honest frozen probe (AdamW WD=1e-2, n_cuts=10 val), 3 seeds
  - Test RMSE + F1@k in {10, 20, 30, 50}

Output: experiments/v18/phase6_multisubset_results.json

Reference (v14 full-sequence, old protocol):
  - FD003 frozen 18.39 / E2E 13.67
  - FD004 frozen 28.08 / E2E 25.27
"""

import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from evaluation.grey_swan_metrics import anomaly_metrics as _anomaly_metrics

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 512
PRED_HIDDEN = 256; EMA_MOMENTUM = 0.99

SEEDS = [42, 123, 456]
K_EVAL_LIST = [10, 20, 30, 50]

CKPTS = {
    'FD003': {
        'v2': V11 / 'best_pretrain_fd003_v2.pt',
        'fullseq': V14 / 'best_pretrain_full_sequence_fd003.pt',
    },
    'FD004': {
        'v2': V11 / 'best_pretrain_fd004_v2.pt',
        'fullseq': V14 / 'best_pretrain_full_sequence_fd004.pt',
    },
}


def load_model(ckpt_path):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN,
    ).to(DEVICE)
    sd = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model


def honest_probe(model, data, seed):
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(D_MODEL, 1), nn.Sigmoid()).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10,
                                   seed=seed + 111)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    best_val = float('inf'); best_state = None; no_impr = 0
    for ep in range(200):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad(): h = model.encode_past(past, mask)
            loss = F.mse_loss(probe(h).squeeze(-1), rul)
            opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy()); tv.append(rul.numpy())
        preds = np.concatenate(pv) * RUL_CAP; targs = np.concatenate(tv) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targs) ** 2)))
        if val_rmse < best_val:
            best_val = val_rmse; best_state = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break

    probe.load_state_dict(best_state)
    probe.eval()
    p_test, t_test = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            p_test.append(probe(h).squeeze(-1).cpu().numpy() * RUL_CAP)
            t_test.append(rul_gt.numpy())
    pt = np.concatenate(p_test); tt = np.concatenate(t_test)
    test_rmse = float(np.sqrt(np.mean((pt - tt) ** 2)))

    f1_by_k = {}
    for ke in K_EVAL_LIST:
        y = (tt <= ke).astype(int); score = -pt
        thr = float(np.percentile(score[y == 0], 95)) if (y == 0).sum() > 0 else 0.0
        m = _anomaly_metrics(score, y, threshold=thr)
        f1_by_k[ke] = {'f1': float(m['f1_non_pa']), 'auc_pr': float(m['auc_pr'])}

    return {'seed': seed, 'val_rmse': best_val, 'test_rmse': test_rmse,
            'f1_by_k': f1_by_k}


def main():
    V18.mkdir(exist_ok=True)
    results = {}
    t0 = time.time()

    for subset_name in ['FD003', 'FD004']:
        results[subset_name] = {}
        print(f"\n=== {subset_name} ===", flush=True)
        data = load_cmapss_subset(subset_name)
        print(f"  train={len(data['train_engines'])} val={len(data['val_engines'])} "
              f"test={len(data['test_engines'])}", flush=True)

        for variant, ckpt_path in CKPTS[subset_name].items():
            if not ckpt_path.exists():
                print(f"  [{variant}] ckpt missing: {ckpt_path}", flush=True)
                continue
            print(f"\n-- variant={variant} --", flush=True)
            model = load_model(ckpt_path)
            rs = []
            for seed in SEEDS:
                r = honest_probe(model, data, seed)
                rs.append(r)
                print(f"   seed={seed} val={r['val_rmse']:.2f} "
                      f"test={r['test_rmse']:.2f} F1@30={r['f1_by_k'][30]['f1']:.3f}",
                      flush=True)
            results[subset_name][variant] = {
                'ckpt': str(ckpt_path),
                'per_seed': rs,
                'test_rmse': {
                    'mean': float(np.mean([r['test_rmse'] for r in rs])),
                    'std': float(np.std([r['test_rmse'] for r in rs])),
                    'per_seed': [r['test_rmse'] for r in rs],
                },
                'val_rmse_mean': float(np.mean([r['val_rmse'] for r in rs])),
                'f1_by_k_mean': {
                    ke: float(np.mean([r['f1_by_k'][ke]['f1'] for r in rs]))
                    for ke in K_EVAL_LIST
                },
            }
            del model; torch.cuda.empty_cache()

        # Save after each subset
        with open(V18 / 'phase6_multisubset_results.json', 'w') as f:
            json.dump({'config': 'v18_phase6_multisubset_honest_probe',
                       'protocol': 'AdamW WD=1e-2 val n_cuts=10',
                       'v14_fullseq_reference_old_protocol': {
                           'FD003_frozen': 18.39, 'FD003_e2e': 13.67,
                           'FD004_frozen': 28.08, 'FD004_e2e': 25.27,
                       },
                       'seeds': SEEDS,
                       'results': results,
                       'runtime_min': (time.time() - t0) / 60}, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("V18 Phase 6: MULTI-SUBSET HONEST FROZEN PROBE SUMMARY")
    print("=" * 70)
    print(f"{'subset':<8} {'variant':<10} {'test_rmse':>18} {'F1@30':>10}")
    for subset in ['FD003', 'FD004']:
        for variant in ['v2', 'fullseq']:
            if variant in results.get(subset, {}):
                s = results[subset][variant]
                print(f"{subset:<8} {variant:<10} "
                      f"{s['test_rmse']['mean']:6.2f}+/-{s['test_rmse']['std']:<4.2f}        "
                      f"{s['f1_by_k_mean'][30]:>7.3f}")
    print(f"\nReference (v14 old protocol): FD003 fullseq frozen 18.39, FD004 28.08")
    print(f"Runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()

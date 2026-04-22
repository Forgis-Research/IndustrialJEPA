"""
V19 Phase 7: Chronos-T5 frozen probe on Paderborn (fault classification).

Extends V18's FAM-beats-Chronos result from turbofan RUL to REAL mechanical
vibration classification. Same protocol as v18 phase8_chronos_rul.py but:
  - Input: 1024-sample windows (not full engine cycle sequence)
  - Channels: 1 (vibration_1 only, matching FAM Paderborn setup)
  - Task: 3-class classification (K001/KA01/KI01)
  - Probe: Linear head + 3-class CE loss

Tests: Chronos-T5-tiny, small, base (skipping large for time).

Output: experiments/v19/phase7_chronos_paderborn_results.json
"""

import sys, json, copy, time, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V19 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v19')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(ROOT))

from chronos import ChronosPipeline
# Reuse Paderborn data pipeline from phase5
sys.path.insert(0, str(V19))
from phase5_paderborn import (CLASSES, load_paderborn_split)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEEDS = [42, 123, 456]

MODELS = [
    ('chronos-t5-tiny',  'amazon/chronos-t5-tiny',  256),
    ('chronos-t5-small', 'amazon/chronos-t5-small', 512),
    ('chronos-t5-base',  'amazon/chronos-t5-base',  768),
]


@torch.no_grad()
def embed_windows(pipe, X):
    """X shape (N, 1024, 1) -> (N, d_emb). Single channel, so batch over N."""
    N = X.shape[0]
    out = []
    batch = 32  # Chronos memory-heavy; keep small
    for i in range(0, N, batch):
        x = torch.from_numpy(X[i:i+batch, :, 0]).float()  # (B, 1024)
        emb, _ = pipe.embed(x)  # emb: (B, 1024+1, d)
        last = emb[:, -1, :].cpu().numpy()  # (B, d)
        out.append(last)
    return np.concatenate(out, axis=0)


def linear_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, seed):
    torch.manual_seed(seed)
    D = X_tr.shape[1]
    probe = nn.Linear(D, 3).to(DEVICE)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    X_tr_t = torch.from_numpy(X_tr).float().to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).long().to(DEVICE)
    X_va_t = torch.from_numpy(X_va).float().to(DEVICE)
    y_va_t = torch.from_numpy(y_va).long().to(DEVICE)
    X_te_t = torch.from_numpy(X_te).float().to(DEVICE)
    best_va = 0.0; best = None; no_impr = 0
    for ep in range(200):
        probe.train()
        idx = torch.randperm(X_tr_t.shape[0])
        for i in range(0, X_tr_t.shape[0], 256):
            b = idx[i:i+256]
            loss = F.cross_entropy(probe(X_tr_t[b]), y_tr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            acc_va = (probe(X_va_t).argmax(-1) == y_va_t).float().mean().item()
        if acc_va > best_va:
            best_va = acc_va; best = copy.deepcopy(probe.state_dict()); no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 25: break
    probe.load_state_dict(best); probe.eval()
    with torch.no_grad():
        pred_te = probe(X_te_t).argmax(-1).cpu().numpy()
    from sklearn.metrics import f1_score
    return {'val_acc': best_va,
            'test_acc': float((pred_te == y_te).mean()),
            'macro_f1': float(f1_score(y_te, pred_te, average='macro')),
            'per_class_f1': f1_score(y_te, pred_te, average=None).tolist()}


def run_model(name, hf_id, d_emb):
    print(f"\n=== {name} ({hf_id}) ===", flush=True)
    t0 = time.time()
    pipe = ChronosPipeline.from_pretrained(hf_id, device_map="cuda", dtype=torch.float32)
    n_params = sum(p.numel() for p in pipe.model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    results = []
    for seed in SEEDS:
        split = load_paderborn_split(seed)
        X_tr = embed_windows(pipe, split['X_train'])
        X_va = embed_windows(pipe, split['X_val'])
        X_te = embed_windows(pipe, split['X_test'])
        print(f"  seed {seed}: emb shapes {X_tr.shape}", flush=True)
        r = linear_probe(X_tr, split['y_train'], X_va, split['y_val'],
                         X_te, split['y_test'], seed)
        r['seed'] = seed
        results.append(r)
        print(f"    test acc {r['test_acc']:.3f}  macro-F1 {r['macro_f1']:.3f}",
              flush=True)

    del pipe; gc.collect(); torch.cuda.empty_cache()
    return {
        'model': name, 'n_params': n_params, 'd_emb': d_emb,
        'per_seed': results,
        'test_acc_mean': float(np.mean([r['test_acc'] for r in results])),
        'test_acc_std':  float(np.std([r['test_acc'] for r in results])),
        'macro_f1_mean': float(np.mean([r['macro_f1'] for r in results])),
        'macro_f1_std':  float(np.std([r['macro_f1'] for r in results])),
        'elapsed_min': (time.time() - t0) / 60,
    }


def main():
    V19.mkdir(exist_ok=True)
    t0 = time.time()
    results = []
    for (name, hf_id, d_emb) in MODELS:
        try:
            r = run_model(name, hf_id, d_emb)
            results.append(r)
            with open(V19/'phase7_chronos_paderborn_results.json', 'w') as f:
                json.dump({'config': 'v19_phase7_chronos_paderborn',
                           'models': results}, f, indent=2, default=float)
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({'model': name, 'error': str(e)})

    print("\n" + "=" * 70)
    print("V19 Phase 7: Chronos frozen-probe on Paderborn (vs FAM 0.783 acc)")
    print("=" * 70)
    print(f"{'model':<22} {'params':>12} {'test_acc':>14} {'macro_f1':>10}")
    for r in results:
        if 'error' in r:
            print(f"{r['model']:<22} FAILED: {r['error']}")
            continue
        print(f"{r['model']:<22} {r['n_params']:>12,} "
              f"{r['test_acc_mean']:>6.3f} +/- {r['test_acc_std']:<4.3f} "
              f"{r['macro_f1_mean']:>6.3f} +/- {r['macro_f1_std']:<4.3f}")
    print(f"{'FAM (ours)':<22} {1260000:>12,} "
          f"{0.783:>6.3f} +/- {0.036:<4.3f} {0.781:>6.3f} +/- {0.035:<4.3f}")
    print(f"\nRuntime: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()

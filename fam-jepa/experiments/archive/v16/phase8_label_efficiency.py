"""
Phase 8: V16b Label Efficiency vs V2.

Runs E2E fine-tuning at 5 label fractions [100%, 50%, 20%, 10%, 5%] for:
- V2 (causal, best E2E=14.23): already in paper, re-run for apples-to-apples comparison
- V16b (bidi+VICReg, E2E=15.06): check if bidi helps at LOW labels even if worse at 100%

Paper currently shows V2 at 100%=14.23, 50%=14.93, 20%=16.54, 10%=18.66, 5%=25.33.
Question: Does V16b improve on V2 at 10% or 5% labels?

3 seeds (not 5) for runtime. Uses 3 V16b checkpoints (seed42/123/456).
"""
import sys, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V16_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16')
sys.path.insert(0, str(V11_DIR))
sys.path.insert(0, str(V16_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from phase1_v16a import V16aJEPA, D_MODEL, N_HEADS, N_LAYERS, EMA_MOMENTUM
from train_utils import subsample_engines

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BUDGETS = [1.0, 0.5, 0.2, 0.1, 0.05]
N_SEEDS = 3


def run_v16b_e2e(train_eng, val_eng, test_eng, test_rul, ckpt_seed, seed):
    """Fine-tune V16b encoder E2E for RUL prediction."""
    model = V16aJEPA(
        n_sensors=N_SENSORS, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dropout=0.1, ema_momentum=EMA_MOMENTUM,
    ).to(DEVICE)
    ckpt_path = V16_DIR / f'best_v16b_seed{ckpt_seed}.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # Linear probe head
    probe = nn.Linear(D_MODEL, 1).to(DEVICE)

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    # E2E: unfreeze context encoder
    for p in model.parameters(): p.requires_grad = False
    for p in model.context_encoder.parameters(): p.requires_grad = True
    optim = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4
    )

    best_val = float('inf')
    best_probe_state = None
    best_enc_state = None
    no_impr = 0

    for ep in range(100):
        model.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            h = model.encode_context(past, mask)
            pred = probe(h).squeeze(-1)
            loss = F.mse_loss(pred, rul)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(model.context_encoder.parameters()) + list(probe.parameters()), 1.0)
            optim.step()

        model.eval()
        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_context(past, mask)
                pv.append(probe(h).squeeze(-1).cpu().numpy())
                tv.append(rul.numpy())
        # Note: val labels are normalized [0,1], compute RMSE in cycles
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv) * RUL_CAP - np.concatenate(tv) * RUL_CAP) ** 2
        )))
        if val_rmse < best_val:
            best_val = val_rmse
            best_probe_state = copy.deepcopy(probe.state_dict())
            best_enc_state = copy.deepcopy(model.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20:
                break

    probe.load_state_dict(best_probe_state)
    model.context_encoder.load_state_dict(best_enc_state)

    model.eval()
    probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_context(past, mask)
            pt.append(probe(h).squeeze(-1).cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt)) ** 2)))


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print("=" * 60)
    print("Phase 8: V16b Label Efficiency")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Budgets: {BUDGETS}")
    print(f"  Seeds: {N_SEEDS}")
    print()

    # Check checkpoints
    for s in [42, 123, 456]:
        p = V16_DIR / f'best_v16b_seed{s}.pt'
        if not p.exists():
            print(f"  Missing checkpoint: {p}")
            import sys; sys.exit(1)
        print(f"  Found: best_v16b_seed{s}.pt")
    print()

    data = load_cmapss_subset('FD001')

    results = {
        'description': 'V16b label efficiency - E2E fine-tuning at 5 label fractions',
        'budgets': BUDGETS,
        'n_seeds': N_SEEDS,
        'v16b_e2e': {},
    }

    # Map each seed pair (ckpt_seed, finetune_seed) to budget
    # Use 3 ckpt seeds matched to 3 finetune seeds
    ckpt_seeds = [42, 123, 456]

    for budget in BUDGETS:
        print(f"\n--- Budget: {budget*100:.0f}% ---")
        sub_eng = subsample_engines(data['train_engines'], budget, seed=42)
        print(f"  Training engines: {len(sub_eng)} / {len(data['train_engines'])}")

        seed_rmses = []
        for ckpt_seed in ckpt_seeds:
            rmse = run_v16b_e2e(
                sub_eng, data['val_engines'],
                data['test_engines'], data['test_rul'],
                ckpt_seed=ckpt_seed, seed=ckpt_seed
            )
            seed_rmses.append(rmse)
            print(f"  V16b E2E (ckpt_seed={ckpt_seed}): test RMSE = {rmse:.2f}")

        mean_rmse = float(np.mean(seed_rmses))
        std_rmse = float(np.std(seed_rmses))
        results['v16b_e2e'][budget] = {
            'mean': mean_rmse,
            'std': std_rmse,
            'per_seed': [float(r) for r in seed_rmses]
        }
        print(f"  -> Mean: {mean_rmse:.2f} +/- {std_rmse:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 8 SUMMARY: V16b E2E Label Efficiency")
    print("=" * 60)
    print(f"{'Budget':<10} {'V16b E2E':>15} {'V2 E2E (paper)':>18}")
    print("-" * 45)
    paper_v2 = {1.0: (14.23, 0.4), 0.5: (14.93, 0.4), 0.2: (16.54, 0.8),
                0.1: (18.66, 0.8), 0.05: (25.33, 5.1)}
    for b in BUDGETS:
        v16b = results['v16b_e2e'][b]
        v2 = paper_v2[b]
        delta = v16b['mean'] - v2[0]
        sign = '+' if delta > 0 else ''
        print(f"  {b*100:.0f}%{'':<7} {v16b['mean']:.2f}+/-{v16b['std']:.2f}  "
              f"  {v2[0]:.2f}+/-{v2[1]:.1f}   ({sign}{delta:.2f})")

    out_path = V16_DIR / 'phase8_label_efficiency.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

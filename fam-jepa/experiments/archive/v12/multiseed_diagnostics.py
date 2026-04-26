"""
Multi-seed trajectory diagnostics for FD001 E2E.
Run 5 seeds and compute mean/std of tracking metrics.
This provides statistical rigor for the Phase 0 verdict.

Output: experiments/v12/multiseed_phase0_diagnostics.json
"""

import json
import sys
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test
)
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'
SEEDS = [42, 123, 456, 789, 1024]

print("Multi-seed Phase 0 diagnostics (5 seeds)")
t0 = time.time()

data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']


def run_e2e_finetune(seed):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
    probe = RULProbe(256).to(DEVICE)

    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    for p in model.context_encoder.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(probe.parameters()),
        lr=1e-4
    )

    best_val, best_enc, best_pb, pc = float('inf'), None, None, 0
    for epoch in range(100):
        model.train(); probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pred = probe(h)
                pv.append(pred.cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_enc = copy.deepcopy(model.context_encoder.state_dict())
            best_pb = copy.deepcopy(probe.state_dict())
            pc = 0
        else:
            pc += 1
            if pc >= 20: break

    model.context_encoder.load_state_dict(best_enc)
    probe.load_state_dict(best_pb)
    model.eval(); probe.eval()

    # Test RMSE + trajectory diagnostics
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            pt.append(pred.cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    test_rmse = float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

    # Trajectory diagnostics for all test engines
    test_engine_ids = sorted(test_engines.keys())
    pred_stds, rhos = [], []
    with torch.no_grad():
        for idx, eid in enumerate(test_engine_ids):
            seq = test_engines[eid]
            T = seq.shape[0]
            oracle_rul = float(test_rul[idx])
            min_c = min(30, T)
            preds, trues = [], []
            for c in range(min_c, T+1):
                prefix = seq[:c]
                x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                h = model.encode_past(x, None)
                pred_norm = probe(h)
                preds.append(float(pred_norm.cpu().numpy()[0]) * RUL_CAP)
                trues.append(float(min(oracle_rul + (T - c), RUL_CAP)))
            preds = np.array(preds); trues = np.array(trues)
            pred_stds.append(float(np.std(preds)))
            if len(preds) >= 3 and preds.std() > 0 and trues.std() > 0:
                rho, _ = spearmanr(preds, trues)
                rhos.append(float(rho) if not np.isnan(rho) else 0.0)
            else:
                rhos.append(0.0)

    return {
        "seed": seed,
        "test_rmse": test_rmse,
        "pred_std_median": float(np.median(pred_stds)),
        "pred_std_mean": float(np.mean(pred_stds)),
        "rho_median": float(np.median(rhos)),
        "rho_mean": float(np.mean(rhos)),
        "n_rho_gt_07": int(np.sum(np.array(rhos) > 0.7)),
        "n_rho_gt_05": int(np.sum(np.array(rhos) > 0.5)),
    }


results = []
for seed in SEEDS:
    print(f"\nSeed {seed}...")
    ts = time.time()
    r = run_e2e_finetune(seed)
    r['wall_time_s'] = float(time.time() - ts)
    print(f"  RMSE={r['test_rmse']:.2f}, pred_std_med={r['pred_std_median']:.2f}, rho_med={r['rho_median']:.3f}")
    results.append(r)

# Aggregate
test_rmses = np.array([r['test_rmse'] for r in results])
pred_std_meds = np.array([r['pred_std_median'] for r in results])
rho_meds = np.array([r['rho_median'] for r in results])

print(f"\n=== 5-Seed Summary ===")
print(f"Test RMSE: {test_rmses.mean():.2f} +/- {test_rmses.std():.2f}")
print(f"Pred std median: {pred_std_meds.mean():.2f} +/- {pred_std_meds.std():.2f}")
print(f"Rho median: {rho_meds.mean():.3f} +/- {rho_meds.std():.3f}")

# All pass thresholds?
all_pass_std = all(v > 10 for v in pred_std_meds)
all_pass_rho = all(v > 0.5 for v in rho_meds)
print(f"\nAll seeds pass pred_std>10: {all_pass_std}")
print(f"All seeds pass rho>0.5: {all_pass_rho}")

summary = {
    "per_seed_results": results,
    "test_rmse_mean": float(test_rmses.mean()),
    "test_rmse_std": float(test_rmses.std()),
    "pred_std_median_mean": float(pred_std_meds.mean()),
    "pred_std_median_std": float(pred_std_meds.std()),
    "rho_median_mean": float(rho_meds.mean()),
    "rho_median_std": float(rho_meds.std()),
    "all_pass_tracking": bool(all_pass_std and all_pass_rho),
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'multiseed_phase0_diagnostics.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Saved to {V12_DIR / 'multiseed_phase0_diagnostics.json'}")

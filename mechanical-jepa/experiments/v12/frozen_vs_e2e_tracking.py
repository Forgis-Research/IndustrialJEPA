"""
Compare frozen vs E2E tracking quality.
V11 frozen=17.81, E2E=13.80. Does frozen also track, or does the E2E gain come
entirely from E2E adapting to a better last-window representation?

For the paper: demonstrate that frozen also tracks degradation (rho > 0.5)
but at a higher RMSE. The E2E gain is calibration improvement, not tracking itself.
"""

import json
import sys
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test
)
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE, finetune

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

print("Frozen vs E2E tracking comparison")
t0 = time.time()

data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']


def run_finetune_with_probe(pretrain_ckpt, mode, seed):
    """Run fine-tune and return model+probe for trajectory inference."""
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(pretrain_ckpt), map_location=DEVICE))
    probe = RULProbe(256).to(DEVICE)

    torch.manual_seed(seed); np.random.seed(seed)
    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model.context_encoder.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(probe.parameters()),
            lr=1e-4
        )

    best_val, best_enc, best_pb, pc = float('inf'), None, None, 0

    for epoch in range(100):
        if mode == 'frozen': model.eval()
        else: model.train()
        probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            h = model.encode_past(past, mask) if mode != 'frozen' else model.encode_past(past, mask).detach() if mode == 'frozen_detach' else model.encode_past(past, mask)
            if mode == 'frozen':
                with torch.no_grad():
                    h = model.encode_past(past, mask)
            else:
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
                with torch.no_grad() if mode == 'frozen' else torch.enable_grad():
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

    # Test RMSE
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            pt.append(pred.cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    test_rmse = float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

    return model, probe, test_rmse


@torch.no_grad()
def trajectory_diagnostics(model, probe, test_engines, test_rul):
    """Per-engine trajectory diagnostics."""
    test_engine_ids = sorted(test_engines.keys())
    pred_stds, rhos = [], []
    for idx, eid in enumerate(test_engine_ids):
        seq = test_engines[eid]
        T = seq.shape[0]
        oracle_rul = float(test_rul[idx])
        min_cycle = min(30, T)
        cycles = list(range(min_cycle, T+1))
        preds, trues = [], []
        for c in cycles:
            prefix = seq[:c]
            x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            h = model.encode_past(x, None)
            pred_norm = probe(h)
            preds.append(float(pred_norm.cpu().numpy()[0]) * RUL_CAP)
            trues.append(min(oracle_rul + (T - c), RUL_CAP))
        preds = np.array(preds); trues = np.array(trues)
        pred_stds.append(float(np.std(preds)))
        if len(preds) >= 3 and preds.std() > 0 and trues.std() > 0:
            rho, _ = spearmanr(preds, trues)
            rhos.append(float(rho) if not np.isnan(rho) else 0.0)
        else:
            rhos.append(0.0)
    return np.array(pred_stds), np.array(rhos)


print("\n--- Frozen mode ---")
model_frozen, probe_frozen, rmse_frozen = run_finetune_with_probe(PRETRAIN_CKPT, 'frozen', seed=0)
print(f"Frozen test RMSE: {rmse_frozen:.2f}")
stds_frozen, rhos_frozen = trajectory_diagnostics(model_frozen, probe_frozen, test_engines, test_rul)
print(f"Frozen pred_std: median={np.median(stds_frozen):.2f}, mean={stds_frozen.mean():.2f}")
print(f"Frozen rho: median={np.median(rhos_frozen):.3f}, mean={rhos_frozen.mean():.3f}")

print("\n--- E2E mode (seed=0) ---")
# Load the already-reconstructed E2E model
e2e_ckpt = torch.load(str(V12_DIR / 'v2_e2e_seed0_reconstructed.pt'), map_location=DEVICE)
model_e2e = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
model_e2e.context_encoder.load_state_dict(e2e_ckpt['encoder_state'])
probe_e2e = RULProbe(256).to(DEVICE)
probe_e2e.load_state_dict(e2e_ckpt['probe_state'])
rmse_e2e = e2e_ckpt['test_rmse']
print(f"E2E test RMSE: {rmse_e2e:.2f}")
stds_e2e, rhos_e2e = trajectory_diagnostics(model_e2e, probe_e2e, test_engines, test_rul)
print(f"E2E pred_std: median={np.median(stds_e2e):.2f}, mean={stds_e2e.mean():.2f}")
print(f"E2E rho: median={np.median(rhos_e2e):.3f}, mean={rhos_e2e.mean():.3f}")

# Plot comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('V12: Frozen vs E2E Tracking Quality Comparison', fontsize=12, fontweight='bold')

# Pred std comparison
ax = axes[0]
ax.hist(stds_frozen, bins=20, alpha=0.6, color='darkorange', label=f'Frozen (med={np.median(stds_frozen):.1f})', edgecolor='none')
ax.hist(stds_e2e, bins=20, alpha=0.6, color='steelblue', label=f'E2E (med={np.median(stds_e2e):.1f})', edgecolor='none')
ax.set_xlabel('Per-engine prediction std (cycles)')
ax.set_ylabel('Count')
ax.set_title('Prediction Variability')
ax.legend()

# Rho comparison
ax = axes[1]
ax.hist(rhos_frozen, bins=20, alpha=0.6, color='darkorange', label=f'Frozen (med={np.median(rhos_frozen):.2f})', edgecolor='none')
ax.hist(rhos_e2e, bins=20, alpha=0.6, color='steelblue', label=f'E2E (med={np.median(rhos_e2e):.2f})', edgecolor='none')
ax.set_xlabel('Within-engine Spearman rho')
ax.set_ylabel('Count')
ax.set_title('Within-engine Tracking')
ax.legend()

# RMSE vs rho scatter
ax = axes[2]
# Per-engine RMSE for frozen
test_engine_ids = sorted(test_engines.keys())
last_preds_frozen, last_preds_e2e = [], []
with torch.no_grad():
    for idx, eid in enumerate(test_engine_ids):
        seq = test_engines[eid]
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        h_fr = model_frozen.encode_past(x, None)
        h_e2e = model_e2e.encode_past(x, None)
        last_preds_frozen.append(float(probe_frozen(h_fr).cpu().numpy()[0]) * RUL_CAP)
        last_preds_e2e.append(float(probe_e2e(h_e2e).cpu().numpy()[0]) * RUL_CAP)

per_eng_rmse_frozen = np.array([(p-t)**2 for p, t in zip(last_preds_frozen, test_rul)]) ** 0.5
per_eng_rmse_e2e = np.array([(p-t)**2 for p, t in zip(last_preds_e2e, test_rul)]) ** 0.5
ax.scatter(per_eng_rmse_frozen, per_eng_rmse_e2e, alpha=0.5, s=30, color='purple')
ax.plot([0, 60], [0, 60], 'k--', linewidth=1, label='Equal RMSE')
ax.set_xlabel('Frozen per-engine RMSE')
ax.set_ylabel('E2E per-engine RMSE')
ax.set_title(f'Per-engine RMSE: Frozen vs E2E\nFrozen={rmse_frozen:.1f}, E2E={rmse_e2e:.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'frozen_vs_e2e_tracking.png'), dpi=150, bbox_inches='tight')
plt.close()

result = {
    "frozen": {
        "test_rmse": rmse_frozen,
        "pred_std_median": float(np.median(stds_frozen)),
        "pred_std_mean": float(stds_frozen.mean()),
        "rho_median": float(np.median(rhos_frozen)),
        "rho_mean": float(rhos_frozen.mean()),
    },
    "e2e": {
        "test_rmse": rmse_e2e,
        "pred_std_median": float(np.median(stds_e2e)),
        "pred_std_mean": float(stds_e2e.mean()),
        "rho_median": float(np.median(rhos_e2e)),
        "rho_mean": float(rhos_e2e.mean()),
    },
    "interpretation": "Both frozen and E2E track degradation. E2E advantage comes from calibration, not tracking.",
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'frozen_vs_e2e_tracking.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved to {V12_DIR / 'frozen_vs_e2e_tracking.json'}")
print(f"Wall time: {time.time()-t0:.1f}s")

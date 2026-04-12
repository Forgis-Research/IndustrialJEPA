"""
Phase 0b.3: Shuffle test at fine-tune time.
If shuffled h_past -> same RMSE, h_past is input-independent bias.

This is informative even when Phase 0 passes - it quantifies how much
h_past content matters vs probe bias.

Output: experiments/v12/shuffle_test.json
"""

import json
import sys
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
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
E2E_CKPT = V12_DIR / 'v2_e2e_seed0_reconstructed.pt'

print("=" * 60)
print("Phase 0b.3: h_past Shuffle Test")
print("=" * 60)
print(f"Device: {DEVICE}")
t0 = time.time()

data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']


@torch.no_grad()
def eval_test_rmse(model, probe, test_engines, test_rul, shuffle_h=False, device=DEVICE):
    """Evaluate test RMSE. If shuffle_h=True, shuffle h_past across batch before probe."""
    model.eval(); probe.eval()

    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_test)

    preds, targets = [], []
    for past, mask, rul_gt in test_loader:
        past, mask = past.to(device), mask.to(device)
        h = model.encode_past(past, mask)
        if shuffle_h:
            idx = torch.randperm(h.shape[0])
            h = h[idx]
        pred_norm = probe(h)
        preds.append(pred_norm.cpu().numpy() * RUL_CAP)
        targets.append(rul_gt.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# Load reconstructed E2E checkpoint
print(f"\nLoading reconstructed E2E model...")
model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2, d_ff=512,
    dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
probe = RULProbe(256).to(DEVICE)

ckpt = torch.load(str(E2E_CKPT), map_location=DEVICE)
model.context_encoder.load_state_dict(ckpt['encoder_state'])
probe.load_state_dict(ckpt['probe_state'])

# Normal test RMSE
normal_rmse = eval_test_rmse(model, probe, test_engines, test_rul, shuffle_h=False)
print(f"Normal test RMSE: {normal_rmse:.2f}")

# Shuffled h_past RMSE (5 repetitions for stability)
shuffled_rmses = []
for seed in range(5):
    torch.manual_seed(seed * 42)
    s_rmse = eval_test_rmse(model, probe, test_engines, test_rul, shuffle_h=True)
    shuffled_rmses.append(s_rmse)
    print(f"  Shuffle seed {seed}: RMSE={s_rmse:.2f}")

shuffled_mean = float(np.mean(shuffled_rmses))
shuffled_std = float(np.std(shuffled_rmses))
rmse_gain = shuffled_mean - normal_rmse

print(f"\nNormal RMSE: {normal_rmse:.2f}")
print(f"Shuffled RMSE: {shuffled_mean:.2f} +/- {shuffled_std:.2f}")
print(f"Gain from unshuffled h_past: {rmse_gain:+.2f}")

if rmse_gain > 5:
    verdict = f"STRONG: h_past carries {rmse_gain:.1f} RMSE of information (shuffle gap). Representation is input-dependent."
elif rmse_gain > 1:
    verdict = f"MODERATE: h_past carries {rmse_gain:.1f} RMSE improvement. Partial dependence on input."
else:
    verdict = f"WEAK: h_past carries only {rmse_gain:.1f} RMSE improvement. h_past may be nearly input-independent bias."

print(f"Verdict: {verdict}")

results = {
    "normal_rmse": normal_rmse,
    "shuffled_rmse_mean": shuffled_mean,
    "shuffled_rmse_std": shuffled_std,
    "shuffled_per_seed": shuffled_rmses,
    "rmse_gain_from_h_past": rmse_gain,
    "verdict": verdict,
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'shuffle_test.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {V12_DIR / 'shuffle_test.json'}")

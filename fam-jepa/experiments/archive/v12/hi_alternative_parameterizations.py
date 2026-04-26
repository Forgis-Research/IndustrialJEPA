"""
Alternative H.I. parameterizations - robustness check for Phase 3.

Standard (Phase 3): piecewise linear with cliff at T-125.
Alternatives:
1. Sigmoid: H.I. = 1/(1+exp(k*(t-t0))) where t0=T-62.5, k=0.1
2. Exponential: H.I. = exp(-lambda*(t-T)) normalized to [0,1]
3. Raw RUL / 125 (no plateau, direct normalization)

If R² is robust across parameterizations, the encoder learning is genuine.
"""

import json
import sys
import numpy as np
import torch
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from models import TrajectoryJEPA
from train_utils import DEVICE

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

print("Alternative H.I. parameterizations")
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']

model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
model.eval()


def compute_hi_piecewise(T, cap=RUL_CAP):
    hi = np.ones(T, dtype=np.float32)
    degrade_start = max(0, T - cap)
    if degrade_start < T:
        n_degrade = T - degrade_start
        hi[degrade_start:] = np.linspace(1.0, 0.0, n_degrade)
    return hi


def compute_hi_sigmoid(T, cap=RUL_CAP, k=0.08):
    t0 = T - cap / 2.0
    t_arr = np.arange(T, dtype=np.float32)
    hi = 1.0 / (1.0 + np.exp(k * (t_arr - t0)))
    return hi.astype(np.float32)


def compute_hi_raw_rul(T, cap=RUL_CAP):
    rul = np.arange(T, 0, -1, dtype=np.float32)
    return np.minimum(rul / cap, 1.0)


@torch.no_grad()
def get_all_embeddings(model, engines, min_cycle=10):
    X_all = []
    y_all_piecewise = []
    y_all_sigmoid = []
    y_all_raw_rul = []

    for eid, seq in engines.items():
        T = seq.shape[0]
        hi_pw = compute_hi_piecewise(T)
        hi_sig = compute_hi_sigmoid(T)
        hi_raw = compute_hi_raw_rul(T)

        for c in range(min_cycle, T+1):
            prefix = seq[:c]
            x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            h = model.encode_past(x, None)
            X_all.append(h.cpu().numpy()[0])
            y_all_piecewise.append(hi_pw[c-1])
            y_all_sigmoid.append(hi_sig[c-1])
            y_all_raw_rul.append(hi_raw[c-1])

    return (np.stack(X_all), np.array(y_all_piecewise, dtype=np.float32),
            np.array(y_all_sigmoid, dtype=np.float32),
            np.array(y_all_raw_rul, dtype=np.float32))


print("Computing embeddings for training engines...")
X_train, y_tr_pw, y_tr_sig, y_tr_raw = get_all_embeddings(model, train_engines)
print(f"Training: {X_train.shape[0]} samples")

print("Computing embeddings for validation engines...")
X_val, y_val_pw, y_val_sig, y_val_raw = get_all_embeddings(model, val_engines)
print(f"Validation: {X_val.shape[0]} samples")

results = {}
for name, y_tr, y_val in [
    ("piecewise_linear", y_tr_pw, y_val_pw),
    ("sigmoid", y_tr_sig, y_val_sig),
    ("raw_rul_norm", y_tr_raw, y_val_raw),
]:
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_tr)
    pred_tr = ridge.predict(X_train)
    pred_val = ridge.predict(X_val)
    r2_tr = float(r2_score(y_tr, pred_tr))
    r2_v = float(r2_score(y_val, pred_val))
    print(f"  {name:25s}: train R²={r2_tr:.4f}, val R²={r2_v:.4f}")
    results[name] = {"r2_train": r2_tr, "r2_val": r2_v}

print("\nConclusion:")
all_val_r2 = [v['r2_val'] for v in results.values()]
if all(r > 0.7 for r in all_val_r2):
    print(f"ALL parameterizations show val R²>0.7 ({[f'{r:.3f}' for r in all_val_r2]}). Robust result.")
else:
    print(f"Some parameterizations < 0.7: {results}")

with open(V12_DIR / 'hi_alternative_params.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {V12_DIR / 'hi_alternative_params.json'}")

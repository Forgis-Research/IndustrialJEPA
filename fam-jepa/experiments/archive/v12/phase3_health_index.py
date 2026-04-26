"""
Phase 3: Health-index recovery probe

Show that frozen V2 encoder's h_past linearly decodes approximate H.I.
H.I. = piecewise linear: 1.0 for [1, T-125], linear 1.0->0.0 for [T-125, T]

Output:
  analysis/plots/v12/health_index_recovery.png
  (R² reported in RESULTS.md)
"""

import json
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from models import TrajectoryJEPA
from train_utils import DEVICE

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

print("=" * 60)
print("Phase 3: Health-Index Recovery")
print("=" * 60)
print(f"Device: {DEVICE}")
t0 = time.time()

# Load data
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']

print(f"Train engines: {len(train_engines)}, Val engines: {len(val_engines)}")


def compute_hi(seq_length: int) -> np.ndarray:
    """
    Compute piecewise-linear health index.
    H.I. = 1.0 for cycles [1, T-125] (healthy plateau)
    H.I. = linear 1.0->0.0 for [T-125, T] (degradation)
    Returns array of shape (T,)
    """
    T = seq_length
    hi = np.ones(T, dtype=np.float32)
    degradation_start = max(0, T - RUL_CAP)

    if degradation_start < T:
        # Linear from 1.0 to 0.0 over [degradation_start, T-1]
        n_degrade = T - degradation_start
        ramp = np.linspace(1.0, 0.0, n_degrade, dtype=np.float32)
        hi[degradation_start:] = ramp

    return hi


# Load V2 model
print(f"\nLoading V2 pretrained model from {PRETRAIN_CKPT}...")
model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1,
    d_model=256, n_heads=4, n_layers=2, d_ff=512,
    dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
model.eval()


@torch.no_grad()
def get_h_past_at_every_cycle(model, seq, min_cycle=10):
    """
    Compute h_past at every cycle from min_cycle to T.
    seq: (T, 14) numpy array
    Returns: embeddings (T-min_cycle+1, d_model), cycle_indices (T-min_cycle+1,)
    """
    T = seq.shape[0]
    embeddings = []
    cycle_indices = list(range(min_cycle, T + 1))

    for c in cycle_indices:
        prefix = seq[:c]  # (c, 14)
        x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        h = model.encode_past(x, None)  # (1, d_model)
        embeddings.append(h.cpu().numpy()[0])

    return np.stack(embeddings), np.array(cycle_indices)


# Compute h_past and H.I. for all training engines
print("\nComputing h_past embeddings at every cycle for training engines...")
t_embed = time.time()

X_train_hi = []
y_train_hi = []

for i, (eid, seq) in enumerate(train_engines.items()):
    T = seq.shape[0]
    hi = compute_hi(T)

    # Get embeddings from cycle 10 to T
    embs, cycles = get_h_past_at_every_cycle(model, seq, min_cycle=10)

    # H.I. at those cycles (0-indexed: cycle c -> hi[c-1])
    hi_at_cycles = hi[np.array(cycles) - 1]

    X_train_hi.append(embs)
    y_train_hi.append(hi_at_cycles)

    if i % 20 == 0:
        print(f"  Processed {i+1}/{len(train_engines)} training engines ({time.time()-t_embed:.1f}s)")

X_train_hi = np.vstack(X_train_hi)
y_train_hi = np.concatenate(y_train_hi)
print(f"Training set: {X_train_hi.shape[0]} samples")

# Validation embeddings
print("\nComputing h_past embeddings for validation engines...")
X_val_hi = []
y_val_hi = []

for eid, seq in val_engines.items():
    T = seq.shape[0]
    hi = compute_hi(T)
    embs, cycles = get_h_past_at_every_cycle(model, seq, min_cycle=10)
    hi_at_cycles = hi[np.array(cycles) - 1]
    X_val_hi.append(embs)
    y_val_hi.append(hi_at_cycles)

X_val_hi = np.vstack(X_val_hi)
y_val_hi = np.concatenate(y_val_hi)
print(f"Validation set: {X_val_hi.shape[0]} samples")

# Fit Ridge regression
print("\nFitting Ridge(alpha=1.0) regression on h_past -> H.I. ...")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_hi, y_train_hi)

pred_train = ridge.predict(X_train_hi)
pred_val = ridge.predict(X_val_hi)

r2_train = float(r2_score(y_train_hi, pred_train))
r2_val = float(r2_score(y_val_hi, pred_val))

print(f"Train R²: {r2_train:.4f}")
print(f"Val R²:   {r2_val:.4f}")
print(f"Target:   > 0.7")

# Also compute MSE for reference
mse_train = float(np.mean((pred_train - y_train_hi) ** 2))
mse_val = float(np.mean((pred_val - y_val_hi) ** 2))
print(f"Train MSE: {mse_train:.6f}")
print(f"Val MSE:   {mse_val:.6f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Phase 3: Health Index Recovery from Frozen h_past\nTrain R²={r2_train:.3f}, Val R²={r2_val:.3f}',
             fontsize=12, fontweight='bold')

# Scatter: true vs predicted on validation
sample_idx = np.random.RandomState(42).choice(len(y_val_hi), size=min(2000, len(y_val_hi)), replace=False)
axes[0].scatter(y_val_hi[sample_idx], pred_val[sample_idx], alpha=0.3, s=5, color='steelblue')
axes[0].plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Perfect')
axes[0].set_xlabel('True H.I.')
axes[0].set_ylabel('Predicted H.I.')
axes[0].set_title(f'Val Scatter (R²={r2_val:.3f})')
axes[0].legend()
axes[0].set_xlim(-0.1, 1.2)
axes[0].set_ylim(-0.1, 1.2)

# Sample H.I. trajectory for 3 val engines
colors = ['steelblue', 'darkorange', 'green']
for i, (eid, seq) in enumerate(list(val_engines.items())[:3]):
    T = seq.shape[0]
    hi_true = compute_hi(T)
    embs, cycles = get_h_past_at_every_cycle(model, seq, min_cycle=10)
    hi_pred = ridge.predict(embs)
    hi_at_cycles = hi_true[np.array(cycles) - 1]
    axes[1].plot(cycles, hi_at_cycles, color=colors[i], linewidth=2, label=f'Eng {eid} true')
    axes[1].plot(cycles, hi_pred, color=colors[i], linewidth=1.5, linestyle='--', label=f'Eng {eid} pred')
axes[1].set_xlabel('Cycle')
axes[1].set_ylabel('Health Index')
axes[1].set_title('Sample Val Engine Trajectories')
axes[1].legend(fontsize=7)
axes[1].set_ylim(-0.1, 1.3)

# R² comparison
categories = ['Train R²', 'Val R²', 'Target (0.7)']
values = [r2_train, r2_val, 0.7]
bar_colors = ['steelblue', 'darkorange', 'gray']
bars = axes[2].bar(categories, values, color=bar_colors, alpha=0.8, edgecolor='black')
axes[2].axhline(0.7, color='red', linestyle='--', linewidth=1.5, label='Target=0.7')
axes[2].set_ylabel('R²')
axes[2].set_title('R² Summary')
axes[2].set_ylim(0, 1.1)
for bar, val in zip(bars, values):
    axes[2].text(bar.get_x() + bar.get_width()/2, val + 0.02, f'{val:.3f}', ha='center', va='bottom')
axes[2].legend()

plt.tight_layout()
hi_plot_path = PLOTS_V12 / 'health_index_recovery.png'
plt.savefig(str(hi_plot_path), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved plot to {hi_plot_path}")

if r2_val > 0.7:
    verdict = f"PASS: Val R²={r2_val:.3f} > 0.7. Frozen encoder linearly decodes health index. Strong SSL evidence."
elif r2_val > 0.4:
    verdict = f"PARTIAL: Val R²={r2_val:.3f} between 0.4-0.7. Encoder partially encodes H.I. but signal is mixed."
else:
    verdict = f"FAIL: Val R²={r2_val:.3f} < 0.4. Kill criterion triggered: SSL objective is not learning degradation."

print(f"\nVerdict: {verdict}")

results = {
    "r2_train": r2_train,
    "r2_val": r2_val,
    "mse_train": mse_train,
    "mse_val": mse_val,
    "target_r2": 0.7,
    "verdict": verdict,
    "n_train_samples": int(X_train_hi.shape[0]),
    "n_val_samples": int(X_val_hi.shape[0]),
    "embedding_dim": int(X_train_hi.shape[1]),
    "wall_time_s": float(time.time() - t0),
}

with open(V12_DIR / 'health_index_recovery.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {V12_DIR / 'health_index_recovery.json'}")
print(f"Wall time: {time.time() - t0:.1f}s")

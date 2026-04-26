"""
Generate prediction trajectory plots for V11 paper.
These require model inference but can run with existing checkpoints.
"""
import os, sys, copy, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings('ignore')
BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
EXP_DIR = os.path.join(BASE, 'experiments/v11')
PLOTS_DIR = os.path.join(BASE, 'analysis/plots/v11')
sys.path.insert(0, EXP_DIR)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test, N_SENSORS, RUL_CAP
)
from models import TrajectoryJEPA, RULProbe
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

def train_e2e_probe(ckpt_path, d_model, n_layers, n_heads, d_ff,
                    train_eng, val_eng, seed):
    """Quick E2E fine-tune to get a good model for visualization."""
    model = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=d_model,
                            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(DEVICE)
    probe = RULProbe(d_model).to(DEVICE)

    for p in model.context_encoder.parameters(): p.requires_grad = True
    optim = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4
    )
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)

    best_v, best_ps, best_es, ni = float('inf'), None, None, 0
    for ep in range(100):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            h = model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward()
            torch.nn.utils.clip_grad_norm_(
                list(probe.parameters()) + list(model.context_encoder.parameters()), 1.0
            )
            optim.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_r < best_v:
            best_v = val_r
            best_ps = copy.deepcopy(probe.state_dict())
            best_es = copy.deepcopy(model.context_encoder.state_dict())
            ni = 0
        else:
            ni += 1
            if ni >= 20: break

    probe.load_state_dict(best_ps)
    model.context_encoder.load_state_dict(best_es)
    model.eval(); probe.eval()
    return model, probe

# Load data and model
print("Loading FD001 data...")
data = load_cmapss_subset('FD001')
ckpt_v2 = os.path.join(EXP_DIR, 'best_pretrain_L1_v2.pt')

print("Training E2E model for visualization...")
model, probe = train_e2e_probe(ckpt_v2, 256, 2, 4, 512,
                                data['train_engines'], data['val_engines'], seed=42)

test_engines = data['test_engines']
test_rul = data['test_rul']
eng_ids = sorted(test_engines.keys())  # list of integer engine IDs

# ============================================================
# Plot 1: Prediction trajectories for 5 engines
# ============================================================
print("Generating prediction trajectory plots...")

# Pick representative engines (by total lifetime)
# test_engines[eid] is a numpy array of shape (T, 14)
lengths = {eid: len(test_engines[eid]) for eid in eng_ids}
sorted_by_length = sorted(lengths.items(), key=lambda x: x[1])
n_total = len(sorted_by_length)
# Short, medium-short, medium, medium-long, long
chosen_idx = [5, int(n_total*0.25), int(n_total*0.5), int(n_total*0.75), n_total-5]
chosen_ids = [sorted_by_length[i][0] for i in chosen_idx]

# Build id-to-index mapping for test_rul lookup
id_to_idx = {eid: idx for idx, eid in enumerate(eng_ids)}

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle('V11 Trajectory JEPA: Predicted vs True RUL (V2 E2E, FD001 Test Engines)',
             fontsize=12, fontweight='bold')

for i, eid in enumerate(chosen_ids):
    eng = test_engines[eid]  # shape (T, 14)
    true_final_rul = float(test_rul[id_to_idx[eid]])
    T = len(eng)
    x_tensor = torch.tensor(eng, dtype=torch.float32)

    # Compute predictions at all cut points from min_past to T
    min_past = 10
    cut_points = list(range(min_past, T))
    preds = []
    true_ruls = []

    with torch.no_grad():
        for t in cut_points:
            past = x_tensor[:t].unsqueeze(0).to(DEVICE)
            mask = torch.ones(1, t, dtype=torch.bool).to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h).item() * RUL_CAP
            preds.append(pred)
            # True RUL at this cut: final_rul + cycles remaining in test window
            true_r = float(min(true_final_rul + (T - t), RUL_CAP))
            true_ruls.append(true_r)

    preds = np.array(preds)
    true_ruls = np.array(true_ruls)

    ax = axes[i]
    ax.plot(cut_points, true_ruls, 'b-', linewidth=2, label='True RUL', alpha=0.8)
    ax.plot(cut_points, preds, 'r--', linewidth=1.5, label='Predicted RUL', alpha=0.8)
    ax.fill_between(cut_points,
                    np.maximum(0, preds - 15),
                    np.minimum(125, preds + 15),
                    alpha=0.15, color='red', label='±15 cycles band')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.axhline(RUL_CAP, color='gray', linewidth=0.5, linestyle=':')
    final_rmse = float(np.sqrt(np.mean((preds[-1] - true_final_rul)**2))) if true_final_rul is not None else float('nan')
    ax.set_title(f'Engine {eid} (T={T} cycles)\nFinal pred={preds[-1]:.0f}, True={true_final_rul:.0f}',
                 fontsize=9)
    ax.set_xlabel('Observation cycle')
    if i == 0:
        ax.set_ylabel('RUL (cycles)')
        ax.legend(fontsize=7)
    ax.set_ylim(-5, 135)
    ax.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(PLOTS_DIR, 'prediction_trajectories_v2.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved prediction_trajectories_v2.png")

# ============================================================
# Plot 2: All test engines final predictions vs true RUL
# ============================================================
print("Generating scatter plot: all test engines...")
te_ds = CMAPSSTestDataset(test_engines, test_rul)
te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

all_preds, all_trues = [], []
with torch.no_grad():
    for past, mask, rg in te:
        past, mask = past.to(DEVICE), mask.to(DEVICE)
        all_preds.append(probe(model.encode_past(past, mask)).cpu().numpy() * RUL_CAP)
        all_trues.append(rg.numpy())

preds_arr = np.concatenate(all_preds)
trues_arr = np.concatenate(all_trues)
rmse = float(np.sqrt(np.mean((preds_arr - trues_arr)**2)))
errors = preds_arr - trues_arr

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'V11 Prediction Analysis: V2 E2E @ 100% Labels (FD001, RMSE={rmse:.2f})',
             fontsize=12, fontweight='bold')

# Scatter: true vs predicted
ax = axes[0]
sc = ax.scatter(trues_arr, preds_arr, c=np.abs(errors), cmap='RdYlGn_r', alpha=0.5, s=15)
ax.plot([0, 125], [0, 125], 'k--', linewidth=1.5, label='Perfect')
ax.plot([0, 125], [15, 140], 'b--', linewidth=0.8, alpha=0.5, label='+15 bias')
ax.plot([0, 125], [-15, 110], 'b--', linewidth=0.8, alpha=0.5)
plt.colorbar(sc, ax=ax, label='|Error|')
ax.set_xlabel('True RUL (cycles)'); ax.set_ylabel('Predicted RUL (cycles)')
ax.set_title('True vs Predicted RUL'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Error distribution
ax2 = axes[1]
ax2.hist(errors, bins=30, color='#3498db', alpha=0.7, edgecolor='white')
ax2.axvline(0, color='black', linestyle='--', linewidth=1.5)
ax2.axvline(np.mean(errors), color='red', linestyle='--', linewidth=1.5,
            label=f'Mean={np.mean(errors):.1f}')
ax2.axvline(np.median(errors), color='orange', linestyle='--', linewidth=1.5,
            label=f'Median={np.median(errors):.1f}')
ax2.set_xlabel('Prediction Error (cycles)'); ax2.set_ylabel('Count')
ax2.set_title(f'Error Distribution (std={np.std(errors):.1f})')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

# Error vs True RUL (bias analysis)
ax3 = axes[2]
ax3.scatter(trues_arr, errors, alpha=0.4, s=15, color='#e74c3c')
# Moving average
sort_idx = np.argsort(trues_arr)
window = 20
if len(trues_arr) > window:
    from scipy.ndimage import uniform_filter1d
    sorted_errors = errors[sort_idx]
    smoothed = uniform_filter1d(sorted_errors, size=window)
    ax3.plot(trues_arr[sort_idx], smoothed, 'b-', linewidth=2, label='Moving avg (20 pts)')
ax3.axhline(0, color='black', linestyle='--', linewidth=1.5)
ax3.set_xlabel('True RUL (cycles)'); ax3.set_ylabel('Prediction Error (cycles)')
ax3.set_title('Error vs True RUL (bias check)')
ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

plt.tight_layout()
out_path2 = os.path.join(PLOTS_DIR, 'prediction_scatter_v2.png')
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved prediction_scatter_v2.png")

print(f"\nFinal RMSE: {rmse:.2f}")
print(f"Mean error: {np.mean(errors):.2f}")
print(f"Median error: {np.median(errors):.2f}")
print(f"Std error: {np.std(errors):.2f}")
late_frac = float(np.mean(errors > 0))
print(f"Late predictions (overestimate RUL): {late_frac:.1%}")

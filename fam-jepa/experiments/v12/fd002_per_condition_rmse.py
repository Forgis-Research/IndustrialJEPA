"""
Extra FD002 per-condition RMSE breakdown.
For each test engine, find its operating condition and compute RMSE per condition.
This shows which conditions are causing the val/test gap.
"""

import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import (
    load_cmapss_subset, load_raw, get_sensor_cols, get_op_cols, N_SENSORS, RUL_CAP,
    CMAPSSTestDataset, collate_test
)
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE, finetune

print("FD002 per-condition RMSE breakdown")
data = load_cmapss_subset('FD002')
train_engines = data['train_engines']
val_engines = data['val_engines']
test_engines = data['test_engines']
test_rul = data['test_rul']
stats = data['stats']

train_df_002, test_df_002, test_rul_arr = load_raw('FD002')
op_cols = get_op_cols()

# Load V11 pretrained FD002 model
model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
model.load_state_dict(torch.load(str(V11_DIR / 'best_pretrain_fd002.pt'), map_location=DEVICE))

# Run one frozen fine-tune
result_frozen = finetune(
    model=model,
    train_engines=train_engines,
    val_engines=val_engines,
    test_engines=test_engines,
    test_rul=test_rul,
    n_epochs=100,
    mode='frozen',
    seed=42,
    verbose=False,
)
print(f"Frozen RMSE (seed=42): {result_frozen['test_rmse']:.2f}")

# Get per-engine predictions
model.eval()
# We need the probe - do this differently
# Re-run fine-tune with access to probe
probe = RULProbe(256).to(DEVICE)
import torch.nn.functional as F
import copy

torch.manual_seed(42); import numpy as np; np.random.seed(42)
from data_utils import CMAPSSFinetuneDataset, collate_finetune
train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=42)
val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
test_ds = CMAPSSTestDataset(test_engines, test_rul)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

for p in model.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
best_val, best_pb, pc = float('inf'), None, 0

for epoch in range(100):
    model.eval(); probe.train()
    for past, mask, rul in train_loader:
        past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
        optimizer.zero_grad()
        with torch.no_grad():
            h = model.encode_past(past, mask)
        pred = probe(h)
        loss = F.mse_loss(pred, rul)
        loss.backward()
        optimizer.step()

    probe.eval()
    preds_v, tgts_v = [], []
    with torch.no_grad():
        for past, mask, rul in val_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            preds_v.append(pred.cpu().numpy()); tgts_v.append(rul.numpy())
    val_rmse = float(np.sqrt(np.mean((np.concatenate(preds_v)*RUL_CAP - np.concatenate(tgts_v)*RUL_CAP)**2)))
    if val_rmse < best_val:
        best_val = val_rmse; best_pb = copy.deepcopy(probe.state_dict()); pc = 0
    else:
        pc += 1
        if pc >= 20: break

probe.load_state_dict(best_pb)

# Per-engine predictions on test set
model.eval(); probe.eval()
test_engine_ids = sorted(test_engines.keys())
all_preds = []
with torch.no_grad():
    for past, mask, rul_gt in test_loader:
        past, mask = past.to(DEVICE), mask.to(DEVICE)
        h = model.encode_past(past, mask)
        pred = probe(h)
        all_preds.extend(pred.cpu().numpy() * RUL_CAP)

all_preds = np.array(all_preds)
test_rmse_total = float(np.sqrt(np.mean((all_preds - test_rul)**2)))
print(f"Reproduced test RMSE: {test_rmse_total:.2f}")

# Assign conditions to each test engine's last window
kmeans = stats['_kmeans']  # from per-condition normalization

test_conds = []
for eid in test_engine_ids:
    eng = test_df_002[test_df_002['engine_id'] == eid].sort_values('cycle')
    # Last row
    last_row = eng.tail(1)[op_cols].values
    cond = int(kmeans.predict(last_row)[0])
    test_conds.append(cond)

test_conds = np.array(test_conds)

# Per-condition RMSE
print("\nPer-condition breakdown:")
per_cond_results = {}
for cond in range(6):
    mask = test_conds == cond
    n = mask.sum()
    if n == 0:
        print(f"  Cond {cond}: 0 engines")
        continue
    rmse_cond = float(np.sqrt(np.mean((all_preds[mask] - test_rul[mask])**2)))
    mean_pred = float(all_preds[mask].mean())
    mean_true = float(test_rul[mask].mean())
    print(f"  Cond {cond}: n={n}, RMSE={rmse_cond:.2f}, mean_pred={mean_pred:.1f}, mean_true={mean_true:.1f}")
    per_cond_results[str(cond)] = {
        "n_engines": int(n),
        "rmse": rmse_cond,
        "mean_pred": mean_pred,
        "mean_true": mean_true,
    }

# Train condition distribution (last window per training engine)
train_engine_ids_fd002 = sorted(train_engines.keys())
train_conds = []
for eid in train_engine_ids_fd002:
    eng = train_df_002[train_df_002['engine_id'] == eid].sort_values('cycle')
    last_row = eng.tail(1)[op_cols].values
    cond = int(kmeans.predict(last_row)[0])
    train_conds.append(cond)
train_cond_counts = np.bincount(train_conds, minlength=6)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('FD002: Per-Condition RMSE Breakdown (Frozen Probe, Seed=42)', fontsize=11)

x = np.arange(6)
cond_rmses = [per_cond_results.get(str(c), {}).get('rmse', 0) for c in range(6)]
cond_ns = [per_cond_results.get(str(c), {}).get('n_engines', 0) for c in range(6)]

bar_colors = ['red' if r > 30 else 'darkorange' if r > 25 else 'steelblue' for r in cond_rmses]
axes[0].bar(x, cond_rmses, color=bar_colors, alpha=0.8, edgecolor='black')
axes[0].axhline(test_rmse_total, color='black', linestyle='--', label=f'Overall={test_rmse_total:.1f}')
for xi, (r, n) in enumerate(zip(cond_rmses, cond_ns)):
    if n > 0:
        axes[0].text(xi, r + 0.3, f'n={n}\n{r:.1f}', ha='center', va='bottom', fontsize=8)
axes[0].set_xlabel('Operating Condition (cluster)')
axes[0].set_ylabel('RMSE (cycles)')
axes[0].set_title('Per-condition RMSE on test set')
axes[0].legend()
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'Cond {i}' for i in range(6)])

# Train vs test distribution
test_cond_counts = np.bincount(test_conds, minlength=6)
width = 0.35
axes[1].bar(x - width/2, train_cond_counts / train_cond_counts.sum(), width,
            label='Train last-window', color='steelblue', alpha=0.8)
axes[1].bar(x + width/2, test_cond_counts / max(1, test_cond_counts.sum()), width,
            label='Test last-window', color='darkorange', alpha=0.8)
axes[1].set_xlabel('Operating Condition')
axes[1].set_ylabel('Fraction')
axes[1].set_title('Train vs Test condition distribution')
axes[1].legend()
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'C{i}' for i in range(6)])

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'fd002_per_condition_rmse.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved fd002_per_condition_rmse.png")

results_out = {
    "total_test_rmse": test_rmse_total,
    "per_condition": per_cond_results,
    "test_condition_counts": test_cond_counts.tolist(),
    "train_condition_counts": train_cond_counts.tolist(),
}
with open(V12_DIR / 'fd002_per_condition_breakdown.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print(f"Saved fd002_per_condition_breakdown.json")

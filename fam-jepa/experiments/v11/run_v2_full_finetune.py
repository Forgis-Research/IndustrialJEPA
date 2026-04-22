"""
Run full label efficiency experiments for V2 (d_model=256) model.
V2 frozen @ 100% = 17.82 (much better than V1 = 21.33)
Run all 5 budgets x 5 seeds.
"""

import os
import sys
import time
import json
import copy
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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
from train_utils import subsample_engines, train_supervised_lstm

from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256

def log(msg):
    print(msg, flush=True)
    with open(os.path.join(EXP_DIR, 'EXPERIMENT_LOG.md'), 'a') as f:
        f.write(msg + '\n')

def save_json(obj, path):
    def default(x):
        if isinstance(x, (np.floating, float)): return float(x)
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return str(x)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=default)

data = load_cmapss_subset('FD001')
ckpt_path_v2 = os.path.join(EXP_DIR, 'best_pretrain_L1_v2.pt')

if not os.path.exists(ckpt_path_v2):
    log("ERROR: V2 checkpoint not found!")
    sys.exit(1)

log("\n" + "="*60)
log("V2 Full Label Efficiency (d_model=256, all 5 budgets)")
log("="*60)

budgets = [1.0, 0.5, 0.2, 0.1, 0.05]
N_SEEDS = 5


def run_finetune_v2(train_eng, val_eng, test_eng, test_rul, mode, seed):
    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
        n_heads=4, n_layers=2, d_ff=512
    )
    model_ft.load_state_dict(torch.load(ckpt_path_v2, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = RULProbe(D_MODEL).to(DEVICE)

    torch.manual_seed(seed); np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model_ft.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model_ft.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(
            list(model_ft.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4
        )

    best_val = float('inf'); best_ps = None; best_es = None; no_impr = 0
    for ep in range(100):
        if mode == 'frozen': model_ft.eval()
        else: model_ft.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model_ft.encode_past(past, mask)
            else: h = model_ft.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optim.step()

        model_ft.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model_ft.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2
        )))
        if val_rmse < best_val:
            best_val = val_rmse
            best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e': best_es = copy.deepcopy(model_ft.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20: break

    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es: model_ft.context_encoder.load_state_dict(best_es)

    model_ft.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model_ft.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))


# Load V1 results for comparison
with open(os.path.join(EXP_DIR, 'finetune_results.json')) as f:
    results_v1 = json.load(f)
    results_v1 = {method: {float(k): v for k, v in d.items()} for method, d in results_v1.items()}

results_v2 = {'jepa_frozen': {}, 'jepa_e2e': {}}

for budget in budgets:
    log(f"\n--- V2 Budget: {budget*100:.0f}% ---")
    sub_eng = subsample_engines(data['train_engines'], budget, seed=42)

    for mode, key in [('frozen', 'jepa_frozen'), ('e2e', 'jepa_e2e')]:
        rmses = []
        for seed in range(N_SEEDS):
            rmse = run_finetune_v2(
                sub_eng, data['val_engines'], data['test_engines'], data['test_rul'],
                mode=mode, seed=seed
            )
            rmses.append(rmse)
            log(f"  V2 {mode} seed={seed}: {rmse:.2f}")
        results_v2[key][budget] = {
            'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses
        }
        log(f"  V2 {key} @ {budget*100:.0f}%: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

    save_json(results_v2, os.path.join(EXP_DIR, 'finetune_results_v2_full.json'))

log("\n--- V2 Final Table ---")
log("Method | 100% | 50% | 20% | 10% | 5%")
for key, name in [('jepa_frozen', 'V2 frozen'), ('jepa_e2e', 'V2 E2E')]:
    row = f"{name} |"
    for b in budgets:
        d = results_v2[key][b]
        row += f" {d['mean']:.2f}+-{d['std']:.2f} |"
    log(row)

log("\n--- Comparison V1 vs V2 ---")
for b in budgets:
    v1_e2e = results_v1['jepa_e2e'][b]['mean']
    v2_e2e = results_v2['jepa_e2e'][b]['mean']
    v1_frz = results_v1['jepa_frozen'][b]['mean']
    v2_frz = results_v2['jepa_frozen'][b]['mean']
    log(f"  {b*100:.0f}%: E2E={v1_e2e:.2f}->{v2_e2e:.2f} "
        f"({'+' if v2_e2e<v1_e2e else '-'}{abs(v2_e2e-v1_e2e):.2f}), "
        f"Frozen={v1_frz:.2f}->{v2_frz:.2f} "
        f"({'+' if v2_frz<v1_frz else '-'}{abs(v2_frz-v1_frz):.2f})")

# Combined plot
STAR_RMSE = 10.61
AE_LSTM_RMSE = 13.99
budget_pcts = [b*100 for b in budgets]
budget_labels = ['100%', '50%', '20%', '10%', '5%']

lstm_means = [results_v1['supervised_lstm'][b]['mean'] for b in budgets]
v1_e2e_means = [results_v1['jepa_e2e'][b]['mean'] for b in budgets]
v2_frozen_means = [results_v2['jepa_frozen'][b]['mean'] for b in budgets]
v2_e2e_means = [results_v2['jepa_e2e'][b]['mean'] for b in budgets]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(budget_pcts, lstm_means, 'o-', color='steelblue', linewidth=2,
        markersize=8, label='Supervised LSTM')
ax.plot(budget_pcts, v1_e2e_means, 's--', color='darkorange', linewidth=2,
        markersize=8, label='Traj JEPA E2E (d=128)', alpha=0.7)
ax.plot(budget_pcts, v2_frozen_means, '^-', color='green', linewidth=2,
        markersize=8, label='Traj JEPA frozen (d=256)')
ax.plot(budget_pcts, v2_e2e_means, 'D-', color='red', linewidth=2,
        markersize=8, label='Traj JEPA E2E (d=256)')
ax.axhline(STAR_RMSE, color='black', linestyle='--', linewidth=2,
           label=f'STAR 2024 supervised SOTA ({STAR_RMSE})')
ax.axhline(AE_LSTM_RMSE, color='purple', linestyle=':', linewidth=2,
           label=f'AE-LSTM SSL reference ({AE_LSTM_RMSE})')
ax.set_xscale('log')
ax.set_xlabel('Label Fraction (%)', fontsize=12)
ax.set_ylabel('Test RMSE (cycles)', fontsize=12)
ax.set_title('Label Efficiency: C-MAPSS FD001 (V1 vs V2 models)', fontsize=13)
ax.set_xticks(budget_pcts)
ax.set_xticklabels(budget_labels)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'label_efficiency_v2.png'), dpi=120)
plt.close()
log("\nSaved label_efficiency_v2.png")

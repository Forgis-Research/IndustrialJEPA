"""
Exp 9: Zero-shot and few-shot transfer evaluation.

Tests whether FD001-pretrained encoder can transfer to FD002 with only a new linear probe
(no encoder fine-tuning), and how many FD002 labels are needed to approach in-domain performance.

Also tests: FD001-pretrained -> FD003 (same conditions, different fault mode)
This is the most direct cross-domain test: same operational context, different degradation mode.
"""
import os, sys, time, json, copy, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
from train_utils import subsample_engines
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LOG_FILE = os.path.join(EXP_DIR, 'EXPERIMENT_LOG.md')
def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def save_json(obj, path):
    def default(x):
        if isinstance(x, (np.floating, float)): return float(x)
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return str(x)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=default)

def run_frozen_probe(ckpt_path, d_model, n_layers, n_heads, d_ff,
                     train_eng, val_eng, test_eng, test_rul, seed, n_epochs=100):
    """Frozen encoder + linear probe only."""
    model = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=d_model,
                            n_heads=n_heads, n_layers=n_layers, d_ff=d_ff)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model = model.to(DEVICE)
    for p in model.parameters(): p.requires_grad = False

    probe = RULProbe(d_model).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    best_v, best_ps, ni = float('inf'), None, 0
    for ep in range(n_epochs):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            with torch.no_grad(): h = model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward(); optim.step()
        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_r < best_v:
            best_v = val_r; best_ps = copy.deepcopy(probe.state_dict()); ni = 0
        else:
            ni += 1
            if ni >= 20: break

    probe.load_state_dict(best_ps); probe.eval(); model.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rg in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pt.append(probe(model.encode_past(past, mask)).cpu().numpy() * RUL_CAP)
            tt.append(rg.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

# Main experiment
log("\n" + "="*60)
log("EXP 9: Zero-shot/Few-shot Cross-subset Transfer")
log("="*60)

ckpt_v2 = os.path.join(EXP_DIR, 'best_pretrain_L1_v2.pt')  # FD001 pretrained
ckpt_fd002 = os.path.join(EXP_DIR, 'best_pretrain_fd002.pt')  # FD002 pretrained

# Load all subsets
log("Loading all C-MAPSS subsets...")
data = {}
for subset in ['FD001', 'FD002', 'FD003']:
    data[subset] = load_cmapss_subset(subset)
    log(f"  {subset}: {len(data[subset]['train_engines'])} train, "
        f"{len(data[subset]['val_engines'])} val, {len(data[subset]['test_engines'])} test")

results = {}

# Exp 9.1: FD001-pretrained -> FD003 (same conditions, different fault mode)
log("\n--- Exp 9.1: FD001->FD003 cross-fault transfer ---")
budgets = [1.0, 0.5, 0.2, 0.1, 0.05]
fd001_to_fd003 = {}
for budget in budgets:
    sub_eng = subsample_engines(data['FD003']['train_engines'], budget, seed=42)
    rmses = []
    for seed in range(5):
        rmse = run_frozen_probe(ckpt_v2, 256, 2, 4, 512,
                                sub_eng, data['FD003']['val_engines'],
                                data['FD003']['test_engines'], data['FD003']['test_rul'], seed)
        rmses.append(rmse)
        log(f"  FD001->FD003 @ {budget:.0%} seed={seed}: {rmse:.2f}")
    fd001_to_fd003[str(budget)] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
    log(f"  FD001->FD003 @ {budget:.0%}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

results['fd001_to_fd003'] = fd001_to_fd003

# Compare with FD001 in-domain at same budgets
import json
with open(os.path.join(EXP_DIR, 'finetune_results_v2_full.json')) as f:
    r_v2 = json.load(f)

log("\nComparison: FD001 in-domain vs FD001->FD003 cross-fault (frozen probe)")
log("Budget | FD001 in-domain | FD001->FD003 | Transfer benefit")
for budget in budgets:
    bk = str(budget)
    fd001_val = r_v2.get('jepa_frozen', {}).get(bk, {}).get('mean', float('nan'))
    fd003_val = fd001_to_fd003[bk]['mean']
    benefit = fd001_val - fd003_val  # positive = transfer helps
    log(f"  {budget:.0%}: FD001={fd001_val:.2f}, FD003={fd003_val:.2f}, benefit={benefit:+.2f}")

# Exp 9.2: FD002-pretrained -> FD003 (different conditions AND fault mode)
log("\n--- Exp 9.2: FD002->FD003 cross-both transfer ---")
fd002_to_fd003 = {}
for budget in [1.0, 0.2, 0.1]:
    sub_eng = subsample_engines(data['FD003']['train_engines'], budget, seed=42)
    rmses = []
    for seed in range(5):
        rmse = run_frozen_probe(ckpt_fd002, 256, 2, 4, 512,
                                sub_eng, data['FD003']['val_engines'],
                                data['FD003']['test_engines'], data['FD003']['test_rul'], seed)
        rmses.append(rmse)
        log(f"  FD002->FD003 @ {budget:.0%} seed={seed}: {rmse:.2f}")
    fd002_to_fd003[str(budget)] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
    log(f"  FD002->FD003 @ {budget:.0%}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

results['fd002_to_fd003'] = fd002_to_fd003

# FD001 in-domain (for FD001 test set reference)
log("\n--- FD001 in-domain reference (frozen, seeds 0-4) ---")
fd001_in_domain = {}
for budget in [1.0, 0.2, 0.1]:
    sub_eng = subsample_engines(data['FD001']['train_engines'], budget, seed=42)
    rmses = []
    for seed in range(5):
        rmse = run_frozen_probe(ckpt_v2, 256, 2, 4, 512,
                                sub_eng, data['FD001']['val_engines'],
                                data['FD001']['test_engines'], data['FD001']['test_rul'], seed)
        rmses.append(rmse)
    fd001_in_domain[str(budget)] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses))}
    log(f"  FD001 in-domain @ {budget:.0%}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

results['fd001_in_domain_rerun'] = fd001_in_domain
save_json(results, os.path.join(EXP_DIR, 'exp9_zero_shot_results.json'))

# Summary
log("\n## Exp 9: Zero-shot/Few-shot Cross-subset Transfer Summary")
log("| Source Pretrain | Fine-tune Target | Budget | Frozen Probe RMSE |")
log("|:---------------:|:---------------:|:------:|:-----------------:|")
for budget in budgets:
    bk = str(budget)
    log(f"| FD001 | FD001 (in-domain) | {budget:.0%} | {r_v2['jepa_frozen'].get(bk, {}).get('mean', float('nan')):.2f} |")
for budget in budgets:
    bk = str(budget)
    log(f"| FD001 | FD003 (cross-fault) | {budget:.0%} | {fd001_to_fd003[bk]['mean']:.2f} |")
for budget in [1.0, 0.2, 0.1]:
    bk = str(budget)
    log(f"| FD002 | FD003 (cross-both) | {budget:.0%} | {fd002_to_fd003[bk]['mean']:.2f} |")
log("")
log("STAR supervised SOTA: FD001=10.61, FD003=10.71")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
x = range(len(budgets))
bks = [str(b) for b in budgets]

fd001_in = [r_v2['jepa_frozen'].get(bk, {}).get('mean', float('nan')) for bk in bks]
fd003_vals = [fd001_to_fd003[bk]['mean'] for bk in bks]

ax.plot(x, fd001_in, 'b-s', label='FD001 in-domain (pretrain+finetune FD001)', linewidth=2)
ax.plot(x, fd003_vals, 'r-^', label='FD001->FD003 cross-fault (pretrain FD001, finetune FD003)', linewidth=2)

ax.axhline(10.71, color='black', linestyle='--', label='STAR FD003 SOTA (10.71)')
ax.set_xticks(x); ax.set_xticklabels([f'{b:.0%}' for b in budgets])
ax.set_xlabel('Label Budget'); ax.set_ylabel('RMSE (cycles)')
ax.set_title('Cross-fault Transfer: FD001 Pretrain -> FD003 Fine-tune')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'cross_fault_transfer.png'), dpi=150, bbox_inches='tight')
plt.close()
log("Saved cross_fault_transfer.png")

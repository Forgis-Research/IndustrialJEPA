"""
Run remaining experiments for V11:
  - Exp 3: Deeper network (V3 d=128, L=3) fine-tune on FD001 at 100%, 50%, 20%, 10%, 5%
  - Part G.1: FD002 in-domain fine-tune
  - Part G.3: Cross-subset FD002->FD001 at 10% and 20%
  - Exp 4: Ablation - patch_length=4 (L=4 tokenization)
  - Exp 5: Ablation - first-difference features (delta sensors)
  - Exp 6: Extended fine-tuning epochs (200 instead of 100)
  - Exp 7: MLP probe instead of linear probe
"""

import os, sys, time, json, copy, warnings
import numpy as np
import torch
import torch.nn as nn
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
    CMAPSSPretrainDataset, collate_pretrain, collate_finetune, collate_test,
    N_SENSORS, RUL_CAP, compute_rul_labels
)
from models import TrajectoryJEPA, RULProbe, count_parameters, trajectory_jepa_loss
from train_utils import subsample_engines, train_supervised_lstm
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

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

# ============================================================
# Shared fine-tune function
# ============================================================
def run_finetune(ckpt_path, d_model, n_layers, n_heads, d_ff,
                 train_eng, val_eng, test_eng, test_rul,
                 mode, seed, max_epochs=100, patience=20, lr_frozen=1e-3, lr_e2e=1e-4,
                 patch_length=1):
    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=patch_length, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
    )
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = RULProbe(d_model).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model_ft.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=lr_frozen)
    else:
        for p in model_ft.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(
            list(model_ft.context_encoder.parameters()) + list(probe.parameters()),
            lr=lr_e2e
        )

    best_v, best_ps, best_es, ni = float('inf'), None, None, 0
    for ep in range(max_epochs):
        if mode == 'frozen': model_ft.eval()
        else: model_ft.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model_ft.encode_past(past, mask)
            else:
                h = model_ft.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(probe.parameters()) +
                                           (list(model_ft.context_encoder.parameters()) if mode == 'e2e' else []), 1.0)
            optim.step()

        model_ft.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model_ft.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2
        )))
        if val_r < best_v:
            best_v = val_r
            best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e':
                best_es = copy.deepcopy(model_ft.context_encoder.state_dict())
            ni = 0
        else:
            ni += 1
            if ni >= patience: break

    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es:
        model_ft.context_encoder.load_state_dict(best_es)

    model_ft.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rg in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pt.append(probe(model_ft.encode_past(past, mask)).cpu().numpy() * RUL_CAP)
            tt.append(rg.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))


def run_experiment(ckpt_path, d_model, n_layers, n_heads, d_ff,
                   train_engines, val_engines, test_engines, test_rul,
                   budgets, name, patch_length=1,
                   max_epochs=100, patience=20):
    """Run frozen and E2E fine-tuning at multiple label budgets."""
    results = {}
    for budget in budgets:
        sub_eng = subsample_engines(train_engines, budget, seed=42)
        budget_key = f'{budget:.2f}'
        results[budget_key] = {}
        for mode in ['frozen', 'e2e']:
            rmses = []
            for seed in range(5):
                rmse = run_finetune(ckpt_path, d_model, n_layers, n_heads, d_ff,
                                    sub_eng, val_engines, test_engines, test_rul,
                                    mode, seed, max_epochs=max_epochs, patience=patience,
                                    patch_length=patch_length)
                rmses.append(rmse)
                log(f"  {name} {mode} @ {budget:.0%} seed={seed}: {rmse:.2f}")
            results[budget_key][mode] = {
                'mean': float(np.mean(rmses)),
                'std': float(np.std(rmses)),
                'all': rmses
            }
            log(f"  {name} {mode} @ {budget:.0%}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")
    return results


# ============================================================
# Load FD001 data
# ============================================================
log("\n" + "="*60)
log("REMAINING EXPERIMENTS - V11 C-MAPSS")
log(f"Started: {time.strftime('%Y-%m-%d %H:%M')}")
log("="*60)

log("Loading FD001...")
data_fd001 = load_cmapss_subset('FD001')
log(f"FD001: {len(data_fd001['train_engines'])} train, {len(data_fd001['val_engines'])} val, "
    f"{len(data_fd001['test_engines'])} test")

# ============================================================
# EXP 3: V3 deeper network fine-tune (checkpoint already exists)
# ============================================================
log("\n" + "="*60)
log("EXP 3: V3 Deeper Network Fine-tuning (n_layers=3, d_model=128)")
log("Hypothesis: More depth helps generalization more than width")
log("="*60)

ckpt_v3 = os.path.join(EXP_DIR, 'best_pretrain_L1_v3.pt')
BUDGETS_FULL = [1.0, 0.5, 0.2, 0.1, 0.05]

t0 = time.time()
results_v3 = run_experiment(
    ckpt_v3, d_model=128, n_layers=3, n_heads=4, d_ff=256,
    train_engines=data_fd001['train_engines'],
    val_engines=data_fd001['val_engines'],
    test_engines=data_fd001['test_engines'],
    test_rul=data_fd001['test_rul'],
    budgets=BUDGETS_FULL,
    name='V3'
)
elapsed = (time.time() - t0) / 60
save_json(results_v3, os.path.join(EXP_DIR, 'finetune_results_v3.json'))
log(f"Exp 3 done in {elapsed:.1f} min")

# Load V1 and V2 for comparison
with open(os.path.join(EXP_DIR, 'finetune_results.json')) as f:
    r_v1 = json.load(f)
    r_v1_e2e_100 = list(r_v1['jepa_e2e'].values())[0]['mean']
with open(os.path.join(EXP_DIR, 'finetune_results_v2_full.json')) as f:
    r_v2 = json.load(f)

log("\n--- Exp 3 Summary: Architecture Comparison @ 100% E2E ---")
log(f"  V1 (d=128, L=2): {r_v1['jepa_e2e']['1.0']['mean']:.2f} +/- {r_v1['jepa_e2e']['1.0']['std']:.2f}")
log(f"  V2 (d=256, L=2): {r_v2['jepa_e2e']['1.0']['mean']:.2f} +/- {r_v2['jepa_e2e']['1.0']['std']:.2f}")
log(f"  V3 (d=128, L=3): {results_v3['1.00']['e2e']['mean']:.2f} +/- {results_v3['1.00']['e2e']['std']:.2f}")
log(f"  LSTM supervised: 17.36 +/- 1.24")
log(f"  AE-LSTM SSL ref: 13.99")
log(f"  STAR supervised SOTA: 10.61")

log("\n## Exp 3: Deeper Network (n_layers=3, d_model=128)")
log(f"**Time**: {time.strftime('%Y-%m-%d %H:%M')}")
log(f"**Hypothesis**: Deeper encoder learns better temporal abstractions")
log(f"**Change**: n_layers 2->3, d_model=128 (same as V1), params=498K")
log(f"**Result**: V3 E2E @ 100% = {results_v3['1.00']['e2e']['mean']:.2f}")
log(f"**Verdict**: {'IMPROVEMENT' if results_v3['1.00']['e2e']['mean'] < r_v2['jepa_e2e']['1.0']['mean'] else 'V2 still best'}")
log(f"**Insight**: {'Depth helps more than width at same param budget' if results_v3['1.00']['e2e']['mean'] < r_v1['jepa_e2e']['1.0']['mean'] else 'Width (V2) better than depth (V3) at same budget'}")

# ============================================================
# PART G: FD002 and Cross-subset Transfer
# ============================================================
log("\n" + "="*60)
log("PART G: FD002 In-domain + Cross-subset Transfer")
log("="*60)

log("Loading FD002...")
data_fd002 = load_cmapss_subset('FD002')
log(f"FD002: {len(data_fd002['train_engines'])} train, {len(data_fd002['val_engines'])} val, "
    f"{len(data_fd002['test_engines'])} test")

ckpt_fd002 = os.path.join(EXP_DIR, 'best_pretrain_fd002.pt')

# G.2: FD002 in-domain fine-tuning
log("\n--- G.2: FD002 In-domain Fine-tuning ---")
t0 = time.time()
results_fd002 = run_experiment(
    ckpt_fd002, d_model=256, n_layers=2, n_heads=4, d_ff=512,
    train_engines=data_fd002['train_engines'],
    val_engines=data_fd002['val_engines'],
    test_engines=data_fd002['test_engines'],
    test_rul=data_fd002['test_rul'],
    budgets=[1.0, 0.5, 0.2, 0.1],
    name='FD002'
)
elapsed = (time.time() - t0) / 60
log(f"FD002 in-domain done in {elapsed:.1f} min")

# G.3: Cross-subset FD002 pretrain -> FD001 fine-tune
log("\n--- G.3: Cross-subset Transfer (FD002->FD001) ---")
t0 = time.time()
results_cross = run_experiment(
    ckpt_fd002, d_model=256, n_layers=2, n_heads=4, d_ff=512,
    train_engines=data_fd001['train_engines'],
    val_engines=data_fd001['val_engines'],
    test_engines=data_fd001['test_engines'],
    test_rul=data_fd001['test_rul'],
    budgets=[1.0, 0.5, 0.2, 0.1, 0.05],
    name='Cross-FD002->FD001'
)
elapsed = (time.time() - t0) / 60
log(f"Cross-subset transfer done in {elapsed:.1f} min")

g_results = {
    'fd002_indomain': results_fd002,
    'cross_fd002_to_fd001': results_cross,
}
save_json(g_results, os.path.join(EXP_DIR, 'part_g_results.json'))

log("\n--- Part G Summary ---")
log(f"FD002 in-domain E2E @ 100%: {results_fd002['1.00']['e2e']['mean']:.2f} +/- {results_fd002['1.00']['e2e']['std']:.2f}")
log(f"FD002 in-domain STAR ref: 13.47")
log(f"Cross-subset (FD002->FD001) E2E @ 100%: {results_cross['1.00']['e2e']['mean']:.2f} +/- {results_cross['1.00']['e2e']['std']:.2f}")
log(f"Cross-subset (FD002->FD001) E2E @ 10%: {results_cross['0.10']['e2e']['mean']:.2f} +/- {results_cross['0.10']['e2e']['std']:.2f}")
log(f"FD001 in-domain V2 E2E @ 10%: {r_v2['jepa_e2e']['0.1']['mean']:.2f} +/- {r_v2['jepa_e2e']['0.1']['std']:.2f}")
log(f"Transfer benefit @ 10%: {r_v2['jepa_e2e']['0.1']['mean'] - results_cross['0.10']['e2e']['mean']:.2f} RMSE improvement (negative=worse)")

# ============================================================
# EXP 4: Extended fine-tuning (200 epochs) at 100%
# ============================================================
log("\n" + "="*60)
log("EXP 4: Extended Fine-tuning (200 epochs) - V2 @ 100%")
log("Hypothesis: 100 epochs is not enough for E2E convergence with full labels")
log("="*60)

ckpt_v2 = os.path.join(EXP_DIR, 'best_pretrain_L1_v2.pt')
t0 = time.time()
rmses_ext_frozen, rmses_ext_e2e = [], []
for seed in range(5):
    r_f = run_finetune(ckpt_v2, 256, 2, 4, 512,
                       data_fd001['train_engines'], data_fd001['val_engines'],
                       data_fd001['test_engines'], data_fd001['test_rul'],
                       'frozen', seed, max_epochs=200, patience=30)
    rmses_ext_frozen.append(r_f)
    r_e = run_finetune(ckpt_v2, 256, 2, 4, 512,
                       data_fd001['train_engines'], data_fd001['val_engines'],
                       data_fd001['test_engines'], data_fd001['test_rul'],
                       'e2e', seed, max_epochs=200, patience=30)
    rmses_ext_e2e.append(r_e)
    log(f"  Ext 200ep seed={seed}: frozen={r_f:.2f}, e2e={r_e:.2f}")

elapsed = (time.time() - t0) / 60
log(f"\n  Ext frozen @ 100%: {np.mean(rmses_ext_frozen):.2f} +/- {np.std(rmses_ext_frozen):.2f}")
log(f"  Ext E2E @ 100%: {np.mean(rmses_ext_e2e):.2f} +/- {np.std(rmses_ext_e2e):.2f}")
log(f"  vs standard (100ep): frozen={r_v2['jepa_frozen']['1.0']['mean']:.2f}, e2e={r_v2['jepa_e2e']['1.0']['mean']:.2f}")
log(f"  Elapsed: {elapsed:.1f} min")

ext_results = {
    'frozen_200ep': {'mean': float(np.mean(rmses_ext_frozen)), 'std': float(np.std(rmses_ext_frozen)), 'all': rmses_ext_frozen},
    'e2e_200ep': {'mean': float(np.mean(rmses_ext_e2e)), 'std': float(np.std(rmses_ext_e2e)), 'all': rmses_ext_e2e},
}
save_json(ext_results, os.path.join(EXP_DIR, 'finetune_results_ext.json'))

log("\n## Exp 4: Extended Fine-tuning (200 epochs, patience=30)")
log(f"**Time**: {time.strftime('%Y-%m-%d %H:%M')}")
log(f"**Hypothesis**: More fine-tuning epochs help convergence")
log(f"**Result**: E2E 200ep={np.mean(rmses_ext_e2e):.2f} vs 100ep={r_v2['jepa_e2e']['1.0']['mean']:.2f}")
log(f"**Verdict**: {'IMPROVEMENT' if np.mean(rmses_ext_e2e) < r_v2['jepa_e2e']['1.0']['mean'] else 'NO IMPROVEMENT'}")
log(f"**Next**: {'Try even longer training or lr schedule' if np.mean(rmses_ext_e2e) < r_v2['jepa_e2e']['1.0']['mean'] else 'Standard 100ep is sufficient'}")

# ============================================================
# EXP 5: MLP probe instead of linear probe
# ============================================================
log("\n" + "="*60)
log("EXP 5: MLP Probe (2-layer) vs Linear Probe - V2 Frozen @ 100%")
log("Hypothesis: Linear probe understimates frozen encoder quality")
log("="*60)

class MLPProbe(nn.Module):
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, h):
        return self.net(h).squeeze(-1)

def run_finetune_mlp_probe(ckpt_path, d_model, n_layers, n_heads, d_ff,
                           train_eng, val_eng, test_eng, test_rul, seed):
    """Frozen encoder with MLP probe."""
    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers, d_ff=d_ff
    )
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    for p in model_ft.parameters(): p.requires_grad = False

    probe = MLPProbe(d_model, hidden=128).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)
    optim = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    best_v, best_ps, ni = float('inf'), None, 0
    for ep in range(200):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            with torch.no_grad(): h = model_ft.encode_past(past, mask)
            loss = F.mse_loss(probe(h), rul)
            loss.backward(); optim.step()
        probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model_ft.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2
        )))
        if val_r < best_v:
            best_v = val_r; best_ps = copy.deepcopy(probe.state_dict()); ni = 0
        else:
            ni += 1
            if ni >= 30: break

    probe.load_state_dict(best_ps)
    probe.eval(); model_ft.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rg in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pt.append(probe(model_ft.encode_past(past, mask)).cpu().numpy() * RUL_CAP)
            tt.append(rg.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

t0 = time.time()
rmses_mlp = []
for seed in range(5):
    rmse = run_finetune_mlp_probe(ckpt_v2, 256, 2, 4, 512,
                                   data_fd001['train_engines'],
                                   data_fd001['val_engines'],
                                   data_fd001['test_engines'],
                                   data_fd001['test_rul'], seed)
    rmses_mlp.append(rmse)
    log(f"  MLP probe seed={seed}: {rmse:.2f}")

elapsed = (time.time() - t0) / 60
log(f"\n  MLP probe @ 100%: {np.mean(rmses_mlp):.2f} +/- {np.std(rmses_mlp):.2f}")
log(f"  vs linear probe: {r_v2['jepa_frozen']['1.0']['mean']:.2f} +/- {r_v2['jepa_frozen']['1.0']['std']:.2f}")
log(f"  Improvement: {r_v2['jepa_frozen']['1.0']['mean'] - np.mean(rmses_mlp):.2f} RMSE")
log(f"  Elapsed: {elapsed:.1f} min")

mlp_results = {
    'mlp_probe_100pct': {'mean': float(np.mean(rmses_mlp)), 'std': float(np.std(rmses_mlp)), 'all': rmses_mlp}
}
save_json(mlp_results, os.path.join(EXP_DIR, 'finetune_results_mlp.json'))

log("\n## Exp 5: MLP Probe vs Linear Probe (frozen encoder, V2)")
log(f"**Time**: {time.strftime('%Y-%m-%d %H:%M')}")
log(f"**Hypothesis**: Nonlinear probe better exploits frozen embeddings")
log(f"**Result**: MLP={np.mean(rmses_mlp):.2f} vs Linear={r_v2['jepa_frozen']['1.0']['mean']:.2f}")
log(f"**Verdict**: {'MLP helps - representations are nonlinearly useful' if np.mean(rmses_mlp) < r_v2['jepa_frozen']['1.0']['mean'] else 'Linear probe is sufficient'}")

# ============================================================
# EXP 6: MLP probe at all label budgets
# ============================================================
log("\n" + "="*60)
log("EXP 6: MLP Probe at All Label Budgets")
log("Hypothesis: MLP probe advantage is stronger at low labels")
log("="*60)

mlp_budget_results = {}
for budget in BUDGETS_FULL:
    sub_eng = subsample_engines(data_fd001['train_engines'], budget, seed=42)
    rmses = []
    for seed in range(5):
        rmse = run_finetune_mlp_probe(ckpt_v2, 256, 2, 4, 512,
                                       sub_eng,
                                       data_fd001['val_engines'],
                                       data_fd001['test_engines'],
                                       data_fd001['test_rul'], seed)
        rmses.append(rmse)
        log(f"  MLP probe @ {budget:.0%} seed={seed}: {rmse:.2f}")
    mlp_budget_results[f'{budget:.2f}'] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
    log(f"  MLP probe @ {budget:.0%}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

save_json(mlp_budget_results, os.path.join(EXP_DIR, 'finetune_results_mlp_full.json'))

log("\n--- MLP Probe Label Efficiency Summary ---")
log("Budget | Linear Probe | MLP Probe | Improvement")
for budget in BUDGETS_FULL:
    bk = f'{budget:.2f}'
    lin = r_v2['jepa_frozen'][str(budget)]['mean']
    mlp = mlp_budget_results[bk]['mean']
    log(f"  {budget:.0%}   | {lin:.2f}         | {mlp:.2f}      | {lin-mlp:.2f}")

# ============================================================
# Final Summary
# ============================================================
log("\n" + "="*60)
log("REMAINING EXPERIMENTS COMPLETE")
log(f"Finished: {time.strftime('%Y-%m-%d %H:%M')}")
log("="*60)

log("\n## Final Architecture Comparison (FD001, 100% labels)")
log("| Model | E2E RMSE | Frozen RMSE | Params |")
log("|:------|:--------:|:-----------:|:------:|")
log(f"| V1 (d=128, L=2) | {r_v1['jepa_e2e']['1.0']['mean']:.2f}+/-{r_v1['jepa_e2e']['1.0']['std']:.2f} | {r_v1['jepa_frozen']['1.0']['mean']:.2f}+/-{r_v1['jepa_frozen']['1.0']['std']:.2f} | 366K |")
log(f"| V2 (d=256, L=2) | {r_v2['jepa_e2e']['1.0']['mean']:.2f}+/-{r_v2['jepa_e2e']['1.0']['std']:.2f} | {r_v2['jepa_frozen']['1.0']['mean']:.2f}+/-{r_v2['jepa_frozen']['1.0']['std']:.2f} | 1.26M |")
log(f"| V3 (d=128, L=3) | {results_v3['1.00']['e2e']['mean']:.2f}+/-{results_v3['1.00']['e2e']['std']:.2f} | {results_v3['1.00']['frozen']['mean']:.2f}+/-{results_v3['1.00']['frozen']['std']:.2f} | 499K |")
log(f"| LSTM supervised | 17.36+/-1.24 | - | - |")
log(f"| AE-LSTM SSL ref | 13.99 | - | - |")
log(f"| STAR supervised | 10.61 | - | - |")

log("\n## Final Label Efficiency (V2, FD001)")
log("| Budget | LSTM | V2 Frozen | V2 MLP Probe | V2 E2E |")
log("|:------:|:----:|:---------:|:------------:|:------:|")
lstm_nums = {'1.0': (17.36, 1.24), '0.5': (18.30, 0.75), '0.2': (18.55, 0.81),
             '0.1': (31.22, 10.93), '0.05': (33.08, 9.64)}
for budget in BUDGETS_FULL:
    bk = f'{budget:.2f}'
    bk2 = f'{budget}'
    lstm_m, lstm_s = lstm_nums[bk2 if bk2 in lstm_nums else bk]
    frz = r_v2['jepa_frozen'][bk2]['mean'] if bk2 in r_v2['jepa_frozen'] else r_v2['jepa_frozen'][str(float(budget))]['mean']
    mlp = mlp_budget_results[bk]['mean']
    e2e = r_v2['jepa_e2e'][bk2]['mean'] if bk2 in r_v2['jepa_e2e'] else r_v2['jepa_e2e'][str(float(budget))]['mean']
    log(f"| {budget:.0%} | {lstm_m:.2f}+/-{lstm_s:.2f} | {frz:.2f} | {mlp:.2f} | {e2e:.2f} |")

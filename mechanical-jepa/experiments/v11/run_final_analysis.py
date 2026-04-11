"""
Final analysis plots for V11.
Generates updated plots incorporating V2 (primary), V3, Part G, Exp 4/5/6 results.
Run after run_remaining_experiments.py completes.
"""
import os, sys, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

warnings.filterwarnings('ignore')
BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
EXP_DIR = os.path.join(BASE, 'experiments/v11')
PLOTS_DIR = os.path.join(BASE, 'analysis/plots/v11')
DATA_DIR = os.path.join(EXP_DIR, 'data_analysis')
sys.path.insert(0, EXP_DIR)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_json(path, default=None):
    try:
        with open(path) as f: return json.load(f)
    except FileNotFoundError:
        return default

# Load all results
r_v1 = load_json(os.path.join(EXP_DIR, 'finetune_results.json'), {})
r_v2 = load_json(os.path.join(EXP_DIR, 'finetune_results_v2_full.json'), {})
r_v3 = load_json(os.path.join(EXP_DIR, 'finetune_results_v3.json'), {})
r_ext = load_json(os.path.join(EXP_DIR, 'finetune_results_ext.json'), {})
r_mlp = load_json(os.path.join(EXP_DIR, 'finetune_results_mlp_full.json'), {})
r_g = load_json(os.path.join(EXP_DIR, 'part_g_results.json'), {})

BUDGETS = [1.0, 0.5, 0.2, 0.1, 0.05]
BUDGET_LABELS = ['100%', '50%', '20%', '10%', '5%']
BUDGET_N = [85, 42, 17, 8, 4]  # approx engines

# Reference numbers
STAR_RMSE = 10.61
AE_LSTM_RMSE = 13.99

# LSTM results (hardcoded from run_experiments.py output)
LSTM_MEAN = [17.36, 18.30, 18.55, 31.22, 33.08]
LSTM_STD = [1.24, 0.75, 0.81, 10.93, 9.64]

def get_v2(mode, budget):
    bk = str(budget)
    if bk in r_v2.get(f'jepa_{mode}', {}):
        return r_v2[f'jepa_{mode}'][bk]
    return None

def get_v3(mode, budget):
    bk = f'{budget:.2f}'
    if r_v3 and bk in r_v3:
        return r_v3[bk].get(mode)
    return None

# ============================================================
# Plot 1: Label Efficiency - Full comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('V11 C-MAPSS FD001: Label Efficiency Results', fontsize=14, fontweight='bold')

# Left: RMSE vs budget
ax = axes[0]
x = np.arange(len(BUDGETS))
w = 0.2

# V2 frozen
v2_frz_m = [get_v2('frozen', b)['mean'] if get_v2('frozen', b) else np.nan for b in BUDGETS]
v2_frz_s = [get_v2('frozen', b)['std'] if get_v2('frozen', b) else 0 for b in BUDGETS]
# V2 E2E
v2_e2e_m = [get_v2('e2e', b)['mean'] if get_v2('e2e', b) else np.nan for b in BUDGETS]
v2_e2e_s = [get_v2('e2e', b)['std'] if get_v2('e2e', b) else 0 for b in BUDGETS]

ax.bar(x - 1.5*w, LSTM_MEAN, w, yerr=LSTM_STD, label='Supervised LSTM', color='#e74c3c', alpha=0.8, capsize=4)
ax.bar(x - 0.5*w, v2_frz_m, w, yerr=v2_frz_s, label='JEPA Frozen (V2)', color='#3498db', alpha=0.8, capsize=4)
ax.bar(x + 0.5*w, v2_e2e_m, w, yerr=v2_e2e_s, label='JEPA E2E (V2)', color='#2ecc71', alpha=0.8, capsize=4)

# MLP probe if available
if r_mlp:
    mlp_m = [r_mlp.get(f'{b:.2f}', {}).get('mean', np.nan) for b in BUDGETS]
    mlp_s = [r_mlp.get(f'{b:.2f}', {}).get('std', 0) for b in BUDGETS]
    ax.bar(x + 1.5*w, mlp_m, w, yerr=mlp_s, label='JEPA MLP Probe (V2)', color='#9b59b6', alpha=0.8, capsize=4)

ax.axhline(STAR_RMSE, color='black', linestyle='--', linewidth=2, label=f'STAR supervised ({STAR_RMSE})')
ax.axhline(AE_LSTM_RMSE, color='gray', linestyle='--', linewidth=1.5, label=f'AE-LSTM SSL ({AE_LSTM_RMSE})')
ax.set_xticks(x); ax.set_xticklabels(BUDGET_LABELS)
ax.set_xlabel('Label Budget'); ax.set_ylabel('RMSE (cycles, RUL cap 125)')
ax.set_title('RMSE by Label Budget'); ax.legend(loc='upper left', fontsize=8)
ax.set_ylim(0, 45)
ax.grid(axis='y', alpha=0.3)

# Right: Line plot for better trend visualization
ax2 = axes[1]
ax2.errorbar(range(len(BUDGETS)), LSTM_MEAN, yerr=LSTM_STD, marker='o', color='#e74c3c',
             label='Supervised LSTM', linewidth=2, markersize=6, capsize=4)
ax2.errorbar(range(len(BUDGETS)), v2_frz_m, yerr=v2_frz_s, marker='s', color='#3498db',
             label='JEPA Frozen (V2)', linewidth=2, markersize=6, capsize=4)
ax2.errorbar(range(len(BUDGETS)), v2_e2e_m, yerr=v2_e2e_s, marker='^', color='#2ecc71',
             label='JEPA E2E (V2)', linewidth=2, markersize=6, capsize=4)

if r_mlp and any(not np.isnan(m) for m in mlp_m):
    ax2.errorbar(range(len(BUDGETS)), mlp_m, yerr=mlp_s, marker='D', color='#9b59b6',
                 label='JEPA MLP Probe (V2)', linewidth=2, markersize=6, capsize=4)

ax2.axhline(STAR_RMSE, color='black', linestyle='--', linewidth=2, label=f'STAR ({STAR_RMSE})')
ax2.axhline(AE_LSTM_RMSE, color='gray', linestyle='--', linewidth=1.5, label=f'AE-LSTM ({AE_LSTM_RMSE})')
ax2.set_xticks(range(len(BUDGETS))); ax2.set_xticklabels(BUDGET_LABELS)
ax2.set_xlabel('Label Budget'); ax2.set_ylabel('RMSE (cycles)')
ax2.set_title('Label Efficiency Curve (lower is better)')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'label_efficiency_final.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved label_efficiency_final.png")

# ============================================================
# Plot 2: Architecture Comparison
# ============================================================
if r_v3:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('V11 Architecture Ablation: FD001 @ 100% Labels', fontsize=13, fontweight='bold')

    arch_names = ['V1\n(d=128,L=2)\n366K', 'V2\n(d=256,L=2)\n1.26M', 'V3\n(d=128,L=3)\n499K']
    e2e_means = [
        r_v1['jepa_e2e']['1.0']['mean'],
        r_v2['jepa_e2e']['1.0']['mean'],
        r_v3.get('1.00', {}).get('e2e', {}).get('mean', np.nan)
    ]
    e2e_stds = [
        r_v1['jepa_e2e']['1.0']['std'],
        r_v2['jepa_e2e']['1.0']['std'],
        r_v3.get('1.00', {}).get('e2e', {}).get('std', 0)
    ]
    frz_means = [
        r_v1['jepa_frozen']['1.0']['mean'],
        r_v2['jepa_frozen']['1.0']['mean'],
        r_v3.get('1.00', {}).get('frozen', {}).get('mean', np.nan)
    ]
    frz_stds = [
        r_v1['jepa_frozen']['1.0']['std'],
        r_v2['jepa_frozen']['1.0']['std'],
        r_v3.get('1.00', {}).get('frozen', {}).get('std', 0)
    ]

    ax = axes[0]
    x = np.arange(3)
    ax.bar(x - 0.2, e2e_means, 0.35, yerr=e2e_stds, label='E2E fine-tune', color='#2ecc71', capsize=5)
    ax.bar(x + 0.2, frz_means, 0.35, yerr=frz_stds, label='Frozen probe', color='#3498db', capsize=5)
    ax.axhline(STAR_RMSE, color='black', linestyle='--', label=f'STAR SOTA ({STAR_RMSE})')
    ax.axhline(AE_LSTM_RMSE, color='gray', linestyle='--', label=f'AE-LSTM SSL ({AE_LSTM_RMSE})')
    ax.axhline(17.36, color='#e74c3c', linestyle=':', label='Supervised LSTM (17.36)')
    ax.set_xticks(x); ax.set_xticklabels(arch_names, fontsize=9)
    ax.set_ylabel('RMSE (cycles)'); ax.set_title('Architecture Comparison at 100% Labels')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(8, 28)

    # Right: Label efficiency across V1/V2/V3 E2E
    ax2 = axes[1]
    v1_e2e = [r_v1['jepa_e2e'][str(b)]['mean'] if str(b) in r_v1.get('jepa_e2e', {}) else np.nan for b in BUDGETS]
    v2_e2e_m_list = [get_v2('e2e', b)['mean'] if get_v2('e2e', b) else np.nan for b in BUDGETS]
    v3_e2e = [r_v3.get(f'{b:.2f}', {}).get('e2e', {}).get('mean', np.nan) for b in BUDGETS]

    ax2.plot(range(len(BUDGETS)), v1_e2e, marker='o', label='V1 E2E (d=128,L=2)', color='#e67e22', linewidth=2)
    ax2.plot(range(len(BUDGETS)), v2_e2e_m_list, marker='s', label='V2 E2E (d=256,L=2)', color='#2ecc71', linewidth=2)
    if any(not np.isnan(v) for v in v3_e2e):
        ax2.plot(range(len(BUDGETS)), v3_e2e, marker='^', label='V3 E2E (d=128,L=3)', color='#9b59b6', linewidth=2)
    ax2.plot(range(len(BUDGETS)), LSTM_MEAN, marker='x', label='LSTM', color='#e74c3c',
             linewidth=2, linestyle='--')
    ax2.axhline(STAR_RMSE, color='black', linestyle='--', linewidth=1.5, label=f'STAR ({STAR_RMSE})')
    ax2.axhline(AE_LSTM_RMSE, color='gray', linestyle='--', linewidth=1.5)
    ax2.set_xticks(range(len(BUDGETS))); ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel('Label Budget'); ax2.set_ylabel('RMSE (cycles)')
    ax2.set_title('Label Efficiency: V1 vs V2 vs V3 E2E')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'architecture_ablation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved architecture_ablation.png")

# ============================================================
# Plot 3: Cross-subset Transfer (Part G)
# ============================================================
if r_g and 'cross_fd002_to_fd001' in r_g:
    cross = r_g['cross_fd002_to_fd001']
    fd002 = r_g.get('fd002_indomain', {})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('V11 Part G: Multi-Subset Results', fontsize=13, fontweight='bold')

    # Left: FD001 in-domain vs FD002->FD001 cross-transfer
    ax = axes[0]
    avail_budgets = [b for b in BUDGETS if f'{b:.2f}' in cross]
    if avail_budgets:
        x = np.arange(len(avail_budgets))
        fd001_e2e = [get_v2('e2e', b)['mean'] if get_v2('e2e', b) else np.nan for b in avail_budgets]
        fd001_frz = [get_v2('frozen', b)['mean'] if get_v2('frozen', b) else np.nan for b in avail_budgets]
        cross_e2e = [cross[f'{b:.2f}']['e2e']['mean'] if f'{b:.2f}' in cross else np.nan for b in avail_budgets]
        cross_frz = [cross[f'{b:.2f}']['frozen']['mean'] if f'{b:.2f}' in cross else np.nan for b in avail_budgets]

        labels_to_show = [f'{b:.0%}' for b in avail_budgets]
        ax.plot(range(len(avail_budgets)), fd001_e2e, marker='s', color='#2ecc71',
                label='FD001 in-domain E2E', linewidth=2)
        ax.plot(range(len(avail_budgets)), cross_e2e, marker='^', color='#9b59b6',
                label='FD002->FD001 transfer E2E', linewidth=2)
        ax.plot(range(len(avail_budgets)), fd001_frz, marker='o', color='#3498db',
                label='FD001 in-domain Frozen', linewidth=2, linestyle='--')
        ax.plot(range(len(avail_budgets)), cross_frz, marker='D', color='#e67e22',
                label='FD002->FD001 transfer Frozen', linewidth=2, linestyle='--')
        ax.axhline(STAR_RMSE, color='black', linestyle=':', label=f'STAR FD001 ({STAR_RMSE})')
        ax.set_xticks(range(len(avail_budgets))); ax.set_xticklabels(labels_to_show)
        ax.set_xlabel('Label Budget (FD001 fine-tune)'); ax.set_ylabel('RMSE (cycles)')
        ax.set_title('FD001: In-domain vs Cross-subset Transfer')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Right: FD002 in-domain performance
    ax2 = axes[1]
    avail_fd002 = [b for b in [1.0, 0.5, 0.2, 0.1] if f'{b:.2f}' in fd002]
    if avail_fd002:
        fd002_e2e = [fd002[f'{b:.2f}']['e2e']['mean'] for b in avail_fd002]
        fd002_frz = [fd002[f'{b:.2f}']['frozen']['mean'] for b in avail_fd002]
        fd002_e2e_s = [fd002[f'{b:.2f}']['e2e']['std'] for b in avail_fd002]
        fd002_frz_s = [fd002[f'{b:.2f}']['frozen']['std'] for b in avail_fd002]

        labels_fd002 = [f'{b:.0%}' for b in avail_fd002]
        ax2.errorbar(range(len(avail_fd002)), fd002_e2e, yerr=fd002_e2e_s, marker='s',
                     color='#2ecc71', label='FD002 JEPA E2E', linewidth=2, capsize=4)
        ax2.errorbar(range(len(avail_fd002)), fd002_frz, yerr=fd002_frz_s, marker='o',
                     color='#3498db', label='FD002 JEPA Frozen', linewidth=2, capsize=4)
        ax2.axhline(13.47, color='black', linestyle='--', label='STAR FD002 (13.47)')
        ax2.set_xticks(range(len(avail_fd002))); ax2.set_xticklabels(labels_fd002)
        ax2.set_xlabel('Label Budget'); ax2.set_ylabel('RMSE (cycles)')
        ax2.set_title('FD002 In-domain Results')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'cross_subset_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved cross_subset_results.png")

# ============================================================
# Plot 4: Extended fine-tuning and MLP probe comparison
# ============================================================
if r_ext and r_mlp:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('V11 Fine-tuning Ablations: V2 on FD001', fontsize=13, fontweight='bold')

    ax = axes[0]
    # E2E: 100ep vs 200ep
    methods = ['JEPA E2E\n100ep', 'JEPA E2E\n200ep', 'JEPA Frozen\n100ep', 'JEPA Frozen\n200ep', 'LSTM\n100ep']
    means = [
        r_v2['jepa_e2e']['1.0']['mean'],
        r_ext.get('e2e_200ep', {}).get('mean', np.nan),
        r_v2['jepa_frozen']['1.0']['mean'],
        r_ext.get('frozen_200ep', {}).get('mean', np.nan),
        17.36
    ]
    stds = [
        r_v2['jepa_e2e']['1.0']['std'],
        r_ext.get('e2e_200ep', {}).get('std', 0),
        r_v2['jepa_frozen']['1.0']['std'],
        r_ext.get('frozen_200ep', {}).get('std', 0),
        1.24
    ]
    colors = ['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e74c3c']
    bars = ax.bar(range(len(methods)), means, yerr=stds, color=colors, capsize=5, alpha=0.8)
    ax.axhline(STAR_RMSE, color='black', linestyle='--', label=f'STAR ({STAR_RMSE})')
    ax.axhline(AE_LSTM_RMSE, color='gray', linestyle='--', label=f'AE-LSTM ({AE_LSTM_RMSE})')
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel('RMSE (cycles)'); ax.set_title('Extended Fine-tuning: 100ep vs 200ep @ 100%')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(8, 25)

    ax2 = axes[1]
    # MLP probe across budgets
    mlp_m = [r_mlp.get(f'{b:.2f}', {}).get('mean', np.nan) for b in BUDGETS]
    mlp_s = [r_mlp.get(f'{b:.2f}', {}).get('std', 0) for b in BUDGETS]
    lin_frz = [get_v2('frozen', b)['mean'] if get_v2('frozen', b) else np.nan for b in BUDGETS]
    lin_frz_s = [get_v2('frozen', b)['std'] if get_v2('frozen', b) else 0 for b in BUDGETS]

    ax2.errorbar(range(len(BUDGETS)), lin_frz, yerr=lin_frz_s, marker='o', color='#3498db',
                 label='Linear Probe (frozen)', linewidth=2, capsize=4)
    ax2.errorbar(range(len(BUDGETS)), mlp_m, yerr=mlp_s, marker='s', color='#9b59b6',
                 label='MLP Probe (frozen)', linewidth=2, capsize=4)
    ax2.errorbar(range(len(BUDGETS)), LSTM_MEAN, yerr=LSTM_STD, marker='x', color='#e74c3c',
                 label='LSTM supervised', linewidth=2, linestyle='--', capsize=4)
    ax2.axhline(STAR_RMSE, color='black', linestyle='--', label=f'STAR ({STAR_RMSE})')
    ax2.set_xticks(range(len(BUDGETS))); ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel('Label Budget'); ax2.set_ylabel('RMSE (cycles)')
    ax2.set_title('Linear vs MLP Probe at All Label Budgets')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'finetuning_ablations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved finetuning_ablations.png")

# ============================================================
# Plot 5: Prediction trajectories (needs model)
# ============================================================
print("Loading model for prediction trajectory plot...")
try:
    import torch
    import torch.nn.functional as F
    from data_utils import (load_cmapss_subset, CMAPSSTestDataset, collate_test,
                            N_SENSORS, RUL_CAP)
    from models import TrajectoryJEPA, RULProbe
    from torch.utils.data import DataLoader

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_v2 = os.path.join(EXP_DIR, 'best_pretrain_L1_v2.pt')
    data = load_cmapss_subset('FD001')

    model = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2, d_ff=512)
    model.load_state_dict(torch.load(ckpt_v2, map_location='cpu'))
    model = model.to(DEVICE); model.eval()

    # Train a quick probe for visualization
    probe = RULProbe(256).to(DEVICE)
    from data_utils import CMAPSSFinetuneDataset, collate_finetune
    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=42)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    optim = torch.optim.Adam(list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4)
    best_probe_state = None; best_model_state = None; best_val = float('inf')
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    for ep in range(50):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            h = model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward()
            torch.nn.utils.clip_grad_norm_(list(probe.parameters()) + list(model.context_encoder.parameters()), 1.0)
            optim.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_r < best_val:
            best_val = val_r
            best_probe_state = {k: v.clone() for k, v in probe.state_dict().items()}
            best_model_state = {k: v.clone() for k, v in model.context_encoder.state_dict().items()}

    probe.load_state_dict(best_probe_state)
    model.context_encoder.load_state_dict(best_model_state)
    model.eval(); probe.eval()

    # Generate predictions for all test engines, then plot 5
    # For prediction trajectory: take each test engine, compute RUL at each cut point
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle('Predicted vs True RUL Trajectories (V2 E2E, Test Engines)', fontsize=12)

    test_engines = data['test_engines']
    test_rul = data['test_rul']

    # test_engines is a dict: {engine_id: numpy_array shape (T, 14)}
    eng_ids = sorted(test_engines.keys())
    id_to_idx = {eid: idx for idx, eid in enumerate(eng_ids)}
    lengths = {eid: len(test_engines[eid]) for eid in eng_ids}
    sorted_by_len = sorted(lengths.items(), key=lambda x: x[1])
    n_total = len(sorted_by_len)
    # Pick short, medium-short, medium, medium-long, long
    chosen_ids = [sorted_by_len[5][0], sorted_by_len[n_total//4][0],
                  sorted_by_len[n_total//2][0], sorted_by_len[3*n_total//4][0],
                  sorted_by_len[n_total-5][0]]

    for i, eid in enumerate(chosen_ids):
        eng = test_engines[eid]  # numpy array (T, 14)
        true_rul_final = float(test_rul[id_to_idx[eid]])
        T = len(eng)
        x_tensor = torch.tensor(eng, dtype=torch.float32)

        cut_points = list(range(max(10, T//4), T, max(1, T//20)))
        preds = []
        true_ruls = []

        for t in cut_points:
            past = x_tensor[:t].unsqueeze(0).to(DEVICE)
            mask = torch.ones(1, t, dtype=torch.bool).to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
                pred = probe(h).item() * RUL_CAP
            preds.append(pred)
            # True RUL at this cut point: final_rul + (T - t)
            true_ruls.append(min(true_rul_final + (T - t), 125))

        ax = axes[i]
        ax.plot(cut_points, true_ruls, 'b-', linewidth=2, label='True RUL')
        ax.plot(cut_points, preds, 'r--', linewidth=2, label='Predicted RUL')
        ax.set_title(f'Engine {eid} (T={T})', fontsize=10)
        ax.set_xlabel('Cycle index (cut point)')
        if i == 0: ax.set_ylabel('RUL (cycles)')
        if i == 0: ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'prediction_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved prediction_trajectories.png")

except Exception as e:
    print(f"Prediction trajectory plot failed: {e}")
    import traceback; traceback.print_exc()

# ============================================================
# Summary table print
# ============================================================
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)

print("\nFD001 Label Efficiency (V2 primary):")
print(f"{'Budget':>8} | {'LSTM':>12} | {'V2 Frozen':>12} | {'V2 E2E':>12} | {'MLP Probe':>12}")
print("-" * 65)
for i, b in enumerate(BUDGETS):
    v2f = get_v2('frozen', b)
    v2e = get_v2('e2e', b)
    mlp = r_mlp.get(f'{b:.2f}', {}) if r_mlp else {}
    print(f"  {BUDGET_LABELS[i]:>5}  | {LSTM_MEAN[i]:>5.2f}+/-{LSTM_STD[i]:<5.2f} | "
          f"{v2f['mean'] if v2f else np.nan:>5.2f}+/-{v2f['std'] if v2f else 0:<5.2f} | "
          f"{v2e['mean'] if v2e else np.nan:>5.2f}+/-{v2e['std'] if v2e else 0:<5.2f} | "
          f"{mlp.get('mean', np.nan):>5.2f}+/-{mlp.get('std', 0):<5.2f}")

print(f"\nSOTA references: STAR supervised = {STAR_RMSE}, AE-LSTM SSL = {AE_LSTM_RMSE}")

if r_v3:
    print(f"\nArchitecture @ 100%:")
    print(f"  V1 (d=128,L=2): E2E={r_v1['jepa_e2e']['1.0']['mean']:.2f}, Frozen={r_v1['jepa_frozen']['1.0']['mean']:.2f}")
    print(f"  V2 (d=256,L=2): E2E={r_v2['jepa_e2e']['1.0']['mean']:.2f}, Frozen={r_v2['jepa_frozen']['1.0']['mean']:.2f}")
    if '1.00' in r_v3:
        print(f"  V3 (d=128,L=3): E2E={r_v3['1.00']['e2e']['mean']:.2f}, Frozen={r_v3['1.00']['frozen']['mean']:.2f}")

if r_g and 'fd002_indomain' in r_g:
    fd002 = r_g['fd002_indomain']
    if '1.00' in fd002:
        print(f"\nFD002 @ 100%: E2E={fd002['1.00']['e2e']['mean']:.2f} (STAR ref: 13.47)")
    if r_g.get('cross_fd002_to_fd001') and '0.10' in r_g['cross_fd002_to_fd001']:
        cross = r_g['cross_fd002_to_fd001']
        print(f"Cross FD002->FD001 @ 10%: E2E={cross['0.10']['e2e']['mean']:.2f} "
              f"vs in-domain={r_v2['jepa_e2e']['0.1']['mean']:.2f}")

print("\nAll plots saved to:", PLOTS_DIR)

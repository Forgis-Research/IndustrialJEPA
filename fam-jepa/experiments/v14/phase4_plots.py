"""
V14 Phase 4: Publication-quality C-MAPSS plots.

4-6 plots total. Every plot must have a clear point.

Output: analysis/plots/v14/ (PNG) + notebooks/plots/ (PDF)

Plots:
  F1. Dataset overview (3 panels)
    A: normalized sensor trajectories for representative-length engine
    B: same sensors for extreme (longest) engine
    C: RUL label distribution showing piecewise-linear cap at 125
  F2. Method illustration (2 panels)
    A: Schematic of trajectory prediction task on one engine's trajectory
    B: True RUL vs predicted RUL for one test engine across cut points
       (Note: this needs inference, which we skip to avoid loading model;
       we substitute with a labeled RUL trajectory showing piecewise shape.)
  F3. Key results: label-efficiency + from-scratch
    A: Label efficiency curves (JEPA E2E, JEPA Frozen, STAR, LSTM)
    B: From-scratch ablation (pretrained vs random init across budgets)
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
PLOTS_PNG = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v14')
PLOTS_PDF = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots')
PLOTS_PNG.mkdir(parents=True, exist_ok=True)
PLOTS_PDF.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(V11_DIR))

from data_utils import load_cmapss_subset, compute_rul_labels, RUL_CAP, SELECTED_SENSORS

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


# Consistent method palette across figures
COLOR = {
    'jepa_e2e':    '#1f77b4',  # blue
    'jepa_frozen': '#2ca02c',  # green
    'star':        '#d62728',  # red
    'lstm':        '#7f7f7f',  # gray
    'scratch':     '#ff7f0e',  # orange
    'pretrain':    '#1f77b4',  # blue
}


def save(fig, name):
    png = PLOTS_PNG / f'{name}.png'
    pdf = PLOTS_PDF / f'{name}.pdf'
    fig.savefig(png, dpi=200, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    print(f"  -> {png}")
    print(f"  -> {pdf}")


# ============================================================
# Load data
# ============================================================
print("Loading FD001...")
data = load_cmapss_subset('FD001')
all_engines = {**data['train_engines'], **data['val_engines']}

# Pick median-length and longest engines
lengths = {eid: len(seq) for eid, seq in all_engines.items()}
ids_by_len = sorted(lengths.items(), key=lambda kv: kv[1])
median_id, median_len = ids_by_len[len(ids_by_len) // 2]
longest_id, longest_len = ids_by_len[-1]
print(f"Median engine: id={median_id}, len={median_len}")
print(f"Longest engine: id={longest_id}, len={longest_len}")

# Sensors to highlight: s2, s7, s12, s21 correspond to
# positions 0, 3, 7, 13 in SELECTED_SENSORS (which is [2,3,4,7,8,9,11,12,13,14,15,17,20,21])
SHOW_SENSORS = {'s2 (LPC out temp)': 0, 's7 (HPC out press)': 3,
                's12 (fuel flow)': 7, 's21 (bleed enthalpy)': 13}


# ============================================================
# F1. Dataset overview
# ============================================================
print("\n=== F1: Dataset overview ===")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

for ax, (eid, label) in zip(axes[:2],
                            [(median_id, f'representative engine (len={median_len})'),
                             (longest_id, f'extreme engine (len={longest_len})')]):
    seq = all_engines[eid]
    cycles = np.arange(len(seq))
    for name, idx in SHOW_SENSORS.items():
        ax.plot(cycles, seq[:, idx], lw=1.3, label=name)
    ax.set_xlabel('cycle')
    ax.set_ylabel('normalized sensor value')
    ax.set_title(label)
    ax.legend(loc='best', frameon=False, fontsize=7)

# Panel C: RUL distribution across all training engines
ax = axes[2]
all_rul = []
for eid, seq in data['train_engines'].items():
    all_rul.append(compute_rul_labels(len(seq), RUL_CAP))
all_rul = np.concatenate(all_rul)
ax.hist(all_rul, bins=50, color='#4477AA', edgecolor='white', linewidth=0.5)
ax.axvline(RUL_CAP, color='#d62728', linestyle='--', lw=1.2,
           label=f'cap = {RUL_CAP}')
plateau_frac = float(np.mean(all_rul == RUL_CAP))
ax.set_xlabel('capped RUL (cycles)')
ax.set_ylabel('count (all training cycles)')
ax.set_title(f'RUL label distribution\nplateau fraction = {plateau_frac:.1%}')
ax.legend(loc='upper left', frameon=False)

fig.suptitle('C-MAPSS FD001: sensor trajectories and label distribution',
             fontsize=11, y=1.02)
fig.tight_layout()
save(fig, 'fig1_cmapss_overview')
plt.close(fig)


# ============================================================
# F2. Method illustration (2 panels) + RUL labels example
# ============================================================
print("\n=== F2: Method illustration ===")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

# Panel A: schematic — one engine trajectory split at cut t
seq = all_engines[median_id]
cycles = np.arange(len(seq))
cut = len(seq) * 2 // 3
ax = axes[0]
# Show one representative sensor
sensor_idx = 0  # s2
past_x = cycles[:cut]
future_x = cycles[cut:cut + 30]   # horizon up to k=30
rest_x = cycles[cut + 30:]
ax.plot(past_x, seq[:cut, sensor_idx], color='#4477AA', lw=2.0,
        label=r'past $x_{1:t}$ (context)')
ax.plot(future_x, seq[cut:cut + 30, sensor_idx], color='#d62728', lw=2.0,
        label=r'future $x_{t+1:t+k}$ (target)')
ax.plot(rest_x, seq[cut + 30:, sensor_idx], color='gray', lw=1.0, alpha=0.4,
        label='unseen')
ax.axvline(cut, color='black', linestyle=':', lw=0.8)
ax.annotate('cutoff t', xy=(cut, seq[cut, sensor_idx]),
            xytext=(cut + 15, seq[cut, sensor_idx] + 0.08),
            fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.6))
# Add text boxes
ax.text(cut / 2, ax.get_ylim()[1] * 0.95,
        r'$h_{\mathrm{past}} = f_\theta(x_{1:t})$',
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='#e8f0f8', ec='#4477AA'))
ax.text(cut + 15, ax.get_ylim()[1] * 0.95,
        r'$h_{\mathrm{future}} = \bar{f}_\xi(x_{t+1:t+k})$',
        ha='center', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='#f8e0e0', ec='#d62728'))
ax.set_xlabel('cycle')
ax.set_ylabel(list(SHOW_SENSORS.keys())[0])
ax.set_title(r'Trajectory JEPA task: predict $h_{\mathrm{future}}$ from $h_{\mathrm{past}}$')
ax.legend(loc='lower left', frameon=False, fontsize=8)

# Panel B: True capped RUL trajectory for the same engine (shows the piecewise label)
ax = axes[1]
rul = compute_rul_labels(len(seq), RUL_CAP)
ax.plot(cycles, rul, color='#4477AA', lw=1.8, label='capped RUL label')
# Mark healthy plateau region
plateau_end = np.argmax(rul < RUL_CAP)
ax.axvspan(0, plateau_end, color='#cfe5c8', alpha=0.5, label='healthy plateau')
ax.axvspan(plateau_end, len(seq), color='#f6cccc', alpha=0.4, label='degradation phase')
ax.set_xlabel('cycle')
ax.set_ylabel('capped RUL (cycles)')
ax.set_title(f'capped RUL label for engine (len={median_len})')
ax.legend(loc='upper right', frameon=False, fontsize=8)

fig.tight_layout()
save(fig, 'fig2_method_schematic')
plt.close(fig)


# ============================================================
# F3. Label efficiency + from-scratch ablation
# ============================================================
print("\n=== F3: Label efficiency + from-scratch ===")

# Numbers from v13/RESULTS.md
budgets = np.array([100, 50, 20, 10, 5])
# From paper-neurips/paper.tex Table 1 and v13 RESULTS
star_rmse = {100: 12.19, 50: 13.26, 20: 17.74, 10: 18.72, 5: 24.55}
star_std  = {100: 0.55, 50: 0.74, 20: 3.62, 10: 2.76, 5: 6.45}
lstm_rmse = {100: 17.36, 50: 18.30, 20: 18.55, 10: 31.22, 5: 33.08}
lstm_std  = {100: 1.2, 50: 0.8, 20: 0.8, 10: 10.9, 5: 9.6}
e2e_rmse  = {100: 14.23, 50: 14.93, 20: 16.54, 10: 18.66, 5: 25.33}
e2e_std   = {100: 0.39, 50: 0.4, 20: 0.8, 10: 0.8, 5: 5.1}
fro_rmse  = {100: 17.81, 50: 18.71, 20: 19.83, 10: 19.93, 5: 21.53}
fro_std   = {100: 1.7, 50: 1.1, 20: 0.3, 10: 0.9, 5: 2.0}

# From-scratch ablation (v13 phase0c): at budgets 100/20/10/5
fs_budgets = np.array([100, 20, 10, 5])
fs_pre = {100: 14.18, 20: 18.00, 10: 19.97, 5: 29.64}
fs_pre_std = {100: 0.55, 20: 1.37, 10: 2.19, 5: 5.27}
fs_scratch = {100: 22.99, 20: 32.50, 10: 35.59, 5: 37.59}
fs_scratch_std = {100: 2.33, 20: 1.50, 10: 2.67, 5: 2.00}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

# Panel A: label efficiency
ax = axes[0]
series = [
    ('LSTM (supervised)',      lstm_rmse, lstm_std, COLOR['lstm'],     '^', '--'),
    ('STAR (supervised SOTA)', star_rmse, star_std, COLOR['star'],     's', '--'),
    ('JEPA E2E (ours)',        e2e_rmse,  e2e_std,  COLOR['jepa_e2e'], 'o', '-'),
    ('JEPA frozen (ours)',     fro_rmse,  fro_std,  COLOR['jepa_frozen'], 'd', '-'),
]
for label, d, s, c, marker, ls in series:
    y = np.array([d[b] for b in budgets])
    yerr = np.array([s[b] for b in budgets])
    ax.errorbar(budgets, y, yerr=yerr, marker=marker, linestyle=ls, label=label,
                color=c, capsize=3, lw=1.6, markersize=6)
# Annotate crossover at 5%
ax.annotate('crossover: JEPA frozen\nbeats STAR at 5%',
            xy=(5, 21.53), xytext=(14, 31),
            fontsize=8, ha='center',
            arrowprops=dict(arrowstyle='->', lw=0.6, color='#333'))
ax.set_xscale('log')
ax.set_xticks(budgets)
ax.set_xticklabels([f'{b}%' for b in budgets])
ax.set_xlabel('label budget (fraction of training engines)')
ax.set_ylabel('test RMSE (cycles)')
ax.set_title('FD001 label efficiency (5 seeds each)')
ax.legend(loc='upper left', frameon=False, fontsize=8)
ax.set_ylim(bottom=10)

# Panel B: from-scratch ablation
ax = axes[1]
y_pre = np.array([fs_pre[b] for b in fs_budgets])
s_pre = np.array([fs_pre_std[b] for b in fs_budgets])
y_scr = np.array([fs_scratch[b] for b in fs_budgets])
s_scr = np.array([fs_scratch_std[b] for b in fs_budgets])
ax.errorbar(fs_budgets, y_pre, yerr=s_pre, marker='o', label='pretrained E2E',
            color=COLOR['pretrain'], capsize=3, lw=1.8, markersize=7)
ax.errorbar(fs_budgets, y_scr, yerr=s_scr, marker='s',
            label='from scratch (random init) E2E',
            color=COLOR['scratch'], capsize=3, lw=1.8, markersize=7)
# Fill between to highlight delta
ax.fill_between(fs_budgets, y_pre, y_scr, color='#cccccc', alpha=0.2,
                label='pretraining contribution')
# Annotate deltas at each budget
for b in fs_budgets:
    d = fs_scratch[b] - fs_pre[b]
    ax.annotate(f'+{d:.1f}', xy=(b, (fs_pre[b] + fs_scratch[b]) / 2),
                fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.8))
ax.set_xscale('log')
ax.set_xticks(fs_budgets)
ax.set_xticklabels([f'{b}%' for b in fs_budgets])
ax.set_xlabel('label budget')
ax.set_ylabel('test RMSE (cycles)')
ax.set_title('From-scratch ablation: the pretraining contribution')
ax.legend(loc='upper left', frameon=False, fontsize=8)

fig.tight_layout()
save(fig, 'fig3_label_efficiency_and_from_scratch')
plt.close(fig)

print("\nDone — all plots saved.")
print(f"PNG:  {PLOTS_PNG}")
print(f"PDF:  {PLOTS_PDF}")

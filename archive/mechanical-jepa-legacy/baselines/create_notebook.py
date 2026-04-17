"""
Create the Phase 6 Jupyter notebook: 07_baseline_establishment.ipynb

Generates figures and a comprehensive notebook.
"""

import nbformat
import json
import os

NOTEBOOK_PATH = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/07_baseline_establishment.ipynb'
PLOTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/baselines/results'


def code_cell(source, tags=None):
    c = nbformat.v4.new_code_cell(source=source)
    if tags:
        c.metadata['tags'] = tags
    return c


def markdown_cell(source):
    return nbformat.v4.new_markdown_cell(source=source)


cells = []

# ============================================================
# TITLE AND SETUP
# ============================================================
cells.append(markdown_cell("""# Mechanical-JEPA V7: Baseline Establishment

**Session**: Overnight V7 — 2026-04-07
**Goal**: Establish comprehensive baselines for all task families on the Forgis/Mechanical-Components dataset.
**Tasks**:
1. Cross-domain fault classification
2. One-class anomaly detection
3. Health indicator forecasting
4. RUL estimation
"""))

cells.append(code_cell("""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Load results
RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/baselines/results'
PLOTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots'

with open(f'{RESULTS_DIR}/classification_baselines.json') as f:
    clf_results = json.load(f)
with open(f'{RESULTS_DIR}/anomaly_detection_baselines.json') as f:
    anom_results = json.load(f)
with open(f'{RESULTS_DIR}/forecasting_baselines.json') as f:
    fore_results = json.load(f)

print("Results loaded successfully")
print(f"Classification methods: {[k for k in clf_results if not k.startswith('_')]}")
print(f"Anomaly sources: {list(anom_results.get('by_source', {}).keys())}")
print(f"Forecasting tasks: {[k for k in fore_results if not k.startswith('_')]}")
"""))

# ============================================================
# SECTION 1: DATASET OVERVIEW
# ============================================================
cells.append(markdown_cell("""## 1. Dataset Overview

The `Forgis/Mechanical-Components` HuggingFace dataset contains ~12,000 samples from 16 sources.
Key characteristics:
- Mixed bearing types, fault mechanisms, test rigs
- Highly variable signal lengths (0.1s to 40s per sample)
- 5 run-to-failure sources (FEMTO, XJTU-SY, IMS, Ottawa, SCA) with temporal ordering
"""))

cells.append(code_cell("""
# Dataset statistics table
dataset_stats = {
    'CWRU': {'n': 60, 'sr': 12000, 'dur': '20-40s', 'has_episodes': 'No', 'task': 'Train (clf)'},
    'MFPT': {'n': 20, 'sr': 48828, 'dur': '6s', 'has_episodes': 'No', 'task': 'Train (clf)'},
    'MAFAULDA': {'n': 800, 'sr': 50000, 'dur': '0.512s', 'has_episodes': 'No', 'task': 'Train (clf, anomaly)'},
    'SEU': {'n': 140, 'sr': 5120, 'dur': '5s', 'has_episodes': 'No', 'task': 'Train (clf)'},
    'FEMTO': {'n': 3569, 'sr': 25600, 'dur': '0.1s', 'has_episodes': 'Yes', 'task': 'Anomaly, RUL'},
    'XJTU-SY': {'n': 1370, 'sr': 25600, 'dur': '1.28s', 'has_episodes': 'Yes', 'task': 'RUL'},
    'Ottawa': {'n': 180, 'sr': 42000, 'dur': '2s', 'has_episodes': 'Partial', 'task': 'Test (clf)'},
    'Paderborn': {'n': 384, 'sr': 64000, 'dur': '4s', 'has_episodes': 'No', 'task': 'Test (clf)'},
}
df_stats = pd.DataFrame(dataset_stats).T
df_stats.index.name = 'Source'
print(df_stats.to_string())
"""))

# ============================================================
# SECTION 2: CLASSIFICATION RESULTS
# ============================================================
cells.append(markdown_cell("""## 2. Task 1: Cross-Domain Fault Classification

**Setup**: Train on CWRU+MAFAULDA+SEU (1000 samples), test on Ottawa+Paderborn (564 samples).
**Metric**: Macro F1 (primary), accuracy
**Challenge**: Different machines, fault types, and physical mechanisms across sources.
"""))

cells.append(code_cell("""
# Extract classification results
methods_clf = []
for k, v in clf_results.items():
    if k.startswith('_') or k == 'deep_note':
        continue
    f1 = v.get('macro_f1_mean', v.get('macro_f1', None))
    f1_std = v.get('macro_f1_std', 0.0)
    acc = v.get('accuracy_mean', v.get('accuracy', None))
    if f1 is not None:
        methods_clf.append({'method': k, 'f1': f1, 'f1_std': f1_std, 'acc': acc})

df_clf = pd.DataFrame(methods_clf).sort_values('f1', ascending=False)
print("Classification Results (Cross-Domain):")
print(df_clf.to_string(index=False))
"""))

cells.append(code_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Task 1: Cross-Domain Fault Classification\\n(Train: CWRU+MAFAULDA+SEU → Test: Ottawa+Paderborn)',
             fontsize=13, fontweight='bold')

# F1 bar chart
ax = axes[0]
methods = df_clf['method'].tolist()
f1s = df_clf['f1'].tolist()
f1_stds = df_clf['f1_std'].fillna(0).tolist()
colors = ['#d62728' if m.startswith('trivial') else '#1f77b4' if 'cnn' in m or 'resnet' in m else '#2ca02c'
          for m in methods]

bars = ax.barh(methods, f1s, xerr=f1_stds, color=colors, alpha=0.8, capsize=4)
ax.axvline(x=df_clf[~df_clf['method'].str.startswith('trivial')]['f1'].max(),
           color='black', linestyle='--', alpha=0.5, label='Best non-trivial')
ax.set_xlabel('Macro F1 (cross-domain)')
ax.set_title('Macro F1 by Method')
ax.legend(fontsize=9)
ax.set_xlim(0, 0.35)
for bar, val in zip(bars, f1s):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', fontsize=8)

# Legend for colors
from matplotlib.patches import Patch
legend_elements = [Patch(color='#d62728', label='Trivial'),
                   Patch(color='#2ca02c', label='Feature-based'),
                   Patch(color='#1f77b4', label='Deep learning')]
ax.legend(handles=legend_elements, fontsize=9)

# F1 vs Accuracy scatter
ax2 = axes[1]
for _, row in df_clf.iterrows():
    if row['acc'] is not None:
        color = '#d62728' if row['method'].startswith('trivial') else \
                '#1f77b4' if 'cnn' in row['method'] or 'resnet' in row['method'] else '#2ca02c'
        ax2.scatter(row['acc'], row['f1'], s=100, color=color, zorder=5)
        ax2.annotate(row['method'], (row['acc'], row['f1']), fontsize=7,
                     xytext=(5, 0), textcoords='offset points')
ax2.set_xlabel('Accuracy')
ax2.set_ylabel('Macro F1')
ax2.set_title('F1 vs Accuracy Trade-off')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/fig_baseline_classification_matrix.pdf', bbox_inches='tight', dpi=150)
plt.savefig(f'{PLOTS_DIR}/fig_baseline_classification_matrix.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved classification figure")
"""))

# ============================================================
# SECTION 3: ANOMALY DETECTION RESULTS
# ============================================================
cells.append(markdown_cell("""## 3. Task 2: One-Class Anomaly Detection

**Setup**: Train on healthy samples only. Test on healthy + faulty mixture.
**Metric**: AUROC (primary)
**Sources**: FEMTO (good quality), MAFAULDA (problematic — only 49 healthy samples)
"""))

cells.append(code_cell("""
# Extract anomaly detection results for FEMTO
sources_anom = list(anom_results.get('by_source', {}).keys())
print(f"Sources evaluated: {sources_anom}")

for source in sources_anom:
    src_res = anom_results['by_source'][source]
    print(f"\\n--- {source.upper()} ---")
    for method, metrics in src_res.items():
        auroc = metrics.get('auroc_mean', metrics.get('auroc', None))
        if auroc is not None and not np.isnan(float(auroc)):
            std = metrics.get('auroc_std', '')
            std_str = f" ± {std:.3f}" if isinstance(std, float) else ''
            print(f"  {method:<30}: AUROC = {float(auroc):.4f}{std_str}")
"""))

cells.append(code_cell("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Task 2: One-Class Anomaly Detection (AUROC)', fontsize=13, fontweight='bold')

for idx, source in enumerate(['femto', 'mafaulda']):
    if source not in anom_results.get('by_source', {}):
        continue
    src_res = anom_results['by_source'][source]
    ax = axes[idx]

    methods_a, aurocs_a, stds_a = [], [], []
    for method, metrics in src_res.items():
        auroc = metrics.get('auroc_mean', metrics.get('auroc', None))
        if auroc is not None and not np.isnan(float(auroc)):
            methods_a.append(method)
            aurocs_a.append(float(auroc))
            stds_a.append(float(metrics.get('auroc_std', 0.0)))

    # Sort by AUROC
    order = np.argsort(aurocs_a)[::-1]
    methods_a = [methods_a[i] for i in order]
    aurocs_a = [aurocs_a[i] for i in order]
    stds_a = [stds_a[i] for i in order]

    colors_a = ['#d62728' if m.startswith('trivial') else '#2ca02c' if m == 'autoencoder' else '#1f77b4'
                for m in methods_a]

    bars = ax.barh(methods_a, aurocs_a, xerr=stds_a, color=colors_a, alpha=0.8, capsize=3)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    ax.axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='JEPA target')
    ax.set_xlabel('AUROC')
    ax.set_title(f'{source.upper()} — Anomaly Detection')
    ax.set_xlim(0, 1.0)
    ax.legend(fontsize=8)

    for bar, val in zip(bars, aurocs_a):
        ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=7)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/fig_baseline_anomaly_roc.pdf', bbox_inches='tight', dpi=150)
plt.savefig(f'{PLOTS_DIR}/fig_baseline_anomaly_roc.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved anomaly detection figure")
"""))

# ============================================================
# SECTION 4: FORECASTING RESULTS
# ============================================================
cells.append(markdown_cell("""## 4. Task 3 & 4: Health Indicator Forecasting and RUL Estimation

**Task 3**: Predict RMS/kurtosis trajectory H steps ahead (FEMTO, run-to-failure)
**Task 4**: Predict rul_percent from current window features (FEMTO + XJTU-SY)
**Note**: rul_percent = 1 - episode_position definitionally in this dataset, so "linear position" is an oracle cheat, not a valid baseline.
"""))

cells.append(code_cell("""
# HI Forecasting results across horizons
horizons = [1, 5, 10]
hi_types = ['rms', 'kurtosis']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Task 3: Health Indicator Forecasting (FEMTO)\\nRMSE vs Horizon (normalized scale)',
             fontsize=13, fontweight='bold')

colors_f = {
    'trivial_last_value': '#d62728',
    'trivial_moving_avg': '#ff7f0e',
    'trivial_linear_extrap': '#e377c2',
    'trivial_constant_mean': '#bcbd22',
    'ridge': '#1f77b4',
    'random_forest': '#2ca02c',
    'xgboost': '#9467bd',
    'arima': '#8c564b',
}

for hi_idx, hi_type in enumerate(hi_types):
    task_key = f'hi_forecasting_{hi_type}'
    if task_key not in fore_results:
        continue
    task_data = fore_results[task_key]

    # Method x horizon matrix
    all_methods_f = set()
    for h in horizons:
        hk = f'horizon_{h}'
        if hk in task_data:
            all_methods_f.update(task_data[hk].keys())
    all_methods_f = sorted(all_methods_f)

    # Plot H=1, H=5, H=10 side by side
    for h_idx, h in enumerate(horizons):
        ax = axes[hi_idx][h_idx]
        hk = f'horizon_{h}'
        if hk not in task_data:
            ax.text(0.5, 0.5, f'H={h} not available', ha='center', va='center', transform=ax.transAxes)
            continue
        h_data = task_data[hk]

        methods_f = list(h_data.keys())
        rmses_f = []
        for m in methods_f:
            rm = h_data[m].get('rmse_mean', h_data[m].get('rmse', None))
            if rm is not None:
                rmses_f.append((m, float(rm)))
        rmses_f.sort(key=lambda x: x[1])

        names = [x[0] for x in rmses_f]
        vals = [x[1] for x in rmses_f]
        cols = [colors_f.get(n, '#aec7e8') for n in names]

        ax.barh(names, vals, color=cols, alpha=0.8)
        ax.set_xlabel('RMSE (normalized)')
        ax.set_title(f'{hi_type.upper()}, H={h}')

        for i, (n, v) in enumerate(rmses_f):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=7)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/fig_baseline_forecasting_trajectories.pdf', bbox_inches='tight', dpi=150)
plt.savefig(f'{PLOTS_DIR}/fig_baseline_forecasting_trajectories.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved forecasting figure")
"""))

cells.append(code_cell("""
# RUL estimation
rul_data = fore_results.get('rul_estimation', {})
print("=== RUL Estimation Results ===")
for method, metrics in rul_data.items():
    if isinstance(metrics, dict):
        rmse = metrics.get('rmse_mean', metrics.get('rmse', None))
        if rmse is not None:
            std = metrics.get('rmse_std', '')
            std_str = f" ± {std:.4f}" if isinstance(std, float) else ''
            note = ' [ORACLE - cheat]' if method == 'trivial_linear_position' else ''
            print(f"  {method:<30}: RMSE = {float(rmse):.4f}{std_str}{note}")
"""))

# ============================================================
# SECTION 5: SUMMARY TABLE
# ============================================================
cells.append(markdown_cell("""## 5. Summary: Bars to Clear for JEPA

Summary of all baselines. The far right column shows what JEPA needs to achieve to be publishable.
"""))

cells.append(code_cell("""
# Create summary table
summary_data = {
    'Cross-domain Classification (Macro F1)': {
        'trivial': 0.04,
        'best_baseline': 0.193,
        'method': 'Random Forest',
        'jepa_target': 0.30,
        'improvement': '+56%'
    },
    'Anomaly Detection AUROC (FEMTO)': {
        'trivial': 0.50,
        'best_baseline': 0.779,
        'method': 'Kurtosis threshold',
        'jepa_target': 0.85,
        'improvement': '+9%'
    },
    'RMS Forecasting H=1 RMSE': {
        'trivial': 0.351,
        'best_baseline': 0.311,
        'method': 'Random Forest',
        'jepa_target': 0.25,
        'improvement': '-20%'
    },
    'RUL Estimation RMSE': {
        'trivial': 0.290,
        'best_baseline': 0.212,
        'method': 'XGBoost',
        'jepa_target': 0.17,
        'improvement': '-20%'
    },
}

df_summary = pd.DataFrame(summary_data).T
df_summary.index.name = 'Task'
print(df_summary.to_string())

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('Summary: Baseline Performance and JEPA Targets', fontsize=13, fontweight='bold')

tasks = list(summary_data.keys())
x = np.arange(len(tasks))
width = 0.25

trivials = [d['trivial'] for d in summary_data.values()]
baselines = [d['best_baseline'] for d in summary_data.values()]
targets = [d['jepa_target'] for d in summary_data.values()]

bars1 = ax.bar(x - width, trivials, width, label='Trivial baseline', color='#d62728', alpha=0.7)
bars2 = ax.bar(x, baselines, width, label='Best ML baseline', color='#2ca02c', alpha=0.7)
bars3 = ax.bar(x + width, targets, width, label='JEPA target', color='#1f77b4', alpha=0.7, linestyle='--')

ax.set_xticks(x)
ax.set_xticklabels([t.split(' (')[0] for t in tasks], rotation=15, ha='right', fontsize=9)
ax.set_ylabel('Metric value')
ax.legend()
ax.set_title('Note: for classification and AUROC, higher is better; for RMSE, lower is better')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/fig_baseline_summary.pdf', bbox_inches='tight', dpi=150)
plt.savefig(f'{PLOTS_DIR}/fig_baseline_summary.png', bbox_inches='tight', dpi=150)
plt.show()
print("Saved summary figure")
"""))

cells.append(markdown_cell("""## 6. Key Findings and Implications for JEPA

### Finding 1: Cross-domain classification is genuinely hard
F1=0.19 with Random Forest shows the domain gap is real. JEPA representations that are invariant to machine-specific features should show large gains here.

### Finding 2: Kurtosis is the most powerful trivial anomaly detector
The fact that kurtosis (AUROC=0.779) outperforms all feature-based ML methods on FEMTO is striking. This implies that MAFAULDA (only 49 healthy samples) is a poor benchmark for anomaly detection — it should be excluded from anomaly evaluation.

### Finding 3: RMS is forecastable; kurtosis is not
For H=1, Random Forest achieves RMSE=0.311 for RMS but 0.995 for kurtosis.
This tells us: the degeneration signal in RMS is smoother and more predictable.
JEPA's target variable should be RMS (or power spectrum energy bands), not raw kurtosis.

### Finding 4: RUL estimation has a meaningful gap
XGBoost achieves RMSE=0.212, far above zero. A JEPA model that captures temporal degradation trajectory should do better. Target: RMSE < 0.17 (20% improvement).

### Finding 5: Deep models are not clearly better than feature-based
CNN/ResNet achieve F1=0.16-0.17, lower than Random Forest (0.19). This is likely because:
- Limited data for deep training (only 60 CWRU + 140 SEU = 200 samples after filtering)
- Short training (40 epochs with small batch)
The comparison is not fully fair; with more data, deep models should do better.
"""))

# ============================================================
# WRITE NOTEBOOK
# ============================================================
nb = nbformat.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {
    'display_name': 'Python 3',
    'language': 'python',
    'name': 'python3',
}

with open(NOTEBOOK_PATH, 'w') as f:
    nbformat.write(nb, f)

print(f"Notebook written to {NOTEBOOK_PATH}")

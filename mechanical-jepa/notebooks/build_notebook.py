"""
Builds the complete 06_v6_walkthrough.ipynb by inserting missing sections
into the existing 22-cell notebook.

Run: python3 build_notebook.py
"""

import json
from pathlib import Path
import hashlib

NB_PATH = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/06_v6_walkthrough.ipynb')


def md(source):
    return {
        "cell_type": "markdown",
        "id": hashlib.md5(source.encode()).hexdigest()[:8],
        "metadata": {},
        "source": source
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": hashlib.md5(source.encode()).hexdigest()[:8],
        "metadata": {},
        "outputs": [],
        "source": source
    }


# ============================================================
# NEW CELLS
# ============================================================

CELL_INTRO = md(r"""## 0. Introduction & Motivation

### What is JEPA?

**JEPA** (Joint-Embedding Predictive Architecture) is a self-supervised learning framework introduced by LeCun (2022) and implemented for images by Assran et al. (I-JEPA, CVPR 2023). The core idea:

> Predict *representations* of masked regions, not raw pixels (or samples).

Unlike masked autoencoders (MAE) that reconstruct in signal space, JEPA trains a predictor to forecast the **target encoder's embeddings** of masked patches. The target encoder is an exponential moving average (EMA) of the online encoder — it cannot be directly optimized, preventing collapse.

```
Context patches ──→ Encoder ──→ Predictor ──→ predicted z_masked
                         ↑                              ↓
Masked patches ──→ EMA Target Encoder ──→  z_masked  (MSE/L1 target)
```

### Why vibration signals?

Industrial bearings generate **non-stationary vibration signals** with fault-specific impulse patterns. The challenge:
- Different machines have different resonance modes
- Different sensors have different frequency responses
- Labeled data is **expensive** — you can't fault-label every machine

**Hypothesis**: JEPA learns patch-level, position-aware representations that capture the *structure* of fault impulses without memorizing machine-specific artifacts. These should transfer across machines.

### Literature gap

| Method | Approach | Transfer? |
|--------|----------|-----------|
| CNN/Transformer supervised | Labels required | Overfits to source domain |
| MAE (reconstruct) | Self-supervised, signal-space | Near-zero transfer (+0.1%) |
| Contrastive (SimCLR etc.) | Self-supervised | Requires augmentation design |
| **JEPA (ours)** | Self-supervised, **latent-space** | **+37.1% transfer gain** |

No prior work has applied JEPA-style latent-predictive pretraining to 1D vibration signals or demonstrated cross-machine bearing fault transfer.
""")


CELL_ARCHITECTURE = md(r"""## Architecture: JEPA V2

The V2 model resolves the **predictor collapse** problem that plagued V1.

### Encoder
- Input: `(B, 3, 4096)` — 3 channels (DE/FE/BA), 4096 samples at 12kHz (~0.34 sec)
- Patching: 16 non-overlapping patches of 256 samples each
- Transformer: 4 layers, d=512, 4 heads, sinusoidal positional encoding
- Output: `(B, 16, 512)` patch embeddings

### JEPA objective
- Mask ratio: **0.625** — 10 of 16 patches masked (higher = harder = better representations)
- Context encoder: standard forward pass on visible patches
- Target encoder: **EMA** of context encoder (momentum=0.996, no gradient)
- Predictor: 2-layer Transformer taking context + masked position tokens → predicted embeddings
- Loss: **L1** between predicted and target embeddings + variance regularization (λ=0.1)

### Why 5 specific fixes were needed (all required)

| Fix | Reason |
|-----|--------|
| **Sinusoidal PE in predictor** | Predictor must know *where* to predict; without it, ignores position |
| **L1 loss (not MSE)** | MSE allows "predict the mean" shortcut; L1 penalizes uniform predictions more harshly |
| **Variance regularization (λ=0.1)** | Direct penalty: if all predictions are similar, add to loss |
| **Mask ratio 0.625 (not 0.5)** | Harder task forces encoder to learn more complete representations |
| **EMA target encoder** | The key JEPA insight: target encoder cannot be directly optimized, preventing collapse |

Remove any one of these → collapse or degraded transfer (see ablation, Section 6).

### Parameters
- Encoder: ~4.0M
- Predictor: ~1.1M
- Total: **5.1M parameters** (same as Transformer supervised baseline)
""")


CELL_DATASET_OVERVIEW_MD = md(r"""## Dataset Overview: The Cross-Domain Challenge

### CWRU (Source Domain — Pretraining)
- **Origin**: Case Western Reserve University Bearing Data Center
- **Sampling rate**: 12kHz (drive end) + 12kHz (fan end) + 12kHz (base)
- **Fault types**: healthy (normal), ball fault, inner race, outer race (4 classes)
- **Windows**: ~2,330 non-overlapping 4096-sample windows
- **Key property**: Lab benchmark, widely used, relatively clean signals

### Paderborn (Target Domain — Transfer Evaluation)
- **Origin**: Paderborn University Bearing Dataset (64kHz)
- **Resampling**: 64kHz → 20kHz (polyphase, anti-aliased) before windowing
- **Classes used**: K001 (healthy), KA01 (outer race), KI01 (inner race)
- **Windows**: ~2,280 windows (20 files/class, 80/20 file-level split)
- **Key property**: Different machine, different sensor, different speed — realistic transfer

### Cross-Domain Challenge
The two datasets differ in:
1. **Sampling rate**: 12kHz vs 64kHz → resampled to 20kHz common standard
2. **Machine**: CWRU motor test rig vs Paderborn accelerated aging rig
3. **Fault mechanism**: CWRU artificially seeded faults vs Paderborn fatigued faults
4. **Frequency content**: Fault harmonics appear at different absolute frequencies

Handcrafted FFT features trained on CWRU achieve F1=1.000 on CWRU but only **F1=0.167** on Paderborn — worse than random chance (0.333 baseline). This is the transfer challenge.
""")


CELL_DATASET_WAVEFORMS = code(r"""# Dataset overview: example waveforms and spectra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import scipy.io
import scipy.signal
import sys

CWRU_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/data/bearings')
PAD_DIR  = Path('/home/sagemaker-user/IndustrialJEPA/datasets/data/paderborn')

def load_cwru_window(bearing_id, start=0, length=4096):
    mat_path = CWRU_DIR / 'raw' / 'cwru' / f'{bearing_id}.mat'
    if not mat_path.exists():
        return None, None
    data = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    for key in data:
        if '_DE_time' in key:
            sig = data[key].astype(np.float32)
            return sig[start:start+length], 12000
    return None, None

def load_paderborn_window(folder, mat_idx=0, start=0, length=None, target_sr=20000):
    folder_path = PAD_DIR / folder
    mat_files = sorted(folder_path.glob('*.mat'))
    if not mat_files or mat_idx >= len(mat_files):
        return None, None
    mat = scipy.io.loadmat(str(mat_files[mat_idx]), squeeze_me=True, simplify_cells=True)
    key = [k for k in mat if not k.startswith('_')][0]
    vib = mat[key]['Y'][6]['Data'].astype(np.float32)
    # Resample 64kHz -> target_sr
    from math import gcd
    g = gcd(64000, target_sr)
    sig = scipy.signal.resample_poly(vib, target_sr // g, 64000 // g).astype(np.float32)
    if length is not None:
        sig = sig[start:start+length]
    return sig, target_sr

# Z-score normalize for display
def znorm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

# Compute PSD
def psd(sig, sr):
    f, p = scipy.signal.welch(sig, fs=sr, nperseg=min(len(sig), 1024))
    return f, 10 * np.log10(p + 1e-12)

# Load example signals
examples = {
    'CWRU\nHealthy':     ('normal_0',    None,  0,     4096, 12000),
    'CWRU\nInner Race':  ('IR007_0',     None,  0,     4096, 12000),
    'CWRU\nBall Fault':  ('B007_0',      None,  0,     4096, 12000),
    'Paderborn\nHealthy (K001)': (None, 'K001', 0,     4096, 20000),
    'Paderborn\nOuter Race (KA01)': (None, 'KA01', 0,  4096, 20000),
    'Paderborn\nInner Race (KI01)': (None, 'KI01', 0,  4096, 20000),
}

fig = plt.figure(figsize=(14, 8))
gs  = gridspec.GridSpec(2, 6, hspace=0.55, wspace=0.35)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336', '#009688']

for col_idx, (label, spec) in enumerate(examples.items()):
    cwru_id, pad_folder, mat_idx, length, sr = spec
    if cwru_id is not None:
        sig, sr = load_cwru_window(cwru_id, length=length)
    else:
        sig, sr = load_paderborn_window(pad_folder, mat_idx=mat_idx, length=length, target_sr=sr)

    if sig is None:
        continue

    sig = znorm(sig[:length])
    t = np.arange(len(sig)) / sr * 1000  # ms
    f, p = psd(sig, sr)

    # Waveform
    ax1 = fig.add_subplot(gs[0, col_idx])
    ax1.plot(t, sig, color=colors[col_idx], linewidth=0.7)
    ax1.set_title(label, fontsize=9, fontweight='bold')
    ax1.set_xlabel('Time (ms)', fontsize=7)
    if col_idx == 0:
        ax1.set_ylabel('Amplitude (z-score)', fontsize=7)
    ax1.tick_params(labelsize=6)
    ax1.set_xlim(0, t[-1])

    # PSD
    ax2 = fig.add_subplot(gs[1, col_idx])
    ax2.plot(f / 1000, p, color=colors[col_idx], linewidth=0.8)
    ax2.set_xlabel('Frequency (kHz)', fontsize=7)
    if col_idx == 0:
        ax2.set_ylabel('PSD (dB/Hz)', fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.set_xlim(0, sr/2/1000)

# Add domain separator
fig.text(0.01, 0.5, 'CWRU (12 kHz)', rotation=90, va='center', fontsize=9,
         color='#1565C0', fontweight='bold', transform=fig.transFigure)

fig.suptitle('Dataset Overview: CWRU (Source) vs Paderborn (Target)\n'
             'Top: Time-domain waveforms | Bottom: Power spectral density',
             fontsize=11, fontweight='bold', y=1.01)

plt.savefig(str(RESULTS_DIR.parent / 'notebooks' / 'plots' / 'dataset_overview.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print("Fig saved: notebooks/plots/dataset_overview.png")
print()
print("Key observation: CWRU and Paderborn signals look superficially similar but have")
print("different fault harmonics. Handcrafted band energies tuned for CWRU don't transfer.")
""")


CELL_RUL_MD = md(r"""## 7. RUL & Prognostics — Honest Negative Result

We tested JEPA embeddings as health indicators for Remaining Useful Life (RUL) estimation
on the IMS bearing run-to-failure dataset (1st_test, 4 bearings, ~983 measurements × 8 time series each).

**Key result: ALL methods fail to beat a constant-mean predictor.**

This is an important negative result that clarifies the scope of JEPA's applicability.
""")


CELL_RUL_CODE = code(r"""# RUL results — loaded from experiment log (no JSON backing for this experiment)
# Source: autoresearch/mechanical_jepa/EXPERIMENT_LOG.md (Exp V5-10) and CONSOLIDATED_RESULTS.md

# ---- Results from log (all methods, IMS 1st_test) ----
rul_results = {
    'Constant baseline\n(predict mean RUL)':  {'rmse': 0.086, 'marker': '*',  'color': '#4CAF50'},
    'RMS → Ridge':                             {'rmse': 0.181, 'marker': 'o',  'color': '#9E9E9E'},
    'Random encoder → Ridge':                 {'rmse': 0.198, 'marker': 's',  'color': '#FF9800'},
    'JEPA-CWRU → Ridge\n(pretrained on CWRU)':{'rmse': 0.202, 'marker': 'D',  'color': '#2196F3'},
    'JEPA-IMS → Ridge\n(pretrained on IMS)':  {'rmse': 0.168, 'marker': '^',  'color': '#F44336'},
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: RMSE bar chart ---
ax = axes[0]
names = list(rul_results.keys())
rmses = [rul_results[k]['rmse'] for k in names]
colors = [rul_results[k]['color'] for k in names]
bars = ax.barh(range(len(names)), rmses, color=colors, edgecolor='k', linewidth=0.5)
ax.axvline(0.086, color='#4CAF50', linestyle='--', linewidth=2,
           label='Constant baseline (reference)')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('RMSE (lower = better)', fontsize=11)
ax.set_title('IMS RUL Regression: All Methods\nFail to Beat Constant Baseline',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
for i, (bar, v) in enumerate(zip(bars, rmses)):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f'{v:.3f}', va='center', fontsize=9)

# --- Right: Root cause explanation ---
ax2 = axes[1]
ax2.axis('off')

ax2.text(0.05, 0.95, 'Root Cause: Label Imbalance', fontsize=13, fontweight='bold',
         transform=ax2.transAxes, va='top', color='#B71C1C')

explanation = """The IMS 1st_test dataset has severe RUL imbalance:

• ~70% of windows have RUL ≈ 1.0 (early life, healthy operation)
• Only ~15% of windows have RUL < 0.5 (degradation phase)
• The constant-mean predictor wins by always predicting RUL ≈ 0.85

Spearman correlation (health indicator quality):
  RMS:             ρ = 0.758 (1st_test), 0.443 (2nd_test)  [GOOD]
  JEPA embedding:  ρ = 0.080 (1st_test), 0.120 (2nd_test)  [POOR]

JEPA embeddings are NOT good health indicators.
The encoder was trained to detect fault TYPE, not fault SEVERITY.

For RUL estimation, you would need:
  1. Pretraining on run-to-failure data (not CWRU fault classes)
  2. A degradation-aware objective (e.g., contrastive temporal ordering)
  3. Domain-specific feature extraction for the degradation signature

This is a clear scope limitation: JEPA → fault detection ✓,
JEPA → prognostics (RUL) ✗
"""

ax2.text(0.05, 0.82, explanation, fontsize=10, transform=ax2.transAxes,
         va='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8))

plt.tight_layout()
plt.savefig(str(RESULTS_DIR.parent / 'notebooks' / 'plots' / 'rul_negative_result.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print("KEY INSIGHT: All methods lose to a trivial constant-mean baseline on IMS RUL.")
print("This is an HONEST NEGATIVE RESULT — JEPA is not a general prognostics tool.")
print()
print("JEPA is good at: cross-domain fault TYPE classification (bearing→bearing)")
print("JEPA is NOT good at: within-domain degradation SEVERITY estimation")
""")


CELL_CROSSCOMP_MD = md(r"""## 8. Cross-Component & Multi-Source Pretraining

Do JEPA representations transfer *across* component types (bearing → gearbox)?

**Hypothesis**: If JEPA learns generic vibration physics, it should transfer across component types.
**Result**: It does NOT. This is expected from a physics perspective.
""")


CELL_CROSSCOMP_CODE = code(r"""# Multi-source pretraining results (from multisource_pretrain.json)
with open(RESULTS_DIR / 'multisource_pretrain.json') as f:
    ms = json.load(f)

s = ms['_summary']

print('=== Multi-Source Pretraining Results (3 seeds, V6 final) ===')
print()
print(f'{"Method":<35} {"CWRU F1":<20} {"Paderborn F1":<22} {"Gear F1"}')
print('-' * 105)

rows = [
    ('CWRU pretrained (bearing)',  s['jepa_v2_cwru_pretrained']),
    ('Gear pretrained (gearbox)',  s['jepa_gear_pretrained']),
    ('Multi-source CWRU+Gear',     s['jepa_multisource']),
    ('Random Init (baseline)',      s['random_init']),
]
for label, v in rows:
    c, p, g = v['cwru_mean'], v['pad_mean'], v['gear_mean']
    cs, ps, gs = v['cwru_std'], v['pad_std'], v['gear_std']
    print(f'{label:<35} {c:.3f} ± {cs:.3f}       {p:.3f} ± {ps:.3f}       {g:.3f} ± {gs:.3f}')

print()
print('Key findings:')
print(f'  1. Gear→bearing transfer (CWRU):  {s["jepa_gear_pretrained"]["cwru_mean"]:.3f} vs random {s["random_init"]["cwru_mean"]:.3f} — NO positive transfer')
print(f'  2. Gear→bearing transfer (Pad):   {s["jepa_gear_pretrained"]["pad_mean"]:.3f} vs CWRU-only {s["jepa_v2_cwru_pretrained"]["pad_mean"]:.3f} — -30.4pp')
print(f'  3. Multi-source Paderborn:        {s["jepa_multisource"]["pad_mean"]:.3f} vs CWRU-only {s["jepa_v2_cwru_pretrained"]["pad_mean"]:.3f} — gear data HURTS bearing transfer')
print(f'  4. Gear domain self-improvement:  {s["jepa_gear_pretrained"]["gear_mean"]:.3f} vs random {s["random_init"]["gear_mean"]:.3f} — +3.8pp (modest but positive)')

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

method_labels = ['CWRU\nPretrained', 'Gear\nPretrained', 'Multi-\nsource', 'Random\nInit']
colors = ['#2196F3', '#FF9800', '#9C27B0', '#9E9E9E']

for ax_idx, (domain, key) in enumerate([('CWRU F1', 'cwru'), ('Paderborn F1', 'pad'), ('Gear F1', 'gear')]):
    ax = axes[ax_idx]
    vals = [s['jepa_v2_cwru_pretrained'][f'{key}_mean'],
            s['jepa_gear_pretrained'][f'{key}_mean'],
            s['jepa_multisource'][f'{key}_mean'],
            s['random_init'][f'{key}_mean']]
    stds = [s['jepa_v2_cwru_pretrained'][f'{key}_std'],
            s['jepa_gear_pretrained'][f'{key}_std'],
            s['jepa_multisource'][f'{key}_std'],
            s['random_init'][f'{key}_std']]

    bars = ax.bar(range(4), vals, yerr=stds, color=colors, edgecolor='k',
                  linewidth=0.5, capsize=5, error_kw={'linewidth': 1.5})
    ax.axhline(vals[3], color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=f'Random init ({vals[3]:.3f})')
    ax.set_xticks(range(4))
    ax.set_xticklabels(method_labels, fontsize=9)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title(domain, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02, f'{v:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

fig.suptitle('Cross-Component Transfer: JEPA Learns Physics-Specific Features\n'
             'Bearing pretraining transfers to bearings, but NOT to gearboxes',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RESULTS_DIR.parent / 'notebooks' / 'plots' / 'cross_component_transfer.png'),
            dpi=150, bbox_inches='tight')
plt.show()

print()
print('Physical interpretation:')
print('  Bearings: impulse-based faults (ball passing outer/inner race)')
print('  Gearboxes: tooth-mesh modulation (cyclic, deterministic pattern)')
print('  These are structurally different signals — JEPA correctly learns distinct features.')
print('  This is EXPECTED physics, not a failure mode.')
""")


CELL_TRIVIAL_BASELINES_MD = md(r"""## 9. Trivial Baselines — Confirming the Transfer Story

To ensure we are not cherry-picking a favorable baseline, we compare against
tree-based classifiers (RF, XGBoost) and a random-init encoder linear probe.
These are the "obvious alternatives" a skeptical reviewer would suggest.
""")


CELL_TRIVIAL_BASELINES_CODE = code(r"""# Load trivial baselines results
with open(RESULTS_DIR / 'trivial_baselines.json') as f:
    tb = json.load(f)

print('=== Trivial Baselines: RF, XGBoost, Random Encoder (3 seeds) ===')
print()
print('Feature set: RMS, peak, crest factor, kurtosis, std, skewness, shape factor,')
print('             impulse factor, spectral entropy, centroid, spread, 4 band energies')
print('             (45 features total, per-channel, 3 channels)')
print()
print(f'{"Method":<35} {"CWRU F1":<22} {"Pad Transfer F1":<22} {"Notes"}')
print('-' * 105)

# From trivial_baselines.json
for method, label, note in [
    ('random_forest',  'Random Forest (200 trees)',      'No retraining on Paderborn'),
    ('xgboost',        'XGBoost (300 trees, depth=6)',   'No retraining on Paderborn'),
    ('random_encoder', 'Random Encoder (lin. probe)',     '5.1M params, random init'),
]:
    if f'_summary_{method}' not in tb:
        continue
    s = tb[f'_summary_{method}']
    print(f'{label:<35} {s["cwru_mean"]:.3f} ± {s["cwru_std"]:.3f}       '
          f'{s["pad_mean"]:.3f} ± {s["pad_std"]:.3f}       {note}')

print()
print('Context (from transfer_baselines_v6_final.json):')
print(f'  JEPA V2 (self-supervised)      CWRU: 0.773 ± 0.018  Pad: 0.900 ± 0.008')
print(f'  CNN Supervised                  CWRU: 1.000 ± 0.000  Pad: 0.987 ± 0.005')
print(f'  Transformer Supervised          CWRU: 0.969 ± 0.026  Pad: 0.673 ± 0.063')
print(f'  Handcrafted + LogReg (transfer) CWRU: 0.999 ± 0.001  Pad: 0.167 ± 0.000')

print()
print('KEY INSIGHT: RF and XGBoost achieve 0.167 Paderborn transfer — SAME as LogReg.')
print('The handcrafted transfer failure is robust to classifier choice.')
print('The problem is the FEATURES, not the classifier.')
print('JEPA (0.900) vs best handcrafted (0.167) = +73.3 percentage points.')

# Visual comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

methods_ordered = [
    ('CNN Supervised',            1.000, 0.000, 0.987, 0.005, '#1565C0'),
    ('JEPA V2 (self-supervised)',  0.773, 0.018, 0.900, 0.008, '#E53935'),
    ('Transformer Supervised',    0.969, 0.026, 0.673, 0.063, '#388E3C'),
    ('Random Forest (handcraft.)',  tb['_summary_random_forest']['cwru_mean'],
                                   tb['_summary_random_forest']['cwru_std'],
                                   tb['_summary_random_forest']['pad_mean'],
                                   tb['_summary_random_forest']['pad_std'], '#FB8C00'),
    ('XGBoost (handcraft.)',       tb['_summary_xgboost']['cwru_mean'],
                                   tb['_summary_xgboost']['cwru_std'],
                                   tb['_summary_xgboost']['pad_mean'],
                                   tb['_summary_xgboost']['pad_std'], '#8E24AA'),
    ('Handcraft. LogReg (transfer)', 0.999, 0.001, 0.167, 0.000, '#757575'),
    ('Random Init',                0.557, 0.012, 0.529, 0.024, '#BDBDBD'),
]

labels = [m[0] for m in methods_ordered]
cwru_means = [m[1] for m in methods_ordered]
cwru_stds  = [m[2] for m in methods_ordered]
pad_means  = [m[3] for m in methods_ordered]
pad_stds   = [m[4] for m in methods_ordered]
clrs       = [m[5] for m in methods_ordered]

for ax, (vals, stds, title) in zip(axes, [
    (cwru_means, cwru_stds,  'CWRU In-Domain F1\n(Source domain — not the main metric)'),
    (pad_means,  pad_stds,   'Paderborn Transfer F1\n(Target domain — the main metric)')
]):
    bars = ax.barh(range(len(labels)), vals, xerr=stds, color=clrs,
                   edgecolor='k', linewidth=0.5, capsize=4, error_kw={'linewidth': 1.5})
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Macro F1', fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1.15)
    if 'Paderborn' in title:
        ax.axvline(0.333, color='gray', linestyle=':', linewidth=1.5,
                   label='Random chance (1/3)')
        ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01, bar.get_y() + bar.get_height()/2,
                f'{v:.3f}', va='center', fontsize=8)

plt.suptitle('Complete Baseline Comparison: RF/XGB confirm handcrafted transfer failure',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(str(RESULTS_DIR.parent / 'notebooks' / 'plots' / 'trivial_baselines_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.show()
""")


# ============================================================
# LOAD EXISTING NOTEBOOK AND INSERT NEW CELLS
# ============================================================

with open(NB_PATH) as f:
    nb = json.load(f)

existing = nb['cells']

# Build new cell list:
# [0] existing title cell
# [NEW] Introduction & Motivation
# [NEW] Architecture
# [NEW] Dataset Overview MD
# [NEW] Dataset waveforms code
# [1] existing setup cell
# [2] existing experimental setup (renumbered to section 5)
# ... rest of existing cells up to cell [20] Discussion
# [NEW] RUL section (between existing 7.Discussion and 8.Reproducibility)
# [NEW] Cross-component section
# [NEW] Trivial baselines section
# [20] existing Discussion (renumber)
# [21] existing Reproducibility

# We'll reorder to:
# 0: title (existing[0])
# 1: intro (NEW)
# 2: architecture (NEW)
# 3: dataset overview md (NEW)
# 4: dataset waveforms (NEW)
# 5: imports (existing[1])
# 6: experimental setup (existing[2])
# 7: Claim 1 md (existing[3])
# 8: Claim 1 code (existing[4])
# 9: Claim 1 fig (existing[5])
# 10: why supervised hurts (existing[6])
# 11: Claim 2 md (existing[7])
# 12: Claim 2 code (existing[8])
# 13: Claim 2 fig (existing[9])
# 14: what this means (existing[10])
# 15: SF-JEPA md (existing[11])
# 16: SF-JEPA code (existing[12])
# 17: SF-JEPA fig (existing[13])
# 18: Complete comparison md (existing[14])
# 19: Complete comparison code (existing[15])
# 20: Complete comparison fig (existing[16])
# 21: Ablation md (existing[17])
# 22: Ablation code (existing[18])
# 23: Ablation fig (existing[19])
# 24: RUL md (NEW)
# 25: RUL code (NEW)
# 26: Cross-component md (NEW)
# 27: Cross-component code (NEW)
# 28: Trivial baselines md (NEW)
# 29: Trivial baselines code (NEW)
# 30: Discussion (existing[20])
# 31: Reproducibility (existing[21])

new_cells = [
    existing[0],      # Title
    CELL_INTRO,       # NEW: Introduction
    CELL_ARCHITECTURE, # NEW: Architecture
    CELL_DATASET_OVERVIEW_MD,  # NEW: Dataset Overview MD
    CELL_DATASET_WAVEFORMS,    # NEW: Dataset waveforms
    existing[1],      # Imports
    existing[2],      # Experimental setup
    existing[3],      # Claim 1 MD
    existing[4],      # Claim 1 code
    existing[5],      # Claim 1 fig
    existing[6],      # Why supervised hurts
    existing[7],      # Claim 2 MD
    existing[8],      # Claim 2 code
    existing[9],      # Claim 2 fig
    existing[10],     # What this means
    existing[11],     # SF-JEPA MD
    existing[12],     # SF-JEPA code
    existing[13],     # SF-JEPA fig
    existing[14],     # Complete comparison MD
    existing[15],     # Complete comparison code
    existing[16],     # Complete comparison fig
    existing[17],     # Ablation MD
    existing[18],     # Ablation code
    existing[19],     # Ablation fig
    CELL_RUL_MD,      # NEW: RUL section MD
    CELL_RUL_CODE,    # NEW: RUL code
    CELL_CROSSCOMP_MD,# NEW: Cross-component MD
    CELL_CROSSCOMP_CODE, # NEW: Cross-component code
    CELL_TRIVIAL_BASELINES_MD,  # NEW: Trivial baselines MD
    CELL_TRIVIAL_BASELINES_CODE, # NEW: Trivial baselines code
    existing[20],     # Discussion
    existing[21],     # Reproducibility
]

# Give unique IDs
import hashlib
for i, cell in enumerate(new_cells):
    src_hash = hashlib.md5((''.join(cell['source']) + str(i)).encode()).hexdigest()[:8]
    cell['id'] = src_hash

nb['cells'] = new_cells

out_path = NB_PATH
with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Written: {out_path}")
print(f"Total cells: {len(new_cells)} (was 22, added {len(new_cells)-22})")

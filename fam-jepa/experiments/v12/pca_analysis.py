"""
PCA analysis of frozen JEPA encoder h_past embeddings.
Shows: PC1 captures degradation (rho vs cycle), PC1 vs H.I. visualization.
This is the visualization for the paper.
"""

import json
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import load_cmapss_subset, N_SENSORS, RUL_CAP
from models import TrajectoryJEPA
from train_utils import DEVICE

PRETRAIN_CKPT = V11_DIR / 'best_pretrain_L1_v2.pt'

print("PCA analysis of JEPA embeddings")
data = load_cmapss_subset('FD001')
train_engines = data['train_engines']
val_engines = data['val_engines']

model = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
    d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
).to(DEVICE)
model.load_state_dict(torch.load(str(PRETRAIN_CKPT), map_location=DEVICE))
model.eval()


def compute_hi_piecewise(T, cap=RUL_CAP):
    hi = np.ones(T, dtype=np.float32)
    degrade_start = max(0, T - cap)
    if degrade_start < T:
        hi[degrade_start:] = np.linspace(1.0, 0.0, T - degrade_start)
    return hi


@torch.no_grad()
def get_embeddings_and_labels(engines, min_cycle=10, max_per_engine=50):
    """Get embeddings sampled from every engine."""
    embs, his, engine_ids, cycles_norm, ruls = [], [], [], [], []

    for eid, seq in engines.items():
        T = seq.shape[0]
        hi = compute_hi_piecewise(T)
        rul_arr = np.minimum(np.arange(T, 0, -1, dtype=np.float32), RUL_CAP)

        # Sample cycles uniformly
        all_cycles = list(range(min_cycle, T+1))
        if len(all_cycles) > max_per_engine:
            step = max(1, len(all_cycles) // max_per_engine)
            sampled = all_cycles[::step][:max_per_engine]
        else:
            sampled = all_cycles

        for c in sampled:
            prefix = seq[:c]
            x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            h = model.encode_past(x, None)
            embs.append(h.cpu().numpy()[0])
            his.append(hi[c-1])
            engine_ids.append(int(eid))
            cycles_norm.append(c / T)  # normalized position
            ruls.append(rul_arr[c-1])

    return (np.stack(embs), np.array(his), np.array(engine_ids),
            np.array(cycles_norm), np.array(ruls))


print("Getting embeddings for training engines...")
X_tr, hi_tr, eids_tr, cnorm_tr, ruls_tr = get_embeddings_and_labels(train_engines, max_per_engine=30)
print(f"Training: {X_tr.shape[0]} samples from {len(train_engines)} engines")

print("Getting embeddings for val engines...")
X_val, hi_val, eids_val, cnorm_val, ruls_val = get_embeddings_and_labels(val_engines, max_per_engine=30)

# PCA
print("Running PCA...")
pca = PCA(n_components=10)
Z_train = pca.fit_transform(X_tr)
Z_val = pca.transform(X_val)

print(f"Explained variance ratios: {pca.explained_variance_ratio_[:5].round(3)}")

# Correlations
print("\nPC1-10 Spearman rho with H.I. (train):")
pc_rhos_hi = []
pc_rhos_rul = []
for i in range(10):
    rho_hi, _ = spearmanr(Z_train[:, i], hi_tr)
    rho_rul, _ = spearmanr(Z_train[:, i], ruls_tr)
    pc_rhos_hi.append(float(rho_hi))
    pc_rhos_rul.append(float(rho_rul))
    if i < 5:
        print(f"  PC{i+1}: rho(H.I.)={rho_hi:.3f}, rho(RUL)={rho_rul:.3f}, var={pca.explained_variance_ratio_[i]:.3f}")

# Figure: 4-panel PCA visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('JEPA Pretrained Encoder: PCA Structure\n(No fine-tuning labels used)', fontsize=13)

# Panel 1: PC1 vs H.I. scatter (colored by engine)
ax = axes[0, 0]
n_colors = min(20, len(train_engines))
sample_engines = list(train_engines.keys())[:n_colors]
color_map = plt.cm.tab20(np.linspace(0, 1, n_colors))
engine_to_color = {eid: color_map[i] for i, eid in enumerate(sample_engines)}

for eid in sample_engines:
    mask = eids_tr == eid
    if not mask.any():
        continue
    color = engine_to_color[eid]
    ax.scatter(Z_train[mask, 0], hi_tr[mask], c=[color], alpha=0.6, s=8)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel('Health Index')
ax.set_title(f'PC1 vs H.I.\nrho={pc_rhos_hi[0]:.3f}')

# Panel 2: PC1 evolution along engine trajectories (3 sample engines)
ax = axes[0, 1]
sample_3 = list(val_engines.keys())[:3]
colors_3 = ['steelblue', 'darkorange', 'green']
for eid, color in zip(sample_3, colors_3):
    mask = eids_val == eid
    if not mask.any():
        continue
    cycles = cnorm_val[mask]
    pc1 = Z_val[mask, 0]
    hi_vals = hi_val[mask]
    sort_idx = np.argsort(cycles)
    ax.plot(cycles[sort_idx], pc1[sort_idx], color=color, linewidth=2, label=f'PC1 eng {eid}', alpha=0.8)
    ax2 = ax.twinx() if eid == sample_3[0] else None
    if eid == sample_3[0]:
        ax2.plot(cycles[sort_idx], hi_vals[sort_idx], color=color, linestyle='--',
                 alpha=0.4, label=f'H.I. eng {eid}')
        ax2.set_ylabel('Health Index', color='gray')

ax.set_xlabel('Normalized cycle position')
ax.set_ylabel('PC1 score')
ax.set_title('PC1 trajectory vs cycle position\n(dashed = H.I. for engine 1)')
ax.legend(fontsize=7)

# Panel 3: PC1 vs PC2 colored by H.I.
ax = axes[1, 0]
sc = ax.scatter(Z_train[:, 0], Z_train[:, 1], c=hi_tr, cmap='RdYlGn',
                alpha=0.3, s=8)
plt.colorbar(sc, ax=ax, label='Health Index')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title('PC1 vs PC2 (colored by H.I.)\nStructure from pretraining only')

# Panel 4: Explained variance + rho bars
ax = axes[1, 1]
x = np.arange(1, 11)
width = 0.35
ax.bar(x - width/2, pca.explained_variance_ratio_[:10], width, label='Explained variance', color='steelblue', alpha=0.8)
ax.bar(x + width/2, np.abs(pc_rhos_hi[:10]), width, label='|rho| with H.I.', color='darkorange', alpha=0.8)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Value')
ax.set_title('PCA: Variance vs Correlation with H.I.\nPC1 dominates H.I. structure')
ax.legend()
ax.set_xticks(x)

plt.tight_layout()
plt.savefig(str(PLOTS_V12 / 'pca_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved pca_analysis.png")

results = {
    "explained_variance_10": pca.explained_variance_ratio_[:10].tolist(),
    "pc_rho_with_hi": pc_rhos_hi[:10],
    "pc_rho_with_rul": pc_rhos_rul[:10],
    "pc1_explained_var": float(pca.explained_variance_ratio_[0]),
    "pc1_rho_hi": float(pc_rhos_hi[0]),
    "pc1_rho_rul": float(pc_rhos_rul[0]),
}
with open(V12_DIR / 'pca_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to {V12_DIR / 'pca_analysis.json'}")
print(f"\nPC1 var={results['pc1_explained_var']:.3f}, rho(H.I.)={results['pc1_rho_hi']:.3f}, rho(RUL)={results['pc1_rho_rul']:.3f}")

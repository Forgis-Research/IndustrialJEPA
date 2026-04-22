"""
Fixed diagnostics: compute h_past at diverse cut points, not just last cycle.
The original error was: use_last_only=True -> all engines at RUL=1, constant variance=0.
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
EXP_DIR = os.path.join(BASE, 'experiments/v11')
PLOTS_DIR = os.path.join(BASE, 'analysis/plots/v11')
sys.path.insert(0, EXP_DIR)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, collate_finetune,
    N_SENSORS, RUL_CAP, get_sensor_cols, compute_rul_labels
)
from models import TrajectoryJEPA, RULProbe

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)

data = load_cmapss_subset('FD001')
model = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=128,
                        n_heads=4, n_layers=2, d_ff=256)
ckpt_path = os.path.join(EXP_DIR, 'best_pretrain_L1.pt')
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

with open(os.path.join(EXP_DIR, 'pretrain_history_L1.json')) as f:
    history = json.load(f)
best_probe_rmse = min(history['probe_rmse'])

log(f"Model loaded. Best probe RMSE from training: {best_probe_rmse:.2f}")


# FIXED: embed at multiple cut points per engine (not just last cycle)
@torch.no_grad()
def get_embeddings_multicut(engines, n_cuts=5, seed=0):
    """Get embeddings at diverse cut points per engine."""
    rng = np.random.default_rng(seed)
    all_h, all_rul = [], []
    for eid, seq in engines.items():
        T = len(seq)
        rul_labels = compute_rul_labels(T, RUL_CAP)
        # Sample cut points from [10, T-1]
        t_min, t_max = 10, T
        if t_max <= t_min:
            continue
        cuts = rng.integers(t_min, t_max, size=min(n_cuts, t_max - t_min))
        for t in cuts:
            past = torch.from_numpy(seq[:t]).unsqueeze(0).to(DEVICE)
            h = model.encode_past(past)
            all_h.append(h.cpu().numpy()[0])
            all_rul.append(float(rul_labels[t - 1]))  # RUL at cycle t
    return np.vstack(all_h), np.array(all_rul)

log("\n--- Computing multi-cut embeddings ---")
all_engines = {**data['train_engines'], **data['val_engines']}
emb, rul_labels = get_embeddings_multicut(all_engines, n_cuts=10)
log(f"Embeddings shape: {emb.shape}")
log(f"RUL range: [{rul_labels.min():.1f}, {rul_labels.max():.1f}]")
log(f"RUL std: {rul_labels.std():.2f}")

emb_std = emb.std(axis=0)
log(f"Embedding std: mean={emb_std.mean():.4f}, min={emb_std.min():.4f}")

# PCA
pca = PCA(n_components=5)
pca_coords = pca.fit_transform(emb)
log(f"Explained variance: {[f'{v:.3f}' for v in pca.explained_variance_ratio_]}")

# Spearman rho
pc_rhos = []
for i in range(5):
    rho, p = spearmanr(pca_coords[:, i], rul_labels)
    pc_rhos.append((i+1, float(rho), float(p)))
    log(f"  PC{i+1} Spearman rho with RUL: {rho:.4f} (p={p:.2e})")

pc1_rho = pc_rhos[0][1]
max_abs_rho = max(abs(r[1]) for r in pc_rhos)
log(f"\nPC1 |rho|: {abs(pc1_rho):.4f}")
log(f"Max component |rho|: {max_abs_rho:.4f}")

# Shuffle test (use val engines only)
log("\n--- Shuffle test ---")
val_emb, val_rul = get_embeddings_multicut(data['val_engines'], n_cuts=10)

# Train quick probe on train embeddings
train_emb, train_rul = get_embeddings_multicut(data['train_engines'], n_cuts=10)
probe = RULProbe(128).to(DEVICE)
te = torch.from_numpy(train_emb).to(DEVICE)
tr = torch.from_numpy(train_rul / RUL_CAP).float().to(DEVICE)
optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
for ep in range(100):
    probe.train()
    idx = torch.randperm(len(te))
    for i in range(0, len(te), 64):
        b = idx[i:i+64]
        pred = probe(te[b])
        loss = F.mse_loss(pred, tr[b])
        optim.zero_grad(); loss.backward(); optim.step()

probe.eval()
ve = torch.from_numpy(val_emb).to(DEVICE)
with torch.no_grad():
    pred_v = probe(ve).cpu().numpy() * RUL_CAP
normal_rmse = float(np.sqrt(np.mean((pred_v - val_rul)**2)))
log(f"  Normal val RMSE: {normal_rmse:.2f}")

# Shuffled test
@torch.no_grad()
def get_embeddings_shuffled_multicut(engines, n_cuts=5, seed=99):
    rng = np.random.default_rng(seed)
    all_h, all_rul = [], []
    for eid, seq in engines.items():
        T = len(seq)
        rul_labels = compute_rul_labels(T, RUL_CAP)
        t_max = T
        if t_max <= 10:
            continue
        cuts = rng.integers(10, t_max, size=min(n_cuts, t_max - 10))
        for t in cuts:
            past = torch.from_numpy(seq[:t]).unsqueeze(0).to(DEVICE)
            # Shuffle time tokens
            perm = torch.randperm(t, device=DEVICE)
            past_shuffled = past[:, perm, :]
            h = model.encode_past(past_shuffled)
            all_h.append(h.cpu().numpy()[0])
            all_rul.append(float(rul_labels[t - 1]))
    return np.vstack(all_h), np.array(all_rul)

val_emb_shuf, val_rul_shuf = get_embeddings_shuffled_multicut(data['val_engines'], n_cuts=10)
ve_shuf = torch.from_numpy(val_emb_shuf).to(DEVICE)
probe.eval()
with torch.no_grad():
    pred_shuf = probe(ve_shuf).cpu().numpy() * RUL_CAP
shuffled_rmse = float(np.sqrt(np.mean((pred_shuf - val_rul_shuf)**2)))
log(f"  Shuffled val RMSE: {shuffled_rmse:.2f}")
log(f"  Temporal signal present: {shuffled_rmse > normal_rmse}")

# Save diagnostics
diag_results = {
    'pc1_rho': float(pc1_rho),
    'max_component_rho': float(max_abs_rho),
    'all_component_rhos': pc_rhos,
    'shuffle_rmse': float(shuffled_rmse),
    'normal_probe_rmse': float(normal_rmse),
    'best_pretrain_probe_rmse': float(best_probe_rmse),
    'embedding_std_mean': float(emb_std.mean()),
    'explained_variance': pca.explained_variance_ratio_.tolist(),
}
with open(os.path.join(EXP_DIR, 'pretrain_diagnostics_fixed.json'), 'w') as f:
    json.dump(diag_results, f, indent=2)
log(f"\nSaved diagnostics to pretrain_diagnostics_fixed.json")

# Plots
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history['loss'], color='steelblue', label='total')
ax1.plot(history['pred_loss'], color='darkorange', linestyle='--', label='pred')
ax1.set_title('Pretraining Loss'); ax1.set_xlabel('Epoch'); ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history['probe_epochs'], history['probe_rmse'], 'g-o', markersize=4)
ax2.axhline(best_probe_rmse, color='red', linestyle='--', label=f'best={best_probe_rmse:.2f}')
ax2.set_title('Linear Probe RMSE over Epochs'); ax2.set_xlabel('Epoch'); ax2.legend()

ax3 = fig.add_subplot(gs[1, 0])
sc = ax3.scatter(pca_coords[:, 0], pca_coords[:, 1],
                  c=rul_labels, cmap='RdYlGn', s=5, alpha=0.5)
plt.colorbar(sc, ax=ax3, label='RUL (cycles)')
ax3.set_title(f'h_past PCA - multi-cut (PC1 rho={pc1_rho:.3f})')
ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(rul_labels, pca_coords[:, 0], alpha=0.3, s=5, color='steelblue')
ax4.set_xlabel('RUL (cycles)'); ax4.set_ylabel('PC1')
ax4.set_title(f'PC1 vs RUL (rho={pc1_rho:.3f})')

plt.suptitle('V11 Pretraining Diagnostics (L=1, FD001, multi-cut)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'pretraining_diagnostics_L1.png'), dpi=120)
plt.close()
log("Saved pretraining_diagnostics_L1.png")

# t-SNE (on subset)
try:
    idx_sub = np.random.choice(len(emb), min(500, len(emb)), replace=False)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_coords = tsne.fit_transform(emb[idx_sub])
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(tsne_coords[:, 0], tsne_coords[:, 1],
                    c=rul_labels[idx_sub], cmap='RdYlGn', s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='RUL (cycles)')
    ax.set_title('t-SNE of h_past (colored by RUL)')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'h_past_tsne.png'), dpi=120)
    plt.close()
    log("Saved h_past_tsne.png")
except Exception as e:
    log(f"t-SNE skipped: {e}")

# h_past PCA
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                c=rul_labels, cmap='RdYlGn', s=5, alpha=0.5)
plt.colorbar(sc, ax=ax, label='RUL (cycles)')
ax.set_title(f'h_past PCA (PC1 rho={pc1_rho:.3f})')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'h_past_pca.png'), dpi=120)
plt.close()

log(f"\nSummary:")
log(f"  PC1 rho: {pc1_rho:.4f}")
log(f"  Max component rho: {max_abs_rho:.4f}")
log(f"  Shuffle signal: {shuffled_rmse:.2f} vs {normal_rmse:.2f}")
log(f"  Best probe RMSE: {best_probe_rmse:.2f}")

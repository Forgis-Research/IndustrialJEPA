"""
Embedding analysis: PCA, t-SNE, Spearman correlation, degradation trajectories.
V9 version. Run after pretraining experiments to generate visualization plots.

Legacy V8 analyze.py code is in v8/analyze.py.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr
from collections import defaultdict

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLOTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

SOURCE_COLORS = {'femto': '#1f77b4', 'xjtu_sy': '#ff7f0e', 'ims': '#2ca02c'}


def get_all_embeddings(model, episodes):
    """Extract all embeddings + RUL + source for a set of episodes."""
    model.eval()
    all_z = []
    all_ruls = []
    all_sources = []

    with torch.no_grad():
        for ep_id, snaps in episodes.items():
            windows = torch.stack([torch.from_numpy(s['window']) for s in snaps], 0)
            windows = windows.unsqueeze(1).to(DEVICE)
            z = model.get_embeddings(windows).cpu().numpy()
            ruls = [s['rul_percent'] for s in snaps]
            src = snaps[0]['source']
            all_z.append(z)
            all_ruls.extend(ruls)
            all_sources.extend([src] * len(snaps))

    return np.vstack(all_z), np.array(all_ruls), all_sources


def compute_embedding_stats(z, ruls):
    """Compute max per-dim Spearman correlation and PC1 correlation."""
    n_dims = z.shape[1]
    max_corr = 0.0
    for d in range(n_dims):
        r, _ = spearmanr(z[:, d], ruls)
        if abs(r) > abs(max_corr):
            max_corr = r

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(z)[:, 0]
    pc1_corr, _ = spearmanr(pc1, ruls)

    return {'max_dim_corr': float(max_corr), 'pc1_corr': float(pc1_corr)}


def plot_pca_tsne_for_model(model, episodes, name, n_tsne_samples=2000):
    """Generate PCA and t-SNE plots for a given encoder."""
    z, ruls, sources = get_all_embeddings(model, episodes)

    print(f"  [{name}] {z.shape[0]} embeddings from {len(episodes)} episodes")
    stats = compute_embedding_stats(z, ruls)
    print(f"  [{name}] max_dim_corr={stats['max_dim_corr']:.3f}, PC1_corr={stats['pc1_corr']:.3f}")

    # PCA
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax1.scatter(coords_pca[:, 0], coords_pca[:, 1], c=ruls, cmap='RdYlGn',
                     alpha=0.6, s=8, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax1, label='RUL%')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title(f'PCA by RUL — {name}\nPC1 corr={stats["pc1_corr"]:.3f}')
    ax1.grid(True, alpha=0.2)

    for src in set(sources):
        mask = np.array([s == src for s in sources])
        color = SOURCE_COLORS.get(src, 'gray')
        ax2.scatter(coords_pca[mask, 0], coords_pca[mask, 1], c=color,
                    alpha=0.5, s=8, label=src)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2')
    ax2.set_title(f'PCA by Source — {name}')
    ax2.legend(fontsize=9, markerscale=2)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'pca_{name}_by_rul_and_source.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # t-SNE (subsample if needed)
    if len(z) > n_tsne_samples:
        idx = np.random.choice(len(z), n_tsne_samples, replace=False)
        z_s, ruls_s, sources_s = z[idx], ruls[idx], [sources[i] for i in idx]
    else:
        z_s, ruls_s, sources_s = z, ruls, sources

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=500)
    coords_tsne = tsne.fit_transform(z_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    sc = ax1.scatter(coords_tsne[:, 0], coords_tsne[:, 1], c=ruls_s, cmap='RdYlGn',
                     alpha=0.6, s=8, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax1, label='RUL%')
    ax1.set_title(f't-SNE by RUL — {name}')
    ax1.grid(True, alpha=0.2)

    for src in set(sources_s):
        mask = np.array([s == src for s in sources_s])
        color = SOURCE_COLORS.get(src, 'gray')
        ax2.scatter(coords_tsne[mask, 0], coords_tsne[mask, 1], c=color,
                    alpha=0.5, s=8, label=src)
    ax2.set_title(f't-SNE by Source — {name}')
    ax2.legend(fontsize=9, markerscale=2)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'tsne_{name}_by_rul_and_source.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return stats, coords_pca


def plot_degradation_trajectories(model, episodes, episode_ids, n_show=7,
                                   save_prefix='degradation'):
    """Plot PC1 embedding trajectory over normalized episode time."""
    z_by_ep = {}
    for ep_id in episode_ids:
        snaps = episodes[ep_id]
        windows = torch.stack([torch.from_numpy(s['window']) for s in snaps], 0)
        windows = windows.unsqueeze(1).to(DEVICE)
        model.eval()
        with torch.no_grad():
            z = model.get_embeddings(windows).cpu().numpy()
        z_by_ep[ep_id] = z

    all_z = np.vstack(list(z_by_ep.values()))
    pca = PCA(n_components=1)
    pca.fit(all_z)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, min(n_show, len(episode_ids))))

    for i, ep_id in enumerate(episode_ids[:n_show]):
        snaps = episodes[ep_id]
        z = z_by_ep[ep_id]
        pc1 = pca.transform(z)[:, 0]
        t = np.arange(len(pc1)) / len(pc1)
        src = snaps[0]['source']
        ax.plot(t, pc1, color=colors[i], alpha=0.8, linewidth=1.5,
                label=f'Ep{i+1} ({src}, n={len(snaps)})')

    ax.set_xlabel('Normalized time (0=start, 1=failure)')
    ax.set_ylabel('PC1')
    ax.set_title('Degradation Trajectories: JEPA PC1 Over Time')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_trajectories.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_deviation_trajectories(model, episodes, episode_ids, n_show=7,
                                 K=10, save_prefix='deviation'):
    """Plot ||z_t - z_baseline|| over normalized episode time."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, min(n_show, len(episode_ids))))

    for i, ep_id in enumerate(episode_ids[:n_show]):
        snaps = episodes[ep_id]
        T = len(snaps)
        windows = torch.stack([torch.from_numpy(s['window']) for s in snaps], 0)
        windows = windows.unsqueeze(1).to(DEVICE)
        model.eval()
        with torch.no_grad():
            z = model.get_embeddings(windows).cpu().numpy()

        z_baseline = z[:min(K, T)].mean(axis=0, keepdims=True)
        deviation_norm = np.linalg.norm(z - z_baseline, axis=1)
        t = np.arange(T) / T
        src = snaps[0]['source']

        ax.plot(t, deviation_norm, color=colors[i], alpha=0.8, linewidth=1.5,
                label=f'Ep{i+1} ({src})')

    ax.set_xlabel('Normalized time (0=start, 1=failure)')
    ax.set_ylabel('||z_t - z_baseline||_2')
    ax.set_title(f'Deviation from Healthy Baseline (K={K} snapshots)\nLarger value = more degraded')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_norm.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def run_all_embedding_analysis(ckpt_dir, episodes):
    """Run complete embedding analysis for all V9 encoders."""
    from jepa_v8 import MechanicalJEPAV8

    models = {}
    encoder_names = ['jepa_v9_all_8', 'jepa_v9_compatible_6', 'jepa_v9_bearing_rul_3',
                     'jepa_v9_block_masking']
    for enc_name in encoder_names:
        ckpt_path = os.path.join(ckpt_dir, f'{enc_name}.pt')
        if os.path.exists(ckpt_path):
            model = MechanicalJEPAV8().to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            models[enc_name] = model
            print(f"Loaded: {enc_name}")

    # Random baseline
    random_model = MechanicalJEPAV8().to(DEVICE)
    random_model.eval()
    models['random'] = random_model

    ep_ids = list(episodes.keys())
    all_stats = {}

    for name, model in models.items():
        print(f"\nGenerating plots for {name}...")
        try:
            stats, _ = plot_pca_tsne_for_model(model, episodes, name)
            all_stats[name] = stats
        except Exception as e:
            print(f"  Error: {e}")

    if 'jepa_v9_compatible_6' in models:
        print("\nGenerating degradation + deviation trajectories...")
        try:
            plot_degradation_trajectories(
                models['jepa_v9_compatible_6'], episodes, ep_ids)
            plot_deviation_trajectories(
                models['jepa_v9_compatible_6'], episodes, ep_ids)
        except Exception as e:
            print(f"  Error: {e}")

    # Save stats summary
    with open(os.path.join(PLOTS_DIR, 'embedding_analysis_summary.json'), 'w') as f:
        json.dump(all_stats, f, indent=2)

    print("\nAll embedding analysis complete.")
    return models, all_stats


if __name__ == '__main__':
    sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
    sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

    from experiments.v9.run_experiments import load_rul_episodes_all
    print("Loading episodes...")
    episodes = load_rul_episodes_all(verbose=True)

    ckpt_dir = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/checkpoints'
    run_all_embedding_analysis(ckpt_dir, episodes)

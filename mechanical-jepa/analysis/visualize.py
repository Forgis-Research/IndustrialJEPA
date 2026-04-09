"""
Visualization utilities for JEPA embedding analysis.
Produces PCA, t-SNE, degradation trajectory, and uncertainty calibration plots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import spearmanr

PLOTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

SOURCE_COLORS = {
    'femto': '#1f77b4',
    'xjtu_sy': '#ff7f0e',
    'ims': '#2ca02c',
    'random': '#7f7f7f',
}


def plot_pca_embeddings(embeddings, ruls, sources, save_prefix='pca', title_suffix=''):
    """PCA plot: color by RUL and by source."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Color by RUL
    sc = ax1.scatter(coords[:, 0], coords[:, 1], c=ruls, cmap='RdYlGn',
                     alpha=0.6, s=10, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax1, label='RUL%')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title(f'PCA Colored by RUL{title_suffix}')
    ax1.grid(True, alpha=0.2)

    # Color by source
    unique_sources = list(set(sources))
    for src in unique_sources:
        mask = np.array([s == src for s in sources])
        color = SOURCE_COLORS.get(src, 'gray')
        ax2.scatter(coords[mask, 0], coords[mask, 1], c=color,
                    alpha=0.5, s=10, label=src)
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax2.set_title(f'PCA Colored by Source{title_suffix}')
    ax2.legend(fontsize=9, markerscale=2)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_by_rul_and_source.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return coords, pca


def plot_tsne_embeddings(embeddings, ruls, sources, save_prefix='tsne', title_suffix='',
                         perplexity=30, n_samples=2000):
    """t-SNE plot: color by RUL and by source."""
    # Subsample if too many points
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_s = embeddings[idx]
        ruls_s = np.array(ruls)[idx]
        sources_s = [sources[i] for i in idx]
    else:
        embeddings_s = embeddings
        ruls_s = np.array(ruls)
        sources_s = sources

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=500)
    coords = tsne.fit_transform(embeddings_s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sc = ax1.scatter(coords[:, 0], coords[:, 1], c=ruls_s, cmap='RdYlGn',
                     alpha=0.6, s=10, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax1, label='RUL%')
    ax1.set_title(f't-SNE Colored by RUL{title_suffix}')
    ax1.grid(True, alpha=0.2)

    unique_sources = list(set(sources_s))
    for src in unique_sources:
        mask = np.array([s == src for s in sources_s])
        color = SOURCE_COLORS.get(src, 'gray')
        ax2.scatter(coords[mask, 0], coords[mask, 1], c=color,
                    alpha=0.5, s=10, label=src)
    ax2.set_title(f't-SNE Colored by Source{title_suffix}')
    ax2.legend(fontsize=9, markerscale=2)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_by_rul_and_source.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    return coords


def plot_degradation_trajectories(embeddings_by_episode, episode_ids, n_episodes=5,
                                  save_prefix='degradation'):
    """Plot PC1 over time for multiple episodes."""
    pca = PCA(n_components=1)
    all_embeds = np.vstack([embeddings_by_episode[ep] for ep in episode_ids])
    pca.fit(all_embeds)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, min(n_episodes, len(episode_ids))))

    for i, ep_id in enumerate(episode_ids[:n_episodes]):
        z = embeddings_by_episode[ep_id]
        pc1 = pca.transform(z)[:, 0]
        t = np.arange(len(pc1)) / len(pc1)
        ax.plot(t, pc1, color=colors[i], alpha=0.8, linewidth=1.5, label=f'Episode {i+1}')

    ax.set_xlabel('Normalized time (0=start, 1=failure)')
    ax.set_ylabel('PC1 (embedding drift direction)')
    ax.set_title('Degradation Trajectories: JEPA PC1 Over Episode Time')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_trajectories.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_deviation_trajectories(episodes_dict, model, test_eps, n_episodes=5,
                                 save_prefix='deviation'):
    """Plot ||z_t - z_baseline|| over time for test episodes."""
    import torch
    device = next(model.parameters()).device

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.plasma(np.linspace(0, 1, min(n_episodes, len(test_eps))))

    for i, ep_id in enumerate(test_eps[:n_episodes]):
        snaps = episodes_dict[ep_id]
        T = len(snaps)

        # Encode all snapshots
        windows = torch.stack([torch.from_numpy(s['window']) for s in snaps], 0)
        windows = windows.unsqueeze(1).to(device)
        model.eval()
        with torch.no_grad():
            z = model.get_embeddings(windows).cpu().numpy()

        # Compute deviation from healthy baseline (first K=10 snapshots)
        K = min(10, T)
        z_baseline = z[:K].mean(axis=0, keepdims=True)
        deviation_norm = np.linalg.norm(z - z_baseline, axis=1)
        t = np.arange(T) / T

        ruls = np.array([s['rul_percent'] for s in snaps])
        ax.plot(t, deviation_norm, color=colors[i], alpha=0.8, linewidth=1.5,
                label=f'Ep{i+1} ({snaps[0]["source"]})')

    ax.set_xlabel('Normalized time (0=start, 1=failure)')
    ax.set_ylabel('||z_t - z_baseline||_2')
    ax.set_title('Deviation from Healthy Baseline Over Time\n(larger = more degraded)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_norm.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_correlation_heatmap(embeddings, ruls, feature_names=None, top_k=20,
                              save_prefix='correlation'):
    """Correlation of top embedding dims with RUL."""
    # Per-dim Spearman correlation
    n_dims = embeddings.shape[1]
    corrs = []
    for d in range(n_dims):
        r, _ = spearmanr(embeddings[:, d], ruls)
        corrs.append(r)
    corrs = np.array(corrs)

    # Top-k most correlated dims
    top_idx = np.argsort(np.abs(corrs))[-top_k:][::-1]
    top_corrs = corrs[top_idx]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ['green' if c > 0 else 'red' for c in top_corrs]
    ax.bar(range(len(top_idx)), np.abs(top_corrs), color=colors, alpha=0.7)
    ax.set_xticks(range(len(top_idx)))
    ax.set_xticklabels([f'dim{d}' for d in top_idx], rotation=45, fontsize=8)
    ax.set_ylabel('|Spearman corr with RUL|')
    ax.set_title(f'Top {top_k} Embedding Dimensions Correlated with RUL\n'
                 f'(max corr = {np.max(np.abs(corrs)):.3f})')
    ax.axhline(y=0.3, color='orange', linestyle='--', label='r=0.3 threshold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_uncertainty_calibration(mu_all, sigma_all, targets_all,
                                 save_prefix='uncertainty_calibration'):
    """Calibration plot: predicted vs actual uncertainty."""
    mu = np.array(mu_all)
    sigma = np.array(sigma_all)
    targets = np.array(targets_all)
    errors = np.abs(mu - targets)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Predicted sigma vs actual error
    ax1.scatter(sigma, errors, alpha=0.2, s=5, color='steelblue')
    max_val = max(sigma.max(), errors.max())
    ax1.plot([0, max_val], [0, max_val], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Predicted sigma')
    ax1.set_ylabel('Actual |error|')
    ax1.set_title('Uncertainty Calibration')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Coverage at different confidence levels
    alphas = np.linspace(0.5, 0.99, 20)
    coverages = []
    for alpha in alphas:
        z_score = np.abs(np.percentile(np.random.randn(10000), [(1 - alpha) / 2 * 100,
                                                                  (1 + alpha) / 2 * 100]))
        z = z_score[1]
        in_interval = np.abs(targets - mu) <= z * sigma
        coverages.append(float(in_interval.mean()))

    ax2.plot(alphas * 100, [c * 100 for c in coverages], 'b-o', markersize=4, label='Actual coverage')
    ax2.plot([50, 99], [50, 99], 'r--', label='Ideal (perfect calibration)')
    ax2.set_xlabel('Confidence level (%)')
    ax2.set_ylabel('Actual coverage (%)')
    ax2.set_title('Coverage vs Confidence Level')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'{save_prefix}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

"""
Phase 7/8: Deep Analysis of V8 Results.

Analyzes:
1. Embedding quality: correlation with health indicators
2. Latent trajectory visualization (PCA)
3. Per-episode breakdown
4. When JEPA helps vs when it doesn't
5. Comparison with published SOTA
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Optional

sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

RESULTS_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/results'
PLOT_DIR = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/notebooks/plots'
os.makedirs(PLOT_DIR, exist_ok=True)


def load_results() -> Dict:
    results = {}
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('.json'):
            with open(os.path.join(RESULTS_DIR, fname)) as f:
                results[fname.replace('.json', '')] = json.load(f)
    return results


def analyze_embedding_quality():
    """Analyze JEPA embedding quality on all FEMTO+XJTU-SY episodes."""
    import torch
    from data_pipeline import load_rul_episodes, compute_handcrafted_features_per_snapshot
    from data_pipeline import compute_envelope_rms_per_snapshot
    from jepa_v8 import MechanicalJEPAV8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_path = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/checkpoints/jepa_v8_best.pt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MechanicalJEPAV8().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    random_model = MechanicalJEPAV8().to(device)
    random_model.eval()

    episodes = load_rul_episodes(['femto', 'xjtu_sy'], verbose=False)
    print(f"Analyzing {len(episodes)} episodes...")

    all_z = []; all_z_rand = []; all_rul = []; all_hc = []; all_env = []
    all_elapsed = []; all_sources = []; all_ep_ids = []

    for ep_id, snaps in episodes.items():
        windows = np.stack([s['window'] for s in snaps])
        X = torch.tensor(windows).unsqueeze(1).to(device)
        with torch.no_grad():
            z = model.get_embeddings(X).cpu().numpy()
            z_rand = random_model.get_embeddings(X).cpu().numpy()

        hc = compute_handcrafted_features_per_snapshot(snaps)
        env = compute_envelope_rms_per_snapshot(snaps)
        rul = np.array([s['rul_percent'] for s in snaps])
        elapsed = np.array([s['episode_position_norm'] for s in snaps])
        src = snaps[0]['source']

        all_z.append(z); all_z_rand.append(z_rand)
        all_rul.append(rul); all_hc.append(hc); all_env.append(env)
        all_elapsed.append(elapsed)
        all_sources.extend([src] * len(snaps))
        all_ep_ids.extend([ep_id] * len(snaps))

    Z = np.concatenate(all_z)
    Z_rand = np.concatenate(all_z_rand)
    RUL = np.concatenate(all_rul)
    HC = np.concatenate(all_hc)
    ENV = np.concatenate(all_env)
    EL = np.concatenate(all_elapsed)
    SRCS = np.array(all_sources)

    print(f"\n=== Embedding Quality Report ===")
    print(f"Total snapshots: {len(RUL)}")

    # Per-dimension correlation
    z_corrs = np.array([abs(spearmanr(Z[:, i], RUL)[0]) for i in range(Z.shape[1])])
    zr_corrs = np.array([abs(spearmanr(Z_rand[:, i], RUL)[0]) for i in range(Z_rand.shape[1])])
    hc_corrs = np.array([abs(spearmanr(HC[:, j], RUL)[0]) for j in range(HC.shape[1])])
    env_corr = abs(spearmanr(ENV, RUL)[0])

    print(f"\nPer-dimension correlations with RUL:")
    print(f"  JEPA (trained): max={z_corrs.max():.3f}, mean={z_corrs.mean():.3f}, "
          f"dims>0.1={np.sum(z_corrs>0.1)}/{len(z_corrs)}")
    print(f"  JEPA (random):  max={zr_corrs.max():.3f}, mean={zr_corrs.mean():.3f}")
    print(f"  Handcrafted:   max={hc_corrs.max():.3f}, mean={hc_corrs.mean():.3f}")
    print(f"  Envelope RMS:  corr={env_corr:.3f}")

    # Linear probing
    print(f"\nLinear probe (Ridge regression, in-sample):")
    for name, feat in [('JEPA', Z), ('Random JEPA', Z_rand),
                        ('Handcrafted', HC), ('Elapsed time', EL.reshape(-1, 1))]:
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        Xs = sc.fit_transform(feat)
        lr = Ridge(alpha=1.0).fit(Xs, RUL)
        pred = lr.predict(Xs).clip(0, 1)
        rmse = float(mean_squared_error(RUL, pred) ** 0.5)
        r, _ = spearmanr(pred, RUL)
        print(f"  {name:<20}: RMSE={rmse:.4f}, Spearman={r:.3f}")

    # PCA analysis
    pca = PCA(n_components=20)
    Z_pca = pca.fit_transform(Z)
    exp_var = pca.explained_variance_ratio_

    print(f"\nPCA explained variance:")
    print(f"  PC1-5: {exp_var[:5].sum()*100:.1f}% total")
    print(f"  PC1: {exp_var[0]*100:.1f}%")

    # Correlation of PCA components with RUL
    print(f"\nTop PCA component correlations with RUL:")
    pca_corrs = [(i, abs(spearmanr(Z_pca[:, i], RUL)[0])) for i in range(20)]
    pca_corrs.sort(key=lambda x: -x[1])
    for rank, (i, r) in enumerate(pca_corrs[:5]):
        print(f"  PC{i+1}: r={r:.3f} (explains {exp_var[i]*100:.1f}% variance)")

    # Spectral entropy (effective rank)
    Z_c = Z - Z.mean(0)
    Z_c = Z_c / (Z_c.std(0) + 1e-10)
    sv = np.linalg.svd(Z_c, compute_uv=False)
    sv_norm = sv / sv.sum()
    entropy = -(sv_norm * np.log(sv_norm + 1e-10)).sum()
    print(f"\nEmbedding effective rank (spectral entropy): {entropy:.2f} / {np.log(256):.2f}")
    print(f"  (1.0 = one dim, {np.log(256):.2f} = perfectly uniform)")

    # Source-wise embedding drift analysis
    print(f"\nEmbedding drift across episode (healthy→faulty):")
    for ep_id, snaps in episodes.items():
        src = snaps[0]['source']
        windows = np.stack([s['window'] for s in snaps])
        X = torch.tensor(windows).unsqueeze(1).to(device)
        with torch.no_grad():
            z_ep = model.get_embeddings(X).cpu().numpy()
        n = len(z_ep)
        early = z_ep[:n//3]
        late = z_ep[-n//3:]
        drift = np.linalg.norm(early.mean(0) - late.mean(0))
        print(f"  {ep_id} ({src}): drift={drift:.4f}")

    return {
        'z_corrs_max': float(z_corrs.max()),
        'z_corrs_mean': float(z_corrs.mean()),
        'zr_corrs_max': float(zr_corrs.max()),
        'hc_corrs_max': float(hc_corrs.max()),
        'spectral_entropy': float(entropy),
    }


def plot_pretrain_history():
    """Plot pretraining loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, fname, title in [
        (axes[0], 'pretrain_history.json', 'Raw JEPA V8 Pretraining'),
        (axes[1], 'pretrain_fft.json', 'FFT JEPA V8 Pretraining'),
    ]:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        hist = data.get('history', [])
        epochs = [h['epoch'] for h in hist]
        train_loss = [h['train_loss'] for h in hist]
        val_loss = [h['val_loss'] for h in hist]
        pred_var = [h['pred_var'] for h in hist]

        ax.plot(epochs, train_loss, label='Train loss', color='blue')
        ax.plot(epochs, val_loss, label='Val loss', color='orange')
        ax2 = ax.twinx()
        ax2.plot(epochs, pred_var, label='Pred variance', color='green', linestyle='--', alpha=0.7)
        ax2.axhline(y=0.01, color='red', linestyle=':', alpha=0.5, label='Collapse threshold')
        ax2.set_ylabel('Prediction Variance', color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('L1 Loss')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'v8_pretrain_history.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'v8_pretrain_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: pretrain history plots")


def plot_results_comparison():
    """Plot RUL results comparison."""
    with open(os.path.join(RESULTS_DIR, 'rul_baselines_linear.json')) as f:
        linear_data = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'rul_baselines_piecewise.json')) as f:
        piecewise_data = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'final_comparison.json')) as f:
        final_data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Linear RUL ---
    ax = axes[0]
    methods_linear = {
        'Constant Mean': (0.2898, 0.0, ''),
        'Elapsed Time': (0.2240, 0.0, ''),
        'Envelope+LSTM': (0.2870, 0.0008, ''),
        'Random JEPA+LSTM': (0.2452, 0.0189, ''),
        'HC+LSTM': (0.1818, 0.0039, ''),
        'CNN-LSTM (E2E)': (0.1949, 0.0048, ''),
        'CNN-GRU-MHA': (0.1850, 0.0051, ''),
        'JEPA+LSTM (opt.)': (0.1886, 0.0154, 'JEPA'),
        'HC+MLP': (0.0851, 0.0042, ''),
        'Transformer+HC': (0.0697, 0.0055, ''),
    }

    colors = ['royalblue' if v[2] == 'JEPA' else 'steelblue' for v in methods_linear.values()]
    names = list(methods_linear.keys())
    rmses = [v[0] for v in methods_linear.values()]
    stds = [v[1] for v in methods_linear.values()]

    bars = ax.barh(names, rmses, xerr=stds, color=colors, alpha=0.8, capsize=4)
    ax.axvline(x=0.2240, color='red', linestyle='--', alpha=0.7, label='Elapsed time baseline')
    ax.set_xlabel('RMSE (lower is better)')
    ax.set_title('Linear RUL Labels')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    # Highlight JEPA bar
    for bar, name in zip(bars, names):
        if 'JEPA' in name and 'Random' not in name:
            bar.set_edgecolor('darkblue')
            bar.set_linewidth(2)

    # --- Piecewise RUL ---
    ax = axes[1]
    methods_piecewise = {
        'Constant Mean': (0.3346, 0.0, ''),
        'Elapsed Time': (0.3367, 0.0, ''),
        'Envelope+LSTM': (0.3437, 0.0087, ''),
        'Random JEPA+LSTM': (0.3000, 0.0338, ''),
        'HC+LSTM': (0.2727, 0.0223, ''),
        'CNN-LSTM (E2E)': (0.2996, 0.0079, ''),
        'CNN-GRU-MHA': (0.3449, 0.0063, ''),
        'JEPA+LSTM': (0.3223, 0.0091, 'JEPA'),
        'HC+MLP': (0.2304, 0.0076, ''),
        'Transformer+HC': (0.1136, 0.0260, ''),
    }

    colors_pw = ['royalblue' if v[2] == 'JEPA' else 'steelblue' for v in methods_piecewise.values()]
    names_pw = list(methods_piecewise.keys())
    rmses_pw = [v[0] for v in methods_piecewise.values()]
    stds_pw = [v[1] for v in methods_piecewise.values()]

    bars_pw = ax.barh(names_pw, rmses_pw, xerr=stds_pw, color=colors_pw, alpha=0.8, capsize=4)
    ax.axvline(x=0.3367, color='red', linestyle='--', alpha=0.7, label='Elapsed time baseline')
    ax.set_xlabel('RMSE (lower is better)')
    ax.set_title('Piecewise-Linear RUL Labels')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('JEPA V8: RUL Prediction Results\n(23 episodes: 16 FEMTO + 7 XJTU-SY, 5-seed average)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'v8_rul_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'v8_rul_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: RUL comparison plot")


def plot_latent_trajectories():
    """Plot JEPA embedding trajectories for FEMTO episodes."""
    import torch
    from data_pipeline import load_rul_episodes
    from jepa_v8 import MechanicalJEPAV8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/checkpoints/jepa_v8_best.pt',
                       map_location=device, weights_only=False)
    model = MechanicalJEPAV8().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    episodes = load_rul_episodes(['femto'], verbose=False)
    # Only use a few episodes for clarity
    ep_subset = sorted(episodes.keys())[:6]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    all_z = []
    for ep_id in ep_subset:
        snaps = episodes[ep_id]
        windows = np.stack([s['window'] for s in snaps])
        X = torch.tensor(windows).unsqueeze(1).to(device)
        with torch.no_grad():
            z = model.get_embeddings(X).cpu().numpy()
        all_z.append(z)

    # Fit PCA on all data
    Z_all = np.concatenate(all_z)
    pca = PCA(n_components=2)
    pca.fit(Z_all)

    for idx, (ep_id, z) in enumerate(zip(ep_subset, all_z)):
        ax = axes[idx]
        z_2d = pca.transform(z)
        T = len(z_2d)
        rul = np.array([s['rul_percent'] for s in episodes[ep_id]])

        # Color by RUL (1=healthy, 0=failed)
        sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=rul, cmap='RdYlGn',
                        s=5, alpha=0.7, vmin=0, vmax=1)
        # Add arrow for trajectory direction
        ax.annotate('', xy=z_2d[-1], xytext=z_2d[0],
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.set_title(f'{ep_id.split("_")[1:3]}\n(T={T} snapshots)', fontsize=8)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=7)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=7)
        plt.colorbar(sc, ax=ax, label='RUL%', fraction=0.046)

    plt.suptitle('JEPA V8 Latent Trajectories (FEMTO Episodes)\nColored by RUL% (green=healthy, red=failed)',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'v8_latent_trajectories.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'v8_latent_trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: latent trajectories plot")


def plot_cross_dataset():
    """Plot cross-dataset transfer results."""
    with open(os.path.join(RESULTS_DIR, 'cross_dataset_linear.json')) as f:
        data = json.load(f)

    configs = ['femto_to_femto', 'xjtu_to_xjtu', 'femto_to_xjtu', 'xjtu_to_femto']
    labels = ['FEMTO→FEMTO\n(within)', 'XJTU→XJTU\n(within)', 'FEMTO→XJTU\n(cross)', 'XJTU→FEMTO\n(cross)']
    methods = ['elapsed_time', 'handcrafted_lstm', 'jepa_lstm']
    method_labels = ['Elapsed Time', 'HC+LSTM', 'JEPA+LSTM']
    colors = ['gray', 'steelblue', 'royalblue']

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(configs))
    width = 0.25

    for i, (m, ml, c) in enumerate(zip(methods, method_labels, colors)):
        rmses = []
        for cfg in configs:
            if cfg in data and m in data[cfg]:
                r = data[cfg][m]
                if isinstance(r, dict):
                    rmse = r.get('rmse_mean', r.get('rmse', float('nan')))
                else:
                    rmse = float('nan')
            else:
                rmse = float('nan')
            rmses.append(rmse)
        ax.bar(x + i * width, rmses, width, label=ml, color=c, alpha=0.8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Cross-Dataset Transfer: RUL Prediction (Linear Labels)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axvline(x=1.5, color='black', linestyle='--', alpha=0.3)
    ax.text(0.5, ax.get_ylim()[1] * 0.95, 'Within-dataset', ha='center', fontsize=9, style='italic')
    ax.text(2.5, ax.get_ylim()[1] * 0.95, 'Cross-dataset', ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'v8_cross_dataset.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(PLOT_DIR, 'v8_cross_dataset.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cross-dataset plot")


def print_sota_comparison():
    """Compare with published SOTA."""
    print("\n=== COMPARISON WITH PUBLISHED SOTA ===")
    print()
    print("Reference: CNN-GRU-MHA on FEMTO (Applied Sciences 2024)")
    print("  Published nRMSE = 0.044 (FEMTO only, time-based RUL, specific train/test split)")
    print()
    print("Our implementation on same data but:")
    print("  - Mixed FEMTO+XJTU-SY dataset (23 episodes)")
    print("  - RUL% formulation (not absolute time)")
    print("  - 75/25 episode-based split")
    print()
    print("Our CNN-GRU-MHA RMSE:")
    with open('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/results/rul_baselines_linear.json') as f:
        data = json.load(f)
    cnn_gru = data['results']['cnn_gru_mha']
    print(f"  RMSE = {cnn_gru['rmse_mean']:.4f} +/- {cnn_gru['rmse_std']:.4f}")
    print()
    print("Discrepancy expected due to:")
    print("  1. Different evaluation protocol (RUL% vs absolute time)")
    print("  2. Mixed dataset (not FEMTO-only)")
    print("  3. Different training/test split")
    print("  4. Our implementation may not exactly replicate theirs")
    print()
    print("On FEMTO-only within-source evaluation:")
    with open('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/v8/results/cross_dataset_linear.json') as f:
        data = json.load(f)
    if 'femto_to_femto' in data and 'jepa_lstm' in data['femto_to_femto']:
        jepa = data['femto_to_femto']['jepa_lstm']
        hc = data['femto_to_femto']['handcrafted_lstm']
        et = data['femto_to_femto']['elapsed_time']
        jepa_rmse = jepa.get('rmse_mean', jepa.get('rmse', 'N/A'))
        hc_rmse = hc.get('rmse_mean', hc.get('rmse', 'N/A'))
        et_rmse = et.get('rmse_mean', et.get('rmse', 'N/A'))
        print(f"  Elapsed time: {et_rmse:.4f}")
        print(f"  HC+LSTM:      {hc_rmse:.4f}")
        print(f"  JEPA+LSTM:    {jepa_rmse:.4f}")


if __name__ == '__main__':
    print("=== V8 Analysis ===\n")

    print("--- Loading results ---")
    results = load_results()
    print(f"Loaded: {list(results.keys())}")

    print("\n--- Embedding quality analysis ---")
    emb_quality = analyze_embedding_quality()

    print("\n--- Generating plots ---")
    plot_pretrain_history()
    plot_results_comparison()
    plot_latent_trajectories()
    plot_cross_dataset()

    print_sota_comparison()

    # Save embedding quality report
    with open(os.path.join(RESULTS_DIR, 'embedding_quality.json'), 'w') as f:
        json.dump(emb_quality, f, indent=2)
    print(f"\nEmbedding quality report saved.")

    print("\n=== Analysis complete ===")
    print(f"Plots saved to: {PLOT_DIR}")

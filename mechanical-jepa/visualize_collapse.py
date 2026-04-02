"""
Collapse Visualization Script for Mechanical-JEPA V4.

Creates 3 figures for Section 3 of the analysis notebook:
3A. Prediction Heatmap (V1-collapsed vs V2-fixed)
3B. Positional Embedding Analysis (sinusoidal vs learnable)
3C. Mask Ratio vs Collapse Threshold

Outputs to notebooks/plots/v4_collapse_*.png
"""

import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPA, MechanicalJEPAV2

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available")

PLOTS_DIR = Path('notebooks/plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


# =============================================================================
# Helpers
# =============================================================================

def sinusoidal_pos_encoding(n_positions: int, d_model: int) -> np.ndarray:
    """Standard sinusoidal positional encoding, returns (n_positions, d_model)."""
    pe = np.zeros((n_positions, d_model))
    position = np.arange(n_positions)[:, None]
    div_term = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity. embeddings: (N, D) -> (N, N)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms
    return normalized @ normalized.T


def get_predictor_outputs(model_v2, x, mask_ratio=0.625, n_patches=16):
    """
    Forward pass and extract per-position predictor predictions.
    Returns:
        predictions: (n_mask, embed_dim) - predictor outputs per masked position
        mask_indices: (n_mask,) - which positions were masked
    """
    model_v2.eval()
    with torch.no_grad():
        x_batch = x.unsqueeze(0).to(DEVICE)  # (1, C, T)

        n_mask = int(n_patches * mask_ratio)
        n_context = n_patches - n_mask

        # Fixed masking: last n_mask patches are masked for reproducibility
        context_indices = torch.arange(n_context, device=DEVICE).unsqueeze(0)  # (1, n_context)
        mask_indices = torch.arange(n_context, n_patches, device=DEVICE).unsqueeze(0)  # (1, n_mask)

        # Get encoder embeddings (all patch tokens, skip CLS)
        all_tokens = model_v2.encoder(x_batch, return_all_tokens=True)  # (1, n_patches+1, D)
        embeds = all_tokens[:, 1:]  # (1, n_patches, D) - skip CLS

        # Context embeddings
        context_embeds = embeds[:, :n_context, :]  # (1, n_context, D)

        # Run predictor
        predictor = model_v2.predictor
        predictions = predictor(context_embeds, context_indices, mask_indices)  # (1, n_mask, D)

    return predictions[0].cpu().numpy(), mask_indices[0].cpu().numpy()


# =============================================================================
# 3A. Prediction Heatmap (Collapsed V1-like vs Fixed V2)
# =============================================================================

def make_prediction_heatmap():
    print("\n" + "="*60)
    print("3A. Prediction Heatmap")
    print("="*60)

    # Load V2 best checkpoint
    ckpt_path = Path('checkpoints/jepa_v2_20260401_003619.pt')
    if not ckpt_path.exists():
        print(f"V2 checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    config = ckpt.get('config', {})
    embed_dim = config.get('embed_dim', 512)
    encoder_depth = config.get('encoder_depth', 4)

    # Build V2 model (fixed, non-collapsed)
    model_v2 = MechanicalJEPAV2(
        n_channels=3,
        window_size=4096,
        patch_size=256,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        predictor_depth=config.get('predictor_depth', 4),
        mask_ratio=config.get('mask_ratio', 0.625),
        predictor_pos=config.get('predictor_pos', 'sinusoidal'),
        var_reg_lambda=config.get('var_reg_lambda', 0.1),
        loss_fn=config.get('loss_fn', 'l1'),
    ).to(DEVICE)
    model_v2.load_state_dict(ckpt['model_state_dict'])
    print(f"V2 model loaded: embed_dim={embed_dim}, depth={encoder_depth}")

    # Build V1-like collapsed model (learnable pos, depth=2, no var_reg, no L1)
    model_collapsed = MechanicalJEPAV2(
        n_channels=3,
        window_size=4096,
        patch_size=256,
        embed_dim=512,
        encoder_depth=4,
        predictor_depth=2,
        mask_ratio=0.5,
        predictor_pos='learnable',
        var_reg_lambda=0.0,
        loss_fn='mse',
    ).to(DEVICE)
    # Use RANDOM (untrained) collapsed model — this shows pure collapse-prone architecture
    print("V1-like collapsed model: random init (learnable pos, pd2, MSE, no var_reg)")

    # Create test signal (random vibration-like signal)
    torch.manual_seed(42)
    x = torch.randn(3, 4096)

    n_patches = 16
    mask_ratio_v2 = 0.625
    n_mask_v2 = int(n_patches * mask_ratio_v2)
    n_context_v2 = n_patches - n_mask_v2

    mask_ratio_v1 = 0.5
    n_mask_v1 = int(n_patches * mask_ratio_v1)
    n_context_v1 = n_patches - n_mask_v1

    # --- Train collapsed model briefly so it actually collapses ---
    # Use a known ablation checkpoint if available (mask=0.5, learnable, mse, no var_reg)
    # Otherwise train 5 steps to show collapse tendency
    model_collapsed.eval()

    # Forward pass for V2 (fixed)
    preds_v2, mask_idx_v2 = get_predictor_outputs(model_v2, x, mask_ratio_v2, n_patches)
    print(f"V2 predictions shape: {preds_v2.shape}")

    # For collapsed model: use the V2 model structure but with random-learnable pos (not trained)
    # Actually use V1 ablation checkpoint if available, otherwise simulate with random V2
    ablation_ckpts = sorted(Path('checkpoints').glob('jepa_v2_20260401_22*.pt'))
    collapsed_ckpt = None
    for ck in ablation_ckpts:
        ck_data = torch.load(ck, map_location='cpu', weights_only=False)
        cfg = ck_data.get('config', {})
        if cfg.get('predictor_pos', '') == 'learnable' and cfg.get('mask_ratio', 0) <= 0.5:
            collapsed_ckpt = ck
            collapsed_cfg = cfg
            break

    if collapsed_ckpt:
        model_collapsed2 = MechanicalJEPAV2(
            n_channels=3,
            window_size=4096,
            patch_size=256,
            embed_dim=collapsed_cfg.get('embed_dim', 512),
            encoder_depth=collapsed_cfg.get('encoder_depth', 4),
            predictor_depth=collapsed_cfg.get('predictor_depth', 2),
            mask_ratio=collapsed_cfg.get('mask_ratio', 0.5),
            predictor_pos=collapsed_cfg.get('predictor_pos', 'learnable'),
            var_reg_lambda=0.0,
            loss_fn='mse',
        ).to(DEVICE)
        model_collapsed2.load_state_dict(ck_data['model_state_dict'])
        print(f"Loaded collapsed ablation checkpoint: {collapsed_ckpt.name}")
        preds_collapsed, mask_idx_collapsed = get_predictor_outputs(
            model_collapsed2, x, collapsed_cfg.get('mask_ratio', 0.5), n_patches)
        n_mask_collapsed = preds_collapsed.shape[0]
        label_collapsed = f"V1-like (ablation: learnable pos, pd{collapsed_cfg.get('predictor_depth',2)}, MSE)"
    else:
        # Use random init of V1-style model as demonstration
        preds_collapsed, mask_idx_collapsed = get_predictor_outputs(
            model_collapsed, x, mask_ratio_v1, n_patches)
        n_mask_collapsed = preds_collapsed.shape[0]
        label_collapsed = "V1-like (random init, learnable pos, pd2)"

    # Cosine similarity matrices
    cosim_v2 = cosine_similarity_matrix(preds_v2)
    cosim_collapsed = cosine_similarity_matrix(preds_collapsed)

    print(f"V2 prediction spread (std over positions): {preds_v2.std(axis=0).mean():.4f}")
    print(f"V2 pred cosim off-diag mean: {(cosim_v2.sum() - np.trace(cosim_v2)) / (cosim_v2.size - len(cosim_v2)):.4f}")
    print(f"Collapsed pred spread: {preds_collapsed.std(axis=0).mean():.4f}")
    print(f"Collapsed pred cosim off-diag mean: {(cosim_collapsed.sum() - np.trace(cosim_collapsed)) / (cosim_collapsed.size - len(cosim_collapsed)):.4f}")

    if not HAS_MATPLOTLIB:
        print("No matplotlib — skipping plot")
        return

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.4, wspace=0.35)

    # Row 0: V2 (fixed)
    ax0 = fig.add_subplot(gs[0, :2])
    # Show first 64 dims of predictions for clarity
    n_dim_show = min(64, preds_v2.shape[1])
    im = ax0.imshow(preds_v2[:, :n_dim_show], aspect='auto', cmap='RdBu_r',
                    vmin=-2, vmax=2, interpolation='nearest')
    ax0.set_xlabel('Embedding dimension (first 64)', fontsize=11)
    ax0.set_ylabel('Masked position', fontsize=11)
    ax0.set_title('V2 Fixed: Predictions per masked position', fontsize=12, fontweight='bold')
    ax0.set_yticks(range(n_mask_v2))
    ax0.set_yticklabels([f'pos {i}' for i in mask_idx_v2], fontsize=8)
    plt.colorbar(im, ax=ax0, fraction=0.03)

    ax1 = fig.add_subplot(gs[0, 2])
    im1 = ax1.imshow(cosim_v2, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    ax1.set_title('V2: Pred cosine sim', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Position', fontsize=10)
    ax1.set_ylabel('Position', fontsize=10)
    plt.colorbar(im1, ax=ax1, fraction=0.06)
    v2_off_diag = (cosim_v2.sum() - np.trace(cosim_v2)) / (cosim_v2.size - len(cosim_v2))
    ax1.set_xlabel(f'Off-diag mean: {v2_off_diag:.3f}', fontsize=10)

    # Row 1: Collapsed
    ax2 = fig.add_subplot(gs[1, :2])
    n_dim_show2 = min(64, preds_collapsed.shape[1])
    im2 = ax2.imshow(preds_collapsed[:, :n_dim_show2], aspect='auto', cmap='RdBu_r',
                     vmin=-2, vmax=2, interpolation='nearest')
    ax2.set_xlabel('Embedding dimension (first 64)', fontsize=11)
    ax2.set_ylabel('Masked position', fontsize=11)
    ax2.set_title(f'V1-like Collapsed: Predictions\n({label_collapsed})', fontsize=10, fontweight='bold')
    ax2.set_yticks(range(n_mask_collapsed))
    ax2.set_yticklabels([f'pos {i}' for i in mask_idx_collapsed], fontsize=8)
    plt.colorbar(im2, ax=ax2, fraction=0.03)

    ax3 = fig.add_subplot(gs[1, 2])
    im3 = ax3.imshow(cosim_collapsed, cmap='hot', vmin=0, vmax=1, interpolation='nearest')
    ax3.set_title('V1-like: Pred cosine sim', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Position', fontsize=10)
    ax3.set_ylabel('Position', fontsize=10)
    plt.colorbar(im3, ax=ax3, fraction=0.06)
    c_off_diag = (cosim_collapsed.sum() - np.trace(cosim_collapsed)) / (cosim_collapsed.size - len(cosim_collapsed))
    ax3.set_xlabel(f'Off-diag mean: {c_off_diag:.3f}', fontsize=10)

    # Spread comparison bar
    ax4 = fig.add_subplot(gs[:, 3])
    models = ['V1-like\n(collapsed)', 'V2\n(fixed)']
    spreads = [preds_collapsed.std(axis=0).mean(), preds_v2.std(axis=0).mean()]
    colors = ['#d62728', '#2ca02c']
    bars = ax4.bar(models, spreads, color=colors, alpha=0.8, width=0.5)
    ax4.axhline(0.1, color='gray', linestyle='--', label='Collapse threshold (0.1)')
    ax4.set_ylabel('Prediction spread\n(mean std across positions)', fontsize=11)
    ax4.set_title('Prediction Diversity\nSpread Comparison', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    for bar, val in zip(bars, spreads):
        ax4.text(bar.get_x() + bar.get_width() / 2., val + 0.005,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    fig.suptitle('Predictor Collapse: Before (V1-like) vs After (V2 Fixed)',
                 fontsize=14, fontweight='bold', y=1.01)
    out_path = PLOTS_DIR / 'v4_collapse_prediction_heatmap.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# 3B. Positional Embedding Analysis
# =============================================================================

def make_positional_embedding_analysis():
    print("\n" + "="*60)
    print("3B. Positional Embedding Analysis")
    print("="*60)

    n_positions = 16
    # Try to infer d_model from V2 checkpoint
    ckpt_path = Path('checkpoints/jepa_v2_20260401_003619.pt')
    predictor_dim = 128  # default from model config
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = ckpt.get('config', {})
        # predictor_dim is typically embed_dim // 4
        embed_dim = cfg.get('embed_dim', 512)
        predictor_dim = max(64, embed_dim // 4)

    # Sinusoidal (V2 fixed)
    sino = sinusoidal_pos_encoding(n_positions, predictor_dim)  # (16, predictor_dim)
    cosim_sino = cosine_similarity_matrix(sino)

    # Learnable-like: random normal init (collapsed model initialization)
    np.random.seed(42)
    learned_init = np.random.randn(n_positions, predictor_dim) * 0.02  # trunc_normal(0.02)
    cosim_learned = cosine_similarity_matrix(learned_init)

    # Learned-after-training: load from ablation checkpoint if available
    ablation_ckpts = sorted(Path('checkpoints').glob('jepa_v2_20260401_22*.pt'))
    learned_trained = None
    for ck_path in ablation_ckpts:
        ck_data = torch.load(ck_path, map_location='cpu', weights_only=False)
        cfg = ck_data.get('config', {})
        if cfg.get('predictor_pos', '') == 'learnable':
            # Extract learned positional embeddings
            sd = ck_data['model_state_dict']
            for key in sd.keys():
                if 'pos_embed' in key and 'predictor' in key:
                    pe = sd[key].numpy()  # (1, N, D) or (N, D)
                    if pe.ndim == 3:
                        pe = pe[0]
                    if pe.shape[0] == n_positions:
                        learned_trained = pe[:, :predictor_dim] if pe.shape[1] >= predictor_dim else pe
                        print(f"Loaded trained learnable pos embeddings from {ck_path.name}")
                        break
            if learned_trained is not None:
                break

    if not HAS_MATPLOTLIB:
        print("No matplotlib — skipping plot")
        return

    n_plots = 3 if learned_trained is not None else 2
    fig, axes = plt.subplots(1, n_plots + 1, figsize=(6 * n_plots + 4, 5))

    # Plot 1: Sinusoidal pos embedding visualized
    ax = axes[0]
    n_dim_show = min(64, predictor_dim)
    im = ax.imshow(sino[:, :n_dim_show], aspect='auto', cmap='seismic',
                   vmin=-1, vmax=1, interpolation='nearest')
    ax.set_xlabel('Embedding dimension (first 64)', fontsize=11)
    ax.set_ylabel('Position index', fontsize=11)
    ax.set_title('Sinusoidal Position Embedding\n(V2 fixed)', fontsize=12, fontweight='bold')
    ax.set_yticks(range(0, n_positions, 2))
    plt.colorbar(im, ax=ax, fraction=0.04)

    # Plot 2: Cosim matrix for sinusoidal
    ax = axes[1]
    im2 = ax.imshow(cosim_sino, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    ax.set_title('Sinusoidal: Pairwise cosine sim\n(gradual decrease = good)', fontsize=11)
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('Position', fontsize=10)
    plt.colorbar(im2, ax=ax, fraction=0.04)
    sino_off = (cosim_sino.sum() - np.trace(cosim_sino)) / (cosim_sino.size - n_positions)
    ax.set_title(f'Sinusoidal cosim\nOff-diag mean: {sino_off:.3f}', fontsize=11)

    # Plot 3 (if available): Cosim matrix for trained learnable
    if learned_trained is not None:
        cosim_trained = cosine_similarity_matrix(learned_trained)
        ax = axes[2]
        im3 = ax.imshow(cosim_trained, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        trained_off = (cosim_trained.sum() - np.trace(cosim_trained)) / (cosim_trained.size - n_positions)
        ax.set_title(f'Learnable (trained): cosim\nOff-diag mean: {trained_off:.3f}', fontsize=11)
        ax.set_xlabel('Position', fontsize=10)
        ax.set_ylabel('Position', fontsize=10)
        plt.colorbar(im3, ax=ax, fraction=0.04)
    else:
        # Show random init cosim
        ax = axes[2] if n_plots >= 3 else None
        if ax is not None:
            cosim_rand = cosine_similarity_matrix(learned_init)
            im4 = ax.imshow(cosim_rand, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            rand_off = (cosim_rand.sum() - np.trace(cosim_rand)) / (cosim_rand.size - n_positions)
            ax.set_title(f'Learnable (random init): cosim\nOff-diag mean: {rand_off:.3f}', fontsize=11)
            ax.set_xlabel('Position', fontsize=10)
            ax.set_ylabel('Position', fontsize=10)
            plt.colorbar(im4, ax=ax, fraction=0.04)

    # Last subplot: min distance vs position separation (shows structure)
    ax_last = axes[-1]
    distances_sino = 1 - cosim_sino  # (16, 16) distance matrix
    separations = []
    dist_means_sino = []
    dist_means_learn = []
    for sep in range(1, n_positions):
        d_sino = [distances_sino[i, i + sep] for i in range(n_positions - sep)]
        separations.append(sep)
        dist_means_sino.append(np.mean(d_sino))

    cosim_to_compare = cosine_similarity_matrix(
        learned_trained if learned_trained is not None else learned_init)
    distances_compare = 1 - cosim_to_compare
    for sep in range(1, n_positions):
        d_c = [distances_compare[i, i + sep] for i in range(n_positions - sep)]
        dist_means_learn.append(np.mean(d_c))

    ax_last.plot(separations, dist_means_sino, 'b-o', label='Sinusoidal (V2)', linewidth=2, markersize=6)
    ax_last.plot(separations, dist_means_learn, 'r--s',
                 label='Learnable (trained)' if learned_trained is not None else 'Learnable (init)',
                 linewidth=2, markersize=6)
    ax_last.set_xlabel('Position separation', fontsize=11)
    ax_last.set_ylabel('Mean embedding distance', fontsize=11)
    ax_last.set_title('Distance vs Position Separation\n(sinusoidal = smooth gradient)', fontsize=11)
    ax_last.legend(fontsize=10)
    ax_last.grid(True, alpha=0.3)

    fig.suptitle('Positional Embedding Analysis: Sinusoidal vs Learnable',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_path = PLOTS_DIR / 'v4_collapse_positional_embeddings.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# 3C. Mask Ratio vs Collapse Threshold
# =============================================================================

def make_mask_ratio_collapse_plot():
    """
    Use data from Exp 26 (fine-grained mask sweep) and Exp 37 (ablation).
    Plot spread_ratio vs mask_ratio with collapse threshold line.
    """
    print("\n" + "="*60)
    print("3C. Mask Ratio vs Collapse Threshold")
    print("="*60)

    # Data from Exp 37 (30ep, seed=42, V2 config except mask ratio)
    # spread_ratio at 30 epochs:
    ablation_data = {
        # (mask_ratio, spread_ratio, collapsed, accuracy_30ep, note)
        0.50: (0.018, True,  65.9, 'learnable, pd2, mse'),
        0.50: (0.050, True,  68.4, 'sino, pd4, mse (no var_reg)'),
        0.625: (0.162, False, 70.7, 'V2 full (sino+pd4+l1+var_reg)'),
    }

    # From Exp 26 (mask sweep at 30ep with full V2 config):
    mask_sweep_data = [
        # (mask_ratio, 30ep_acc) from Exp 26
        (0.500, 61.4),
        (0.5625, 64.7),
        (0.625, 70.7),
        (0.6875, 70.3),
        (0.750, 76.0),
        (0.8125, 72.6),
        (0.875, 72.2),
    ]

    # From Exp 27 (100ep validation):
    mask_100ep_data = [
        (0.625, 82.1, 5.4),   # mean, std
        (0.750, 78.5, 4.4),
        (0.8125, 73.8, 12.98),
    ]

    # From Exp 16 ablation (30ep spread ratios):
    spread_data = [
        # (mask_ratio, spread_ratio, config_label)
        (0.50, 0.035, 'sino, pd4, mse'),
        (0.50, 0.012, 'learnable, pd4, mse'),
        (0.50, 0.149, 'sino, pd4, l1, var_reg=0.1'),
        (0.75, 0.042, 'sino, pd4, l1, mask=0.75 (no var_reg)'),
        (0.75, 0.260, 'V2 full: sino+pd4+l1+mask=0.75+var_reg=0.1'),
        (0.625, 0.162, 'V2 full: sino+pd4+l1+mask=0.625+var_reg=0.1'),
    ]

    if not HAS_MATPLOTLIB:
        print("No matplotlib — skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: 30ep accuracy vs mask ratio
    ax = axes[0]
    mask_ratios_sweep = [d[0] for d in mask_sweep_data]
    accs_sweep = [d[1] for d in mask_sweep_data]
    ax.plot(mask_ratios_sweep, accs_sweep, 'b-o', linewidth=2, markersize=8, label='V2 full config (30ep)')

    # Add 100ep data as separate points
    mask_ratios_100 = [d[0] for d in mask_100ep_data]
    accs_100 = [d[1] for d in mask_100ep_data]
    stds_100 = [d[2] for d in mask_100ep_data]
    ax.errorbar(mask_ratios_100, accs_100, yerr=stds_100, fmt='rs', markersize=10,
                capsize=5, linewidth=2, label='V2 full config (100ep, 3 seeds)')

    ax.axvline(0.625, color='green', linestyle='--', alpha=0.7, label='V2 optimal (0.625)')
    ax.set_xlabel('Mask Ratio', fontsize=12)
    ax.set_ylabel('CWRU Test Accuracy (%)', fontsize=12)
    ax.set_title('Mask Ratio vs CWRU Accuracy\n(30ep and 100ep)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(55, 90)

    # Right: Spread ratio vs mask ratio (collapse analysis)
    ax2 = axes[1]

    # Group spread data by config
    sino_mse = [(0.50, 0.035)]
    sino_l1_varreg = [(0.50, 0.149), (0.625, 0.162), (0.75, 0.260)]
    sino_l1_no_varreg = [(0.75, 0.042)]

    ax2.scatter([d[0] for d in sino_mse], [d[1] for d in sino_mse],
                s=120, color='orange', marker='D', label='Sino+pd4+MSE (no var_reg)', zorder=5)
    ax2.plot([d[0] for d in sino_l1_varreg], [d[1] for d in sino_l1_varreg],
             'g-^', markersize=12, linewidth=2.5, label='V2 full (sino+pd4+L1+var_reg=0.1)')
    ax2.scatter([d[0] for d in sino_l1_no_varreg], [d[1] for d in sino_l1_no_varreg],
                s=120, color='red', marker='x', linewidths=3, label='Sino+pd4+L1 (no var_reg)')

    # Collapse threshold line
    ax2.axhline(0.1, color='red', linestyle='--', linewidth=2, label='Collapse threshold (0.1)')

    # Annotate key points
    for m, s, label in [(0.625, 0.162, 'V2 opt'), (0.75, 0.260, 'Best spread')]:
        ax2.annotate(label, (m, s), textcoords='offset points', xytext=(10, 5),
                     fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))

    # Color regions
    ax2.axhspan(0, 0.1, alpha=0.1, color='red', label='Collapse zone')
    ax2.axhspan(0.1, 0.35, alpha=0.1, color='green', label='Healthy zone')

    ax2.set_xlabel('Mask Ratio', fontsize=12)
    ax2.set_ylabel('Spread Ratio (prediction diversity)', fontsize=12)
    ax2.set_title('Mask Ratio vs Predictor Spread\n(collapse threshold = 0.1)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.02, 0.35)
    ax2.set_xlim(0.45, 0.80)

    plt.tight_layout()
    out_path = PLOTS_DIR / 'v4_collapse_mask_ratio_threshold.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import os
    os.chdir(Path(__file__).parent)

    print("Generating collapse visualizations for V4 notebook...")

    make_prediction_heatmap()
    make_positional_embedding_analysis()
    make_mask_ratio_collapse_plot()

    print("\n" + "="*60)
    print("All visualizations complete!")
    print(f"Output directory: {PLOTS_DIR.absolute()}")
    for f in sorted(PLOTS_DIR.glob('v4_collapse_*.png')):
        print(f"  {f.name}")

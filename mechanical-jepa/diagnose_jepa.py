"""
Diagnostic script to test encoder and predictor in isolation.

Run from mechanical-jepa directory:
    python diagnose_jepa.py --checkpoint checkpoints/your_checkpoint.pt
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import MechanicalJEPA
from src.data import create_dataloaders


def load_model_and_data(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint and get test data."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint
    config = ckpt.get('config', {})
    embed_dim = config.get('embed_dim', 512)
    encoder_depth = config.get('encoder_depth', 4)

    print(f"  embed_dim={embed_dim}, encoder_depth={encoder_depth}")

    # Create model
    model = MechanicalJEPA(
        n_channels=3,
        window_size=4096,
        patch_size=256,
        embed_dim=embed_dim,
        encoder_depth=encoder_depth,
        predictor_depth=2,
        mask_ratio=0.5,
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Get data
    data_dir = Path(__file__).parent / 'data' / 'bearings'
    _, test_loader, _ = create_dataloaders(data_dir=data_dir, batch_size=32)

    return model, test_loader, config


def test_encoder_diversity(model, test_loader, device):
    """
    Test 1: Does the encoder produce diverse embeddings?

    Checks:
    - Variance of embeddings (should be high, not collapsed)
    - Inter-class vs intra-class distances
    - Embedding distribution
    """
    print("\n" + "="*60)
    print("TEST 1: ENCODER DIVERSITY")
    print("="*60)

    all_embeddings = []
    all_labels = []
    all_patch_embeddings = []

    with torch.no_grad():
        for batch_idx, (x, y, _) in enumerate(test_loader):
            if batch_idx >= 5:  # Enough samples
                break
            x = x.to(device)

            # Get CLS embedding
            cls_embed = model.encoder(x, return_all_tokens=False)  # (B, D)
            all_embeddings.append(cls_embed.cpu())
            all_labels.append(y)

            # Get all patch embeddings
            patch_embed = model.encoder(x, return_all_tokens=True)[:, 1:]  # (B, N, D)
            all_patch_embeddings.append(patch_embed.cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    patch_embeddings = torch.cat(all_patch_embeddings, dim=0).numpy()  # (total_samples, N, D)

    print(f"\nSamples: {len(embeddings)}, Classes: {len(np.unique(labels))}")

    # Check 1: Overall embedding variance
    embed_var = embeddings.var(axis=0).mean()
    print(f"\n[1a] CLS Embedding variance (mean over dims): {embed_var:.6f}")
    print(f"     {'[OK] GOOD' if embed_var > 0.01 else '[X] LOW - possible collapse'}")

    # Check 2: Per-class embedding variance
    print(f"\n[1b] Per-class variance:")
    for c in np.unique(labels):
        class_embeds = embeddings[labels == c]
        class_var = class_embeds.var(axis=0).mean()
        print(f"     Class {c}: {class_var:.6f} (n={len(class_embeds)})")

    # Check 3: Inter-class distance vs intra-class distance
    print(f"\n[1c] Class separation:")
    class_centers = {}
    for c in np.unique(labels):
        class_centers[c] = embeddings[labels == c].mean(axis=0)

    # Intra-class distance (average distance to class center)
    intra_dists = []
    for c in np.unique(labels):
        class_embeds = embeddings[labels == c]
        dists = np.linalg.norm(class_embeds - class_centers[c], axis=1)
        intra_dists.extend(dists)
    mean_intra = np.mean(intra_dists)

    # Inter-class distance (distance between class centers)
    centers = np.array(list(class_centers.values()))
    inter_dists = []
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            inter_dists.append(np.linalg.norm(centers[i] - centers[j]))
    mean_inter = np.mean(inter_dists)

    ratio = mean_inter / (mean_intra + 1e-8)
    print(f"     Mean intra-class distance: {mean_intra:.4f}")
    print(f"     Mean inter-class distance: {mean_inter:.4f}")
    print(f"     Ratio (inter/intra): {ratio:.4f}")
    print(f"     {'[OK] GOOD' if ratio > 1.5 else '[!] LOW - classes not well separated'}")

    # Check 4: Patch embedding diversity (within a sample)
    print(f"\n[1d] Patch diversity (within samples):")
    patch_vars = []
    for sample_patches in patch_embeddings[:20]:  # First 20 samples
        # Variance across patches for this sample
        var = sample_patches.var(axis=0).mean()
        patch_vars.append(var)

    mean_patch_var = np.mean(patch_vars)
    print(f"     Mean patch variance (across N patches): {mean_patch_var:.6f}")
    print(f"     {'[OK] GOOD' if mean_patch_var > 0.01 else '[X] LOW - patches too similar'}")

    return {
        'embed_var': embed_var,
        'mean_intra': mean_intra,
        'mean_inter': mean_inter,
        'separation_ratio': ratio,
        'patch_var': mean_patch_var,
        'embeddings': embeddings,
        'labels': labels,
    }


def test_predictor_sensitivity(model, test_loader, device):
    """
    Test 2: Does the predictor produce different outputs for different positions?

    Checks:
    - Same context, different mask positions -> different predictions?
    - Prediction variance across positions
    - Sensitivity to positional information
    """
    print("\n" + "="*60)
    print("TEST 2: PREDICTOR SENSITIVITY TO POSITION")
    print("="*60)

    # Get one batch
    x, y, _ = next(iter(test_loader))
    x = x[:4].to(device)  # Just 4 samples

    with torch.no_grad():
        # Get context embeddings (using first half as context)
        n_patches = model.n_patches
        n_context = n_patches // 2
        n_mask = n_patches - n_context

        # Fixed context indices (first half)
        context_indices = torch.arange(n_context).unsqueeze(0).expand(x.shape[0], -1).to(device)

        # Get context embeddings
        context_embeds = model.encoder(x, return_all_tokens=True)[:, 1:]  # (B, N, D)
        context_embeds = context_embeds[:, :n_context]  # (B, n_context, D)

        print(f"\nSetup: {n_context} context patches, {n_mask} mask positions")
        print(f"Context indices: {context_indices[0].tolist()}")

        # Test 1: Predict each masked position individually
        print(f"\n[2a] Single position predictions:")
        single_preds = []
        for mask_pos in range(n_context, n_patches):
            mask_idx = torch.tensor([[mask_pos]]).expand(x.shape[0], -1).to(device)
            pred = model.predictor(context_embeds, context_indices, mask_idx)  # (B, 1, D)
            single_preds.append(pred[:, 0].cpu())

        single_preds = torch.stack(single_preds, dim=1).numpy()  # (B, n_mask, D)

        # Check variance across positions
        pos_variance = single_preds.var(axis=1).mean()
        print(f"     Variance across positions: {pos_variance:.6f}")
        print(f"     {'[OK] GOOD' if pos_variance > 0.001 else '[X] LOW - same prediction for all positions'}")

        # Check pairwise distances between position predictions
        print(f"\n[2b] Pairwise distances between position predictions:")
        for sample_idx in range(min(2, x.shape[0])):
            sample_preds = single_preds[sample_idx]  # (n_mask, D)
            dists = []
            for i in range(len(sample_preds)):
                for j in range(i+1, len(sample_preds)):
                    d = np.linalg.norm(sample_preds[i] - sample_preds[j])
                    dists.append(d)
            mean_dist = np.mean(dists)
            print(f"     Sample {sample_idx}: mean pairwise distance = {mean_dist:.4f}")

        # Test 2: Compare predictions for same position across different samples
        print(f"\n[2c] Same position, different samples:")
        for pos_idx in [0, n_mask//2, n_mask-1]:
            preds_at_pos = single_preds[:, pos_idx]  # (B, D)
            cross_sample_var = preds_at_pos.var(axis=0).mean()
            print(f"     Position {pos_idx + n_context}: cross-sample variance = {cross_sample_var:.6f}")

        # Test 3: All positions at once
        print(f"\n[2d] All masked positions at once:")
        mask_indices = torch.arange(n_context, n_patches).unsqueeze(0).expand(x.shape[0], -1).to(device)
        all_preds = model.predictor(context_embeds, context_indices, mask_indices)  # (B, n_mask, D)
        all_preds_np = all_preds.cpu().numpy()

        # Compare with single predictions
        diff = np.abs(all_preds_np - single_preds).mean()
        print(f"     Diff from single predictions: {diff:.6f}")
        print(f"     (Should be small - predictor is consistent)")

    return {
        'pos_variance': pos_variance,
        'single_preds': single_preds,
    }


def test_target_encoder(model, test_loader, device):
    """
    Test 3: Are target encoder outputs different from context encoder?

    Checks:
    - EMA has diverged from main encoder
    - Target produces sensible targets
    """
    print("\n" + "="*60)
    print("TEST 3: TARGET ENCODER vs CONTEXT ENCODER")
    print("="*60)

    x, _, _ = next(iter(test_loader))
    x = x[:4].to(device)

    with torch.no_grad():
        # Get embeddings from both encoders
        context_out = model.encoder(x, return_all_tokens=True)[:, 1:]  # (B, N, D)
        target_out = model.target_encoder(x, return_all_tokens=True)[:, 1:]  # (B, N, D)

        # Compute difference
        diff = (context_out - target_out).abs().mean().item()

        # Compute cosine similarity
        context_flat = context_out.reshape(-1, context_out.shape[-1])
        target_flat = target_out.reshape(-1, target_out.shape[-1])
        cos_sim = F.cosine_similarity(context_flat, target_flat, dim=-1).mean().item()

        print(f"\n[3a] Encoder comparison:")
        print(f"     Mean absolute difference: {diff:.6f}")
        print(f"     Mean cosine similarity: {cos_sim:.4f}")

        if diff < 0.001:
            print(f"     [!] WARNING: Encoders nearly identical - EMA may not be updating")
        elif cos_sim > 0.99:
            print(f"     [!] NOTE: Very high similarity - EMA decay may be too high")
        else:
            print(f"     [OK] GOOD: Encoders are different (EMA is working)")

        # Check target embedding variance
        target_var = target_out.var(dim=1).mean().item()
        print(f"\n[3b] Target embedding variance: {target_var:.6f}")
        print(f"     {'[OK] GOOD' if target_var > 0.01 else '[X] LOW - targets may be collapsed'}")


def visualize_predictions(model, test_loader, device, save_path=None):
    """Create detailed visualization of predictions vs targets."""
    print("\n" + "="*60)
    print("VISUALIZATION: Predictions vs Targets")
    print("="*60)

    x, y, _ = next(iter(test_loader))
    x = x[:1].to(device)  # Single sample

    with torch.no_grad():
        # Run forward pass
        loss, predictions, targets = model(x)

        pred_np = predictions[0].cpu().numpy()  # (n_mask, D)
        target_np = targets[0].cpu().numpy()  # (n_mask, D)

        n_mask = pred_np.shape[0]

        # Cosine similarities
        cos_sims = []
        for i in range(n_mask):
            sim = F.cosine_similarity(
                torch.tensor(pred_np[i:i+1]),
                torch.tensor(target_np[i:i+1])
            ).item()
            cos_sims.append(sim)

        print(f"\nPer-position cosine similarity:")
        for i, sim in enumerate(cos_sims):
            print(f"  Position {i}: {sim:.4f}")
        print(f"Mean: {np.mean(cos_sims):.4f}, Std: {np.std(cos_sims):.4f}")

        # PCA visualization
        combined = np.vstack([pred_np, target_np])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        pred_2d = combined_2d[:n_mask]
        target_2d = combined_2d[n_mask:]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: PCA with connections
        ax = axes[0]
        ax.scatter(target_2d[:, 0], target_2d[:, 1], c='blue', s=100, label='Target', zorder=3)
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], c='orange', s=100, marker='x', label='Predicted', zorder=3)
        for i in range(n_mask):
            ax.plot([target_2d[i, 0], pred_2d[i, 0]],
                   [target_2d[i, 1], pred_2d[i, 1]],
                   'gray', alpha=0.5, linewidth=1)
            ax.annotate(str(i), (target_2d[i, 0], target_2d[i, 1]), fontsize=8)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA: Targets (blue) vs Predictions (orange)')
        ax.legend()

        # Plot 2: Cosine similarity per position
        ax = axes[1]
        ax.bar(range(n_mask), cos_sims, color='steelblue')
        ax.axhline(y=np.mean(cos_sims), color='red', linestyle='--',
                  label=f'Mean: {np.mean(cos_sims):.3f}')
        ax.set_xlabel('Masked Patch Index')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Cosine Similarity per Position')
        ax.legend()
        ax.set_ylim(0, 1.1)

        # Plot 3: Prediction variance check
        ax = axes[2]
        pred_var = pred_np.var(axis=0)
        target_var = target_np.var(axis=0)
        ax.hist(pred_var, bins=30, alpha=0.5, label=f'Pred var (mean={pred_var.mean():.4f})')
        ax.hist(target_var, bins=30, alpha=0.5, label=f'Target var (mean={target_var.mean():.4f})')
        ax.set_xlabel('Variance per dimension')
        ax.set_ylabel('Count')
        ax.set_title('Embedding Variance Distribution')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nSaved visualization to {save_path}")

        plt.show()

        # Key diagnostic
        pred_spread = np.std(pred_2d, axis=0).mean()
        target_spread = np.std(target_2d, axis=0).mean()

        print(f"\n[DIAGNOSTIC] Prediction spread in PCA: {pred_spread:.4f}")
        print(f"[DIAGNOSTIC] Target spread in PCA: {target_spread:.4f}")
        print(f"[DIAGNOSTIC] Ratio (pred/target): {pred_spread/target_spread:.4f}")

        if pred_spread / target_spread < 0.1:
            print("\n[!] PROBLEM: Predictions are collapsing to a small region!")
            print("  Possible causes:")
            print("  1. Predictor too weak (not enough depth)")
            print("  2. Predictor too strong (ignoring positional info)")
            print("  3. Learning rate too high (predictor overshoots)")
            print("  4. Not enough training")


def main():
    parser = argparse.ArgumentParser(description='Diagnose JEPA encoder and predictor')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--save-fig', type=str, default='diagnose_jepa.png', help='Save figure path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and data
    model, test_loader, config = load_model_and_data(args.checkpoint, device)

    # Run tests
    encoder_results = test_encoder_diversity(model, test_loader, device)
    predictor_results = test_predictor_sensitivity(model, test_loader, device)
    test_target_encoder(model, test_loader, device)
    visualize_predictions(model, test_loader, device, save_path=args.save_fig)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    issues = []

    if encoder_results['embed_var'] < 0.01:
        issues.append("Encoder embeddings have low variance (possible collapse)")

    if encoder_results['separation_ratio'] < 1.5:
        issues.append("Classes not well separated in embedding space")

    if predictor_results['pos_variance'] < 0.001:
        issues.append("Predictor outputs same embedding for all positions (collapse)")

    if encoder_results['patch_var'] < 0.01:
        issues.append("Patch embeddings too similar within samples")

    if issues:
        print("\n[!] ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n[OK] No major issues detected")

    print("\nRun complete.")


if __name__ == '__main__':
    main()

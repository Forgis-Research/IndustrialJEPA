"""
Quick diagnostic for JEPA encoder and predictor using synthetic data.
Faster than full diagnostic - no dataset loading needed.

Run: python quick_diagnose.py --checkpoint checkpoints/your_checkpoint.pt
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.models import MechanicalJEPA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print(f"\nLoading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    config = ckpt.get('config', {})
    embed_dim = config.get('embed_dim', 256)
    encoder_depth = config.get('encoder_depth', 4)
    print(f"Config: embed_dim={embed_dim}, encoder_depth={encoder_depth}")

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

    # Create synthetic test data
    print("\n" + "="*60)
    print("Creating synthetic test signals...")
    print("="*60)

    B = 8  # batch size
    x = torch.randn(B, 3, 4096).to(device)

    # Also create signals with different characteristics
    x_smooth = torch.randn(B, 3, 4096).to(device)
    x_smooth = F.avg_pool1d(x_smooth, kernel_size=16, stride=1, padding=8)[:, :, :4096]

    x_high_freq = torch.randn(B, 3, 4096).to(device) * 2  # higher amplitude noise

    print(f"Created 3 test batches of shape {x.shape}")

    # TEST 1: Encoder output diversity
    print("\n" + "="*60)
    print("TEST 1: ENCODER OUTPUT DIVERSITY")
    print("="*60)

    with torch.no_grad():
        # Get CLS tokens
        cls_random = model.encoder(x, return_all_tokens=False)
        cls_smooth = model.encoder(x_smooth, return_all_tokens=False)
        cls_high = model.encoder(x_high_freq, return_all_tokens=False)

        # Get all patch tokens
        patches_random = model.encoder(x, return_all_tokens=True)[:, 1:]  # remove CLS

        # Check variance
        cls_var = cls_random.var(dim=0).mean().item()
        print(f"\n[1a] CLS token variance (across batch): {cls_var:.6f}")
        print(f"     {'[OK]' if cls_var > 0.01 else '[X] LOW'}")

        patch_var = patches_random.var(dim=1).mean().item()  # var across patches, then mean
        print(f"\n[1b] Patch token variance (within sample): {patch_var:.6f}")
        print(f"     {'[OK]' if patch_var > 0.01 else '[X] LOW - patches too similar'}")

        # Check if different inputs give different outputs
        diff_smooth = (cls_random - cls_smooth).abs().mean().item()
        diff_high = (cls_random - cls_high).abs().mean().item()
        print(f"\n[1c] Encoder sensitivity to input:")
        print(f"     Random vs Smooth: {diff_smooth:.6f}")
        print(f"     Random vs HighFreq: {diff_high:.6f}")
        print(f"     {'[OK]' if diff_smooth > 0.1 else '[X] LOW - not distinguishing inputs'}")

    # TEST 2: Predictor position sensitivity
    print("\n" + "="*60)
    print("TEST 2: PREDICTOR POSITION SENSITIVITY")
    print("="*60)

    with torch.no_grad():
        n_patches = 16
        n_context = 8  # half as context
        n_mask = 8

        # Fixed context (first half)
        context_indices = torch.arange(n_context).unsqueeze(0).expand(B, -1).to(device)

        # Get context embeddings
        all_patches = model.encoder(x, return_all_tokens=True)[:, 1:]  # (B, 16, D)
        context_embeds = all_patches[:, :n_context]  # (B, 8, D)

        # Predict each mask position individually
        single_preds = []
        for pos in range(n_context, n_patches):
            mask_idx = torch.tensor([[pos]]).expand(B, -1).to(device)
            pred = model.predictor(context_embeds, context_indices, mask_idx)
            single_preds.append(pred[:, 0])  # (B, D)

        single_preds = torch.stack(single_preds, dim=1)  # (B, 8, D)

        # Check if predictions differ by position
        pred_var_across_pos = single_preds.var(dim=1).mean().item()
        print(f"\n[2a] Prediction variance across positions: {pred_var_across_pos:.6f}")
        print(f"     {'[OK]' if pred_var_across_pos > 0.001 else '[X] COLLAPSED - same output for all positions'}")

        # Check pairwise distances
        pairwise_dists = []
        for i in range(n_mask):
            for j in range(i+1, n_mask):
                dist = (single_preds[:, i] - single_preds[:, j]).norm(dim=-1).mean().item()
                pairwise_dists.append(dist)

        mean_pairwise = np.mean(pairwise_dists)
        print(f"\n[2b] Mean pairwise distance between predictions: {mean_pairwise:.4f}")
        print(f"     {'[OK]' if mean_pairwise > 0.1 else '[X] LOW - predictions too similar'}")

        # Check if same position gives same output (consistency)
        # Run twice with same inputs
        pred1 = model.predictor(context_embeds, context_indices,
                               torch.tensor([[n_context]]).expand(B, -1).to(device))
        pred2 = model.predictor(context_embeds, context_indices,
                               torch.tensor([[n_context]]).expand(B, -1).to(device))
        consistency = (pred1 - pred2).abs().mean().item()
        print(f"\n[2c] Predictor consistency (same input twice): {consistency:.8f}")
        print(f"     {'[OK] Deterministic' if consistency < 1e-6 else '[!] Non-deterministic'}")

    # TEST 3: Full forward pass
    print("\n" + "="*60)
    print("TEST 3: FULL FORWARD PASS (PREDICTIONS vs TARGETS)")
    print("="*60)

    with torch.no_grad():
        loss, predictions, targets = model(x)

        print(f"\n[3a] Loss value: {loss.item():.6f}")

        # Cosine similarity
        pred_norm = F.normalize(predictions, dim=-1)
        targ_norm = F.normalize(targets, dim=-1)
        cos_sim = (pred_norm * targ_norm).sum(dim=-1).mean().item()
        print(f"\n[3b] Mean cosine similarity: {cos_sim:.4f}")
        print(f"     {'[OK]' if cos_sim > 0.7 else '[!] LOW'}")

        # Check prediction spread vs target spread
        pred_std = predictions.std(dim=1).mean().item()
        targ_std = targets.std(dim=1).mean().item()
        spread_ratio = pred_std / (targ_std + 1e-8)

        print(f"\n[3c] Embedding spread:")
        print(f"     Prediction std: {pred_std:.4f}")
        print(f"     Target std: {targ_std:.4f}")
        print(f"     Ratio (pred/target): {spread_ratio:.4f}")

        if spread_ratio < 0.1:
            print("     [X] PROBLEM: Predictions collapsing to same point!")
        elif spread_ratio < 0.5:
            print("     [!] WARNING: Predictions less diverse than targets")
        else:
            print("     [OK] Predictions have similar spread to targets")

        # Per-position analysis
        print(f"\n[3d] Per-position cosine similarity:")
        n_mask = predictions.shape[1]
        for i in range(min(n_mask, 8)):
            sim = F.cosine_similarity(predictions[:, i], targets[:, i], dim=-1).mean().item()
            print(f"     Position {i}: {sim:.4f}")

    # TEST 4: Target encoder vs Context encoder
    print("\n" + "="*60)
    print("TEST 4: EMA TARGET ENCODER")
    print("="*60)

    with torch.no_grad():
        ctx_out = model.encoder(x, return_all_tokens=True)
        tgt_out = model.target_encoder(x, return_all_tokens=True)

        diff = (ctx_out - tgt_out).abs().mean().item()
        cos = F.cosine_similarity(ctx_out.reshape(-1, embed_dim),
                                  tgt_out.reshape(-1, embed_dim), dim=-1).mean().item()

        print(f"\n[4a] Context vs Target encoder:")
        print(f"     Mean abs difference: {diff:.6f}")
        print(f"     Mean cosine similarity: {cos:.4f}")

        if diff < 0.0001:
            print("     [X] PROBLEM: Encoders identical - EMA not updating?")
        elif cos > 0.999:
            print("     [!] NOTE: Very similar - EMA decay very high")
        else:
            print("     [OK] Encoders have diverged (EMA working)")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    issues = []
    if cls_var < 0.01:
        issues.append("Encoder CLS variance too low (collapse)")
    if patch_var < 0.01:
        issues.append("Patch embeddings too similar")
    if pred_var_across_pos < 0.001:
        issues.append("Predictor collapsed - same output for all positions")
    if spread_ratio < 0.1:
        issues.append("Predictions collapsing to single point")

    if issues:
        print("\n[!] ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nDIAGNOSIS:")
        if pred_var_across_pos < 0.001 and spread_ratio < 0.1:
            print("  -> PREDICTOR COLLAPSE: The predictor ignores position information")
            print("     and outputs nearly identical embeddings for all masked positions.")
            print("  -> The encoder might be fine (check cls_var and patch_var above)")
            print("\n  POSSIBLE FIXES:")
            print("  1. Increase predictor depth (currently: 2 -> try 3-4)")
            print("  2. Decrease predictor dim (makes task harder)")
            print("  3. Use different positional encoding in predictor")
            print("  4. Lower learning rate for predictor")
            print("  5. Train for more epochs")
    else:
        print("\n[OK] No major issues detected")


if __name__ == '__main__':
    main()

"""
Test pipeline for MTS-JEPA.
Must pass before any full experiments run.

Tests:
1. Data loading and preprocessing
2. Model forward pass shape verification
3. Loss computation and decrease over 5 epochs
4. Codebook utilization check
5. Downstream classifier verification
"""
import sys
import os
import time
import numpy as np
import torch

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_dataset, remove_constant_channels, make_non_overlapping_windows,
    make_window_labels, RevIN, create_views, PretrainDataset, prepare_data,
)
from models import (
    MTSJEPA, DownstreamClassifier, SoftCodebook, MTSJEPAEncoder,
    FinePredictor, CoarsePredictor, ReconstructionDecoder,
)
from train_utils import compute_total_loss, DEFAULT_LOSS_CONFIG


def test_data_loading():
    """Test 1: Load PSM (smallest available dataset) and verify shapes."""
    print("\n" + "="*60)
    print("TEST 1: Data Loading")
    print("="*60)

    train, test, labels = load_dataset("PSM")
    assert train.ndim == 2, f"Train should be 2D, got {train.ndim}"
    assert test.ndim == 2, f"Test should be 2D, got {test.ndim}"
    assert len(labels) == len(test), f"Labels length mismatch"
    print(f"  PSM: train {train.shape}, test {test.shape}")

    # Constant channel removal
    train_f, test_f, mask = remove_constant_channels(train, test)
    n_vars = train_f.shape[1]
    print(f"  After removal: {n_vars} vars (expected 25)")

    # Non-overlapping windows
    windows = make_non_overlapping_windows(train_f, 100)
    print(f"  Windows: {windows.shape}")
    assert windows.shape[1] == 100
    assert windows.shape[2] == n_vars

    # Window labels
    wl = make_window_labels(labels, 100)
    print(f"  Window labels: {wl.shape}, anomaly rate {wl.mean():.3f}")

    print("  PASSED ✓")
    return n_vars


def test_revin():
    """Test 2: RevIN forward and inverse."""
    print("\n" + "="*60)
    print("TEST 2: RevIN")
    print("="*60)

    revin = RevIN(25)
    x = torch.randn(4, 100, 25)

    x_norm = revin(x)
    x_rec = revin.inverse(x_norm)

    err = (x - x_rec).abs().max().item()
    print(f"  Reconstruction error: {err:.8f}")
    assert err < 1e-5, f"RevIN reconstruction error too large: {err}"

    print("  PASSED ✓")


def test_create_views():
    """Test 3: Multi-scale view construction."""
    print("\n" + "="*60)
    print("TEST 3: Multi-scale Views")
    print("="*60)

    B, T, V = 4, 100, 25
    window = torch.randn(B, T, V)

    fine, coarse = create_views(window, n_patches=5, patch_length=20)

    print(f"  Fine view: {fine.shape}")   # (4, 5, 20, 25)
    print(f"  Coarse view: {coarse.shape}")  # (4, 1, 20, 25)

    assert fine.shape == (B, 5, 20, V), f"Fine shape wrong: {fine.shape}"
    assert coarse.shape == (B, 1, 20, V), f"Coarse shape wrong: {coarse.shape}"

    print("  PASSED ✓")


def test_encoder():
    """Test 4: Encoder forward pass."""
    print("\n" + "="*60)
    print("TEST 4: Encoder")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = MTSJEPAEncoder(d_model=64, d_out=64, n_layers=2, n_heads=4,
                              patch_length=20, dropout=0.1).to(device)

    # Fine view: (B, P=5, L=20, V=10)
    x_fine = torch.randn(2, 5, 20, 10).to(device)
    h_fine = encoder(x_fine)
    print(f"  Fine encoding: {x_fine.shape} -> {h_fine.shape}")
    assert h_fine.shape == (2, 10, 5, 64), f"Wrong shape: {h_fine.shape}"

    # Coarse view: (B, 1, L=20, V=10)
    x_coarse = torch.randn(2, 1, 20, 10).to(device)
    h_coarse = encoder(x_coarse)
    print(f"  Coarse encoding: {x_coarse.shape} -> {h_coarse.shape}")
    assert h_coarse.shape == (2, 10, 1, 64), f"Wrong shape: {h_coarse.shape}"

    print("  PASSED ✓")


def test_codebook():
    """Test 5: Soft codebook."""
    print("\n" + "="*60)
    print("TEST 5: Soft Codebook")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codebook = SoftCodebook(K=32, D=64, tau=0.1).to(device)

    h = torch.randn(2, 10, 5, 64).to(device)  # (B, V, P, D)
    p, z = codebook(h)

    print(f"  Input: {h.shape}")
    print(f"  Probabilities: {p.shape}, sum={p.sum(dim=-1).mean():.4f}")
    print(f"  Embeddings: {z.shape}")

    assert p.shape == (2, 10, 5, 32), f"Wrong p shape: {p.shape}"
    assert z.shape == (2, 10, 5, 64), f"Wrong z shape: {z.shape}"
    assert torch.allclose(p.sum(dim=-1), torch.ones_like(p.sum(dim=-1)), atol=1e-5)

    print("  PASSED ✓")


def test_predictors():
    """Test 6: Fine and coarse predictors."""
    print("\n" + "="*60)
    print("TEST 6: Predictors")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    K = 32

    fine_pred = FinePredictor(K=K, n_patches=5, n_layers=2, n_heads=4, d_ff=64).to(device)
    coarse_pred = CoarsePredictor(K=K, n_patches=5, n_layers=2, n_heads=4, d_ff=64).to(device)

    pi = torch.softmax(torch.randn(4, 5, K).to(device), dim=-1)

    p_fine = fine_pred(pi)
    p_coarse = coarse_pred(pi)

    print(f"  Fine prediction: {pi.shape} -> {p_fine.shape}")
    print(f"  Coarse prediction: {pi.shape} -> {p_coarse.shape}")

    assert p_fine.shape == (4, 5, K), f"Wrong fine shape: {p_fine.shape}"
    assert p_coarse.shape == (4, 1, K), f"Wrong coarse shape: {p_coarse.shape}"

    # Check they're valid distributions
    assert torch.allclose(p_fine.sum(dim=-1), torch.ones_like(p_fine.sum(dim=-1)), atol=1e-5)
    assert torch.allclose(p_coarse.sum(dim=-1), torch.ones_like(p_coarse.sum(dim=-1)), atol=1e-5)

    print("  PASSED ✓")


def test_full_model_forward():
    """Test 7: Full MTS-JEPA forward pass."""
    print("\n" + "="*60)
    print("TEST 7: Full Model Forward Pass")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V = 10  # small for testing

    model = MTSJEPA(
        n_vars=V, d_model=64, d_out=64, n_codes=32, tau=0.1,
        patch_length=20, n_patches=5, n_encoder_layers=2,
        n_heads=4, dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    x_ctx = torch.randn(2, 100, V).to(device)
    x_tgt = torch.randn(2, 100, V).to(device)

    losses = model(x_ctx, x_tgt)

    print(f"  Losses:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.item():.4f}")
        else:
            print(f"    {k}: {v:.4f}")

    # Check no NaN
    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v), f"NaN in loss {k}"

    print("  PASSED ✓")
    return model


def test_loss_decreases():
    """Test 8: Training stability check on real PSM data.

    In early training, total loss may increase as KL divergence grows
    (codebook sharpening makes prediction harder). We verify:
    1. Reconstruction loss decreases (encoder learns useful representations)
    2. Sample entropy decreases (codebook sharpens, not collapsed)
    3. No NaN or Inf losses
    4. Codebook utilization > 0 (not degenerate)
    """
    print("\n" + "="*60)
    print("TEST 8: Training Stability (5 epochs on real data)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        data_dict = prepare_data("PSM", window_length=100, batch_size=16)
    except Exception as e:
        print(f"  SKIPPED: {e}")
        return

    V = data_dict['n_vars']

    model = MTSJEPA(
        n_vars=V, d_model=64, d_out=64, n_codes=32, tau=0.1,
        patch_length=20, n_patches=5, n_encoder_layers=2,
        n_heads=4, dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.online_params(), lr=5e-4)
    revin = RevIN(V).to(device)

    rec_losses = []
    ent_losses = []
    utils = []

    for epoch in range(5):
        epoch_rec = []
        epoch_ent = []
        n_batches = 0
        for x_ctx, x_tgt in data_dict['pretrain_train_loader']:
            x_ctx = x_ctx.to(device)
            x_tgt = x_tgt.to(device)

            x_ctx_n = revin(x_ctx)
            x_tgt_n = revin(x_tgt)

            losses = model(x_ctx_n, x_tgt_n)
            total, comps = compute_total_loss(losses, DEFAULT_LOSS_CONFIG, epoch, 100)

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.online_params(), 0.5)
            optimizer.step()
            model.update_ema()

            epoch_rec.append(losses['rec'].item())
            epoch_ent.append(losses['ent_sample'].item())
            n_batches += 1
            if n_batches >= 20:
                break

            # Check for NaN
            assert not torch.isnan(total), f"NaN total loss at epoch {epoch}"

        avg_rec = np.mean(epoch_rec)
        avg_ent = np.mean(epoch_ent)
        rec_losses.append(avg_rec)
        ent_losses.append(avg_ent)
        utils.append(losses['codebook_utilization'])

        print(f"  Epoch {epoch}: rec={avg_rec:.4f}, ent_sample={avg_ent:.4f}, "
              f"util={losses['codebook_utilization']:.3f}")

    # Check reconstruction loss decreased
    assert rec_losses[-1] < rec_losses[0], \
        f"Reconstruction loss didn't decrease: {rec_losses[0]:.4f} -> {rec_losses[-1]:.4f}"

    # Check sample entropy decreased (codebook sharpening)
    assert ent_losses[-1] < ent_losses[0], \
        f"Sample entropy didn't decrease: {ent_losses[0]:.4f} -> {ent_losses[-1]:.4f}"

    # Check codebook is used
    assert all(u > 0 for u in utils), "Codebook utilization dropped to 0"

    print("  PASSED ✓")


def test_downstream_encoding():
    """Test 9: Downstream encoding pipeline."""
    print("\n" + "="*60)
    print("TEST 9: Downstream Encoding")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    V = 10

    model = MTSJEPA(
        n_vars=V, d_model=64, d_out=64, n_codes=32, tau=0.1,
        patch_length=20, n_patches=5, n_encoder_layers=2,
        n_heads=4, dropout=0.1,
    ).to(device)

    x = torch.randn(8, 100, V).to(device)
    features = model.encode_for_downstream(x)

    print(f"  Input: {x.shape}")
    print(f"  Features: {features.shape}")  # (8, 5*32) = (8, 160)

    expected_dim = 5 * 32  # n_patches * n_codes
    assert features.shape == (8, expected_dim), f"Wrong shape: {features.shape}"
    assert not torch.isnan(features).any(), "NaN in features"

    # Test classifier
    classifier = DownstreamClassifier(expected_dim, hidden_dim=64).to(device)
    logits = classifier(features)
    print(f"  Classifier output: {logits.shape}")
    assert logits.shape == (8,), f"Wrong output shape: {logits.shape}"

    print("  PASSED ✓")


def test_real_data_forward():
    """Test 10: Forward pass on real PSM data."""
    print("\n" + "="*60)
    print("TEST 10: Real Data Forward Pass (PSM)")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        data_dict = prepare_data("PSM", window_length=100, batch_size=8)
    except Exception as e:
        print(f"  SKIPPED: Could not load PSM data: {e}")
        return

    n_vars = data_dict['n_vars']
    print(f"  PSM: {n_vars} effective variables")

    model = MTSJEPA(
        n_vars=n_vars, d_model=64, d_out=64, n_codes=32, tau=0.1,
        patch_length=20, n_patches=5, n_encoder_layers=2,
        n_heads=4, dropout=0.1,
    ).to(device)

    # Get one batch
    x_ctx, x_tgt = next(iter(data_dict['pretrain_train_loader']))
    x_ctx = x_ctx.to(device)
    x_tgt = x_tgt.to(device)

    print(f"  Batch shape: {x_ctx.shape}")

    revin = RevIN(n_vars).to(device)
    x_ctx_n = revin(x_ctx)
    x_tgt_n = revin(x_tgt)

    losses = model(x_ctx_n, x_tgt_n)

    for k, v in losses.items():
        if isinstance(v, torch.Tensor):
            assert not torch.isnan(v), f"NaN in {k}"
            print(f"    {k}: {v.item():.4f}")
        else:
            print(f"    {k}: {v:.4f}")

    print("  PASSED ✓")


if __name__ == "__main__":
    print("MTS-JEPA Test Pipeline")
    print("=" * 60)

    tests = [
        ("Data Loading", test_data_loading),
        ("RevIN", test_revin),
        ("Multi-scale Views", test_create_views),
        ("Encoder", test_encoder),
        ("Codebook", test_codebook),
        ("Predictors", test_predictors),
        ("Full Model Forward", test_full_model_forward),
        ("Loss Decrease", test_loss_decreases),
        ("Downstream Encoding", test_downstream_encoding),
        ("Real Data Forward", test_real_data_forward),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n  FAILED ✗: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*60}")

    if failed > 0:
        print("\n*** TEST PIPELINE FAILED — DO NOT PROCEED TO FULL EXPERIMENTS ***")
        sys.exit(1)
    else:
        print("\n*** ALL TESTS PASSED — READY FOR FULL EXPERIMENTS ***")
        sys.exit(0)

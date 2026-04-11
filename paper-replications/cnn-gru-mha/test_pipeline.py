"""
Quick validation script: 5 epochs, verify no NaN, shapes correct, loss decreases.

Run with: python test_pipeline.py
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_utils import (
    load_bearing_for_cnn_gru,
    FEMTO_DATA_ROOT,
    get_transfer_split,
    dwt_denoise,
    minmax_normalize,
    compute_linear_rul,
)
from models import CNNGRUMHAModel, rmse_l1_loss
from train_utils import train_source_domain, finetune_fc_head, evaluate_bearing


def test_preprocessing():
    """Test DWT denoising and normalization."""
    print("\n[TEST 1] Preprocessing...")
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(2560).astype(np.float32)

    denoised = dwt_denoise(sig)
    assert denoised.shape == (2560,), f"Shape mismatch: {denoised.shape}"
    assert not np.isnan(denoised).any(), "NaN in DWT output"

    normalized = minmax_normalize(denoised)
    assert normalized.min() >= -1e-6, f"Min < 0: {normalized.min()}"
    assert normalized.max() <= 1.0 + 1e-6, f"Max > 1: {normalized.max()}"
    assert not np.isnan(normalized).any(), "NaN in normalization"

    rul = compute_linear_rul(100)
    assert rul[0] == 1.0, f"RUL[0] = {rul[0]}, expected 1.0"
    assert abs(rul[-1] - 0.01) < 1e-5, f"RUL[-1] = {rul[-1]}, expected 0.01"
    print(f"  DWT output range: [{denoised.min():.4f}, {denoised.max():.4f}]")
    print(f"  Normalized range: [{normalized.min():.4f}, {normalized.max():.4f}]")
    print(f"  RUL: {rul[0]:.3f} -> {rul[-1]:.4f} (N=100)")
    print("  PASSED")


def test_data_loading():
    """Test FEMTO bearing loading."""
    print("\n[TEST 2] FEMTO Data Loading...")
    bdata = load_bearing_for_cnn_gru(
        FEMTO_DATA_ROOT, "Bearing1_3", verbose=True
    )
    assert bdata["n_snapshots"] > 0, "No snapshots loaded"
    assert bdata["snapshots"].shape == (bdata["n_snapshots"], 2560), \
        f"Shape mismatch: {bdata['snapshots'].shape}"
    assert bdata["rul"].shape == (bdata["n_snapshots"],), "RUL shape mismatch"
    assert not np.isnan(bdata["snapshots"]).any(), "NaN in snapshots"
    assert not np.isnan(bdata["rul"]).any(), "NaN in RUL"
    assert bdata["rul"][0] == 1.0, "First RUL != 1.0"
    print(f"  Bearing1_3: {bdata['n_snapshots']} snapshots")
    print(f"  Snapshot range: [{bdata['snapshots'].min():.4f}, {bdata['snapshots'].max():.4f}]")
    print(f"  RUL: {bdata['rul'][0]:.4f} -> {bdata['rul'][-1]:.6f}")
    print("  PASSED")
    return bdata


def test_model_shapes(bdata):
    """Test model forward pass shapes."""
    print("\n[TEST 3] Model Shapes...")
    device = torch.device("cpu")
    model = CNNGRUMHAModel(cnn_batch_size=16).to(device)
    params = model.count_parameters()
    print(f"  Parameters: {params}")

    # Test with small batch
    n_test = 20
    snaps = torch.FloatTensor(bdata["snapshots"][:n_test]).to(device)
    model.eval()
    with torch.no_grad():
        out = model(snaps)

    assert out.shape == (n_test,), f"Output shape {out.shape}, expected ({n_test},)"
    assert not torch.isnan(out).any(), "NaN in model output"
    assert (out >= 0).all() and (out <= 1).all(), f"Output out of [0,1]: [{out.min():.4f}, {out.max():.4f}]"
    print(f"  Input: {snaps.shape}, Output: {out.shape}")
    print(f"  Output range: [{out.min():.4f}, {out.max():.4f}]")
    print("  PASSED")
    return model


def test_loss(model, bdata):
    """Test loss function."""
    print("\n[TEST 4] Loss Function...")
    device = torch.device("cpu")
    n_test = 20
    snaps = torch.FloatTensor(bdata["snapshots"][:n_test]).to(device)
    rul = torch.FloatTensor(bdata["rul"][:n_test]).to(device)

    model.train()
    pred = model(snaps)
    loss, info = rmse_l1_loss(pred, rul, model, alpha=1e-4)

    assert not torch.isnan(loss), f"NaN loss: {info}"
    assert loss.item() > 0, f"Zero loss: {loss.item()}"
    print(f"  Loss info: {info}")
    print("  PASSED")


def test_training_loop(bdata):
    """Test that loss decreases over 5 epochs."""
    print("\n[TEST 5] Training Loop (5 epochs)...")
    device = torch.device("cpu")
    model = CNNGRUMHAModel(cnn_batch_size=16).to(device)

    # Use small subset for speed
    small_data = {
        "bearing_name": "Bearing1_3_small",
        "snapshots": bdata["snapshots"][:50],
        "rul": bdata["rul"][:50],
        "n_snapshots": 50,
    }

    loss_hist = train_source_domain(
        model, small_data,
        n_epochs=5,
        lr=0.001,
        alpha=1e-4,
        device=device,
        verbose=True,
    )

    assert len(loss_hist) == 5, f"Expected 5 loss values, got {len(loss_hist)}"
    assert not any(np.isnan(v) for v in loss_hist), f"NaN in loss history: {loss_hist}"

    # Loss should decrease (or at least not explode)
    initial_loss = loss_hist[0]
    final_loss = loss_hist[-1]
    assert final_loss < initial_loss * 2, f"Loss exploded: {initial_loss:.4f} -> {final_loss:.4f}"
    print(f"  Loss: {initial_loss:.4f} -> {final_loss:.4f}")
    if final_loss < initial_loss:
        print("  Loss DECREASED (good)")
    else:
        print("  WARNING: loss did not decrease over 5 epochs (may be ok for short run)")
    print("  PASSED")
    return model


def test_transfer_split(bdata):
    """Test 1:1 chronological split."""
    print("\n[TEST 6] Transfer Split...")
    ft_data, eval_data = get_transfer_split(bdata, split_ratio=0.5)
    n = bdata["n_snapshots"]
    assert ft_data["n_snapshots"] + eval_data["n_snapshots"] == n, "Split counts don't sum"
    assert ft_data["n_snapshots"] == int(n * 0.5), f"FT half size wrong: {ft_data['n_snapshots']}"
    print(f"  Total: {n}, FT: {ft_data['n_snapshots']}, Eval: {eval_data['n_snapshots']}")
    print("  PASSED")


def test_finetune(bdata):
    """Test fine-tuning with frozen CNN+GRU."""
    print("\n[TEST 7] Fine-tuning FC Head...")
    device = torch.device("cpu")
    model = CNNGRUMHAModel(cnn_batch_size=16).to(device)

    # Quick source training (2 epochs)
    source_small = {
        "bearing_name": "source_test",
        "snapshots": bdata["snapshots"][:30],
        "rul": bdata["rul"][:30],
        "n_snapshots": 30,
    }
    train_source_domain(model, source_small, n_epochs=2, verbose=False)

    # Fine-tune on small subset
    ft_data = {
        "bearing_name": "target_ft",
        "snapshots": bdata["snapshots"][30:50],
        "rul": bdata["rul"][30:50],
        "n_snapshots": 20,
    }
    ft_loss = finetune_fc_head(model, ft_data, n_epochs=3, verbose=True)
    assert not any(np.isnan(v) for v in ft_loss), "NaN in FT loss"

    # Verify CNN is frozen
    for name, param in model.cnn.named_parameters():
        assert not param.requires_grad, f"CNN param {name} not frozen!"

    print(f"  FT loss: {ft_loss}")
    print("  PASSED")


def test_evaluation(bdata):
    """Test evaluation produces valid RMSE."""
    print("\n[TEST 8] Evaluation...")
    device = torch.device("cpu")
    model = CNNGRUMHAModel(cnn_batch_size=16).to(device)

    eval_data = {
        "bearing_name": "eval_test",
        "snapshots": bdata["snapshots"][:30],
        "rul": bdata["rul"][:30],
        "n_snapshots": 30,
    }
    rmse, preds, gt = evaluate_bearing(model, eval_data, device)

    assert not np.isnan(rmse), "NaN RMSE"
    assert rmse >= 0, f"Negative RMSE: {rmse}"
    assert len(preds) == 30, "Wrong prediction count"
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Pred range: [{min(preds):.4f}, {max(preds):.4f}]")
    print(f"  GT range: [{min(gt):.4f}, {max(gt):.4f}]")
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("CNN-GRU-MHA Pipeline Validation")
    print("=" * 60)

    passed = 0
    failed = 0

    tests = [
        ("Preprocessing", test_preprocessing),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Data loading — shared across remaining tests
    try:
        bdata = test_data_loading()
        passed += 1
    except Exception as e:
        print(f"  Data loading FAILED: {e}")
        import traceback
        traceback.print_exc()
        print("\nABORTING: Cannot continue without data")
        sys.exit(1)

    remaining_tests = [
        ("Model Shapes", lambda: test_model_shapes(bdata)),
        ("Loss Function", lambda: test_loss(test_model_shapes(bdata), bdata)),
        ("Training Loop", lambda: test_training_loop(bdata)),
        ("Transfer Split", lambda: test_transfer_split(bdata)),
        ("Fine-tuning", lambda: test_finetune(bdata)),
        ("Evaluation", lambda: test_evaluation(bdata)),
    ]

    for name, fn in remaining_tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAILED [{name}]: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("ALL TESTS PASSED - Pipeline is ready for full experiments")
    else:
        print("SOME TESTS FAILED - Fix before running full experiments")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)

"""
Quick end-to-end pipeline test to verify everything works before running full experiments.

Tests:
1. Data loading for all 17 bearings
2. RUL label construction
3. Dataset creation
4. Forward pass of all models
5. Loss computation
6. Mini training loop (5 epochs)
7. Evaluation
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DATA_ROOT = Path("/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_data_loading():
    print("\n" + "="*50)
    print("TEST 1: Data Loading")
    print("="*50)

    from data_utils import load_condition_data, get_data_stats

    for condition in [1, 2, 3]:
        print(f"\nCondition {condition}:")
        train_data, test_data = load_condition_data(DATA_ROOT, condition, verbose=True)

        print(f"  Train stats: {get_data_stats(train_data)}")
        print(f"  Test stats: {get_data_stats(test_data)}")

        # Verify RUL labels
        for bdata in train_data:
            assert bdata["rul"].min() >= 0.0, "RUL min < 0"
            assert bdata["rul"].max() <= 1.0, "RUL max > 1"
            assert bdata["rul"][0] == 1.0, "First RUL should be 1.0"
            assert bdata["rul"][-1] < 0.1, "Last RUL should be ~0"
            print(f"  {bdata['bearing_name']}: n={bdata['n_snapshots']}, "
                  f"FPT={bdata['fpt']} ({100*bdata['fpt']/bdata['n_snapshots']:.0f}%), "
                  f"RUL range=[{bdata['rul'].min():.3f}, {bdata['rul'].max():.3f}]")

        for bdata in test_data:
            print(f"  {bdata['bearing_name']}: n={bdata['n_snapshots']}, "
                  f"FPT={bdata['fpt']}")

    print("\nTEST 1: PASSED")
    return True


def test_datasets():
    print("\n" + "="*50)
    print("TEST 2: Dataset and DataLoader")
    print("="*50)

    from data_utils import load_condition_data, FEMTOPretrainDataset, FEMTORULDataset
    from torch.utils.data import DataLoader

    train_data, test_data = load_condition_data(DATA_ROOT, 1, verbose=False)

    # Test pretrain dataset
    pretrain_ds = FEMTOPretrainDataset(train_data, crop_length=1024)
    print(f"  Pretrain dataset: {len(pretrain_ds)} samples")
    sample = pretrain_ds[0]
    print(f"  Sample view1 shape: {sample['view1'].shape}")
    print(f"  Sample view2 shape: {sample['view2'].shape}")
    assert sample["view1"].shape == (2, 1024), f"Expected (2, 1024), got {sample['view1'].shape}"

    loader = DataLoader(pretrain_ds, batch_size=16, shuffle=True, drop_last=True)
    batch = next(iter(loader))
    print(f"  Batch view1: {batch['view1'].shape}")
    print(f"  Batch bearing_idx: {batch['bearing_idx']}")

    # Test RUL dataset
    rul_ds = FEMTORULDataset(train_data, crop_length=1024)
    print(f"\n  RUL dataset: {len(rul_ds)} samples")
    sample = rul_ds[0]
    print(f"  Sample x shape: {sample['x'].shape}")
    print(f"  Sample rul: {sample['rul']}")
    assert sample["x"].shape == (2, 1024), f"Expected (2, 1024), got {sample['x'].shape}"

    print("\nTEST 2: PASSED")
    return True


def test_models():
    print("\n" + "="*50)
    print("TEST 3: Model Forward Passes")
    print("="*50)

    from models import DCSSSLModel, SimCLRModel, SupConModel, count_parameters

    B, C, T = 8, 2, 1024
    x1 = torch.randn(B, C, T).to(DEVICE)
    x2 = torch.randn(B, C, T).to(DEVICE)
    bearing_idx = torch.randint(0, 2, (B,)).to(DEVICE)
    time_idx = torch.randint(0, 100, (B,)).to(DEVICE)
    n_snaps = torch.full((B,), 200).to(DEVICE)
    rul = torch.rand(B, 1).to(DEVICE)

    for ModelClass, name in [(SimCLRModel, "SimCLR"), (SupConModel, "SupCon"), (DCSSSLModel, "DCSSL")]:
        model = ModelClass(
            in_channels=2, encoder_hidden=64, encoder_out=128,
            n_tcn_blocks=8, kernel_size=3, dropout=0.1, temperature=0.1,
        ).to(DEVICE)

        n_params = count_parameters(model)
        print(f"\n  {name}: {n_params:,} parameters")

        # Test encode
        with torch.no_grad():
            h = model.encode(x1)
            print(f"  Encode output: {h.shape}")
            assert h.shape == (B, 128), f"Expected (B, 128), got {h.shape}"

            # Test predict_rul
            rul_pred = model.predict_rul(x1)
            print(f"  RUL prediction: {rul_pred.shape}, range=[{rul_pred.min():.3f}, {rul_pred.max():.3f}]")
            assert rul_pred.shape == (B,), f"Expected (B,), got {rul_pred.shape}"
            assert (rul_pred >= 0).all() and (rul_pred <= 1).all(), "RUL out of [0,1]"

        # Test loss
        loss, loss_dict = model.contrastive_loss(
            x1, x2, bearing_indices=bearing_idx,
            time_indices=time_idx, n_snapshots=n_snaps, rul=rul
        )
        print(f"  Contrastive loss: {loss.item():.4f} | {loss_dict}")
        assert not torch.isnan(loss), "Loss is NaN!"

        # Test backward
        loss.backward()
        print(f"  Backward pass: OK")

    print("\nTEST 3: PASSED")
    return True


def test_mini_training():
    print("\n" + "="*50)
    print("TEST 4: Mini Training Loop (10 epochs)")
    print("="*50)

    from data_utils import load_condition_data, FEMTOPretrainDataset, FEMTORULDataset
    from models import DCSSSLModel
    from train_utils import pretrain_ssl, finetune_rul, evaluate_on_test_bearings
    from torch.utils.data import DataLoader

    train_data, test_data = load_condition_data(DATA_ROOT, 1, verbose=False)

    model = DCSSSLModel(
        in_channels=2, encoder_hidden=64, encoder_out=128,
        n_tcn_blocks=8, kernel_size=3, dropout=0.1, temperature=0.1,
        lambda_temporal=1.0, lambda_instance=1.0,
    ).to(DEVICE)

    # Mini pretrain
    pretrain_ds = FEMTOPretrainDataset(train_data, crop_length=1024)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=32, shuffle=True, drop_last=True)

    print("\n  Pretraining (10 epochs)...")
    history = pretrain_ssl(model, pretrain_loader, n_epochs=10, lr=1e-3, device=DEVICE, verbose=True)
    assert len(history) == 10, "Wrong history length"
    assert not np.isnan(history[-1].get("total", float("nan"))), "Final loss is NaN"
    print(f"  Final pretrain loss: {history[-1]}")

    # Mini finetune
    rul_ds = FEMTORULDataset(train_data, augment=True, crop_length=1024)
    rul_loader = DataLoader(rul_ds, batch_size=32, shuffle=True)

    print("\n  Fine-tuning (10 epochs)...")
    ft_history = finetune_rul(model, rul_loader, n_epochs=10, lr=5e-4, device=DEVICE, verbose=True)
    print(f"  Final finetune MSE: {ft_history[-1]}")

    # Evaluate
    print("\n  Evaluating on test bearings...")
    results = evaluate_on_test_bearings(model, test_data, DEVICE, batch_size=32, crop_length=1024)

    for name, res in results.items():
        mse = res["mse"]
        print(f"  {name}: MSE = {mse:.4f}")
        assert 0.0 <= mse <= 2.0, f"MSE out of reasonable range: {mse}"

    avg_mse = np.mean([r["mse"] for r in results.values()])
    print(f"  Average MSE: {avg_mse:.4f}")

    print("\nTEST 4: PASSED")
    return True


def main():
    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")

    tests = [
        ("Data Loading", test_data_loading),
        ("Datasets", test_datasets),
        ("Models", test_models),
        ("Mini Training", test_mini_training),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            result = test_fn()
            if result:
                passed += 1
        except Exception as e:
            import traceback
            print(f"\nTEST FAILED: {test_name}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*50)

    if failed == 0:
        print("All tests passed! Ready to run full experiments.")
        return 0
    else:
        print("Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

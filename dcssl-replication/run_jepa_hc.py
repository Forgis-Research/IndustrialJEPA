"""
Run JEPA+HC baseline on FEMTO for comparison with DCSSL.

This script:
1. Pretrains TCN encoder with JEPA-style masked prediction
2. Fine-tunes with HC features
3. Evaluates on test bearings

Usage:
    python run_jepa_hc.py --condition 1
    python run_jepa_hc.py --condition all
"""

import argparse
import json
import warnings
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from data_utils import load_condition_data, FEMTOPretrainDataset
from models import count_parameters
from train_utils import pretrain_ssl
from jepa_femto import JEPAHCModel, FEMTORULDatasetHC, finetune_jepa_hc, evaluate_jepa_hc
from torch.utils.data import DataLoader

CONDITION_INFO = {
    1: {"train": ["Bearing1_1", "Bearing1_2"],
        "test": ["Bearing1_3", "Bearing1_4", "Bearing1_5", "Bearing1_6", "Bearing1_7"]},
    2: {"train": ["Bearing2_1", "Bearing2_2"],
        "test": ["Bearing2_3", "Bearing2_4", "Bearing2_5", "Bearing2_6", "Bearing2_7"]},
    3: {"train": ["Bearing3_1", "Bearing3_2"],
        "test": ["Bearing3_3"]},
}


def run_jepa_hc_condition(
    condition: int,
    data_root: Path,
    output_dir: Path,
    pretrain_epochs: int = 300,
    finetune_epochs: int = 150,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 5e-4,
    batch_size: int = 64,
    crop_length: int = 1024,
    device: torch.device = None,
    verbose: bool = True,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"JEPA+HC Hybrid — Condition {condition}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading condition {condition} data...")
    train_data, test_data = load_condition_data(data_root, condition, verbose=verbose)

    if not train_data or not test_data:
        return {"error": "Data loading failed"}

    # Create model
    model = JEPAHCModel(
        in_channels=2, encoder_hidden=64, encoder_out=128,
        n_tcn_blocks=8, kernel_size=3, dropout=0.1,
        n_hc_features=18, rul_hidden=128,
    ).to(device)
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Stage 1: JEPA pretraining
    print(f"\nStage 1: JEPA Pretraining ({pretrain_epochs} epochs)")
    pretrain_ds = FEMTOPretrainDataset(train_data, crop_length=crop_length)
    pretrain_loader = DataLoader(
        pretrain_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    exp_dir = output_dir / f"jepa_hc_cond{condition}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    pretrain_ckpt = exp_dir / "jepa_hc_pretrain.pt"

    pretrain_history = pretrain_ssl(
        model, pretrain_loader,
        n_epochs=pretrain_epochs, lr=pretrain_lr,
        device=device, verbose=verbose,
        checkpoint_path=pretrain_ckpt,
    )

    if pretrain_ckpt.exists():
        ckpt = torch.load(pretrain_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best pretrained (loss={ckpt['loss']:.4f})")

    # Stage 2: Fine-tune with HC features
    print(f"\nStage 2: Fine-tuning with HC features ({finetune_epochs} epochs)")
    train_rul_ds = FEMTORULDatasetHC(
        train_data, augment=True, crop_length=crop_length, normalize_hc=True
    )
    train_rul_loader = DataLoader(
        train_rul_ds, batch_size=batch_size, shuffle=True, num_workers=2
    )

    ft_history = finetune_jepa_hc(
        model, train_rul_loader, n_epochs=finetune_epochs,
        lr=finetune_lr, device=device, verbose=verbose,
    )

    # Stage 3: Evaluation
    print(f"\nStage 3: Evaluation")
    results = evaluate_jepa_hc(
        model, test_data, device,
        batch_size=batch_size, crop_length=crop_length,
        train_feat_mean=train_rul_ds.feat_mean,
        train_feat_std=train_rul_ds.feat_std,
    )

    mse_values = [r["mse"] for r in results.values()]
    avg_mse = float(np.mean(mse_values))

    print(f"\n  Results:")
    for name, res in results.items():
        print(f"    {name}: MSE = {res['mse']:.4f}")
    print(f"    Average: MSE = {avg_mse:.4f}")

    full_results = {
        "model": "jepa_hc",
        "condition": condition,
        "avg_mse": avg_mse,
        "per_bearing": results,
        "pretrain_history": pretrain_history[-10:] if pretrain_history else [],
        "finetune_history": ft_history[-10:] if ft_history else [],
    }

    results_path = exp_dir / "jepa_hc_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"  Saved to {results_path}")

    return full_results


def main():
    parser = argparse.ArgumentParser(description="JEPA+HC on FEMTO")
    parser.add_argument("--data_root", type=str,
                        default="/mnt/sagemaker-nvme/femto_data/10. FEMTO Bearing")
    parser.add_argument("--output_dir", type=str,
                        default="/home/sagemaker-user/IndustrialJEPA/dcssl-replication/results")
    parser.add_argument("--condition", type=str, default="all",
                        help="Condition to run: 1, 2, 3, or 'all'")
    parser.add_argument("--pretrain_epochs", type=int, default=300)
    parser.add_argument("--finetune_epochs", type=int, default=150)
    parser.add_argument("--pretrain_lr", type=float, default=1e-3)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--crop_length", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [1, 2, 3] if args.condition == "all" else [int(args.condition)]

    all_results = {}
    for cond in conditions:
        results = run_jepa_hc_condition(
            condition=cond,
            data_root=data_root,
            output_dir=output_dir,
            pretrain_epochs=args.pretrain_epochs,
            finetune_epochs=args.finetune_epochs,
            pretrain_lr=args.pretrain_lr,
            finetune_lr=args.finetune_lr,
            batch_size=args.batch_size,
            crop_length=args.crop_length,
            device=device,
        )
        all_results[f"cond{cond}"] = results

    # Final results summary
    print("\n" + "="*60)
    print("JEPA+HC Final Summary")
    print("="*60)
    all_mse = []
    for cond_key, res in all_results.items():
        if "error" not in res:
            for name, r in res.get("per_bearing", {}).items():
                print(f"  {name}: {r['mse']:.4f}")
                all_mse.append(r["mse"])

    if all_mse:
        print(f"\n  Overall Average MSE: {np.mean(all_mse):.4f}")

    # Save combined results
    final_path = output_dir / "jepa_hc_all_results.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {final_path}")


if __name__ == "__main__":
    main()

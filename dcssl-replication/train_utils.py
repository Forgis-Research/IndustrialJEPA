"""
Training utilities for DCSSL replication.

Handles:
- SSL pretraining loop
- Supervised fine-tuning loop
- Evaluation / MSE computation
- Checkpoint save/load
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =====================================================================
# SSL Pretraining
# =====================================================================

def pretrain_ssl(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
    checkpoint_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Self-supervised pretraining loop.

    Returns:
        List of per-epoch loss dicts
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    best_loss = float("inf")

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {}
        n_batches = 0

        for batch in train_loader:
            view1 = batch["view1"].to(device)
            view2 = batch["view2"].to(device)
            bearing_idx = batch["bearing_idx"].to(device)
            time_idx = batch["time_idx"].to(device)
            n_snapshots = batch["n_snapshots"].to(device)
            rul = batch.get("rul", None)
            if rul is not None:
                rul = rul.to(device)

            # Compute contrastive loss (handles both DCSSL and SimCLR/SupCon)
            loss, loss_dict = model.contrastive_loss(
                view1, view2,
                bearing_indices=bearing_idx,
                time_indices=time_idx,
                n_snapshots=n_snapshots,
                rul=rul,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            n_batches += 1

        scheduler.step()

        if n_batches > 0:
            avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        else:
            avg_losses = {"total": float("nan")}

        history.append(avg_losses)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            loss_str = " | ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
            print(f"  Epoch {epoch+1:4d}/{n_epochs} | {loss_str}")

        # Save best checkpoint
        if checkpoint_path is not None and avg_losses.get("total", float("inf")) < best_loss:
            best_loss = avg_losses.get("total", float("inf"))
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, checkpoint_path)

    return history


# =====================================================================
# Fine-tuning
# =====================================================================

def finetune_rul(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    n_epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
    freeze_encoder: bool = False,
    checkpoint_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Fine-tune (or train from scratch) the RUL prediction head.

    Args:
        freeze_encoder: if True, only train the prediction head
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    criterion = nn.MSELoss()

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.rul_head.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    best_val_mse = float("inf")

    for epoch in range(n_epochs):
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            rul = batch["rul"].to(device).squeeze()

            pred = model.predict_rul(x)
            if pred.dim() > 1:
                pred = pred.squeeze()

            loss = criterion(pred, rul)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        avg_train = np.mean(train_losses) if train_losses else float("nan")
        epoch_info = {"train_mse": avg_train}

        # Validation
        if val_loader is not None:
            val_mse = evaluate_rul(model, val_loader, device, return_predictions=False)
            epoch_info["val_mse"] = val_mse

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                if checkpoint_path is not None:
                    torch.save(model.state_dict(), checkpoint_path)

        history.append(epoch_info)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            info_str = f"  Epoch {epoch+1:4d}/{n_epochs} | Train MSE={avg_train:.4f}"
            if "val_mse" in epoch_info:
                info_str += f" | Val MSE={epoch_info['val_mse']:.4f}"
            print(info_str)

    # Re-enable encoder params if frozen
    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = True

    return history


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_rul(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    return_predictions: bool = True,
) -> float | Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate MSE on a data loader.

    Returns:
        If return_predictions: (mse, predictions, targets)
        Else: mse
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch["x"].to(device)
            rul = batch["rul"].squeeze().cpu().numpy()

            pred = model.predict_rul(x)
            if pred.dim() > 1:
                pred = pred.squeeze()
            pred = pred.cpu().numpy()

            # Handle single-sample batches
            if pred.ndim == 0:
                pred = pred.reshape(1)
            if rul.ndim == 0:
                rul = rul.reshape(1)

            all_preds.append(pred)
            all_targets.append(rul)

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    mse = float(np.mean((preds - targets) ** 2))

    if return_predictions:
        return mse, preds, targets
    return mse


def evaluate_on_test_bearings(
    model: nn.Module,
    test_data_list: List[Dict],
    device: torch.device,
    batch_size: int = 64,
    crop_length: int = 2560,
) -> Dict:
    """
    Evaluate MSE on each test bearing individually.

    Returns:
        Dict mapping bearing_name → {mse, predictions, targets}
    """
    from data_utils import FEMTORULDataset

    results = {}
    for bdata in test_data_list:
        bearing_name = bdata["bearing_name"]
        dataset = FEMTORULDataset([bdata], augment=False, crop_length=crop_length)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        mse, preds, targets = evaluate_rul(model, loader, device, return_predictions=True)

        results[bearing_name] = {
            "mse": mse,
            "predictions": preds.tolist(),
            "targets": targets.tolist(),
            "n_snapshots": bdata["n_snapshots"],
        }

    return results


# =====================================================================
# Full Pipeline
# =====================================================================

def run_full_pipeline(
    model: nn.Module,
    train_data: List[Dict],
    test_data: List[Dict],
    output_dir: Path,
    model_name: str = "model",
    pretrain_epochs: int = 200,
    finetune_epochs: int = 100,
    pretrain_lr: float = 1e-3,
    finetune_lr: float = 5e-4,
    batch_size: int = 64,
    crop_length: int = 1024,
    device: torch.device = None,
    skip_pretrain: bool = False,
    verbose: bool = True,
) -> Dict:
    """
    Full pipeline: pretrain → finetune → evaluate.

    Returns:
        Results dict with per-bearing MSE and average
    """
    from data_utils import FEMTOPretrainDataset, FEMTORULDataset

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1: SSL Pretraining ----
    if not skip_pretrain:
        if verbose:
            print(f"\n[{model_name}] Stage 1: SSL Pretraining ({pretrain_epochs} epochs)")
        pretrain_dataset = FEMTOPretrainDataset(train_data, crop_length=crop_length)
        pretrain_loader = DataLoader(
            pretrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=2, pin_memory=True,
        )
        pretrain_ckpt = output_dir / f"{model_name}_pretrain_best.pt"
        pretrain_history = pretrain_ssl(
            model, pretrain_loader,
            n_epochs=pretrain_epochs, lr=pretrain_lr,
            device=device, verbose=verbose,
            checkpoint_path=pretrain_ckpt,
        )
        # Load best pretrained weights
        if pretrain_ckpt.exists():
            ckpt = torch.load(pretrain_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            if verbose:
                print(f"  Loaded best pretrained checkpoint (loss={ckpt['loss']:.4f})")
    else:
        if verbose:
            print(f"\n[{model_name}] Skipping pretraining")
        pretrain_history = []

    # ---- Stage 2: Fine-tuning ----
    if verbose:
        print(f"\n[{model_name}] Stage 2: Fine-tuning ({finetune_epochs} epochs)")

    finetune_dataset = FEMTORULDataset(train_data, augment=True, crop_length=crop_length)
    finetune_loader = DataLoader(
        finetune_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )
    finetune_ckpt = output_dir / f"{model_name}_finetune_best.pt"
    finetune_history = finetune_rul(
        model, finetune_loader,
        n_epochs=finetune_epochs, lr=finetune_lr,
        device=device, verbose=verbose,
        freeze_encoder=False,
        checkpoint_path=finetune_ckpt,
    )
    if finetune_ckpt.exists():
        model.load_state_dict(torch.load(finetune_ckpt, map_location=device))
        if verbose:
            print(f"  Loaded best fine-tuned checkpoint")

    # ---- Stage 3: Evaluation ----
    if verbose:
        print(f"\n[{model_name}] Stage 3: Evaluation")
    test_results = evaluate_on_test_bearings(
        model, test_data, device, batch_size=batch_size, crop_length=crop_length
    )

    mse_values = [v["mse"] for v in test_results.values()]
    avg_mse = float(np.mean(mse_values))

    if verbose:
        print(f"\n  Results:")
        for name, res in test_results.items():
            print(f"    {name}: MSE = {res['mse']:.4f}")
        print(f"    Average: MSE = {avg_mse:.4f}")

    # Save results
    full_results = {
        "model": model_name,
        "avg_mse": avg_mse,
        "per_bearing": test_results,
        "pretrain_history": pretrain_history[-10:] if pretrain_history else [],
        "finetune_history": finetune_history[-10:] if finetune_history else [],
    }
    results_path = output_dir / f"{model_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(full_results, f, indent=2)

    return full_results

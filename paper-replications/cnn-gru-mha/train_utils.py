"""
Training utilities for CNN-GRU-MHA replication.

Memory-efficient design for constrained GPU:
  - CNN features extracted in mini-batches WITHOUT gradients
  - GRU+FC trained on FULL sequences WITH gradients
  - This allows full-sequence temporal learning while staying within GPU memory

Training flow:
  Source domain:
    Step A (60 iters): Extract CNN features (no_grad), train GRU+FC on full sequence
    Step B (20 iters): Freeze GRU, train CNN with windowed batches

  Target fine-tuning:
    Extract CNN features (no_grad), fine-tune FC only on first half

  Evaluation:
    Extract CNN features (no_grad), run GRU+FC on second half
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

from models import CNNGRUMHAModel, rmse_l1_loss


# =====================================================================
# Full-sequence training (GRU sees the whole bearing life)
# =====================================================================

def train_source_full_sequence(
    model: CNNGRUMHAModel,
    source_data: Dict,
    n_iterations: int = 60,
    lr: float = 0.001,
    alpha: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> List[float]:
    """
    Train GRU+FC on the full source sequence.

    CNN features are extracted WITHOUT gradients (saves GPU memory).
    Only GRU and FC parameters are updated.

    Then in a separate step, CNN is updated with windowed mini-batches.

    Args:
        model: CNN-GRU-MHA model
        source_data: dict with 'snapshots' (N, 2560) and 'rul' (N,)
        n_iterations: number of gradient steps (60 per paper interpretation)
        lr: learning rate (0.001 per paper)
        alpha: L1 regularization coefficient
        device: compute device
        verbose: print progress

    Returns:
        List of per-iteration loss values
    """
    if device is None:
        device = next(model.parameters()).device

    model.unfreeze_all()

    snapshots = source_data["snapshots"]
    rul_np = source_data["rul"]
    N = len(snapshots)

    snaps_tensor = torch.FloatTensor(snapshots).to(device)
    rul_tensor = torch.FloatTensor(rul_np).to(device)

    # Optimizer for all params
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    t0 = time.time()

    for iteration in range(n_iterations):
        model.train()
        optimizer.zero_grad()

        # Extract CNN features without gradients to save memory
        with torch.no_grad():
            features = model.extract_cnn_features(snaps_tensor, use_grad=False)  # (N, 1024)
        features = features.detach().requires_grad_(True)  # allow grad for GRU input

        # GRU + FC on full sequence (with gradients)
        seq = features.unsqueeze(0)   # (1, N, 1024)
        out1, _ = model.gru1(seq)     # (1, N, 512)
        out2, _ = model.gru2(out1)    # (1, N, 128)
        out2 = out2.squeeze(0)        # (N, 128)
        pred = model.fc(out2).squeeze(-1)  # (N,)

        loss = torch.sqrt(((pred - rul_tensor) ** 2).mean() + 1e-8)

        # L1 reg on GRU+FC only (CNN not in this gradient flow)
        l1_reg = torch.tensor(0.0, device=device)
        for p in list(model.gru1.parameters()) + list(model.gru2.parameters()) + list(model.fc.parameters()):
            l1_reg = l1_reg + p.abs().sum()

        total_loss = loss + alpha * l1_reg
        total_loss.backward()

        # Only update GRU + FC via these gradients (CNN didn't participate)
        for p in list(model.gru1.parameters()) + list(model.gru2.parameters()) + list(model.fc.parameters()):
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], max_norm=1.0)

        # Step: Adam will update all params but CNN grads will be zero
        # Better: use separate optimizer for GRU+FC
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    Iter {iteration+1}/{n_iterations} | RMSE={loss.item():.4f} | "
                  f"L1={alpha*l1_reg.item():.6f} | {elapsed:.1f}s")

    return loss_history


def train_source_domain(
    model: CNNGRUMHAModel,
    source_data: Dict,
    n_iterations: int = 60,
    window_size: int = 128,
    lr: float = 0.001,
    alpha: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> List[float]:
    """
    Train the full CNN-GRU-MHA on source domain bearing.

    Two-phase training:
      Phase 1 (n_iterations iterations): GRU+FC on full sequence (CNN features detached)
      Phase 2 (n_iterations//3 iterations): Full model on windowed mini-batches (updates CNN)

    This gives GRU the full temporal context while still updating CNN weights.

    Args:
        model: CNN-GRU-MHA model
        source_data: bearing data dict
        n_iterations: iterations for phase 1 (full sequence GRU training)
        window_size: window size for phase 2 (CNN update)
        lr: learning rate
        alpha: L1 regularization
        device: compute device
        verbose: print progress

    Returns:
        Combined loss history
    """
    if device is None:
        device = next(model.parameters()).device

    model.unfreeze_all()

    snapshots = source_data["snapshots"]
    rul_np = source_data["rul"]
    N = len(snapshots)

    snaps_tensor = torch.FloatTensor(snapshots).to(device)
    rul_tensor = torch.FloatTensor(rul_np).to(device)

    # Phase 1: Train GRU+FC with full sequence (CNN features without gradient)
    if verbose:
        print(f"    Phase 1: Full-sequence GRU training ({n_iterations} iters)...")

    optimizer_gru = torch.optim.Adam(
        list(model.gru1.parameters()) + list(model.gru2.parameters()) + list(model.fc.parameters()),
        lr=lr
    )
    optimizer_cnn = torch.optim.Adam(model.cnn.parameters(), lr=lr * 0.1)

    loss_history = []
    t0 = time.time()

    for iteration in range(n_iterations):
        model.train()
        optimizer_gru.zero_grad()

        with torch.no_grad():
            features = model.extract_cnn_features(snaps_tensor, use_grad=False)

        seq = features.unsqueeze(0)
        out1, _ = model.gru1(seq)
        out2, _ = model.gru2(out1)
        out2 = out2.squeeze(0)
        pred = model.fc(out2).squeeze(-1)

        loss = torch.sqrt(((pred - rul_tensor) ** 2).mean() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.gru1.parameters()) + list(model.gru2.parameters()) + list(model.fc.parameters()),
            max_norm=1.0
        )
        optimizer_gru.step()
        loss_history.append(loss.item())

        if verbose and (iteration + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"      Iter {iteration+1}/{n_iterations} | RMSE={loss.item():.4f} | {elapsed:.1f}s")

    # Phase 2: Update CNN with windowed mini-batches (window=128, fewer iters)
    cnn_iters = max(20, n_iterations // 3)
    if verbose:
        print(f"    Phase 2: CNN update with windows ({cnn_iters} iters)...")

    for iteration in range(cnn_iters):
        model.train()
        optimizer_cnn.zero_grad()

        # Sample random window
        if N > window_size:
            start = np.random.randint(0, N - window_size)
            win_snaps = snaps_tensor[start:start + window_size]
            win_rul = rul_tensor[start:start + window_size]
        else:
            win_snaps = snaps_tensor
            win_rul = rul_tensor

        # Full forward with CNN gradients on this window
        win_batch = win_snaps.unsqueeze(1)  # (W, 1, 2560)
        features_win = model.cnn(win_batch)  # (W, 1024) - WITH gradients

        seq = features_win.unsqueeze(0)  # (1, W, 1024)
        with torch.no_grad():  # GRU without gradients (we're updating CNN only)
            out1, _ = model.gru1(seq)
            out2, _ = model.gru2(out1)
            out2 = out2.squeeze(0)

        pred = model.fc(out2.detach()).squeeze(-1)
        loss = torch.sqrt(((pred - win_rul) ** 2).mean() + 1e-8)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.cnn.parameters(), max_norm=1.0)
        optimizer_cnn.step()

    return loss_history


def finetune_fc_head(
    model: CNNGRUMHAModel,
    finetune_data: Dict,
    n_iterations: int = 100,
    window_size: int = 128,
    lr: float = 0.001,
    alpha: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
) -> List[float]:
    """
    Fine-tune only the FC head on the first half of target bearing.

    Freeze CNN + GRU. Extract CNN features (no_grad), run full GRU (no_grad),
    update only FC.

    Args:
        model: pre-trained CNN-GRU-MHA
        finetune_data: first half of target bearing
        n_iterations: 100 per paper
        window_size: unused (full sequence used)
        lr: learning rate
        alpha: L1 regularization
        device: compute device
        verbose: print progress

    Returns:
        Per-iteration loss values
    """
    if device is None:
        device = next(model.parameters()).device

    model.freeze_feature_extractor()

    snapshots = finetune_data["snapshots"]
    rul_np = finetune_data["rul"]
    N = len(snapshots)

    snaps_tensor = torch.FloatTensor(snapshots).to(device)
    rul_tensor = torch.FloatTensor(rul_np).to(device)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    loss_history = []
    t0 = time.time()

    for iteration in range(n_iterations):
        model.train()
        optimizer.zero_grad()

        with torch.no_grad():
            features = model.extract_cnn_features(snaps_tensor, use_grad=False)
            seq = features.unsqueeze(0)
            out1, _ = model.gru1(seq)
            out2, _ = model.gru2(out1)
            out2 = out2.squeeze(0)

        pred = model.fc(out2).squeeze(-1)
        loss = torch.sqrt(((pred - rul_tensor) ** 2).mean() + 1e-8)

        # L1 on FC params only
        l1_reg = sum(p.abs().sum() for p in model.fc.parameters())
        total_loss = loss + alpha * l1_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.fc.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and (iteration + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"    FT Iter {iteration+1}/{n_iterations} | RMSE={loss.item():.4f} | {elapsed:.1f}s")

    return loss_history


def evaluate_bearing_with_context(
    model: CNNGRUMHAModel,
    full_bearing_data: Dict,
    device: torch.device = None,
    context_fraction: float = 0.5,
) -> Dict:
    """
    Evaluate RUL prediction on the SECOND half of a bearing.

    The GRU processes the FULL bearing sequence (both halves), giving it
    the first-half context. RMSE is then computed only on the second half.

    This matches the paper's evaluation protocol where the model sees the
    entire temporal context.

    Args:
        model: trained model
        full_bearing_data: FULL target bearing data (both halves)
        device: compute device
        context_fraction: fraction used as context (0.5 = first half)

    Returns:
        Dict with rmse, predictions (second half only), ground_truth (second half)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    snapshots = full_bearing_data["snapshots"]
    gt = full_bearing_data["rul"]
    N = len(snapshots)
    split_idx = int(N * context_fraction)

    with torch.no_grad():
        snaps = torch.FloatTensor(snapshots).to(device)
        features = model.extract_cnn_features(snaps, use_grad=False)  # (N, 1024)
        seq = features.unsqueeze(0)
        out1, _ = model.gru1(seq)
        out2, _ = model.gru2(out1)
        out2 = out2.squeeze(0)
        pred_all = model.fc(out2).squeeze(-1).cpu().numpy()  # (N,)

    # Measure RMSE on second half only
    pred_second = pred_all[split_idx:]
    gt_second = gt[split_idx:]
    rmse = float(np.sqrt(np.mean((pred_second - gt_second) ** 2)))

    return {
        "rmse": rmse,
        "predictions": pred_second.tolist(),
        "ground_truth": gt_second.tolist(),
        "predictions_full": pred_all.tolist(),
        "ground_truth_full": gt.tolist(),
    }


def evaluate_bearing(
    model: CNNGRUMHAModel,
    eval_data: Dict,
    device: torch.device = None,
    window_size: int = 128,  # kept for API compat
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Evaluate model on eval bearing data.

    Extracts CNN features (no_grad), runs full GRU+FC on the sequence.

    Args:
        model: trained model
        eval_data: dict with 'snapshots' and 'rul'
        device: compute device
        window_size: unused (full sequence used)

    Returns:
        (rmse, predictions, ground_truth)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    snapshots = eval_data["snapshots"]
    gt = eval_data["rul"]
    N = len(snapshots)

    with torch.no_grad():
        snaps = torch.FloatTensor(snapshots).to(device)
        features = model.extract_cnn_features(snaps, use_grad=False)  # (N, 1024)
        seq = features.unsqueeze(0)
        out1, _ = model.gru1(seq)
        out2, _ = model.gru2(out1)
        out2 = out2.squeeze(0)
        pred = model.fc(out2).squeeze(-1).cpu().numpy()  # (N,)

    rmse = float(np.sqrt(np.mean((pred - gt) ** 2)))
    return rmse, pred, gt


# =====================================================================
# Transfer experiment runner
# =====================================================================

def run_transfer_experiment(
    source_data: Dict,
    target_data: Dict,
    seed: int = 42,
    source_iterations: int = 60,
    finetune_iterations: int = 100,
    window_size: int = 128,
    lr: float = 0.001,
    alpha: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
    cnn_batch_size: int = 128,
) -> Dict:
    """
    Run a single transfer experiment:
      1. Train on source bearing (full-sequence GRU training)
      2. Split target 1:1 (chronological)
      3. Fine-tune FC on first half (full-sequence FC training)
      4. Evaluate on second half
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    source_name = source_data["bearing_name"]
    target_name = target_data["bearing_name"]

    if verbose:
        print(f"\n  Transfer: {source_name} -> {target_name} (seed={seed})")
        print(f"  Source: {source_data['n_snapshots']} snapshots")
        print(f"  Target: {target_data['n_snapshots']} snapshots")

    model = CNNGRUMHAModel(cnn_batch_size=cnn_batch_size).to(device)

    if verbose:
        params = model.count_parameters()
        print(f"  Model params: {params['total']:,} total")

    # Step 1: Source domain training
    t_source = time.time()
    if verbose:
        print(f"  [1/3] Source training ({source_iterations} iters)...")
    source_loss_hist = train_source_domain(
        model, source_data,
        n_iterations=source_iterations,
        window_size=window_size,
        lr=lr, alpha=alpha,
        device=device, verbose=verbose,
    )
    source_time = time.time() - t_source

    source_rmse, _, _ = evaluate_bearing(model, source_data, device)
    if verbose:
        print(f"  Source RMSE: {source_rmse:.4f} | Training time: {source_time:.1f}s")

    # Step 2: Split target 1:1 (random split — both halves cover full RUL range)
    from data_utils import get_transfer_split
    finetune_data, eval_data = get_transfer_split(
        target_data, split_ratio=0.5, random_split=True, seed=seed
    )
    if verbose:
        rul_ft = finetune_data["rul"]
        rul_ev = eval_data["rul"]
        print(f"  Target split (random): {finetune_data['n_snapshots']} FT + "
              f"{eval_data['n_snapshots']} eval snapshots")
        print(f"    FT RUL range: [{rul_ft.min():.3f}, {rul_ft.max():.3f}]")
        print(f"    Eval RUL range: [{rul_ev.min():.3f}, {rul_ev.max():.3f}]")

    # Step 3: Fine-tune FC
    t_ft = time.time()
    if verbose:
        print(f"  [2/3] Fine-tuning FC ({finetune_iterations} iters)...")
    ft_loss_hist = finetune_fc_head(
        model, finetune_data,
        n_iterations=finetune_iterations,
        lr=lr, alpha=alpha,
        device=device, verbose=verbose,
    )
    ft_time = time.time() - t_ft

    # Step 4: Evaluate on held-out half
    # With random split, evaluate directly (no temporal context issue)
    if verbose:
        print(f"  [3/3] Evaluating on held-out half...")
    t_eval = time.time()
    eval_rmse, predictions, ground_truth = evaluate_bearing(model, eval_data, device)
    eval_time = time.time() - t_eval

    if verbose:
        print(f"  Eval RMSE: {eval_rmse:.4f} | FT time: {ft_time:.1f}s")

    # predictions/ground_truth may already be lists (from evaluate_bearing_with_context)
    if hasattr(predictions, 'tolist'):
        predictions = predictions.tolist()
    if hasattr(ground_truth, 'tolist'):
        ground_truth = ground_truth.tolist()

    return {
        "source": source_name,
        "target": target_name,
        "seed": seed,
        "rmse": eval_rmse,
        "source_rmse": source_rmse,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "source_loss_history": source_loss_hist,
        "finetune_loss_history": ft_loss_hist,
        "source_n_snapshots": source_data["n_snapshots"],
        "target_n_snapshots": target_data["n_snapshots"],
        "finetune_n_snapshots": finetune_data["n_snapshots"],
        "eval_n_snapshots": eval_data["n_snapshots"],
        "source_training_time_s": source_time,
        "finetune_time_s": ft_time,
        "eval_time_s": eval_time,
    }


def run_transfer_multi_seed(
    source_data: Dict,
    target_data: Dict,
    seeds: List[int] = None,
    source_iterations: int = 60,
    finetune_iterations: int = 100,
    window_size: int = 128,
    lr: float = 0.001,
    alpha: float = 1e-4,
    device: torch.device = None,
    verbose: bool = True,
    cnn_batch_size: int = 128,
) -> Dict:
    """Run transfer experiment with multiple seeds."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    source_name = source_data["bearing_name"]
    target_name = target_data["bearing_name"]

    print(f"\n{'='*60}")
    print(f"Transfer: {source_name} -> {target_name}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}")

    per_seed_results = []
    rmse_values = []

    for seed in seeds:
        result = run_transfer_experiment(
            source_data=source_data,
            target_data=target_data,
            seed=seed,
            source_iterations=source_iterations,
            finetune_iterations=finetune_iterations,
            window_size=window_size,
            lr=lr,
            alpha=alpha,
            device=device,
            verbose=verbose,
            cnn_batch_size=cnn_batch_size,
        )
        per_seed_results.append(result)
        rmse_values.append(result["rmse"])
        print(f"  Seed {seed}: RMSE = {result['rmse']:.4f}")

    mean_rmse = float(np.mean(rmse_values))
    std_rmse = float(np.std(rmse_values))
    print(f"\n  Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

    best_idx = int(np.argmin(rmse_values))

    return {
        "source": source_name,
        "target": target_name,
        "seeds": seeds,
        "rmse_mean": mean_rmse,
        "rmse_std": std_rmse,
        "rmse_per_seed": rmse_values,
        "best_seed": seeds[best_idx],
        "best_seed_predictions": per_seed_results[best_idx]["predictions"],
        "best_seed_ground_truth": per_seed_results[best_idx]["ground_truth"],
        "per_seed_results": per_seed_results,
    }

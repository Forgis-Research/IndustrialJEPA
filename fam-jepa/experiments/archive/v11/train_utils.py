"""
V11 Training Utilities: Pretraining and Fine-tuning loops
"""

import os
import math
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple

from models import TrajectoryJEPA, RULProbe, SupervisedLSTM, trajectory_jepa_loss
from data_utils import (
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test, RUL_CAP
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Pretraining
# ============================================================

def pretrain_one_epoch(model: TrajectoryJEPA,
                        loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        lambda_var: float = 0.01) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_pred = 0.0
    total_var = 0.0
    n = 0

    for batch in loader:
        past, past_mask, future, future_mask, k, t = [x.to(DEVICE) for x in batch]

        optimizer.zero_grad()
        pred_future, h_future, h_past = model.forward_pretrain(
            past, past_mask, future, future_mask, k
        )

        loss, pred_l, var_l = trajectory_jepa_loss(pred_future, h_future, lambda_var)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_ema()

        B = past.shape[0]
        total_loss += loss.item() * B
        total_pred += pred_l.item() * B
        total_var += var_l.item() * B
        n += B

    return {
        'loss': total_loss / n,
        'pred_loss': total_pred / n,
        'var_loss': total_var / n,
    }


@torch.no_grad()
def compute_h_past_embeddings(model: TrajectoryJEPA,
                               engines: Dict[int, np.ndarray],
                               batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute h_past embeddings for all engines (at their last cycle).
    Returns: (embeddings array (N, d), rul_labels array (N,))
    """
    model.eval()
    from data_utils import CMAPSSFinetuneDataset, collate_finetune, compute_rul_labels

    ds = CMAPSSFinetuneDataset(engines, use_last_only=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_finetune)

    all_h = []
    all_rul = []
    for past, mask, rul in loader:
        past, mask = past.to(DEVICE), mask.to(DEVICE)
        h = model.encode_past(past, mask)
        all_h.append(h.cpu().numpy())
        all_rul.append(rul.numpy())

    return np.vstack(all_h), np.concatenate(all_rul) * RUL_CAP  # back to raw RUL


def linear_probe_rmse(model: TrajectoryJEPA,
                       train_engines: Dict[int, np.ndarray],
                       val_engines: Dict[int, np.ndarray],
                       n_epochs: int = 100,
                       lr: float = 1e-3) -> float:
    """
    Train a linear probe on frozen JEPA embeddings.
    Returns validation RMSE (raw RUL scale, cap=125).
    """
    from data_utils import CMAPSSFinetuneDataset, collate_finetune

    model.eval()
    probe = RULProbe(model.d_model).to(DEVICE)
    optim = torch.optim.Adam(probe.parameters(), lr=lr)

    # Build dataloaders with multiple cuts per engine
    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                             collate_fn=collate_finetune)

    best_rmse = float('inf')
    patience = 20
    no_improve = 0

    for ep in range(n_epochs):
        probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # Validation RMSE
        probe.eval()
        preds, targets = [], []
        for past, mask, rul in val_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
                pred = probe(h)
            preds.append(pred.cpu().numpy())
            targets.append(rul.numpy())

        preds = np.concatenate(preds) * RUL_CAP
        targets = np.concatenate(targets) * RUL_CAP
        rmse = np.sqrt(np.mean((preds - targets) ** 2))

        if rmse < best_rmse:
            best_rmse = rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_rmse


def pretrain(model: TrajectoryJEPA,
             train_engines: Dict[int, np.ndarray],
             val_engines: Dict[int, np.ndarray],
             n_epochs: int = 200,
             batch_size: int = 8,
             lr: float = 3e-4,
             weight_decay: float = 0.01,
             n_cuts_per_epoch: int = 20,
             min_past: int = 10,
             min_horizon: int = 5,
             max_horizon: int = 30,
             lambda_var: float = 0.01,
             probe_every: int = 10,
             checkpoint_path: Optional[str] = None,
             verbose: bool = True) -> Dict:
    """Full pretraining loop with periodic linear probe validation."""

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    train_ds = CMAPSSPretrainDataset(
        train_engines, n_cuts_per_engine=n_cuts_per_epoch,
        min_past=min_past, min_horizon=min_horizon, max_horizon=max_horizon
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_pretrain, num_workers=0)

    history = {
        'loss': [], 'pred_loss': [], 'var_loss': [],
        'probe_rmse': [], 'probe_epochs': []
    }
    best_probe_rmse = float('inf')
    best_state = None

    for epoch in range(1, n_epochs + 1):
        # Rebuild dataset each epoch for fresh random cuts
        train_ds = CMAPSSPretrainDataset(
            train_engines, n_cuts_per_engine=n_cuts_per_epoch,
            min_past=min_past, min_horizon=min_horizon, max_horizon=max_horizon,
            seed=epoch  # different cuts each epoch
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_pretrain, num_workers=0)

        metrics = pretrain_one_epoch(model, train_loader, optimizer, lambda_var)
        history['loss'].append(metrics['loss'])
        history['pred_loss'].append(metrics['pred_loss'])
        history['var_loss'].append(metrics['var_loss'])
        scheduler.step()

        if epoch % probe_every == 0 or epoch == 1:
            probe_rmse = linear_probe_rmse(model, train_engines, val_engines)
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)

            if probe_rmse < best_probe_rmse:
                best_probe_rmse = probe_rmse
                best_state = copy.deepcopy(model.state_dict())
                if checkpoint_path:
                    torch.save(best_state, checkpoint_path)

            if verbose:
                print(f"Ep {epoch:3d} | loss={metrics['loss']:.4f} "
                      f"pred={metrics['pred_loss']:.4f} "
                      f"var={metrics['var_loss']:.4f} | "
                      f"probe_RMSE={probe_rmse:.2f} (best={best_probe_rmse:.2f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    return history, best_probe_rmse


# ============================================================
# Fine-tuning
# ============================================================

def finetune(model: TrajectoryJEPA,
             train_engines: Dict[int, np.ndarray],
             val_engines: Dict[int, np.ndarray],
             test_engines: Dict[int, np.ndarray],
             test_rul: np.ndarray,
             n_epochs: int = 100,
             lr_probe: float = 1e-3,
             lr_e2e: float = 1e-4,
             batch_size: int = 16,
             early_stop_patience: int = 20,
             mode: str = 'frozen',
             seed: int = 42,
             verbose: bool = False) -> Dict[str, float]:
    """
    Fine-tune JEPA for RUL prediction.

    mode='frozen': freeze encoder, train only probe
    mode='e2e': fine-tune entire model end-to-end
    """
    from data_utils import CMAPSSFinetuneDataset, CMAPSSTestDataset

    model = model.to(DEVICE)
    probe = RULProbe(model.d_model).to(DEVICE)

    if mode == 'frozen':
        for p in model.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr_probe)
    else:  # e2e
        for p in model.context_encoder.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(
            list(model.context_encoder.parameters()) +
            list(model.predictor.parameters()) +
            list(probe.parameters()),
            lr=lr_e2e
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_test)

    best_val_rmse = float('inf')
    best_probe_state = None
    best_encoder_state = None
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        if mode == 'frozen':
            model.eval()
        else:
            model.train()
        probe.train()

        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            if mode == 'frozen':
                with torch.no_grad():
                    h = model.encode_past(past, mask)
            else:
                h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

        # Validation RMSE
        val_rmse = _eval_rmse(model, probe, val_loader, raw_scale=True)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_probe_state = copy.deepcopy(probe.state_dict())
            if mode == 'e2e':
                best_encoder_state = copy.deepcopy(model.context_encoder.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break

    # Load best state
    probe.load_state_dict(best_probe_state)
    if mode == 'e2e' and best_encoder_state is not None:
        model.context_encoder.load_state_dict(best_encoder_state)

    # Final test RMSE
    test_rmse = _eval_test_rmse(model, probe, test_loader)

    return {
        'val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'mode': mode,
    }


@torch.no_grad()
def _eval_rmse(model, probe, loader, raw_scale: bool = True) -> float:
    model.eval()
    probe.eval()
    preds, targets = [], []
    for past, mask, rul in loader:
        past, mask = past.to(DEVICE), mask.to(DEVICE)
        h = model.encode_past(past, mask)
        pred = probe(h)
        preds.append(pred.cpu().numpy())
        targets.append(rul.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    if raw_scale:
        preds = preds * RUL_CAP
        targets = targets * RUL_CAP
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


@torch.no_grad()
def _eval_test_rmse(model, probe, test_loader) -> float:
    """Test RMSE where RUL targets are already in raw cycles."""
    model.eval()
    probe.eval()
    preds, targets = [], []
    for past, mask, rul_gt in test_loader:
        past, mask = past.to(DEVICE), mask.to(DEVICE)
        h = model.encode_past(past, mask)
        pred_norm = probe(h)
        # pred is in [0,1], scale back
        pred_raw = pred_norm.cpu().numpy() * RUL_CAP
        preds.append(pred_raw)
        targets.append(rul_gt.numpy())  # already raw cycles
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ============================================================
# Supervised LSTM baseline
# ============================================================

def train_supervised_lstm(train_engines: Dict[int, np.ndarray],
                            val_engines: Dict[int, np.ndarray],
                            test_engines: Dict[int, np.ndarray],
                            test_rul: np.ndarray,
                            n_epochs: int = 150,
                            batch_size: int = 16,
                            lr: float = 1e-3,
                            hidden_size: int = 64,
                            seed: int = 42,
                            verbose: bool = False) -> Dict[str, float]:
    """Train supervised LSTM from scratch."""
    from data_utils import CMAPSSFinetuneDataset, CMAPSSTestDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SupervisedLSTM(n_sensors=14, hidden_size=hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_test)

    best_val_rmse = float('inf')
    best_state = None
    patience_count = 0
    patience = 20

    for epoch in range(1, n_epochs + 1):
        model.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            pred = model(past, mask)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pred = model(past, mask)
                preds.append(pred.cpu().numpy())
                targets.append(rul.numpy())
        preds = np.concatenate(preds) * RUL_CAP
        targets = np.concatenate(targets) * RUL_CAP
        val_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    model.load_state_dict(best_state)

    # Test RMSE
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pred = model(past, mask)
            preds.append(pred.cpu().numpy() * RUL_CAP)
            targets.append(rul_gt.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    test_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))

    return {
        'val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
    }


# ============================================================
# Label efficiency: subsample training engines
# ============================================================

def subsample_engines(engines: Dict[int, np.ndarray],
                       fraction: float,
                       seed: int = 42) -> Dict[int, np.ndarray]:
    """Sample a fraction of training engines."""
    ids = sorted(engines.keys())
    rng = np.random.default_rng(seed)
    n = max(1, int(fraction * len(ids)))
    selected = rng.choice(ids, size=n, replace=False)
    return {i: engines[i] for i in selected}


# ============================================================
# Diagnostics
# ============================================================

def compute_pretraining_diagnostics(model: TrajectoryJEPA,
                                     train_engines: Dict[int, np.ndarray],
                                     val_engines: Dict[int, np.ndarray]) -> Dict:
    """
    Compute pretraining diagnostics:
    1. h_past PC1 Spearman rho with RUL
    2. Shuffle test
    """
    from sklearn.decomposition import PCA

    model.eval()

    # Get embeddings and RUL labels for train+val engines
    all_engines = {**train_engines, **val_engines}
    embeddings, rul_labels = compute_h_past_embeddings(model, all_engines)

    # PC1 Spearman
    pca = PCA(n_components=min(5, embeddings.shape[1]))
    pca_coords = pca.fit_transform(embeddings)
    pc1_rho, pc1_p = spearmanr(pca_coords[:, 0], rul_labels)

    # All components
    rhos = []
    for i in range(pca_coords.shape[1]):
        rho, _ = spearmanr(pca_coords[:, i], rul_labels)
        rhos.append(abs(rho))
    max_rho = max(rhos)

    # Shuffle test: permute past tokens, re-encode
    from data_utils import CMAPSSFinetuneDataset, collate_finetune

    def get_probe_rmse_shuffled():
        ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            collate_fn=collate_finetune)

        probe = RULProbe(model.d_model).to(DEVICE)
        # Quick train on real embeddings first
        train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=3)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  collate_fn=collate_finetune)
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(50):
            for past, mask, rul in train_loader:
                past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
                with torch.no_grad():
                    h = model.encode_past(past, mask)
                pred = probe(h)
                loss = F.mse_loss(pred, rul)
                optim.zero_grad()
                loss.backward()
                optim.step()

        probe.eval()
        preds, targets = [], []
        for past, mask, rul in loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            # Shuffle tokens along time dimension
            B, T, S = past.shape
            idx = torch.stack([torch.randperm(T) for _ in range(B)]).to(DEVICE)
            past_shuffled = past.gather(1, idx.unsqueeze(-1).expand_as(past))
            with torch.no_grad():
                h = model.encode_past(past_shuffled, mask)
                pred = probe(h)
            preds.append(pred.cpu().numpy())
            targets.append(rul.numpy())
        preds = np.concatenate(preds) * RUL_CAP
        targets = np.concatenate(targets) * RUL_CAP
        return float(np.sqrt(np.mean((preds - targets) ** 2)))

    shuffle_rmse = get_probe_rmse_shuffled()

    return {
        'pc1_rho': float(pc1_rho),
        'pc1_p': float(pc1_p),
        'max_component_rho': float(max_rho),
        'all_component_rhos': rhos,
        'shuffle_rmse': shuffle_rmse,
        'embeddings': embeddings,
        'rul_labels': rul_labels,
        'pca': pca,
        'pca_coords': pca_coords,
        'explained_variance': pca.explained_variance_ratio_.tolist(),
    }

"""
Training utilities for MTS-JEPA replication.
Pre-training loop (Algorithm 1), downstream evaluation (Algorithm 2),
and all supporting functions.
"""
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from data_utils import RevIN
from models import (
    MTSJEPA, DownstreamClassifier,
    kl_divergence, embedding_mse, codebook_alignment_loss,
    dual_entropy_loss, reconstruction_loss,
)

# ============================================================================
# Loss weight configuration (from paper)
# ============================================================================

DEFAULT_LOSS_CONFIG = {
    'lambda_f': 1.0,           # Fine prediction weight
    'lambda_c': 0.5,           # Coarse prediction weight
    'gamma': 0.5,              # MSE weight within prediction loss
    'kl_scale': 0.1,           # KL scale (paper=1.0 explodes on small batch; 0.1 is stable)
    'lambda_emb': 1.0,         # Embedding alignment
    'lambda_com': 0.25,        # Commitment loss
    'lambda_ent_sample': 0.005,  # Sample entropy (minimize) — paper: 0.005
    'lambda_ent_batch': 0.01,    # Batch entropy (maximize) — paper: 0.01
    'lambda_r_start': 0.5,      # Reconstruction weight at epoch 0
    'lambda_r_end': 0.1,        # Reconstruction weight at epoch 99
}


def compute_total_loss(losses, loss_config, epoch, total_epochs):
    """
    Compute total loss following the paper's formulation:

    L_pred = lambda_f * (L_KL_fine + gamma * L_MSE_fine) + lambda_c * L_KL_coarse
    L_code = lambda_emb * L_emb + lambda_com * L_com
             + lambda_ent_sample * L_ent_sample - lambda_ent_batch * L_ent_batch
    L_rec  = lambda_r(epoch) * L_rec
    L_total = L_pred + L_code + L_rec
    """
    cfg = loss_config

    # Anneal reconstruction weight linearly
    progress = epoch / max(total_epochs - 1, 1)
    lambda_r = cfg['lambda_r_start'] - (cfg['lambda_r_start'] - cfg['lambda_r_end']) * progress

    # Prediction loss: paper formulation — KL primary, MSE secondary (gamma=0.1)
    kl_scale = cfg.get('kl_scale', 1.0)
    L_pred = cfg['lambda_f'] * (kl_scale * losses['kl_fine'] + cfg['gamma'] * losses['mse_fine']) \
           + cfg['lambda_c'] * (kl_scale * losses['kl_coarse'])

    # Codebook loss
    L_code = cfg['lambda_emb'] * losses['emb'] \
           + cfg['lambda_com'] * losses['com'] \
           + cfg['lambda_ent_sample'] * losses['ent_sample'] \
           - cfg['lambda_ent_batch'] * losses['ent_batch']

    # Reconstruction loss
    L_rec = lambda_r * losses['rec']

    total = L_pred + L_code + L_rec

    return total, {
        'L_pred': L_pred.item(),
        'L_code': L_code.item(),
        'L_rec': L_rec.item(),
        'L_total': total.item(),
        'lambda_r': lambda_r,
    }


# ============================================================================
# Pre-training
# ============================================================================

def pretrain_mtsjepa(model, train_loader, val_loader, n_vars, config,
                     device='cuda', checkpoint_dir=None, verbose=True):
    """
    Pre-train MTS-JEPA following Algorithm 1.

    Args:
        model: MTSJEPA instance
        train_loader: yields (x_context, x_target) pairs
        val_loader: yields (x_context, x_target) pairs
        n_vars: number of variables
        config: dict with training hyperparameters
        device: torch device
        checkpoint_dir: where to save best model
        verbose: print progress

    Returns:
        dict with training history and best model info
    """
    lr = config.get('lr', 5e-4)
    weight_decay = config.get('weight_decay', 1e-5)
    n_epochs = config.get('n_epochs', 100)
    patience = config.get('patience', 10)
    patience_start = config.get('patience_start', 50)
    max_grad_norm = config.get('max_grad_norm', 0.5)
    loss_config = config.get('loss_config', DEFAULT_LOSS_CONFIG)

    optimizer = torch.optim.Adam(model.online_params(), lr=lr, weight_decay=weight_decay)

    revin = RevIN(n_vars).to(device)

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = {
        'train_losses': [],
        'val_losses': [],
        'codebook_util': [],
        'codebook_perplexity': [],
    }

    start_time = time.time()

    for epoch in range(n_epochs):
        model.train()
        revin.train()
        epoch_losses = []
        epoch_util = []
        epoch_perp = []

        for batch_idx, (x_ctx, x_tgt) in enumerate(train_loader):
            x_ctx = x_ctx.to(device)  # (B, T, V)
            x_tgt = x_tgt.to(device)

            # Apply RevIN to both context and target
            x_ctx_norm = revin(x_ctx)
            x_tgt_norm = revin(x_tgt)  # Uses the same cached stats from context

            # Forward pass
            losses = model(x_ctx_norm, x_tgt_norm)

            # Compute total loss
            total_loss, loss_components = compute_total_loss(
                losses, loss_config, epoch, n_epochs
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.online_params(), max_norm=max_grad_norm)

            optimizer.step()

            # EMA update
            model.update_ema()

            epoch_losses.append(loss_components['L_total'])
            epoch_util.append(losses['codebook_utilization'])
            epoch_perp.append(losses['codebook_perplexity'])

        # Validation
        model.eval()
        revin.eval()
        val_losses_epoch = []

        with torch.no_grad():
            for x_ctx, x_tgt in val_loader:
                x_ctx = x_ctx.to(device)
                x_tgt = x_tgt.to(device)
                x_ctx_norm = revin(x_ctx)
                x_tgt_norm = revin(x_tgt)

                losses = model(x_ctx_norm, x_tgt_norm)
                total_loss, loss_components = compute_total_loss(
                    losses, loss_config, epoch, n_epochs
                )
                val_losses_epoch.append(loss_components['L_total'])

        mean_train_loss = np.mean(epoch_losses)
        mean_val_loss = np.mean(val_losses_epoch) if val_losses_epoch else mean_train_loss
        mean_util = np.mean(epoch_util)
        mean_perp = np.mean(epoch_perp)

        history['train_losses'].append(mean_train_loss)
        history['val_losses'].append(mean_val_loss)
        history['codebook_util'].append(mean_util)
        history['codebook_perplexity'].append(mean_perp)

        # Early stopping (after patience_start epochs)
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        elif epoch >= patience_start:
            patience_counter += 1

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1 or patience_counter == 0):
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:3d} | train {mean_train_loss:.4f} | "
                  f"val {mean_val_loss:.4f} | util {mean_util:.2f} | "
                  f"perp {mean_perp:.1f} | {elapsed:.0f}s")

        if patience_counter >= patience and epoch >= patience_start:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (best: {best_epoch})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(best_state, checkpoint_path)

    wall_time = time.time() - start_time

    return {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'wall_time': wall_time,
        'history': history,
    }


# ============================================================================
# Downstream Evaluation
# ============================================================================

def encode_windows(model, windows, n_vars, device, batch_size=256):
    """
    Encode context windows using frozen MTS-JEPA encoder.

    windows: (N, T, V) numpy array
    Returns: (N, P*K) numpy array of code representations
    """
    model.eval()
    revin = RevIN(n_vars).to(device)
    revin.eval()

    all_features = []

    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            batch = torch.tensor(windows[i:i+batch_size], dtype=torch.float32).to(device)
            batch_norm = revin(batch)
            features = model.encode_for_downstream(batch_norm)  # (B, P*K)
            all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def train_downstream_classifier(features_train, labels_train,
                                features_val, labels_val,
                                input_dim, device='cuda',
                                n_epochs=100, lr=1e-3, batch_size=64):
    """
    Train MLP classifier on encoded features.

    Returns: trained classifier
    """
    classifier = DownstreamClassifier(input_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    # Handle class imbalance
    n_pos = labels_train.sum()
    n_neg = len(labels_train) - n_pos
    if n_pos > 0:
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    else:
        pos_weight = torch.ones(1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_train = torch.tensor(features_train, dtype=torch.float32)
    y_train = torch.tensor(labels_train, dtype=torch.float32)

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_val_f1 = -1
    best_state = None

    for epoch in range(n_epochs):
        classifier.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = classifier(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate every 10 epochs
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            classifier.eval()
            with torch.no_grad():
                X_val = torch.tensor(features_val, dtype=torch.float32).to(device)
                val_logits = classifier(X_val).cpu().numpy()
                val_probs = 1 / (1 + np.exp(-val_logits))

                # Use 0.5 threshold for quick validation
                val_preds = (val_probs > 0.5).astype(int)
                if labels_val.sum() > 0:
                    val_f1 = f1_score(labels_val, val_preds, zero_division=0)
                else:
                    val_f1 = 0.0

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_state = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}

    if best_state is not None:
        classifier.load_state_dict(best_state)

    return classifier


def select_threshold(classifier, features_val, labels_val, device='cuda'):
    """
    Select threshold delta* on validation set to maximize F1.
    """
    classifier.eval()
    with torch.no_grad():
        X_val = torch.tensor(features_val, dtype=torch.float32).to(device)
        logits = classifier(X_val).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

    best_f1 = -1
    best_threshold = 0.5

    for threshold in np.arange(0.01, 1.0, 0.01):
        preds = (probs > threshold).astype(int)
        if labels_val.sum() > 0:
            f1 = f1_score(labels_val, preds, zero_division=0)
        else:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_downstream(classifier, features_test, labels_test, threshold, device='cuda'):
    """
    Evaluate on test split using selected threshold.

    Returns: dict with F1, AUC, Precision, Recall
    """
    classifier.eval()
    with torch.no_grad():
        X_test = torch.tensor(features_test, dtype=torch.float32).to(device)
        logits = classifier(X_test).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))

    preds = (probs > threshold).astype(int)

    results = {}
    if labels_test.sum() > 0 and (1 - labels_test).sum() > 0:
        results['f1'] = f1_score(labels_test, preds, zero_division=0) * 100
        results['auc'] = roc_auc_score(labels_test, probs) * 100
        results['precision'] = precision_score(labels_test, preds, zero_division=0) * 100
        results['recall'] = recall_score(labels_test, preds, zero_division=0) * 100
    else:
        results['f1'] = 0.0
        results['auc'] = 50.0
        results['precision'] = 0.0
        results['recall'] = 0.0

    results['threshold'] = threshold
    results['n_test'] = len(labels_test)
    results['n_anomalous'] = int(labels_test.sum())
    results['anomaly_rate'] = float(labels_test.mean())

    return results


def full_downstream_evaluation(model, data_dict, device='cuda', seed=42):
    """
    Complete downstream evaluation pipeline (Algorithm 2).

    1. Freeze encoder + codebook
    2. Encode all context windows
    3. Train classifier on 60% of test
    4. Select threshold on 20% validation
    5. Evaluate on final 20%

    Returns: dict with all metrics
    """
    n_vars = data_dict['n_vars']
    window_length = data_dict['window_length']
    n_patches = model.n_patches
    n_codes = model.n_codes
    input_dim = n_patches * n_codes  # P*K = 5*128 = 640

    # Freeze model
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Encode all windows
    ds_train_ctx, ds_train_labels = data_dict['downstream_train']
    ds_val_ctx, ds_val_labels = data_dict['downstream_val']
    ds_test_ctx, ds_test_labels = data_dict['downstream_test']

    print(f"  Encoding windows...")
    feat_train = encode_windows(model, ds_train_ctx, n_vars, device)
    feat_val = encode_windows(model, ds_val_ctx, n_vars, device)
    feat_test = encode_windows(model, ds_test_ctx, n_vars, device)

    print(f"  Feature shape: {feat_train.shape}")
    print(f"  Train anomaly rate: {ds_train_labels.mean():.3f}")

    # Set seed for classifier training
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Train classifier
    print(f"  Training classifier...")
    classifier = train_downstream_classifier(
        feat_train, ds_train_labels,
        feat_val, ds_val_labels,
        input_dim, device,
    )

    # Select threshold
    threshold, val_f1 = select_threshold(classifier, feat_val, ds_val_labels, device)
    print(f"  Selected threshold: {threshold:.2f} (val F1: {val_f1:.3f})")

    # Evaluate
    results = evaluate_downstream(classifier, feat_test, ds_test_labels, threshold, device)
    results['val_f1'] = val_f1 * 100

    print(f"  Test F1: {results['f1']:.2f}, AUC: {results['auc']:.2f}, "
          f"Precision: {results['precision']:.2f}, Recall: {results['recall']:.2f}")

    # Unfreeze for potential further training
    for p in model.parameters():
        p.requires_grad = True

    return results


# ============================================================================
# Experiment orchestration
# ============================================================================

def run_single_experiment(dataset_name, seed, data_dict, device='cuda',
                          config=None, checkpoint_dir=None, verbose=True):
    """
    Run a single pre-train + downstream experiment.

    Returns: dict with pretrain info and downstream metrics
    """
    if config is None:
        config = {}

    n_vars = data_dict['n_vars']
    window_length = data_dict['window_length']

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Build model
    model = MTSJEPA(
        n_vars=n_vars,
        d_model=config.get('d_model', 256),
        d_out=config.get('d_out', 256),
        n_codes=config.get('n_codes', 128),
        tau=config.get('tau', 0.1),
        patch_length=config.get('patch_length', 20),
        n_patches=config.get('n_patches', 5),
        n_encoder_layers=config.get('n_encoder_layers', 6),
        n_heads=config.get('n_heads', 8),
        dropout=config.get('dropout', 0.1),
        ema_rho=config.get('ema_rho', 0.996),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {dataset_name} | seed={seed} | params={n_params:,}")
        print(f"{'='*60}")

    # Pre-train
    if verbose:
        print(f"\nPhase 1: Pre-training...")
    pretrain_info = pretrain_mtsjepa(
        model,
        data_dict['pretrain_train_loader'],
        data_dict['pretrain_val_loader'],
        n_vars,
        config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )

    # Downstream evaluation
    if verbose:
        print(f"\nPhase 2: Downstream evaluation...")
    downstream_results = full_downstream_evaluation(model, data_dict, device, seed)

    return {
        'dataset': dataset_name,
        'seed': seed,
        'n_params': n_params,
        'pretrain': {
            'best_epoch': pretrain_info['best_epoch'],
            'val_loss': pretrain_info['best_val_loss'],
            'wall_time_seconds': pretrain_info['wall_time'],
            'codebook_utilization': pretrain_info['history']['codebook_util'][-1],
            'codebook_perplexity': pretrain_info['history']['codebook_perplexity'][-1],
        },
        'downstream': downstream_results,
    }

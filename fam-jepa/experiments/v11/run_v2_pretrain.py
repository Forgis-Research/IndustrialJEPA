"""
V11 Experiment 2: Improved pretraining hyperparameters.
Key changes from V1:
1. d_model=256 (vs 128) - more capacity
2. Probe-based early stopping (stop when probe RMSE stops improving for 5 consecutive checks)
3. Smaller batch_size (4 instead of 8) for better coverage
4. More cuts per engine (30 instead of 20)
5. EMA momentum 0.99 (faster adaptation) vs 0.996
"""

import os
import sys
import time
import json
import copy
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
EXP_DIR = os.path.join(BASE, 'experiments/v11')
PLOTS_DIR = os.path.join(BASE, 'analysis/plots/v11')
sys.path.insert(0, EXP_DIR)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    CMAPSSPretrainDataset, collate_pretrain, collate_finetune, collate_test,
    N_SENSORS, RUL_CAP, get_sensor_cols, compute_rul_labels
)
from models import TrajectoryJEPA, RULProbe, SupervisedLSTM, count_parameters, trajectory_jepa_loss
from train_utils import subsample_engines, _eval_test_rmse, train_supervised_lstm

from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)
    fname = os.path.join(EXP_DIR, 'EXPERIMENT_LOG.md')
    with open(fname, 'a') as f:
        f.write(msg + '\n')

def save_json(obj, path):
    def default(x):
        if isinstance(x, (np.floating, float)):
            return float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=default)


log("\n" + "="*60)
log("EXP 2: Improved Pretraining (d_model=256, more cuts)")
log("="*60)

data = load_cmapss_subset('FD001')

# ============================================================
# New model: d_model=256
# ============================================================
D_MODEL = 256

model_v2 = TrajectoryJEPA(
    n_sensors=N_SENSORS,
    patch_length=1,
    d_model=D_MODEL,
    n_heads=4,
    n_layers=2,
    d_ff=512,
    dropout=0.1,
    ema_momentum=0.99,  # faster EMA
    predictor_hidden=256,
)
log(f"V2 model params: {count_parameters(model_v2):,}")
model_v2 = model_v2.to(DEVICE)

optimizer = torch.optim.AdamW(
    [p for p in model_v2.parameters() if p.requires_grad],
    lr=3e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

# Training config
N_EPOCHS = 200
BATCH_SIZE = 4
N_CUTS = 30
LAMBDA_VAR = 0.01
PROBE_EVERY = 5  # more frequent probe checks

history_v2 = {'loss': [], 'pred_loss': [], 'var_loss': [], 'probe_rmse': [], 'probe_epochs': []}
best_probe_rmse = float('inf')
best_state = None
ckpt_path_v2 = os.path.join(EXP_DIR, 'best_pretrain_L1_v2.pt')

# Quick linear probe evaluation
def eval_probe_rmse(enc_model, train_eng, val_eng, n_probe_epochs=50):
    enc_model.eval()
    from data_utils import CMAPSSFinetuneDataset, collate_finetune
    probe = RULProbe(D_MODEL).to(DEVICE)
    optim_probe = torch.optim.Adam(probe.parameters(), lr=1e-3)

    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=3)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=False, n_cuts_per_engine=10)
    tr_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    for ep in range(n_probe_epochs):
        probe.train()
        for past, mask, rul in tr_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = enc_model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            optim_probe.zero_grad(); loss.backward(); optim_probe.step()

    probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in va_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = enc_model.encode_past(past, mask)
            preds.append(probe(h).cpu().numpy())
            targets.append(rul.numpy())
    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets) * RUL_CAP
    return float(np.sqrt(np.mean((preds - targets)**2)))

log("Starting pretraining...")
t0 = time.time()
patience_probe = 10  # stop if probe doesn't improve for 10 consecutive checks
no_improve = 0

for epoch in range(1, N_EPOCHS + 1):
    train_ds = CMAPSSPretrainDataset(
        data['train_engines'], n_cuts_per_engine=N_CUTS,
        min_past=10, min_horizon=5, max_horizon=30, seed=epoch
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_pretrain, num_workers=0)

    model_v2.train()
    total_loss, total_pred, total_var, n = 0.0, 0.0, 0.0, 0
    for past, past_mask, future, future_mask, k, t in train_loader:
        past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
        future, future_mask = future.to(DEVICE), future_mask.to(DEVICE)
        k = k.to(DEVICE)
        optimizer.zero_grad()
        pred_future, h_future, h_past = model_v2.forward_pretrain(
            past, past_mask, future, future_mask, k
        )
        loss, pred_l, var_l = trajectory_jepa_loss(pred_future, h_future, LAMBDA_VAR)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v2.parameters(), 1.0)
        optimizer.step()
        model_v2.update_ema()
        B = past.shape[0]
        total_loss += loss.item() * B
        total_pred += pred_l.item() * B
        total_var += var_l.item() * B
        n += B

    history_v2['loss'].append(total_loss / n)
    history_v2['pred_loss'].append(total_pred / n)
    history_v2['var_loss'].append(total_var / n)
    scheduler.step()

    if epoch % PROBE_EVERY == 0 or epoch == 1:
        probe_rmse = eval_probe_rmse(model_v2, data['train_engines'], data['val_engines'])
        history_v2['probe_rmse'].append(probe_rmse)
        history_v2['probe_epochs'].append(epoch)

        if probe_rmse < best_probe_rmse:
            best_probe_rmse = probe_rmse
            best_state = copy.deepcopy(model_v2.state_dict())
            torch.save(best_state, ckpt_path_v2)
            no_improve = 0
        else:
            no_improve += 1

        log(f"Ep {epoch:3d} | loss={total_loss/n:.4f} | probe_RMSE={probe_rmse:.2f} "
            f"(best={best_probe_rmse:.2f}, no_improve={no_improve})")

        if no_improve >= patience_probe:
            log(f"  Early stopping at epoch {epoch} (patience {patience_probe} exhausted)")
            break
    else:
        if epoch % 20 == 0:
            log(f"Ep {epoch:3d} | loss={total_loss/n:.4f}")

elapsed = (time.time() - t0) / 60
log(f"\nPretraining V2 complete in {elapsed:.1f} min")
log(f"Best probe RMSE: {best_probe_rmse:.2f}")
log(f"Final loss: {history_v2['loss'][-1]:.4f} (started: {history_v2['loss'][0]:.4f})")

# Load best checkpoint
if best_state is not None:
    model_v2.load_state_dict(best_state)

save_json(history_v2, os.path.join(EXP_DIR, 'pretrain_history_L1_v2.json'))

# ============================================================
# Diagnostics for V2
# ============================================================
log("\n--- Diagnostics for V2 ---")

@torch.no_grad()
def get_embeddings_multicut(model, engines, n_cuts=10, seed=0):
    rng = np.random.default_rng(seed)
    all_h, all_rul = [], []
    for eid, seq in engines.items():
        T = len(seq)
        rul_labels = compute_rul_labels(T, RUL_CAP)
        t_max = T
        if t_max <= 10:
            continue
        cuts = rng.integers(10, t_max, size=min(n_cuts, t_max - 10))
        for t in cuts:
            past = torch.from_numpy(seq[:t]).unsqueeze(0).to(DEVICE)
            h = model.encode_past(past)
            all_h.append(h.cpu().numpy()[0])
            all_rul.append(float(rul_labels[t - 1]))
    return np.vstack(all_h), np.array(all_rul)

all_engines = {**data['train_engines'], **data['val_engines']}
emb_v2, rul_labels_v2 = get_embeddings_multicut(model_v2, all_engines, n_cuts=10)
log(f"Embeddings: {emb_v2.shape}, RUL range: [{rul_labels_v2.min():.1f}, {rul_labels_v2.max():.1f}]")

pca_v2 = PCA(n_components=5)
pca_coords_v2 = pca_v2.fit_transform(emb_v2)
log(f"Explained variance: {[f'{v:.3f}' for v in pca_v2.explained_variance_ratio_]}")

pc_rhos_v2 = []
for i in range(5):
    rho, p = spearmanr(pca_coords_v2[:, i], rul_labels_v2)
    pc_rhos_v2.append((i+1, float(rho), float(p)))
    log(f"  PC{i+1} rho={rho:.4f} (p={p:.2e})")

pc1_rho_v2 = pc_rhos_v2[0][1]
log(f"\nV2 PC1 |rho|: {abs(pc1_rho_v2):.4f} (V1 was 0.814)")

# ============================================================
# Fine-tuning V2 at 100% labels
# ============================================================
log("\n--- V2 Fine-tuning at 100% ---")

def run_finetune_v2(train_eng, val_eng, test_eng, test_rul, mode, seed):
    model_ft = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
        n_heads=4, n_layers=2, d_ff=512
    )
    model_ft.load_state_dict(torch.load(ckpt_path_v2, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)

    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)

    from data_utils import CMAPSSFinetuneDataset, CMAPSSTestDataset
    train_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    val_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model_ft.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model_ft.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(
            list(model_ft.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4
        )

    best_val = float('inf'); best_ps = None; best_es = None; no_impr = 0
    for ep in range(100):
        if mode == 'frozen': model_ft.eval()
        else: model_ft.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model_ft.encode_past(past, mask)
            else: h = model_ft.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optim.step()

        model_ft.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model_ft.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e': best_es = copy.deepcopy(model_ft.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20: break

    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es: model_ft.context_encoder.load_state_dict(best_es)

    model_ft.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model_ft.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

N_SEEDS = 5
results_v2 = {}
for mode in ['frozen', 'e2e']:
    rmses = []
    for seed in range(N_SEEDS):
        rmse = run_finetune_v2(
            data['train_engines'], data['val_engines'],
            data['test_engines'], data['test_rul'],
            mode=mode, seed=seed
        )
        rmses.append(rmse)
        log(f"  V2 {mode} seed={seed}: {rmse:.2f}")
    results_v2[mode] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
    log(f"  V2 {mode} @ 100%: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

save_json(results_v2, os.path.join(EXP_DIR, 'finetune_results_v2.json'))

log("\n--- Comparison: V1 vs V2 ---")
with open(os.path.join(EXP_DIR, 'finetune_results.json')) as f:
    results_v1 = json.load(f)
    results_v1 = {method: {float(k): v for k, v in d.items()} for method, d in results_v1.items()}

log(f"  V1 E2E @ 100%: {results_v1['jepa_e2e'][1.0]['mean']:.2f} +/- {results_v1['jepa_e2e'][1.0]['std']:.2f}")
log(f"  V2 E2E @ 100%: {results_v2['e2e']['mean']:.2f} +/- {results_v2['e2e']['std']:.2f}")
log(f"  V1 frozen @ 100%: {results_v1['jepa_frozen'][1.0]['mean']:.2f}")
log(f"  V2 frozen @ 100%: {results_v2['frozen']['mean']:.2f}")
log(f"  LSTM @ 100%: {results_v1['supervised_lstm'][1.0]['mean']:.2f}")

# Log experiment
with open(os.path.join(EXP_DIR, 'EXPERIMENT_LOG.md'), 'a') as f:
    f.write(f"""
## Exp 2: Improved Pretraining (d_model=256, faster EMA, more cuts)

**Time**: {time.strftime('%Y-%m-%d %H:%M')}
**Hypothesis**: Larger model + faster EMA + more cuts will improve downstream RUL prediction
**Change**: d_model 128->256, n_ff 256->512, ema 0.996->0.99, n_cuts 20->30, batch_size 4
**Result**: V2 E2E @ 100% = {results_v2['e2e']['mean']:.2f} vs V1 = {results_v1['jepa_e2e'][1.0]['mean']:.2f}
**Verdict**: {"KEEP" if results_v2['e2e']['mean'] < results_v1['jepa_e2e'][1.0]['mean'] else "REVERT"}
**Insight**: Best PC1 rho = {abs(pc1_rho_v2):.4f}
""")

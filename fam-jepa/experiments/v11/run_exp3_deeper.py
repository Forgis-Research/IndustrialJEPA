"""
Exp 3: Deeper network (3 layers instead of 2) with d_model=128.
Hypothesis: V1 bottleneck is depth, not width.
"""

import os, sys, time, json, copy, warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
BASE = '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa'
EXP_DIR = os.path.join(BASE, 'experiments/v11')
PLOTS_DIR = os.path.join(BASE, 'analysis/plots/v11')
sys.path.insert(0, EXP_DIR)

from data_utils import (
    load_cmapss_subset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    CMAPSSPretrainDataset, collate_pretrain, collate_finetune, collate_test,
    N_SENSORS, RUL_CAP, compute_rul_labels
)
from models import TrajectoryJEPA, RULProbe, count_parameters, trajectory_jepa_loss
from train_utils import subsample_engines, train_supervised_lstm
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log(msg):
    print(msg, flush=True)
    with open(os.path.join(EXP_DIR, 'EXPERIMENT_LOG.md'), 'a') as f:
        f.write(msg + '\n')

def save_json(obj, path):
    def default(x):
        if isinstance(x, (np.floating, float)): return float(x)
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return str(x)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=default)

log("\n" + "="*60)
log("EXP 3: Deeper network (n_layers=3, d_model=128)")
log("="*60)

data = load_cmapss_subset('FD001')

D_MODEL = 128
N_LAYERS = 3
model_v3 = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
    n_heads=4, n_layers=N_LAYERS, d_ff=256, dropout=0.1, ema_momentum=0.996
)
log(f"V3 model params: {count_parameters(model_v3):,}")
model_v3 = model_v3.to(DEVICE)

optimizer = torch.optim.AdamW(
    [p for p in model_v3.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

# Quick probe eval
def eval_probe_rmse(enc_model, train_eng, val_eng, d_model, n_epochs=50):
    enc_model.eval()
    probe = RULProbe(d_model).to(DEVICE)
    optim_p = torch.optim.Adam(probe.parameters(), lr=1e-3)
    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=3)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=False, n_cuts_per_engine=10)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)
    for ep in range(n_epochs):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = enc_model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            optim_p.zero_grad(); loss.backward(); optim_p.step()
    probe.eval()
    pv, tv = [], []
    with torch.no_grad():
        for past, mask, rul in va:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = enc_model.encode_past(past, mask)
            pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))

# Pretraining
ckpt_path_v3 = os.path.join(EXP_DIR, 'best_pretrain_L1_v3.pt')
history_v3 = {'loss': [], 'pred_loss': [], 'probe_rmse': [], 'probe_epochs': []}
best_probe_rmse = float('inf'); best_state = None; no_improve = 0

t0 = time.time()
for epoch in range(1, 201):
    train_ds = CMAPSSPretrainDataset(data['train_engines'], n_cuts_per_engine=20,
                                     min_past=10, min_horizon=5, max_horizon=30, seed=epoch)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=collate_pretrain, num_workers=0)
    model_v3.train()
    total_loss, total_pred, n = 0.0, 0.0, 0
    for past, pm, future, fm, k, t in train_loader:
        past, pm = past.to(DEVICE), pm.to(DEVICE)
        future, fm = future.to(DEVICE), fm.to(DEVICE)
        k = k.to(DEVICE)
        optimizer.zero_grad()
        pred_f, h_f, h_p = model_v3.forward_pretrain(past, pm, future, fm, k)
        loss, pred_l, var_l = trajectory_jepa_loss(pred_f, h_f, 0.01)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_v3.parameters(), 1.0)
        optimizer.step(); model_v3.update_ema()
        B = past.shape[0]
        total_loss += loss.item() * B; total_pred += pred_l.item() * B; n += B
    history_v3['loss'].append(total_loss/n)
    history_v3['pred_loss'].append(total_pred/n)
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        rmse = eval_probe_rmse(model_v3, data['train_engines'], data['val_engines'], D_MODEL)
        history_v3['probe_rmse'].append(rmse)
        history_v3['probe_epochs'].append(epoch)
        if rmse < best_probe_rmse:
            best_probe_rmse = rmse
            best_state = copy.deepcopy(model_v3.state_dict())
            torch.save(best_state, ckpt_path_v3)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 10:
                log(f"  Early stopping at ep {epoch}")
                break
        log(f"V3 Ep {epoch:3d} | loss={total_loss/n:.4f} | probe={rmse:.2f} (best={best_probe_rmse:.2f}, no_impr={no_improve})")
    elif epoch % 50 == 0:
        log(f"V3 Ep {epoch:3d} | loss={total_loss/n:.4f}")

elapsed = (time.time() - t0) / 60
log(f"V3 pretraining done in {elapsed:.1f} min. Best probe RMSE: {best_probe_rmse:.2f}")
if best_state: model_v3.load_state_dict(best_state)

# Fine-tune at 100% and 20%
def run_finetune_v3(train_eng, val_eng, test_eng, test_rul, mode, seed):
    model_ft = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
                               n_heads=4, n_layers=N_LAYERS, d_ff=256)
    model_ft.load_state_dict(torch.load(ckpt_path_v3, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)
    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=True)
    te_ds = CMAPSSTestDataset(test_eng, test_rul)
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)
    if mode == 'frozen':
        for p in model_ft.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model_ft.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(list(model_ft.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4)
    best_v, best_ps, best_es, ni = float('inf'), None, None, 0
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
            loss = F.mse_loss(probe(h), rul)
            loss.backward(); torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0); optim.step()
        model_ft.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model_ft.encode_past(past, mask)).cpu().numpy()); tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_r < best_v:
            best_v = val_r; best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e': best_es = copy.deepcopy(model_ft.context_encoder.state_dict())
            ni = 0
        else:
            ni += 1
            if ni >= 20: break
    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es: model_ft.context_encoder.load_state_dict(best_es)
    model_ft.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rg in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pt.append(probe(model_ft.encode_past(past, mask)).cpu().numpy() * RUL_CAP); tt.append(rg.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

results_v3 = {}
for budget, label in [(1.0, '100%'), (0.2, '20%'), (0.1, '10%')]:
    sub_eng = subsample_engines(data['train_engines'], budget, seed=42)
    for mode in ['frozen', 'e2e']:
        key = f"{mode}_{label}"
        rmses = [run_finetune_v3(sub_eng, data['val_engines'], data['test_engines'],
                                  data['test_rul'], mode, seed) for seed in range(5)]
        results_v3[key] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
        log(f"V3 {mode} @ {label}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

save_json(results_v3, os.path.join(EXP_DIR, 'finetune_results_v3.json'))

# Compare V1/V2/V3
log("\n--- V1 vs V2 vs V3 @ 100% E2E ---")
with open(os.path.join(EXP_DIR, 'finetune_results.json')) as f:
    r_v1 = json.load(f)
    r_v1 = {m: {float(k): v for k, v in d.items()} for m, d in r_v1.items()}
with open(os.path.join(EXP_DIR, 'finetune_results_v2_full.json')) as f:
    r_v2 = json.load(f)
log(f"  V1 (d=128, L=2): E2E={r_v1['jepa_e2e'][1.0]['mean']:.2f}, frozen={r_v1['jepa_frozen'][1.0]['mean']:.2f}")
log(f"  V2 (d=256, L=2): E2E={r_v2['jepa_e2e']['1.0']['mean']:.2f}, frozen={r_v2['jepa_frozen']['1.0']['mean']:.2f}")
log(f"  V3 (d=128, L=3): E2E={results_v3['e2e_100%']['mean']:.2f}, frozen={results_v3['frozen_100%']['mean']:.2f}")
log(f"  STAR supervised: 10.61")
log(f"  AE-LSTM SSL ref: 13.99")

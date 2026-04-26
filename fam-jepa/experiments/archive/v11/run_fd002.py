"""
Part G: FD002 pretraining and cross-subset transfer.
Uses best architecture from FD001 experiments: d_model=256 (V2).
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
from train_utils import subsample_engines
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256

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
log("PART G: FD002 Pretraining + Cross-subset Transfer")
log("="*60)

# Load FD002
log("Loading FD002...")
data_fd002 = load_cmapss_subset('FD002')
log(f"FD002: {len(data_fd002['train_engines'])} train, {len(data_fd002['val_engines'])} val, "
    f"{len(data_fd002['test_engines'])} test engines")

# Also load FD001 for cross-subset eval
data_fd001 = load_cmapss_subset('FD001')

# ============================================================
# G.1: FD002 in-domain pretraining
# ============================================================
log("\n--- G.1: FD002 In-domain Pretraining ---")

model_fd002 = TrajectoryJEPA(
    n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
    n_heads=4, n_layers=2, d_ff=512, dropout=0.1, ema_momentum=0.99
)
log(f"FD002 model params: {count_parameters(model_fd002):,}")
model_fd002 = model_fd002.to(DEVICE)

optimizer = torch.optim.AdamW(
    [p for p in model_fd002.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

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

ckpt_fd002 = os.path.join(EXP_DIR, 'best_pretrain_fd002.pt')
best_pr = float('inf'); best_st = None; ni = 0
t0 = time.time()

for epoch in range(1, 201):
    train_ds = CMAPSSPretrainDataset(data_fd002['train_engines'], n_cuts_per_engine=20,
                                     min_past=10, min_horizon=5, max_horizon=30, seed=epoch)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=collate_pretrain, num_workers=0)
    model_fd002.train()
    total_loss, n = 0.0, 0
    for past, pm, future, fm, k, t in train_loader:
        past, pm = past.to(DEVICE), pm.to(DEVICE)
        future, fm = future.to(DEVICE), fm.to(DEVICE)
        k = k.to(DEVICE)
        optimizer.zero_grad()
        pred_f, h_f, h_p = model_fd002.forward_pretrain(past, pm, future, fm, k)
        loss, _, _ = trajectory_jepa_loss(pred_f, h_f, 0.01)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_fd002.parameters(), 1.0)
        optimizer.step(); model_fd002.update_ema()
        total_loss += loss.item() * past.shape[0]; n += past.shape[0]
    scheduler.step()

    if epoch % 10 == 0 or epoch == 1:
        rmse = eval_probe_rmse(model_fd002, data_fd002['train_engines'], data_fd002['val_engines'], D_MODEL)
        if rmse < best_pr:
            best_pr = rmse; best_st = copy.deepcopy(model_fd002.state_dict())
            torch.save(best_st, ckpt_fd002); ni = 0
        else:
            ni += 1
            if ni >= 10:
                log(f"  FD002 early stop at ep {epoch}")
                break
        log(f"FD002 Ep {epoch:3d} | loss={total_loss/n:.4f} | probe={rmse:.2f} (best={best_pr:.2f})")

elapsed = (time.time() - t0) / 60
log(f"FD002 pretraining done in {elapsed:.1f} min. Best probe RMSE: {best_pr:.2f}")
if best_st: model_fd002.load_state_dict(best_st)

# ============================================================
# G.2: Fine-tune FD002 in-domain
# ============================================================
log("\n--- G.2: FD002 in-domain fine-tuning ---")

def run_finetune(ckpt_path, d_model, n_layers, train_eng, val_eng, test_eng, test_rul, mode, seed):
    model_ft = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=d_model,
                               n_heads=4, n_layers=n_layers, d_ff=512 if d_model==256 else 256)
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model_ft = model_ft.to(DEVICE)
    probe = RULProbe(d_model).to(DEVICE)
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

fd002_results = {}
for mode in ['frozen', 'e2e']:
    rmses = []
    for seed in range(5):
        rmse = run_finetune(ckpt_fd002, D_MODEL, 2,
                             data_fd002['train_engines'], data_fd002['val_engines'],
                             data_fd002['test_engines'], data_fd002['test_rul'],
                             mode, seed)
        rmses.append(rmse)
        log(f"FD002 in-domain {mode} seed={seed}: {rmse:.2f}")
    fd002_results[mode] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
    log(f"FD002 {mode} @ 100%: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

# ============================================================
# G.3: Cross-subset transfer FD002 -> FD001 at 10% labels
# ============================================================
log("\n--- G.3: Cross-subset FD002->FD001 @ 10% labels ---")

cross_results = {}
for mode in ['frozen', 'e2e']:
    rmses = []
    sub_fd001 = subsample_engines(data_fd001['train_engines'], 0.1, seed=42)
    log(f"FD001 10% subset: {len(sub_fd001)} engines")
    for seed in range(5):
        rmse = run_finetune(ckpt_fd002, D_MODEL, 2,
                             sub_fd001, data_fd001['val_engines'],
                             data_fd001['test_engines'], data_fd001['test_rul'],
                             mode, seed)
        rmses.append(rmse)
        log(f"Cross-subset {mode} seed={seed}: {rmse:.2f}")
    cross_results[mode] = {'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses}
    log(f"Cross-subset FD002->FD001 {mode} @ 10%: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

g_results = {
    'fd002_probe_rmse': float(best_pr),
    'fd002_finetune': fd002_results,
    'cross_subset_10pct': cross_results,
}
save_json(g_results, os.path.join(EXP_DIR, 'part_g_results.json'))

# Compare with FD001 in-domain at 10%
with open(os.path.join(EXP_DIR, 'finetune_results_v2_full.json')) as f:
    r_v2 = json.load(f)

log("\n--- Summary: FD001 vs FD002 ---")
log(f"FD001 in-domain STAR ref: 10.61")
log(f"FD002 in-domain STAR ref: 13.47")
log(f"FD001 in-domain V2 E2E@100%: {r_v2['jepa_e2e']['1.0']['mean']:.2f}")
log(f"FD002 in-domain V2 E2E@100%: {fd002_results['e2e']['mean']:.2f}")
log(f"FD001 V2 frozen @ 10%: {r_v2['jepa_frozen']['0.1']['mean']:.2f}")
log(f"FD002->FD001 cross-subset frozen @ 10%: {cross_results['frozen']['mean']:.2f}")
log(f"FD002->FD001 cross-subset E2E @ 10%: {cross_results['e2e']['mean']:.2f}")

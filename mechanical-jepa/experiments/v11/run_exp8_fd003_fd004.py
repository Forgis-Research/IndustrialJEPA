"""
Exp 8: FD003 and FD004 in-domain experiments.
FD003: 100 engines, single condition, 2 fault modes (similar structure to FD001 but harder)
FD004: 249 engines, 6 conditions, 2 fault modes (hardest subset)

This extends Part G to all 4 subsets.
Uses the SAME V2 architecture (d_model=256).
For FD003/FD004, we need to retrain from scratch (no pretrained checkpoints exist).
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
    N_SENSORS, RUL_CAP
)
from models import TrajectoryJEPA, RULProbe, count_parameters, trajectory_jepa_loss
from train_utils import subsample_engines
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256

LOG_FILE = os.path.join(EXP_DIR, 'EXPERIMENT_LOG.md')
def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def save_json(obj, path):
    def default(x):
        if isinstance(x, (np.floating, float)): return float(x)
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, np.ndarray): return x.tolist()
        return str(x)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=default)

def eval_probe_rmse(enc_model, train_eng, val_eng, d_model=D_MODEL, n_epochs=50):
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
            optim_p.zero_grad()
            with torch.no_grad(): h = enc_model.encode_past(past, mask)
            F.mse_loss(probe(h), rul).backward()
            optim_p.step()
    probe.eval()
    pv, tv = [], []
    with torch.no_grad():
        for past, mask, rul in va:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pv.append(probe(enc_model.encode_past(past, mask)).cpu().numpy())
            tv.append(rul.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))

def pretrain_subset(subset, data):
    """Pretrain JEPA on a given subset."""
    ckpt_path = os.path.join(EXP_DIR, f'best_pretrain_{subset.lower()}_v2.pt')

    # Check if checkpoint already exists
    if os.path.exists(ckpt_path):
        log(f"  Checkpoint found: {ckpt_path}, skipping pretraining")
        return ckpt_path

    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
        n_heads=4, n_layers=2, d_ff=512, dropout=0.1, ema_momentum=0.99
    )
    model = model.to(DEVICE)
    log(f"  {subset} model params: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

    best_pr = float('inf'); best_st = None; ni = 0
    t0 = time.time()

    for epoch in range(1, 201):
        train_ds = CMAPSSPretrainDataset(data['train_engines'], n_cuts_per_engine=20,
                                         min_past=10, min_horizon=5, max_horizon=30, seed=epoch)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                                  collate_fn=collate_pretrain, num_workers=0)
        model.train()
        total_loss, n = 0.0, 0
        for past, pm, future, fm, k, t in train_loader:
            past, pm = past.to(DEVICE), pm.to(DEVICE)
            future, fm = future.to(DEVICE), fm.to(DEVICE)
            k = k.to(DEVICE)
            optimizer.zero_grad()
            pred_f, h_f, h_p = model.forward_pretrain(past, pm, future, fm, k)
            loss, _, _ = trajectory_jepa_loss(pred_f, h_f, 0.01)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); model.update_ema()
            total_loss += loss.item() * past.shape[0]; n += past.shape[0]
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            rmse = eval_probe_rmse(model, data['train_engines'], data['val_engines'])
            if rmse < best_pr:
                best_pr = rmse; best_st = copy.deepcopy(model.state_dict())
                torch.save(best_st, ckpt_path); ni = 0
            else:
                ni += 1
                if ni >= 10:
                    log(f"  {subset} early stop at ep {epoch}")
                    break
            log(f"{subset} Ep {epoch:3d} | loss={total_loss/n:.4f} | probe={rmse:.2f} (best={best_pr:.2f})")

    elapsed = (time.time() - t0) / 60
    log(f"{subset} pretraining done in {elapsed:.1f} min. Best probe RMSE: {best_pr:.2f}")
    return ckpt_path

def run_finetune(ckpt_path, train_eng, val_eng, test_eng, test_rul, mode, seed):
    model_ft = TrajectoryJEPA(n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL,
                               n_heads=4, n_layers=2, d_ff=512)
    model_ft.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
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
        optim = torch.optim.Adam(
            list(model_ft.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4
        )

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
            F.mse_loss(probe(h), rul).backward()
            torch.nn.utils.clip_grad_norm_(list(probe.parameters()) +
                (list(model_ft.context_encoder.parameters()) if mode == 'e2e' else []), 1.0)
            optim.step()

        model_ft.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pv.append(probe(model_ft.encode_past(past, mask)).cpu().numpy())
                tv.append(rul.numpy())
        val_r = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2
        )))
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
            pt.append(probe(model_ft.encode_past(past, mask)).cpu().numpy() * RUL_CAP)
            tt.append(rg.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))

# ============================================================
# Main
# ============================================================
log("\n" + "="*60)
log("EXP 8: FD003 and FD004 In-domain Experiments")
log("="*60)

STAR_REFS = {'FD001': 10.61, 'FD002': 13.47, 'FD003': 10.71, 'FD004': 14.25}
all_results = {}

for subset in ['FD003', 'FD004']:
    log(f"\n--- {subset} ---")
    log(f"Loading {subset}...")
    data = load_cmapss_subset(subset)
    log(f"{subset}: {len(data['train_engines'])} train, {len(data['val_engines'])} val, "
        f"{len(data['test_engines'])} test")

    ckpt_path = pretrain_subset(subset, data)

    # Fine-tune at 100% and 20%
    log(f"\n{subset} Fine-tuning at 100% and 20%...")
    subset_results = {}
    for budget in [1.0, 0.2, 0.1]:
        sub_eng = subsample_engines(data['train_engines'], budget, seed=42)
        subset_results[str(budget)] = {}
        for mode in ['frozen', 'e2e']:
            rmses = []
            for seed in range(5):
                rmse = run_finetune(ckpt_path, sub_eng, data['val_engines'],
                                    data['test_engines'], data['test_rul'], mode, seed)
                rmses.append(rmse)
                log(f"  {subset} {mode} @ {budget:.0%} seed={seed}: {rmse:.2f}")
            subset_results[str(budget)][mode] = {
                'mean': float(np.mean(rmses)), 'std': float(np.std(rmses)), 'all': rmses
            }
            log(f"  {subset} {mode} @ {budget:.0%}: {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}")

    all_results[subset] = subset_results
    log(f"\n{subset} Summary:")
    log(f"  E2E @ 100%: {subset_results['1.0']['e2e']['mean']:.2f} vs STAR {STAR_REFS[subset]}")
    log(f"  Frozen @ 100%: {subset_results['1.0']['frozen']['mean']:.2f}")
    log(f"  E2E @ 20%: {subset_results['0.2']['e2e']['mean']:.2f}")

save_json(all_results, os.path.join(EXP_DIR, 'exp8_fd3_fd4_results.json'))

# Summary table
log("\n## Exp 8: All-subset Summary")
log("| Subset | JEPA E2E@100% | JEPA Frozen@100% | STAR ref |")
log("|:------:|:-------------:|:----------------:|:--------:|")
for subset in ['FD003', 'FD004']:
    if subset in all_results and '1.0' in all_results[subset]:
        e2e_m = all_results[subset]['1.0']['e2e']['mean']
        e2e_s = all_results[subset]['1.0']['e2e']['std']
        frz_m = all_results[subset]['1.0']['frozen']['mean']
        log(f"| {subset} | {e2e_m:.2f}+/-{e2e_s:.2f} | {frz_m:.2f} | {STAR_REFS[subset]} |")

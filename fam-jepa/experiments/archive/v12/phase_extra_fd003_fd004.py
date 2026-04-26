"""
Extra diagnostics: Replicate Phase 0 verdict logic on FD003 and FD004.
Do FD003 and FD004 also show real tracking?

For each subset:
- Load pretrained checkpoint
- Reconstruct E2E (seed=0)
- Run trajectory inference at every cycle for 10 test engines
- Report per-engine pred std, rho
- Compare to engine-summary regressor

Output: experiments/v12/extra_fd003_fd004_diagnostics.json
"""

import json
import sys
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
PLOTS_V12 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/analysis/plots/v12')

sys.path.insert(0, str(V11_DIR))
from data_utils import (
    load_cmapss_subset, load_raw, get_sensor_cols, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset, collate_finetune, collate_test,
    fit_normalizer, build_engine_sequences
)
from models import TrajectoryJEPA, RULProbe
from train_utils import DEVICE, finetune

CHECKPOINTS = {
    'FD003': V11_DIR / 'best_pretrain_fd003_v2.pt',
    'FD004': V11_DIR / 'best_pretrain_fd004_v2.pt',
}
V11_RESULTS = {
    'FD003': {'e2e': 15.37, 'frozen': 19.25},
    'FD004': {'e2e': 25.62, 'frozen': 29.35},
}

print("=" * 60)
print("Extra: FD003 and FD004 Phase 0 Replication")
print("=" * 60)

results_all = {}

for subset in ['FD003', 'FD004']:
    print(f"\n{'='*50}")
    print(f"Subset: {subset}")
    print(f"{'='*50}")
    t0 = time.time()

    ckpt_path = CHECKPOINTS[subset]
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        continue

    data = load_cmapss_subset(subset)
    train_engines = data['train_engines']
    val_engines = data['val_engines']
    test_engines = data['test_engines']
    test_rul = data['test_rul']
    print(f"  Train: {len(train_engines)}, Val: {len(val_engines)}, Test: {len(test_engines)}")

    # Build model
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=256, n_heads=4, n_layers=2,
        d_ff=512, dropout=0.1, ema_momentum=0.996, predictor_hidden=256
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))

    # E2E fine-tune at seed=0
    print(f"  Fine-tuning E2E (seed=0)...")
    probe = RULProbe(256).to(DEVICE)
    torch.manual_seed(0); np.random.seed(0)

    train_ds = CMAPSSFinetuneDataset(train_engines, n_cuts_per_engine=5, seed=0)
    val_ds = CMAPSSFinetuneDataset(val_engines, use_last_only=True)
    test_ds = CMAPSSTestDataset(test_engines, test_rul)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    for p in model.context_encoder.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters()) + list(model.predictor.parameters()) + list(probe.parameters()),
        lr=1e-4
    )

    best_val, best_enc, best_pb = float('inf'), None, None
    patience_count = 0

    for epoch in range(1, 101):
        model.train(); probe.train()
        for past, mask, rul in train_loader:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optimizer.zero_grad()
            h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

        model.eval(); probe.eval()
        preds_v, tgts_v = [], []
        with torch.no_grad():
            for past, mask, rul in val_loader:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pred = probe(h)
                preds_v.append(pred.cpu().numpy())
                tgts_v.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean((np.concatenate(preds_v)*RUL_CAP - np.concatenate(tgts_v)*RUL_CAP)**2)))

        if val_rmse < best_val:
            best_val = val_rmse
            best_enc = copy.deepcopy(model.context_encoder.state_dict())
            best_pb = copy.deepcopy(probe.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 20:
                break

    model.context_encoder.load_state_dict(best_enc)
    probe.load_state_dict(best_pb)

    # Test RMSE
    model.eval(); probe.eval()
    preds_t, tgts_t = [], []
    with torch.no_grad():
        for past, mask, rul_gt in test_loader:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            preds_t.append(pred.cpu().numpy() * RUL_CAP)
            tgts_t.append(rul_gt.numpy())
    preds_t = np.concatenate(preds_t)
    tgts_t = np.concatenate(tgts_t)
    test_rmse_seed0 = float(np.sqrt(np.mean((preds_t - tgts_t)**2)))
    print(f"  Test RMSE (seed=0): {test_rmse_seed0:.2f} (V11 reported: {V11_RESULTS[subset]['e2e']:.2f})")

    # Trajectory inference at every cycle for first 10 test engines
    test_engine_ids = sorted(test_engines.keys())

    @torch.no_grad()
    def infer_at_cycle(model, probe_net, seq, cycle_idx):
        prefix = seq[:cycle_idx+1]
        x = torch.tensor(prefix, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        h = model.encode_past(x, None)
        pred_norm = probe_net(h)
        return float(pred_norm.cpu().numpy()[0]) * RUL_CAP

    pred_stds = []
    rhos = []

    n_track = min(20, len(test_engine_ids))  # Use 20 engines for FD003/004 analysis
    for idx in range(n_track):
        eid = test_engine_ids[idx]
        seq = test_engines[eid]
        T = seq.shape[0]
        oracle_rul = float(test_rul[idx])
        min_cycle = min(30, T)
        cycles = list(range(min_cycle, T+1))

        preds = []
        true_ruls = []
        for c in cycles:
            true_rul_c = min(oracle_rul + (T - c), RUL_CAP)
            pred = infer_at_cycle(model, probe, seq, c-1)
            preds.append(pred)
            true_ruls.append(true_rul_c)

        preds = np.array(preds)
        true_ruls = np.array(true_ruls)
        pred_std = float(np.std(preds))
        if len(preds) >= 3 and preds.std() > 0 and true_ruls.std() > 0:
            rho, _ = spearmanr(preds, true_ruls)
            if np.isnan(rho): rho = 0.0
        else:
            rho = 0.0

        pred_stds.append(pred_std)
        rhos.append(float(rho))

    pred_stds = np.array(pred_stds)
    rhos = np.array(rhos)
    print(f"  Pred std (first {n_track} engines): mean={pred_stds.mean():.2f}, median={np.median(pred_stds):.2f}")
    print(f"  Rho (first {n_track} engines): mean={rhos.mean():.3f}, median={np.median(rhos):.3f}")

    # Engine-summary regressor for FD003/FD004
    train_df, test_df, test_rul_arr = load_raw(subset)
    stats = fit_normalizer(train_df, per_condition=(subset in ('FD002', 'FD004')))
    all_train_seqs = build_engine_sequences(train_df, stats, per_condition=(subset in ('FD002', 'FD004')))
    all_test_seqs = build_engine_sequences(test_df, stats, per_condition=(subset in ('FD002', 'FD004')))
    all_ids = sorted(all_train_seqs.keys())
    mean_T = np.mean([all_train_seqs[i].shape[0] for i in all_ids])
    test_eids = sorted(all_test_seqs.keys())

    def compute_eng_feats(seq, mean_T):
        T = seq.shape[0]
        last_n = min(30, T); first_n = min(30, T)
        last_seg = seq[-last_n:]; first_seg = seq[:first_n]
        feats = [float(T), float(T)/mean_T]
        feats.extend(last_seg.mean(axis=0).tolist())
        feats.extend(last_seg.std(axis=0).tolist())
        if last_n >= 2:
            x = np.arange(last_n, dtype=np.float32); xm = x.mean(); xv = np.sum((x-xm)**2)
            if xv > 0:
                slopes = np.sum((x[:,None]-xm)*(last_seg-last_seg.mean(axis=0)),axis=0)/xv
            else:
                slopes = np.zeros(last_seg.shape[1])
        else:
            slopes = np.zeros(last_seg.shape[1])
        feats.extend(slopes.tolist())
        feats.extend((last_seg.mean(axis=0) - first_seg.mean(axis=0)).tolist())
        return np.array(feats, dtype=np.float32)

    rng = np.random.default_rng(42)
    n_val_reg = max(1, int(0.15 * len(all_ids)))
    val_ids_reg = set(rng.choice(all_ids, size=n_val_reg, replace=False).tolist())
    train_ids_reg = [i for i in all_ids if i not in val_ids_reg]

    X_multi = []; y_multi = []
    for eid in train_ids_reg:
        seq = all_train_seqs[eid]; T = seq.shape[0]
        for cp in sorted(set([30, max(30, T//2), max(30, 3*T//4), T])):
            if cp > T: cp = T
            prefix = seq[:cp]
            feats = compute_eng_feats(prefix, mean_T)
            rul_at_cut = float(min(T - cp + 1, RUL_CAP))
            X_multi.append(feats); y_multi.append(rul_at_cut)

    X_test_reg = np.stack([compute_eng_feats(all_test_seqs[eid], mean_T) for eid in test_eids])
    y_test_reg = test_rul_arr

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(np.stack(X_multi))
    X_test_scaled = scaler.transform(X_test_reg)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, np.array(y_multi))
    preds_reg = np.clip(ridge.predict(X_test_scaled), 0, RUL_CAP)
    reg_rmse = float(np.sqrt(np.mean((preds_reg - y_test_reg)**2)))
    print(f"  Engine-summary regressor RMSE: {reg_rmse:.2f}")
    print(f"  V11 E2E RMSE: {V11_RESULTS[subset]['e2e']:.2f}")
    print(f"  Delta (JEPA vs regressor): {V11_RESULTS[subset]['e2e'] - reg_rmse:+.2f}")

    results_all[subset] = {
        "test_rmse_seed0": test_rmse_seed0,
        "v11_reported_e2e": V11_RESULTS[subset]['e2e'],
        "v11_reported_frozen": V11_RESULTS[subset]['frozen'],
        "pred_std_mean": float(pred_stds.mean()),
        "pred_std_median": float(np.median(pred_stds)),
        "rho_mean": float(rhos.mean()),
        "rho_median": float(np.median(rhos)),
        "engine_summary_reg_rmse": reg_rmse,
        "delta_jepa_vs_regressor": float(V11_RESULTS[subset]['e2e'] - reg_rmse),
        "n_engines_analyzed": n_track,
        "wall_time_s": float(time.time() - t0),
    }

with open(V12_DIR / 'extra_fd003_fd004_diagnostics.json', 'w') as f:
    json.dump(results_all, f, indent=2)

print(f"\nSaved to {V12_DIR / 'extra_fd003_fd004_diagnostics.json'}")

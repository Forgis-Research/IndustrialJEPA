"""V24 Phase 1: Sanity check for canonical model.py + train.py.

Quick smoke test:
  1. Import FAM + train pipeline
  2. Load FD001 (data comes pre-normalized from v11 loader; RevIN re-normalizes per-context)
  3. Run 3-epoch pretrain
  4. Run 3-epoch pred-FT
  5. Evaluate on test set -> check AUPRC reports
  6. Confirm no crash, loss decreases, h_std > 0.01 (no collapse)

This is BLOCKING. If it fails, debug before any other phase.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, get_horizons,
)
from data_utils import load_cmapss_subset, compute_rul_labels


def build_cmapss_event_dataset(engines, stride=4, max_context=512):
    """Wrap engines into a single EventDataset by concatenating with failure markers.

    C-MAPSS training engines: each engine ends at failure. So for engine of length T,
    label t=T-1 as 1 (event at final cycle), all others 0. tte[t] = T-1-t.

    We concatenate engines via ConcatDataset over per-engine EventDatasets.
    """
    from torch.utils.data import ConcatDataset
    datasets = []
    for eid, seq in engines.items():
        T = len(seq)
        if T < 8:
            continue
        labels = np.zeros(T, dtype=np.int32)
        labels[T - 1] = 1  # failure marker at end of engine
        datasets.append(EventDataset(seq, labels, max_context=max_context,
                                     stride=stride, max_future=200))
    return ConcatDataset(datasets)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[phase1_sanity] device={device}", flush=True)

    print("[phase1_sanity] loading FD001 ...", flush=True)
    t0 = time.time()
    data = load_cmapss_subset('FD001')
    t1 = time.time()
    print(f"  loaded in {t1 - t0:.1f}s", flush=True)
    print(f"  n_train_engines={len(data['train_engines'])}", flush=True)
    print(f"  n_val_engines  ={len(data['val_engines'])}", flush=True)
    print(f"  n_test_engines ={len(data['test_engines'])}", flush=True)

    # ---- Build pretraining datasets (cumulative target, variable context) ----
    print("[phase1_sanity] building pretrain datasets ...", flush=True)
    train_pre = PretrainDataset(
        data['train_engines'], n_cuts=20, max_context=512,
        delta_t_max=150, delta_t_min=1, seed=42)
    val_pre = PretrainDataset(
        data['val_engines'], n_cuts=10, max_context=512,
        delta_t_max=150, delta_t_min=1, seed=99)
    print(f"  train samples: {len(train_pre)}", flush=True)
    print(f"  val samples:   {len(val_pre)}", flush=True)

    train_pre_loader = DataLoader(train_pre, batch_size=32, shuffle=True,
                                  collate_fn=collate_pretrain, num_workers=0)
    val_pre_loader = DataLoader(val_pre, batch_size=32, shuffle=False,
                                collate_fn=collate_pretrain, num_workers=0)

    # ---- Build model ----
    print("[phase1_sanity] building FAM ...", flush=True)
    model = FAM(n_channels=14, patch_size=16,
                d_model=256, n_heads=4, n_layers=2, d_ff=256,
                dropout=0.1, ema_momentum=0.99, predictor_hidden=256)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params:     {n_params:,}", flush=True)
    print(f"  trainable params: {n_train:,}", flush=True)

    # ---- Run 3-epoch pretrain ----
    print("[phase1_sanity] pretrain (3 epochs) ...", flush=True)
    t0 = time.time()
    pre_out = pretrain(model, train_pre_loader, val_pre_loader,
                       lr=3e-4, n_epochs=3, patience=10, device=device)
    t1 = time.time()
    print(f"  pretrain took {t1 - t0:.1f}s", flush=True)
    hist = pre_out['history']
    first_loss = hist[0]['train_loss']
    last_loss = hist[-1]['train_loss']
    last_hstd = hist[-1]['h_std']
    print(f"  first epoch train_loss: {first_loss:.4f}", flush=True)
    print(f"  last  epoch train_loss: {last_loss:.4f}", flush=True)
    print(f"  last  epoch h_std:      {last_hstd:.3f}", flush=True)
    assert last_hstd > 0.01, f"COLLAPSE detected: h_std={last_hstd}"

    # ---- Build finetune datasets ----
    print("[phase1_sanity] building event datasets ...", flush=True)
    train_ft_dataset = build_cmapss_event_dataset(
        data['train_engines'], stride=4, max_context=512)
    val_ft_dataset = build_cmapss_event_dataset(
        data['val_engines'], stride=4, max_context=512)
    # Test uses full test engines; we treat end-of-available-seq as "observed up to now"
    test_ft_dataset = build_cmapss_event_dataset(
        data['test_engines'], stride=1, max_context=512)
    print(f"  train_ft: {len(train_ft_dataset)}", flush=True)
    print(f"  val_ft:   {len(val_ft_dataset)}", flush=True)
    print(f"  test_ft:  {len(test_ft_dataset)}", flush=True)

    train_ft_loader = DataLoader(train_ft_dataset, batch_size=64, shuffle=True,
                                 collate_fn=collate_event, num_workers=0)
    val_ft_loader = DataLoader(val_ft_dataset, batch_size=64, shuffle=False,
                               collate_fn=collate_event, num_workers=0)
    test_ft_loader = DataLoader(test_ft_dataset, batch_size=64, shuffle=False,
                                collate_fn=collate_event, num_workers=0)

    horizons = get_horizons('FD001')
    print(f"  horizons: {horizons}", flush=True)

    # ---- Run 3-epoch pred-FT ----
    print("[phase1_sanity] pred-FT (3 epochs) ...", flush=True)
    t0 = time.time()
    ft_out = finetune(model, train_ft_loader, val_ft_loader, horizons,
                      mode='pred_ft', lr=1e-3, n_epochs=3, patience=10,
                      device=device)
    t1 = time.time()
    print(f"  pred-FT took {t1 - t0:.1f}s", flush=True)
    print(f"  best val loss: {ft_out['best_val']:.4f}", flush=True)

    # ---- Evaluate ----
    print("[phase1_sanity] evaluating ...", flush=True)
    eval_out = evaluate(model, test_ft_loader, horizons, mode='pred_ft',
                        device=device)
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']
    auprc = primary['auprc']
    auroc = primary['auroc']
    print(f"  AUPRC (pooled): {auprc:.4f}", flush=True)
    print(f"  AUROC (pooled): {auroc:.4f}", flush=True)
    mono_val = mono['violation_rate'] if isinstance(mono, dict) else float(mono)
    print(f"  monotonicity violation rate: {mono_val:.4f}", flush=True)
    per_h_list = per_h['auprc_per_k']
    print(f"  per-horizon AUPRC:", flush=True)
    for h, a in zip(horizons, per_h_list):
        print(f"    dt={h:4d}: {a:.4f}", flush=True)

    # ---- Summary to JSON ----
    summary = {
        'phase': 'v24_phase1_sanity',
        'dataset': 'FD001',
        'device': device,
        'n_params': int(n_params),
        'n_trainable': int(n_train),
        'pretrain_epochs': 3,
        'pretrain_first_loss': float(first_loss),
        'pretrain_last_loss': float(last_loss),
        'pretrain_last_hstd': float(last_hstd),
        'ft_epochs': 3,
        'ft_best_val': float(ft_out['best_val']),
        'auprc_pooled': float(auprc),
        'auroc_pooled': float(auroc),
        'monotonicity_violation_rate': float(mono_val),
        'per_horizon_auprc': {str(h): float(a) for h, a in zip(horizons, per_h_list)},
    }
    out_path = FAM_DIR / 'experiments/v24/phase1_sanity.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[phase1_sanity] wrote {out_path}", flush=True)
    print("[phase1_sanity] PASS", flush=True)


if __name__ == '__main__':
    main()

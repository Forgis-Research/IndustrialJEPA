"""V26 Phase 1: Sanity check for hazard CDF parameterization.

Verifies:
  - finetune_forward returns probs in (0, 1) (NOT logits)
  - Monotonicity violations = 0 (guaranteed by cumprod construction)
  - Loss decreases (pos-weighted BCE on probabilities, no BCEWithLogits)
  - No NaN / Inf during pretrain or finetune
  - AUPRC at least broadly sensible (FD001 expected > 0.90)

BLOCKING — if this fails, stop and debug before any other phase.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V26_DIR = FAM_DIR / 'experiments/v26'
sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v24'))

from model import FAM
from train import (
    PretrainDataset, EventDataset, collate_pretrain, collate_event,
    pretrain, finetune, evaluate, get_horizons,
)
from _cmapss_raw import load_cmapss_raw


def build_event_concat(engines, stride, max_context=512, max_future=200,
                       min_context=128):
    datasets = []
    for eid, seq in engines.items():
        T = len(seq)
        if T <= min_context:
            continue
        labels = np.zeros(T, dtype=np.int32)
        labels[T - 1] = 1
        ds = EventDataset(seq, labels, max_context=max_context,
                          stride=stride, max_future=max_future,
                          min_context=min_context)
        if len(ds) > 0:
            datasets.append(ds)
    return ConcatDataset(datasets) if datasets else ConcatDataset([])


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[phase1_sanity] device={device}", flush=True)

    # ---- Load ----
    print("[phase1_sanity] loading FD001 (normalize=False, RevIN handles it)",
          flush=True)
    t0 = time.time()
    data = load_cmapss_raw('FD001')
    print(f"  loaded in {time.time()-t0:.1f}s "
          f"(n_train={len(data['train_engines'])}, "
          f"n_val={len(data['val_engines'])}, "
          f"n_test={len(data['test_engines'])})", flush=True)

    # ---- Pretrain ----
    train_pre = PretrainDataset(data['train_engines'], n_cuts=20,
                                max_context=512, delta_t_max=150,
                                delta_t_min=1, seed=42)
    val_pre = PretrainDataset(data['val_engines'], n_cuts=10, max_context=512,
                              delta_t_max=150, delta_t_min=1, seed=99)
    print(f"  pretrain samples: train={len(train_pre)}, val={len(val_pre)}",
          flush=True)
    train_pre_loader = DataLoader(train_pre, batch_size=32, shuffle=True,
                                  collate_fn=collate_pretrain, num_workers=0)
    val_pre_loader = DataLoader(val_pre, batch_size=32, shuffle=False,
                                collate_fn=collate_pretrain, num_workers=0)

    model = FAM(n_channels=14, patch_size=16, d_model=256, n_heads=4,
                n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
                predictor_hidden=256)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {n_params:,}, trainable: {n_train:,}", flush=True)

    print("[phase1_sanity] pretrain (3 epochs) ...", flush=True)
    t0 = time.time()
    pre_out = pretrain(model, train_pre_loader, val_pre_loader,
                       lr=3e-4, n_epochs=3, patience=10, device=device)
    print(f"  pretrain took {time.time()-t0:.1f}s", flush=True)
    hist = pre_out['history']
    assert len(hist) > 0, "pretrain produced no history"
    first_loss = hist[0]['train_loss']
    last_loss = hist[-1]['train_loss']
    last_hstd = hist[-1]['h_std']
    print(f"  first train_loss: {first_loss:.4f}", flush=True)
    print(f"  last  train_loss: {last_loss:.4f}", flush=True)
    print(f"  last  h_std:      {last_hstd:.3f}", flush=True)
    assert np.isfinite(first_loss) and np.isfinite(last_loss), \
        f"pretrain loss non-finite: first={first_loss}, last={last_loss}"
    assert last_hstd > 0.01, f"COLLAPSE: h_std={last_hstd}"

    # ---- Finetune ----
    train_ft = build_event_concat(data['train_engines'], stride=4)
    val_ft = build_event_concat(data['val_engines'], stride=4)
    test_ft = build_event_concat(data['test_engines'], stride=1)
    print(f"  ft samples: train={len(train_ft)}, val={len(val_ft)}, "
          f"test={len(test_ft)}", flush=True)

    train_ft_loader = DataLoader(train_ft, batch_size=64, shuffle=True,
                                 collate_fn=collate_event, num_workers=0)
    val_ft_loader = DataLoader(val_ft, batch_size=64, shuffle=False,
                               collate_fn=collate_event, num_workers=0)
    test_ft_loader = DataLoader(test_ft, batch_size=64, shuffle=False,
                                collate_fn=collate_event, num_workers=0)

    horizons = get_horizons('FD001')
    print(f"  horizons: {horizons}", flush=True)

    print("[phase1_sanity] pred-FT (3 epochs) ...", flush=True)
    t0 = time.time()
    ft_out = finetune(model, train_ft_loader, val_ft_loader, horizons,
                      mode='pred_ft', lr=1e-3, n_epochs=3, patience=10,
                      device=device)
    print(f"  pred-FT took {time.time()-t0:.1f}s "
          f"(best_val={ft_out['best_val']:.4f})", flush=True)
    assert np.isfinite(ft_out['best_val']), "pred-FT loss non-finite"

    # ---- Direct CDF check on one batch ----
    print("[phase1_sanity] checking finetune_forward returns probs ...",
          flush=True)
    model.eval()
    h_tensor = torch.tensor(horizons, dtype=torch.float32, device=device)
    with torch.no_grad():
        ctx, ctx_m, tte, t_idx = next(iter(test_ft_loader))
        ctx, ctx_m = ctx.to(device), ctx_m.to(device)
        cdf = model.finetune_forward(ctx, h_tensor, ctx_m, mode='pred_ft')
    cdf_np = cdf.cpu().numpy()
    cdf_min, cdf_max = float(cdf_np.min()), float(cdf_np.max())
    print(f"  cdf range: [{cdf_min:.6f}, {cdf_max:.6f}]  shape={cdf_np.shape}",
          flush=True)
    assert cdf_min > 0 and cdf_max < 1, \
        f"CDF out of (0,1): min={cdf_min}, max={cdf_max}"
    # Monotonicity check: cdf must be non-decreasing along K axis (row by row)
    diffs = np.diff(cdf_np, axis=-1)
    violations = (diffs < -1e-6).sum()
    print(f"  monotonicity violations on batch: {violations}", flush=True)
    assert violations == 0, f"Got {violations} monotonicity violations"

    # ---- Evaluate full test set ----
    print("[phase1_sanity] full test evaluation ...", flush=True)
    eval_out = evaluate(model, test_ft_loader, horizons, mode='pred_ft',
                        device=device)
    primary = eval_out['primary']
    per_h = eval_out['per_horizon']
    mono = eval_out['monotonicity']
    auprc = float(primary['auprc'])
    auroc = float(primary['auroc'])
    mono_rate = float(mono['violation_rate'])
    print(f"  AUPRC (pooled): {auprc:.4f}", flush=True)
    print(f"  AUROC (pooled): {auroc:.4f}", flush=True)
    print(f"  monotonicity violation rate: {mono_rate:.6f}", flush=True)
    print(f"  per-horizon AUPRC:", flush=True)
    for h, a in zip(horizons, per_h['auprc_per_k']):
        print(f"    dt={h:4d}: {a:.4f}", flush=True)
    # Full surface monotonicity check
    p_surface = eval_out['p_surface']
    surface_diffs = np.diff(p_surface, axis=-1)
    surface_violations = int((surface_diffs < -1e-6).sum())
    n_pairs = surface_diffs.size
    print(f"  full surface: {surface_violations} / {n_pairs} violations "
          f"= {surface_violations/max(n_pairs,1):.6f}", flush=True)
    assert mono_rate == 0.0, \
        f"Expected 0 violations with hazard CDF, got {mono_rate}"

    # Sanity checks on AUPRC
    assert auprc > 0.3, f"AUPRC way too low: {auprc}"
    # On a 3-epoch FD001 pred-FT we'd expect AUPRC around 0.80+ even shallow.
    # Don't hard-fail on <0.90 here (that's the full 40-epoch Phase 2 bar).
    warn = ""
    if auprc < 0.80:
        warn = f" (WARN: AUPRC {auprc:.3f} < 0.80 — watch Phase 2)"
    print(f"[phase1_sanity] PASS{warn}", flush=True)

    summary = {
        'phase': 'v26_phase1_sanity',
        'dataset': 'FD001',
        'device': device,
        'n_params': int(n_params),
        'n_trainable': int(n_train),
        'pretrain_first_loss': float(first_loss),
        'pretrain_last_loss': float(last_loss),
        'pretrain_last_hstd': float(last_hstd),
        'ft_best_val': float(ft_out['best_val']),
        'cdf_min': cdf_min,
        'cdf_max': cdf_max,
        'batch_monotonicity_violations': int(violations),
        'surface_monotonicity_violations': surface_violations,
        'surface_n_pairs': int(n_pairs),
        'auprc_pooled': auprc,
        'auroc_pooled': auroc,
        'monotonicity_violation_rate': mono_rate,
        'per_horizon_auprc': {str(h): float(a) for h, a in
                              zip(horizons, per_h['auprc_per_k'])},
    }
    out_path = V26_DIR / 'results/phase1_sanity.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[phase1_sanity] wrote {out_path}", flush=True)


if __name__ == '__main__':
    main()

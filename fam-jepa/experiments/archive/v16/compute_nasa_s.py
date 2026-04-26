"""
Compute NASA-S (PHM asymmetric scoring function) for Traj JEPA V2 on C-MAPSS FD001.
Uses V12 reconstructed E2E checkpoint (seed 0).
Runs on CPU only (GPU is busy with Phase 3 SMAP).

NASA-S formula: S = sum_i[exp(d_i/a) - 1]
  where d_i = pred_i - true_i, a=10 if d_i>=0 (late), a=13 if d_i<0 (early)
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V12_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v12')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, CMAPSSTestDataset, collate_test, N_SENSORS, RUL_CAP
)
from models import TrajectoryJEPA, RULProbe

DEVICE = torch.device('cpu')  # CPU only


def nasa_s(pred, true):
    """NASA asymmetric scoring function (PHM 2008 challenge)."""
    d = pred - true  # positive = late (underestimate RUL), more penalized
    score = np.where(d < 0, np.exp(-d/13.0) - 1, np.exp(d/10.0) - 1)
    return float(np.sum(score))


def run_inference(model, probe, test_eng, test_rul):
    """Run frozen encoder + probe on test set, return predictions and true values."""
    test_ds = CMAPSSTestDataset(test_eng, test_rul)
    te = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    model.eval()
    probe.eval()
    preds, trues = [], []
    with torch.no_grad():
        for past, mask, rul in te:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            h = model.encode_past(past, mask)
            pred = probe(h)
            preds.extend(pred.cpu().numpy())
            trues.extend(rul.cpu().numpy())

    preds = np.array(preds) * RUL_CAP  # probe outputs [0,1], unnormalize to cycles
    trues = np.array(trues)  # CMAPSSTestDataset already in raw cycles
    return preds, trues


def main():
    print("=" * 50)
    print("NASA-S Computation for Traj JEPA V2 on FD001")
    print("=" * 50)

    # Load data
    subset = load_cmapss_subset('FD001')
    test_eng = subset['test_engines']
    test_rul = subset['test_rul']
    print(f"Test engines: {len(test_eng)}, test RUL points: {len(test_rul)}")

    # Load V12 reconstructed checkpoint (seed 0)
    ckpt_path = V12_DIR / 'v2_e2e_seed0_reconstructed.pt'
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"Checkpoint test_rmse: {ckpt['test_rmse']:.4f}")

    # Reconstruct model
    # V2 architecture: 2 layers, d_model=256, d_ff=512 (inferred from checkpoint)
    d_model = 256
    n_heads = 4
    n_layers = 2
    dropout = 0.1

    # Try to infer architecture from checkpoint
    enc_keys = list(ckpt['encoder_state'].keys())
    print("Encoder state keys (first 5):", enc_keys[:5])

    # d_ff=512 (inferred from checkpoint: ff.0 shape is [512, 256])
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ff=512, dropout=dropout
    ).to(DEVICE)

    # Load encoder state
    try:
        model.context_encoder.load_state_dict(ckpt['encoder_state'])
        print("Encoder loaded successfully")
    except Exception as e:
        print(f"Encoder load failed: {e}")
        print("Trying alternative load...")
        # Try loading with strict=False
        missing, unexpected = model.context_encoder.load_state_dict(ckpt['encoder_state'], strict=False)
        print(f"Missing: {missing[:3]}, Unexpected: {unexpected[:3]}")

    # Load probe
    probe = RULProbe(d_model).to(DEVICE)
    try:
        probe.load_state_dict(ckpt['probe_state'])
        print("Probe loaded successfully")
    except Exception as e:
        print(f"Probe load failed: {e}")
        missing, unexpected = probe.load_state_dict(ckpt['probe_state'], strict=False)
        print(f"Missing: {missing[:3]}, Unexpected: {unexpected[:3]}")

    # Run inference
    print("\nRunning inference on test set...")
    preds, trues = run_inference(model, probe, test_eng, test_rul)

    rmse = float(np.sqrt(np.mean((preds - trues)**2)))
    nasa = nasa_s(preds, trues)

    print(f"\nResults:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  NASA-S: {nasa:.2f}")
    print(f"  N predictions: {len(preds)}")
    print(f"  Pred mean: {preds.mean():.2f}, std: {preds.std():.2f}")
    print(f"  True mean: {trues.mean():.2f}, std: {trues.std():.2f}")

    # d_i statistics
    d = preds - trues
    print(f"  d = pred - true: mean={d.mean():.2f}, std={d.std():.2f}")
    print(f"  Late predictions (d>0): {(d>0).sum()} / {len(d)}")
    print(f"  Early predictions (d<0): {(d<0).sum()} / {len(d)}")

    results = {
        'checkpoint': str(ckpt_path),
        'checkpoint_test_rmse': float(ckpt['test_rmse']),
        'computed_rmse': rmse,
        'nasa_s': nasa,
        'n_predictions': len(preds),
        'd_mean': float(d.mean()),
        'd_std': float(d.std()),
        'n_late': int((d>0).sum()),
        'n_early': int((d<0).sum()),
    }

    out_path = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v16/nasa_s_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()

"""
V18 Phase 11: Paired FAM vs STAR test (per-engine, addresses round-3/4 path c).

Round-4 reviewer-A: "A true paired test on matched engine-subset splits
requires re-running STAR with seed-matched subsets". Actually, both STAR and
FAM evaluate on the SAME 100 FD001 test engines - so we CAN do a paired
per-engine residual test even without matched training seeds.

Setup:
  STAR: 5 seeds (42, 123, 456, 789, 1024), per-seed test predictions saved
        in paper-replications/star/results/FD001_results.json
  FAM E2E 100%: 5 seeds (0, 1, 2, 3, 4), per-seed predictions newly saved
        in phase9_nasa_score.json (seeds 0, 1, 2 - re-use those)
  FAM E2E 100% needs seeds 3, 4 too - we retrain those here.

Test:
  For each test engine i: residual_i^FAM = avg_over_seeds(pred_i - true_i)
                         residual_i^STAR = same for STAR
  Paired comparison: paired t-test on (|res^FAM| - |res^STAR|) across 100 engines.
  Also bootstrap CI.

Output: experiments/v18/phase11_paired_star.json
"""

import sys, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy import stats

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
V18 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v18')
STAR = Path('/home/sagemaker-user/IndustrialJEPA/paper-replications/star')
ROOT = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')
sys.path.insert(0, str(V11)); sys.path.insert(0, str(ROOT))

from models import TrajectoryJEPA, RULProbe
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                        CMAPSSFinetuneDataset, CMAPSSTestDataset,
                        collate_finetune, collate_test)
from train_utils import subsample_engines

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D_MODEL = 256; N_HEADS = 4; N_LAYERS = 2; D_FF = 4 * D_MODEL
PRED_HIDDEN = D_FF; EMA_MOMENTUM = 0.99
CKPT_SEED = 42
E2E_LR = 1e-4; E2E_EPOCHS = 50; E2E_PATIENCE = 15
FAM_NEW_SEEDS = [3, 4]   # seeds not in Phase 9 yet


def load_v17():
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=PRED_HIDDEN,
    ).to(DEVICE)
    sd = torch.load(V17/'ckpts'/f'v17_seed{CKPT_SEED}_best.pt',
                    map_location=DEVICE, weights_only=False)
    model.load_state_dict(sd)
    return model


def run_fam_e2e(data, budget, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = load_v17(); probe = RULProbe(D_MODEL).to(DEVICE)
    sub = subsample_engines(data['train_engines'], budget, seed=seed)
    tr = DataLoader(CMAPSSFinetuneDataset(sub, n_cuts_per_engine=5, seed=seed),
                    batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], n_cuts_per_engine=10,
                                           seed=seed+111),
                    batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=16, shuffle=False, collate_fn=collate_test)
    for p in model.context_encoder.parameters(): p.requires_grad = True
    for p in model.predictor.parameters(): p.requires_grad = True
    params = (list(model.context_encoder.parameters())
              + list(model.predictor.parameters())
              + list(probe.parameters()))
    opt = torch.optim.Adam(params, lr=E2E_LR)
    best_val = float('inf'); best_pr = best_e = best_p = None; no_impr = 0
    for ep in range(E2E_EPOCHS):
        model.train(); probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            opt.zero_grad()
            h = model.encode_past(past, mask); pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        model.eval(); probe.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                h = model.encode_past(past, mask)
                pv.append(probe(h).cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_pr = copy.deepcopy(probe.state_dict())
            best_e = copy.deepcopy(model.context_encoder.state_dict())
            best_p = copy.deepcopy(model.predictor.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= E2E_PATIENCE: break
    probe.load_state_dict(best_pr)
    model.context_encoder.load_state_dict(best_e)
    model.predictor.load_state_dict(best_p)
    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy()*RUL_CAP); tt.append(rul_gt.numpy())
    return np.concatenate(pt), np.concatenate(tt)


def main():
    t0 = time.time()
    # Load STAR per-seed preds
    with open(STAR/'results'/'FD001_results.json') as f:
        star = json.load(f)
    star_preds = np.array([s['preds'] for s in star['per_seed']])  # (5, 100)
    star_trues = np.array(star['per_seed'][0]['trues'])            # (100,)
    print(f"STAR: {star_preds.shape}, true: {star_trues.shape}", flush=True)

    # Assemble FAM per-seed preds: seeds 0,1,2 from Phase 9 + seeds 3,4 new
    with open(V18/'phase9_nasa_score.json') as f:
        p9 = json.load(f)
    p9_100 = p9['results']['100%']['per_seed']
    fam_preds_old = [np.array(s['preds']) for s in p9_100 if s['seed'] in [0, 1, 2]]
    fam_trues_old = np.array(p9_100[0]['targets'])
    print(f"FAM existing: {len(fam_preds_old)} seeds", flush=True)

    # Sanity: FAM and STAR test engine targets should match
    if not np.allclose(star_trues, fam_trues_old, atol=0.5):
        print(f"  WARNING: STAR trues ({star_trues[:5]}) vs FAM trues "
              f"({fam_trues_old[:5]}) don't match exactly", flush=True)
        # Use STAR trues as canonical (they're from RUL_FD001.txt)

    data = load_cmapss_subset('FD001')
    print(f"Running 2 additional FAM E2E seeds (3, 4)...", flush=True)
    fam_preds_new = []
    for seed in FAM_NEW_SEEDS:
        p, t = run_fam_e2e(data, 1.0, seed)
        fam_preds_new.append(p)
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        print(f"  FAM seed={seed}: RMSE={rmse:.2f}", flush=True)

    fam_preds = np.array(fam_preds_old + fam_preds_new)  # (5, 100)
    print(f"FAM full: {fam_preds.shape}", flush=True)

    # True RUL (use STAR's - canonical from RUL_FD001.txt)
    trues = star_trues

    # Paired test: per-engine seed-averaged absolute residual
    fam_abs_res = np.abs(fam_preds.mean(axis=0) - trues)     # (100,)
    star_abs_res = np.abs(star_preds.mean(axis=0) - trues)   # (100,)
    delta = fam_abs_res - star_abs_res                       # (100,) per-engine

    # Paired t-test and Wilcoxon signed-rank
    t_stat, t_p = stats.ttest_rel(fam_abs_res, star_abs_res)
    w_stat, w_p = stats.wilcoxon(fam_abs_res, star_abs_res)
    # Bootstrap CI on mean delta
    rng = np.random.RandomState(42)
    diffs = np.array([rng.choice(delta, 100, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # Also per-seed comparison (both have 5 seeds, cross-seed avg residual)
    fam_rmse_per_seed = np.sqrt(((fam_preds - trues) ** 2).mean(axis=1))
    star_rmse_per_seed = np.sqrt(((star_preds - trues) ** 2).mean(axis=1))
    t_rmse, t_rmse_p = stats.ttest_ind(fam_rmse_per_seed, star_rmse_per_seed, equal_var=False)

    summary = {
        'config': 'v18_phase11_paired_star',
        'fam_seeds': [0, 1, 2] + FAM_NEW_SEEDS,
        'star_seeds': [42, 123, 456, 789, 1024],
        'fam_rmse_per_seed': fam_rmse_per_seed.tolist(),
        'star_rmse_per_seed': star_rmse_per_seed.tolist(),
        'fam_rmse_mean': float(fam_rmse_per_seed.mean()),
        'fam_rmse_std': float(fam_rmse_per_seed.std()),
        'star_rmse_mean': float(star_rmse_per_seed.mean()),
        'star_rmse_std': float(star_rmse_per_seed.std()),
        'unpaired_welch': {'t': float(t_rmse), 'p': float(t_rmse_p)},
        'paired_per_engine_abs_residual': {
            'fam_mean_abs_res': float(fam_abs_res.mean()),
            'star_mean_abs_res': float(star_abs_res.mean()),
            'delta_mean': float(delta.mean()),
            'paired_t': float(t_stat),
            'paired_t_p': float(t_p),
            'wilcoxon_W': float(w_stat),
            'wilcoxon_p': float(w_p),
            'bootstrap_95_ci': [float(lo), float(hi)],
        },
        'runtime_min': (time.time() - t0) / 60,
    }
    with open(V18/'phase11_paired_star.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 65)
    print("V18 Phase 11: Paired FAM vs STAR on FD001 test engines")
    print("=" * 65)
    print(f"FAM RMSE:  {summary['fam_rmse_mean']:.2f} +/- {summary['fam_rmse_std']:.2f}")
    print(f"STAR RMSE: {summary['star_rmse_mean']:.2f} +/- {summary['star_rmse_std']:.2f}")
    print(f"Welch's unpaired t: t={summary['unpaired_welch']['t']:+.2f}, "
          f"p={summary['unpaired_welch']['p']:.4f}")
    print(f"\nPaired per-engine |residual|:")
    pe = summary['paired_per_engine_abs_residual']
    print(f"  FAM mean |res|:   {pe['fam_mean_abs_res']:.2f}")
    print(f"  STAR mean |res|:  {pe['star_mean_abs_res']:.2f}")
    print(f"  Delta (FAM-STAR): {pe['delta_mean']:+.2f}")
    print(f"  Paired t:   t={pe['paired_t']:+.2f}, p={pe['paired_t_p']:.4f}")
    print(f"  Wilcoxon:   W={pe['wilcoxon_W']:.0f}, p={pe['wilcoxon_p']:.4f}")
    print(f"  Bootstrap 95% CI: [{pe['bootstrap_95_ci'][0]:+.2f}, "
          f"{pe['bootstrap_95_ci'][1]:+.2f}]")


if __name__ == '__main__':
    main()

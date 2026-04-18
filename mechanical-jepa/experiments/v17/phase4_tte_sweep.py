"""
V17 Phase 4: TTE via trajectory sweep.

Using best Phase 1 checkpoint (or Phase 3 if better) with encoder+predictor FROZEN.

Ground truth: on C-MAPSS FD001, pick sensor s14 (corrected fan speed).
  Baseline: first 50 cycles -> mu, sigma (per engine).
  Exceedance: sensor value outside mu +/- 3*sigma.
  TTE(t) = first exceedance after t  (NaN if never).

Event-boundary probe:
  Input  : gamma(k) = predictor(h_past, k), 256-dim (frozen)
  Output : p_k = sigmoid(linear(gamma(k))) = probability of crossing within k cycles
  Label  : y_k = 1 if TTE(t) <= k else 0
  Loss   : BCE, sample k from LogUniform[1, 150] during training

Inference:
  For each test sample at time t, sweep k in {1..150}; TTE_hat = min{k : p_k > 0.5}
  Compare to ground-truth TTE.

Metrics (primary): F1, precision, recall, AUC-PR on the binary "fail within 30" task.
Secondary: TTE RMSE.
"""

import sys, math, json, copy, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

V11 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V17 = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v17')
sys.path.insert(0, str(V11))
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/mechanical-jepa')

from models import TrajectoryJEPA
from data_utils import (load_cmapss_subset, N_SENSORS, RUL_CAP,
                         SELECTED_SENSORS)
from evaluation.grey_swan_metrics import (anomaly_metrics as _anomaly_metrics,
                                            tte_metrics, compute_tte_labels)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
D_FF = 4 * D_MODEL
EMA_MOMENTUM = 0.99

# s14 is at index 9 in SELECTED_SENSORS = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
S14_IDX = SELECTED_SENSORS.index(14)   # = 9
BASELINE_N = 50
N_SIGMA = 3.0
K_MAX = 150
K_EVAL_F1 = 30

PROBE_EPOCHS = 150
PROBE_LR = 1e-3
BATCH_SIZE = 128
SEEDS = [42, 123, 456]
CKPT_DIR = V17 / 'ckpts'
MIN_PAST = 15
N_CUTS_PER_ENGINE = 30


def compute_engine_tte(arr, s_idx=S14_IDX, baseline_n=BASELINE_N, n_sigma=N_SIGMA):
    """
    Per-engine TTE labels (length T). Uses compute_tte_labels on sensor s_idx.
    NaN if no exceedance, otherwise cycles until first exceedance.
    """
    T = len(arr)
    if T < baseline_n + 1:
        return np.full(T, np.nan)
    return compute_tte_labels(arr[:, s_idx], baseline_window=baseline_n,
                              n_sigma=n_sigma, method='first')


class TTEProbeDataset(Dataset):
    """
    (past_seq, k, y_k) tuples.
    y_k = 1 if engine has an exceedance within k cycles of t, else 0.
    Skip samples where TTE is undefined at time t (no exceedance in engine).
    """

    def __init__(self, engines, tte_labels_per_engine, K_max=K_MAX,
                 n_cuts=N_CUTS_PER_ENGINE, min_past=MIN_PAST, seed=42):
        rng = np.random.RandomState(seed)
        self.engines = engines
        self.K_max = K_max
        self.samples = []
        for eid, arr in engines.items():
            T = len(arr)
            tte = tte_labels_per_engine[eid]
            if tte is None:
                continue
            # Allowed t: min_past <= t <= T-1 (need some future)
            for _ in range(n_cuts):
                t_hi = T - 1
                if t_hi < min_past:
                    continue
                t = int(rng.randint(min_past, t_hi + 1))
                # Sample k LogUniform[1, K_max]
                k = int(math.exp(rng.uniform(0.0, math.log(K_max))))
                k = max(1, min(k, K_max))
                tte_t = tte[t - 1] if t - 1 < len(tte) else np.nan
                if np.isnan(tte_t):
                    # No exceedance ahead -> y_k = 0 only if within-window certain
                    # We include these as hard negatives (y=0)
                    y = 0.0
                else:
                    y = 1.0 if tte_t <= k else 0.0
                self.samples.append((eid, t, k, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eid, t, k, y = self.samples[idx]
        arr = self.engines[eid]
        past = torch.from_numpy(arr[:t]).float()
        return past, k, y


def collate_tte(batch):
    pasts, ks, ys = zip(*batch)
    T_max = max(p.shape[0] for p in pasts)
    B = len(pasts); S = pasts[0].shape[1]
    x_past = torch.zeros(B, T_max, S)
    past_mask = torch.ones(B, T_max, dtype=torch.bool)
    for i, p in enumerate(pasts):
        x_past[i, :p.shape[0]] = p
        past_mask[i, :p.shape[0]] = False
    k_t = torch.tensor(ks, dtype=torch.long)
    y_t = torch.tensor(ys, dtype=torch.float32)
    return x_past, past_mask, k_t, y_t


def load_model(seed, phase='phase1'):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, d_ff=D_FF,
        dropout=0.1, ema_momentum=EMA_MOMENTUM, predictor_hidden=D_FF,
    ).to(DEVICE)
    if phase == 'phase1':
        ck = CKPT_DIR / f'v17_seed{seed}_best.pt'
    else:
        ck = CKPT_DIR / f'v17_phase3_{phase}_seed{seed}_best.pt'
    if not ck.exists():
        return None
    model.load_state_dict(torch.load(ck, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def train_tte_probe(model, train_engines, val_engines, test_engines,
                     tte_train, tte_val, tte_test, seed=42):
    """Train boundary probe on gamma(k) -> p(TTE <= k). Returns test metrics."""
    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(D_MODEL, 1)).to(DEVICE)  # logits
    opt = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)

    tr_ds = TTEProbeDataset(train_engines, tte_train, seed=seed)
    va_ds = TTEProbeDataset(val_engines, tte_val, seed=seed + 1)
    print(f"    TTE probe train/val: {len(tr_ds)}/{len(va_ds)}", flush=True)
    tr = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_tte)
    va = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_tte)

    best_val_f1 = 0.0
    best_state = None
    no_impr = 0

    for ep in range(PROBE_EPOCHS):
        probe.train()
        for past, mask, k, y in tr:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            k, y = k.to(DEVICE), y.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
                g = model.predictor(h, k)  # (B, D)
            logit = probe(g).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, y)
            opt.zero_grad(); loss.backward(); opt.step()

        # Val metrics
        probe.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for past, mask, k, y in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                k = k.to(DEVICE)
                h = model.encode_past(past, mask)
                g = model.predictor(h, k)
                p = torch.sigmoid(probe(g).squeeze(-1)).cpu().numpy()
                all_p.append(p); all_y.append(y.numpy())
        all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)
        try:
            val_metrics = _anomaly_metrics(all_p, all_y.astype(int),
                                             threshold=0.5)
        except Exception:
            continue
        vf1 = val_metrics['f1_non_pa']
        if vf1 > best_val_f1:
            best_val_f1 = vf1
            best_state = copy.deepcopy(probe.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20:
                break

    if best_state is not None:
        probe.load_state_dict(best_state)

    # ---- Test-set TTE sweep ----
    probe.eval()
    # For each test engine, for the FULL sequence (last cycle), sweep k=1..K_MAX
    all_pred_tte = []   # predicted first k where p>0.5
    all_true_tte = []
    # Binary "within K_EVAL_F1 cycles" task
    all_bin_pred = []   # p_{k=K_EVAL_F1}
    all_bin_true = []

    with torch.no_grad():
        for eid, arr in test_engines.items():
            T = len(arr)
            if T < MIN_PAST:
                continue
            tte_eng = tte_test.get(eid)
            if tte_eng is None:
                continue
            # Last-cycle only (match CMAPSSTestDataset style)
            t = T
            past = torch.from_numpy(arr[:t]).float().unsqueeze(0).to(DEVICE)
            mask = torch.zeros(1, t, dtype=torch.bool, device=DEVICE)
            h = model.encode_past(past, mask)  # (1, D)

            # Sweep k
            ks = torch.arange(1, K_MAX + 1, dtype=torch.long, device=DEVICE)
            h_rep = h.expand(K_MAX, -1)
            g = model.predictor(h_rep, ks)  # (K, D)
            probs = torch.sigmoid(probe(g).squeeze(-1)).cpu().numpy()  # (K,)

            # TTE_hat = first k with p>0.5, else K_MAX+1
            fired = np.where(probs > 0.5)[0]
            pred_tte = float(fired[0] + 1) if len(fired) > 0 else float(K_MAX + 1)

            # Ground-truth TTE at last cycle
            true_tte = tte_eng[t - 1] if t - 1 < len(tte_eng) else np.nan

            all_pred_tte.append(pred_tte)
            all_true_tte.append(float(true_tte) if not np.isnan(true_tte) else np.nan)

            # Binary task: within K_EVAL_F1 cycles
            bin_pred = float(probs[K_EVAL_F1 - 1])  # score
            if np.isnan(true_tte):
                bin_true = 0
            else:
                bin_true = 1 if true_tte <= K_EVAL_F1 else 0
            all_bin_pred.append(bin_pred)
            all_bin_true.append(bin_true)

    all_pred_tte = np.array(all_pred_tte)
    all_true_tte = np.array(all_true_tte)
    all_bin_pred = np.array(all_bin_pred)
    all_bin_true = np.array(all_bin_true, dtype=int)

    # TTE RMSE on valid samples
    tte_m = tte_metrics(all_pred_tte, all_true_tte, max_tte=K_MAX)

    # Binary F1/AUC-PR
    try:
        thr = float(np.percentile(all_bin_pred[all_bin_true == 0], 95)) \
            if (all_bin_true == 0).sum() > 0 else 0.5
    except Exception:
        thr = 0.5
    m_bin = _anomaly_metrics(all_bin_pred, all_bin_true, threshold=thr)

    return {
        'seed': seed,
        'best_val_f1': float(best_val_f1),
        'tte_rmse': tte_m['rmse'],
        'tte_nrmse': tte_m['nrmse'],
        'tte_n_valid': tte_m['n_valid'],
        'bin_f1': m_bin['f1_non_pa'],
        'bin_auc_pr': m_bin['auc_pr'],
        'bin_precision': m_bin['precision_non_pa'],
        'bin_recall': m_bin['recall_non_pa'],
        'n_test': int(len(all_bin_true)),
        'n_pos': int(all_bin_true.sum()),
    }


def main():
    data = load_cmapss_subset('FD001')
    print(f"FD001 loaded", flush=True)

    # Compute per-engine TTE labels (s14, 3-sigma, baseline=50)
    tte_train = {eid: compute_engine_tte(arr)
                 for eid, arr in data['train_engines'].items()}
    tte_val = {eid: compute_engine_tte(arr)
               for eid, arr in data['val_engines'].items()}
    tte_test = {eid: compute_engine_tte(arr)
                for eid, arr in data['test_engines'].items()}

    # Summary of label distribution
    n_tr_has = sum(1 for t in tte_train.values()
                   if t is not None and np.isfinite(t).any())
    print(f"  train engines with s14 exceedance: "
          f"{n_tr_has}/{len(tte_train)}", flush=True)

    all_results = []
    t0 = time.time()
    for seed in SEEDS:
        model = load_model(seed, phase='phase1')
        if model is None:
            print(f"  [seed {seed}] phase1 ckpt missing, skipping", flush=True)
            continue
        print(f"\n=== seed {seed} ===", flush=True)
        r = train_tte_probe(model, data['train_engines'], data['val_engines'],
                             data['test_engines'], tte_train, tte_val, tte_test,
                             seed=seed)
        all_results.append(r)
        print(f"  seed {seed}: bin_F1={r['bin_f1']:.3f} "
              f"AUC-PR={r['bin_auc_pr']:.3f} "
              f"TTE_RMSE={r['tte_rmse']:.2f} "
              f"(n_valid={r['tte_n_valid']})", flush=True)
        del model
        torch.cuda.empty_cache()

    summary = {
        'config': 'v17_phase4_tte',
        'k_max': K_MAX, 'k_eval_f1': K_EVAL_F1,
        'sensor': 's14', 's14_idx': S14_IDX,
        'baseline_window': BASELINE_N, 'n_sigma': N_SIGMA,
        'seeds': SEEDS,
        'per_seed': all_results,
    }
    if all_results:
        summary['bin_f1_mean'] = float(np.mean([r['bin_f1'] for r in all_results]))
        summary['bin_auc_pr_mean'] = float(np.mean([r['bin_auc_pr'] for r in all_results]))
        summary['tte_rmse_mean'] = float(np.mean([r['tte_rmse'] for r in all_results
                                                    if r['tte_n_valid'] > 0]))
    summary['runtime_minutes'] = (time.time() - t0) / 60

    with open(V17 / 'phase4_tte_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n" + "=" * 60)
    print("V17 Phase 4 TTE SUMMARY")
    print("=" * 60)
    if all_results:
        print(f"Binary F1 (within {K_EVAL_F1} cycles): "
              f"{summary['bin_f1_mean']:.3f}")
        print(f"Binary AUC-PR                 : {summary['bin_auc_pr_mean']:.3f}")
        print(f"TTE RMSE                      : {summary['tte_rmse_mean']:.2f}")


if __name__ == '__main__':
    main()

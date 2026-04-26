"""
V14 Phase 2: Full-Sequence Prediction Experiment.

Hypothesis: Currently the target encoder sees ONLY x_{t+1:t+k}. If instead
the target encoder processes x_{1:t+k} (the whole trajectory up to t+k),
the target representation is richer and may produce more informative
gradients.

Objective:
  - Context encoder (causal): x_{1:t} -> h_past       [unchanged]
  - Target encoder (bidir): x_{1:t+k} -> h_full        [NEW]
  - Predictor: (h_past, k) -> h_full_hat               [unchanged]
  - Loss: L1(F.normalize(h_full_hat), F.normalize(h_full))

Everything else identical to V2 baseline. Evaluation: pretrain 150 epochs
(probe early-stop patience 10), frozen + E2E at 100% labels, 3 seeds.
Compare directly to V2 (frozen 17.81, E2E 14.23).

Kill criterion: if frozen RMSE > 18.5, revert. (documented, not automated)

Output: experiments/v14/full_sequence_prediction.json
        experiments/v14/best_pretrain_full_sequence.pt
"""

import os
import sys
import time
import json
import copy
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_finetune, collate_test,
)
from models import TrajectoryJEPA, RULProbe, trajectory_jepa_loss

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Custom dataset: returns (past, full_seq_upto_t_plus_k, k, t)
# ============================================================

class CMAPSSFullSequenceDataset(Dataset):
    """Like CMAPSSPretrainDataset, but returns x_{1:t+k} as the 'target sequence'
    instead of x_{t+1:t+k}. The context is still x_{1:t}."""

    def __init__(self, engines, n_cuts_per_engine=30, min_past=10,
                 min_horizon=5, max_horizon=30, seed=42):
        self.rng = np.random.default_rng(seed)
        self.items = []
        for eid, seq in engines.items():
            T = len(seq)
            for _ in range(n_cuts_per_engine):
                k = int(self.rng.integers(min_horizon, max_horizon + 1))
                t_min = min_past
                t_max = T - k
                if t_min > t_max:
                    continue
                t = int(self.rng.integers(t_min, t_max + 1))
                self.items.append((seq, t, k))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, t, k = self.items[idx]
        past = torch.from_numpy(seq[:t])          # (t, 14)
        full = torch.from_numpy(seq[:t + k])      # (t+k, 14)  <-- CHANGED
        return past, full, k, t


def collate_full_sequence(batch):
    past_list, full_list, k_list, t_list = zip(*batch)
    B = len(past_list)
    S = past_list[0].shape[1]

    max_p = max(p.shape[0] for p in past_list)
    past_padded = torch.zeros(B, max_p, S)
    past_mask = torch.zeros(B, max_p, dtype=torch.bool)
    for i, p in enumerate(past_list):
        past_padded[i, :p.shape[0]] = p
        past_mask[i, p.shape[0]:] = True

    max_f = max(f.shape[0] for f in full_list)
    full_padded = torch.zeros(B, max_f, S)
    full_mask = torch.zeros(B, max_f, dtype=torch.bool)
    for i, f in enumerate(full_list):
        full_padded[i, :f.shape[0]] = f
        full_mask[i, f.shape[0]:] = True

    return (past_padded, past_mask, full_padded, full_mask,
            torch.tensor(k_list, dtype=torch.long),
            torch.tensor(t_list, dtype=torch.long))


# ============================================================
# Probe-based validation (same as v11)
# ============================================================

def eval_probe_rmse(model, train_eng, val_eng, d_model=256, n_probe_epochs=50):
    model.eval()
    probe = RULProbe(d_model).to(DEVICE)
    optim_probe = torch.optim.Adam(probe.parameters(), lr=1e-3)

    tr_ds = CMAPSSFinetuneDataset(train_eng, n_cuts_per_engine=3)
    va_ds = CMAPSSFinetuneDataset(val_eng, use_last_only=False, n_cuts_per_engine=10)
    tr = DataLoader(tr_ds, batch_size=32, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=32, shuffle=False, collate_fn=collate_finetune)

    for _ in range(n_probe_epochs):
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            with torch.no_grad():
                h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            optim_probe.zero_grad(); loss.backward(); optim_probe.step()

    probe.eval()
    preds, targets = [], []
    with torch.no_grad():
        for past, mask, rul in va:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            preds.append(probe(h).cpu().numpy())
            targets.append(rul.numpy())
    preds = np.concatenate(preds) * RUL_CAP
    targets = np.concatenate(targets) * RUL_CAP
    return float(np.sqrt(np.mean((preds - targets) ** 2)))


# ============================================================
# Pretraining
# ============================================================

D_MODEL = 256
N_EPOCHS = 150
BATCH_SIZE = 4
N_CUTS = 30
LAMBDA_VAR = 0.01
PROBE_EVERY = 5
PATIENCE_PROBE = 10

def pretrain_full_sequence(data, ckpt_path, log_path, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.1,
        ema_momentum=0.99, predictor_hidden=256,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    history = {'loss': [], 'pred_loss': [], 'var_loss': [],
               'probe_rmse': [], 'probe_epochs': []}
    best_probe_rmse = float('inf')
    best_state = None
    no_improve = 0

    run = None
    if HAS_WANDB:
        try:
            run = wandb.init(
                project='industrialjepa',
                name=f'v14-phase2-fullseq-pretrain-s{seed}',
                tags=['v14-phase2-full-sequence'],
                config={'phase': '2', 'seed': seed, 'd_model': D_MODEL,
                        'n_epochs': N_EPOCHS, 'batch_size': BATCH_SIZE},
                reinit=True,
            )
        except Exception:
            pass

    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        ds = CMAPSSFullSequenceDataset(
            data['train_engines'], n_cuts_per_engine=N_CUTS,
            min_past=10, min_horizon=5, max_horizon=30, seed=epoch,
        )
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                            collate_fn=collate_full_sequence)

        model.train()
        total_loss = total_pred = total_var = 0.0
        n = 0
        for past, past_mask, full, full_mask, k, t_tensor in loader:
            past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
            full, full_mask = full.to(DEVICE), full_mask.to(DEVICE)
            k = k.to(DEVICE)
            optimizer.zero_grad()
            # h_past: causal over past
            h_past = model.context_encoder(past, past_mask)
            # h_full: EMA target encoder processes full sequence x_{1:t+k}
            with torch.no_grad():
                h_full = model.target_encoder(full, full_mask)
            # Predictor (unchanged): (h_past, k) -> h_full_hat
            pred = model.predictor(h_past, k)
            loss, pred_l, var_l = trajectory_jepa_loss(pred, h_full, LAMBDA_VAR)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.update_ema()

            B = past.shape[0]
            total_loss += loss.item() * B
            total_pred += pred_l.item() * B
            total_var += var_l.item() * B
            n += B

        history['loss'].append(total_loss / n)
        history['pred_loss'].append(total_pred / n)
        history['var_loss'].append(total_var / n)
        scheduler.step()

        msg_extra = ''
        if epoch % PROBE_EVERY == 0 or epoch == 1:
            probe_rmse = eval_probe_rmse(model, data['train_engines'], data['val_engines'])
            history['probe_rmse'].append(probe_rmse)
            history['probe_epochs'].append(epoch)
            if probe_rmse < best_probe_rmse:
                best_probe_rmse = probe_rmse
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, ckpt_path)
                no_improve = 0
            else:
                no_improve += 1
            msg_extra = f" | probe_RMSE={probe_rmse:.2f} (best={best_probe_rmse:.2f}, no_improve={no_improve})"

        if run is not None:
            try:
                log_dict = {'epoch': epoch, 'loss': total_loss / n,
                            'pred_loss': total_pred / n, 'var_loss': total_var / n}
                if epoch % PROBE_EVERY == 0 or epoch == 1:
                    log_dict['probe_rmse'] = probe_rmse
                    log_dict['best_probe_rmse'] = best_probe_rmse
                wandb.log(log_dict)
            except Exception:
                pass

        with open(log_path, 'a') as f:
            f.write(f"Ep {epoch:3d} | loss={total_loss/n:.4f}{msg_extra}\n")
        print(f"Ep {epoch:3d} | loss={total_loss/n:.4f}{msg_extra}", flush=True)

        if no_improve >= PATIENCE_PROBE:
            print(f"  Early stopping at epoch {epoch}", flush=True)
            break

    elapsed = (time.time() - t0) / 60
    print(f"\nPretraining complete in {elapsed:.1f} min, best probe RMSE={best_probe_rmse:.2f}",
          flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)

    if run is not None:
        try: wandb.finish()
        except Exception: pass

    return model, history, best_probe_rmse


# ============================================================
# Fine-tuning (frozen + E2E)
# ============================================================

def run_finetune(ckpt_path, data, mode, seed):
    model = TrajectoryJEPA(
        n_sensors=N_SENSORS, patch_length=1, d_model=D_MODEL, n_heads=4,
        n_layers=2, d_ff=512, dropout=0.1,
    ).to(DEVICE)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=DEVICE))

    probe = RULProbe(D_MODEL).to(DEVICE)
    torch.manual_seed(seed); np.random.seed(seed)

    tr_ds = CMAPSSFinetuneDataset(data['train_engines'], n_cuts_per_engine=5, seed=seed)
    va_ds = CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True)
    te_ds = CMAPSSTestDataset(data['test_engines'], data['test_rul'])
    tr = DataLoader(tr_ds, batch_size=16, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(va_ds, batch_size=16, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(te_ds, batch_size=16, shuffle=False, collate_fn=collate_test)

    if mode == 'frozen':
        for p in model.parameters(): p.requires_grad = False
        optim = torch.optim.Adam(probe.parameters(), lr=1e-3)
    else:
        for p in model.context_encoder.parameters(): p.requires_grad = True
        optim = torch.optim.Adam(
            list(model.context_encoder.parameters()) + list(probe.parameters()), lr=1e-4)

    best_val = float('inf'); best_ps = None; best_es = None; no_impr = 0
    for ep in range(100):
        if mode == 'frozen': model.eval()
        else: model.train()
        probe.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            optim.zero_grad()
            if mode == 'frozen':
                with torch.no_grad(): h = model.encode_past(past, mask)
            else:
                h = model.encode_past(past, mask)
            pred = probe(h)
            loss = F.mse_loss(pred, rul)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optim.step()

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
            best_ps = copy.deepcopy(probe.state_dict())
            if mode == 'e2e':
                best_es = copy.deepcopy(model.context_encoder.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 20: break

    probe.load_state_dict(best_ps)
    if mode == 'e2e' and best_es is not None:
        model.context_encoder.load_state_dict(best_es)

    model.eval(); probe.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            h = model.encode_past(past, mask)
            pt.append(probe(h).cpu().numpy() * RUL_CAP)
            tt.append(rul_gt.numpy())
    return float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2))), best_val


# ============================================================
# Main
# ============================================================

def main():
    V14_DIR.mkdir(exist_ok=True)
    ckpt_path = V14_DIR / 'best_pretrain_full_sequence.pt'
    log_path = V14_DIR / 'phase2_output.log'
    out_path = V14_DIR / 'full_sequence_prediction.json'

    with open(log_path, 'w') as f:
        f.write("V14 Phase 2: Full-sequence prediction\n")
        f.write(f"Device: {DEVICE}\n")

    print(f"V14 Phase 2: Full-sequence prediction experiment")
    print(f"Device: {DEVICE}")
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    print(f"Loaded FD001: {len(data['train_engines'])} train, "
          f"{len(data['val_engines'])} val, {len(data['test_engines'])} test")

    # ---------------- Pretrain (single seed) ----------------
    print("\n==== PRETRAINING (seed 42) ====")
    model, history, best_probe = pretrain_full_sequence(
        data, ckpt_path=ckpt_path, log_path=log_path, seed=42)

    # ---------------- Finetune (3 seeds, frozen + E2E) ----------------
    print("\n==== FINE-TUNING ====")
    seeds = [42, 123, 456]
    results = {'pretrain_best_probe': best_probe, 'pretrain_history': history,
               'seeds': seeds, 'frozen': [], 'e2e': []}
    for seed in seeds:
        for mode in ['frozen', 'e2e']:
            rmse, val = run_finetune(ckpt_path, data, mode, seed)
            print(f"  seed={seed} mode={mode:6s} | test RMSE={rmse:.3f} | val RMSE={val:.3f}",
                  flush=True)
            results[mode].append({'seed': seed, 'test_rmse': rmse, 'val_rmse': val})

    for mode in ['frozen', 'e2e']:
        vals = [r['test_rmse'] for r in results[mode]]
        results[f'{mode}_mean'] = float(np.mean(vals))
        results[f'{mode}_std'] = float(np.std(vals))

    results['wall_time_s'] = time.time() - t0

    # Baseline V2: frozen 17.81, E2E 14.23
    results['baseline_v2'] = {'frozen_mean': 17.81, 'frozen_std': 1.7,
                              'e2e_mean': 14.23, 'e2e_std': 0.39}
    delta_frozen = results['frozen_mean'] - 17.81
    delta_e2e = results['e2e_mean'] - 14.23
    results['delta_vs_v2'] = {'frozen': delta_frozen, 'e2e': delta_e2e}

    print(f"\nFrozen: {results['frozen_mean']:.3f} +/- {results['frozen_std']:.3f} "
          f"(V2: 17.81, delta={delta_frozen:+.3f})")
    print(f"E2E:    {results['e2e_mean']:.3f} +/- {results['e2e_std']:.3f} "
          f"(V2: 14.23, delta={delta_e2e:+.3f})")

    if results['frozen_mean'] > 18.5:
        verdict = 'KILL - frozen RMSE > 18.5, full-sequence prediction hurts'
    elif delta_frozen > 0.3:
        verdict = 'NEGATIVE - full-sequence does not improve frozen'
    elif delta_frozen < -0.3:
        verdict = 'POSITIVE - full-sequence improves frozen, keep'
    else:
        verdict = 'NEUTRAL - within noise of V2'
    results['verdict'] = verdict
    print(f"Verdict: {verdict}")

    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out_path}")
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()

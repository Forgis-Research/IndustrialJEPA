"""
V14 Phase 5b.3: AE-LSTM head-to-head replication.

Reconstruction autoencoder (FC 14->64->32->32->64->14) followed by a
1-layer LSTM regressor (hidden=64) on the latent sequence. Stage 1
pretrain (MSE reconstruction, no labels), Stage 2 joint fine-tune
(unfreeze AE encoder + LSTM head + linear, MSE on capped RUL).

Matched to our pipeline: same FD001 splits, same RUL cap (125),
same canonical last-window test protocol. 5 seeds at 100% labels.

Directly comparable to our Trajectory JEPA E2E (14.23 +/- 0.39).

Output: experiments/v14/aelstm_replication.json
"""

import sys, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

V11_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v11')
V14_DIR = Path('/home/sagemaker-user/IndustrialJEPA/mechanical-jepa/experiments/v14')
sys.path.insert(0, str(V11_DIR))

from data_utils import (
    load_cmapss_subset, N_SENSORS, RUL_CAP,
    CMAPSSPretrainDataset, CMAPSSFinetuneDataset, CMAPSSTestDataset,
    collate_pretrain, collate_finetune, collate_test,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Model
# ============================================================

class AEEncoder(nn.Module):
    def __init__(self, n_sensors=14, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, latent),
        )

    def forward(self, x):  # (B, T, 14) -> (B, T, latent)
        return self.net(x)


class AEDecoder(nn.Module):
    def __init__(self, n_sensors=14, hidden=64, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, n_sensors),
        )

    def forward(self, z):
        return self.net(z)


class AELSTMRegressor(nn.Module):
    def __init__(self, n_sensors=14, hidden=64, latent=32, lstm_hidden=64):
        super().__init__()
        self.encoder = AEEncoder(n_sensors, hidden, latent)
        self.decoder = AEDecoder(n_sensors, hidden, latent)
        self.lstm = nn.LSTM(latent, lstm_hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def reconstruct(self, x, mask=None):
        z = self.encoder(x)
        xhat = self.decoder(z)
        if mask is not None:
            # Zero out reconstruction loss at padding positions
            valid = (~mask).float().unsqueeze(-1)
            return F.mse_loss(xhat * valid, x * valid, reduction='sum') / valid.sum().clamp(min=1) / x.shape[-1]
        return F.mse_loss(xhat, x)

    def rul_predict(self, x, mask=None):
        z = self.encoder(x)                               # (B, T, latent)
        if mask is not None:
            lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
            z_p = nn.utils.rnn.pack_padded_sequence(
                z, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(z_p)
        else:
            _, (h_n, _) = self.lstm(z)
        h = h_n[-1]  # (B, lstm_hidden)
        return self.head(h).squeeze(-1)


# ============================================================
# Stage 1: reconstruction pretrain
# ============================================================

def pretrain_ae(model, train_engines, val_engines, n_epochs=80, lr=1e-3,
                 seed=42, batch_size=32):
    torch.manual_seed(seed); np.random.seed(seed)
    optim = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)

    # Reuse CMAPSSPretrainDataset's 'past' as the reconstruction input
    # (we ignore the 'future' part). Each epoch draws fresh cuts.
    best_val = float('inf'); best_state = None; no_impr = 0
    for epoch in range(1, n_epochs + 1):
        ds = CMAPSSPretrainDataset(train_engines, n_cuts_per_engine=10,
                                    min_past=10, min_horizon=5, max_horizon=30,
                                    seed=epoch)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_pretrain)
        model.train()
        for past, past_mask, _, _, _, _ in loader:
            past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
            loss = model.reconstruct(past, past_mask)
            optim.zero_grad(); loss.backward(); optim.step()

        # Val reconstruction
        model.eval()
        val_ds = CMAPSSPretrainDataset(val_engines, n_cuts_per_engine=3,
                                        min_past=10, min_horizon=5, max_horizon=30,
                                        seed=seed)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                collate_fn=collate_pretrain)
        with torch.no_grad():
            tot = 0.0; n = 0
            for past, past_mask, _, _, _, _ in val_loader:
                past, past_mask = past.to(DEVICE), past_mask.to(DEVICE)
                val_loss = model.reconstruct(past, past_mask).item()
                tot += val_loss; n += 1
            val_recon = tot / max(n, 1)

        if val_recon < best_val:
            best_val = val_recon
            best_state = copy.deepcopy(
                {'encoder': model.encoder.state_dict(),
                 'decoder': model.decoder.state_dict()})
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= 10:
                break
    if best_state is not None:
        model.encoder.load_state_dict(best_state['encoder'])
        model.decoder.load_state_dict(best_state['decoder'])
    return best_val


# ============================================================
# Stage 2: joint RUL fine-tune (encoder + LSTM + head)
# ============================================================

def finetune_rul(model, data, seed, n_epochs=100, patience=20, lr=1e-3,
                  batch_size=16):
    torch.manual_seed(seed); np.random.seed(seed)
    # All params trainable
    for p in model.parameters(): p.requires_grad = True
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    tr = DataLoader(CMAPSSFinetuneDataset(data['train_engines'],
                                           n_cuts_per_engine=5, seed=seed),
                    batch_size=batch_size, shuffle=True, collate_fn=collate_finetune)
    va = DataLoader(CMAPSSFinetuneDataset(data['val_engines'], use_last_only=True),
                    batch_size=batch_size, shuffle=False, collate_fn=collate_finetune)
    te = DataLoader(CMAPSSTestDataset(data['test_engines'], data['test_rul']),
                    batch_size=batch_size, shuffle=False, collate_fn=collate_test)

    best_val = float('inf'); best_state = None; no_impr = 0
    for ep in range(n_epochs):
        model.train()
        for past, mask, rul in tr:
            past, mask, rul = past.to(DEVICE), mask.to(DEVICE), rul.to(DEVICE)
            pred = model.rul_predict(past, mask)
            loss = F.mse_loss(pred, rul)
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        model.eval()
        pv, tv = [], []
        with torch.no_grad():
            for past, mask, rul in va:
                past, mask = past.to(DEVICE), mask.to(DEVICE)
                pred = model.rul_predict(past, mask)
                pv.append(pred.cpu().numpy()); tv.append(rul.numpy())
        val_rmse = float(np.sqrt(np.mean(
            (np.concatenate(pv)*RUL_CAP - np.concatenate(tv)*RUL_CAP)**2)))
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            no_impr = 0
        else:
            no_impr += 1
            if no_impr >= patience: break

    model.load_state_dict(best_state); model.eval()
    pt, tt = [], []
    with torch.no_grad():
        for past, mask, rul_gt in te:
            past, mask = past.to(DEVICE), mask.to(DEVICE)
            pred = model.rul_predict(past, mask)
            pt.append(pred.cpu().numpy() * RUL_CAP); tt.append(rul_gt.numpy())
    test_rmse = float(np.sqrt(np.mean((np.concatenate(pt) - np.concatenate(tt))**2)))
    return test_rmse, best_val


# ============================================================
# Main
# ============================================================

def main():
    print(f"V14 Phase 5b.3: AE-LSTM head-to-head replication")
    print(f"Device: {DEVICE}")
    t0 = time.time()

    data = load_cmapss_subset('FD001')
    seeds = [42, 123, 456, 789, 1024]

    results = {'seeds': seeds, 'per_seed': []}
    rmses = []
    for seed in seeds:
        print(f"\n--- seed={seed} ---")
        model = AELSTMRegressor(n_sensors=N_SENSORS, hidden=64, latent=32,
                                  lstm_hidden=64).to(DEVICE)
        t_s = time.time()
        best_recon = pretrain_ae(model, data['train_engines'], data['val_engines'],
                                   n_epochs=80, seed=seed)
        print(f"  Stage 1 AE pretrain: best val recon={best_recon:.4f} "
              f"({(time.time()-t_s)/60:.1f} min)")
        t_s = time.time()
        test_rmse, val_rmse = finetune_rul(model, data, seed)
        print(f"  Stage 2 joint fine-tune: test RMSE={test_rmse:.3f}, val={val_rmse:.3f} "
              f"({(time.time()-t_s)/60:.1f} min)")
        rmses.append(test_rmse)
        results['per_seed'].append({
            'seed': seed, 'test_rmse': test_rmse, 'val_rmse': val_rmse,
            'ae_val_recon': best_recon,
        })

    results['test_rmse_mean'] = float(np.mean(rmses))
    results['test_rmse_std'] = float(np.std(rmses))
    results['wall_time_s'] = time.time() - t0

    print(f"\n=== SUMMARY ===")
    print(f"AE-LSTM (ours replication): {results['test_rmse_mean']:.3f} +/- "
          f"{results['test_rmse_std']:.3f}")
    print(f"AE-LSTM (paper, LeCam 2025):  13.99  (best-of-28-configs, no variance)")
    print(f"Trajectory JEPA E2E (ours):   14.23 +/- 0.39  (5 seeds, fixed hyperparams)")
    print(f"Per-seed: {[f'{r:.3f}' for r in rmses]}")
    print(f"Wall time: {(time.time()-t0)/60:.1f} min")

    out = V14_DIR / 'aelstm_replication.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()

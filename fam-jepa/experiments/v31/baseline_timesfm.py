"""V31 TimesFM-1.3.0 baseline: frozen encoder + MLP head for event detection.

TimesFM-1.3.0 is Google's patched decoder model for time series forecasting.
We use it as a frozen feature extractor: extract the hidden state from the
last transformer block (the final patch representation) and train the same
198K-param dt-MLP head as used for MOMENT and Chr2 baselines.

Architecture: 32-layer transformer, hidden_size=1280, model_dims=1280 -> ~500M params.
Feature extraction: run per-channel (univariate), take last-patch hidden state (1280-d),
mean-pool across channels.

Protocol:
- Same 4 datasets as MOMENT baseline: FD001, FD003, MBA, BATADAL
- Same 198K-param dt-conditioned MLP head (same as Chr2MLP)
- Same training protocol: 100% labels, 3 seeds
- Comparison: FAM vs TimesFM vs MOMENT vs Chr2

IMPORTANT: Run with py310 conda env (timesfm requires Python <=3.11):
  conda run -n py310 python3 baseline_timesfm.py

Note: timesfm 1.3.0 does not expose an embedding API directly.
We extract hidden states by monkey-patching PatchedTimeSeriesDecoder.forward
to also return model_output (the transformer output before the projection head).
"""
from __future__ import annotations

import json
import sys
import time
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V31_DIR = FAM_DIR / 'experiments/v31'
V30_DIR = FAM_DIR / 'experiments/v30'
FEAT_DIR = V31_DIR / 'timesfm_features'
SURF_DIR = V31_DIR / 'surfaces'
RES_DIR = V31_DIR / 'results'
PNG_DIR = RES_DIR / 'surface_pngs'
for d in [FEAT_DIR, SURF_DIR, RES_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v24'))
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v11'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v27'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v28'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v29'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v30'))
sys.path.insert(0, str(FAM_DIR / 'experiments/v31'))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}", flush=True)

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False
    print("WARNING: wandb not found", flush=True)


# ---------------------------------------------------------------------------
# Data loading (reuse same protocol as MOMENT baseline)
# ---------------------------------------------------------------------------

def _build_anomaly(entity_list, stride=1):
    from train import EventDataset
    datasets = []
    for e in entity_list:
        X = np.array(e['test'], dtype=np.float32)
        y = np.array(e['labels'], dtype=np.int32)
        T = len(X)
        if T <= 128:
            continue
        d = EventDataset(X, y, max_context=512, stride=stride,
                         max_future=200, min_context=128)
        if len(d) > 0:
            datasets.append(d)
    return ConcatDataset(datasets)


def load_dataset(dataset: str):
    """Return (train_ds, val_ds, test_ds, n_channels, horizons)."""
    from train import EventDataset

    CMAPSS_HORIZONS = [1, 5, 10, 20, 50, 100, 150]
    ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]

    if dataset.startswith('FD'):
        from _cmapss_raw import load_cmapss_raw

        def build_cmapss(engines, stride):
            datasets = []
            for eid, seq in engines.items():
                T = len(seq)
                if T <= 128:
                    continue
                labels = np.zeros(T, dtype=np.int32)
                labels[T - 1] = 1
                d = EventDataset(seq, labels, max_context=512, stride=stride,
                                 max_future=200, min_context=128)
                if len(d) > 0:
                    datasets.append(d)
            return ConcatDataset(datasets)

        data = load_cmapss_raw(dataset)
        tr = build_cmapss(data['train_engines'], stride=4)
        va = build_cmapss(data['val_engines'], stride=4)
        te = build_cmapss(data['test_engines'], stride=1)
        return tr, va, te, 14, CMAPSS_HORIZONS

    if dataset == 'MBA':
        from data.mba import load_mba
        d = load_mba(normalize=False)
        test = np.asarray(d['test'], dtype=np.float32)
        labels = np.asarray(d['labels'], dtype=np.int32)
        T = len(test)
        t1, t2, gap = int(0.6 * T), int(0.7 * T), 200
        ent_tr = [{'test': test[:t1], 'labels': labels[:t1]}]
        ent_va = [{'test': test[t1 + gap:t2], 'labels': labels[t1 + gap:t2]}]
        ent_te = [{'test': test[t2 + gap:], 'labels': labels[t2 + gap:]}]
        n_ch = int(d.get('n_channels', test.shape[1]))
        return (_build_anomaly(ent_tr, 4), _build_anomaly(ent_va, 4),
                _build_anomaly(ent_te, 1), n_ch, ANOMALY_HORIZONS)

    if dataset == 'BATADAL':
        from data.batadal import load_batadal
        d = load_batadal(normalize=False)
        n_ch = d.get('n_channels', 43)
        ent_tr = d.get('ft_train', [])
        ent_va = d.get('ft_val', [])
        ent_te = d.get('ft_test', [])
        return _build_anomaly(ent_tr, 4), _build_anomaly(ent_va, 4), _build_anomaly(ent_te, 1), n_ch, ANOMALY_HORIZONS

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# TimesFM feature extractor
# ---------------------------------------------------------------------------

class TimesFMExtractor:
    """Extracts 1280-d embeddings from TimesFM-1.0-200M using hidden state hooking.

    Uses google/timesfm-1.0-200m-pytorch (203M params, 20 transformer layers).
    The 500M variant (google/timesfm-2.0-500m-pytorch) has 50 layers but
    timesfm 1.3.0 code only supports up to 20 layers - architecture mismatch.
    We use the 200M variant which loads cleanly.

    Embedding strategy: install a forward hook on stacked_transformer,
    run the model forward, capture transformer output (B, N_patches, 1280),
    mean-pool over patches to get (B, 1280) per time series.
    """

    MODEL_HF_ID = 'google/timesfm-1.0-200m-pytorch'
    CONTEXT_LEN = 512
    HIDDEN_SIZE = 1280  # TimesFM model_dims
    N_PARAMS_M = 203.6

    def __init__(self, device: str):
        import timesfm
        print("Loading TimesFM-1.0-200M model...", flush=True)

        hparams = timesfm.TimesFmHparams(
            backend='gpu' if 'cuda' in device else 'cpu',
            per_core_batch_size=32,
            horizon_len=128,
            context_len=self.CONTEXT_LEN,
        )
        checkpoint = timesfm.TimesFmCheckpoint(
            huggingface_repo_id=self.MODEL_HF_ID,
        )
        self.tfm = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        self.tfm.load_from_checkpoint(checkpoint=checkpoint)

        # Get the underlying PyTorch model
        self._model = self.tfm._model  # PatchedTimeSeriesDecoder

        # Count parameters
        n_params = sum(p.numel() for p in self._model.parameters()) / 1e6
        print(f"  TimesFM loaded: {n_params:.1f}M params ({len(list(self._model.stacked_transformer.layers))} layers)", flush=True)

        self.device = device
        self._hidden_cache = None

        # Install hook to capture hidden state from stacked_transformer
        self._install_hook()

    def _install_hook(self):
        """Install a forward hook on stacked_transformer to capture output."""
        def _hook(module, input, output):
            # output shape: (B, N_patches, hidden_size)
            self._hidden_cache = output.detach().clone()

        self._model.stacked_transformer.register_forward_hook(_hook)

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) - multivariate time series windows
        Returns: (B, 1280) - last-patch hidden state mean-pooled across channels
        """
        B, T, C = x.shape
        seq_len = self.CONTEXT_LEN

        # Pad or trim to context_len
        if T > seq_len:
            x = x[:, -seq_len:, :]
        elif T < seq_len:
            pad = torch.zeros(B, seq_len - T, C, dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=1)

        # Process each channel independently (TimesFM is univariate)
        # x: (B, seq_len, C) -> process as (B*C, seq_len) batch
        x_flat = x.permute(0, 2, 1).reshape(B * C, seq_len)  # (B*C, seq_len)
        x_np = x_flat.cpu().numpy()

        # Run forward to trigger hook (which captures stacked_transformer output)
        input_ts_list = [x_np[i] for i in range(x_np.shape[0])]
        freq_list = [0] * len(input_ts_list)  # high frequency

        # Use internal _preprocess_input + stacked_transformer directly
        # to capture hidden states without going through the full forecast
        hidden_states = self._embed_via_hook(x_flat)  # (B*C, 1280)

        # Reshape and mean-pool across channels
        hidden_states = hidden_states.reshape(B, C, -1).mean(dim=1)  # (B, 1280)
        return hidden_states

    @torch.no_grad()
    def _embed_via_hook(self, x_flat: torch.Tensor) -> torch.Tensor:
        """
        Run input through model up to stacked_transformer, capture hidden state.

        x_flat: (N, seq_len) tensor (may be on CPU)
        Returns: (N, hidden_size) tensor on CPU
        """
        N, T = x_flat.shape
        batch_size = 32
        all_hidden = []

        for i in range(0, N, batch_size):
            batch = x_flat[i:i + batch_size]  # (B_local, T)
            B_local = batch.shape[0]

            # Move to device
            t_input_ts = batch.float().to(self.device)
            t_input_padding = torch.zeros(B_local, T, device=self.device)
            t_freq = torch.zeros(B_local, 1, dtype=torch.long, device=self.device)

            # Run model forward (hook will capture hidden state from stacked_transformer)
            self._model(t_input_ts, t_input_padding, t_freq)

            # _hidden_cache shape: (B_local, N_patches, hidden_size)
            h = self._hidden_cache  # (B_local, N_patches, 1280)
            # Mean-pool over patches to get sequence-level representation
            h_pooled = h.mean(dim=1)  # (B_local, 1280)
            all_hidden.append(h_pooled.cpu())

        return torch.cat(all_hidden, dim=0)  # (N, 1280)


# ---------------------------------------------------------------------------
# MLP head (same 198K-param architecture as Chr2MLP and MOMENTHead)
# ---------------------------------------------------------------------------

class TimesFMHead(nn.Module):
    """198K-param dt-conditioned MLP head."""

    def __init__(self, d_input: int = 1280, d_hidden: int = 256, n_horizons: int = 7):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.dt_embed = nn.Embedding(256, d_hidden)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, feat: torch.Tensor, dt_idx: torch.Tensor) -> torch.Tensor:
        """feat: (B, d_input), dt_idx: (B,) -> (B, 1)."""
        h = self.proj(feat) + self.dt_embed(dt_idx.clamp(0, 255))
        return self.mlp(h)  # (B, 1)


# ---------------------------------------------------------------------------
# Feature extraction and caching
# ---------------------------------------------------------------------------

def extract_features(extractor: TimesFMExtractor,
                     loader: DataLoader,
                     feat_path: Path,
                     device: str) -> tuple:
    """Extract and cache TimesFM features for a DataLoader."""
    if feat_path.exists():
        print(f"  Loading cached features from {feat_path}", flush=True)
        cached = np.load(feat_path)
        return torch.from_numpy(cached['feats']), torch.from_numpy(cached['labels']), torch.from_numpy(cached['dt_indices'])

    print(f"  Extracting features...", flush=True)
    all_feats, all_labels, all_dts = [], [], []
    n_batches = len(loader)

    for bidx, batch in enumerate(loader):
        if (bidx + 1) % 50 == 0:
            print(f"    batch {bidx + 1}/{n_batches}", flush=True)
        x, y, dt_indices = batch[0], batch[1], batch[2]
        # x: (B, T, C), y: (B, K) event labels, dt_indices: (B, K)
        B, T, C = x.shape
        with torch.no_grad():
            feats = extractor.embed_batch(x.float())  # (B, 1280)
        all_feats.append(feats.cpu())
        all_labels.append(y.cpu())
        all_dts.append(dt_indices.cpu())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    dts = torch.cat(all_dts, dim=0)

    np.savez(feat_path,
             feats=feats.numpy(),
             labels=labels.numpy(),
             dt_indices=dts.numpy())
    print(f"  Saved features: {feats.shape}", flush=True)
    return feats, labels, dts


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_head(feat_train, label_train, dt_train,
               feat_val, label_val, dt_val,
               n_horizons: int, seed: int, run_name: str) -> TimesFMHead:
    """Train the 198K MLP head on frozen TimesFM features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_input = feat_train.shape[1]  # 1280
    head = TimesFMHead(d_input=d_input, d_hidden=256, n_horizons=n_horizons).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)

    # Build flat dataset: (feat, dt_idx, label)
    N_train = feat_train.shape[0]
    K = label_train.shape[1]

    def make_flat(feats, labels, dts):
        # feats: (N, D), labels: (N, K), dts: (N, K) -> K*N samples
        feats_exp = feats.unsqueeze(1).expand(-1, K, -1).reshape(-1, feats.shape[1])
        labels_flat = labels.reshape(-1).float()
        dts_flat = dts.reshape(-1).long()
        return feats_exp, labels_flat, dts_flat

    f_tr, l_tr, d_tr = make_flat(feat_train, label_train, dt_train)
    f_va, l_va, d_va = make_flat(feat_val, label_val, dt_val)

    BS = 512
    best_val_loss = float('inf')
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(50):
        head.train()
        perm = torch.randperm(len(f_tr))
        epoch_loss = 0.0
        n_steps = 0
        for s in range(0, len(f_tr), BS):
            idx = perm[s:s + BS]
            f_b = f_tr[idx].to(DEVICE)
            l_b = l_tr[idx].to(DEVICE)
            d_b = d_tr[idx].to(DEVICE)
            logits = head(f_b, d_b).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, l_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        head.eval()
        with torch.no_grad():
            val_logits = []
            val_labels = []
            for s in range(0, len(f_va), BS):
                f_b = f_va[s:s + BS].to(DEVICE)
                d_b = d_va[s:s + BS].to(DEVICE)
                l_b = l_va[s:s + BS]
                logits = head(f_b, d_b).squeeze(-1)
                val_logits.append(logits.cpu())
                val_labels.append(l_b)
            val_logits = torch.cat(val_logits)
            val_labels_cat = torch.cat(val_labels)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, val_labels_cat.float()).item()

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}: train_loss={epoch_loss/n_steps:.4f} val_loss={val_loss:.4f}", flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"    early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def evaluate_head(head: TimesFMHead,
                  feat_test, label_test, dt_test,
                  horizons: list) -> dict:
    """Compute per-horizon AUROC and mean h-AUROC."""
    from sklearn.metrics import roc_auc_score
    head.eval()
    N, K = label_test.shape
    BS = 512

    all_probs = np.zeros((N, K), dtype=np.float32)

    with torch.no_grad():
        for k in range(K):
            dt_k = dt_test[:, k]
            probs_k = []
            for s in range(0, N, BS):
                f_b = feat_test[s:s + BS].to(DEVICE)
                d_b = dt_k[s:s + BS].to(DEVICE)
                logits = head(f_b, d_b).squeeze(-1)
                probs_k.append(torch.sigmoid(logits).cpu().numpy())
            all_probs[:, k] = np.concatenate(probs_k)

    labels_np = label_test.numpy()
    aurocs = []
    for k in range(K):
        y = labels_np[:, k]
        p = all_probs[:, k]
        if y.sum() < 2 or (1 - y).sum() < 2:
            continue
        auroc = roc_auc_score(y, p)
        aurocs.append(auroc)

    return {
        'h_auroc_per_horizon': aurocs,
        'h_auroc_mean': float(np.mean(aurocs)) if aurocs else float('nan'),
        'n_valid_horizons': len(aurocs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_baseline(dataset: str, seeds: list, extractor: TimesFMExtractor):
    """Run TimesFM baseline on one dataset, multiple seeds."""
    print(f"\n{'='*60}", flush=True)
    print(f"Dataset: {dataset}", flush=True)

    # Load data
    from train import collate_event
    tr_ds, va_ds, te_ds, n_ch, horizons = load_dataset(dataset)
    K = len(horizons)
    print(f"  horizons: {horizons}", flush=True)
    print(f"  n_channels: {n_ch}", flush=True)

    # Build loaders
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=False, num_workers=0,
                           collate_fn=lambda b: collate_event(b, horizons))
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=0,
                           collate_fn=lambda b: collate_event(b, horizons))
    te_loader = DataLoader(te_ds, batch_size=64, shuffle=False, num_workers=0,
                           collate_fn=lambda b: collate_event(b, horizons))

    # Extract features (cached)
    feat_tr_path = FEAT_DIR / f"{dataset}_train.npz"
    feat_va_path = FEAT_DIR / f"{dataset}_val.npz"
    feat_te_path = FEAT_DIR / f"{dataset}_test.npz"

    feat_tr, lbl_tr, dt_tr = extract_features(extractor, tr_loader, feat_tr_path, DEVICE)
    feat_va, lbl_va, dt_va = extract_features(extractor, va_loader, feat_va_path, DEVICE)
    feat_te, lbl_te, dt_te = extract_features(extractor, te_loader, feat_te_path, DEVICE)

    seed_results = []
    for seed in seeds:
        print(f"\n  Seed {seed}:", flush=True)

        run_name = f"v31-timesfm-{dataset}-seed{seed}"
        if WANDB_OK:
            wandb.init(project='industrialjepa', name=run_name,
                       config={'dataset': dataset, 'seed': seed,
                               'model': 'TimesFM-1.3.0', 'head': 'dt-MLP-198K',
                               'n_channels': n_ch, 'horizons': horizons},
                       reinit=True, tags=['v31', 'timesfm', 'baseline'])

        head = train_head(feat_tr, lbl_tr, dt_tr,
                          feat_va, lbl_va, dt_va,
                          n_horizons=K, seed=seed, run_name=run_name)
        metrics = evaluate_head(head, feat_te, lbl_te, dt_te, horizons)
        print(f"    h-AUROC: {metrics['h_auroc_mean']:.4f} "
              f"(n_horizons={metrics['n_valid_horizons']})", flush=True)
        seed_results.append(metrics['h_auroc_mean'])

        if WANDB_OK:
            wandb.log({'h_auroc': metrics['h_auroc_mean']})
            wandb.finish()

    mean = float(np.mean(seed_results))
    std = float(np.std(seed_results))
    print(f"\n  {dataset} FINAL: h-AUROC = {mean:.4f} ± {std:.4f} "
          f"({len(seed_results)} seeds)", flush=True)
    print(f"  Per-seed: {[f'{v:.4f}' for v in seed_results]}", flush=True)

    return {
        'dataset': dataset,
        'model': 'TimesFM-1.3.0',
        'head': 'dt-MLP-198K',
        'horizons': horizons,
        'seeds': seeds,
        'per_seed_h_auroc': seed_results,
        'mean_h_auroc': mean,
        'std_h_auroc': std,
        'n_seeds': len(seeds),
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+',
                        default=['FD001', 'FD003', 'MBA', 'BATADAL'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    args = parser.parse_args()

    print("V31 TimesFM Baseline", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Seeds: {args.seeds}", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    # Load TimesFM
    extractor = TimesFMExtractor(device=DEVICE)

    results = {}
    for ds in args.datasets:
        try:
            r = run_baseline(ds, args.seeds, extractor)
            results[ds] = r
        except Exception as e:
            import traceback
            print(f"ERROR on {ds}: {e}", flush=True)
            traceback.print_exc()
            results[ds] = {'error': str(e)}

    # Save results
    out_path = RES_DIR / 'timesfm_baseline.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY: TimesFM-1.3.0 h-AUROC")
    for ds, r in results.items():
        if 'error' in r:
            print(f"  {ds}: ERROR - {r['error']}")
        else:
            print(f"  {ds}: {r['mean_h_auroc']:.4f} ± {r['std_h_auroc']:.4f}")

"""V31 MOMENT-1-large baseline: frozen encoder + MLP head for event detection.

MOMENT is a 341M-param foundation model trained on a large corpus of time series.
This baseline is the fairest comparison to Chronos-2 (120M, value-forecasting).

Protocol:
- MOMENT processes univariate channels independently (seq_len=512)
- For multivariate data: run MOMENT per channel, mean-pool across channels
  -> 1024-d embedding per observation window
- Fit the same Chr2MLP head as used in Chr2 baseline (198K params)
- Report h-AUROC on the same test set

Datasets: FD001, FD003, MBA, BATADAL (same as Chr2 comparison)
Seeds: 42, 123, 456

IMPORTANT: Run this script with the py310 conda env:
  conda run -n py310 python3 baseline_moment.py

Note: MOMENT requires Python <=3.11 (momentfm package).
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
FEAT_DIR = V31_DIR / 'moment_features'
SURF_DIR = V31_DIR / 'surfaces'
RES_DIR = V31_DIR / 'results'
PNG_DIR = RES_DIR / 'surface_pngs'
for d in [FEAT_DIR, SURF_DIR, RES_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v24'))  # _cmapss_raw
sys.path.insert(0, str(FAM_DIR / 'experiments/archive/v11'))  # data_utils
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
# Data loading (reuse same protocol as Chr2 baseline)
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
        tr = _build_anomaly(d.get('ft_train', [d.get('train', [])]), stride=4)
        va = _build_anomaly(d.get('ft_val', []), stride=4)
        te = _build_anomaly(d.get('ft_test', [d.get('test', [])]), stride=1)
        n_ch = d.get('n_channels', 2)
        return tr, va, te, n_ch, ANOMALY_HORIZONS

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
# MOMENT feature extractor
# ---------------------------------------------------------------------------

class MOMENTExtractor:
    """Extracts fixed-size 1024-d embeddings from MOMENT-1-large."""

    def __init__(self, device):
        from momentfm import MOMENTPipeline
        print("Loading MOMENT-1-large...", flush=True)
        self.model = MOMENTPipeline.from_pretrained(
            'AutonLab/MOMENT-1-large',
            model_kwargs={'task_name': 'embedding'}
        )
        self.model.eval()
        self.model.to(device)
        self.device = device
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"  MOMENT loaded: {n_params:.1f}M params, seq_len={self.model.seq_len}", flush=True)

    @torch.no_grad()
    def embed_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, C) - multivariate time series windows
        Returns: (B, 1024) - mean-pooled embeddings across channels
        """
        B, T, C = x.shape
        # MOMENT takes (B, 1, seq_len) for univariate - process each channel
        # Pad or trim to seq_len=512
        seq_len = self.model.seq_len  # 512
        if T > seq_len:
            x = x[:, -seq_len:, :]  # use last seq_len steps
        elif T < seq_len:
            # pad left with zeros
            pad = torch.zeros(B, seq_len - T, C, dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=1)

        # Process all channels together as a batch of univariate series
        # (B*C, 1, seq_len)
        x_flat = x.permute(0, 2, 1).reshape(B * C, 1, seq_len)
        out = self.model.embed(x_enc=x_flat, reduction='mean')
        embeddings = out.embeddings  # (B*C, 1024)

        # Reshape and mean-pool across channels
        embeddings = embeddings.reshape(B, C, -1).mean(dim=1)  # (B, 1024)
        return embeddings


# ---------------------------------------------------------------------------
# Chr2MLP head (same as v30 baseline for fair comparison)
# ---------------------------------------------------------------------------

class MOMENTHead(nn.Module):
    """198K-param dt-conditioned MLP head (same architecture as Chr2MLP)."""

    def __init__(self, d_input: int = 1024, d_hidden: int = 256, n_horizons: int = 7):
        super().__init__()
        self.proj = nn.Linear(d_input, d_hidden)
        self.dt_embed = nn.Embedding(256, d_hidden)  # max 256 horizons
        self.mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_horizons),
        )

    def forward(self, x: torch.Tensor, dt_indices: torch.Tensor) -> torch.Tensor:
        """
        x: (B, d_input)
        dt_indices: (B, K) - horizon indices (integers 0..K-1)
        Returns: (B, K) logits
        """
        h = self.proj(x)  # (B, d_hidden)
        # For each horizon: add dt embedding and predict
        K = dt_indices.shape[1]
        out = []
        for k in range(K):
            dt_k = dt_indices[:, k]  # (B,)
            h_k = h + self.dt_embed(dt_k)
            logits_k = self.mlp(h_k)  # (B, K) - we pick k-th output
            out.append(logits_k[:, k:k+1])  # (B, 1)
        return torch.cat(out, dim=1)  # (B, K)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def extract_features(extractor, ds, batch_size=64, device=DEVICE):
    """Extract MOMENT features from a dataset.

    collate_event returns (ctx_padded, ctx_mask, ttes, ts) - a tuple of 4 elements.
    ctx_padded: (B, T, C), ttes: (B,) time-to-event.
    """
    from train import collate_event

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_event, num_workers=0)

    all_feats, all_tte = [], []
    for batch in loader:
        x, ctx_mask, ttes, ts = batch  # unpack tuple
        x = x.to(device)  # (B, T, C)
        feats = extractor.embed_batch(x)  # (B, 1024)
        all_feats.append(feats.cpu())
        all_tte.append(ttes)

    return torch.cat(all_feats, dim=0), torch.cat(all_tte, dim=0)


def train_moment_head(X_tr, y_tr, X_va, y_va, horizons, seed=42, max_epochs=100):
    """Train MOMENT head on extracted features."""
    from evaluation.losses import build_label_surface

    torch.manual_seed(seed)
    np.random.seed(seed)

    h_t = torch.tensor(horizons, dtype=torch.float32)
    K = len(horizons)

    # Build label surfaces. build_label_surface returns (N, 1, K); squeeze to (N, K).
    y_tr_surf = build_label_surface(y_tr.unsqueeze(1), h_t).squeeze(1).to(DEVICE)
    y_va_surf = build_label_surface(y_va.unsqueeze(1), h_t).squeeze(1).to(DEVICE)
    X_tr = X_tr.to(DEVICE)
    X_va = X_va.to(DEVICE)

    # Compute pos_weight
    n_pos = y_tr_surf.sum().item()
    n_tot = y_tr_surf.numel()
    pw = torch.tensor(max(1.0, min(1000.0, (n_tot - n_pos) / max(n_pos, 1))),
                      device=DEVICE)

    head = MOMENTHead(d_input=X_tr.shape[1], d_hidden=256, n_horizons=K).to(DEVICE)
    print(f"  MOMENT head: {head.n_params():,} params", flush=True)

    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)

    B = 2048
    dt_idx = torch.arange(K, device=DEVICE).unsqueeze(0)  # (1, K)

    best_state, best_val, wait = None, float('inf'), 0
    for ep in range(max_epochs):
        head.train()
        perm = torch.randperm(len(X_tr), device=DEVICE)
        losses = []
        for i in range(0, len(X_tr), B):
            idx = perm[i:i+B]
            x = X_tr[idx]
            y = y_tr_surf[idx]
            dt_idx_b = dt_idx.expand(x.shape[0], -1)
            logits = head(x, dt_idx_b)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sch.step()

        head.eval()
        with torch.no_grad():
            dt_idx_va = dt_idx.expand(X_va.shape[0], -1)
            logits_va = head(X_va, dt_idx_va)
            vl = F.binary_cross_entropy_with_logits(logits_va, y_va_surf, pos_weight=pw).item()

        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(head.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= 8:
                break

    head.load_state_dict(best_state)
    return head


def run_moment_baseline(dataset: str, seed: int, extractor, results_list: list):
    """Run full MOMENT baseline for one dataset/seed."""
    from evaluation.losses import build_label_surface
    from evaluation.surface_metrics import evaluate_probability_surface

    tag = f"{dataset}_moment-mlp_s{seed}"
    print(f"\n[{tag}] Starting...", flush=True)
    t0 = time.time()

    # Load data
    tr_ds, va_ds, te_ds, n_channels, horizons = load_dataset(dataset)
    K = len(horizons)
    print(f"  n_channels={n_channels}, K={K}, horizons={horizons}", flush=True)
    print(f"  tr={len(tr_ds)}, va={len(va_ds)}, te={len(te_ds)}", flush=True)

    # Extract features (may be cached)
    feat_path = FEAT_DIR / f"{dataset}_s{seed}_moment.pt"
    if feat_path.exists():
        print(f"  Loading cached features: {feat_path}", flush=True)
        cache = torch.load(feat_path, map_location='cpu')
        X_tr, y_tr = cache['tr']
        X_va, y_va = cache['va']
        X_te, y_te = cache['te']
    else:
        print(f"  Extracting features...", flush=True)
        X_tr, y_tr = extract_features(extractor, tr_ds)
        X_va, y_va = extract_features(extractor, va_ds)
        X_te, y_te = extract_features(extractor, te_ds)
        torch.save({'tr': (X_tr, y_tr), 'va': (X_va, y_va), 'te': (X_te, y_te)},
                   feat_path)
        print(f"  Saved features: X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}", flush=True)

    # Train head
    torch.manual_seed(seed)
    head = train_moment_head(X_tr, y_tr, X_va, y_va, horizons, seed=seed)

    # Evaluate
    head.eval()
    h_t = torch.tensor(horizons, dtype=torch.float32)
    # build_label_surface returns (N, 1, K); squeeze to (N, K)
    y_te_surf = build_label_surface(y_te.unsqueeze(1), h_t).squeeze(1).numpy().astype(np.int32)

    X_te_dev = X_te.to(DEVICE)
    dt_idx = torch.arange(K, device=DEVICE).unsqueeze(0).expand(X_te.shape[0], -1)
    with torch.no_grad():
        p_te = torch.sigmoid(head(X_te_dev, dt_idx)).cpu().numpy()

    # h-AUROC
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for k in range(K):
        y_k = y_te_surf[:, k]
        p_k = p_te[:, k]
        if y_k.sum() > 0 and y_k.sum() < len(y_k):
            aurocs.append(float(roc_auc_score(y_k, p_k)))
    mean_h_auroc = float(np.mean(aurocs)) if aurocs else float('nan')
    base = 0.5  # chance

    # AUPRC
    from sklearn.metrics import average_precision_score
    y_flat = y_te_surf.ravel()
    p_flat = p_te.ravel()
    auprc = float(average_precision_score(y_flat, p_flat)) if y_flat.sum() > 0 else float('nan')

    elapsed = time.time() - t0
    print(f"[{tag}] mean h-AUROC={mean_h_auroc:.4f} (base={base:.4f}, Δ={mean_h_auroc-base:+.4f})",
          flush=True)
    print(f"  pooled AUPRC={auprc:.4f}, elapsed={elapsed:.1f}s", flush=True)

    # Save surface
    np.savez(SURF_DIR / f"{tag}.npz",
             p_surface=p_te.astype(np.float32),
             y_surface=y_te_surf.astype(np.int8),
             horizons=np.array(horizons, dtype=np.int32))

    result = {
        'tag': tag,
        'dataset': dataset,
        'seed': seed,
        'n_channels': n_channels,
        'n_params_encoder': 341_200_000,
        'n_params_head': head.n_params(),
        'mean_h_auroc': mean_h_auroc,
        'mean_h_auroc_base': base,
        'pooled_auprc': auprc,
        'elapsed_s': elapsed,
        'horizons': horizons,
    }
    results_list.append(result)

    # Log to wandb
    if WANDB_OK:
        try:
            wandb.init(
                project='industrialjepa',
                name=f'v31-moment-mlp-{dataset}-s{seed}',
                config={
                    'experiment': 'v31_moment_baseline',
                    'dataset': dataset,
                    'seed': seed,
                    'model': 'MOMENT-1-large',
                    'head': 'MLP-198K',
                    'n_encoder_params': 341_200_000,
                },
                tags=['v31', 'moment-baseline', dataset],
                reinit=True,
            )
            wandb.log({
                'eval/mean_h_auroc': mean_h_auroc,
                'eval/pooled_auprc': auprc,
                'eval/delta_over_base': mean_h_auroc - base,
            })
            wandb.finish()
        except Exception as e:
            print(f"  wandb error: {e}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    datasets = ['FD001', 'FD003', 'MBA', 'BATADAL']
    seeds = [42, 123, 456]

    print("=" * 60, flush=True)
    print("V31 MOMENT-1-large Baseline", flush=True)
    print(f"Datasets: {datasets}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print("=" * 60, flush=True)

    # Load MOMENT once
    extractor = MOMENTExtractor(DEVICE)

    all_results = []
    for dataset in datasets:
        ds_results = []
        for seed in seeds:
            try:
                r = run_moment_baseline(dataset, seed, extractor, ds_results)
            except Exception as e:
                print(f"ERROR [{dataset} s{seed}]: {e}", flush=True)
                import traceback
                traceback.print_exc()

        all_results.extend(ds_results)

        # Per-dataset summary
        if ds_results:
            means = [r['mean_h_auroc'] for r in ds_results]
            print(f"\n[{dataset}] h-AUROC: {np.mean(means):.4f} +/- {np.std(means, ddof=0):.4f} "
                  f"(seeds: {[f'{m:.4f}' for m in means]})", flush=True)

    # Save results
    out_path = RES_DIR / 'moment_baseline.json'
    with open(out_path, 'w') as f:
        json.dump({'results': all_results, 'model': 'MOMENT-1-large',
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, f, indent=2)
    print(f"\nResults saved: {out_path}", flush=True)

    # Summary table
    print("\n=== MOMENT-1-large Summary (h-AUROC) ===", flush=True)
    by_ds = {}
    for r in all_results:
        ds = r['dataset']
        by_ds.setdefault(ds, []).append(r['mean_h_auroc'])

    for ds, scores in by_ds.items():
        print(f"  {ds:10s}: {np.mean(scores):.4f} +/- {np.std(scores, ddof=0):.4f}", flush=True)


if __name__ == '__main__':
    main()

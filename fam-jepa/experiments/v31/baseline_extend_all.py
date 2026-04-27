"""V31 Baseline Extension: MOMENT, TimesFM-1.0, Moirai across all 11 datasets.

Extends the 4-dataset baselines to the full 11-dataset benchmark:
  FD001, FD002, FD003, SMAP, PSM, MBA, GECCO, BATADAL, SKAB, ETTm1, SMD

Runs in append-only mode: loads existing JSON, skips already-done
(dataset, seed, label_fraction) tuples, saves merged JSON.

Usage:
  conda run -n py310 python3 baseline_extend_all.py --model moment [--datasets ...]
  conda run -n py310 python3 baseline_extend_all.py --model timesfm [--datasets ...]
  conda run -n py310 python3 baseline_extend_all.py --model moirai [--datasets ...]

Run with py310 conda env (requires momentfm, timesfm, uni2ts packages).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
V31_DIR = FAM_DIR / 'experiments/v31'

MOMENT_FEAT_DIR = V31_DIR / 'moment_features'
TFM_FEAT_DIR = V31_DIR / 'timesfm_features'
MOIRAI_FEAT_DIR = V31_DIR / 'moirai_features'
SURF_DIR = V31_DIR / 'surfaces'
RES_DIR = V31_DIR / 'results'
PNG_DIR = RES_DIR / 'surface_pngs'

for d in [MOMENT_FEAT_DIR, TFM_FEAT_DIR, MOIRAI_FEAT_DIR, SURF_DIR, RES_DIR, PNG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Set up sys.path
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

CMAPSS_HORIZONS = [1, 5, 10, 20, 50, 100, 150]
ANOMALY_HORIZONS = [1, 5, 10, 20, 50, 100, 150, 200]

ALL_11_DATASETS = ['FD001', 'FD002', 'FD003', 'SMAP', 'PSM', 'MBA',
                   'GECCO', 'BATADAL', 'SKAB', 'ETTm1', 'SMD']

# Datasets already covered in original 4-dataset baseline
ALREADY_COVERED_MOMENT = {'FD001', 'FD003', 'MBA', 'BATADAL'}
ALREADY_COVERED_TFM = {'FD001', 'FD003', 'MBA', 'BATADAL'}
ALREADY_COVERED_MOIRAI = {'FD001', 'FD003', 'MBA', 'BATADAL'}


# ---------------------------------------------------------------------------
# Unified data loading using v27/v29 bundle format
# ---------------------------------------------------------------------------

def load_bundle(dataset: str) -> dict:
    """Load dataset into standard bundle format via v27/v29 LOADERS."""
    from _runner_v29 import LOADERS
    loader_fn = LOADERS.get(dataset)
    if loader_fn is None:
        raise ValueError(f"No loader for dataset: {dataset}")
    return loader_fn()


def bundle_to_datasets(bundle: dict, dataset: str, stride_train: int = 4):
    """Convert bundle to (train_ds, val_ds, test_ds, n_channels, horizons)."""
    from train import EventDataset

    horizons = bundle['horizons']
    n_channels = bundle['n_channels']

    def build_ds(entities, stride=1):
        dsets = []
        for e in entities:
            X = np.array(e['test'], dtype=np.float32)
            y = np.array(e['labels'], dtype=np.int32)
            if len(X) <= 128:
                continue
            d = EventDataset(X, y, max_context=512, stride=stride,
                             max_future=200, min_context=128)
            if len(d) > 0:
                dsets.append(d)
        if not dsets:
            return ConcatDataset([EventDataset(
                np.zeros((200, n_channels), dtype=np.float32),
                np.zeros(200, dtype=np.int32),
                max_context=512, stride=stride, max_future=200, min_context=128
            )])
        return ConcatDataset(dsets)

    tr = build_ds(bundle['ft_train'], stride=stride_train)
    va = build_ds(bundle['ft_val'], stride=4)
    te = build_ds(bundle['ft_test'], stride=1)

    return tr, va, te, n_channels, horizons


# ---------------------------------------------------------------------------
# MLP head (shared architecture for all baselines: 198K params)
# ---------------------------------------------------------------------------

class UnifiedMLPHead(nn.Module):
    """~198K-param dt-conditioned MLP head (same for MOMENT, TimesFM, Moirai)."""

    def __init__(self, d_input: int, d_hidden: int = 256, n_horizons: int = 7):
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

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Feature extraction (common pattern)
# ---------------------------------------------------------------------------

def extract_features_to_flat(extractor_fn, ds, horizons, batch_size=64):
    """
    Extract features and labels from a dataset.

    Returns (feats, labels_flat, dt_flat) where:
      feats: (N, D)
      labels_flat: (N*K,) binary event labels
      dt_flat: (N*K,) horizon indices 0..K-1
    """
    from train import collate_event
    from evaluation.losses import build_label_surface

    K = len(horizons)
    h_t = torch.tensor(horizons, dtype=torch.float32)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_event, num_workers=0)

    all_feats, all_labels, all_dts = [], [], []
    for batch in loader:
        x, ctx_mask, ttes, ts = batch  # (B,T,C), (B,T), (B,), (B,)
        B = x.shape[0]

        with torch.no_grad():
            feats = extractor_fn(x.float())  # (B, D)
        all_feats.append(feats.cpu())

        # Build label surface: (B, 1, K) -> (B, K)
        y_surf = build_label_surface(ttes.unsqueeze(1), h_t).squeeze(1)  # (B, K)
        all_labels.append(y_surf)

        # dt indices: for each sample, K horizon indices 0..K-1
        dt_idx = torch.arange(K).unsqueeze(0).expand(B, -1)  # (B, K)
        all_dts.append(dt_idx)

    feats = torch.cat(all_feats, dim=0)  # (N, D)
    labels_surf = torch.cat(all_labels, dim=0)  # (N, K)
    dt_surf = torch.cat(all_dts, dim=0)  # (N, K)

    # Flatten for head training
    N = feats.shape[0]
    feats_exp = feats.unsqueeze(1).expand(-1, K, -1).reshape(N * K, -1)  # (N*K, D)
    labels_flat = labels_surf.reshape(-1).float()  # (N*K,)
    dt_flat = dt_surf.reshape(-1).long()  # (N*K,)

    return feats, feats_exp, labels_surf, labels_flat, dt_flat


def train_head_on_feats(feat_train_flat, label_train_flat, dt_train_flat,
                         feat_val_flat, label_val_flat, dt_val_flat,
                         d_input: int, K: int, seed: int, run_name: str = ''):
    """Train MLP head on pre-extracted flat features."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = UnifiedMLPHead(d_input=d_input, d_hidden=256, n_horizons=K).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)

    BS = 512
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    PATIENCE = 8

    for epoch in range(80):
        head.train()
        perm = torch.randperm(len(feat_train_flat))
        epoch_loss = 0.0
        n_steps = 0
        for s in range(0, len(feat_train_flat), BS):
            idx = perm[s:s + BS]
            f_b = feat_train_flat[idx].to(DEVICE)
            l_b = label_train_flat[idx].to(DEVICE)
            d_b = dt_train_flat[idx].to(DEVICE)
            logits = head(f_b, d_b).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, l_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_steps += 1

        head.eval()
        with torch.no_grad():
            val_logits, val_labs = [], []
            for s in range(0, len(feat_val_flat), BS):
                f_b = feat_val_flat[s:s + BS].to(DEVICE)
                d_b = dt_val_flat[s:s + BS].to(DEVICE)
                logits = head(f_b, d_b).squeeze(-1)
                val_logits.append(logits.cpu())
                val_labs.append(label_val_flat[s:s + BS])
            val_logits = torch.cat(val_logits)
            val_labs_cat = torch.cat(val_labs)
            val_loss = F.binary_cross_entropy_with_logits(
                val_logits, val_labs_cat.float()).item()

        if epoch % 10 == 0 or epoch < 3:
            print(f"    ep {epoch+1}: train={epoch_loss/n_steps:.4f} val={val_loss:.4f}",
                  flush=True)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = copy.deepcopy(head.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"    early stop at epoch {epoch + 1}", flush=True)
            break

    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def evaluate_head_on_feats(head, feat_test, label_test_surf, K: int, horizons: list):
    """Evaluate trained head. Returns metrics dict."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    head.eval()

    N = feat_test.shape[0]
    all_probs = np.zeros((N, K), dtype=np.float32)
    BS = 512

    with torch.no_grad():
        for k in range(K):
            dt_k = torch.full((N,), k, dtype=torch.long)
            probs_k = []
            for s in range(0, N, BS):
                f_b = feat_test[s:s + BS].to(DEVICE)
                d_b = dt_k[s:s + BS].to(DEVICE)
                logits = head(f_b, d_b).squeeze(-1)
                probs_k.append(torch.sigmoid(logits).cpu().numpy())
            all_probs[:, k] = np.concatenate(probs_k)

    y_np = label_test_surf.numpy().astype(int)
    aurocs = []
    for k in range(K):
        y = y_np[:, k]
        p = all_probs[:, k]
        if y.sum() >= 2 and (1 - y).sum() >= 2:
            aurocs.append(float(roc_auc_score(y, p)))

    mean_h_auroc = float(np.mean(aurocs)) if aurocs else float('nan')

    # Pooled AUPRC
    y_flat = y_np.ravel()
    p_flat = all_probs.ravel()
    pooled_auprc = float(average_precision_score(y_flat, p_flat)) if y_flat.sum() > 0 else float('nan')

    return {
        'mean_h_auroc': mean_h_auroc,
        'pooled_auprc': pooled_auprc,
        'n_valid_horizons': len(aurocs),
        'p_surface': all_probs,
        'y_surface': y_np.astype(np.int8),
    }


# ---------------------------------------------------------------------------
# MOMENT extractor
# ---------------------------------------------------------------------------

def build_moment_extractor():
    from momentfm import MOMENTPipeline
    print("Loading MOMENT-1-large...", flush=True)
    model = MOMENTPipeline.from_pretrained(
        'AutonLab/MOMENT-1-large',
        model_kwargs={'task_name': 'embedding'}
    )
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  MOMENT loaded: {n_params:.1f}M params", flush=True)
    seq_len = model.seq_len  # 512

    @torch.no_grad()
    def embed(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, 1024)"""
        B, T, C = x.shape
        if T > seq_len:
            x = x[:, -seq_len:, :]
        elif T < seq_len:
            pad = torch.zeros(B, seq_len - T, C, dtype=x.dtype, device=x.device)
            x = torch.cat([pad, x], dim=1)
        x_flat = x.permute(0, 2, 1).reshape(B * C, 1, seq_len)
        out = model.embed(x_enc=x_flat.to(DEVICE), reduction='mean')
        embs = out.embeddings.reshape(B, C, -1).mean(dim=1)
        return embs.cpu()

    return embed, n_params * 1e6


# ---------------------------------------------------------------------------
# TimesFM-1.0 extractor
# ---------------------------------------------------------------------------

def build_timesfm_extractor():
    import timesfm
    print("Loading TimesFM-1.0-200M...", flush=True)
    hparams = timesfm.TimesFmHparams(
        backend='gpu' if 'cuda' in DEVICE else 'cpu',
        per_core_batch_size=32,
        horizon_len=128,
        context_len=512,
    )
    ckpt = timesfm.TimesFmCheckpoint(
        huggingface_repo_id='google/timesfm-1.0-200m-pytorch',
    )
    tfm = timesfm.TimesFm(hparams=hparams, checkpoint=ckpt)
    tfm.load_from_checkpoint(checkpoint=ckpt)
    _model = tfm._model
    n_params = sum(p.numel() for p in _model.parameters()) / 1e6
    print(f"  TimesFM-1.0 loaded: {n_params:.1f}M params", flush=True)

    CONTEXT_LEN = 512
    HIDDEN_SIZE = 1280
    _hidden_cache = [None]

    def _hook(module, input, output):
        _hidden_cache[0] = output.detach().clone()

    _model.stacked_transformer.register_forward_hook(_hook)

    @torch.no_grad()
    def embed(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, 1280)"""
        B, T, C = x.shape
        if T > CONTEXT_LEN:
            x = x[:, -CONTEXT_LEN:, :]
        elif T < CONTEXT_LEN:
            pad = torch.zeros(B, CONTEXT_LEN - T, C, dtype=x.dtype, device='cpu')
            x = torch.cat([pad, x.cpu()], dim=1)
        x_flat = x.permute(0, 2, 1).reshape(B * C, CONTEXT_LEN).float()

        all_hidden = []
        BS = 32
        for s in range(0, B * C, BS):
            b = x_flat[s:s + BS]
            B_loc = b.shape[0]
            t_ts = b.to(DEVICE)
            t_pad = torch.zeros(B_loc, CONTEXT_LEN, device=DEVICE)
            t_freq = torch.zeros(B_loc, 1, dtype=torch.long, device=DEVICE)
            _model(t_ts, t_pad, t_freq)
            h = _hidden_cache[0].mean(dim=1)  # (B_loc, 1280)
            all_hidden.append(h.cpu())

        hidden = torch.cat(all_hidden, dim=0)  # (B*C, 1280)
        return hidden.reshape(B, C, -1).mean(dim=1)  # (B, 1280)

    return embed, n_params * 1e6


# ---------------------------------------------------------------------------
# Moirai extractor (mirrors run_all_baselines.py: MoiraiModule.from_pretrained)
# ---------------------------------------------------------------------------

def build_moirai_extractor(size: str = 'base'):
    from uni2ts.model.moirai import MoiraiModule
    hf_id = f'Salesforce/moirai-1.1-R-{size}'
    print(f"Loading Moirai-1.1-R-{size} from {hf_id}...", flush=True)
    model = MoiraiModule.from_pretrained(hf_id)
    model.eval()
    model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Moirai loaded: {n_params:.1f}M params, d_model={model.d_model}", flush=True)

    D_MODEL = model.d_model
    PATCH_SIZE = 32  # fixed from supported patch_sizes
    CONTEXT_LEN = 512
    N_PATCHES = CONTEXT_LEN // PATCH_SIZE  # 16
    MAX_PATCH = model.patch_sizes[-1]  # 128

    hidden_cache = [None]

    def hook(module, input, output):
        hidden_cache[0] = output.detach().clone()

    model.encoder.register_forward_hook(hook)

    @torch.no_grad()
    def embed(x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, D_MODEL). Sub-batched over B*C for memory safety."""
        B, T, C = x.shape
        if T > CONTEXT_LEN:
            x = x[:, -CONTEXT_LEN:, :]
        elif T < CONTEXT_LEN:
            pad = torch.zeros(B, CONTEXT_LEN - T, C, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)

        # Process each channel as a separate univariate sequence
        x_flat = x.permute(0, 2, 1).reshape(B * C, CONTEXT_LEN)  # (B*C, T)
        N = B * C

        # Sub-batch over N for memory safety (mirrors TimesFM extractor)
        BS_MOIRAI = 64
        all_hidden = []
        for s in range(0, N, BS_MOIRAI):
            x_sub = x_flat[s:s + BS_MOIRAI].to(DEVICE)  # (bs, T)
            n_sub = x_sub.shape[0]

            # Build patched input
            target = x_sub.view(n_sub, N_PATCHES, PATCH_SIZE)
            target_padded = torch.zeros(n_sub, N_PATCHES, MAX_PATCH, device=DEVICE)
            target_padded[:, :, :PATCH_SIZE] = target
            observed = torch.zeros(n_sub, N_PATCHES, MAX_PATCH, dtype=torch.bool, device=DEVICE)
            observed[:, :, :PATCH_SIZE] = True

            sample_id = torch.arange(n_sub, device=DEVICE).unsqueeze(1).expand(-1, N_PATCHES)
            time_id = torch.arange(N_PATCHES, device=DEVICE).unsqueeze(0).expand(n_sub, -1)
            variate_id = torch.zeros(n_sub, N_PATCHES, dtype=torch.long, device=DEVICE)
            pred_mask = torch.zeros(n_sub, N_PATCHES, dtype=torch.bool, device=DEVICE)
            ps_tensor = torch.full((n_sub, N_PATCHES), PATCH_SIZE, dtype=torch.long, device=DEVICE)

            model(target_padded, observed, sample_id, time_id, variate_id, pred_mask, ps_tensor)
            h = hidden_cache[0]  # (n_sub, N_PATCHES, D_MODEL)
            h_pooled = h.mean(dim=1)  # (n_sub, D_MODEL)
            all_hidden.append(h_pooled.cpu())

        hidden = torch.cat(all_hidden, dim=0)  # (N, D_MODEL)
        return hidden.reshape(B, C, D_MODEL).mean(dim=1)  # (B, D_MODEL)

    return embed, n_params * 1e6


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def get_feat_cache_path(model_name: str, dataset: str) -> Path:
    """Return path for the unified flat cache file."""
    if model_name == 'moment':
        return MOMENT_FEAT_DIR / f"{dataset}_v31ext_flat.npz"
    elif model_name == 'timesfm':
        return TFM_FEAT_DIR / f"{dataset}_v31ext_flat.npz"
    elif model_name == 'moirai':
        return MOIRAI_FEAT_DIR / f"{dataset}_v31ext_flat.npz"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def check_legacy_split_cache(model_name: str, dataset: str):
    """Check if legacy split-keyed caches exist (timesfm/moirai from run_all_baselines.py).

    Legacy format has separate _train.npz, _val.npz, _test.npz files with keys:
    'feats' (N,D), 'labels' (N,K), 'dt_indices' (N,K)

    Returns (tr_data, va_data, te_data) or None if not found.
    """
    if model_name == 'moment':
        return None  # MOMENT used .pt format, skip
    ds_safe = dataset.replace('/', '_')
    if model_name == 'timesfm':
        feat_dir = TFM_FEAT_DIR
    else:
        feat_dir = MOIRAI_FEAT_DIR

    tr_path = feat_dir / f"{ds_safe}_train.npz"
    va_path = feat_dir / f"{ds_safe}_val.npz"
    te_path = feat_dir / f"{ds_safe}_test.npz"

    if tr_path.exists() and va_path.exists() and te_path.exists():
        tr = np.load(tr_path)
        va = np.load(va_path)
        te = np.load(te_path)
        # Legacy format: feats (N,D), labels (N,K), dt_indices (N,K)
        return (torch.from_numpy(tr['feats']),
                torch.from_numpy(tr['labels']),
                torch.from_numpy(tr['dt_indices']),
                torch.from_numpy(va['feats']),
                torch.from_numpy(va['labels']),
                torch.from_numpy(va['dt_indices']),
                torch.from_numpy(te['feats']),
                torch.from_numpy(te['labels']),
                torch.from_numpy(te['dt_indices']))
    return None


# (cache_features removed; feature caching is inlined in run_one)


# ---------------------------------------------------------------------------
# Load existing results JSON (append-only mode)
# ---------------------------------------------------------------------------

def load_existing_results(json_path: Path) -> list:
    if not json_path.exists():
        return []
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # New format: {'results': [...], 'model': ..., 'timestamp': ...}
        if 'results' in data:
            return data['results']
        # Old format (moirai/timesfm): {'FD001': {dataset, seeds, per_seed, mean}, ...}
        # Convert to flat result list (one entry per dataset, seeds=[42,123,456])
        converted = []
        for key, val in data.items():
            if key in ('model_info',) or not isinstance(val, dict):
                continue
            if 'dataset' not in val:
                continue
            # Old format has per_seed list; expand to per-seed records
            ds = val['dataset']
            per_seed = val.get('per_seed', [])
            seeds = val.get('seeds', [42, 123, 456])
            horizons = val.get('horizons', CMAPSS_HORIZONS)
            for s, auroc in zip(seeds, per_seed):
                converted.append({
                    'dataset': ds,
                    'seed': s,
                    'label_fraction': 1.0,
                    'mean_h_auroc': auroc,
                    'horizons': horizons,
                })
        return converted
    return []


def already_done(results: list, dataset: str, seed: int,
                 label_fraction: float = 1.0) -> bool:
    for r in results:
        if (r.get('dataset') == dataset and
                r.get('seed') == seed and
                abs(r.get('label_fraction', 1.0) - label_fraction) < 0.01):
            return True
    return False


# ---------------------------------------------------------------------------
# Main run function for one (model, dataset, seed)
# ---------------------------------------------------------------------------

def run_one(model_name: str, dataset: str, seed: int,
            embed_fn, n_encoder_params: float,
            bundle: dict, results_list: list,
            label_fraction: float = 1.0):
    """Run one (model, dataset, seed) baseline. Appends result to results_list."""
    from train import EventDataset
    from _runner_v31 import _apply_label_fraction

    tag = f"{dataset}_{model_name}-mlp_s{seed}_lf{int(label_fraction*100)}"
    print(f"\n{'='*60}", flush=True)
    print(f"[{tag}]", flush=True)

    # Apply label fraction to training entities
    ft_train = bundle['ft_train']
    if label_fraction < 1.0:
        ft_train = _apply_label_fraction(ft_train, label_fraction, seed, dataset)

    # Build modified bundle for cache (full labels used for val/test)
    bundle_lf = dict(bundle)
    bundle_lf['ft_train'] = ft_train

    # Get features (cache per dataset, not per label_fraction).
    # Priority: 1) unified flat cache  2) legacy split-keyed cache (timesfm/moirai)  3) extract
    cache_path = get_feat_cache_path(model_name, dataset)
    legacy = check_legacy_split_cache(model_name, dataset)

    if cache_path.exists():
        print(f"  Loading unified cache: {cache_path}", flush=True)
        d = np.load(cache_path, allow_pickle=True)
        feat_tr_full = torch.from_numpy(d['tr_feats'])
        feat_tr_flat_full = torch.from_numpy(d['tr_feats_flat'])
        lbl_tr_surf_full = torch.from_numpy(d['tr_labels_surf'])
        lbl_tr_flat_full = torch.from_numpy(d['tr_labels_flat'])
        dt_tr_flat_full = torch.from_numpy(d['tr_dt_flat'])
        feat_va_flat = torch.from_numpy(d['va_feats_flat'])
        lbl_va_flat = torch.from_numpy(d['va_labels_flat'])
        dt_va_flat = torch.from_numpy(d['va_dt_flat'])
        feat_te = torch.from_numpy(d['te_feats'])
        lbl_te_surf = torch.from_numpy(d['te_labels_surf'])
        horizons = list(d['horizons'].tolist())
        n_ch = int(d['n_channels'][0])

    elif legacy is not None:
        print(f"  Converting legacy split cache for {model_name}/{dataset}...", flush=True)
        (feat_tr_full, lbl_tr_surf_l, dt_tr_surf,
         feat_va_full, lbl_va_surf_l, dt_va_surf,
         feat_te, lbl_te_surf, _dt_te) = legacy
        # lbl_tr_surf_l is (N,K) float, dt_tr_surf is (N,K) long
        lbl_tr_surf_full = lbl_tr_surf_l
        K = lbl_tr_surf_full.shape[1]
        N_tr = feat_tr_full.shape[0]
        feat_tr_flat_full = feat_tr_full.unsqueeze(1).expand(-1, K, -1).reshape(N_tr * K, -1).contiguous()
        lbl_tr_flat_full = lbl_tr_surf_full.reshape(-1).float()
        dt_tr_flat_full = dt_tr_surf.reshape(-1).long()
        N_va = feat_va_full.shape[0]
        feat_va_flat = feat_va_full.unsqueeze(1).expand(-1, K, -1).reshape(N_va * K, -1).contiguous()
        lbl_va_flat = lbl_va_surf_l.reshape(-1).float()
        dt_va_flat = dt_va_surf.reshape(-1).long()
        horizons = bundle['horizons']
        n_ch = bundle['n_channels']
        # Save unified cache for reuse
        np.savez(cache_path,
                 tr_feats=feat_tr_full.numpy(),
                 tr_feats_flat=feat_tr_flat_full.numpy(),
                 tr_labels_surf=lbl_tr_surf_full.numpy(),
                 tr_labels_flat=lbl_tr_flat_full.numpy(),
                 tr_dt_flat=dt_tr_flat_full.numpy(),
                 va_feats=feat_va_full.numpy(),
                 va_feats_flat=feat_va_flat.numpy(),
                 va_labels_surf=lbl_va_surf_l.numpy(),
                 va_labels_flat=lbl_va_flat.numpy(),
                 va_dt_flat=dt_va_flat.numpy(),
                 te_feats=feat_te.numpy(),
                 te_labels_surf=lbl_te_surf.numpy(),
                 horizons=np.array(horizons, dtype=np.int32),
                 n_channels=np.array([n_ch], dtype=np.int32))
        print(f"  Legacy cache converted and saved: {cache_path}", flush=True)

    else:
        print(f"  Extracting features for {model_name}/{dataset}...", flush=True)
        tr_ds, va_ds, te_ds, n_ch, horizons = bundle_to_datasets(bundle, dataset)
        tr = extract_features_to_flat(embed_fn, tr_ds, horizons)
        va = extract_features_to_flat(embed_fn, va_ds, horizons)
        te = extract_features_to_flat(embed_fn, te_ds, horizons)
        feat_tr_full = tr[0]
        feat_tr_flat_full = tr[1]
        lbl_tr_surf_full = tr[2]
        lbl_tr_flat_full = tr[3]
        dt_tr_flat_full = tr[4]
        feat_va_flat = va[1]
        lbl_va_flat = va[3]
        dt_va_flat = va[4]
        feat_te = te[0]
        lbl_te_surf = te[2]
        np.savez(cache_path,
                 tr_feats=feat_tr_full.numpy(),
                 tr_feats_flat=feat_tr_flat_full.numpy(),
                 tr_labels_surf=lbl_tr_surf_full.numpy(),
                 tr_labels_flat=lbl_tr_flat_full.numpy(),
                 tr_dt_flat=dt_tr_flat_full.numpy(),
                 va_feats=va[0].numpy(),
                 va_feats_flat=feat_va_flat.numpy(),
                 va_labels_surf=va[2].numpy(),
                 va_labels_flat=lbl_va_flat.numpy(),
                 va_dt_flat=dt_va_flat.numpy(),
                 te_feats=feat_te.numpy(),
                 te_labels_surf=lbl_te_surf.numpy(),
                 horizons=np.array(horizons, dtype=np.int32),
                 n_channels=np.array([n_ch], dtype=np.int32))
        print(f"  Saved features: tr={feat_tr_full.shape}, va={va[0].shape}, te={feat_te.shape}",
              flush=True)

    K = len(horizons)

    # Apply label fraction: subsample flat training features
    if label_fraction < 1.0:
        N_tr = feat_tr_full.shape[0]
        N_keep = max(1, int(round(N_tr * label_fraction)))
        rng = np.random.RandomState(seed + 7777)
        keep_idx = rng.choice(N_tr, size=N_keep, replace=False)
        keep_idx_flat = np.concatenate([keep_idx * K + k for k in range(K)])
        feat_tr_flat = feat_tr_flat_full[keep_idx_flat]
        lbl_tr_flat = lbl_tr_flat_full[keep_idx_flat]
        dt_tr_flat = dt_tr_flat_full[keep_idx_flat]
        print(f"  lf={label_fraction:.1f}: keeping {N_keep}/{N_tr} train samples",
              flush=True)
    else:
        feat_tr_flat = feat_tr_flat_full
        lbl_tr_flat = lbl_tr_flat_full
        dt_tr_flat = dt_tr_flat_full

    d_input = feat_tr_flat.shape[1]
    print(f"  d_input={d_input}, K={K}, horizons={horizons}, n_ch={n_ch}", flush=True)
    print(f"  tr_flat={feat_tr_flat.shape}, va_flat={feat_va_flat.shape}, te={feat_te.shape}",
          flush=True)

    t0 = time.time()

    # Train head
    run_name = f"v31-{model_name}-{dataset}-s{seed}"
    if WANDB_OK:
        wandb.init(project='industrialjepa', name=run_name,
                   config={'dataset': dataset, 'seed': seed, 'model': model_name,
                           'label_fraction': label_fraction, 'head': 'dt-MLP-unified',
                           'n_channels': n_ch, 'horizons': horizons},
                   reinit=True, tags=['v31', model_name, 'baseline', dataset])

    head = train_head_on_feats(
        feat_tr_flat, lbl_tr_flat, dt_tr_flat,
        feat_va_flat, lbl_va_flat, dt_va_flat,
        d_input=d_input, K=K, seed=seed, run_name=run_name,
    )

    # Evaluate
    metrics = evaluate_head_on_feats(head, feat_te, lbl_te_surf, K, horizons)
    elapsed = time.time() - t0

    print(f"[{tag}] h-AUROC={metrics['mean_h_auroc']:.4f} "
          f"AUPRC={metrics['pooled_auprc']:.4f} "
          f"elapsed={elapsed:.1f}s", flush=True)

    if WANDB_OK:
        wandb.log({'h_auroc': metrics['mean_h_auroc'],
                   'pooled_auprc': metrics['pooled_auprc']})
        wandb.finish()

    # Save surface
    np.savez(SURF_DIR / f"{tag}.npz",
             p_surface=metrics['p_surface'],
             y_surface=metrics['y_surface'],
             horizons=np.array(horizons, dtype=np.int32))

    result = {
        'tag': tag,
        'dataset': dataset,
        'seed': seed,
        'label_fraction': label_fraction,
        'model': model_name,
        'n_channels': n_ch,
        'n_encoder_params': int(n_encoder_params),
        'n_head_params': head.n_params(),
        'mean_h_auroc': metrics['mean_h_auroc'],
        'pooled_auprc': metrics['pooled_auprc'],
        'n_valid_horizons': metrics['n_valid_horizons'],
        'horizons': horizons,
        'elapsed_s': elapsed,
        'timestamp': datetime.now().isoformat(),
    }
    results_list.append(result)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True,
                        choices=['moment', 'timesfm', 'moirai'],
                        help='Which foundation model backbone to use')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all 11)')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456])
    parser.add_argument('--label-fractions', nargs='+', type=float, default=[1.0],
                        help='Label fractions to run (e.g., 1.0 0.1)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if already in results JSON')
    parser.add_argument('--delete-cache', action='store_true',
                        help='Delete flat feature cache after each dataset (saves disk space)')
    args = parser.parse_args()

    model_name = args.model
    seeds = args.seeds
    label_fractions = args.label_fractions
    datasets = args.datasets if args.datasets else ALL_11_DATASETS

    # JSON paths per model
    json_paths = {
        'moment': RES_DIR / 'moment_baseline.json',
        'timesfm': RES_DIR / 'timesfm_baseline.json',
        'moirai': RES_DIR / 'moirai_baseline.json',
    }
    out_path = json_paths[model_name]

    print(f"\nV31 Baseline Extension: {model_name.upper()}", flush=True)
    print(f"Datasets: {datasets}", flush=True)
    print(f"Seeds: {seeds}", flush=True)
    print(f"Label fractions: {label_fractions}", flush=True)
    print(f"Output: {out_path}", flush=True)
    print("=" * 60, flush=True)

    # Load existing results
    existing_results = load_existing_results(out_path)
    print(f"Loaded {len(existing_results)} existing results from {out_path}", flush=True)

    # Build extractor once
    if model_name == 'moment':
        embed_fn, n_params = build_moment_extractor()
    elif model_name == 'timesfm':
        embed_fn, n_params = build_timesfm_extractor()
    elif model_name == 'moirai':
        embed_fn, n_params = build_moirai_extractor()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    all_results = list(existing_results)
    new_results = []

    for dataset in datasets:
        print(f"\n{'='*60}", flush=True)
        print(f"Loading bundle: {dataset}", flush=True)
        try:
            bundle = load_bundle(dataset)
        except Exception as e:
            print(f"  ERROR loading {dataset}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

        for lf in label_fractions:
            for seed in seeds:
                if not args.force and already_done(all_results, dataset, seed, lf):
                    print(f"  SKIP: {dataset} seed={seed} lf={lf} (already done)",
                          flush=True)
                    continue

                try:
                    r = run_one(model_name=model_name, dataset=dataset, seed=seed,
                                embed_fn=embed_fn, n_encoder_params=n_params,
                                bundle=bundle, results_list=all_results,
                                label_fraction=lf)
                    new_results.append(r)

                    # Save after each run (append-only)
                    with open(out_path, 'w') as f:
                        json.dump({'results': all_results,
                                   'model': model_name,
                                   'timestamp': datetime.now().isoformat()},
                                  f, indent=2)
                    print(f"  Saved: {out_path} ({len(all_results)} total results)",
                          flush=True)

                except Exception as e:
                    print(f"  ERROR [{dataset} s{seed} lf={lf}]: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

        # Delete flat cache for this dataset after all lf/seed runs complete (disk management)
        if args.delete_cache:
            cache_p = get_feat_cache_path(model_name, dataset)
            if cache_p.exists():
                size_mb = cache_p.stat().st_size / 1e6
                cache_p.unlink()
                print(f"  Cache deleted: {cache_p.name} ({size_mb:.0f} MB freed)", flush=True)

    # Final summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: {model_name.upper()} h-AUROC (mean +/- std, 3 seeds)")
    from collections import defaultdict
    by_ds_lf = defaultdict(list)
    for r in all_results:
        if 'mean_h_auroc' in r:
            k = (r['dataset'], r.get('label_fraction', 1.0))
            by_ds_lf[k].append(r['mean_h_auroc'])

    for (ds, lf), scores in sorted(by_ds_lf.items()):
        if scores:
            print(f"  {ds:10s} lf={lf:.1f}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} "
                  f"({len(scores)} seeds)")

    print(f"\nNew results this run: {len(new_results)}", flush=True)
    print(f"Total results in JSON: {len(all_results)}", flush=True)


if __name__ == '__main__':
    main()

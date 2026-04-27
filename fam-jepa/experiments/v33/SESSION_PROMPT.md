# V33 Overnight Session -- Spatiotemporal Masking for Cross-Channel JEPA

**Duration**: Full overnight (~8-10 hours). Do NOT stop early.
**Goal**: Add cross-channel learning to FAM via spatiotemporal masking. Two levels: (1) channel dropout as go/no-go gate, (2) full ST-JEPA with factored attention if gate passes. Produce paper-ready ablation on 3 datasets.
**Commit cadence**: `git add -A && git commit -m "v33 phase X: <description>" && git push` after each phase. At minimum once per hour.
**Self-check**: After each phase, verify outputs exist and are reasonable. Compare to baselines. If a result looks wrong, investigate before moving on.
**Codex will review** your work -- be thorough, document everything, no shortcuts.

---

## CRITICAL: Lessons from Past Failures

**READ THIS BEFORE WRITING ANY CODE.** Two prior attempts at cross-channel attention failed. You MUST avoid repeating these mistakes.

### V14 Failure: Sensor-ID Shortcut Learning
- Architecture: 14 sensor-as-token with **learnable sensor-ID embeddings** + alternating temporal/cross-sensor attention
- Result: FD001 frozen@100% = 14.98 (good), but @20% std=10.19 (catastrophic variance). FD003 regressed +5.24 RMSE vs baseline.
- Root cause: Learnable sensor-ID embeddings let the model learn `h = f(sensor_ID, sensor_mean)` instead of temporal context. The model memorized per-sensor univariate dynamics. Training loss went to 0 by epoch 20 but probe RMSE was 75 (below random).
- **HARD RULE: NO learnable per-channel/per-sensor embeddings. Ever.**

### V22 Failure: Protocol Artifact Masquerading as Architecture Win
- Architecture: variantA (pure iTransformer, Linear(T=100, d) per channel + cross-channel MHA) and variantB (hybrid temporal + cross-channel fusion).
- Initial result: variantB SMAP AUPRC 0.384 vs baseline 0.290 -- looked like +0.094 win!
- **But**: baseline was trained with variable-past; variants used fixed-past=100. When baseline was retrained with matched protocol (fixed-past=100), it scored 0.382 +/- 0.050. variantB scored 0.373. Paired t-test: p=0.716. **The "win" was entirely a protocol artifact.**
- After matched-protocol re-evaluation across 6 datasets: cross-channel attention only helped on PSM (p=0.015). Everywhere else, within noise.
- **HARD RULE: All comparisons MUST use identical protocol (same data splits, same n_cuts, same max_context, same Dt sampling, same seeds). Never compare across different pretraining configs.**

### What v14/v22 Tell Us
1. Channel identity information (learnable or fixed) enables shortcut learning
2. Protocol differences dominate architecture differences
3. Cross-channel attention only clearly helps on PSM (26 correlated server metrics)
4. The causal temporal transformer is already very strong -- cross-channel is an addition, not a replacement

---

## Datasets for This Session

Three datasets, chosen for maximum information:

| Dataset | Channels | Entities | Expected Benefit | Role |
|---------|----------|----------|-----------------|------|
| **PSM** | 25 | 1 (single stream) | HIGH -- only v22 winner, correlated server metrics | Should improve |
| **SMAP** | 25 | 55 (independent) | NONE -- independent spacecraft subsystems | Negative control |
| **FD001** | 14 | 100 engines | MODERATE -- physically coupled sensors | Reference |

**Current baseline h-AUROC** (v30/v31, 3 seeds):
- PSM 100%: 0.562 +/- 0.013
- SMAP 100%: 0.598 +/- 0.036
- FD001 100%: 0.786 +/- 0.033

These are the numbers to beat (or at least not regress).

---

## Phase 0: Environment Setup (~15 min)

```bash
cd /home/sagemaker-user/IndustrialJEPA/fam-jepa
pip install wandb psutil --quiet
```

### Verify data loading
```python
import sys
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v29')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v27')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v28')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/archive/v24')
sys.path.insert(0, '/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/archive/v11')

from _runner_v29 import LOADERS
for ds in ['FD001', 'SMAP', 'PSM']:
    bundle = LOADERS[ds]()
    seqs = bundle['pretrain_seqs']
    first_key = list(seqs.keys())[0]
    C = seqs[first_key].shape[-1]
    n_entities = len(seqs)
    total_steps = sum(s.shape[0] for s in seqs.values())
    print(f"{ds}: C={C}, entities={n_entities}, total_steps={total_steps}")
```

Expected: PSM C=25, SMAP C=25, FD001 C=14.

### Create directories
```python
from pathlib import Path
V33_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa/experiments/v33')
for d in ['ckpts', 'surfaces', 'results', 'results/phase1', 'results/phase2',
          'results/phase3', 'results/phase4']:
    (V33_DIR / d).mkdir(parents=True, exist_ok=True)
```

**Commit after Phase 0.**

---

## Phase 1: Baseline Re-Pretrain with Matched Protocol (~2 hours)

**CRITICAL**: Before testing ANY new architecture, establish a fresh baseline with EXACTLY the protocol that all variants will use. This prevents the v22 mistake.

### Protocol (fixed for ALL experiments in this session)
```python
PROTOCOL = {
    'max_context': 512,        # fixed context window
    'patch_size': 16,
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 256,
    'dropout': 0.1,
    'ema_momentum': 0.99,
    'predictor_hidden': 256,
    'norm_mode': 'revin',
    'predictor_kind': 'mlp',
    'event_head_kind': 'discrete_hazard',
    # Pretraining
    'pre_epochs': 50,          # more than default 30 for fair comparison
    'pre_batch': 64,
    'pre_lr': 3e-4,
    'pre_patience': 8,
    'n_cuts': 40,              # windows per entity
    'delta_t_min': 1,
    'delta_t_max': 150,
    'lambda_var': 0.04,
    # Finetuning
    'ft_epochs': 40,
    'ft_batch': 128,
    'ft_lr': 1e-3,
    'ft_patience': 8,
    'label_fraction': 1.0,
    # Seeds
    'seeds': [42, 123, 456],
}
```

### For each dataset (PSM, SMAP, FD001), for each seed:
1. Pretrain standard FAM (channel-fusion, no dropout) with PROTOCOL above
2. Pred-FT on 100% labels
3. Evaluate: h-AUROC, h-AUPRC, save surface as .npz
4. Save checkpoint as `ckpts/{dataset}_baseline_s{seed}.pt`

### Output
Save as `results/phase1/baseline_{dataset}.json`:
```json
{
  "dataset": "PSM",
  "protocol": {... exact PROTOCOL dict ...},
  "seeds": {
    "42": {"h_auroc": X, "h_auprc": X, "pretrain_loss": X, "ft_val_loss": X, "pretrain_epochs": X},
    "123": {...},
    "456": {...}
  },
  "mean_h_auroc": X,
  "std_h_auroc": X,
  "mean_h_auprc": X,
  "std_h_auprc": X
}
```

### Sanity check
Compare to v30/v31 baselines. If fresh baseline is >0.03 worse than v30, investigate before continuing. Acceptable variance: +/- 0.02.

**Commit after Phase 1.**

---

## Phase 2: Channel Dropout Gate (~2 hours)

This is the go/no-go experiment. Modify the existing PatchEmbedding to drop channels during pretraining. If this helps on PSM without hurting SMAP/FD001, proceed to Phase 3. If not, pivot to Phase 3-ALT.

### Implementation

Add channel dropout to `PatchEmbedding.forward()`. Create a new class or add a parameter -- do NOT modify the original class in model.py. Instead, create `v33/model_v33.py` that subclasses or wraps:

```python
class ChannelDropoutPatchEmbedding(nn.Module):
    """PatchEmbedding with inverted channel dropout during training."""
    
    def __init__(self, n_channels: int, patch_size: int = 16,
                 d_model: int = 256, channel_drop_rate: float = 0.0):
        super().__init__()
        self.P = patch_size
        self.proj = nn.Linear(n_channels * patch_size, d_model)
        self.channel_drop_rate = channel_drop_rate
        self.n_channels = n_channels
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        P = self.P
        remainder = T % P
        if remainder != 0:
            x = F.pad(x, (0, 0, 0, P - remainder))
            T = x.shape[1]
        
        # Channel dropout: zero out random channels, scale up survivors
        if self.training and self.channel_drop_rate > 0:
            # Per-sample channel mask (same mask across all timesteps)
            keep_prob = 1.0 - self.channel_drop_rate
            mask = torch.bernoulli(
                torch.full((B, 1, C), keep_prob, device=x.device)
            )
            # Ensure at least 1 channel survives
            all_zero = (mask.sum(dim=-1, keepdim=True) == 0)
            if all_zero.any():
                # Force one random channel on
                fix_idx = torch.randint(0, C, (all_zero.sum().item(),), device=x.device)
                mask[all_zero.squeeze(-1), 0, fix_idx] = 1.0
            x = x * mask / keep_prob  # inverted dropout scaling
        
        x = x.reshape(B, T // P, C * P)
        return self.proj(x)
```

Then create a FAM variant that uses this embedding:

```python
class FAM_ChannelDrop(FAM):
    """FAM with channel dropout in context encoder only."""
    def __init__(self, channel_drop_rate: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        # Replace encoder's patch embedding
        self.encoder.patch_embed = ChannelDropoutPatchEmbedding(
            kwargs['n_channels'], kwargs.get('patch_size', 16),
            kwargs.get('d_model', 256), channel_drop_rate
        )
        # Target encoder keeps original (no dropout) -- sees full signal
        # This creates an asymmetry: context encoder learns to be robust
        # to missing channels; target encoder provides clean targets.
```

**IMPORTANT**: Channel dropout applies ONLY to the context encoder, NOT the target encoder. The target encoder always sees the full signal. This is the correct JEPA asymmetry: the context is degraded, the target is clean, the predictor must bridge the gap.

### Sweep
For each dataset (PSM, SMAP, FD001), for each dropout rate (0.0, 0.1, 0.3, 0.5):
- Pretrain with PROTOCOL (same as Phase 1)
- Pred-FT 100% labels
- Evaluate h-AUROC, h-AUPRC
- Use **seed 42 only** for the sweep (save compute)
- Use **3 seeds** for the best dropout rate per dataset

### Go/No-Go Decision

Compute delta = h_AUROC(dropout=best) - h_AUROC(dropout=0.0) for each dataset:
- **PSM delta > +0.01**: PASS (cross-channel robustness helps)
- **SMAP delta > -0.02**: PASS (doesn't hurt the negative control)
- **FD001 delta > -0.02**: PASS (doesn't hurt the reference)

If PSM passes AND neither SMAP nor FD001 regresses: proceed to Phase 3 (full ST-JEPA).
If PSM does NOT pass: proceed to Phase 3-ALT (skip ST-JEPA, run other v33 work).

### Output
Save as `results/phase2/channel_dropout_{dataset}.json`:
```json
{
  "dataset": "PSM",
  "sweep": {
    "0.0": {"h_auroc": X, "h_auprc": X, "seed": 42},
    "0.1": {"h_auroc": X, "h_auprc": X, "seed": 42},
    "0.3": {"h_auroc": X, "h_auprc": X, "seed": 42},
    "0.5": {"h_auroc": X, "h_auprc": X, "seed": 42}
  },
  "best_rate": 0.3,
  "best_3seed": {"mean_h_auroc": X, "std_h_auroc": X, "seeds": {...}},
  "delta_vs_baseline": X,
  "go_nogo": "PASS" or "FAIL"
}
```

Also save a `results/phase2/GATE_DECISION.md` with:
- The sweep results table
- The go/no-go decision with justification
- If FAIL: which Phase 3-ALT tasks to prioritize

**Commit after Phase 2.**

---

## Phase 3: Full ST-JEPA (Only if Phase 2 PASSES) (~3-4 hours)

### Architecture: Per-Channel Tokenization + Factored Attention + Channel Masking

Create `v33/model_stjepa.py`. This is a new encoder variant, NOT a modification of model.py.

#### 3A: Per-Channel Patch Embedding

```python
class PerChannelPatchEmbedding(nn.Module):
    """Per-channel patching: shared projection across channels.
    
    Input: (B, T, C) -> Output: (B, N_patches, C, d_model)
    Each channel gets its own token sequence. The Linear(P, d) projection
    is SHARED across all channels (no per-channel parameters).
    """
    def __init__(self, patch_size: int = 16, d_model: int = 256):
        super().__init__()
        self.P = patch_size
        # SHARED projection -- same weights for every channel
        # This is the anti-v14 safeguard: cannot learn per-sensor clusters
        self.proj = nn.Linear(patch_size, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, N, C, d)"""
        B, T, C = x.shape
        P = self.P
        remainder = T % P
        if remainder != 0:
            x = F.pad(x, (0, 0, 0, P - remainder))
            T = x.shape[1]
        N = T // P
        # (B, T, C) -> (B, N, P, C) -> (B, N, C, P)
        x = x.reshape(B, N, P, C).permute(0, 1, 3, 2)
        # (B, N, C, P) -> Linear -> (B, N, C, d)
        return self.proj(x)
```

**NO channel positional encoding.** Channels are distinguished ONLY by their content (the actual sensor values). No sinusoidal channel PE, no learned embeddings, no random fixed vectors. This is a deliberate design choice:
- Avoids v14's shortcut learning
- Makes the architecture permutation-equivariant over channels
- Forces the cross-channel attention to learn content-based correlations

Temporal PE is applied as before (sinusoidal over patch index), broadcast across the channel dimension.

#### 3B: Factored Attention Encoder

```python
class FactoredTransformerBlock(nn.Module):
    """One temporal attention layer + one cross-channel attention layer."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        # Temporal attention (causal, per-channel)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.temporal_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.temporal_ff_norm = nn.LayerNorm(d_model)
        
        # Cross-channel attention (non-causal, per-timestep)
        self.channel_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.channel_ff_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask, channel_mask=None):
        """
        x: (B, N, C, d)
        causal_mask: (N, N) causal attention mask
        channel_mask: (B, N, C) bool, True=masked (invisible)
        Returns: (B, N, C, d)
        """
        B, N, C, d = x.shape
        
        # --- Temporal attention (per-channel, causal) ---
        # Reshape: (B, N, C, d) -> (B*C, N, d)
        xt = x.permute(0, 2, 1, 3).reshape(B * C, N, d)
        xt2 = self.temporal_norm(xt)
        a, _ = self.temporal_attn(xt2, xt2, xt2, attn_mask=causal_mask)
        xt = xt + self.drop(a)
        xt = xt + self.temporal_ff(self.temporal_ff_norm(xt))
        x = xt.reshape(B, C, N, d).permute(0, 2, 1, 3)  # back to (B, N, C, d)
        
        # --- Cross-channel attention (per-timestep, non-causal) ---
        # Reshape: (B, N, C, d) -> (B*N, C, d)
        xc = x.reshape(B * N, C, d)
        xc2 = self.channel_norm(xc)
        
        # Build per-timestep channel key_padding_mask if channel_mask provided
        ch_kpm = None
        if channel_mask is not None:
            # channel_mask: (B, N, C) -> (B*N, C)
            ch_kpm = channel_mask.reshape(B * N, C)
        
        a, _ = self.channel_attn(xc2, xc2, xc2, key_padding_mask=ch_kpm)
        xc = xc + self.drop(a)
        xc = xc + self.channel_ff(self.channel_ff_norm(xc))
        x = xc.reshape(B, N, C, d)
        
        return x
```

#### 3C: ST-JEPA Context Encoder

```python
class STJEPAEncoder(nn.Module):
    """Spatiotemporal JEPA encoder with factored attention.
    
    Tokenization: per-channel patching (shared projection)
    Attention: temporal causal + cross-channel non-causal (factored)
    Masking: random channel dropout per timestep during pretraining
    Output: h_t (B, d) -- pooled representation of visible tokens
    """
    
    def __init__(self, n_channels, patch_size=16, d_model=256,
                 n_heads=4, n_layers=2, d_ff=256, dropout=0.1,
                 norm_mode='revin', channel_mask_ratio=0.0):
        super().__init__()
        self.d_model = d_model
        self.P = patch_size
        self.n_channels = n_channels
        self.channel_mask_ratio = channel_mask_ratio
        
        if norm_mode == 'revin':
            self.revin = RevIN()  # from model.py
        else:
            self.revin = None
        
        self.patch_embed = PerChannelPatchEmbedding(patch_size, d_model)
        self.layers = nn.ModuleList([
            FactoredTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
        # Attention pool: query attends over all visible tokens
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.0, batch_first=True)
    
    def _generate_channel_mask(self, B, N, C, device):
        """Generate random per-timestep channel mask.
        
        Returns: (B, N, C) bool tensor, True = masked (invisible).
        Guarantees: at least 1 channel visible per timestep.
        """
        if self.channel_mask_ratio <= 0 or not self.training:
            return None
        
        # Variable masking ratio per sample: Uniform(ratio*0.5, min(ratio*1.5, 0.8))
        lo = self.channel_mask_ratio * 0.5
        hi = min(self.channel_mask_ratio * 1.5, 0.8)
        
        mask = torch.zeros(B, N, C, dtype=torch.bool, device=device)
        for b in range(B):
            ratio = torch.empty(1).uniform_(lo, hi).item()
            n_mask = max(1, min(int(ratio * C), C - 1))  # mask at least 1, keep at least 1
            for n in range(N):
                perm = torch.randperm(C, device=device)[:n_mask]
                mask[b, n, perm] = True
        
        return mask
    
    def forward(self, x, mask=None):
        """
        x: (B, T, C). mask: (B, T) bool, True=padding.
        Returns: h_t (B, d) -- pooled over visible tokens at last valid timestep.
        """
        B, T, C = x.shape
        
        # RevIN
        if self.revin is not None:
            x, _ = self.revin(x, mask)
        
        # Per-channel patching -> (B, N, C, d)
        tokens = self.patch_embed(x)
        N = tokens.shape[1]
        
        # Temporal PE (broadcast over channels)
        pe = sinusoidal_pe(torch.arange(N, device=x.device), self.d_model)
        tokens = tokens + pe.unsqueeze(1)  # (N, d) -> (1, N, 1, d) broadcast
        
        # Channel masking
        ch_mask = self._generate_channel_mask(B, N, C, x.device)
        
        # Causal mask for temporal attention
        causal = torch.nn.Transformer.generate_square_subsequent_mask(
            N, device=x.device)
        
        # Factored transformer layers
        h = tokens
        for layer in self.layers:
            h = layer(h, causal_mask=causal, channel_mask=ch_mask)
        h = self.norm(h)
        
        # Pool over visible tokens at last valid timestep
        # Flatten (B, N, C, d) -> (B, N*C, d) for attention pooling
        h_flat = h.reshape(B, N * C, self.d_model)
        
        # Build pooling mask: mask out invisible (channel-masked) tokens
        pool_mask = None
        if ch_mask is not None:
            pool_mask = ch_mask.reshape(B, N * C)  # True = masked
        
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, h_flat, h_flat,
                                    key_padding_mask=pool_mask)
        return pooled.squeeze(1)  # (B, d)
```

#### 3D: Full ST-JEPA Model

Create `FAM_STJEPA` that replaces the context encoder but keeps the same target encoder, predictor, and event head:

```python
class FAM_STJEPA(FAM):
    """FAM with spatiotemporal JEPA context encoder."""
    
    def __init__(self, channel_mask_ratio=0.4, **kwargs):
        # Initialize base FAM (which creates the standard encoder)
        super().__init__(**kwargs)
        # Replace context encoder with ST-JEPA encoder
        self.encoder = STJEPAEncoder(
            n_channels=kwargs['n_channels'],
            patch_size=kwargs.get('patch_size', 16),
            d_model=kwargs.get('d_model', 256),
            n_heads=kwargs.get('n_heads', 4),
            n_layers=kwargs.get('n_layers', 2),
            d_ff=kwargs.get('d_ff', 256),
            dropout=kwargs.get('dropout', 0.1),
            norm_mode=kwargs.get('norm_mode', 'revin'),
            channel_mask_ratio=channel_mask_ratio,
        )
        # Re-initialize target encoder from the NEW encoder
        self._init_target_encoder()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
```

**WAIT**: The target encoder is the standard `TargetEncoder` (channel-fusion, bidirectional, attention pool). The context encoder is now `STJEPAEncoder` (per-channel, factored, with channel masking). This is an **asymmetric design**:
- Context encoder: per-channel tokens, sees partial channels (masked), causal temporal
- Target encoder: channel-fusion tokens, sees all channels, bidirectional

This asymmetry is intentional and correct:
- The target encoder provides a clean, information-rich target h*
- The context encoder must learn robust representations from partial, causal input
- The predictor bridges the gap

BUT: `_init_target_encoder()` copies weights from encoder to target_encoder. With different architectures, the weight copying will only transfer matching keys. The `patch_embed` weights will NOT match (different shapes: Linear(P, d) vs Linear(C*P, d)). This is fine -- the target encoder's patch_embed stays at random init, and EMA updates won't apply to it (shape mismatch check in `update_ema`). The target encoder trains from scratch via the EMA signal from transformer layers that DO match.

**Actually, this is risky.** If the target encoder's patch_embed is random and never updated by EMA, the target representations may be poor and the pretraining signal garbage. Two solutions:

**Option A (safer):** Also make the target encoder per-channel. Create `TargetEncoderPerChannel` that mirrors the per-channel tokenization but uses bidirectional attention and no channel masking. Then `_init_target_encoder` and `update_ema` work correctly because architectures match.

**Option B (simpler):** Keep the channel-fusion target encoder but initialize its `proj` weight matrix such that the per-channel blocks of the matrix match the shared projection. I.e., `target.proj.weight[:, c*P:(c+1)*P] = encoder.proj.weight` for all c. Then EMA updates apply to the transformer layers (which have matching shapes).

**Use Option A.** It's more work but architecturally clean. Create:

```python
class PerChannelTargetEncoder(nn.Module):
    """Bidirectional per-channel target encoder + attention pool."""
    # Same as STJEPAEncoder but:
    # - Bidirectional (no causal mask in temporal attention)
    # - No channel masking (sees everything)
    # - Attention pool -> h* (B, d)
```

### 3E: Training Loop

Use the same `PretrainDataset` and `collate_pretrain` from train.py. The data pipeline is unchanged -- it still produces (context, target, delta_t) tuples. Only the model changes.

Pretraining:
```python
# Same as standard pretrain() but with FAM_STJEPA
model = FAM_STJEPA(
    channel_mask_ratio=BEST_FROM_PHASE2,  # or 0.4 default
    n_channels=C, patch_size=16, d_model=256, n_heads=4,
    n_layers=2, d_ff=256, dropout=0.1, ema_momentum=0.99,
    predictor_hidden=256, norm_mode='revin',
    predictor_kind='mlp', event_head_kind='discrete_hazard',
)
```

**Training budget**: 50 epochs (same as Phase 1). Monitor for convergence -- if loss is still decreasing at epoch 50, extend to 80. The factored attention is ~6x slower per epoch, so budget ~30 min per dataset instead of 5 min.

### 3F: Channel Mask Ratio Sweep (seed 42 only)

For each dataset, sweep channel_mask_ratio in {0.0, 0.2, 0.4, 0.6}:
- 0.0 = per-channel tokenization + factored attention, but no masking (tests architecture alone)
- 0.2, 0.4, 0.6 = increasing masking (tests whether masking adds value beyond architecture)

This disentangles two effects:
1. Does per-channel tokenization + factored attention help? (compare 0.0 to Phase 1 baseline)
2. Does channel masking help on top of that? (compare 0.2-0.6 to 0.0)

### 3G: Full 3-Seed Evaluation

Pick best mask ratio per dataset. Run 3 seeds. Full eval: h-AUROC, h-AUPRC, save surfaces.

### Output
Save as `results/phase3/{dataset}_stjepa.json`:
```json
{
  "dataset": "PSM",
  "mask_ratio_sweep": {
    "0.0": {"h_auroc": X, "seed": 42},
    "0.2": {"h_auroc": X, "seed": 42},
    "0.4": {"h_auroc": X, "seed": 42},
    "0.6": {"h_auroc": X, "seed": 42}
  },
  "best_mask_ratio": 0.4,
  "best_3seed": {"mean_h_auroc": X, "std_h_auroc": X, "seeds": {...}},
  "delta_vs_baseline": X,
  "delta_vs_channel_dropout": X
}
```

**Commit after Phase 3.**

---

## Phase 3-ALT: Pivot Tasks (Only if Phase 2 FAILS)

If channel dropout shows no signal, do NOT waste time on ST-JEPA. Instead:

### 3-ALT-A: MSL Re-Pretrain with p3 Predictor (~1.5h)
The v30 MSL surface is broken (AUROC 0.37, anti-correlated). MSL used `predictor_kind='p2'` while all other datasets used `'p3'`. Re-pretrain MSL with standard config.

### 3-ALT-B: GECCO lf10 with Higher Pos-Weight (~1h)
The v31 GECCO lf10 surface collapsed (AUROC 0.50 = chance). The finetuning had too few positive labels. Try pos_weight in {5, 10, 20} with the v30 pretrained encoder.

### 3-ALT-C: FD002 Per-Condition Normalization (~2h)
FD002 (6 operating conditions) has RMSE 32.4 vs STAR's 13.5. Hypothesis: global z-score hurts because operating conditions have different baselines. Try per-condition normalization in the encoder (condition-stratified RevIN).

**Commit after each sub-phase.**

---

## Phase 4: Ablation Table + Paper-Ready Summary (~1.5 hours)

### 4A: Compile Results Table

Build the full comparison table:

```
| Dataset | Baseline (v33) | Ch-Drop (best) | ST-JEPA (best) | v30 Reference |
|---------|---------------|----------------|----------------|---------------|
| PSM     |               |                |                | 0.562         |
| SMAP    |               |                |                | 0.598         |
| FD001   |               |                |                | 0.786         |
```

For each cell: mean +/- std (3 seeds). Bold the winner per row.

### 4B: Statistical Tests

For the key comparison (best variant vs baseline), compute paired t-test over 3 seeds:
- Report t-statistic, p-value, Cohen's d
- A result is "significant" only if p < 0.05
- With only 3 seeds, significance is hard to achieve -- report effect size even if not significant

### 4C: Attention Pattern Analysis (if ST-JEPA was run)

For PSM (should have cross-channel structure) and SMAP (should not):
1. Extract cross-channel attention weights from the trained ST-JEPA encoder
2. Compute mean attention entropy across channels per timestep
3. For PSM: do certain channel pairs get high attention? Which ones?
4. For SMAP: is attention approximately uniform? (confirms no spurious patterns)
5. Save attention heatmaps as PNGs

### 4D: Training Dynamics Comparison

Plot for each dataset:
- Pretrain loss curves: baseline vs channel-dropout vs ST-JEPA
- Convergence speed (epoch to best val loss)
- Wall-clock time per epoch

### 4E: Session Summary

Write `results/SESSION_SUMMARY.md`:
- What was the go/no-go decision and why
- Key numbers for each variant
- Whether cross-channel learning helps, and on which datasets
- Statistical significance of any improvements
- Recommendation for paper: include ST-JEPA results? As main table? As appendix ablation?
- Honest failure analysis if nothing worked

### 4F: LaTeX Snippet

If any variant beats baseline significantly on PSM:
- Generate the LaTeX for a paper ablation table
- Suggest where it fits (Section 5 Experiments? Appendix?)

### 4G: Update RESULTS.md

Add v33 results to `experiments/RESULTS.md`.

**Final commit with everything.**

---

## Timing Budget

| Phase | Estimated Time | Cumulative |
|-------|---------------|------------|
| 0. Setup | 15 min | 0:15 |
| 1. Baseline (3 datasets x 3 seeds x ~5 min pretrain + 3 min FT) | 1:30 | 1:45 |
| 2. Channel dropout (3 datasets x 4 rates x ~5 min) | 1:00 | 2:45 |
| 2. Channel dropout 3-seed (3 datasets x best rate x 3 seeds) | 0:45 | 3:30 |
| 3. ST-JEPA sweep (3 datasets x 4 rates x ~30 min) | 3:00-4:00 | 6:30-7:30 |
| 3. ST-JEPA 3-seed (3 datasets x 3 seeds x ~30 min) | 1:30 | 8:00-9:00 |
| 4. Analysis + summary | 1:00 | 9:00-10:00 |

If ST-JEPA is skipped (Phase 2 FAIL): phases 3-ALT fill ~4 hours.

---

## File Locations

- Model code: `fam-jepa/model.py` (DO NOT MODIFY -- create v33 variants)
- Training code: `fam-jepa/train.py` (import PretrainDataset, pretrain, etc.)
- Data loaders: `from _runner_v29 import LOADERS` (via sys.path to experiments/v29)
- Runner reference: `fam-jepa/experiments/v31/_runner_v31.py` (copy patterns from here)
- v22 variant code: `fam-jepa/experiments/archive/v22/models_variants.py` (reference, do NOT copy)
- Surface eval: `fam-jepa/evaluation/surface_metrics.py`
- This session's code: `fam-jepa/experiments/v33/` (all new files go here)

## Principles

- **Matched protocol above all.** Every comparison uses IDENTICAL data, splits, seeds, epochs, lr, batch size. The ONLY variable is the model architecture.
- **No learnable channel embeddings.** Learned, fixed, or sinusoidal channel PE are ALL banned. Content-only channel distinction.
- **Gate before invest.** Phase 2 (channel dropout) is cheap. Phase 3 (ST-JEPA) is expensive. Do not skip Phase 2.
- **Report honestly.** If nothing works, say so. "Cross-channel attention does not improve FAM on these datasets" is a valid finding. Do not cherry-pick seeds or metrics.
- **Commit hourly.** Even partial results are valuable.
- **Don't stop early.** Use the full overnight window. If main phases finish early, run additional analysis (more seeds, lf10 comparison, per-horizon breakdown, additional datasets like SMD or BATADAL).

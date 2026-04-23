# FAM Architecture Audit: What's Off-The-Shelf vs. Novel

**Date**: 2026-04-23 (v24)
**Purpose**: Explicit audit of `fam-jepa/model.py` and `fam-jepa/train.py` in terms of borrowed components vs. our contribution. Ensure paper claims match the implementation.

Guiding principle (paper Sec. 1 "Contributions" and user request): we do NOT reinvent components that Chronos-2, TimesFM-2.5, PatchTST, MOMENT, I-JEPA, V-JEPA, LeJEPA, or SIGReg already nailed. We innovate only on:
1. **JEPA** as the self-supervised pretext for multivariate time series (established by TS-JEPA and MTS-JEPA; our twist: cumulative interval target)
2. **Predictor finetuning**: freeze encoder, repurpose the JEPA predictor as the downstream event head
3. **Probability surface $p(t, \Delta t)$** as a discrete-CDF treatment of time-to-event, with interval arithmetic
4. **SIGReg** as an ablation toward target-encoder-free JEPA (Balestriero 2025)

Everything else should be identifiably *imported* or *standard*.

---

## Component-by-component audit

| Component | File / line | Origin | Our contribution? | Keep as-is? |
|---|---|---|---|---|
| **Sinusoidal positional encoding** | `model.py:29` | Attention Is All You Need (Vaswani 2017) | No | Keep - textbook. |
| **RevIN (per-window instance norm)** | `model.py:44-67` | Kim et al. 2022, "Reversible Instance Normalization" | No | Keep. We use the affine-free form; PatchTST/TimesFM/MOMENT all use RevIN or equivalent. |
| **Patch embedding** | `model.py:74-92` | PatchTST (Nie+23) | No | Keep - identical to PatchTST (patch, flatten channels, linear to d). |
| **Causal Transformer block (pre-norm, MHA, FFN)** | `model.py:99-119` | Vaswani 2017 + pre-norm convention (Xiong+20) | No | Keep. Matches Chronos / PatchTST / MOMENT. |
| **Causal attention mask** | `model.py:160` (`generate_square_subsequent_mask`) | PyTorch built-in | No | Keep. |
| **Padding mask for variable-length batching** | `model.py:144-185`, `train.py:collate_*` | PyTorch `MultiheadAttention(key_padding_mask=)` | No | Keep. |
| **Bidirectional target encoder + attention pool** | `model.py:192-244` | I-JEPA / V-JEPA style target encoder; attention pool is ALBERT/CLS-query style | No (borrow) | Keep. The "query + MHA + pool" pattern is standard. |
| **EMA target encoder update** | `model.py:329-341` | I-JEPA (Assran 2023), BYOL (Grill 2020) | No | Keep. Tau=0.99 is standard. |
| **Horizon-conditioned MLP predictor** | `model.py:251-267` | Closest analogue: V-JEPA's "predictor" is also an MLP conditioned on a mask token. Our horizon scalar is novel in this formulation. | **Partial**. The architecture is a standard MLP; the horizon conditioning as a scalar input is a simple novelty that TTE literature uses (DeepHit, DRSA) but JEPA literature does not. | Keep. |
| **Event head (LayerNorm + Linear -> sigmoid)** | `model.py:274-284` | Standard prediction head | No | Keep. |
| **Cumulative target interval $x_{(t, t+\Delta t]}$** | `train.py:PretrainDataset` | **Ours (v24)**. Previous JEPA formulations use fixed-length future windows; we use variable-length cumulative intervals so downstream event probability CDF matches pretrain interval semantics. | **Yes** | Keep - this is the main pretrain novelty. |
| **L1 loss on L2-normalized representations** | `train.py:pretrain:244-247` | I-JEPA uses L1 or smooth-L1; L2 normalization is standard in contrastive/JEPA lit | No | Keep. |
| **Variance regularizer (prevent collapse)** | `train.py:pretrain:249` | VICReg-family (Bardes+22), LeJEPA / SIGReg (Balestriero 2025) | No (borrow) | Keep. LAMBDA_VAR=0.04. |
| **SIGReg (drop target encoder, spectral reg)** | (ablation-only, v23) | Balestriero 2025 | No | Not in default pipeline; included for ablation only. |
| **Positive-weighted BCE for class imbalance** | `train.py:finetune` | `torch.nn.BCEWithLogitsLoss(pos_weight=)` | No | Keep. We use $w^+ = N_{\text{neg}}/N_{\text{pos}}$, clamped to [1, 1000]. |
| **Binary label surface from TTE** | `evaluation/losses.py:build_label_surface` | TTE / DRSA-style discretization | No | Keep. $y(t, \Delta t_k) = \mathbb{1}[\text{TTE}(t) \leq \Delta t_k]$. |
| **Pooled AUPRC over $(t, \Delta t)$ cells** | `evaluation/surface_metrics.py:evaluate_probability_surface` | Standard average-precision on flattened surface | No (scoring convention is ours) | Keep. **This is the paper's evaluation contribution.** |
| **Monotonicity violation rate diagnostic** | `evaluation/surface_metrics.py:monotonicity_violation_rate` | **Ours**. Trivial derivation from CDF semantics but not standard in existing benchmarks. | **Yes (minor)** | Keep. |
| **Interval arithmetic $P(\text{event in }(t+a, t+b]) = p(t,b) - p(t,a)$** | Paper only; not explicit in code but emerges from the CDF | **Ours (framing)**. | **Yes** | Paper-only construct. |
| **Minimum context floor (skip predictions with <8 tokens)** | `train.py:EventDataset(min_context=128)` | ARCHITECTURE.md convention derived from self-attention failure mode at small N | No (standard practice) | Keep. |
| **AdamW + cosine schedule** | `train.py:pretrain` / `finetune` | Loshchilov+19; standard | No | Keep. |
| **Early stopping on val loss** | `train.py:pretrain` / `finetune` | Standard | No | Keep. |
| **Gradient clip norm=1.0** | `train.py:pretrain` | Standard | No | Keep. |

---

## Where we could lean harder on proven components (future work, not tonight)

- **Tokenizer**: currently a simple patch + linear projection. PatchTST, Chronos-2, TimesFM all use essentially this with per-patch RevIN. We could import a pre-built tokenizer layer from `transformers` or a `chronos.Tokenizer` but this is a 20-line component that costs nothing to own.
- **Attention impl**: we use `nn.MultiheadAttention`. Flash-Attention / PyTorch 2 SDPA would be faster but has no correctness benefit on our small (32-token) context.
- **Per-horizon head sharing**: we use *one* linear head $\mathbf{w}, b$ applied to $\hat{\mathbf{h}}_{(t,t+\Delta t]}$ regardless of $\Delta t$. This is parameter-efficient and the horizon information flows in via the predictor. Alternative (not used): per-horizon head $(\mathbf{w}_k, b_k)$. Our choice keeps the head at 513 parameters vs $K \cdot 513$.
- **Foundation-model encoder**: we could directly replace the encoder with frozen Chronos-2 embeddings (`baseline_chronos2.py` does exactly this for comparison). In the paper table FAM(ours) vs Chronos-2 probe show this is a *complementary* direction, not a replacement - dataset-specific JEPA pretraining shines on SMAP (+0.11 AUPRC) but ties on FD001.

---

## Verification against paper.tex

Current paper.tex Method section (Sec. 3) is **consistent** with `model.py` + `train.py` after the v24 update:

- Patch $P=16$ global, $P=1$ for sepsis: ✓ (paper L112; `train.py:P=16`, `phase6_sepsis.py:SEPSIS_P=1`)
- RevIN: ✓ (paper L114; `model.py:RevIN`)
- Causal encoder, $d=256$, 2 layers, 4 heads, $d_{ff}=256$: ✓ (paper L147; `train.py:D_MODEL=256 / D_FF=256 / N_LAYERS=2 / N_HEADS=4`)
- Predictor MLP with horizon scalar input: ✓ (paper L153; `model.py:Predictor`)
- EMA target encoder, $\tau=0.99$: ✓ (paper L160; `model.py:FAM(ema_momentum=0.99)`)
- L1 loss on L2-normalized + var-reg: ✓ (paper L162; `train.py:244-250`)
- Cumulative target $x_{(t,t+\Delta t]}$ - no gap: ✓ (paper L87-89; `train.py:PretrainDataset.__getitem__`)
- Pred-FT freezes encoder, trains predictor + event head: ✓ (paper L165-167; `train.py:finetune(mode='pred_ft')`)
- Pos-weighted BCE: ✓ (paper L190; `train.py:finetune`)
- Monotonicity violation as diagnostic (not constraint): ✓ (paper L197; `evaluation/surface_metrics.py`)
- Interval arithmetic: ✓ (paper L198-199; implicit in the stored surface; not code)

**No drift** between paper description and implementation. The only things labeled "legacy" in the paper are the ablation tables (tab:finetune_ablation, tab:label_efficiency) which use the V17 backbone with F1w metric - that is explicitly called out via the scope note at the top of the Ablations section.

---

## Summary

FAM is a JEPA-style SSL pipeline built from standard Transformer + PatchTST + RevIN + I-JEPA components. Three things are ours:

1. **Cumulative-interval pretrain target** ($x_{(t,t+\Delta t]}$), aligning pretrain and finetune interval semantics.
2. **Predictor finetuning** as the downstream recipe (freeze encoder, fine-tune predictor + 513-param event head).
3. **Probability surface $p(t, \Delta t)$ with pooled AUPRC evaluation and interval arithmetic.

Everything else is battle-tested and imported in spirit (not always as a literal library call, but matching the published reference design).

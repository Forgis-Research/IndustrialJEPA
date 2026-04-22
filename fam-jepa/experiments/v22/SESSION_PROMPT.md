# V22 Overnight Session — Fix Anomaly Pred-FT + Cross-Channel Attention

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: overnight (~10 hours on A10G)
**Prereqs**: Read CLAUDE.md, experiments/RESULTS.md, v21 SESSION_PROMPT.md

---

## Two goals

### Goal A: Anomaly pred-FT with correct splits (critical)

V21 anomaly results are Mahalanobis-only — the predictor was never finetuned
on SMAP/MSL/SMD because a chronological split of the concatenated test array
crossed entity boundaries and produced AUROC 0.38 (anti-correlated).

**Root cause**: SMAP=55 entities, MSL=27, SMD=28 machines, all concatenated
along time.  A 60/10/30 chrono split trained on entities 1-33, tested on 39-55.

**Fix (already committed)**: `data/smap_msl.py` now has `split_smap_entities()`
which does an INTRA-entity chronological split.  For each entity independently:
```
[--- ft_train 60% ---|-- gap 100 --|-- ft_val 10% --|-- gap 100 --|-- ft_test 30% --]
```
Every entity appears in all three splits.  The predictor sees all channels
during training.  Gap = window_size prevents temporal leakage.

**What to run**: pred-FT with BCE on SMAP, MSL, SMD using the new entity splits.
3 seeds.  Store surfaces.  Compute AUPRC + legacy metrics.
PSM and MBA are single streams — use chronological split with gap.

### Goal B: Cross-channel attention in the encoder (important)

Add cross-channel structure to the **encoder** (pretrained self-supervised,
no labels).  The encoder is frozen during finetuning, so downstream trainable
params (predictor + head) are identical with or without this change.  Low-label
performance is NOT a concern — the encoder sees unlimited unlabeled data.

V14 tried cross-channel attention but failed because of **sensor-ID embeddings**
(shortcut learning: loss→0, probe below random).  The architecture itself
produced the best-ever frozen result (14.98 RMSE @ 100%).  The lesson: no
learnable per-sensor embeddings; use fixed positional encoding or nothing.

**Architecture: cross-channel refinement on h_past**

```
Input: (B, T, C)

Temporal path (unchanged):
  Linear(C, d=256) → (B, T, 256) → causal Transformer → h_past (B, 256)

Channel path (NEW — operates on the LAST window's raw sensor values):
  x_recent: (B, W_ch, C)  — last W_ch timesteps of input (e.g. W_ch=10)
  Per-channel: Linear(W_ch, d_ch) for each of C channels → (B, C, d_ch)
  + sinusoidal PE(channel_index)  — fixed, not learned
  → channel_tokens: (B, C, d_ch)

Fusion:
  h_refined = h_past + α · CrossAttn(Q=h_past, KV=channel_tokens)
  α: learnable scalar, init=0 → identity at start of training
```

Key choices:
- **No learnable sensor-ID embeddings** (V15 shortcut lesson)
- **Zero-init α** → model starts as current architecture, earns cross-channel
- **Small d_ch (64)** — channel tokens are cheap; C=14 → 14 KV pairs
- **Shared Linear(W_ch, d_ch)** across channels — no per-channel parameters
  except the sinusoidal PE (which is fixed)
- Uses raw sensor values, not encoded tokens — cross-channel sees the
  actual signal, not the temporal encoder's already-mixed representation

**Evaluation**: compare pretraining loss curves (cross-channel vs baseline),
then freeze both encoders and run identical pred-FT on FD001 (3 seeds).
Better representations → better AUPRC, regardless of label budget.

---

## Phase plan

| Phase | What | Est. time |
|-------|------|-----------|
| 0 | Infra: adapt anomaly_runner to use entity splits | 1h |
| 1 | **Anomaly pred-FT**: SMAP, MSL, SMD (3 seeds each) | 2h |
| 2 | Anomaly pred-FT: PSM, MBA (chrono split with gap) | 1h |
| 3 | Update RESULTS.md + paper.tex with real pred-FT numbers | 0.5h |
| 4 | Cross-channel: implement in models.py, pretrain FD001 | 2h |
| 5 | Compare: freeze both encoders, pred-FT FD001 (3 seeds) | 1h |
| 6 | If cross-channel helps: pretrain SMAP, run anomaly pred-FT | 1.5h |

**Critical path**: Phases 0-3 (~4.5h).  Phase 4-6 are the architecture work.

---

## Prior art on cross-channel (READ THIS — avoid repeating failures)

| Attempt | Result | Failure mode |
|---------|--------|-------------|
| V14 cross-sensor (iTransformer-style) | 14.98 frozen@100% (best ever), but std=10.19@20% | Sensor-ID embeddings → shortcut |
| V15 sensor-ID embeddings | loss→0 epoch 20, probe=75 (below random) | Model memorized per-sensor univariate dynamics |
| V15 bidirectional full-seq | Collapsed (anisotropy 1e15) | Shared prefix made prediction trivial |
| V18 cross-sensor + LogU k | Never executed (stretch goal) | — |

**Key insight**: V14's low-label regression was likely the sensor-ID shortcut,
not the cross-channel attention itself.  With fixed PE (no learnable sensor
embeddings) and zero-init α, the architecture should degrade gracefully.

---

## Ground rules

1. **Use `split_smap_entities()` / `split_msl_entities()` / `split_smd_entities()`** for anomaly pred-FT.  Never split the concatenated array chronologically.
2. Store surfaces as .npz.  Compute AUPRC (primary) + legacy metrics from surfaces.
3. Reporting: `mean ± std (Ns)`.  Decompose F1 → P + R.
4. Cross-channel block: **residual + zero-init α + fixed sinusoidal channel PE**.  NO learnable sensor-ID embeddings.
5. Commit + push hourly.
6. Update RESULTS.md after every phase.

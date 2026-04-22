# V22 Overnight Session — Fix Anomaly Pred-FT + Cross-Channel Encoder Variants

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: full overnight (~10-12 hours on A10G). Use ALL available time.
**Prereqs**: Read CLAUDE.md, experiments/RESULTS.md, v21 SESSION_PROMPT.md

---

## Three goals

### Goal A: Anomaly pred-FT with correct splits (critical, do first)

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

### Goal B: Pure iTransformer encoder (variant A — fixed window)

Replace the temporal transformer with an iTransformer-style encoder.
This is the clean, faithful iTransformer design adapted for causal JEPA.

```
Input: (B, T, C)  where T = fixed window (e.g. 100)

Step 1 — build per-channel tokens:
  Transpose: (B, T, C) → (B, C, T)
  Shared projection: Linear(T, d) → (B, C, d)
  + sinusoidal PE(channel_index)  — fixed, not learned
  Each channel's full temporal history becomes one d-dim token.

Step 2 — cross-channel self-attention (this IS iTransformer):
  (B, C, d) → Transformer (L=2, H=4, causal=False) → (B, C, d)
  Channels attend to each other. C=14 for CMAPSS, C=25 for SMAP.

Step 3 — pool to h_past:
  Mean-pool over C tokens → h_past (B, d)
```

**Constraint**: T must be fixed (Linear(T, d) is position-specific).
Use T=100 (our standard window).  Pad shorter inputs.

**Why this is clean**: no alternating layers, no sensor-ID embeddings,
no α scaling.  The encoder just IS an iTransformer on the causal window.
Temporal structure is in the Linear(T, d) projection; cross-channel
structure is in the self-attention.  ~Same param count as current encoder.

### Goal C: Hybrid encoder (variant B — variable-length compatible)

Keep the temporal causal transformer, add cross-channel attention after it.
Compatible with variable-length inputs (no fixed T requirement).

```
Input: (B, T, C)  — T can vary

Step 1 — temporal path (unchanged from current):
  Linear(C, d) → (B, T, d) → causal Transformer (L=2, H=4) → (B, T, d)

Step 2 — extract per-channel summaries from temporal output:
  Reshape temporal output: (B, T, d) back to per-channel view
  Use the LAST temporal token per channel? No — channels are mixed.
  
  Instead: project raw input at last timestep through per-channel lens:
  x_last: (B, C) → unsqueeze → (B, C, 1)
  Or use last W timesteps: (B, C, W) with W=10
  Shared Linear(W, d_ch) → (B, C, d_ch)
  + sinusoidal PE(channel_index)

Step 3 — cross-channel self-attention:
  (B, C, d_ch) → MultiheadAttention (1 layer, H=4) → (B, C, d_ch)

Step 4 — fuse with temporal representation:
  h_temporal = last token from step 1: (B, d)
  h_channel = mean-pool step 3: (B, d_ch)
  h_past = Linear(d + d_ch, d) applied to concat [h_temporal, h_channel]
```

**Why this is different from variant A**: preserves variable-length temporal
processing, adds cross-channel as a parallel stream.  More complex but
backward-compatible with existing pretraining infrastructure.

---

## Phase plan

| Phase | What | Est. time | Priority |
|-------|------|-----------|----------|
| 0 | Infra: adapt anomaly_runner to use entity splits | 1h | Critical |
| 1 | **Anomaly pred-FT**: SMAP, MSL, SMD (3 seeds each) | 2h | Critical |
| 2 | Anomaly pred-FT: PSM, MBA (chrono split with gap) | 1h | Critical |
| 3 | Update RESULTS.md + paper.tex with pred-FT numbers | 0.5h | Critical |
| 4 | **Variant A** (pure iTransformer): implement + pretrain FD001 | 1.5h | Important |
| 5 | **Variant B** (hybrid): implement + pretrain FD001 | 1.5h | Important |
| 6 | Compare: freeze all 3 encoders (baseline + A + B), pred-FT FD001 3 seeds | 1.5h | Important |
| 7 | Winner on FD001: pretrain on SMAP, run anomaly pred-FT with entity splits | 1.5h | If time |
| 8 | Update RESULTS.md with architecture comparison | 0.5h | If time |

**Critical path**: Phases 0-3 (~4.5h).
**Architecture work**: Phases 4-8 (~6.5h).
**Total**: ~11h.  Use all available time.  If a phase finishes early, start next.

---

## Prior art on cross-channel (READ THIS — avoid repeating failures)

| Attempt | Result | Failure mode |
|---------|--------|-------------|
| V14 cross-sensor (iTransformer-style) | 14.98 frozen@100% (best ever), but std=10.19@20% | Sensor-ID embeddings → shortcut |
| V15 sensor-ID embeddings | loss→0 epoch 20, probe=75 (below random) | Model memorized per-sensor univariate dynamics |
| V15 bidirectional full-seq | Collapsed (anisotropy 1e15) | Shared prefix made prediction trivial |
| V18 cross-sensor + LogU k | Never executed (stretch goal) | — |

**Key insight**: cross-channel attention is a PRETRAINING change.  The encoder
is frozen during finetuning, so downstream trainable params are identical.
V14's low-label regression was the sensor-ID shortcut, not cross-channel
attention itself.  With fixed sinusoidal channel PE, this should not recur.

**Hard rule**: NO learnable per-sensor/per-channel embeddings.  Use sinusoidal
PE(channel_index) or nothing.

---

## Evaluation protocol for encoder variants

For each encoder variant (baseline, A, B):
1. **Pretrain** on FD001 train set (early stopping, patience=5, max 50ep)
2. **Freeze encoder**, run pred-FT with BCE (3 seeds, 100% labels)
3. Report: pretraining loss curve, AUPRC, RMSE (legacy)
4. If AUPRC improves: repeat on SMAP with entity splits

The comparison is **representation quality**, not downstream capacity.
Same predictor, same head, same labels — only the frozen encoder differs.

---

## Ground rules

1. **Use `split_smap_entities()` / `split_msl_entities()` / `split_smd_entities()`** for anomaly pred-FT.  Never split the concatenated array chronologically.
2. Store surfaces as .npz.  Compute AUPRC (primary) + legacy metrics from surfaces.
3. Reporting: `mean ± std (Ns)`.  Decompose F1 → P + R.
4. **NO learnable sensor-ID / channel-ID embeddings.** Fixed sinusoidal PE only.
5. Commit + push hourly.  Tag each phase completion.
6. Update RESULTS.md after every phase.
7. **Use all available time.**  If phases finish early, expand seeds or add datasets.
8. If an encoder variant clearly fails during pretraining (loss doesn't decrease, collapse), document why and move on.  Don't debug for >30 min.

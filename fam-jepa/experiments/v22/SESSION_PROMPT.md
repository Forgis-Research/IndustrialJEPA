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

### Goal B: Cross-channel attention architecture (important)

Replace the flat per-timestep Linear(C, d) with a design that lets the model
attend across channels.  Must NOT regress at low labels (V14's failure mode).

**Architecture sketch — "temporal-first, then cross-channel"**:

```
Input: (B, T, C)

Step 1 — per-timestep channel mixing (lightweight):
  For each t:  x_t ∈ R^C  →  Linear(C, d)  →  token_t ∈ R^d
  (Same as current — cheap, preserves all channel info in one vector)

Step 2 — temporal causal transformer (unchanged):
  (B, T, d) → causal self-attention (L=2, H=4) → (B, T, d)
  Extract last token → h_past ∈ R^d

Step 3 — cross-channel refinement (NEW, small):
  Reshape input at time t: (B, C) → per-channel embed → (B, C, d_small)
  Cross-attend: h_past queries the C channel embeddings
  h_past_refined = h_past + CrossAttn(Q=h_past, KV=channel_embeds)
```

Key design choices to avoid V14/V15 failures:
- **No learnable sensor-ID embeddings** (caused shortcut learning in V15)
- Cross-channel block is a **refinement** on h_past, not a replacement
- Small d_small (32-64) to limit capacity at low labels
- Residual connection: if cross-channel attention learns nothing useful,
  h_past passes through unchanged (graceful degradation)

**Ablation**: run with and without cross-channel block at 100% and 5% labels
on FD001.  Must not regress at 5%.

---

## Phase plan

| Phase | What | Est. time |
|-------|------|-----------|
| 0 | Infra: adapt anomaly_runner to use entity splits | 1h |
| 1 | **Anomaly pred-FT**: SMAP, MSL, SMD (3 seeds each) | 2h |
| 2 | Anomaly pred-FT: PSM, MBA (chrono split with gap) | 1h |
| 3 | Update RESULTS.md + paper.tex with real pred-FT numbers | 0.5h |
| 4 | Cross-channel attention: implement + pretrain FD001 | 2h |
| 5 | Cross-channel ablation: 100% vs 5% labels, ±cross-attn | 1.5h |
| 6 | If cross-channel works: pretrain SMAP, run pred-FT | 1.5h |

**Critical path**: Phases 0-3 (~4.5h).  Phase 4-6 are the architecture work.

---

## Ground rules

1. **Use `split_smap_entities()` / `split_msl_entities()` / `split_smd_entities()`** for anomaly pred-FT.  Never split the concatenated array chronologically.
2. Store surfaces as .npz.  Compute AUPRC (primary) + legacy metrics from surfaces.
3. Reporting: `mean ± std (Ns)`.  Decompose F1 → P + R.
4. Cross-channel block: **residual + small capacity**.  No sensor-ID embeddings.
5. Ablation must include 5% labels.  If cross-channel regresses at 5%, report it honestly and keep the channel-fusion default.
6. Commit + push hourly.
7. Update RESULTS.md after every phase.

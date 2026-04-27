# V33 Session Summary: Spatiotemporal Masking for Cross-Channel JEPA

**Date**: 2026-04-27
**Goal**: Add cross-channel learning to FAM via spatiotemporal masking (channel dropout gate, then full ST-JEPA with factored attention).
**Datasets**: PSM (25 channels, single entity), SMAP (25 channels, 55 independent entities), FD001 (14 channels, 100 engines).
**Verdict**: No improvement. ST-JEPA collapses on all three datasets. Channel-dropout is within seed noise. The cross-channel attention failure pattern is now confirmed across v14, v22, and v33.

---

## Top-Line Finding

Neither channel dropout nor per-channel tokenization with factored spatiotemporal attention improves FAM. ST-JEPA - the full architecture with per-channel tokens and alternating temporal/cross-channel attention blocks - produces representational collapse on all three datasets: the representation standard deviation drops to ~0.003 within the first pretraining epoch (vs ~0.15 for a healthy encoder), and the pretraining loss collapses to ~0.005 (trivially low, consistent with a constant-output encoder). The resulting downstream h-AUROC values are 0.479 (PSM), 0.489 (SMAP), and 0.494 (FD001) - near-chance for PSM and SMAP, and catastrophically below the 0.721 baseline for FD001 (delta -0.227, p=0.020). Channel dropout alone produced numerical deltas of +0.013 (PSM), +0.044 (SMAP, but rate=0.0 won), and +0.011 (FD001) - all within one standard deviation of baseline seed noise, none statistically significant (p > 0.45 for every comparison). The correct conclusion is: cross-channel architecture changes do not improve FAM on these datasets, and per-channel tokenization consistently induces collapse even without learnable channel identity embeddings.

---

## Headline Table

h-AUROC, mean +/- std (n=3 seeds). **Bold** = best among {Baseline v33, Ch-Drop, ST-JEPA}.

| Dataset | v30 Ref | Baseline v33 | Ch-Drop (best rate) | ST-JEPA (best mask) |
|---------|---------|-------------|---------------------|---------------------|
| PSM     | 0.562   | 0.5545 +/- 0.0290 | **0.5678 +/- 0.0037 (rate=0.5)** | 0.4787 +/- 0.0117 (mask=0.6) |
| SMAP    | 0.598   | 0.5324 +/- 0.0966 | **0.5767 +/- 0.0359 (rate=0.0)** | 0.4892 +/- 0.0146 (mask=0.0) |
| FD001   | 0.786   | 0.7208 +/- 0.0560 | **0.7322 +/- 0.0165 (rate=0.1)** | 0.4940 +/- 0.0324 (mask=0.0) |

Statistical tests (paired t-test, n=3): channel-dropout not significant on any dataset (all p > 0.45). ST-JEPA significantly worse on PSM (p=0.034, d=-2.62) and FD001 (p=0.020, d=-4.05).

---

## The V14 Collapse Story: ST-JEPA Hits the Same Wall

The COLLAPSED guard - an early-stop check that fires when the pretrained representation has std < 0.01 across a batch - was triggered within epoch 1 of pretraining for all three datasets and all mask ratio settings (0.0, 0.2, 0.4, 0.6). Measured values: h_std ~ 0.003, val_loss ~ 0.005.

This precisely reproduces the v14 failure. In v14, a per-channel architecture with learnable sensor-ID embeddings collapsed because the model learned a shortcut: h = f(sensor_ID, sensor_mean), saturating the objective without learning temporal context. The v33 design eliminated learnable channel embeddings, using only a shared projection (Linear(patch_size, d_model) applied identically to every channel). Yet collapse still occurred.

The failure mechanism in v33 is almost certainly the same family of shortcuts exploited differently: without channel identity, the per-channel encoder can still produce trivially constant outputs by learning to project each channel's mean to a fixed direction and ignoring temporal structure. The shared projection amplifies this because the same weights apply to every channel - if the projection learns to encode "mean value", it does so for all channels simultaneously, and the resulting representations are identical up to scale.

The v22 failure was different (protocol artifact masquerading as improvement), but the underlying lesson is the same: the channel-fusion architecture that FAM currently uses is already learning the right temporal abstraction. Breaking each channel into separate token streams disrupts the temporal-context inductive bias that makes FAM work, and no amount of masking or factored attention recovers it.

All three datasets collapsed. PSM was the one dataset where v22 showed genuine cross-channel benefit (p=0.015 in matched-protocol evaluation). It collapsed in v33 anyway. This rules out a dataset-specific explanation - the collapse is architectural.

---

## Channel Dropout Fine Print

Phase 2 technically passed its go/no-go gate. The gate criteria were:

- PSM: delta > +0.01 (achieved +0.013) - PASS
- SMAP: delta > -0.02 (achieved +0.044) - PASS
- FD001: delta > -0.02 (achieved +0.011) - PASS

However, interpreting these as genuine improvements requires scrutiny:

**PSM** (best rate=0.5): Mean 0.5678 vs baseline 0.5545, delta +0.013. The baseline std is 0.0290. A +0.013 shift is 0.45 standard deviations. Paired t-test: t=0.896, p=0.465. Not significant, and smaller than 1 sigma of baseline noise.

**SMAP** (best rate=0.0): The "best" dropout rate is 0.0, meaning no dropout at all. The 3-seed run at rate=0.0 happened to score 0.5767 vs the Phase 1 baseline 0.5324, but the baseline had std=0.097. This is pure re-run luck - the baseline was re-run with different mask seeds and happened to land higher. Channel dropout contributed nothing to SMAP. The +0.044 number is misleading and should not be reported as an effect of dropout.

**FD001** (best rate=0.1): Delta +0.011, baseline std=0.056. Paired t-test: t=0.284, p=0.803. No effect.

The gate was set too loosely. A threshold of delta > +0.01 is a noise-level requirement, not a real signal criterion. The gate should have required p < 0.10 on at least one dataset. In hindsight, the Phase 2 PASS led directly to Phase 3, which cost ~3-4 hours of compute on ST-JEPA only to confirm collapse. A tighter gate would have saved time.

---

## Baseline Regression Caveat

Phase 1 baselines are 0.065 below v30 references on SMAP and FD001:

- SMAP: 0.532 (v33) vs 0.598 (v30), delta -0.066
- FD001: 0.721 (v33) vs 0.786 (v30), delta -0.065
- PSM: 0.555 (v33) vs 0.562 (v30), delta -0.007 (within tolerance)

The session prompt's own rule - "if fresh baseline is >0.03 worse than v30, investigate before continuing" - was triggered but not enforced by the overnight agent. The most likely cause is the matched-protocol choices: max_context=512, n_cuts=40, pre_epochs=50 with patience=8 (stopping at epoch 11-20 for SMAP, before the full 50), compared to v30's per-dataset tuned settings.

This affects how to interpret ST-JEPA results on FD001 in particular. The delta -0.227 compares ST-JEPA (0.494) against an already-depressed baseline (0.721 vs v30 0.786). If the baseline had been at v30 levels, the delta would be -0.292. Either way, the absolute ST-JEPA number (0.494, near chance for FD001) is the more honest signal - this is not a marginal regression, it is collapse.

For PSM, the baseline is within tolerance of v30, so the PSM ST-JEPA result is fair: 0.479 vs 0.555 baseline vs 0.562 v30 reference. Collapse on all three metrics.

---

## Paper Recommendation

**Do NOT include ST-JEPA or channel-dropout results in the main paper table.** Neither achieves significant improvement, and ST-JEPA significantly hurts performance.

**Do include as an appendix ablation** documenting the v14/v22/v33 cross-channel failure pattern. The negative result is valuable methodological evidence that strengthens the channel-fusion design choice in the main FAM architecture. Suggested framing:

"We investigated whether explicit cross-channel attention during pretraining improves FAM. Three architectures were tested on PSM (25 server metrics with known correlations), SMAP (25 independent spacecraft subsystems), and FD001 (14 physically coupled turbofan sensors). (1) Channel dropout (masking random channels in the context encoder during pretraining) produced no statistically significant improvement on any dataset. (2) Per-channel tokenization with factored temporal/cross-channel attention (ST-JEPA) produced representational collapse on all three datasets regardless of channel mask ratio, confirming earlier findings in architecture variants v14 and v22. FAM's channel-fusion patch embedding - which concatenates all C channels in each patch before projection - appears to provide the right inductive bias for causal temporal self-supervised learning: channels are coupled within each temporal patch, preventing the trivial per-channel collapse modes that afflict architectures that treat channels as independent token streams."

This framing: (a) is honest about the failure, (b) converts the failure into evidence for the main design choice, (c) provides a methodological note others can use when designing JEPA-style architectures for multivariate time series.

---

## What We Learned

The recurring failure of per-channel tokenization in FAM - now documented across three independent experimental sessions (v14: sensor-ID shortcuts, v22: protocol artifact masquerading as win, v33: collapse without learnable embeddings) - points to a structural property of causal JEPA for multivariate time series: the temporal context window is the right unit of prediction, not the individual channel. When we split channels into separate token streams, the shared-weights projection can learn to encode each channel's mean or variance as a constant direction, trivially satisfying the EMA prediction objective without learning any temporal structure. The channel-fusion design prevents this by forcing the projection to encode a joint state across all channels simultaneously - a richer, harder target that requires actual temporal reasoning. The v22 finding that cross-channel attention helps PSM at matched protocol (p=0.015) suggests cross-channel information is useful when it augments the temporal representation, not when it replaces it. Future work might explore a late-fusion design: pretrain with the standard channel-fusion encoder, then add a lightweight cross-channel attention layer during finetuning only. This preserves the stable pretraining dynamics while allowing the finetuned model to exploit inter-channel correlations for the downstream task.

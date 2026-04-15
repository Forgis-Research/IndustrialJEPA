# V16 Plan: Stabilized Architecture + Cross-Machine Generalization

**Session date**: 2026-04-16 (following V15)
**Context**: V15 confirmed bidirectional EMA collapses due to shared prefix between
context (x_{0:t}) and target (x_{0:t+k}). SIGReg is pending completion.

## Priority Ranking

Priority: P0=critical, P1=high, P2=medium, P3=low/aspirational.

---

## P0: Fix V15-EMA Collapse

**Problem**: V15-EMA collapses (PC1=0.777, anisotropy=1.9e15x) because the
bidirectional full-sequence encoder produces near-identical representations for
context (x_{0:t}) and target (x_{0:t+k}) - they share the prefix x_{0:t}.
The predictor learns to copy, not predict.

**Root cause**: Target is NOT informative beyond context when they share a prefix.
The causal V2 architecture avoids this: context=x_{0:t}, target=x_{t+1:t+k}
(NO shared prefix). The latent prediction task is genuinely non-trivial.

**Fix options** (ranked by simplicity):

1. **V16a - Causal target, bidirectional context**: Keep V2 target convention
   (target encoder sees only x_{t+1:t+k}), but use bidirectional encoder for
   context. This preserves the non-trivial prediction task while testing whether
   bidirectional context helps. Simplest change: replace ContextEncoder in V11
   with BidiTransformerEncoder from V15. Expected: no collapse, slight improvement.

2. **V16b - Disjoint segments**: Context=x_{0:t}, target=x_{t+k:t+2k} (no overlap).
   Harder prediction task, requires k >= min_gap (e.g., 5) to prevent trivial solutions.
   Risk: information gap too large for learning to converge.

3. **V16c - EMA with encoder-level SIGReg**: Keep full-sequence target, add
   SIGReg penalty directly on context encoder outputs h_t. Force embeddings to
   stay approximately Gaussian. More complex, but tests whether the collapse is
   correctable in-place.

**Recommendation for V16**: Start with V16a (bidirectional context, causal target).
Implement in 1h, run 3 seeds 200 epochs, compare frozen RMSE against V2 (17.81)
and V14-full-seq (15.70).

---

## P0: Validate V15-SIGReg (Awaiting Phase 1 Completion)

V15-SIGReg (3 seeds) is queued after V15-EMA in Phase 1. Results needed to
determine if SIGReg alone prevents collapse in bidirectional architecture.

**Expected**: SIGReg prevents collapse (isotropy penalty forces spread), but may
not beat V2 baseline due to information structure mismatch.

**If V15-SIGReg beats V2 (< 17.81)**: Architecture is sound, SIGReg is key.
Scale to V16.

**If V15-SIGReg fails (>= 17.81)**: Fix prefix-sharing problem (V16a) first.
SIGReg is insufficient alone.

---

## P1: Phase 2 - Cross-Sensor Encoder (FIX REQUIRED)

**NEGATIVE RESULT FROM V15**: Sensor ID embeddings cause shortcut learning.
V15 Phase 2 showed: loss=0.0014 at epoch 20 (shortcut) + probe=75.41 (worse than random).
Root cause: sensor ID embeddings let model predict future sensor values from identity
alone, without encoding temporal degradation context.

**V16 fix**: Remove sensor ID embeddings. Use:
- Relative positional encoding across sensors (based on correlation structure from Phase 4)
- Or no sensor-specific encoding (let cross-attention learn it implicitly)
- Keep sensor dropout 20% (good regularization, but remove ID shortcut)

**Alternative**: Re-run V14 cross-sensor architecture (no ID embeds) with sensor dropout only.
This isolates the benefit of dropout without the ID shortcut. Expected: small improvement
over V14 baseline (14.98 -> ~14.5 with dropout) without shortcut issues.

V14 cross-sensor reference: frozen=14.98+/-0.22

---

## P1: Phase 5a - TTE Probe with V15 Encoder

Phase 5a done with V14 encoder: frozen TTE probe RMSE=37.02, WORSE than hand
features (32.98). V14 encoder optimized for RUL, not TTE.

**For V16**: After training stable encoder (V16a), run TTE probe. Hypothesis: a
model that better captures temporal dynamics (bidirectional + stable) may better
encode the rate-of-change signal relevant to threshold exceedance.

**Also needed**: Add s14 exceedance as an auxiliary pretraining signal. If we
include "time to s14 threshold" in the pretraining loss, the encoder should
encode TTE-relevant features. This would require adding a TTE head to pretraining.

---

## P1: SMAP Anomaly - Longer Pretraining

Phase 3 SMAP result: non-PA F1=0.069 (random baseline=0.071) after only 20 epochs.
The model did NOT beat random baseline on non-PA F1.

**Root cause**: 20 epochs on 20K samples is insufficient. The anomaly score
distribution was near-constant (mean=0.838, std=0.039), suggesting the model
hasn't learned to distinguish normal from anomalous patterns.

**V16 fix**: Run 100 epochs on full 135K SMAP training set. Expected runtime: ~30
min. This should give the encoder enough signal to differentiate normal windows.

**Alternative approach**: Use point-contrastive learning (normal vs. injected
anomalies during pretraining) rather than pure prediction. This provides explicit
anomaly signal.

---

## P2: Cross-Machine Generalization (FD001 -> FD002, FD003, FD004)

V12 JEPA showed FD002 performance degrades significantly (RMSE gap ~8-10 cycles)
due to multiple operating conditions. V15 bidirectional encoder may generalize
better if representations are condition-agnostic.

**V16 experiment**:
1. Pretrain on FD001 only.
2. Evaluate frozen probe on FD002, FD003, FD004 (zero-shot transfer).
3. Compare V2 baseline transfer vs V16a transfer.
4. Hypothesis: bidirectional encoder with SIGReg creates more disentangled
   representations, transfers better across operating conditions.

---

## P2: SWaT Dataset

SWaT adapter (swat.py) is implemented but data requires registration at:
  https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/

Register and obtain data before V16 session. Expected: ~1-2 day approval.
SWaT has 51 sensors, 177K attack timesteps, 41 attack events.

---

## P3: Degradation Clock Visualization (Phase 1d)

The `plot_degradation_clock()` function was implemented and is called at end of
Phase 1, but Phase 1 is still running. This will complete during Phase 1.

For V16: Add t-SNE visualization in addition to PCA. Check whether the degradation
trajectory forms a smooth manifold (clock-like) vs. scattered clusters.

---

## Technical Debt to Address

1. **Mask convention documentation**: collate_finetune uses True=padding. Document
   this clearly in a comment at the top of data_utils.py. The confusion caused a
   bug during V15 (mask flip bug, subsequently fixed).

2. **EP-SIGReg uses linear quadrature**: The quadrature grid is equally-spaced
   in [0, t_max], not Gauss-Hermite. The EP test result is qualitatively correct
   but not exactly the EP statistics from the paper. For V16 paper submission,
   use scipy.integrate.quad for more accurate integration.

3. **Phase 1d degradation clock**: plot_degradation_clock() has a bug in
   extract_all_embeddings() (returns None, None, None placeholder). The
   extract_embeddings_simple() function works correctly. Fix the broken function
   or remove it.

---

## V16 Priority Execution Order

```
Hour 1-2:  Implement V16a (bidir context + causal target). Smoke test.
Hour 2-5:  Run V16a 3 seeds 200 epochs (V15-EMA replacement). Collect RMSE.
Hour 5-7:  Run phase2_cross_sensor_improved.py (sensor dropout + ID embeds).
Hour 7-8:  Run SMAP 100 epochs on full data. Collect non-PA F1.
Hour 8-9:  Cross-machine generalization eval (FD001 -> FD002/FD003/FD004).
Hour 9-10: Update paper appendix with V16 results. Write RESULTS.md. Commit/push.
```

---

## Success Criteria for V16

| Metric | V2 baseline | V16 target | Stretch goal |
|--------|-------------|------------|--------------|
| FD001 frozen RMSE | 17.81 | <16.0 | <15.0 |
| FD001 E2E RMSE | 14.23 | <13.5 | <12.5 |
| SMAP non-PA F1 | 0.07 (random) | >0.10 | >0.15 |
| TTE nRMSE | 0.218 (V14) | <0.150 | <0.120 |
| Cross-machine: FD002 frozen | ~28 (V2 ref) | <25 | <22 |

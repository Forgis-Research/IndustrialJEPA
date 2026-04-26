# V17 Results

Session: 2026-04-18 (overnight ~1h25m compute + ~30m eval)
Hardware: A10G (shared — had to restart Phase 1 once).
Dataset: C-MAPSS FD001 (Phases 1-4), SMAP (Phase 5).

## Summary

| Phase | Task / variant | Test RMSE | F1 (binary) | AUC-PR |
|-------|----------------|-----------|-------------|--------|
| 1 | V17 baseline (h_past, LogU k, w=10) | 20.08 +/- 1.18 | 0.829 @ k=30 | 0.926 |
| 2 | Probe on h_past (WD 1e-2, multi-cut val) | **15.53 +/- 1.68** | **0.919 @ k=30** | **0.974** |
| 2 | Probe on concat [h, γ(5..100)] (1536-d) | 21.92 +/- 9.40 | 0.860 | 0.932 |
| 2 | Probe on γ(5) alone | 18.62 +/- 1.38 | 0.855 | 0.936 |
| 2 | Probe on γ(10) alone | 18.39 +/- 1.49 | 0.847 | 0.931 |
| 2 | Probe on γ(20) alone | 18.03 +/- 2.01 | 0.876 | 0.933 |
| 2 | Probe on γ(50) alone | 17.67 +/- 2.38 | 0.869 | 0.936 |
| 2 | Probe on γ(100) alone | 17.93 +/- 1.99 | 0.869 | 0.937 |
| 3 | Curriculum SIGReg on encoder | 22.43 +/- 1.50 | 0.807 @ k=30 | 0.930 |
| 3 | **Curriculum SIGReg on predictor** | **15.38 +/- 1.08** | **0.891 @ k=30** | **0.952** |
| 4 | TTE probe + k-sweep (s14, 3-sigma) | TTE RMSE 99.38 | 0.079 @ k=30 | 0.430 |
| 5 | SMAP anomaly (50ep EMA, multi-k score) | N/A | **non-PA 0.038 / PA 0.219** | 0.097 |

**References:**
V2 baseline test RMSE on FD001 : 17.81 (5 seeds, V14 establishment).
MTS-JEPA SMAP PA-F1            : 0.336.

## Takeaways

1. **V17 baseline beats V2 on frozen-probe RMSE** when probe regularization is honest.
   Phase 2 h_past probe achieves 15.53 (V2: 17.81). Phase 1's 20.08 reflects a
   misleading probe protocol (overfit on small val set). Fix: weight decay 1e-2
   + multiple val cuts.

2. **Trajectory concatenation hurts RUL**. Concatenated γ(5..100) features (1536-d)
   degrade the probe with only ~500 training samples. Individual γ(k) probes
   match h_past within 2-3 RMSE points but don't beat it. *The predictor at
   inference is most valuable for event-specific queries, not for regression.*

3. **SIGReg placement matters** (Phase 3). Applying SIGReg to h_past pushes the
   encoder into near-isotropy (PC1=0.22) but *destroys* the RUL gradient
   (+7 RMSE). Applying SIGReg to predictor output (γ(k)) preserves h_past's
   usefulness (15.38 RMSE, matching Phase 2 h_past). The curriculum successfully
   drops the EMA target network in the final 50 epochs — pred variant.

4. **TTE via trajectory sweep is weak** (F1 0.08). The predictor was trained on
   generic reconstruction, not on s14-threshold events. A task-aware predictor
   or joint pretraining would be required.

5. **SMAP anomaly detection fails again** (non-PA 0.038). Bug fixes from V16
   (argument order, EMA target) plus shorter training (50 ep) and LogU k were
   insufficient. Root cause is structural: SMAP anomalies are recurrent
   patterns a well-trained JEPA learns to predict — so prediction error score
   is *anti-correlated* with labels (gap = -0.61 in favor of normals). A
   different scoring function (sensitivity, spectral) is needed.

## Methodology notes / gotchas

- Phase 1 got killed mid-run by an unrelated process on the shared GPU. Seed 42
  completed under the original script (`phase1_v17_baseline.py`); seeds 123 and
  456 ran under `phase1_restart_seeds.py` which merged results.
- Probe overfitting on the 15-engine val set caused misleadingly low val RMSE
  (0.1-1.0) in the first Phase 2 attempt. Fixed with weight_decay=1e-2 and
  n_cuts_per_engine=10 on val; now val and test RMSE track each other.
- Phase 4 original implementation evaluated TTE only at the last cycle of each
  test engine -> n_valid=0 (TTE is NaN past the first exceedance by construction).
  Fixed to sample multiple cycles per test engine.
- Phase 3 SG-mode spike at ep151: prediction loss jumps 4x when switching from
  EMA target_encoder to context_encoder targets. The network recovers over
  ~20 epochs. PC1 keeps dropping throughout curriculum.
- All ckpts saved in `experiments/v17/ckpts/`.

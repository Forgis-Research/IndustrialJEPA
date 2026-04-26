# V25 Session — Discrete Hazard CDF + Dense Horizons

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v24 SESSION_SUMMARY.md

---

## What changed since v24

### Discrete hazard → CDF parameterization (in model.py)

`finetune_forward()` now returns CDF probabilities, not independent logits.
The event head produces per-interval hazard logits; sigmoid gives
conditional hazards λ_k; cumprod gives the survival function; 1 - survival
gives the CDF.

```python
hazard_logits = self.event_head(h_pred)          # (B, K)
lambdas = torch.sigmoid(hazard_logits)            # (B, K) ∈ (0,1)
survival = torch.cumprod(1 - lambdas, dim=-1)     # non-increasing
cdf = 1 - survival                                # non-decreasing
```

**Monotonicity is guaranteed by construction.** Zero violations by design.
No post-hoc cummax enforcement.

v24 had 7–25% monotonicity violation rates on anomaly/medical datasets.
This fix eliminates them structurally.

### train.py loss change

Loss changed from `BCEWithLogitsLoss` (on independent logits) to manual
pos-weighted BCE on CDF probabilities. evaluate() no longer applies
sigmoid — finetune_forward already returns probabilities.

---

## Session goals

### Goal A: Re-run v24 benchmark with hazard CDF (Critical)

Re-pretrain + pred-FT on all 12 datasets with the hazard parameterization.
Compare AUPRC to v24 (independent logits). Expect:
- C-MAPSS: minimal change (0% violations in v24)
- Anomaly/medical: improvement (7–25% violations in v24 are now eliminated)

3 seeds per dataset. Same protocol as v24.

### Goal B: Dense horizon evaluation (Important)

Evaluate at every Δt from 1 to max_horizon (not just 7 sparse points).
The predictor takes continuous Δt — just run it at 150 values instead of 7.

This gives:
- Smooth probability surfaces for visualization
- More accurate AUPRC (150 cells per timestep vs 7)
- Better monotonicity verification (every consecutive pair checked)

**During finetuning**: still train on sparse horizons (7 points) for
compute efficiency. The model interpolates to dense horizons at eval time.

**During evaluation**: run predictor at Δt = 1, 2, 3, ..., 150 for C-MAPSS
or Δt = 1, 2, 3, ..., 200 for anomaly datasets. Store full dense surface.

### Goal C: Notebook with proper surface heatmaps (Always)

With dense horizons, the heatmaps will be smooth with real data (not
interpolated). Use log-scaled y-axis (pcolormesh) to show horizon
structure honestly.

---

## Ground rules

1. **Import from model.py and train.py.** The hazard CDF is already
   implemented — do NOT modify the model.
2. **finetune_forward returns probabilities (CDF), not logits.** Do NOT
   apply sigmoid after calling it.
3. Store surfaces as .npz. Dense surfaces will be larger (~20× more
   columns) but still manageable.
4. Commit + push after each phase. Update RESULTS.md after every phase.
5. Compare v25 vs v24 explicitly in the notebook.

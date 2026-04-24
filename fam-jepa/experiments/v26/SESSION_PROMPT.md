# V26 Session — Hazard CDF Benchmark + Architecture Figure

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: 8-10 hours on A10G. Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`, v24 SESSION_SUMMARY.md

---

## What changed since v24

### Discrete hazard CDF (in model.py, already committed)

`finetune_forward()` now returns CDF probabilities, not independent logits.

```python
hazard_logits = self.event_head(h_pred)          # (B, K)
lambdas = torch.sigmoid(hazard_logits)            # (B, K) in (0,1)
survival = torch.cumprod(1 - lambdas, dim=-1)     # non-increasing
cdf = 1 - survival                                # non-decreasing
```

**Monotonicity is guaranteed by construction.** v24 had 7-25% violation
rates on anomaly/medical datasets. This fix eliminates them structurally.

### train.py loss change

Loss changed from `BCEWithLogitsLoss` on independent logits to manual
pos-weighted BCE on CDF probabilities. `evaluate()` no longer applies
sigmoid --- `finetune_forward` already returns probabilities.

**IMPORTANT**: do NOT apply sigmoid after `finetune_forward`. It returns
probabilities, not logits.

---

## Datasets (9 total, no Sepsis)

Focus on the core benchmark. Sepsis is deferred (hourly data requires
P=1 exception, distracts from the main story).

| Dataset | Domain | Channels | P | Context | Horizons |
|---------|--------|----------|---|---------|----------|
| FD001 | Turbofan | 14 | 16 | Full history | {1,5,10,20,50,100,150} |
| FD002 | Turbofan | 14 | 16 | Full history | {1,5,10,20,50,100,150} |
| FD003 | Turbofan | 14 | 16 | Full history | {1,5,10,20,50,100,150} |
| SMAP | Spacecraft | 25 | 16 | Sliding 512 | {1,5,10,20,50,100,150,200} |
| MSL | Spacecraft | 55 | 16 | Sliding 512 | {1,5,10,20,50,100,150,200} |
| PSM | Server | 25 | 16 | Sliding 512 | {1,5,10,20,50,100,150,200} |
| SMD | Server | 38 | 16 | Sliding 512 | {1,5,10,20,50,100,150,200} |
| MBA | Cardiac | 2 | 16 | Sliding 512 | {1,5,10,20,50,100,150,200} |
| PhysioNet 2012 | ICU mortality | 36 | 16 | Full stay | {1,2,3,6,12,24,48} |

All loaders: `normalize=False` (RevIN handles it).

---

## Phase plan

### Phase 1: Sanity check (15 min)

Quick 3-epoch pretrain + finetune on FD001 with the hazard CDF.
Verify:
- `finetune_forward` returns probabilities in (0, 1)
- Monotonicity violations = 0
- Loss decreases
- No NaN/Inf

### Phase 2: C-MAPSS FD001/002/003 (1.5 h)

Pretrain + pred-FT, 3 seeds each. Compare to v24.
Expected: C-MAPSS had 0% violations in v24, so hazard CDF should
give similar results. Any improvement comes from better gradient flow
(cumprod couples all horizons).

### Phase 3: SMAP, MSL, PSM, SMD, MBA (3 h)

Pretrain + pred-FT, 3 seeds each. Compare to v24.
Expected: these had 7-25% violations in v24. The hazard CDF should
improve AUPRC by eliminating wasted capacity on learning monotonicity.

### Phase 4: PhysioNet 2012 (1 h)

Pretrain + pred-FT, 3 seeds. Compare to v24 (AUROC 0.858).

### Phase 5: Dense horizon evaluation (1 h)

For FD001, SMAP, and MBA: re-evaluate the best checkpoint at
dense horizons (every integer Δt from 1 to max_horizon).

- FD001: Δt = 1, 2, 3, ..., 150
- SMAP: Δt = 1, 2, 3, ..., 200
- MBA: Δt = 1, 2, 3, ..., 200

Store dense surfaces. These give smooth heatmaps and more accurate AUPRC.
**Dense evaluation only** --- still train on sparse horizons.

### Phase 6: Chronos-2 comparison (1 h)

Re-run the Chronos-2 baseline on the same datasets (reuse v24 code).
Ensure comparison is on the same test splits.

### Phase 7: Architecture figure (30 min)

Check if `paper-neurips/figures/fig_architecture.tex` was created.
If so, compile it and verify the PDF looks correct.
If not, the TikZ source should be created manually showing:

```
Pretraining:
  x → RevIN → Patch(P=16) → Causal Transformer → h_t
                                                    ↓
  x(t:t+Δt] → same tokenizer → Bidir Transformer → h* (EMA)
                                                    ↓
  Predictor(h_t, Δt) → ĥ ----L1 loss---- h*

Finetuning (encoder frozen):
  x → encoder → h_t → Predictor(Δt₁..Δt_K) → Event Head
                         → λ_k (hazards) → cumprod → CDF p(t, Δt)
```

Use TikZ. Clean, minimal, 2-3 colors. NeurIPS column width.

### Phase 8: PA-F1 from surfaces (15 min)

Compute PA-F1 from v26 surfaces for literature comparability.

### Phase 9: Update RESULTS.md + Quarto notebook (30 min)

- v26 section in RESULTS.md
- `notebooks/26_v26_analysis.qmd` with:
  - v26 vs v24 comparison table
  - Monotonicity violation comparison (should be 0 everywhere now)
  - Dense surface heatmaps (log y-axis, actual Δt values)
  - Per-horizon AUPRC curves
  - Chronos-2 comparison
- Render: `quarto render notebooks/26_v26_analysis.qmd`

---

## Phase priorities

| Phase | What | Est. | Priority |
|-------|------|------|----------|
| 1 | Sanity check | 15 min | BLOCKING |
| 2 | C-MAPSS | 1.5 h | Critical |
| 3 | Anomaly + MBA | 3 h | Critical |
| 4 | PhysioNet 2012 | 1 h | Important |
| 5 | Dense horizons | 1 h | Important |
| 6 | Chronos-2 | 1 h | Important |
| 7 | Architecture figure | 30 min | Important |
| 8 | PA-F1 | 15 min | Easy |
| 9 | RESULTS.md + notebook | 30 min | Always |

**Total**: ~9h.

---

## Ground rules

1. **Import from model.py and train.py.** Do NOT copy model code.
2. **finetune_forward returns probabilities (CDF), not logits.** Do NOT
   apply sigmoid. Do NOT use BCEWithLogitsLoss.
3. **All loaders: normalize=False.** RevIN handles normalization.
4. **P=16 everywhere.** No exceptions in this session.
5. **Minimum context = 128 timesteps** (8 tokens). Enforced by EventDataset.
6. Stride=1 at evaluation, stride=4 at training.
7. Store surfaces as .npz. Compute AUPRC + legacy from surfaces.
8. Reporting: `mean +/- std (Ns)`.
9. Commit + push after each phase. Update RESULTS.md after every phase.
10. If FD001 AUPRC drops below 0.90, stop and debug.

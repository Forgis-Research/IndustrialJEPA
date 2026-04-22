# V13 Overnight Session

**Goal**: Close open loopholes from v12, then close the JEPA-STAR gap.

**Commit cadence**: ~10 commits over the session. Commit + push after each
phase completes and after any result that took >30 min of compute. Do NOT
batch everything to the end — the VM can crash at any time.

**W&B**: Log every training run to `wandb.init(project="industrialjepa")`.
Log VRAM/RAM/disk every 60s via `psutil`. Tag runs with `v13-phase{N}-{name}`.

All outputs under `mechanical-jepa/experiments/v13/`. Do NOT modify v11 or
v12 artifacts.

---

## What v12 established (do not re-run)

- V11 is real: H.I. R²=0.926, rho=0.830, shuffle +41.5, regressor margin 5.4
- FD002 gap is distribution shift (conditions 1/2/5 overrepresented in test)
- 17-channel global normalization failed catastrophically on FD002
- Frozen tracks better (rho 0.856) than E2E (0.804); E2E calibrates, doesn't detect
- V2 (d=256, L=2) is the primary config: E2E 14.23, frozen 17.81, STAR 12.19

---

## Phase 0 — Carry-over from v12 crash (RUN FIRST, gates everything)

### 0a. STAR label-efficiency sweep [background, ~3-4h]

Launch FIRST, runs unattended while other phases execute.

Using `paper-replications/star/run_experiments.py` with FD001 config:
- Label budgets: 50%, 20%, 10%, 5% (100% already done = 12.19)
- 5 seeds: 42, 123, 456, 789, 1024
- Output: `experiments/v13/star_label_efficiency.json`

**Kill criterion**: if STAR@20% <= 14 RMSE, the label-efficiency pitch
is dead. Paper pivots to H.I. recovery headline exclusively. If
STAR@20% > 16, the pitch is strong. Between 14-16: survives but weakened.

### 0b. STAR FD004 [background, ~5-7h]

Also crashed. Complete the 5-seed FD004 sweep with FD004-specific
hyperparams (bs=64, w=64, scales=4, dm=64, nh=4).
Output: `experiments/v13/star_fd004_results.json`

Launch alongside 0a if GPU memory allows, otherwise queue after.

### 0c. From-scratch ablation [~1h]

**The most important experiment in v13.** Quantifies the pretraining
contribution under E2E fine-tuning.

Same V2 transformer encoder (d=256, L=2, same param count) + same
linear probe + same E2E protocol (LR=1e-4, AdamW, patience=20) — but
initialized from **random weights** instead of the pretrained checkpoint.

Run at 4 label budgets: 100%, 20%, 10%, 5%.
5 seeds each: 42, 123, 456, 789, 1024.

```python
# The only difference: skip checkpoint loading
model = TrajectoryJEPA(d_model=256, n_layers=2, ...)  # random init
probe = RULProbe(model.d_model)
# Same optimizer, same training loop, same eval as V11 E2E
```

Output: `experiments/v13/from_scratch_ablation.json`

**Interpretation**:
- delta > 3 RMSE → pretraining does real work under E2E (strong SSL claim)
- delta 1-3 RMSE → helps modestly (paper leads with frozen/H.I., not E2E)
- delta < 1 RMSE → negligible (E2E is supervised learning in a transformer)

If the delta grows as labels decrease, that's the pitch: "pretraining
matters most when labels are scarce." Report the delta at each budget.

### 0d. Length-vs-content ablation [~15 min, inference only]

Disentangles whether the encoder reads sensor degradation or just
encodes sequence length via positional encoding. Three tests, all
inference-only on the frozen V2 encoder — no training needed.

**Test 1: Constant input.** Feed sequences where every row is identical
(first cycle's sensors repeated t times). Vary t = 30, 50, 80, 110,
140, 170, 200. If predictions change with t, the PE is doing the work.
If predictions are constant, the encoder reads content not length.

**Test 2: Length-matched cross-engine swap.** Pick 10 engine pairs with
similar total length (within 10 cycles). At a shared cut point t,
compute h_past for both. Same length, different sensors. Report cosine
similarity and probe output difference. If h_past is similar despite
different sensor content, length dominates.

**Test 3: Temporal shuffle (strongest).** For each test engine, randomly
permute the temporal order of sensor rows (keep length and value set
identical, destroy temporal structure). Report rho_median and RMSE vs
original. If rho collapses, the encoder reads temporal degradation
patterns. If rho holds, it just counts.

Output: `experiments/v13/length_vs_content_ablation.json`

**If the encoder fails** (primarily encodes length), the H.I. R²=0.926
is a length artifact and the paper narrative needs fundamental
rethinking. Flag this prominently and STOP before Phase 1.

### Phase 0 decision point

Read all Phase 0 results before proceeding. Three gates:

1. STAR@20% result → determines label-efficiency narrative
2. From-scratch delta → determines whether E2E is an SSL claim
3. Length-vs-content → determines whether representation claims hold

If any gate fails, write up the failure in RESULTS.md and adjust
Phase 1 priorities accordingly. Do not blindly proceed.

**COMMIT + PUSH here.** This is the most important checkpoint of the session.

---

## Phase 1 — Close the JEPA-STAR gap (only if Phase 0 clears)

The gap is JEPA E2E 14.23 vs STAR 12.19 (~2 RMSE). Three hypotheses.

### 1a. Warmup-freeze fine-tuning [~1h]

Freeze encoder for first 20 epochs (probe-only warmup), then unfreeze
for E2E. Rationale: letting the probe converge first prevents early
gradient noise from wrecking pretrained encoder weights.

5 seeds, 100% labels, FD001. Compare vs standard E2E (14.23).

### 1b. Weight decay [~1h]

Add L2 weight decay (1e-4) to E2E. Addresses high variance at 5% labels
(E2E std=5.1 at 5% vs frozen std=2.0).

5 seeds, 100% + 5% labels, FD001.

### 1c. Longer prediction horizon [~3h]

Pretrain from scratch with k in [5, 50] (vs current [5, 30]). Predicting
further ahead forces the encoder to learn slower dynamics. Requires new
pretraining run (200 epochs) + full fine-tuning sweep.

Kill criterion: if probe RMSE after new pretraining doesn't improve vs
current 19.22, horizon isn't the bottleneck.

### 1d. Deeper architecture (V4: d=256, L=4) [~4h]

Scale from 2 to 4 transformer layers (~3.5M params). Requires new
pretraining + fine-tuning. Compare frozen and E2E at 100%.

Kill criterion: if V4 frozen doesn't improve over V2 frozen, depth
doesn't help (consistent with v11 finding that width > depth).

### Phase 1 decision point

If any experiment produces E2E < 13.0 on FD001 (5 seeds), adopt that
config. If none improve by > 0.5 RMSE, the gap to STAR is architectural
(STAR uses hierarchical patch merging) not a fine-tuning/pretraining issue.

**COMMIT + PUSH after Phase 1.**

---

## Phase 2 — FD002 fix (lower priority, run if time permits)

### 2a. Condition token

Keep 14 sensor channels with per-condition KMeans normalization. Prepend
a learnable condition embedding (6-way, looked up from KMeans cluster ID)
to each sequence. This lets the encoder know which operating regime it's
in without overloading the sensor channels.

Pretrain on FD002, fine-tune frozen + E2E at 100%, 5 seeds.
Target: frozen FD002 RMSE < 20 (vs current 26.33).

---

## Success criteria

| Criterion | Target | Current |
|:---|:---|:---|
| From-scratch delta @ 100% | > 3 RMSE | unknown |
| Length-vs-content | encoder reads content | unknown |
| STAR label sweep | STAR@20% > 16 | unknown |
| E2E FD001 (Phase 1 best) | < 13.5 | 14.23 |
| FD002 frozen (Phase 2) | < 20 | 26.33 |

---

## Execution timeline

```
T+0:00  Launch Phase 0a (STAR sweep) + 0b (STAR FD004) in background
T+0:00  Run Phase 0d (length-vs-content, 15 min)
T+0:15  Run Phase 0c (from-scratch ablation, ~1h)
T+1:15  Phase 0 decision point — COMMIT + PUSH
T+1:15  Start Phase 1a (warmup-freeze, 1h)
T+2:15  Start Phase 1b (weight decay, 1h)
T+3:15  Phase 0a (STAR sweep) likely done — collect + COMMIT + PUSH
T+3:15  Start Phase 1c (longer horizon, 3h) or 1d (deeper arch, 4h)
T+6:00  Phase 1 decision point — COMMIT + PUSH
T+6:00  Start Phase 2a (FD002 condition token) if time permits
T+8:00  Session wrap-up — COMMIT + PUSH + write RESULTS.md
```

---

## Commit protocol

Target ~10 commits over the session:
1. After Phase 0d (length-vs-content)
2. After Phase 0c (from-scratch ablation)
3. Phase 0 decision point (RESULTS.md update)
4. After Phase 0a results collected (STAR sweep)
5. After Phase 1a
6. After Phase 1b
7. After Phase 0b results collected (STAR FD004)
8. After Phase 1c or 1d
9. Phase 1 decision point
10. Session wrap-up (RESULTS.md final, EXPERIMENT_LOG.md)

Push after every 1-2 commits. Never let >2 unpushed commits accumulate.

---

## One-sentence success criterion

**By morning, we know whether pretraining contributes under E2E, whether
the encoder reads sensors or counts timesteps, whether STAR kills the
label-efficiency pitch, and whether any fine-tuning variant closes the
gap to STAR — and all of this is committed and pushed.**

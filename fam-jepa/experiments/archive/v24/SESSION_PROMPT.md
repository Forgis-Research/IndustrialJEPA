# V24 Session — Canonical Architecture First Run

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: 8–10 hours on A10G. Use ALL available time.
**Prereqs**: Read CLAUDE.md, `fam-jepa/ARCHITECTURE.md`, `fam-jepa/model.py`,
`fam-jepa/train.py`, `experiments/RESULTS.md`

---

## What changed (read first)

V24 is the first run with the **canonical architecture** (`model.py` + `train.py`).
All previous experiment scripts (v11–v23) used ad-hoc model definitions scattered
across experiment directories. V24 uses a single clean codebase.

Key changes from v22/v23:

1. **Cumulative target interval**: target = x(t : t+Δt]. No skip-gap.
   Pretraining and finetuning use the same interval semantics.
2. **P=16 globally**: one patch size for all datasets (including C-MAPSS).
3. **RevIN**: per-context, per-channel instance normalization (replaces
   global train-set normalization).
4. **Variable-length context**: full history for C-MAPSS/Sepsis, sliding
   max 512 for anomaly streams. Padding + mask.
5. **Stride=1 at evaluation**: dense probability surface.
6. **d_model=256, d_ff=256, predictor_hidden=256**: ~2.16M params total.
   Matches the v11–v22 param budget.

Architecture defined in: `model.py` (components) and `train.py` (loops).
**Do NOT create new model files.** Import from these.

---

## What is novel vs off-the-shelf

**Off-the-shelf (don't reinvent):**
- Patch tokenization P=16 (PatchTST, MOMENT standard)
- RevIN instance normalization (Kim et al. 2022)
- Causal transformer encoder with sinusoidal PE
- EMA target encoder (I-JEPA, V-JEPA)
- L1 loss on L2-normalized representations + variance regularizer
- AdamW + cosine schedule + early stopping

**Our contributions (these are what the paper is about):**
- **Predictor finetuning**: freeze encoder, finetune predictor + sigmoid head.
  One architecture across N datasets and M event types.
- **Cumulative interval pretraining**: target = x(t : t+Δt], same interval
  used downstream for p(t, Δt) = P(event in (t, t+Δt]).
- **Probability surface as CDF**: p(t, Δt) is non-decreasing in Δt.
  Interval arithmetic: P(event in (t+a, t+b]) = p(t,b) − p(t,a).
- **Multi-domain benchmark**: turbofan, spacecraft, server, cardiac, ICU —
  same architecture, same hyperparameters, different data.

---

## Minimum context length

**128 timesteps minimum** (8 tokens at P=16). Below this the transformer
degenerates. Enforce in data loading:
- C-MAPSS: shortest engines are ~128 cycles — right at the boundary. Pad
  if shorter.
- Anomaly datasets: sliding context of 512, always above minimum.
- Sepsis: **use P=1** (hourly data, most stays < 128 hours). This is the
  one exception to P=16. The paper says "P=16 for all datasets with
  sufficient temporal resolution."

---

## Phase 0: Read the dataset reference (5 min)

Read `experiments/v24/DATASETS.md` — comprehensive characterization of all
9 datasets including: scale, event types, label definitions, SOTA methods
and metrics, potential problems, and data loader notes.

Key things to check in Phase 0:
- All data loaders must pass `normalize=False` (RevIN handles normalization)
- SMAP/MSL/SMD use intra-entity splits; PSM/MBA use chronological with gap
- Sepsis uses P=1 (exception), all others P=16
- C-MAPSS shortest engines (128 cycles) = 8 tokens at P=16 — right at minimum

---

## Phase plan

### Phase 1: Sanity check (30 min)

Verify `model.py` and `train.py` work on the VM:
```python
from model import FAM
from train import PretrainDataset, collate_pretrain, pretrain
# Quick 3-epoch pretrain on FD001, check loss decreases and no crash
```
Fix any import issues. Commit.

### Phase 1: C-MAPSS FD001 pretrain + pred-FT (1.5 h)

**Pretrain** (3 seeds):
- Data: `data_utils.load_cmapss_subset('FD001')` → dict of engine sequences
- Full history context (variable length, no max_context cap — engines are
  128–362 cycles = 8–23 tokens at P=16)
- Δt ~ LogUniform[1, 150], Δt_max=150
- Batch=64, lr=3e-4, patience=5, max 50 epochs
- Save checkpoint per seed

**Pred-FT** (3 seeds × 3 pretrain seeds = 9 runs, but just use matched seeds):
- Horizons: {1, 5, 10, 20, 50, 100, 150} cycles
- EventDataset with full engine history, stride=4 for train, stride=1 for test
- Batch=256, lr=1e-3, patience=8, max 40 epochs
- Evaluate: AUPRC, AUROC, RMSE (legacy), monotonicity violation rate
- Save surfaces as .npz

**Report**: compare to v21 results (AUPRC 0.945±0.016).
This is the key sanity check — if the new architecture regresses badly
on FD001, debug before continuing.

### Phase 2: C-MAPSS FD002, FD003 (1 h)

Same protocol as Phase 1, 3 seeds each.
FD002 has multiple operating conditions — test if RevIN handles this
(it should, since it normalizes per-context rather than per-condition).

### Phase 3: SMAP anomaly pred-FT (1.5 h)

**Pretrain** on SMAP train stream (3 seeds):
- n_channels=25, max_context=512, Δt_max=150
- Save checkpoints

**Pred-FT** with intra-entity splits (3 seeds):
- Use `data.smap_msl.split_smap_entities()` for entity splits
- Build EventDataset per entity, ConcatDataset
- Stride=4 train, stride=1 test
- Evaluate AUPRC, AUROC, non-PA F1, PA-F1

**Report**: compare to v22 baseline (AUPRC 0.290±0.042) and v23 patch
L=10 result (AUPRC 0.433). The new architecture has P=16 + RevIN +
cumulative target — all three changes at once. If AUPRC improves, the
new architecture is validated. If it regresses, isolate which change hurt.

### Phase 4: MSL, PSM, SMD, MBA (2 h)

Same protocol as Phase 3 for each dataset. 3 seeds each.
Use existing data loaders (`data/*.py`).
Entity splits for MSL/SMD, chronological for PSM/MBA.

### Phase 5: Sepsis (1 h)

Use the data loader from v23 (`data/sepsis.py`).
- n_channels=40, **P=1** (exception: hourly data, most stays < 128 hours)
- This means: build a separate FAM with `patch_size=1` for Sepsis only
- Variable-length context (full stay, up to ~336 tokens)
- Pretrain on set A training split
- Pred-FT with patient-level splits, 3 seeds
- Horizons: {1, 2, 3, 6, 12, 24, 48} hours
- Report AUPRC and AUROC (for literature comparison)
- Exclude stays shorter than 8 timesteps (8 hours — minimum for transformer)

### Phase 6: PA-F1 from surfaces (15 min)

Run `experiments/v22/compute_pa_f1_from_surfaces.py` on v24 surfaces.
Apples-to-apples comparison with MTS-JEPA.

### Phase 7: Update RESULTS.md + Quarto notebook (30 min)

- Add v24 section to RESULTS.md with all numbers
- Create `notebooks/24_v24_analysis.qmd` with:
  - Full results table: v24 vs v22 vs v21 (did architecture changes help?)
  - AUPRC per-horizon curves
  - Monotonicity violation rates
  - Surface heatmaps (at least one per dataset)
- Render: `quarto render notebooks/24_v24_analysis.qmd`

### Phase 8: If time — label efficiency on FD001 (1 h)

Pred-FT vs E2E at 5%, 10%, 50%, 100% labels. 5 seeds.
This directly tests the paper's core claim. Use the v21 label-subsampling
protocol.

---

## Phase priorities

| Phase | What | Est. | Priority |
|-------|------|------|----------|
| 0 | Dataset analysis table | 30 min | Do first |
| 1 | Sanity check on VM | 30 min | BLOCKING |
| 2 | FD001 pretrain + pred-FT | 1.5 h | Critical |
| 3 | FD002, FD003 | 1 h | Critical |
| 4 | SMAP | 1.5 h | Critical |
| 5 | MSL, PSM, SMD, MBA | 2 h | Important |
| 6 | Sepsis (P=1 exception) | 1 h | Important |
| 7 | PA-F1 from surfaces | 15 min | Easy |
| 8 | RESULTS.md + notebook | 30 min | Always |
| 9 | Label efficiency | 1 h | If time |

**Total**: ~9.5h. Fits an overnight session.

---

## Ground rules

1. **Import from `model.py` and `train.py`.** Do NOT copy-paste model code
   into experiment scripts. Write thin wrappers that load data and call
   `pretrain()`, `finetune()`, `evaluate()`.
2. **Cumulative target**: target = x(t : t+Δt]. No gap. No fixed w.
3. **P=16 everywhere except Sepsis** (P=1, hourly data below resolution floor).
4. **RevIN in the encoder** (already built into `CausalEncoder` and
   `TargetEncoder`). Do NOT also apply global normalization in data loaders.
   Pass `normalize=False` to all data loaders.
5. **Stride=1 at evaluation**. Stride=4 for train/val is fine.
6. **Collapse monitoring**: pretrain() already checks h_std < 0.01.
7. Store surfaces as .npz. Compute AUPRC + legacy metrics from surfaces.
8. Reporting: `mean ± std (Ns)`. Decompose F1 into P + R.
9. Commit + push after each phase. Update RESULTS.md after every phase.
10. If FD001 AUPRC drops below 0.90, stop and debug. This is a regression.
11. If sepsis gives <3 tokens context, document it. This is informative.

---

## Data loader notes

**C-MAPSS**: use `experiments/v11/data_utils.py:load_cmapss_subset()`.
Pass `normalize=False` (RevIN handles normalization now).
Warning: the old data loader normalizes by default — check!

**Anomaly datasets**: use `data/*.py` loaders. Pass `normalize=False`.
Entity splits: `data/smap_msl.py:split_smap_entities()` etc.

**Sepsis**: use `data/sepsis.py` (created in v23).

For all datasets: the data loaders return raw numpy arrays. Wrap them
in `PretrainDataset` or `EventDataset` from `train.py`.

---

## What a good experiment script looks like

```python
"""V24 Phase 1: FD001 pretrain + pred-FT."""
import sys, json, torch
from pathlib import Path

FAM_DIR = Path('/home/sagemaker-user/IndustrialJEPA/fam-jepa')
sys.path.insert(0, str(FAM_DIR))
sys.path.insert(0, str(FAM_DIR / 'experiments/v11'))

from model import FAM
from train import (PretrainDataset, EventDataset, collate_pretrain,
                   collate_event, pretrain, finetune, evaluate,
                   save_surface, get_horizons)
from data_utils import load_cmapss_subset

SEEDS = [42, 123, 456]
DEVICE = 'cuda'
V24 = FAM_DIR / 'experiments/v24'

# Load data (normalize=False — RevIN handles it)
data = load_cmapss_subset('FD001', normalize=False)
# ... build datasets, call pretrain(), finetune(), evaluate()
# ... save results JSON + surfaces
```

Keep experiment scripts short. All logic lives in `model.py` and `train.py`.

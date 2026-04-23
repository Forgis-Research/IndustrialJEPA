# V23 Session — SIGReg, Patch Tokenization, Sepsis Prediction

**Usage**: Paste as opening prompt to a Claude Code session on the GPU VM.
**Working directory**: `IndustrialJEPA/fam-jepa/`
**Duration**: ~3 hours on A10G.
**Prereqs**: Read CLAUDE.md, experiments/RESULTS.md, v22 SESSION_PROMPT.md

---

## Four goals

### Goal A: Drop EMA — SIGReg with curriculum (ablation)

Replace EMA target encoder with explicit collapse prevention (SIGReg).
This simplifies the method and removes a hyperparameter.

**Prior work in this repo:**
- V17 phase 3: curriculum EMA→SIGReg on FD001 (100 epochs EMA warm-up,
  then ramp λ_sig 0→0.05 while removing EMA). Worked but never re-evaluated
  under AUPRC.
- V20 phase 3: SIGReg-pred beat EMA on F1w+RMSE at 100% labels (0.451 vs
  0.391 F1w), but was never tested under AUPRC or on other datasets.
- V22 pretraining already uses a variance regularizer (λ_var=0.04,
  VICReg-style `relu(1-std)`) on top of EMA.

**What to run:**

1. **FD001 SIGReg from scratch** (no EMA at any point). Use full VICReg
   triplet (invariance L1 + variance relu(1-std) + covariance off-diag)
   with curriculum on k: start k∈[1,10] for first 20 epochs, expand to
   k∈[1,150] by epoch 40. This stabilizes early training when the
   predictor hasn't learned anything yet.
   - Loss = L1_normalized + λ_var * var_loss + λ_cov * cov_loss
   - λ_var=0.04, λ_cov=0.02 (start here, adjust if collapse)
   - Target = context_encoder(future).detach() (stop-grad, no EMA)
   - 3 seeds, early stop patience=5, max 50 epochs

2. **Compare to EMA baseline**: use existing v22 pretrained baseline
   checkpoints (FD001). Freeze both encoders, run pred-FT 3 seeds.
   Report AUPRC under identical downstream protocol.

3. **If SIGReg matches baseline on FD001**: repeat on SMAP with entity
   splits. If it doesn't match, document why and move on (<30 min debug).

**Collapse detection**: monitor std of h_past per dimension each epoch.
If mean(std) < 0.01 → collapsed, abort that seed.

### Goal B: Patch tokenization on SMAP/PSM (ablation)

Current tokenization: `patch_length=1`, i.e. `Linear(C, d)` per timestep.
Each timestep is one token. For 25-channel SMAP with window=100, that's
100 tokens of dim 256 — the temporal transformer sees each timestep but
channels are mixed in the input projection.

**Patch tokenization**: group L consecutive timesteps into one token via
`Linear(C * L, d)`. With L=10 and window=100, that's 10 tokens. This:
- Reduces sequence length (faster attention, more scalable)
- Forces the model to learn local temporal patterns within each patch
- Is standard in PatchTST / TimesFM / Chronos

**What to run:**

1. Modify `SensorProjection` (already supports `patch_length` parameter).
   Test L ∈ {5, 10, 20} on SMAP (25 channels, window=100).
2. Pretrain each variant (same protocol as v22: L1 + var_reg, early stop).
3. Pred-FT with entity splits, 3 seeds. Report AUPRC.
4. **Only test on SMAP and PSM** — these have long, high-frequency streams
   where patching should help most. C-MAPSS cycles are already semantically
   meaningful units, patching there makes less sense.

**Key question**: does patching help or hurt when the downstream task is
event prediction (not forecasting)? Longer patches blur the temporal
resolution of the probability surface.

### Goal C: PhysioNet 2019 Sepsis — new domain, clear onset event

Add a medical event-prediction dataset with well-defined onset, strong
literature baselines, and a fundamentally different event type (clinical
deterioration, not mechanical/anomaly).

**Dataset**: PhysioNet Computing in Cardiology 2019 Challenge (early
prediction of sepsis in ICU patients).
- ~40k ICU stays, 40 clinical variables (vitals, labs, demographics)
- Binary label `SepsisLabel` with clear onset time (flips 0→1)
- Hourly resolution, variable-length stays (6–336 hours), missing values
- Literature SOTA: AUROC ~0.78–0.85, utility score ~0.36–0.45
- Free download: `wget -r -N -c -np https://physionet.org/files/challenge-2019/1.0.0/`

**Why this dataset matters for the paper:**
- **Clear onset time**: sepsis has a defined start — our surface p(t, Δt)
  predicts "sepsis within Δt hours from time t". This is the predictive
  use case (not just detection of ongoing anomaly).
- **Established SOTA**: many published baselines to compare against.
- **Medical domain**: different from all current datasets. Strengthens
  the "forecast anything" claim across turbofan/spacecraft/server/cardiac/ICU.
- **Missing values + irregular features**: tests whether the architecture
  generalises beyond clean, regularly sampled sensor streams.

**What to run:**

1. **Data loader** (`data/sepsis.py`):
   - Download training sets A and B (set C is hidden).
   - Each patient is a `.psv` file (pipe-separated). Load all, concat.
   - Handle missing values: forward-fill then zero-fill (standard).
   - Drop static demographics (Age, Gender, HospAdmTime) or append as
     constant channels — try both if time, default to drop.
   - Normalize per-feature on training set.
   - Split: use set A for train/val (80/20 patient-level), set B for test.
   - Return dict with `train`, `test`, `labels_test`, same interface as
     other loaders.

2. **Pretrain** on set A training split (normal data only — stays without
   sepsis, or pre-onset portions of sepsis stays). Same protocol: causal
   transformer, L1 + var_reg, early stop. Window = 24h (24 timesteps).
   k ∈ [1, 48] (predict up to 48h ahead).

3. **Pred-FT** with entity splits (each patient is an entity). Freeze
   encoder, finetune predictor + sigmoid head. 3 seeds.
   Compute AUPRC on probability surface + per-horizon AUPRC.

4. **SOTA comparison**: report AUROC (standard in sepsis literature) and
   our AUPRC. Key baselines:
   - MGP-AttTCN (AUROC 0.852) — Futoma et al.
   - InceptionTime (AUROC 0.847) — Ismail Fawaz et al.
   - Challenge winner (utility 0.364) — Reyna et al. 2020

**Practical note**: the full dataset is ~6 GB of .psv files (~40k patients).
Download time ~5 min on the VM. Preprocessing into numpy arrays ~2 min.
The small size per patient (mean ~40 timesteps) means pretraining is fast.

### Goal D: PA-F1 from stored surfaces (quick, first)

Run `experiments/v22/compute_pa_f1_from_surfaces.py`.
Produces `pa_f1_from_surfaces.json` with PA-F1 for all 5 anomaly datasets.
Update RESULTS.md. Takes <1 min, do this first.

---

## Phase plan

| Phase | What | Est. time | Priority |
|-------|------|-----------|----------|
| 0 | PA-F1 from surfaces, update RESULTS.md | 5 min | Do first |
| 1 | SIGReg pretrain FD001 (3 seeds) | 40 min | Critical |
| 2 | SIGReg vs EMA: pred-FT comparison on FD001 | 20 min | Critical |
| 3 | Sepsis data loader + download + sanity check | 20 min | Critical |
| 4 | Sepsis pretrain + pred-FT (3 seeds) | 40 min | Critical |
| 5 | Patch tokenization: pretrain SMAP L={5,10,20} | 30 min | Important |
| 6 | Patch pred-FT on SMAP (best L + baseline) | 20 min | Important |
| 7 | If SIGReg works on FD001: repeat on SMAP | 20 min | If time |
| 8 | Update RESULTS.md + render notebook | 15 min | Always |

**Total**: ~3.5h (tight — skip phase 7 if behind schedule).

---

## Ground rules

1. **Collapse monitoring**: log mean(std(h_past)) each epoch. Abort seed if < 0.01.
2. **SIGReg = no EMA, stop-grad only.** Target = context_encoder(future).detach().
3. **Curriculum**: ramp k range, not λ. Start k∈[1,10], expand to full range by epoch 40.
4. Store surfaces as .npz. Compute AUPRC + legacy metrics from surfaces.
5. Reporting: `mean ± std (Ns)`. Decompose F1 into P + R.
6. Commit + push after each phase.
7. Update RESULTS.md after every phase.
8. If something clearly fails (collapse, loss plateau), document and move on. Max 30 min debug per issue.

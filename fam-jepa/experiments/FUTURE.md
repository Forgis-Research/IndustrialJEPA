# Future Work (pre-submission, not next session)

Target: complete before NeurIPS 2026 submission deadline.

---

## 1. Cross-domain transfer

The paper's current weakness: we pretrain per-dataset. A foundation model
pretrains on a union. Three variants to test:

### 1a. Multi-domain pretrain, finetune on test domain

Pretrain one encoder on ALL training data (FD001+FD002+FD003+SMAP+MSL+
PSM+SMD+MBA+PhysioNet). Then pred-FT per dataset on that dataset's
labeled data. Does the shared encoder learn universal dynamics?

### 1b. Multi-domain pretrain, finetune on unseen domain

Pretrain on N-1 datasets, pred-FT on the held-out dataset.
Leave-one-domain-out cross-validation. This tests zero-shot transfer:
can dynamics learned from turbofan+server+spacecraft help predict
cardiac arrhythmias?

### 1c. Use someone else's pretrained encoder

Take a frozen Chronos-2 or TimesFM-2.5 encoder (already pretrained on
massive corpora). Apply our pred-FT recipe on top. If this works well,
it validates that pred-FT is the contribution, not the encoder.
v24 already has partial results (Chronos-2 baseline), but those use a
linear probe, not our full predictor. Run pred-FT on Chronos-2
embeddings for a fair comparison.

### 1d. Pretrain on many, test on completely unseen dataset

Find a dataset NOT in our benchmark (e.g. SWaT, WADI, bearing vibration).
Pretrain on our full corpus. Pred-FT with the unseen dataset's labels.
This is the strongest "forecast anything" claim: the encoder has never
seen this domain during pretraining.

---

## 2. Dense horizon finetuning

Currently we finetune on 7 sparse horizons and evaluate at 7.
Test: finetune on dense horizons (every integer Δt) and evaluate dense.
Expected: better CDF quality, smoother surfaces, potentially better AUPRC
(more supervision signal per sample).

## 3. Predictor architecture ablation

MLP vs small transformer predictor. The predictor needs to learn dynamics
in representation space conditioned on Δt. A transformer could attend
over learned "dynamics queries." Low priority unless MLP is clearly
the bottleneck (FD003 regression suggests it might be).

## 4. FD003 capacity investigation

FD003 regressed from 0.932 (v21, P=1, 790K predictor) to 0.766 (v24,
P=16, 198K predictor). Two hypotheses:
- P=16 loses per-cycle resolution needed for multi-fault degradation
- 198K predictor is too small for two interacting fault modes
Test each independently: P=1 with 198K predictor, P=16 with 790K predictor.

## 5. Remaining version labels in paper

~22 internal version references (v20, v21, v22, v24, V17) remain in
the ablation sections. Clean all before submission.

## 6. Paper figures

- Surface heatmap figure for the paper (not just notebook): pick FD001
  and SMAP, show predicted vs ground truth with log-scaled Δt axis
- Label efficiency curve figure (pred-FT vs E2E vs scratch at 5%/10%/20%/50%/100%)

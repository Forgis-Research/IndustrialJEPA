# IndustrialJEPA: Proof-of-Concept Paper Execution Plan

**Target:** Workshop paper (NeurIPS/ICML) or arXiv preprint
**Timeline:** 4 weeks
**Budget:** ~$100-200 compute

---

## 1. Paper Thesis (One Sentence)

> By predicting Effort from Setpoint in latent space, JEPA learns the physics of machine behavior rather than hardware-specific statistics, enabling fault detection that transfers across different robot types.

---

## 2. Critical Assessment

### What We Have
| Asset | Status | Usability |
|-------|--------|-----------|
| FactoryNet Hackathon dataset | 18M rows, 5 machines | Ready |
| Unified Setpoint/Effort/Feedback schema | Implemented | Ready |
| JEPA training code (`train_jepa.py`) | Exists but for C-MAPSS | Needs adaptation |
| Q&A framework (4 tiers) | Defined in FactoryNet paper | Needs implementation |

### What We Don't Have
| Gap | Risk | Mitigation |
|-----|------|------------|
| Proof that JEPA > MAE on FactoryNet | Core hypothesis unvalidated | Week 1 experiment |
| Cross-machine transfer evidence | Hero result missing | Week 2 experiment |
| Q&A evaluation implementation | No code exists | Classifiers on JEPA embeddings |
| Baseline comparisons | No published baselines | Implement MAE, Autoencoder |

### Honest Risks
1. **Cross-machine transfer might fail** — different tasks (screwdriving vs pick-place) may dominate over shared physics
2. **JEPA might not beat MAE** — FactoryNet data might be clean enough that reconstruction works fine
3. **Scale might be insufficient** — 18M rows sounds big but episodes are few (7K)
4. **Q&A tiers 3-4 are hard** — Counterfactual and Decision-Making require sophisticated reasoning

---

## 3. Q&A Tiers

| Tier | Name | Question Type | Implementation | Difficulty |
|------|------|---------------|----------------|------------|
| 1 | **State** | "Is the robot carrying a load?" | Binary classifier on JEPA embeddings | Easy |
| 2 | **Intervention** | "If we increase speed by 20%, what happens to effort?" | Conditional prediction head | Medium |
| 3 | **Counterfactual** | "If payload were 3kg instead of 1kg, what would effort be?" | Generative model / simulation | Hard |
| 4 | **Decision-Making** | "Should we stop the robot given current state?" | Policy head / threshold-based | Hard |

### Proof-of-Concept Scope
- **In scope:** Tiers 1-2 (State, Intervention)
- **Future work:** Tiers 3-4 (Counterfactual, Decision-Making)

---

## 4. Experiments

### Experiment 1: JEPA vs Baselines (Week 1)
**Goal:** Prove JEPA learns better representations than alternatives

```
Dataset: AURSAD (UR3e, 6.2M rows, 4094 episodes)
Split: 80% train (healthy), 20% test (healthy + faults)

Models:
  1. JEPA: Predict Effort embedding from Setpoint (ours)
  2. MAE: Reconstruct raw Effort from Setpoint
  3. Autoencoder: Reconstruct Effort (no Setpoint input)
  4. Contrastive: Same-episode pairs positive

Task: Fault detection (anomaly = high prediction error)
Metric: AUC-ROC, F1, Precision, Recall

Success Criteria:
  - JEPA AUC > MAE AUC by >5%
  - JEPA converges faster (fewer epochs)
```

### Experiment 2: Causal Structure Ablation (Week 1)
**Goal:** Prove Setpoint matters (causal structure is key)

```
Variants:
  A. JEPA with Setpoint → Effort (full causal)
  B. JEPA with Effort only (no command info)
  C. JEPA with Feedback → Effort (wrong direction)

Expected: A >> B >> C
This proves the Setpoint is essential.
```

### Experiment 3: Cross-Machine Transfer (Week 2)
**Goal:** The hero result — prove physics transfers

```
Train: AURSAD (UR3e, screwdriving)
Test: voraus-AD (Yu-Cobot, pick-and-place) — ZERO SHOT

Metrics:
  - Fault detection AUC on voraus-AD
  - Compare to: supervised baseline trained on voraus-AD

Scenarios:
  A. Zero-shot (no voraus-AD training)
  B. Few-shot (10% voraus-AD labels)
  C. Full supervised (100% voraus-AD)

Success Criteria:
  - Zero-shot achieves >50% of supervised performance
  - Few-shot matches or exceeds supervised
```

### Experiment 4: Multi-Dataset Pretraining (Week 2)
**Goal:** Does combining datasets help?

```
Pretraining Configs:
  A. AURSAD only
  B. All 4 robot datasets (exclude NASA Milling)
  C. All 5 datasets (include NASA Milling)

Test: Fine-tune on AURSAD, measure fault detection

Expected:
  - B > A (multi-robot helps)
  - C ≈ B (CNC doesn't help robots)
```

### Experiment 5: Q&A Evaluation (Week 3)
**Goal:** Show JEPA representations support machine understanding

```
Tier 1 (State):
  Q: "Is there an undeclared payload?"
  Implementation: Binary classifier on JEPA embedding
  Labels: From fault annotations

Tier 2 (Intervention):
  Q: "If we run at 50% speed, will peak effort increase or decrease?"
  Implementation: Predict effort at modified setpoint
  Evaluation: Compare predicted vs actual at different speeds

Metrics: Accuracy, F1 for classification; MSE for prediction
```

---

## 5. Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| **1** | Setup + Experiments 1-2 | JEPA vs baselines results, ablation table |
| **2** | Experiments 3-4 | Cross-machine transfer results, multi-dataset results |
| **3** | Experiment 5 + Analysis | Q&A results, visualizations, error analysis |
| **4** | Writing | Full paper draft |

---

## 6. Paper Outline

```
Title: JEPA Learns Physics, Not Hardware: Causal Structure Enables
       Cross-Machine Transfer in Industrial Time Series

1. Introduction (1 page)
   - Industrial AI gap: sensor statistics vs physics
   - Key insight: Setpoint → Effort is learnable and transferable
   - Contributions

2. Related Work (0.5 pages)
   - JEPA (I-JEPA, V-JEPA, TS-JEPA)
   - Industrial anomaly detection
   - Cross-machine transfer

3. FactoryNet and Causal Structure (1 page)
   - Setpoint/Effort/Feedback schema
   - Why this enables transfer (the physics argument)
   - Dataset statistics

4. Method: IndustrialJEPA (1.5 pages)
   - Architecture: Encoder, Predictor, EMA target
   - Training objective
   - Fault detection via prediction error

5. Experiments (2 pages)
   - Exp 1: JEPA vs baselines
   - Exp 2: Ablation (Setpoint matters)
   - Exp 3: Cross-machine transfer (hero result)
   - Exp 4: Multi-dataset pretraining
   - Exp 5: Q&A evaluation (Tiers 1-2)

6. Analysis (0.5 pages)
   - What does JEPA learn? (visualizations)
   - When does transfer fail?

7. Conclusion (0.5 pages)
   - Summary
   - Limitations
   - Future: Tiers 3-4, LLM integration, larger scale

Total: ~7-8 pages (workshop) or 9 pages (main conference)
```

---

## 7. Code Changes Needed

### Adapt `train_jepa.py` for FactoryNet
```python
# Current: C-MAPSS dataset (single machine, RUL task)
# Needed: FactoryNet dataset (multi-machine, fault detection)

Changes:
1. New dataloader for FactoryNet parquet format
2. Handle Setpoint/Effort/Feedback column naming
3. Episode-based splitting (not random)
4. Per-episode normalization
5. Fault labels for evaluation
```

### Implement Baselines
```python
# MAE Baseline
class MAEBaseline(nn.Module):
    # Reconstruct raw Effort from Setpoint

# Autoencoder Baseline
class AutoencoderBaseline(nn.Module):
    # Reconstruct Effort without Setpoint

# Contrastive Baseline
class ContrastiveBaseline(nn.Module):
    # InfoNCE on same-episode pairs
```

### Implement Q&A Heads
```python
# Tier 1: State classification
class StateClassifier(nn.Module):
    def __init__(self, jepa_encoder):
        self.encoder = jepa_encoder
        self.head = nn.Linear(embed_dim, num_states)

# Tier 2: Intervention prediction
class InterventionPredictor(nn.Module):
    def __init__(self, jepa_encoder):
        self.encoder = jepa_encoder
        self.head = nn.Linear(embed_dim + action_dim, effort_dim)
```

---

## 8. Compute Budget & GPU Requirements

### GPU vs CPU Analysis

Based on initial CPU training on AURSAD hackathon subset (6.2M rows, 32K windows):
- **Batch time:** ~2.7 seconds per batch (batch_size=32, seq_len=256)
- **Epoch time:** ~45 minutes per epoch on CPU
- **Model size:** 14.2M trainable parameters

| Dataset | Rows | Est. Windows | CPU Time/Epoch | GPU Time/Epoch |
|---------|------|--------------|----------------|----------------|
| AURSAD (hackathon) | 6.2M | ~32K | ~45 min | ~3-5 min |
| Full FactoryNet | 17.9M | ~100K+ | ~2.5 hours | ~10-15 min |

### AWS SageMaker Recommendations

| Instance | GPU | VRAM | Cost/hr | Best For |
|----------|-----|------|---------|----------|
| ml.g4dn.xlarge | T4 | 16GB | $0.74 | Development, debugging |
| ml.g5.xlarge | A10G | 24GB | $1.41 | Standard training |
| ml.p3.2xlarge | V100 | 16GB | $3.82 | Fast iteration |

**Recommendation:** `ml.g5.xlarge` for production training - good balance of speed and cost.

### Estimated Costs

| Task | GPU Hours | Cost @ $1.41/hr |
|------|-----------|-----------------|
| Experiment 1 (4 models × 3 runs) | 8 | $11 |
| Experiment 2 (3 variants × 3 runs) | 6 | $8 |
| Experiment 3 (3 configs × 3 runs) | 6 | $8 |
| Experiment 4 (3 configs × 3 runs) | 10 | $14 |
| Experiment 5 (Q&A training) | 4 | $6 |
| Debugging/iteration | 15 | $21 |
| **Total** | **~50** | **~$70** |

### Quick Start (No GPU)

For validation and debugging, CPU training works but is slow:
```bash
# Quick validation (CPU, ~30 min)
python scripts/train_world_model.py --epochs 5 --batch_size 16 --window_size 128 --subset AURSAD

# With wandb tracking
python scripts/train_world_model.py --epochs 5 --wandb --wandb_project industrialjepa --subset AURSAD
```

### Production Training (GPU)

```bash
# Full FactoryNet training (GPU recommended)
python scripts/train_world_model.py \
    --dataset Forgis/FactoryNet_Dataset \
    --epochs 100 \
    --batch_size 64 \
    --wandb \
    --wandb_project industrialjepa
```

---

## 9. Success Criteria

### Minimum Viable Paper
- [ ] JEPA beats MAE on AURSAD fault detection
- [ ] Setpoint ablation shows causal structure matters
- [ ] Cross-machine transfer shows any positive signal (>30% of supervised)
- [ ] Q&A Tier 1 accuracy >80%

### Strong Paper
- [ ] JEPA beats all baselines by >10%
- [ ] Cross-machine transfer achieves >60% of supervised
- [ ] Multi-dataset pretraining helps
- [ ] Q&A Tier 2 works (intervention prediction)
- [ ] Clear visualization of learned representations

### If Things Go Wrong
| Problem | Pivot |
|---------|-------|
| Transfer fails completely | Focus on "what's needed for transfer" — negative result paper |
| JEPA ≈ MAE | Focus on convergence speed, sample efficiency |
| Q&A doesn't work | Drop Q&A, focus on fault detection |
| Data quality issues | Switch to single best dataset (AURSAD) |

---

## 10. Repository Cleanup Recommendations

### DELETE (not relevant to new direction)
```
src/opentslm/                    # Medical time series (ECG, HAR, Sleep)
evaluation/                       # OpenTSLM evaluation code
demo/                            # HuggingFace demos for medical models
curriculum_learning.py           # OpenTSLM training script
test/                            # OpenTSLM tests
```

### KEEP (relevant to IndustrialJEPA)
```
src/industrialworldlm/           # Core model code
scripts/train_jepa.py            # JEPA training (needs adaptation)
scripts/evaluate_jepa.py         # Evaluation (needs adaptation)
configs/                         # Configuration files
paper_draft/                     # Paper files
```

### RENAME Repo
- Current: `OpenTSLM`
- Suggested: `IndustrialJEPA` or `FactoryJEPA`

---

## 11. Immediate Next Steps

1. **Today:** Create FactoryNet dataloader for AURSAD
2. **Tomorrow:** Run JEPA vs MAE on AURSAD (sanity check)
3. **Day 3:** If sanity check passes, run full Experiment 1
4. **Week 1 end:** Go/no-go decision based on Experiments 1-2

---

## 12. Open Questions

1. **Normalization:** Per-episode z-score or global? (Test both)
2. **Episode length:** Fixed windows or full episodes? (Start with windows)
3. **Fault labels:** Use original dataset labels or unified schema?
4. **NASA Milling:** Include or exclude? (Exclude initially)
5. **Franka datasets (RH20T, REASSEMBLE):** 7-DOF vs 6-DOF — does this matter?

---

*Last updated: 2026-03-04*
*Status: Planning phase*

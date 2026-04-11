# Recommendation: Which Direction to Pursue

**Date**: 2026-03-23
**Based on**: 35+ papers reviewed, 38 experiments completed, root cause analysis of JEPA failures

---

## Most Promising Direction: Slot-Concept Transformer

### Why

1. **Genuine novelty**: No prior work applies slot attention to multivariate industrial sensors for component discovery. SlotFM (2025) does frequency decomposition on accelerometers — ours would discover physical components.

2. **Unifies our findings**: Our Role-Transformer uses *fixed* physics-informed grouping and gets 35% better transfer (p=0.005). The Slot-Concept Transformer *learns* the grouping — generalizing Role-Trans to settings where physics is unknown.

3. **Built-in validation**: C-MAPSS has known component structure (fan, HPC, combustor, turbine, nozzle). We can directly measure if learned slots match physics.

4. **Low implementation risk**: It's fundamentally a Role-Transformer where the channel groupings are learned via slot attention instead of hardcoded. Our existing infrastructure (training loop, evaluation, baselines) transfers directly.

5. **Strong paper narrative**: "Can neural networks discover the physical structure of industrial systems from sensor data alone? We show that slot attention, applied to multivariate sensor time series, recovers 80%+ of known turbofan component structure and provides transfer performance comparable to expert-specified groupings."

### How It Extends Our Results

| What We Know | What Slot-Concept Adds |
|-------------|----------------------|
| Role-Trans beats CI-Trans for cross-condition transfer | Do learned concepts also transfer? |
| Weight sharing is the key mechanism | Does slot attention discover weight-sharing opportunities? |
| Physics grouping > random > none | Is learned grouping ≥ physics grouping? |
| JEPA pretraining doesn't help | Does concept-based prediction change this? |
| Encoder quality matters for frozen transfer | Are slot-based representations more transferable frozen? |

---

## Second Priority: Mechanical-JEPA (Fixed)

### Why Second

- Higher risk (3 prior failures) but compelling narrative
- Root causes now understood (codebook, high masking, deep encoder)
- If it works, "Why JEPA Fails and How to Fix It" is a strong NeurIPS title
- Implement ONLY after Slot-Concept validates that concepts work

### Dependency

If Slot-Concept works → try Slot-JEPA hybrid (JEPA in slot space with codebook)
If Slot-Concept fails → Mechanical-JEPA is the backup with proven fixes

---

## Do NOT Pursue: Sparse Graph Learning (Direction 1)

### Why Not

- Incremental over MTGNN/GTS which already exist
- Our Role-Trans is already a physics-informed graph — sparse graph would be a small delta
- Graph learning methods have known failure modes (collapse to dense/trivial)
- Less impactful than concept discovery

---

## Concrete Next Steps

### Immediate (tonight)
1. Implement Slot-Concept Transformer on C-MAPSS
2. Train on FD001, analyze slot assignments
3. Compare to Role-Trans and CI-Trans on FD001 and FD001→FD002
4. If slots discover meaningful groupings: this is the paper

### This Week
5. Ablate number of slots (K=3, 5, 7, 10, 14)
6. Test slot stability across seeds
7. Frozen encoder transfer experiments
8. If promising: implement Slot-JEPA hybrid

### Paper Structure (if results support)
1. **Introduction**: Channel-independence vs dependence paradox
2. **Related Work**: Graph learning, concept models, JEPA
3. **Method**: Slot-Concept Transformer
4. **Experiments**:
   - Concept discovery: do slots match physics?
   - Transfer: do concepts transfer?
   - Ablations: K, slot attention iterations, etc.
   - Comparison to Role-Trans, CI-Trans, MTGNN
5. **Analysis**: Why concepts work for transfer

### Timeline to NeurIPS Submission
- March 23-25: Implement and validate Slot-Concept
- March 25-28: Full experiments + ablations
- March 28-31: Paper writing
- April 1-7: Revisions
- NeurIPS 2026 deadline: ~May 2026

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Slots don't discover meaningful components | Initialize slot centroids with physics groupings |
| Slot assignments collapse (all → one slot) | Entropy regularization on assignments |
| Learned grouping doesn't beat fixed physics | Still interesting: "neural nets can't beat domain knowledge" |
| Transfer doesn't improve over Role-Trans | Report honestly; the concept discovery itself is novel |

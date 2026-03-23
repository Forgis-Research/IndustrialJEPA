# Is Cross-Machine Physics Transfer Utopian?

An honest analysis of what transfers, what doesn't, and how to test it.

---

## The Spectrum of "Physics Transfer"

```
DEFINITELY TRANSFERS          MAYBE TRANSFERS              DEFINITELY DOESN'T
        |                           |                              |
   Conservation laws         Degradation patterns           Frequency signatures
   F = ma                    "Vibration grows"              Ball-pass frequency
   Energy balance            Friction increases             Gear mesh harmonics
   Control loop structure    Efficiency drops               Thermal time constants
```

---

## Evidence From Other Fields

### Physics-Informed Neural Networks (PINNs)

PINNs encode equations like `∂u/∂t = α∇²u` directly. These transfer perfectly because **the equation is the prior**, not learned.

**Lesson**: If you *know* the physics, encode it. Don't learn it.

### Foundation Models in Other Domains

| Domain | What Transfers | What Doesn't |
|--------|----------------|--------------|
| Vision (ImageNet→X) | Edges, textures, shapes | Domain-specific objects |
| Language (GPT→X) | Grammar, reasoning patterns | Domain vocabulary |
| Protein (ESM→X) | Amino acid interactions | Specific protein functions |

**Pattern**: Low-level structure transfers. High-level semantics don't.

### Our C-MAPSS Results

We showed that **within-component relationships** transfer across operating conditions:
- Fan speed ↔ bypass ratio relationship is invariant
- But the *absolute values* change with conditions

This is physics transfer at the **relational** level, not the **parametric** level.

---

## Three Levels of Physics Transfer

### Level 1: Parametric (Utopian ❌)

**Claim**: "The model learns that `τ = I·α + b·ω` and transfers the exact coefficients"

**Reality**: Different machines have different `I` (inertia), `b` (damping). The parameters don't transfer.

**Our evidence**: Role-Trans doesn't learn exact sensor values. It learns relationships.

### Level 2: Relational (Plausible ✓)

**Claim**: "The model learns that torque and acceleration are proportional, even if the constant differs"

**Reality**: This is what our physics grouping captures. Within a component, sensors have consistent relationships.

**Evidence**:
- Physics grouping (3.69 ratio) beats random (4.99) by 26%
- The *structure* of relationships matters, not the specific values
- Per-condition normalization eliminates the advantage → Role-Trans learns condition-*invariant* features

### Level 3: Structural (Very Plausible ✓✓)

**Claim**: "The model learns that setpoint→effort→feedback is a causal chain, regardless of machine"

**Reality**: This is universal across controlled systems:
- Robot: position command → motor torque → achieved position
- Turbofan: throttle → fuel flow → thrust
- HVAC: temperature setpoint → compressor power → room temperature

**This is our strongest transferable prior.**

---

## Summary Table

| Aspect | Utopian? | Why |
|--------|----------|-----|
| Learning exact physical constants | **Yes** | Parameters are machine-specific |
| Learning physical relationships | **No** | Relationships are invariant (our C-MAPSS proof) |
| Learning causal structure | **No** | Control loops are universal |
| Learning degradation patterns | **Partially** | "Things get worse" transfers; specific signatures don't |
| Learning frequency signatures | **Yes** | Completely hardware-specific |

---

## What Brain-JEPA Actually Did (Instructive)

Brain-JEPA's "gradient positioning" doesn't learn specific neuron firing rates. It learns:
- **Functional organization**: Which brain regions co-activate
- **Hierarchical structure**: Primary sensory → association → higher cognition
- **Connectivity patterns**: Not exact weights, but graph structure

This transfers because **brains share architecture** even if individuals differ in parameters.

**Our analog**: Machines share **control architecture** (setpoint→effort→feedback) even if parameters differ.

---

## The Key Experiment That Would Settle This

**Synthetic experiment with ground truth**:

```python
# Physics: y = A @ x + noise
# Machine 1: A1 = [[1.0, 0.5], [0.5, 1.0]]
# Machine 2: A2 = [[2.0, 1.0], [1.0, 2.0]]  # Same structure, different scale

# Train on Machine 1
# Test on Machine 2

# If model learns "y depends on both x components" → transfers
# If model learns "y ≈ 1.0*x1 + 0.5*x2" → doesn't transfer
```

If a model can learn **"y depends on x1 and x2 jointly"** rather than **"y = 1.0x1 + 0.5x2"**, cross-machine transfer works.

Our physics grouping does exactly this: it says "these sensors are related" without specifying *how*.

---

## Verdict

| Claim | Verdict |
|-------|---------|
| "We can learn a universal physics model" | ❌ Utopian |
| "We can learn that certain channels are related" | ✅ Proven (C-MAPSS) |
| "We can learn causal structure (setpoint→effort)" | ✅ Plausible, testable |
| "We can transfer across machine *types*" | ⚠️ Unknown, needs PoC |
| "We can transfer across machine *instances*" | ✅ Likely (same physics, different params) |

**The non-utopian version of our claim**:

> "Physics-informed architectures learn *which* channels are related, not *how* they're related. This structural knowledge transfers across machines that share control architecture, even when parameters differ."

---

## Quick Sanity Check (30 minutes)

Look at FD001→FD002 transfer results. The physics grouping works because:

1. Fan sensors (T2, P2, Nf, BPR) have relationships that hold across conditions
2. These relationships are **scale-invariant** (ratios, correlations)
3. Global normalization handles the scale; grouping handles the structure

**Test**: If you normalize per-sensor (z-score each channel independently), does physics grouping still win?

- If **yes**: We're learning structure, not scale → transferable
- If **no**: We're learning scale relationships → less transferable

---

## PoC Experiments to Validate

### Exp A: Synthetic Physics (1-2 hours)
Generate data from `y = A @ x + noise` with different A matrices.
- Same structure (which x's affect y), different parameters
- If transfer works here, relational learning is real

### Exp B: Cross-Bearing-Rig (2-4 hours)
Train on CWRU, test on NASA Bearing.
- Different hardware, same degradation physics
- Acid test for cross-machine transfer

### Exp C: Cross-Engine-ID (1 hour)
Train on engines 1-50, test on engines 51-100 (within FD001).
- If this fails, we're memorizing engine signatures

### Exp D: Channel Permutation (30 min)
At test time, shuffle channels within each physics group.
- If performance stable: learning physics
- If performance drops: learning positions

---

## Bottom Line

**Cross-machine physics transfer is NOT utopian if we're precise about what "physics" means.**

- Utopian: Learning exact equations and constants
- Realistic: Learning which channels relate and causal flow direction
- Our evidence: C-MAPSS shows relational transfer works across conditions
- Open question: Does it extend to different machine types?

The PoC experiments (especially synthetic) will definitively answer this.

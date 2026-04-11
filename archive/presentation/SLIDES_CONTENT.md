# ETH AI Center Presentation Slides

## 5-Minute Talk: World Models for Physical Systems

---

## SLIDE 1: Title (15 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              WORLD MODELS FOR PHYSICAL SYSTEMS                  │
│                                                                 │
│      Learning Transferable Dynamics via Koopman Theory          │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│                        Forgis Research                          │
│                                                                 │
│              FactoryNet  •  KH-JEPA  •  OpenTSLM                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Say**: "We're building world models for physical systems. The key insight: industrial machines obey physics - and physics is learnable."

---

## SLIDE 2: Two Types of Time Series (45 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│           NOT ALL TIME SERIES ARE CREATED EQUAL                 │
│                                                                 │
│  ┌───────────────────────┐    ┌───────────────────────┐        │
│  │   PHYSICAL SYSTEMS    │    │  FINANCIAL MARKETS    │        │
│  │                       │    │                       │        │
│  │  dx/dt = f(x)        │    │  Adversarial games    │        │
│  │  Newton, thermo...   │    │  EMH applies          │        │
│  │                       │    │                       │        │
│  │  ✓ Koopman-linear    │    │  ✗ No true dynamics   │        │
│  │  ✓ Transferable      │    │  ✗ Prediction erases  │        │
│  │  ✓ Interpretable     │    │    predictability     │        │
│  └───────────────────────┘    └───────────────────────┘        │
│                                                                 │
│         We focus where foundation models make sense.            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Say**: "Physical systems obey differential equations. Financial markets don't. Koopman theory tells us physics becomes linear in a lifted space. We exploit this - foundation models for physics, not speculation."

---

## SLIDE 3: Our Stack (60 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    THE FORGIS STACK                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  OpenTSLM: Foundation Model for Physical Time Series    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  KH-JEPA: Koopman + JEPA + SIGReg                       │   │
│  │                                                          │   │
│  │  • Latent prediction (not reconstruction)               │   │
│  │  • Linear dynamics: z_{t+k} = K^k z_t                   │   │
│  │  • Provable collapse prevention                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│  ┌──────────────────────┐  ┌────────────────────────────────┐  │
│  │     FactoryNet       │  │        FactoryBench            │  │
│  │                      │  │                                │  │
│  │  18M timesteps       │  │  • Cross-machine transfer     │  │
│  │  5 robots            │  │  • Long-horizon stability     │  │
│  │  Unified schema      │  │  • Anomaly detection          │  │
│  └──────────────────────┘  └────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Say**: "Three layers. FactoryNet provides unified data across 5 robot types - first cross-robot benchmark. KH-JEPA is our architecture: JEPA for latent prediction, Koopman for linear dynamics, SIGReg for provable training. OpenTSLM is the foundation model this enables."

---

## SLIDE 4: Results (60 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    AUTORESEARCH RESULTS                         │
│                                                                 │
│  ┌───────────────────┬─────────┬──────────┬─────────┐          │
│  │      Metric       │  Result │  Target  │  Status │          │
│  ├───────────────────┼─────────┼──────────┼─────────┤          │
│  │ MSE (sanity)      │  0.403  │  < 0.45  │   ✓     │          │
│  │ Transfer h1→h2    │  0.75x  │  < 1.5x  │   ✓     │          │
│  │ Long-horizon 720  │  1.49x  │  < 3.0x  │   ✓     │          │
│  └───────────────────┴─────────┴──────────┴─────────┘          │
│                                                                 │
│  Key: MSE is NOT the goal. Transfer + stability is.            │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  What worked:                                                   │
│  • Channel-independent processing (7x effective data)          │
│  • SIGReg: provable collapse prevention (LeJEPA, 2024)         │
│  • Koopman: eigenvalues < 1 → stable long-horizon              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Say**: "Results from overnight autoresearch. MSE passes sanity check, but that's not the goal. The real tests: can it transfer across distributions? 0.75x ratio - better on h2 than h1. Can it predict 720 steps out? Only 1.49x degradation. This is world model behavior - stable learned dynamics."

---

## SLIDE 5: Differentiation (30 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│              WHY THIS IS DIFFERENT                              │
│                                                                 │
│  ┌────────────────────────┬────────────────────────────┐       │
│  │    General FMs         │    Physics World Models    │       │
│  │    (Chronos, TimesFM)  │    (Ours)                  │       │
│  ├────────────────────────┼────────────────────────────┤       │
│  │ Optimize MSE           │ Learn transferable dynamics│       │
│  │ Scale (billions)       │ Inductive bias (Koopman)   │       │
│  │ Treat all TS same      │ Exploit causal structure   │       │
│  │ Closed                 │ Open (data, code, models)  │       │
│  └────────────────────────┴────────────────────────────┘       │
│                                                                 │
│       We don't beat Chronos at MSE.                            │
│       We do what Chronos can't: transfer across machines.      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Say**: "We're not competing on forecasting MSE. We're building world models that learn physics. The Koopman operator captures invariant dynamics - same physics, different hardware. That's our differentiator."

---

## SLIDE 6: Call to Action (30 seconds)

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                      OPEN RESEARCH                              │
│                                                                 │
│  All open source:                                               │
│  • github.com/ForgisX/FactoryNet                               │
│  • github.com/ForgisX/IndustrialJEPA                           │
│  • huggingface.co/Forgis                                       │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Looking for collaborations:                                    │
│                                                                 │
│  • More industrial datasets (manufacturing, energy, robotics)  │
│  • Benchmark adoption                                          │
│  • Theory: Koopman transfer guarantees                         │
│  • Scaling: Industrial pretraining at 1B params                │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│                    [Contact info / QR code]                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Say**: "Everything is open - data, code, models. We're looking for collaborations. More datasets. Theoretical analysis of when Koopman transfer works. Scaling to foundation model size. Happy to discuss."

---

## BACKUP SLIDES (for Q&A)

### Backup 1: Koopman Theory

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    KOOPMAN OPERATOR THEORY                      │
│                                                                 │
│  Physical system:      dx/dt = f(x)     [nonlinear]            │
│                                                                 │
│  Koopman form:         dz/dt = K·z      [linear!]              │
│                                                                 │
│  where z = φ(x) is a learned lifting function                  │
│                                                                 │
│  ─────────────────────────────────────────────────────────────  │
│                                                                 │
│  Benefits:                                                      │
│  • O(1) prediction: z_{t+k} = K^k z_t (matrix power)           │
│  • Interpretability: eigenvalues = system modes                │
│  • Transfer: same physics → same K                             │
│                                                                 │
│  This isn't a choice - it's mathematics.                       │
│  Any smooth dynamical system has a Koopman representation.     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Backup 2: FactoryNet Details

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    FACTORYNET DATASET                           │
│                                                                 │
│  ┌─────────┬─────────┬─────┬───────────────┬─────────┐         │
│  │ Dataset │  Robot  │ DOF │     Task      │  Rows   │         │
│  ├─────────┼─────────┼─────┼───────────────┼─────────┤         │
│  │ AURSAD  │  UR3e   │  6  │ Screwdriving  │  6.2M   │         │
│  │ voraus  │Yu-Cobot │  6  │ Pick-place    │  2.3M   │         │
│  │ RH20T   │ Franka  │  7  │ Manipulation  │  1.2M   │         │
│  │ NASA    │  CNC    │  3  │ Milling       │  1.5M   │         │
│  └─────────┴─────────┴─────┴───────────────┴─────────┘         │
│                                                                 │
│  Unified schema:                                                │
│  • Setpoint: 14 dims (7 DOF × pos + vel)                       │
│  • Effort: 13 dims (7 torques + 6 forces)                      │
│  • Validity masks for heterogeneous robots                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Backup 3: Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│                    KH-JEPA ARCHITECTURE                         │
│                                                                 │
│  Input: x ∈ R^{T×C}                                            │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │ Patch Embedding │  (channel-independent)                    │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │   Transformer   │  (per-channel temporal attention)         │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │  Latent z ∈ R^d │                                           │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │    Koopman K    │  z_{t+k} = K^k z_t                        │
│  └────────┬────────┘                                           │
│           ▼                                                     │
│  ┌─────────────────┐                                           │
│  │     Decoder     │  → ŷ ∈ R^{H×C}                            │
│  └─────────────────┘                                           │
│                                                                 │
│  + SIGReg on z for provable collapse prevention                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Timing Summary

| Slide | Content | Time | Cumulative |
|-------|---------|------|------------|
| 1 | Title | 15s | 0:15 |
| 2 | Two Types of TS | 45s | 1:00 |
| 3 | Our Stack | 60s | 2:00 |
| 4 | Results | 60s | 3:00 |
| 5 | Differentiation | 30s | 3:30 |
| 6 | Call to Action | 30s | 4:00 |
| - | Buffer/transition | 60s | 5:00 |

Total: 5 minutes exactly.

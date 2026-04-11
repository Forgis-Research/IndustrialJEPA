# Overnight Autoresearch: 3-Tier Physics Grouping Validation

**Mission:** Prove physics-informed channel grouping beats learned attention across synthetic, mechanical, and physical systems.

**Architecture:** 2D treatment — Mamba for time, Attention for space (channels/axes).

---

## Before You Start

Read these files:
1. `.claude/agents/ml_researcher.md` — Operating instructions
2. `autoresearch/LESSONS_LEARNED.md` — Critical: what failed before
3. `autoresearch/EXPERIMENT_LOG.md` — Current best results

**Key lessons from previous work:**
- RevIN hurts on C-MAPSS (don't use it)
- JEPA pretraining hurts transfer (don't use it)
- Weight sharing within groups is the critical ingredient
- Shorter sequences transfer better

---

## The Architecture (2D Patch Treatment)

```
Input: [Batch, Time, Channels, Features]
       e.g., [32, 7500, 7, 4]

Step 1: Patch along time → [B, num_patches, C, patch_dim]
Step 2: Mamba within each channel (time dimension)
Step 3: Attention across channels (space dimension)
Step 4: Decode to predictions
```

**Why this works:**
- Mamba handles long sequences efficiently (O(n) not O(n²))
- Attention only sees #channels tokens (7 for pendulum, 14 for C-MAPSS, 21 for weather)
- Physics grouping = constrained attention mask or shared Mamba encoders within groups

---

## Baselines (ALWAYS INCLUDE)

### Trivial Baselines
```python
TRIVIAL = {
    "mean": lambda x: train_mean.expand_as(target),  # Predict training mean
    "last_value": lambda x: x[:, -1:, :].expand_as(target),  # Persistence
    "random": lambda x: torch.randn_like(target) * train_std + train_mean,
}
```

### Simple Model Baselines
```python
SIMPLE = {
    "linear": nn.Linear(lookback * channels, horizon * channels),
    "mlp": nn.Sequential(Linear, ReLU, Linear),
}
```

### Architecture Baselines
```python
ARCHITECTURE = {
    "ci_transformer": "Channel-independent (no cross-channel attention)",
    "full_attention": "All-to-all attention (learns relationships from data)",
    "role_transformer": "Physics-grouped attention (our method)",
}
```

---

## Metrics (Unified)

```python
METRICS = {
    "mse": Mean Squared Error (primary),
    "mae": Mean Absolute Error (secondary),
    "transfer_ratio": MSE_target / MSE_source,  # Lower = better transfer
}

# Interpretation
# - Transfer ratio < 2.0 = good transfer
# - Transfer ratio > 5.0 = poor transfer
# - Role-Trans should beat CI-Trans on transfer ratio
```

---

## Online Learning Protocol

```
Source:  [============ 100% train ============]
Target:  [10% adapt][======= 90% test =======]
```

```python
def online_eval(model, source_data, target_data):
    # 1. Train on source
    model.fit(source_data)
    zero_shot = model.evaluate(target_data)

    # 2. Adapt on first 10% of target (chronological!)
    target_adapt = target_data[:len(target_data)//10]
    target_test = target_data[len(target_data)//10:]
    model.finetune(target_adapt, epochs=10)
    adapted = model.evaluate(target_test)

    # 3. Upper bound: full target training
    model.fit(target_data)
    full = model.evaluate(target_data)  # or held-out test

    return {"zero_shot": zero_shot, "10%_adapted": adapted, "full": full}
```

---

# Tier 1: Double Pendulum (Synthetic)

## Setup

**Generate once, use forever:**
```bash
python autoresearch/experiments/generate_pendulum.py \
    --n_trajectories 10000 \
    --timesteps 1000 \
    --dt 0.01 \
    --output data/pendulum.csv
```

**CSV Schema:**
```
trajectory_id, timestep, theta1, omega1, theta2, omega2, m1, m2, l1, l2, g
```

**Physics Groups:**
```python
PENDULUM_GROUPS = {
    "mass_1": ["theta1", "omega1"],
    "mass_2": ["theta2", "omega2"],
}
```

## Transfer Test

- **Source:** m1/m2 = 1.0 (balanced masses)
- **Target:** m1/m2 = 0.5 (unbalanced masses)

Split trajectories by mass ratio in the CSV.

## Experiments

| Exp | Model | Groups | What it tests |
|-----|-------|--------|---------------|
| P1 | Mean baseline | - | Sanity check |
| P2 | Last-value | - | Persistence baseline |
| P3 | Linear | - | Trivial model baseline |
| P4 | CI-Transformer | None | Channel-independent |
| P5 | Full-Attention | None | Learned relationships |
| P6 | **Role-Trans** | Physics | Our method |

## Success Criteria

| Metric | Target |
|--------|--------|
| In-domain MSE | Beat linear |
| Transfer ratio | Role-Trans < CI-Trans |
| Online (10%) | Faster adaptation than CI |

---

# Tier 2: C-MAPSS (Mechanical Real)

## Setup

**Data:** Already downloaded at `data/cmapss/`

**Sensors (14 informative after dropping constants):**
```python
CMAPSS_SENSORS = [
    "s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
    "s13", "s14", "s15", "s17", "s20", "s21"
]
```

**Physics Groups (turbofan components):**
```python
CMAPSS_GROUPS = {
    "fan": ["s2", "s8", "s12", "s21"],      # Fan-related sensors
    "hpc": ["s3", "s7", "s11", "s20"],      # High-pressure compressor
    "combustor": ["s9", "s14"],              # Combustor
    "turbine": ["s4", "s13"],                # Turbine
    "nozzle": ["s15", "s17"],                # Nozzle
}
```

## 2D Patch Treatment

```python
# Input: [B, T=30, C=14]
# Patch: [B, num_patches=6, C=14, patch_dim=5]

# Mamba: process each channel's patches (temporal)
# Attention: process across channels at each patch (spatial)
```

**Key insight from student's proposal:**
- Mamba drives down each "lane" (channel) through time
- Attention bridges across lanes at each patch/timestep

## Transfer Tests

| Source | Target | Challenge |
|--------|--------|-----------|
| FD001 | FD002 | 1 → 6 operating conditions |
| FD001 | FD003 | Same conditions, +fault mode |
| FD001 | FD004 | Different conditions + fault |

## Experiments

| Exp | Model | Groups | Transfer |
|-----|-------|--------|----------|
| C1 | Mean baseline | - | FD001→FD002 |
| C2 | Last-value | - | FD001→FD002 |
| C3 | Linear | - | FD001→FD002 |
| C4 | CI-Transformer | None | FD001→FD002 |
| C5 | Full-Attention | None | FD001→FD002 |
| C6 | **Mamba+Attn** | Physics | FD001→FD002 |
| C7 | **Mamba+Attn** | Physics | FD001→FD003 |
| C8 | **Mamba+Attn** | Physics | FD001→FD004 |

## Success Criteria

| Metric | Target | Previous Best |
|--------|--------|---------------|
| FD001 RMSE | < 13.0 | 12.22 ± 0.38 |
| FD001→FD002 ratio | < 4.0 | 4.00 |
| Beat CI-Trans on cross-condition | Yes | 36% better |

---

# Tier 3: Weather (Physical Real)

## Setup

**Data:** Download from GIFT-Eval or Jena Climate

```bash
# Option 1: GIFT-Eval
pip install gift-eval
python -c "from gift_eval.data import download; download('jena_weather')"

# Option 2: Direct download
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip
```

**Variables (21 total):**
```python
WEATHER_VARS = [
    "p", "T", "Tpot", "Tdew", "rh", "VPmax", "VPact", "VPdef",
    "sh", "H2OC", "rho", "wv", "max_wv", "wd", "rain", "raining",
    "SWDR", "PAR", "max_PAR", "Tlog", "CO2"
]
```

**Physics Groups:**
```python
WEATHER_GROUPS = {
    "temperature": ["T", "Tpot", "Tdew", "Tlog"],
    "pressure": ["p", "VPmax", "VPact", "VPdef"],
    "humidity": ["rh", "sh", "H2OC", "rho"],
    "wind": ["wv", "max_wv", "wd"],
    "radiation": ["SWDR", "PAR", "max_PAR"],
    "other": ["rain", "raining", "CO2"],
}
```

## Transfer Test

- **Source:** Year 2015
- **Target:** Year 2016

Temporal distribution shift (different weather patterns year-to-year).

## Experiments

| Exp | Model | Groups | Horizon |
|-----|-------|--------|---------|
| W1 | Mean baseline | - | 96 |
| W2 | Last-value | - | 96 |
| W3 | Linear | - | 96 |
| W4 | iTransformer | None | 96 |
| W5 | CI-Transformer | None | 96 |
| W6 | **Mamba+Attn** | Physics | 96 |
| W7 | **Mamba+Attn** | Physics | 336 |
| W8 | **Mamba+Attn** | Physics | 720 |

## Success Criteria

| Metric | Target | iTransformer Baseline |
|--------|--------|----------------------|
| H=96 MSE | < 0.174 | 0.174 |
| H=336 MSE | < 0.200 | ~0.20 |
| Transfer (2015→2016) | ratio < 1.5 | ? |

---

# Experiment Protocol

## For Each Tier

```python
for tier in [PENDULUM, CMAPSS, WEATHER]:
    # 1. Load data
    train, val, test = load_data(tier)

    # 2. Run trivial baselines
    for baseline in [mean, last_value, random]:
        log(baseline.evaluate(test))

    # 3. Run simple baselines
    for model in [linear, mlp]:
        model.fit(train)
        log(model.evaluate(test))

    # 4. Run architecture comparison
    for arch in [ci_trans, full_attn, mamba_attn]:
        for groups in [None, "physics"]:
            model = arch(groups=groups)
            model.fit(train)
            log(model.evaluate(test))

    # 5. Transfer experiment
    model = mamba_attn(groups="physics")
    model.fit(source_train)

    zero_shot = model.evaluate(target_test)
    model.finetune(target_train[:10%])
    adapted = model.evaluate(target_test[10%:])

    log(transfer_ratio=zero_shot / source_test)
```

## Seeds

**3 seeds minimum:** 42, 123, 456

Report: mean ± std

---

# Logging Requirements

## Every Experiment

```markdown
## Exp [Tier][N]: [Description]

**Time**: HH:MM
**Tier**: Pendulum / C-MAPSS / Weather
**Model**: [architecture]
**Groups**: None / Physics
**Hypothesis**: ...

**Results (3 seeds):**
| Metric | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|--------|---------|----------|----------|------------|
| MSE    | ...     | ...      | ...      | ...        |

**Transfer:**
| Setting | MSE | Ratio |
|---------|-----|-------|
| In-domain | ... | 1.0 |
| Zero-shot | ... | ... |
| 10%-adapted | ... | ... |

**Verdict**: KEEP / REVERT
**Insight**: ...
**Next**: ...
```

## Commit Protocol

```bash
# After each tier completion
git add -A && git commit -m "Tier N: [summary of results]" && git push
```

---

# Success Metrics

| Tier | Metric | Target | Stretch |
|------|--------|--------|---------|
| 1. Pendulum | Transfer ratio | Role < CI | Role < 0.5 * CI |
| 2. C-MAPSS | FD001→FD002 ratio | < 4.0 | < 3.5 |
| 3. Weather | H=96 MSE | < 0.174 | < 0.165 |
| **All** | Physics beats CI | 2/3 tiers | 3/3 tiers |

**Paper threshold:** Beat CI-Trans on 2/3 tiers with physics grouping.

---

# What NOT to Do

- Don't use RevIN on C-MAPSS (confirmed harmful)
- Don't use JEPA pretraining (confirmed harmful for transfer)
- Don't skip trivial baselines (mean, last-value, linear)
- Don't run >10 min per experiment (fast iteration)
- Don't claim "breakthrough" without statistical significance
- Don't modify core `src/industrialjepa/` unless necessary

---

# File Structure When Done

```
data/
├── pendulum.csv              # Generated (Tier 1)
├── cmapss/                   # Existing (Tier 2)
└── weather/                  # Downloaded (Tier 3)

autoresearch/experiments/
├── generate_pendulum.py      # One-time generation
├── mamba_attention.py        # Core architecture (2D treatment)
├── tier1_pendulum.py         # Tier 1 experiments
├── tier2_cmapss.py           # Tier 2 experiments
├── tier3_weather.py          # Tier 3 experiments
└── baselines.py              # Trivial + simple baselines

autoresearch/
├── EXPERIMENT_LOG.md         # Updated with all results
└── program_v2.md             # This file
```

---

# Architecture Implementation Notes

## Mamba + Attention (2D Treatment)

```python
class MambaAttentionBlock(nn.Module):
    """
    2D treatment: Mamba for time, Attention for space.

    Input: [B, T, C, D]  (Batch, Time, Channels, Features)
    """
    def __init__(self, d_model, n_channels, groups=None):
        self.mamba = Mamba(d_model)  # Shared or per-group
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.groups = groups  # Physics groupings

    def forward(self, x):
        B, T, C, D = x.shape

        # Step 1: Mamba along time (within each channel)
        x = rearrange(x, 'b t c d -> (b c) t d')
        x = self.mamba(x)  # Each channel processed independently
        x = rearrange(x, '(b c) t d -> b t c d', c=C)

        # Step 2: Attention across channels (at each timestep)
        # If physics groups: use attention mask
        x = rearrange(x, 'b t c d -> (b t) c d')
        if self.groups:
            mask = self.make_group_mask(self.groups)
            x = self.attention(x, x, x, attn_mask=mask)
        else:
            x = self.attention(x, x, x)
        x = rearrange(x, '(b t) c d -> b t c d', t=T)

        return x
```

## Physics Group Mask

```python
def make_group_mask(groups, n_channels):
    """
    Create attention mask that only allows within-group + cross-group attention.
    Within-group: full attention (physics relationships)
    Cross-group: single representative per group (compositional)
    """
    mask = torch.zeros(n_channels, n_channels)
    for group_channels in groups.values():
        for i in group_channels:
            for j in group_channels:
                mask[i, j] = 1  # Within-group: allow
    # Add cross-group connections (first channel of each group)
    representatives = [channels[0] for channels in groups.values()]
    for i in representatives:
        for j in representatives:
            mask[i, j] = 1
    return mask
```

---

# When You're Done

1. Update `EXPERIMENT_LOG.md` with all results
2. Write summary table comparing all tiers
3. Honest assessment: Does physics grouping help universally?
4. Identify which transfer scenarios benefit most
5. Recommend next steps

**The claim (if results support):**
> "Physics-informed channel grouping with 2D Mamba-Attention processing provides consistent transfer improvements across synthetic (double pendulum), mechanical (turbofan), and physical (weather) systems, beating channel-independent and learned-attention baselines by X-Y%."

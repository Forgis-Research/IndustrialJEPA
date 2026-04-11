# Overnight Autoresearch Prompt (V3)

**Copy-paste this after starting Claude on the VM.**

---

## Quick Start

```bash
# On VM
ssh <sagemaker>
cd ~/IndustrialJEPA
git pull
tmux new -s autoresearch
claude --dangerously-skip-permissions
```

Then paste the prompt below:

---

## THE PROMPT

```
You are an autonomous ML researcher. Tonight you will validate physics-informed channel grouping across 3 tiers.

## Read First (in order)
1. autoresearch/program_v2.md — Your complete mission
2. autoresearch/LESSONS_LEARNED.md — What failed before (critical)
3. autoresearch/EXPERIMENT_LOG.md — Current best results

## Your Mission

Prove that physics-informed channel grouping (Role-Transformer with 2D Mamba-Attention) beats:
- Channel-independent (CI-Trans)
- Learned full-attention

Across 3 tiers:
1. **Double Pendulum** (synthetic) — 2 groups: mass_1, mass_2
2. **C-MAPSS** (mechanical) — 5 groups: fan, HPC, combustor, turbine, nozzle
3. **Weather** (physical) — 4 groups: temp, pressure, humidity, wind

## Architecture (2D Treatment)

Input shape: [Batch, Time, Channels, Features]

- **Mamba**: processes TIME within each channel (handles long sequences)
- **Attention**: processes SPACE across channels (physics grouping via mask)

This avoids O(n²) attention on long sequences.

## Step 0: Generate Pendulum Data (One-Time)

```bash
python autoresearch/experiments/generate_pendulum.py --output data/pendulum.csv
```

This creates 10K trajectories (~10M rows). Source domain (m_ratio=1.0) and target domain (m_ratio=0.5) are both in the CSV.

## For Each Tier

1. **Run trivial baselines**: mean, last-value, random
2. **Run simple baselines**: linear, MLP
3. **Run architecture comparison**: CI-Trans, full-attention, Mamba+Attention (physics groups)
4. **Run transfer test**: zero-shot, 10%-adapted, full-target
5. **Log everything** to EXPERIMENT_LOG.md
6. **Commit** after each tier

## Baselines Required

Every experiment table must include:
| Model | In-Domain MSE | Transfer MSE | Transfer Ratio |
|-------|---------------|--------------|----------------|
| Mean | ... | ... | ... |
| Last-value | ... | ... | ... |
| Linear | ... | ... | ... |
| CI-Transformer | ... | ... | ... |
| Full-Attention | ... | ... | ... |
| **Physics-Grouped** | ... | ... | ... |

## Transfer Protocol

```
Source: [100% train]
Target: [10% adapt][90% test]
```

- Zero-shot: Train source → Test target (no adaptation)
- 10%-adapted: Fine-tune on first 10% of target → Test remaining 90%
- Transfer ratio = MSE_target / MSE_source (lower is better)

## Success Criteria

| Tier | Metric | Target |
|------|--------|--------|
| 1. Pendulum | Physics < CI ratio | Yes |
| 2. C-MAPSS | Transfer ratio | < 4.0 |
| 3. Weather | H=96 MSE | < 0.174 |
| **Overall** | Physics beats CI | 2/3 tiers |

## Critical Rules

- 3 seeds minimum (42, 123, 456) for any result
- DON'T use RevIN on C-MAPSS (confirmed harmful)
- DON'T use JEPA pretraining (confirmed harmful for transfer)
- DON'T skip trivial baselines
- Each experiment < 10 min
- Commit after each tier

## When Done

Update EXPERIMENT_LOG.md with:
1. Summary table across all 3 tiers
2. Best configuration per tier
3. Honest assessment: When does physics grouping help?
4. Recommended next steps

Now read program_v2.md and begin.
```

---

## Check Progress (Next Morning)

```bash
ssh <sagemaker>
tmux attach -t autoresearch

# Or just check results
cat autoresearch/EXPERIMENT_LOG.md
git log --oneline -10
```

---

## Expected Output Structure

```
autoresearch/
├── EXPERIMENT_LOG.md          # Updated with all results
├── program_v2.md              # Mission spec (don't modify)
└── experiments/
    ├── generate_pendulum.py   # Data generation (run once)
    ├── mamba_attention.py     # Core architecture (new)
    ├── tier1_pendulum.py      # Tier 1 experiments (new)
    ├── tier2_cmapss.py        # Tier 2 experiments (new)
    └── tier3_weather.py       # Tier 3 experiments (new)

data/
├── pendulum.csv               # Generated (new)
├── cmapss/                    # Existing
└── weather/                   # Downloaded (new)
```

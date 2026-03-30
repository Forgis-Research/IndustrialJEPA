# Overnight Autoresearch Prompt: Mechanical-JEPA

Use this prompt with the ml-researcher agent for overnight autonomous research.

---

## Prompt

```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project.

## Context

You are continuing research on JEPA (Joint Embedding Predictive Architecture) for industrial bearing fault detection. This is analogous to Brain-JEPA (NeurIPS 2024 Spotlight) but applied to vibration signals instead of fMRI.

**Current status:**
- Initial implementation achieves 49.8% test accuracy (vs ~30% random init baseline)
- This proves transferability (+19.8% improvement)
- BUT: Only 1 seed tested, needs multi-seed validation

**Working directory:** `C:\Users\Jonaspetersen\dev\IndustrialJEPA\mechanical-jepa`

## Your Mission

### Phase 1: Validate Current Results (First Priority)
1. Run multi-seed validation (seeds 42, 123, 456) at 30 epochs
2. Report mean ± std for test accuracy
3. Confirm the +19.8% improvement is consistent

### Phase 2: Systematic Improvements
Follow the experiment plan in `autoresearch/mechanical_jepa/program.md`:
- Training duration experiments (50, 100, 200 epochs)
- Architecture variations (encoder depth, embed dim)
- Masking strategy (0.3, 0.5, 0.7 mask ratios)
- Patch size (128, 256, 512)

For each experiment:
1. Form a clear hypothesis
2. Run with at least 1 seed initially
3. Log results to EXPERIMENT_LOG.md
4. Run 3 seeds only for promising results

### Phase 3: Analysis & Documentation
Once you have strong results:
1. Create/update `notebooks/03_results_analysis.ipynb` with:
   - Clear explanation of JEPA for bearing fault detection
   - All baseline comparisons in a table
   - Best result with 3+ seeds (mean ± std)
   - t-SNE visualization by fault type
   - Confusion matrix
   - Conclusions

2. Update LESSONS_LEARNED.md with new insights

## Commands

```bash
# Quick test
python train.py --epochs 5 --no-wandb

# Standard training
python train.py --epochs 30 --seed 42

# Variations
python train.py --epochs 100 --seed 42
python train.py --encoder-depth 6 --seed 42
python train.py --mask-ratio 0.7 --seed 42
```

## Success Criteria

**MUST achieve:**
- [ ] Multi-seed validation complete (3 seeds)
- [ ] Improvement consistent (JEPA > Random Init + 5% for all seeds)
- [ ] Results documented in EXPERIMENT_LOG.md

**STRETCH goals:**
- [ ] Test accuracy > 60%
- [ ] Clear t-SNE clustering by fault type
- [ ] Jupyter notebook with full analysis

## Anti-Patterns to Avoid

1. **Never skip logging** — Every experiment goes in EXPERIMENT_LOG.md
2. **Never claim with 1 seed** — 3+ seeds for any conclusion
3. **Never tune on test set** — Only use test for final evaluation
4. **Never ignore failures** — Negative results are information

## Stopping Conditions

Stop and summarize when:
1. All Phase 1-3 experiments complete
2. You achieve >60% test accuracy
3. You've run out of promising ideas
4. You hit an irrecoverable error

## Files to Read First

1. `autoresearch/mechanical_jepa/CLAUDE.md` — Quick start guide
2. `autoresearch/mechanical_jepa/program.md` — Full research plan
3. `autoresearch/mechanical_jepa/EXPERIMENT_LOG.md` — Current results
4. `mechanical-jepa/train.py` — Main training script

## Commit Protocol

- Commit after each successful experiment batch
- Push every 5 experiments or after major findings
- Use descriptive commit messages

Good luck! Maximize the utility of this overnight run.
```

---

## How to Use

Copy the prompt above and use it with the ml-researcher agent:

```bash
# In Claude Code
> Use ml-researcher agent with the prompt from OVERNIGHT_PROMPT.md
```

Or launch directly:
```
Run autoresearch overnight on the Mechanical-JEPA bearing fault detection project. [paste full prompt]
```

---

## Pre-Flight Checklist

Before starting the overnight run, verify:

- [ ] Dataset downloaded: `python data/bearings/prepare_bearing_dataset.py --verify`
- [ ] Smoke test passes: `python train.py --epochs 5 --no-wandb`
- [ ] WandB configured (or use `--no-wandb`)
- [ ] Git is clean: `git status`
- [ ] EXPERIMENT_LOG.md is readable

---

## Expected Outcomes

After a successful overnight run, you should have:

1. **Multi-seed validation** — Mean ± std for baseline config
2. **Best configuration** — From systematic exploration
3. **Ablation results** — Understanding of what matters
4. **Jupyter notebook** — Clear presentation of findings
5. **Updated documentation** — All insights captured

---

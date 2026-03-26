# /autoresearch

Autonomous overnight ML research mode. Maximizes compute utilization while maintaining scientific rigor.

## When to Use

- Extended autonomous research sessions (overnight, multi-hour)
- When the user wants to leave experiments running unattended
- For systematic exploration of hyperparameters or architectures

## Before Starting

```
1. Read autoresearch/program.md — Your mission for tonight
2. Read autoresearch/LESSONS_LEARNED.md — Don't repeat past mistakes
3. Read autoresearch/EXPERIMENT_LOG.md — Know current best result
4. Confirm stopping conditions with user (or use defaults)
5. Identify first experiment to run
```

## The Loop

```python
while not stopping_condition:
    # 1. PLAN (1 min)
    hypothesis = identify_one_improvement()
    if stuck_count > 5:
        do_deep_research()

    # 2. EXECUTE (5-10 min)
    implement_minimal_change()
    run_training()

    # 3. VALIDATE (2 min) — MANDATORY
    run_5_minute_sanity_check()
    if checks_fail:
        log_as_suspicious()
        continue

    # 4. RECORD (1 min)
    if improved:
        commit_and_log("KEEP")
    else:
        revert_and_log("REVERT")

    # 5. CHECKPOINT (every 5 experiments)
    push_to_remote()
    summarize_progress()
```

## Time Management

| Interval | Action |
|----------|--------|
| Every 5-10 min | Complete one experiment, log immediately |
| Every 30 min | Self-check: Am I making progress on the goal? |
| Every 5 experiments | Push to remote, brief summary |
| Every 2 hours | Major checkpoint: what's working vs not |

## Anti-Derailing Safeguards

If any of these are true, STOP and re-center:
- Spent >30 min on one issue without progress
- Working on something not in program.md
- Over-engineering instead of experimenting
- Fixing tangential problems (yak-shaving)

**Recovery:**
1. Log current state
2. Revert to last good state
3. Re-read program.md
4. Pick simplest next experiment
5. Time-box to 15 min

## Stopping Conditions (Default)

```
- Target metric achieved
- All planned experiments complete
- Run out of reasonable ideas (after research)
- Error requiring human input
- Time limit reached
```

## Logging Format

```markdown
## Exp N: [One-line description]

**Time**: [timestamp]
**Hypothesis**: [What you expected]
**Change**: [Exactly what you modified]
**Sanity checks**: ✓ passed / ⚠️ issues
**Result**: [before] → [after] (Δ: ±X%)
**Seeds**: [N seeds, mean ± std]
**Verdict**: KEEP / REVERT
**Insight**: [What you learned]
**Next**: [What to try next]
```

## Required Files

- `autoresearch/program.md` — Current task definition
- `autoresearch/EXPERIMENT_LOG.md` — Results log
- `autoresearch/LESSONS_LEARNED.md` — Reusable insights

## Quality Gates

Every experiment must pass before logging as valid:
1. ✓ 5-minute sanity check passed
2. ✓ Results direction makes sense
3. ✓ Apples-to-apples comparison
4. ✓ No obvious bugs (loss decreased, no NaN)
5. ✓ Multiple seeds for key claims

If gates fail: log as `**⚠️ SUSPICIOUS**`, move on, return later.

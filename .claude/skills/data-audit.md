# /data-audit

Quick quality assessment for any dataset. Run this before using data for training.

## When to Use

- First time looking at a new dataset
- After any data pipeline changes
- When results seem suspicious (data issue?)
- Before committing processed data

## The 10-Minute Audit

```python
def audit(df, name="dataset"):
    print(f"=== AUDIT: {name} ===\n")

    # 1. SHAPE
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    # 2. TYPES
    print("Types:")
    print(df.dtypes.value_counts())
    print()

    # 3. MISSING
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    if missing.sum() > 0:
        print("Missing values:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
        print()
    else:
        print("Missing: None\n")

    # 4. DUPLICATES
    n_dup = df.duplicated().sum()
    print(f"Duplicates: {n_dup:,} ({n_dup/len(df)*100:.1f}%)\n")

    # 5. NUMERIC SUMMARY
    numeric = df.select_dtypes(include=[np.number])
    if len(numeric.columns) > 0:
        print("Numeric ranges:")
        for col in numeric.columns:
            print(f"  {col}: [{df[col].min():.2f}, {df[col].max():.2f}] "
                  f"mean={df[col].mean():.2f} std={df[col].std():.2f}")
        print()

    # 6. CATEGORICAL SUMMARY
    categorical = df.select_dtypes(include=['object', 'category'])
    if len(categorical.columns) > 0:
        print("Categorical:")
        for col in categorical.columns:
            n_unique = df[col].nunique()
            print(f"  {col}: {n_unique} unique")
            if n_unique <= 10:
                print(f"    Values: {df[col].unique().tolist()}")
        print()

    # 7. QUICK CHECKS
    print("Quick checks:")
    print(f"  [{'✓' if df.notna().any().all() else '✗'}] All columns have data")
    print(f"  [{'✓' if n_dup == 0 else '✗'}] No duplicates")
    print(f"  [{'✓' if missing.sum() == 0 else '✗'}] No missing values")

    # Check for constant columns
    constant = [c for c in df.columns if df[c].nunique() <= 1]
    print(f"  [{'✓' if len(constant) == 0 else '✗'}] No constant columns")
    if constant:
        print(f"    Constant: {constant}")
```

## Red Flags

| Flag | Action |
|------|--------|
| >5% missing | Investigate cause before imputing |
| Duplicates | Check if ETL bug or legitimate |
| Constant columns | Drop (useless) |
| High cardinality | May need encoding strategy |
| Outliers >5 std | Verify sensor range |
| Perfect r=1.0 | Check for derived/leaked columns |

## Output Format

```markdown
## Data Audit: [Dataset Name]

**Date**: [timestamp]
**Shape**: N rows × M cols
**Memory**: X MB

### Quality Summary
| Metric | Value | Status |
|--------|-------|--------|
| Completeness | X% | ✓/⚠️ |
| Uniqueness | X% | ✓/⚠️ |
| Valid ranges | X% | ✓/⚠️ |

### Issues Found
1. [Issue description]
2. [Issue description]

### Recommendations
1. [What to fix]
2. [What to investigate]
```

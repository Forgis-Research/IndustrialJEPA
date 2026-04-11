# IndustrialJEPA - Technical Debt & TODO

## High Priority

### Memory: Dataset loaded 3x for train/val/test splits
**Status:** FIXED
**Impact:** OOM on 16GB machines with full AURSAD dataset (6.2M rows)
**Symptoms:** Process killed after ~15min during validation dataset loading

**Root cause:** `FactoryNetDataset` loads the entire HuggingFace dataset into pandas for each split (train, val, test). With 6.2M rows × 3 splits = ~18M rows in memory.

**Current workaround:** Use `--data-source hackathon` for small dataset

**Proposed fix:**
1. Load dataset once in `create_world_model_dataloaders()`
2. Pass shared DataFrame to each split's `FactoryNetDataset`
3. Each split filters to its episodes without duplicating data

**Files to modify:**
- `src/industrialjepa/data/factorynet.py` - Accept pre-loaded DataFrame
- `src/industrialjepa/data/world_model_dataset.py` - Share data between splits

---

### Memory: Episode metadata extraction is O(n²)
**Status:** FIXED (uses groupby now)
**Impact:** ~15min to extract metadata from 4k episodes in 6.2M rows

**Root cause:** For each episode, we filter the entire DataFrame:
```python
ep_data = self.df[self.df["episode_id"] == ep_id].iloc[0]
```

**Proposed fix:**
1. Use `groupby("episode_id").first()` to get one row per episode in one pass
2. Or build episode index during initial load

---

## Medium Priority

### Add `--max-episodes` flag for quick iteration
Allow limiting number of episodes for faster debugging:
```bash
python scripts/train_world_model.py --max-episodes 100 --epochs 5
```

### Cross-machine transfer experiments
- Train on AURSAD, evaluate on Voraus
- Both are 6-DOF robots, should share physics
- Need unified effort signal handling (current vs torque)

---

## Low Priority

### Streaming mode for very large datasets
HuggingFace datasets streaming to avoid full download:
```python
load_dataset(..., streaming=True)
```
Tradeoff: No full shuffle, only buffer shuffle.

### Multi-GPU training
Add DistributedDataParallel support for faster training on multi-GPU instances.

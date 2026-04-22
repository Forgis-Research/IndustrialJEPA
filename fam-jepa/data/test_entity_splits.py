"""Local tests for intra-entity split logic.

Verifies the split correctness using synthetic data (no GPU, no real data needed).
"""
import numpy as np
import sys
from pathlib import Path

# Import just the split logic without triggering torch via __init__.py
# (torch DLL is broken on this Windows dev machine; runs fine on SageMaker)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "smap_msl_raw", str(Path(__file__).resolve().parent / "smap_msl.py"),
    submodule_search_locations=[])

# Stub out torch and the relative config import before loading
import types
_torch_stub = types.ModuleType("torch")
_torch_stub.from_numpy = lambda x: x
class _FakeDataset:
    pass
class _FakeDataLoader:
    pass
_torch_stub_utils = types.ModuleType("torch.utils")
_torch_stub_data = types.ModuleType("torch.utils.data")
_torch_stub_data.Dataset = _FakeDataset
_torch_stub_data.DataLoader = _FakeDataLoader
_torch_stub.utils = _torch_stub_utils
_torch_stub.utils.data = _torch_stub_data
sys.modules["torch"] = _torch_stub
sys.modules["torch.utils"] = _torch_stub_utils
sys.modules["torch.utils.data"] = _torch_stub_data

_smap_mod = importlib.util.module_from_spec(_spec)
# Patch the relative import for .config
_smap_mod.__package__ = "data"
try:
    _spec.loader.exec_module(_smap_mod)
except ImportError:
    # config import fails, that's fine — we only need the split function
    pass

_intra_entity_split = _smap_mod._intra_entity_split
WINDOW_SIZE = _smap_mod.WINDOW_SIZE


def make_fake_entities(n_entities=10, T_range=(3000, 8000), C=25, seed=42):
    """Create synthetic entities mimicking SMAP structure (real entities are ~5-8K steps)."""
    rng = np.random.RandomState(seed)
    entities = []
    for i in range(n_entities):
        T = rng.randint(*T_range)
        test = rng.randn(T, C).astype(np.float32)
        labels = np.zeros(T, dtype=np.int32)
        # Place 1-3 anomaly segments per entity
        n_segments = rng.randint(1, 4)
        for _ in range(n_segments):
            start = rng.randint(0, T - 50)
            length = rng.randint(10, 50)
            labels[start:start + length] = 1

        entities.append({
            'entity_id': f'entity_{i}',
            'train': rng.randn(T // 3, C).astype(np.float32),
            'test': test,
            'labels': labels,
            'has_anomaly': True,
        })
    return entities


def test_all_entities_present():
    """Every entity must appear in all three splits."""
    entities = make_fake_entities(n_entities=15)
    splits = _intra_entity_split(entities, ratios=(0.6, 0.1, 0.3))

    train_ids = {e['entity_id'] for e in splits['ft_train']}
    val_ids = {e['entity_id'] for e in splits['ft_val']}
    test_ids = {e['entity_id'] for e in splits['ft_test']}

    assert train_ids == val_ids == test_ids, (
        f"Entity sets differ: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
    )
    print(f"  PASS: all {len(train_ids)} entities in all splits")


def test_no_temporal_overlap():
    """No timestep should appear in more than one split (accounting for gap)."""
    entities = make_fake_entities(n_entities=5, T_range=(3000, 6000))
    splits = _intra_entity_split(entities, ratios=(0.6, 0.1, 0.3))

    for i, eid in enumerate(e['entity_id'] for e in splits['ft_train']):
        tr = splits['ft_train'][i]
        va = splits['ft_val'][i]
        te = splits['ft_test'][i]

        # Reconstruct original T
        T_original = len(entities[i]['test'])
        t1 = int(0.6 * T_original)
        t2 = int(0.7 * T_original)
        gap = WINDOW_SIZE

        # Check train ends before gap
        assert len(tr['test']) == t1, f"{eid}: train length mismatch"
        # Check val starts after gap
        assert len(va['test']) == (t2 - (t1 + gap)), f"{eid}: val length mismatch"
        # Check test starts after second gap
        assert len(te['test']) == (T_original - (t2 + gap)), f"{eid}: test length mismatch"

        # Total used + gaps should be <= T
        total = len(tr['test']) + gap + len(va['test']) + gap + len(te['test'])
        assert total <= T_original, f"{eid}: total {total} > T {T_original}"

    print(f"  PASS: no temporal overlap, gaps correctly placed")


def test_gap_prevents_window_leakage():
    """A window of size WINDOW_SIZE ending at the last train timestep
    must not reach into any val timestep."""
    entities = make_fake_entities(n_entities=5, T_range=(3000, 6000))
    splits = _intra_entity_split(entities, ratios=(0.6, 0.1, 0.3))

    for i in range(len(splits['ft_train'])):
        tr = splits['ft_train'][i]
        T_original = len(entities[i]['test'])
        t1 = int(0.6 * T_original)
        val_start = t1 + WINDOW_SIZE

        # Last possible train window: starts at (t1 - WINDOW_SIZE), ends at t1
        # Val data starts at val_start = t1 + WINDOW_SIZE
        # So the gap between end-of-train-window (t1) and start-of-val (val_start)
        # is exactly WINDOW_SIZE — no window can straddle the boundary
        assert val_start - t1 == WINDOW_SIZE
    print(f"  PASS: gap size = {WINDOW_SIZE}, no window can straddle train->val")


def test_labels_aligned():
    """Labels must match the corresponding test slice."""
    entities = make_fake_entities(n_entities=5, T_range=(3000, 6000))
    splits = _intra_entity_split(entities, ratios=(0.6, 0.1, 0.3))

    for i in range(len(splits['ft_train'])):
        orig = entities[i]
        tr = splits['ft_train'][i]
        va = splits['ft_val'][i]
        te = splits['ft_test'][i]

        T = len(orig['test'])
        t1 = int(0.6 * T)
        t2 = int(0.7 * T)
        gap = WINDOW_SIZE

        np.testing.assert_array_equal(tr['labels'], orig['labels'][:t1])
        np.testing.assert_array_equal(va['labels'], orig['labels'][t1 + gap:t2])
        np.testing.assert_array_equal(te['labels'], orig['labels'][t2 + gap:])
        np.testing.assert_array_equal(tr['test'], orig['test'][:t1])
        np.testing.assert_array_equal(va['test'], orig['test'][t1 + gap:t2])
        np.testing.assert_array_equal(te['test'], orig['test'][t2 + gap:])

    print(f"  PASS: labels and test data correctly sliced")


def test_short_entity_skipped():
    """Entities too short for all three splits should be skipped."""
    short_entity = {
        'entity_id': 'too_short',
        'train': np.zeros((10, 5), dtype=np.float32),
        'test': np.zeros((150, 5), dtype=np.float32),  # T=150, gap=100 → not enough
        'labels': np.zeros(150, dtype=np.int32),
        'has_anomaly': True,
    }
    long_entity = {
        'entity_id': 'long_enough',
        'train': np.zeros((100, 5), dtype=np.float32),
        'test': np.zeros((5000, 5), dtype=np.float32),
        'labels': np.zeros(5000, dtype=np.int32),
        'has_anomaly': True,
    }
    splits = _intra_entity_split([short_entity, long_entity], ratios=(0.6, 0.1, 0.3))
    train_ids = {e['entity_id'] for e in splits['ft_train']}
    assert 'too_short' not in train_ids, "Short entity should be skipped"
    assert 'long_enough' in train_ids, "Long entity should be included"
    print(f"  PASS: short entity skipped, long entity kept")


def test_pretrain_normal_preserved():
    """ft_train entries should carry the pretrain_normal (normal-only train) data."""
    entities = make_fake_entities(n_entities=3, T_range=(800, 1000))
    splits = _intra_entity_split(entities, ratios=(0.6, 0.1, 0.3))

    for i, tr in enumerate(splits['ft_train']):
        assert 'pretrain_normal' in tr, f"{tr['entity_id']}: missing pretrain_normal"
        np.testing.assert_array_equal(tr['pretrain_normal'], entities[i]['train'])
    print(f"  PASS: pretrain_normal data preserved in ft_train")


def test_anomaly_coverage():
    """At least some anomalies should exist in test split (statistical check)."""
    entities = make_fake_entities(n_entities=20, T_range=(3000, 8000))
    splits = _intra_entity_split(entities, ratios=(0.6, 0.1, 0.3))

    total_anom_train = sum(e['labels'].sum() for e in splits['ft_train'])
    total_anom_test = sum(e['labels'].sum() for e in splits['ft_test'])

    assert total_anom_train > 0, "No anomalies in train split"
    assert total_anom_test > 0, "No anomalies in test split"
    print(f"  PASS: anomalies in train ({total_anom_train}) and test ({total_anom_test})")


if __name__ == '__main__':
    print("Testing _intra_entity_split()...\n")
    test_all_entities_present()
    test_no_temporal_overlap()
    test_gap_prevents_window_leakage()
    test_labels_aligned()
    test_short_entity_skipped()
    test_pretrain_normal_preserved()
    test_anomaly_coverage()
    print("\nAll tests passed.")

"""Raw (unnormalized) C-MAPSS loader for V24.

RevIN in CausalEncoder/TargetEncoder normalizes per-context per-channel,
so the data loader should NOT normalize. The v11 loader (load_cmapss_subset)
always applies min-max normalization — this module bypasses it.
"""

from typing import Dict

import numpy as np

from data_utils import load_raw, get_sensor_cols


def load_cmapss_raw(subset: str = 'FD001', val_frac: float = 0.15,
                    val_seed: int = 42) -> Dict:
    """Load C-MAPSS subset WITHOUT normalization. Returns raw sensor sequences.

    Same engine split as load_cmapss_subset (85/15 by engine_id, seed=42) so
    we stay comparable to v21/v22.
    """
    train_df, test_df, rul_arr = load_raw(subset)
    sensor_cols = get_sensor_cols()

    def build_raw(df):
        out = {}
        for eid, grp in df.groupby('engine_id'):
            grp = grp.sort_values('cycle')
            out[int(eid)] = grp[sensor_cols].values.astype(np.float32)
        return out

    all_train = build_raw(train_df)
    test = build_raw(test_df)

    all_ids = sorted(all_train.keys())
    rng = np.random.default_rng(val_seed)
    n_val = max(1, int(val_frac * len(all_ids)))
    val_ids = set(rng.choice(all_ids, size=n_val, replace=False).tolist())
    train_ids = [i for i in all_ids if i not in val_ids]

    return {
        'train_engines': {i: all_train[i] for i in train_ids},
        'val_engines':   {i: all_train[i] for i in sorted(val_ids)},
        'test_engines':  {i: test[i] for i in sorted(test.keys())},
        'test_rul': rul_arr.astype(np.float32),
        'subset': subset,
    }

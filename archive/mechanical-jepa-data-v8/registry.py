"""
Dataset registry: metadata for all 8 bearing sources.
Single source of truth for native SR, channel map, domain, and compatibility.
"""

SOURCES = {
    'cwru': {
        'native_sr': 12000,
        'channels': ['drive_end', 'fan_end'],
        'primary_channel': 0,
        'domain': 'bearing_fault_classification',
        'has_run_to_failure': False,
        'machine_type': 'motor_bearing',
        'load_conditions': [0, 1, 2, 3],  # HP
        'fault_types': ['normal', 'inner_race', 'outer_race', 'ball'],
        'fault_diameter_mils': [7, 14, 21],
        'n_signals': 40,
        'notes': 'Pre-seeded faults, no degradation trajectory. Fixed faults.',
        'hf_parquet': 'extra_cwru_mfpt.parquet',
    },
    'femto': {
        'native_sr': 25600,
        'channels': ['horizontal_accel', 'vertical_accel'],
        'primary_channel': 0,
        'domain': 'bearing_run_to_failure',
        'has_run_to_failure': True,
        'machine_type': 'ball_bearing',
        'load_conditions': ['1800rpm_4kN', '1650rpm_4.2kN', '1500rpm_5kN'],
        'n_episodes': 17,
        'snapshot_interval_s': 10,
        'snapshot_duration_s': 0.1,
        'lifetime_range_h': [1.0, 7.0],
        'notes': 'PHM 2012 challenge. 0.1s snapshots every 10s. Very short snapshots (~2560 samples).',
        'hf_parquet': 'train-{shard:05d}-of-00005.parquet',
        'shards': [0, 1, 2, 3],
    },
    'xjtu_sy': {
        'native_sr': 25600,
        'channels': ['horizontal_accel', 'vertical_accel'],
        'primary_channel': 0,
        'domain': 'bearing_run_to_failure',
        'has_run_to_failure': True,
        'machine_type': 'ball_bearing',
        'load_conditions': ['35Hz_12kN', '37.5Hz_11kN', '40Hz_10kN'],
        'n_episodes': 15,  # 5 per condition
        'snapshot_interval_s': 60,
        'snapshot_duration_s': 1.28,
        'lifetime_range_h': [0.5, 2.6],
        'notes': 'XJTU-SY 2019. 1.28s snapshots every 60s. More variable lifetimes than FEMTO.',
        'hf_parquet': 'train-{shard:05d}-of-00005.parquet',
        'shards': [3],
    },
    'ims': {
        'native_sr': 20480,
        'channels': ['bearing1', 'bearing2', 'bearing3', 'bearing4'],
        'primary_channel': 0,
        'domain': 'bearing_run_to_failure',
        'has_run_to_failure': True,
        'machine_type': 'roller_bearing',
        'load_conditions': ['2000rpm_6000lb'],
        'n_episodes': 4,  # 3 test runs with 4 bearings each
        'snapshot_interval_s': 600,  # 10 minutes
        'snapshot_duration_s': 1.0,
        'lifetime_range_h': [50, 164],
        'notes': 'IMS NASA dataset. 4 bearings per run. Long runs (days).',
        'hf_parquet': 'extra_ims.parquet',
    },
    'mfpt': {
        'native_sr': 48828,
        'channels': ['vibration'],
        'primary_channel': 0,
        'domain': 'bearing_fault_classification',
        'has_run_to_failure': False,
        'machine_type': 'bearing',
        'load_conditions': [270, 25, 50, 100, 150, 200, 250, 300],  # lbs
        'fault_types': ['normal', 'inner_race', 'outer_race'],
        'n_signals': 23,
        'notes': 'MFPT Society dataset. Variable loads. No degradation trajectory.',
        'hf_parquet': 'extra_cwru_mfpt.parquet',
    },
    'paderborn': {
        'native_sr': 64000,
        'channels': ['vibration', 'microphone'],
        'primary_channel': 0,
        'domain': 'bearing_fault_classification',
        'has_run_to_failure': False,
        'machine_type': 'ball_bearing',
        'load_conditions': ['N09_M07_F10', 'N15_M01_F10', 'N15_M07_F04'],
        'fault_types': ['normal', 'artificial', 'real_degradation'],
        'n_signals': 32,
        'notes': 'Paderborn KAT dataset. Highest native SR. Mix of artificial and real faults.',
        'hf_parquet': 'train-{shard:05d}-of-00005.parquet',
        'shards': [4],
    },
    'ottawa': {
        'native_sr': 42000,
        'channels': ['vibration'],
        'primary_channel': 0,
        'domain': 'bearing_run_to_failure',
        'has_run_to_failure': True,
        'machine_type': 'ball_bearing',
        'load_conditions': ['0.6hp', '1.2hp', '1.8hp', '2.4hp'],
        'n_episodes': 3,  # small dataset
        'snapshot_interval_s': None,  # continuous recording
        'notes': 'Ottawa bearing dataset. Relatively short run-to-failure. 4 load conditions.',
        'hf_parquet': 'ottawa_bearings.parquet',
    },
    'mafaulda': {
        'native_sr': 50000,
        'channels': ['underhang_accel', 'tach', 'overhang_accel_radial', 'overhang_accel_tangential'],
        'primary_channel': 0,
        'domain': 'rotating_machine_fault_classification',
        'has_run_to_failure': False,
        'machine_type': 'centrifugal_pump_bearing',
        'load_conditions': [0, 0.25, 0.5, 0.75, 1.0],  # kgf
        'fault_types': ['normal', 'imbalance', 'horizontal_misalignment', 'vertical_misalignment',
                        'overhang_ball', 'overhang_inner_race', 'overhang_outer_race',
                        'underhang_ball', 'underhang_inner_race', 'underhang_outer_race'],
        'n_signals': 1951,
        'notes': 'MAFAULDA (Machinery Fault Database). Centrifugal pump, not pure bearing. '
                 'Most diverse fault types. Different machine type from bearing RUL sources.',
        'hf_parquet': 'mafaulda_{shard:03d}.parquet',
        'shards': list(range(8)),
    },
}

# Convenience groupings
RUL_SOURCES = ['femto', 'xjtu_sy', 'ims', 'ottawa']
CLASSIFICATION_SOURCES = ['cwru', 'mfpt', 'paderborn', 'mafaulda']
BEARING_SOURCES = ['cwru', 'femto', 'xjtu_sy', 'ims', 'mfpt', 'paderborn', 'ottawa']
ALL_SOURCES = list(SOURCES.keys())

# Native sampling rates
SOURCE_SR = {k: v['native_sr'] for k, v in SOURCES.items()}
# Add aliases used in V8 data_pipeline
SOURCE_SR['ottawa_bearing'] = 42000


def get_source_info(source_id: str) -> dict:
    """Get metadata for a source."""
    sid = source_id.lower().replace('ottawa_bearing', 'ottawa')
    return SOURCES.get(sid, {})


def get_compatible_groups():
    """Return pre-defined compatibility groups for pretraining."""
    return {
        'all_8': ['cwru', 'femto', 'xjtu_sy', 'ims', 'mfpt', 'paderborn', 'ottawa', 'mafaulda'],
        'bearing_rul': ['femto', 'xjtu_sy', 'ims'],
        'bearing_all': ['cwru', 'femto', 'xjtu_sy', 'ims', 'mfpt', 'paderborn', 'ottawa'],
        'compatible': None,  # Filled in after dataset_compatibility.py analysis
    }


if __name__ == '__main__':
    print("Dataset Registry")
    print("=" * 50)
    for src, meta in SOURCES.items():
        sr = meta['native_sr']
        has_rtf = meta['has_run_to_failure']
        n = meta.get('n_episodes', meta.get('n_signals', '?'))
        print(f"  {src:20s} SR={sr:6d}  RTF={'Y' if has_rtf else 'N'}  n={n}")

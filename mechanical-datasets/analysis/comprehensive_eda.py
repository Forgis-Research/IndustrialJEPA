#!/usr/bin/env python3
"""
Comprehensive EDA of Mechanical-Components Dataset (streaming mode).
"""

import os
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import signal as scipy_signal, stats

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)

OUT_DIR = "/home/sagemaker-user/IndustrialJEPA/mechanical-datasets/analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_ID = "Forgis/Mechanical-Components"

from datasets import load_dataset

# ============================================================
# 1. LOAD SOURCE METADATA (tiny, non-streaming)
# ============================================================
print("=" * 70)
print("LOADING SOURCE METADATA")
print("=" * 70)
meta_ds = load_dataset(DATASET_ID, "source_metadata")
meta = meta_ds["train"]
print(f"Sources: {len(meta)}")
for row in meta:
    sid = row.get('source_id', '?')
    ns = row.get('n_samples', 0)
    sr = row.get('sampling_rate_hz', 0)
    ct = row.get('component_type', '?')
    mods = row.get('available_modalities', [])
    faults = row.get('available_fault_types', [])
    print(f"  {sid:20s} | {ct:25s} | {sr:>6}Hz | n={ns:>5} | mods={mods} | faults={faults}")

# ============================================================
# 2. STREAM v2_train - COLLECT METADATA + SAMPLE SIGNALS
# ============================================================
print("\n" + "=" * 70)
print("STREAMING v2_train")
print("=" * 70)

metadata = defaultdict(list)
signal_samples = defaultdict(list)  # source -> list of (signal, row_meta)
MAX_SIGNALS_PER_SOURCE = 30
total_count = 0

for split_name in ["train", "validation", "test"]:
    print(f"\n  Streaming split: {split_name}...")
    try:
        ds_stream = load_dataset(DATASET_ID, "v2_train", split=split_name, streaming=True)
    except Exception as e:
        print(f"    FAILED: {e}")
        continue

    count = 0
    for row in ds_stream:
        count += 1
        total_count += 1

        # Collect metadata (no signal)
        src = row.get('source_id', 'unknown')
        metadata['source_id'].append(src)
        metadata['_split'].append(split_name)

        for col in ['health_state', 'fault_type', 'rpm', 'episode_id',
                     'rul_percent', 'is_transition', 'valid_length', 'fault_severity']:
            metadata[col].append(row.get(col))

        # Sample signals for detailed analysis
        if len(signal_samples[src]) < MAX_SIGNALS_PER_SOURCE:
            sig = np.array(row['signal'], dtype=np.float32)
            row_meta = {k: row.get(k) for k in ['source_id', 'health_state', 'fault_type',
                                                   'valid_length', 'rpm', 'rul_percent']}
            row_meta['split'] = split_name
            signal_samples[src].append((sig, row_meta))

        if count % 5000 == 0:
            print(f"    ...{count} rows")

    print(f"    {split_name}: {count} samples")

print(f"\nTotal samples: {total_count}")

# ============================================================
# 3. DISTRIBUTION ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("DISTRIBUTIONS")
print("=" * 70)

print("\n--- Source Distribution ---")
source_counts = Counter(metadata['source_id'])
for src, cnt in source_counts.most_common():
    pct = 100 * cnt / total_count
    print(f"  {src:20s}: {cnt:>6} ({pct:5.1f}%)")

print("\n--- Health State Distribution ---")
health_counts = Counter(h for h in metadata['health_state'] if h)
for state, cnt in health_counts.most_common():
    print(f"  {state:30s}: {cnt:>6} ({100*cnt/total_count:5.1f}%)")

print("\n--- Fault Type Distribution ---")
fault_counts = Counter(f for f in metadata['fault_type'] if f)
for ft, cnt in fault_counts.most_common():
    print(f"  {ft:30s}: {cnt:>6} ({100*cnt/total_count:5.1f}%)")

print("\n--- Split Distribution ---")
split_counts = Counter(metadata['_split'])
for sp, cnt in split_counts.most_common():
    print(f"  {sp:15s}: {cnt:>6} ({100*cnt/total_count:5.1f}%)")

print("\n--- RPM Distribution ---")
rpms = [r for r in metadata['rpm'] if r is not None and r > 0]
if rpms:
    print(f"  With RPM: {len(rpms)}/{total_count}")
    print(f"  Range: {min(rpms)} - {max(rpms)}")
    print(f"  Mean: {np.mean(rpms):.0f}, Median: {np.median(rpms):.0f}")

print("\n--- Episode & RUL Data ---")
episodes = [e for e in metadata['episode_id'] if e and e != 'none' and e != '']
print(f"  With episode_id: {len(episodes)}/{total_count}")
ruls = [r for r in metadata['rul_percent'] if r is not None and r >= 0]
print(f"  With rul_percent: {len(ruls)}/{total_count}")
transitions = sum(1 for t in metadata['is_transition'] if t == True)
print(f"  Transition samples: {transitions}/{total_count}")

print("\n--- Valid Length ---")
vlens = np.array([v for v in metadata['valid_length'] if v is not None])
if len(vlens) > 0:
    full = np.sum(vlens == 16384)
    short = vlens[vlens < 16384]
    print(f"  Full (16384): {full} ({100*full/len(vlens):.1f}%)")
    if len(short) > 0:
        print(f"  Padded: {len(short)} ({100*len(short)/len(vlens):.1f}%)")
        print(f"  Short lengths: min={short.min()}, max={short.max()}, mean={short.mean():.0f}")

# ============================================================
# 4. SIGNAL QUALITY ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("SIGNAL QUALITY ANALYSIS")
print("=" * 70)

signal_stats_by_source = {}
for src in sorted(signal_samples.keys()):
    samples = signal_samples[src]
    n = len(samples)
    stat_list = []
    for sig, rmeta in samples:
        vl = rmeta.get('valid_length') or len(sig)
        sig_v = sig[:vl]
        if len(sig_v) < 10:
            continue
        stat_list.append({
            'mean': float(np.mean(sig_v)),
            'std': float(np.std(sig_v)),
            'max_abs': float(np.max(np.abs(sig_v))),
            'rms': float(np.sqrt(np.mean(sig_v**2))),
            'kurtosis': float(stats.kurtosis(sig_v)),
            'skewness': float(stats.skew(sig_v)),
            'crest_factor': float(np.max(np.abs(sig_v)) / (np.sqrt(np.mean(sig_v**2)) + 1e-10)),
            'length': vl,
            'n_zeros': int(np.sum(sig_v == 0)),
            'health_state': rmeta.get('health_state', 'unknown'),
        })
    signal_stats_by_source[src] = stat_list

    means = [s['mean'] for s in stat_list]
    stds_vals = [s['std'] for s in stat_list]
    kurts = [s['kurtosis'] for s in stat_list]
    crests = [s['crest_factor'] for s in stat_list]
    max_abs = [s['max_abs'] for s in stat_list]
    n_zeros = [s['n_zeros'] for s in stat_list]

    print(f"\n--- {src} (n={n}) ---")
    print(f"  Mean of means: {np.mean(means):.6f} (target ~0)")
    print(f"  Mean of stds:  {np.mean(stds_vals):.4f} (target ~1)")
    print(f"  Kurtosis:      {np.mean(kurts):.2f} +/- {np.std(kurts):.2f}")
    print(f"  Crest factor:  {np.mean(crests):.2f} +/- {np.std(crests):.2f}")
    print(f"  Max |amp|:     {np.max(max_abs):.2f}")
    print(f"  Zero padding:  {np.mean(n_zeros):.0f} avg zeros")

    issues = []
    if abs(np.mean(means)) > 0.05:
        issues.append(f"Non-zero mean ({np.mean(means):.4f})")
    if abs(np.mean(stds_vals) - 1.0) > 0.15:
        issues.append(f"Std far from 1.0 ({np.mean(stds_vals):.4f})")
    if np.mean(kurts) > 30:
        issues.append(f"Very high kurtosis ({np.mean(kurts):.1f})")
    if np.max(max_abs) > 100:
        issues.append(f"Extreme amplitudes ({np.max(max_abs):.1f})")
    if np.mean(n_zeros) > 5000:
        issues.append(f"Heavy padding ({np.mean(n_zeros):.0f} zeros)")
    if issues:
        for iss in issues:
            print(f"  WARNING: {iss}")
    else:
        print(f"  OK")

# ============================================================
# 5. VISUALIZATIONS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)

# --- Figure 1: Dataset Composition Overview ---
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Mechanical-Components Dataset: Composition Overview", fontsize=16, fontweight='bold')

# 1a: Samples per source
ax = axes[0, 0]
sources_sorted = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
names = [s[0] for s in sources_sorted]
counts = [s[1] for s in sources_sorted]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(names)))
bars = ax.barh(names[::-1], counts[::-1], color=colors[::-1])
ax.set_xlabel("Number of Samples")
ax.set_title("Samples per Source")
for bar, cnt in zip(bars, counts[::-1]):
    ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
            f'{cnt}', va='center', fontsize=9)

# 1b: Health state
ax = axes[0, 1]
health_sorted = sorted(health_counts.items(), key=lambda x: x[1], reverse=True)
h_names = [h[0] for h in health_sorted[:15]]
h_counts = [h[1] for h in health_sorted[:15]]
cmap_health = {'healthy': '#2ecc71', 'faulty': '#e74c3c', 'degrading': '#f39c12'}
h_colors = [cmap_health.get(n, '#3498db') for n in h_names]
ax.barh(h_names[::-1], h_counts[::-1], color=h_colors[::-1])
ax.set_xlabel("Number of Samples")
ax.set_title("Health State Distribution")

# 1c: Fault type
ax = axes[1, 0]
fault_sorted = sorted(fault_counts.items(), key=lambda x: x[1], reverse=True)
f_names = [f[0][:30] for f in fault_sorted[:15]]
f_counts = [f[1] for f in fault_sorted[:15]]
ax.barh(f_names[::-1], f_counts[::-1], color=plt.cm.Set2(np.linspace(0, 1, len(f_names))))
ax.set_xlabel("Number of Samples")
ax.set_title("Fault Type Distribution (Top 15)")

# 1d: Split composition
ax = axes[1, 1]
split_source = defaultdict(lambda: defaultdict(int))
for sp, src in zip(metadata['_split'], metadata['source_id']):
    split_source[sp][src] += 1
all_sources = sorted(set(metadata['source_id']))
split_names = sorted(split_source.keys())
bottom = np.zeros(len(split_names))
for src in all_sources:
    vals = [split_source[sp].get(src, 0) for sp in split_names]
    ax.bar(split_names, vals, bottom=bottom, label=src, width=0.6)
    bottom += vals
ax.set_ylabel("Number of Samples")
ax.set_title("Split Composition by Source")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_composition_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("  01_composition_overview.png")

# --- Figure 2: Source x Split Heatmap ---
fig, ax = plt.subplots(figsize=(10, 12))
heatmap_data = np.zeros((len(all_sources), len(split_names)))
for i, src in enumerate(all_sources):
    for j, sp in enumerate(split_names):
        heatmap_data[i, j] = split_source[sp].get(src, 0)
sns.heatmap(heatmap_data, annot=True, fmt='.0f', xticklabels=split_names,
            yticklabels=all_sources, cmap='YlOrRd', ax=ax)
ax.set_title("Source x Split (Source-Disjoint Verification)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_source_split_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  02_source_split_heatmap.png")

# --- Figure 3: Signal Quality Boxplots ---
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("Signal Quality Metrics by Source", fontsize=16, fontweight='bold')

metrics = [('std', 'Standard Deviation (post-norm)'),
           ('kurtosis', 'Kurtosis'),
           ('crest_factor', 'Crest Factor'),
           ('max_abs', 'Max |Amplitude|')]

for ax, (metric, title) in zip(axes.flat, metrics):
    data_by_src = {}
    for src in sorted(signal_stats_by_source.keys()):
        vals = [s[metric] for s in signal_stats_by_source[src] if not np.isnan(s[metric]) and not np.isinf(s[metric])]
        if vals:
            data_by_src[src] = vals
    if data_by_src:
        bp = ax.boxplot(data_by_src.values(), labels=data_by_src.keys(),
                       vert=True, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.viridis(np.linspace(0.2, 0.9, len(data_by_src)))):
            patch.set_facecolor(color)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_signal_quality_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()
print("  03_signal_quality_boxplots.png")

# --- Figure 4: Example Waveforms ---
source_list = sorted(signal_samples.keys())
n_sources = len(source_list)
fig, axes = plt.subplots(n_sources, 1, figsize=(18, 2.5*n_sources))
if n_sources == 1:
    axes = [axes]

for ax, src in zip(axes, source_list):
    if signal_samples[src]:
        sig, rmeta = signal_samples[src][0]
        vl = rmeta.get('valid_length') or len(sig)
        t = np.arange(vl) / 12800.0
        ax.plot(t, sig[:vl], linewidth=0.3, color='#2c3e50')
        if vl < len(sig):
            ax.axvline(x=vl/12800.0, color='red', linestyle='--', alpha=0.5, label='pad')
        hs = rmeta.get('health_state', '?')
        ft = rmeta.get('fault_type', '?')
        ax.set_title(f"{src} | {hs} | {ft}", fontsize=10, loc='left')
        ax.set_ylabel("Amp")

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("Example Waveforms (v2_train, 12.8 kHz)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_example_waveforms.png", dpi=150, bbox_inches='tight')
plt.close()
print("  04_example_waveforms.png")

# --- Figure 5: Power Spectral Density ---
fig, axes = plt.subplots(n_sources, 1, figsize=(18, 2.5*n_sources))
if n_sources == 1:
    axes = [axes]

for ax, src in zip(axes, source_list):
    if signal_samples[src]:
        sig, rmeta = signal_samples[src][0]
        vl = rmeta.get('valid_length') or len(sig)
        sig_v = sig[:vl]
        nperseg = min(1024, vl)
        f, psd = scipy_signal.welch(sig_v, fs=12800, nperseg=nperseg, noverlap=nperseg//2)
        ax.semilogy(f, psd, linewidth=0.8, color='#2c3e50')
        ax.set_title(f"{src}", fontsize=10, loc='left')
        ax.set_ylabel("PSD")
        ax.set_xlim([0, 6400])

axes[-1].set_xlabel("Frequency (Hz)")
fig.suptitle("Power Spectral Density (Welch, 12.8 kHz)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_frequency_spectra.png", dpi=150, bbox_inches='tight')
plt.close()
print("  05_frequency_spectra.png")

# --- Figure 6: Health x Source Heatmap ---
fig, ax = plt.subplots(figsize=(14, 10))
health_source = defaultdict(lambda: defaultdict(int))
for hs, src in zip(metadata['health_state'], metadata['source_id']):
    if hs:
        health_source[src][hs] += 1
all_health = sorted(set(h for h in metadata['health_state'] if h))
hm = np.zeros((len(all_sources), len(all_health)))
for i, src in enumerate(all_sources):
    for j, hs in enumerate(all_health):
        hm[i, j] = health_source[src].get(hs, 0)
sns.heatmap(hm, annot=True, fmt='.0f', xticklabels=all_health,
            yticklabels=all_sources, cmap='Blues', ax=ax)
ax.set_title("Health State x Source")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_health_source_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  06_health_source_heatmap.png")

# --- Figure 7: Fault Type x Source Heatmap ---
fig, ax = plt.subplots(figsize=(18, 10))
fault_source = defaultdict(lambda: defaultdict(int))
for ft, src in zip(metadata['fault_type'], metadata['source_id']):
    if ft:
        fault_source[src][ft] += 1
all_faults = sorted(set(f for f in metadata['fault_type'] if f))
hm2 = np.zeros((len(all_sources), len(all_faults)))
for i, src in enumerate(all_sources):
    for j, ft in enumerate(all_faults):
        hm2[i, j] = fault_source[src].get(ft, 0)
sns.heatmap(hm2, annot=True, fmt='.0f', xticklabels=all_faults,
            yticklabels=all_sources, cmap='Oranges', ax=ax)
ax.set_title("Fault Type x Source")
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_fault_source_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("  07_fault_source_heatmap.png")

# --- Figure 8: Operating Conditions ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Operating Conditions", fontsize=14, fontweight='bold')

rpm_by_source = defaultdict(list)
for rpm, src in zip(metadata['rpm'], metadata['source_id']):
    if rpm and rpm > 0:
        rpm_by_source[src].append(rpm)

ax = axes[0]
if rpm_by_source:
    srcs_rpm = sorted(rpm_by_source.keys())
    bp = ax.boxplot([rpm_by_source[s] for s in srcs_rpm],
                    labels=srcs_rpm, vert=True, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(srcs_rpm)))):
        patch.set_facecolor(color)
    ax.set_ylabel("RPM")
    ax.set_title("RPM by Source")
    ax.tick_params(axis='x', rotation=45)

ax = axes[1]
if rpms:
    ax.hist(rpms, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel("RPM")
    ax.set_ylabel("Count")
    ax.set_title(f"RPM Distribution (n={len(rpms)})")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_operating_conditions.png", dpi=150, bbox_inches='tight')
plt.close()
print("  08_operating_conditions.png")

# --- Figure 9: Valid Length Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Signal Length & Padding Analysis", fontsize=14, fontweight='bold')

if len(vlens) > 0:
    ax = axes[0]
    ax.hist(vlens, bins=100, color='#2ecc71', edgecolor='white', alpha=0.8)
    ax.axvline(x=16384, color='red', linestyle='--', label='Full (16384)')
    ax.set_xlabel("Valid Length")
    ax.set_ylabel("Count")
    ax.set_title("Valid Length Distribution")
    ax.legend()

    ax = axes[1]
    vlen_by_src = defaultdict(list)
    for vl, src in zip(metadata['valid_length'], metadata['source_id']):
        if vl is not None:
            vlen_by_src[src].append(vl)
    srcs_vl = sorted(vlen_by_src.keys())
    bp = ax.boxplot([vlen_by_src[s] for s in srcs_vl],
                    labels=srcs_vl, vert=True, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Pastel1(np.linspace(0, 1, len(srcs_vl)))):
        patch.set_facecolor(color)
    ax.axhline(y=16384, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel("Valid Length")
    ax.set_title("Valid Length by Source")
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/09_valid_length_analysis.png", dpi=150, bbox_inches='tight')
plt.close()
print("  09_valid_length_analysis.png")

# --- Figure 10: Signal Statistics Scatter ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Signal Statistics: Cross-Source Comparison", fontsize=14, fontweight='bold')

all_stats = []
for src in signal_stats_by_source:
    for s in signal_stats_by_source[src]:
        s['source_id'] = src
        all_stats.append(s)

if all_stats:
    sources_u = sorted(set(s['source_id'] for s in all_stats))
    cmap_s = {src: plt.cm.tab20(i/max(len(sources_u),1)) for i, src in enumerate(sources_u)}

    ax = axes[0]
    for src in sources_u:
        pts = [s for s in all_stats if s['source_id'] == src]
        ax.scatter([s['kurtosis'] for s in pts], [s['crest_factor'] for s in pts],
                  c=[cmap_s[src]], label=src, alpha=0.6, s=25)
    ax.set_xlabel("Kurtosis"); ax.set_ylabel("Crest Factor"); ax.set_title("Kurtosis vs Crest Factor")

    ax = axes[1]
    for src in sources_u:
        pts = [s for s in all_stats if s['source_id'] == src]
        ax.scatter([s['skewness'] for s in pts], [s['kurtosis'] for s in pts],
                  c=[cmap_s[src]], label=src, alpha=0.6, s=25)
    ax.set_xlabel("Skewness"); ax.set_ylabel("Kurtosis"); ax.set_title("Skewness vs Kurtosis")

    ax = axes[2]
    for src in sources_u:
        pts = [s for s in all_stats if s['source_id'] == src]
        ax.scatter([s['rms'] for s in pts], [s['max_abs'] for s in pts],
                  c=[cmap_s[src]], label=src, alpha=0.6, s=25)
    ax.set_xlabel("RMS"); ax.set_ylabel("Max |Amp|"); ax.set_title("RMS vs Max Amplitude")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/10_signal_statistics_scatter.png", dpi=150, bbox_inches='tight')
plt.close()
print("  10_signal_statistics_scatter.png")

# --- Figure 11: Spectrograms ---
print("  Computing spectrograms...")
fig, axes = plt.subplots(n_sources, 1, figsize=(18, 2.5*n_sources))
if n_sources == 1:
    axes = [axes]

for ax, src in zip(axes, source_list):
    if signal_samples[src]:
        sig, rmeta = signal_samples[src][0]
        vl = rmeta.get('valid_length') or len(sig)
        sig_v = sig[:vl]
        nperseg = min(256, vl)
        noverlap = nperseg * 3 // 4
        f_spec, t_spec, Sxx = scipy_signal.spectrogram(sig_v, fs=12800, nperseg=nperseg, noverlap=noverlap)
        ax.pcolormesh(t_spec, f_spec, 10*np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        ax.set_title(f"{src}", fontsize=10, loc='left')
        ax.set_ylabel("Hz")

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Spectrograms by Source (v2_train, 12.8 kHz)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/11_spectrograms.png", dpi=150, bbox_inches='tight')
plt.close()
print("  11_spectrograms.png")

# --- Figure 12: RUL Analysis ---
if ruls:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Remaining Useful Life (RUL) Analysis", fontsize=14, fontweight='bold')

    ax = axes[0]
    ax.hist(ruls, bins=50, color='#e74c3c', edgecolor='white', alpha=0.8)
    ax.set_xlabel("RUL %"); ax.set_ylabel("Count"); ax.set_title(f"RUL Distribution (n={len(ruls)})")

    ax = axes[1]
    rul_by_src = defaultdict(list)
    for rul, src in zip(metadata['rul_percent'], metadata['source_id']):
        if rul is not None and rul >= 0:
            rul_by_src[src].append(rul)
    srcs_rul = sorted(s for s in rul_by_src if rul_by_src[s])
    if srcs_rul:
        bp = ax.boxplot([rul_by_src[s] for s in srcs_rul],
                        labels=srcs_rul, vert=True, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Reds(np.linspace(0.3, 0.8, len(srcs_rul)))):
            patch.set_facecolor(color)
        ax.set_ylabel("RUL %"); ax.set_title("RUL by Source")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/12_rul_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  12_rul_analysis.png")

# --- Figure 13: Healthy vs Faulty comparison per source ---
fig, axes = plt.subplots(min(n_sources, 6), 2, figsize=(18, 3*min(n_sources, 6)))
fig.suptitle("Healthy vs Faulty: Time Domain & Frequency Domain", fontsize=14, fontweight='bold')

plot_idx = 0
for src in source_list[:6]:
    healthy_sigs = [(s, m) for s, m in signal_samples[src] if m.get('health_state') == 'healthy']
    faulty_sigs = [(s, m) for s, m in signal_samples[src] if m.get('health_state') == 'faulty']

    if not healthy_sigs or not faulty_sigs:
        # Use first 2 signals instead
        if len(signal_samples[src]) >= 2:
            healthy_sigs = [signal_samples[src][0]]
            faulty_sigs = [signal_samples[src][1]]
        else:
            plot_idx += 1
            continue

    h_sig, h_meta = healthy_sigs[0]
    f_sig, f_meta = faulty_sigs[0]

    # Time domain
    ax = axes[plot_idx, 0] if n_sources > 1 else axes[0]
    h_vl = h_meta.get('valid_length') or len(h_sig)
    f_vl = f_meta.get('valid_length') or len(f_sig)
    t_h = np.arange(min(h_vl, 4000)) / 12800.0
    t_f = np.arange(min(f_vl, 4000)) / 12800.0
    ax.plot(t_h, h_sig[:len(t_h)], linewidth=0.4, alpha=0.8, label=f'H: {h_meta.get("fault_type","?")}', color='#2ecc71')
    ax.plot(t_f, f_sig[:len(t_f)], linewidth=0.4, alpha=0.8, label=f'F: {f_meta.get("fault_type","?")}', color='#e74c3c')
    ax.set_title(f"{src} - Time Domain", fontsize=9, loc='left')
    ax.legend(fontsize=7)

    # Frequency domain
    ax = axes[plot_idx, 1] if n_sources > 1 else axes[1]
    nperseg_h = min(1024, h_vl)
    nperseg_f = min(1024, f_vl)
    fh, psd_h = scipy_signal.welch(h_sig[:h_vl], fs=12800, nperseg=nperseg_h)
    ff, psd_f = scipy_signal.welch(f_sig[:f_vl], fs=12800, nperseg=nperseg_f)
    ax.semilogy(fh, psd_h, linewidth=0.8, alpha=0.8, label='Healthy', color='#2ecc71')
    ax.semilogy(ff, psd_f, linewidth=0.8, alpha=0.8, label='Faulty', color='#e74c3c')
    ax.set_title(f"{src} - PSD", fontsize=9, loc='left')
    ax.set_xlim([0, 6400])
    ax.legend(fontsize=7)

    plot_idx += 1

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/13_healthy_vs_faulty.png", dpi=150, bbox_inches='tight')
plt.close()
print("  13_healthy_vs_faulty.png")

# --- Figure 14: Dataset Imbalance Summary ---
fig, ax = plt.subplots(figsize=(16, 8))
health_by_src = defaultdict(lambda: defaultdict(int))
for hs, src in zip(metadata['health_state'], metadata['source_id']):
    if hs:
        health_by_src[src][hs] += 1

srcs_bar = sorted(health_by_src.keys())
all_hs = sorted(set(h for h in metadata['health_state'] if h))
bottom_arr = np.zeros(len(srcs_bar))
hs_colors_map = {'healthy': '#2ecc71', 'faulty': '#e74c3c', 'degrading': '#f39c12'}

for hs in all_hs:
    vals = [health_by_src[src].get(hs, 0) for src in srcs_bar]
    color = hs_colors_map.get(hs, plt.cm.Set2(hash(hs) % 8 / 8))
    ax.bar(srcs_bar, vals, bottom=bottom_arr, label=hs, color=color)
    bottom_arr += np.array(vals)

ax.set_ylabel("Samples")
ax.set_title("Health State by Source (Stacked)")
ax.legend()
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/14_health_by_source_stacked.png", dpi=150, bbox_inches='tight')
plt.close()
print("  14_health_by_source_stacked.png")

# ============================================================
# 6. FIELD COMPLETENESS & COVERAGE
# ============================================================
print("\n" + "=" * 70)
print("FIELD COMPLETENESS")
print("=" * 70)

for col in sorted(metadata.keys()):
    if col.startswith('_'):
        continue
    vals = metadata[col]
    n_total = len(vals)
    n_none = sum(1 for v in vals if v is None or v == '' or v == 'none' or v == 'unknown')
    pct = 100 * (n_total - n_none) / n_total if n_total > 0 else 0
    print(f"  {col:25s}: {n_total-n_none:>6}/{n_total} ({pct:5.1f}%)")

# ============================================================
# 7. GAPS & RECOMMENDATIONS
# ============================================================
print("\n" + "=" * 70)
print("GAPS & RECOMMENDATIONS")
print("=" * 70)

# Component coverage
fault_set = set(f for f in metadata['fault_type'] if f)
bearing_faults = {'inner_race', 'outer_race', 'ball', 'cage', 'bearing_fault'}
gear_faults = {'gear_crack', 'gear_wear', 'missing_tooth', 'tooth_break', 'chipped_tooth'}
shaft_faults = {'imbalance', 'misalignment', 'horizontal_misalignment', 'vertical_misalignment', 'unbalance'}

has_bearing = bool(fault_set & bearing_faults)
has_gear = bool(fault_set & gear_faults)
has_shaft = bool(fault_set & shaft_faults)
has_motor = any('stator' in f or 'rotor' in f for f in fault_set)

print(f"\n  Bearing faults: {'YES' if has_bearing else 'NO'} ({fault_set & bearing_faults})")
print(f"  Gear faults:    {'YES' if has_gear else 'NO'} ({fault_set & gear_faults})")
print(f"  Shaft faults:   {'YES' if has_shaft else 'NO'} ({fault_set & shaft_faults})")
print(f"  Motor faults:   {'YES' if has_motor else 'NO'}")

print(f"\n  Total samples: {total_count}")
print(f"  Total sources: {len(source_counts)}")
print(f"  Unique health states: {len(health_counts)}")
print(f"  Unique fault types: {len(fault_counts)}")

print("\n  IDENTIFIED GAPS:")
gaps = []
if not has_motor:
    gaps.append("- No motor-specific faults (stator winding, rotor bar, eccentricity)")
if total_count < 50000:
    gaps.append(f"- Modest size ({total_count} samples) for foundation model pretraining")
if len(source_counts) < 20:
    gaps.append(f"- {len(source_counts)} sources — more diversity beneficial")
if transitions < 500:
    gaps.append(f"- Only {transitions} transition samples — limited action-conditioning data")
if len(ruls) < total_count * 0.3:
    gaps.append(f"- RUL data sparse ({len(ruls)}/{total_count} = {100*len(ruls)/total_count:.0f}%)")

# Check class balance
biggest = max(source_counts.values())
smallest = min(source_counts.values())
if biggest / max(smallest, 1) > 100:
    gaps.append(f"- Extreme source imbalance: {biggest}x vs {smallest}x ({biggest/max(smallest,1):.0f}x ratio)")

for g in gaps:
    print(f"  {g}")

print("\n  POTENTIAL ADDITIONS (from NEW_DATASETS_INVENTORY):")
print("  - NLN-EMP (Dutch Navy): 11+ fault types spanning motor/coupling/shaft/bearing")
print("  - COMFAULDA: Compound/combined fault scenarios")
print("  - KAIST Run-to-Failure: Vibration + temperature degradation")
print("  - DIRG Aerospace: High-speed bearings (30,000 RPM)")
print("  - Tsinghua Motor: Motor faults with current + vibration")

print("\n\nDONE! All figures saved to:", OUT_DIR)

# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS-INFORMED HYBRID DATASET CONSTRUCTION
#
# PROBLEM WITH IBM CALIBRATION DATA:
#   IBM publishes calibration snapshots AFTER recalibration — hardware is at
#   its best at snapshot time. Standard IBM data is dominated by healthy
#   post-calibration states, with drift events visible only as statistical
#   outliers relative to each qubit's own baseline.
#
# SOLUTION — TWO-SOURCE HYBRID DATASET:
#   Source 1 (sim_data.csv, 12,000 rows):
#     Qiskit Aer simulation with physically grounded noise models drawn from
#     IBM Heron r2 parameter distributions (Carroll 2022, ISCA 2025).
#     Covers the mid-cycle drift regime that IBM calibration data cannot.
#
#   Source 2 (all_backends_raw.csv, ~3,000 extreme rows):
#     Real IBM calibration snapshots where drift is unambiguous — either
#     clearly stable (hardware at peak) or clearly drifted (anomalous).
#     Grey zone rows discarded. Analytical fidelity proxy formulas estimate
#     what the canary circuits would have measured.
#
# LABELING PHYSICS:
#
#   EXTREME STABLE (label=0) — ALL THREE conditions must hold:
#     T1 within ±8% of 30-day rolling median
#     CZ error within +20% of rolling median (or missing)
#     Readout error within +20% of rolling median
#     → Hardware in healthy operational regime. All metrics within
#       normal daily variation. Computations are reliable.
#
#   EXTREME DRIFTED (label=1) — ANY ONE condition sufficient:
#     T1 dropped >=35% below rolling median
#       ← TLS spectral diffusion event (Carroll et al. 2022:
#          T1 fluctuations 30-50% from defect resonance)
#     CZ error rose >=100% above median
#       ← Gate severely miscalibrated (ISCA 2025: 2Q errors
#          reach up to 5x between calibrations)
#     Readout error rose >=50% above median
#       ← Measurement chain degraded beyond reliable operation
#     → Hardware has undergone a genuine drift event. Computational
#       results are materially compromised.
#
#   GREY ZONE — discarded:
#     Mild degradation in post-recalibration snapshots is ambiguous.
#     Could be normal qubit-to-qubit variation, not true drift.
#     Including it would inject noise into label assignments.
#
# WHY PERCENTAGE DEVIATION NOT Z-SCORES:
#   Some qubits have very stable T1 over their 30-day window (near-zero std).
#   Z-scores explode in this case. Percentage deviation is scale-invariant,
#   physically interpretable, and robust to near-zero denominators.
#
# FIDELITY PROXY FORMULAS (updated for new circuit depths):
#   F_bell:      H + CX   → t_circuit = t_sx + t_cz   (unchanged)
#   F_coherence: H x20    → t_circuit = 20 x t_sx     (was 2 x t_sx)
#   F_gate:      X x20    → t_circuit = 20 x t_x      (was 8 x t_x)
#   These match the actual canary circuits in 1e_pull_data.py exactly.
#   Using actual per-row gate durations from IBM calibration data.
#   Reference: Escofet et al. 2025 (analytical depolarizing fidelity model)
#
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json, os

os.makedirs('data',    exist_ok=True)
os.makedirs('figures', exist_ok=True)

np.random.seed(42)

# ── 1. LOAD AND SORT ──────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend', 'qubit_id', 'timestamp']).reset_index(drop=True)
print(f"  {len(df):,} rows | backends: {list(df['backend'].unique())}")
print(f"  Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}\n")

# ── 2. FILL MISSING GATE DURATIONS ───────────────────────────────────────────
# Use actual per-row values from IBM. Only fall back for NaN.
# ibm_fez: t_sx~24ns, ibm_kingston: t_sx~32ns, ibm_marrakesh: t_sx~36ns
df['t_sx_ns'] = df['t_sx_ns'].fillna(30.0)
df['t_x_ns']  = df['t_x_ns'].fillna(30.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(75.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

# Remove physically impossible CZ values (> 0.5 = measurement artifact)
df.loc[df['cz_error'] > 0.5, 'cz_error'] = np.nan

# ── 3. 30-DAY ROLLING BASELINE PER QUBIT ─────────────────────────────────────
# Median not mean: median is robust to the outlier drift events themselves.
# If a qubit has occasional TLS spikes, mean baseline would be pulled down,
# making subsequent spikes harder to detect. Median is resistant to outliers.
# min_periods=14: need 2 weeks of data before baseline is reliable.
# shift(1): strict temporal causality — today never contaminates its own baseline.

print("Computing 30-day rolling baselines (median, shift-1 causality)...")

backend_col = df['backend'].values
qubit_col   = df['qubit_id'].values

def add_rolling(group):
    group['T1_roll_med'] = group['T1_us'].rolling(
        window=30, min_periods=14).median().shift(1)
    group['T1_roll_std'] = group['T1_us'].rolling(
        window=30, min_periods=14).std().shift(1)
    group['CZ_roll_med'] = group['cz_error'].rolling(
        window=30, min_periods=14).median().shift(1)
    group['RO_roll_med'] = group['readout_error'].rolling(
        window=30, min_periods=14).median().shift(1)
    return group

df = df.groupby(['backend', 'qubit_id'], group_keys=False).apply(add_rolling)
df = df.reset_index(drop=True)
df['backend']  = backend_col
df['qubit_id'] = qubit_col

df = df.dropna(subset=['T1_roll_med', 'RO_roll_med']).reset_index(drop=True)
print(f"  Rows with valid baselines: {len(df):,}\n")

# ── 4. PERCENTAGE DEVIATIONS FROM BASELINE ───────────────────────────────────
EPS = 1e-8

df['pct_T1_drop'] = (
    (df['T1_roll_med'] - df['T1_us']) / (df['T1_roll_med'] + EPS)
).clip(-2, 2)

df['pct_T1_dev'] = (
    (df['T1_us'] - df['T1_roll_med']).abs() / (df['T1_roll_med'] + EPS)
).clip(0, 2)

cz_valid = df['cz_error'].notna() & df['CZ_roll_med'].notna()
df['pct_CZ_rise'] = np.nan
df.loc[cz_valid, 'pct_CZ_rise'] = (
    (df.loc[cz_valid, 'cz_error'] - df.loc[cz_valid, 'CZ_roll_med']) /
    (df.loc[cz_valid, 'CZ_roll_med'] + EPS)
).clip(-1, 10)

df['pct_RO_rise'] = (
    (df['readout_error'] - df['RO_roll_med']) / (df['RO_roll_med'] + EPS)
).clip(-1, 10)

# ── 5. CLASSIFY INTO ZONES ────────────────────────────────────────────────────
print("Classifying into extreme stable / extreme drifted / grey zone...")

def classify_zone(row):
    if pd.isna(row['T1_roll_med']) or row['T1_roll_med'] <= 0:
        return 'grey'

    t1_drop = row['pct_T1_drop']
    t1_dev  = row['pct_T1_dev']
    ro_rise = row['pct_RO_rise']
    cz_rise = row['pct_CZ_rise']

    # EXTREME DRIFTED — any single severe anomaly is sufficient
    if t1_drop >= 0.35:
        return 'extreme_drifted'      # TLS event: T1 >=35% below baseline
    if not pd.isna(cz_rise) and cz_rise >= 1.00:
        return 'extreme_drifted'      # Gate miscalibration: CZ doubled
    if ro_rise >= 0.50:
        return 'extreme_drifted'      # Readout degraded >=50%

    # EXTREME STABLE — all metrics must simultaneously be near baseline
    t1_ok = t1_dev <= 0.08
    ro_ok = ro_rise <= 0.20
    cz_ok = pd.isna(cz_rise) or (cz_rise <= 0.20)
    if t1_ok and ro_ok and cz_ok:
        return 'extreme_stable'

    return 'grey'

df['zone'] = df.apply(classify_zone, axis=1)

zone_counts = df['zone'].value_counts()
print(f"  Extreme stable  : {zone_counts.get('extreme_stable',  0):,}")
print(f"  Extreme drifted : {zone_counts.get('extreme_drifted', 0):,}")
print(f"  Grey zone       : {zone_counts.get('grey', 0):,} (discarded)\n")

# ── 6. ANALYTICAL FIDELITY PROXY FORMULAS ────────────────────────────────────
# Estimates what the 3 canary circuits would have measured on this hardware.
# CRITICAL: Circuit depths match 1e_pull_data.py exactly:
#   F_coherence: 20 H gates x t_sx (not 2 x t_sx as in older versions)
#   F_gate:      20 X gates x t_x  (not 8 x t_x)
# Using actual per-row gate durations from the IBM calibration record.

print("Computing analytical fidelity proxy formulas...")
print("  F_bell      : H + CX   = t_sx + t_cz ns (per-row gate times)")
print("  F_coherence : H x 20   = 20 x t_sx ns")
print("  F_gate      : X x 20   = 20 x t_x  ns")

T2_ns      = (df['T2_us'] * 1e3).replace(0, np.nan)
T1_us_safe = df['T1_us'].replace(0, np.nan).fillna(df['T1_us'].median())
T1_ns      = T1_us_safe * 1e3
T2_ns      = T2_ns.fillna(T1_ns * 0.9)
T2_ns      = T2_ns.clip(upper=2.0 * T1_ns)   # Physical: T2 <= 2*T1
T2_ns      = T2_ns.replace(0, np.nan).fillna(T1_ns * 0.9)

cz = df['cz_error'].fillna(df['sx_error'])

# Bell state: H(q0), CX(q0->q1), measure
t_bell = df['t_sx_ns'] + df['t_cz_ns']
df['F_bell'] = (
    (1 - df['sx_error']) *
    (1 - cz) *
    (1 - df['readout_error'])**2 *
    np.exp(-t_bell / T1_ns) *
    np.exp(-t_bell / T2_ns)
).clip(0, 1)

# Coherence: H x 20 (identity) — maximally sensitive to T2
t_coh = 20.0 * df['t_sx_ns']
df['F_coherence'] = (
    (1 - df['sx_error'])**20 *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-t_coh / T1_ns) * np.exp(-t_coh / T2_ns))
).clip(0, 1)

# Gate error: X x 20 (identity) — error amplification
t_gate = 20.0 * df['t_x_ns']
df['F_gate'] = (
    (1 - df['x_error'])**20 *
    (1 - df['readout_error']) *
    np.exp(-t_gate / T1_ns) *
    np.exp(-t_gate / T2_ns)
).clip(0, 1)

print(f"\n  IBM fidelity means by zone:")
print(df.groupby('zone')[['F_bell','F_coherence','F_gate']].mean().round(4))

# ── 7. EXTRACT EXTREME IBM ROWS ───────────────────────────────────────────────
# Sort by unambiguity: stable = closest to baseline first,
# drifted = most extreme T1 drops first (clearest TLS events)

N_IBM_MAX = 1500

ibm_stable_df  = df[df['zone'] == 'extreme_stable'].copy()
ibm_drifted_df = df[df['zone'] == 'extreme_drifted'].copy()

ibm_stable_sel  = ibm_stable_df.sort_values(
    'pct_T1_dev', ascending=True).head(N_IBM_MAX).copy()
ibm_drifted_sel = ibm_drifted_df.sort_values(
    'pct_T1_drop', ascending=False).head(N_IBM_MAX).copy()

ibm_stable_sel['drifted']  = 0
ibm_drifted_sel['drifted'] = 1

print(f"\nExtreme IBM rows selected:")
print(f"  Stable  (label=0): {len(ibm_stable_sel):,}")
print(f"  Drifted (label=1): {len(ibm_drifted_sel):,}")
print(f"\n  IBM stable fidelity means:")
print(ibm_stable_sel[['F_bell','F_coherence','F_gate']].mean().round(4))
print(f"\n  IBM drifted fidelity means:")
print(ibm_drifted_sel[['F_bell','F_coherence','F_gate']].mean().round(4))
print(f"\n  IBM drifted T1 drop distribution:")
print(ibm_drifted_sel['pct_T1_drop'].describe().round(3))

# Save labeled IBM file for reference
full_cols = ['timestamp', 'backend', 'qubit_id', 'T1_us', 'T2_us',
             'sx_error', 'x_error', 'cz_error', 'readout_error',
             't_sx_ns', 't_x_ns', 't_cz_ns',
             'T1_roll_med', 'CZ_roll_med', 'RO_roll_med',
             'pct_T1_drop', 'pct_CZ_rise', 'pct_RO_rise',
             'F_bell', 'F_coherence', 'F_gate', 'zone']
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("\n  ✓ Saved data/ibm_calibration_labeled.csv")

# ── 8. LOAD SIMULATION DATA ───────────────────────────────────────────────────
print("\nLoading data/sim_data.csv ...")
sim       = pd.read_csv('data/sim_data.csv')
sim_clean = sim[['F_bell', 'F_coherence', 'F_gate', 'drifted']].copy()
print(f"  {len(sim_clean):,} simulation rows")
print(f"  Sim fidelity means by class:")
print(sim_clean.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))

# ── 9. COMBINE ────────────────────────────────────────────────────────────────
ibm_feat = ['F_bell', 'F_coherence', 'F_gate', 'drifted']
ibm_combined = pd.concat(
    [ibm_stable_sel[ibm_feat], ibm_drifted_sel[ibm_feat]],
    ignore_index=True
)
combined = pd.concat([sim_clean, ibm_combined], ignore_index=True)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

drift_rate = round(combined['drifted'].mean() * 100, 1)

print(f"\nCombined dataset:")
print(f"  Simulation rows  : {len(sim_clean):,}")
print(f"  Extreme IBM rows : {len(ibm_combined):,}")
print(f"  Total            : {len(combined):,}")
print(f"  Stable  (0)      : {(combined['drifted']==0).sum():,}")
print(f"  Drifted (1)      : {(combined['drifted']==1).sum():,}")
print(f"  Drift rate       : {drift_rate}%")

print(f"\n  Combined fidelity means by class:")
print(combined.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))

sep_bell = (combined[combined['drifted']==0]['F_bell'].mean() -
            combined[combined['drifted']==1]['F_bell'].mean())
sep_coh  = (combined[combined['drifted']==0]['F_coherence'].mean() -
            combined[combined['drifted']==1]['F_coherence'].mean())
sep_gate = (combined[combined['drifted']==0]['F_gate'].mean() -
            combined[combined['drifted']==1]['F_gate'].mean())

print(f"\n  Class separation (stable - drifted):")
print(f"    F_bell      : {sep_bell:.4f}")
print(f"    F_coherence : {sep_coh:.4f}")
print(f"    F_gate      : {sep_gate:.4f}")

# ── 10. SAVE ──────────────────────────────────────────────────────────────────
combined[['F_bell', 'F_coherence', 'F_gate', 'drifted']].to_csv(
    'data/features_data.csv', index=False
)
print("\n  ✓ Saved data/features_data.csv")

# ── 11. STRATIFIED SPLIT 70/15/15 ────────────────────────────────────────────
all_idx = np.arange(len(combined))
y_all   = combined['drifted'].values

train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.30, stratify=y_all, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=y_all[temp_idx], random_state=42
)
split = {
    "train": sorted(train_idx.tolist()),
    "val":   sorted(val_idx.tolist()),
    "test":  sorted(test_idx.tolist())
}
with open('data/split_indices.json', 'w') as f:
    json.dump(split, f)

print(f"\n  Train: {len(split['train']):,} | "
      f"Val: {len(split['val']):,} | "
      f"Test: {len(split['test']):,}")
for name, idx in split.items():
    d_pct = round(combined.iloc[idx]['drifted'].mean() * 100, 1)
    print(f"    {name:5s}: drift={d_pct}%  ✓ stratified")

# ── 12. FIGURES ───────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Figure 1: T1 over time with zone markers
drift_by_qubit = df[df['zone']=='extreme_drifted'].groupby(
    ['backend','qubit_id']).size()
if len(drift_by_qubit) > 0:
    fig1_backend, fig1_qubit = drift_by_qubit.idxmax()
else:
    fig1_backend = df['backend'].iloc[0]
    fig1_qubit   = 0

fd = df[(df['backend']==fig1_backend) &
        (df['qubit_id']==fig1_qubit)].sort_values('timestamp')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(fd['timestamp'], fd['T1_us'], color='steelblue', lw=1.5,
        label='T1 (µs)', zorder=3)
ax.plot(fd['timestamp'], fd['T1_roll_med'], color='orange', lw=1.3,
        linestyle='--', label='30-day rolling median', zorder=2)
for _, row in fd[fd['zone']=='extreme_drifted'].iterrows():
    ax.axvline(row['timestamp'], color='crimson', alpha=0.3, lw=0.8, zorder=1)
for _, row in fd[fd['zone']=='extreme_stable'].iterrows():
    ax.axvline(row['timestamp'], color='steelblue', alpha=0.12, lw=0.8, zorder=1)
drift_dates  = fd[fd['zone']=='extreme_drifted']['timestamp']
stable_dates = fd[fd['zone']=='extreme_stable']['timestamp']
if len(drift_dates) > 0:
    ax.axvline(drift_dates.iloc[0], color='crimson', alpha=0.8, lw=0.8,
               label=f'Extreme drift (n={len(drift_dates)})')
if len(stable_dates) > 0:
    ax.axvline(stable_dates.iloc[0], color='steelblue', alpha=0.4, lw=0.8,
               label=f'Extreme stable (n={len(stable_dates)})')
ax.set_title(f'Qubit stability — {fig1_backend} qubit {fig1_qubit}\n'
             f'Grey zone discarded · extreme events only kept',
             fontsize=9, fontweight='bold')
ax.set_xlabel('Date', fontsize=9); ax.set_ylabel('T1 (µs)', fontsize=9)
ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_qubit_stability.png")

# Figure 2: Fidelity distributions
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
feats = ['F_bell', 'F_coherence', 'F_gate']
flabs = ['Bell state fidelity', 'Coherence fidelity', 'Gate error fidelity']
s_sim = sim_clean[sim_clean['drifted']==0]
d_sim = sim_clean[sim_clean['drifted']==1]

for ax, feat, lab in zip(axes, feats, flabs):
    all_vals = pd.concat([s_sim[feat], d_sim[feat],
                          ibm_stable_sel[feat], ibm_drifted_sel[feat]])
    bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 55)
    ax.hist(s_sim[feat], bins=bins, alpha=0.5, color='steelblue',
            label=f'Sim stable (n={len(s_sim):,})', density=True)
    ax.hist(d_sim[feat], bins=bins, alpha=0.5, color='crimson',
            label=f'Sim drifted (n={len(d_sim):,})', density=True)
    ax.hist(ibm_stable_sel[feat], bins=bins, alpha=0.5, color='navy',
            label=f'IBM stable (n={len(ibm_stable_sel):,})',
            density=True, histtype='step', linewidth=2)
    ax.hist(ibm_drifted_sel[feat], bins=bins, alpha=0.5, color='darkred',
            label=f'IBM drifted (n={len(ibm_drifted_sel):,})',
            density=True, histtype='step', linewidth=2)
    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)
plt.suptitle('Canary circuit fidelity distributions\n'
             'Simulation (filled) + extreme IBM snapshots (outline)',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  LABELING COMPLETE")
print(f"{'='*60}")
print(f"  Simulation rows      : {len(sim_clean):,}")
print(f"  Extreme IBM stable   : {len(ibm_stable_sel):,}")
print(f"  Extreme IBM drifted  : {len(ibm_drifted_sel):,}")
print(f"  Grey zone discarded  : {zone_counts.get('grey', 0):,}")
print(f"  Total training rows  : {len(combined):,}")
print(f"  Drift rate           : {drift_rate}%")
print(f"\n  Labeling physics:")
print(f"    Stable  : T1 within +-8% AND CZ/readout within +20% of 30d median")
print(f"    Drifted : T1 drop >=35% (TLS) OR CZ rise >=100% OR readout rise >=50%")
print(f"    Refs    : Carroll et al. 2022 · ISCA 2025")
print(f"\n  Fidelity formula circuit depths:")
print(f"    F_bell      : H + CX  = t_sx + t_cz ns (actual per-row gate times)")
print(f"    F_coherence : H x 20  = 20 x t_sx ns")
print(f"    F_gate      : X x 20  = 20 x t_x  ns")
print(f"    Ref         : Escofet et al. 2025")

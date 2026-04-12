# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS-INFORMED HYBRID DATASET CONSTRUCTION
#
# DATASET DESIGN RATIONALE:
#
#   Stable class  (label=0): 6,000 sim stable + 1,500 IBM extreme stable
#     IBM stable rows anchor the model in real hardware behavior.
#     The IBM stable outline tracks the simulation stable distribution
#     almost perfectly — these rows genuinely teach "healthy real hardware."
#
#   Drifted class (label=1): 6,000 sim drifted ONLY
#     IBM extreme drifted rows are NOT included in the drifted class.
#     Reason: even rows with T1 drop >=35% produce fidelity values nearly
#     identical to stable rows (IBM T1 baseline is so high that a 55% drop
#     from 300us to 135us barely affects a 640ns circuit).
#     Including them would contaminate the decision boundary the MLP needs
#     to learn, adding noise rather than signal.
#
#   Result: 13,500 total rows, real hardware anchored on stable side,
#     simulation covers the drift regime IBM calibration data cannot.
#
# LABELING PHYSICS:
#   EXTREME STABLE (label=0) — ALL conditions:
#     T1 within +-8% of 30-day rolling median
#     CZ within +20% of rolling median (or missing)
#     Readout within +20% of rolling median
#
#   EXTREME DRIFTED (IBM) — ANY condition (used for diagnostics only):
#     T1 dropped >=35% (TLS spectral diffusion, Carroll et al. 2022)
#     CZ rose >=100% (gate miscalibration, ISCA 2025)
#     Readout rose >=50% (measurement chain degraded)
#
# FIDELITY PROXY FORMULAS (match actual canary circuits in 1e_pull_data.py):
#   F_bell      : H + CX   = t_sx + t_cz ns
#   F_coherence : H x 20   = 20 x t_sx ns
#   F_gate      : X x 20   = 20 x t_x  ns
#   Reference   : Escofet et al. 2025
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
df['t_sx_ns'] = df['t_sx_ns'].fillna(30.0)
df['t_x_ns']  = df['t_x_ns'].fillna(30.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(75.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])
df.loc[df['cz_error'] > 0.5, 'cz_error'] = np.nan

# ── 3. 30-DAY ROLLING BASELINE PER QUBIT ─────────────────────────────────────
# Median: robust to outlier drift events in the time series.
# shift(1): strict temporal causality.
# min_periods=14: require 2 weeks of data before baseline is reliable.
print("Computing 30-day rolling baselines (median, shift-1 causality)...")

backend_col = df['backend'].values
qubit_col   = df['qubit_id'].values

def add_rolling(group):
    group['T1_roll_med'] = group['T1_us'].rolling(
        window=30, min_periods=14).median().shift(1)
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

# ── 4. PERCENTAGE DEVIATIONS ──────────────────────────────────────────────────
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

# ── 5. CLASSIFY IBM ZONES ────────────────────────────────────────────────────
print("Classifying IBM rows into extreme stable / extreme drifted / grey zone...")

def classify_zone(row):
    if pd.isna(row['T1_roll_med']) or row['T1_roll_med'] <= 0:
        return 'grey'

    t1_drop = row['pct_T1_drop']
    t1_dev  = row['pct_T1_dev']
    ro_rise = row['pct_RO_rise']
    cz_rise = row['pct_CZ_rise']

    # Extreme drifted (for diagnostics — not used in training drifted class)
    if t1_drop >= 0.35:
        return 'extreme_drifted'
    if not pd.isna(cz_rise) and cz_rise >= 1.00:
        return 'extreme_drifted'
    if ro_rise >= 0.50:
        return 'extreme_drifted'

    # Extreme stable — all metrics simultaneously near baseline
    t1_ok = t1_dev <= 0.08
    ro_ok = ro_rise <= 0.20
    cz_ok = pd.isna(cz_rise) or (cz_rise <= 0.20)
    if t1_ok and ro_ok and cz_ok:
        return 'extreme_stable'

    return 'grey'

df['zone'] = df.apply(classify_zone, axis=1)
zone_counts = df['zone'].value_counts()
print(f"  Extreme stable  : {zone_counts.get('extreme_stable',  0):,}")
print(f"  Extreme drifted : {zone_counts.get('extreme_drifted', 0):,} (diagnostics only — not in training drifted class)")
print(f"  Grey zone       : {zone_counts.get('grey', 0):,} (discarded)\n")

# ── 6. ANALYTICAL FIDELITY PROXY FORMULAS ────────────────────────────────────
print("Computing analytical fidelity proxy formulas...")
print("  F_bell      : H + CX  = t_sx + t_cz ns")
print("  F_coherence : H x 20  = 20 x t_sx ns")
print("  F_gate      : X x 20  = 20 x t_x  ns")

T2_ns      = (df['T2_us'] * 1e3).replace(0, np.nan)
T1_us_safe = df['T1_us'].replace(0, np.nan).fillna(df['T1_us'].median())
T1_ns      = T1_us_safe * 1e3
T2_ns      = T2_ns.fillna(T1_ns * 0.9)
T2_ns      = T2_ns.clip(upper=2.0 * T1_ns)
T2_ns      = T2_ns.replace(0, np.nan).fillna(T1_ns * 0.9)

cz = df['cz_error'].fillna(df['sx_error'])

t_bell = df['t_sx_ns'] + df['t_cz_ns']
df['F_bell'] = (
    (1 - df['sx_error']) *
    (1 - cz) *
    (1 - df['readout_error'])**2 *
    np.exp(-t_bell / T1_ns) *
    np.exp(-t_bell / T2_ns)
).clip(0, 1)

t_coh = 20.0 * df['t_sx_ns']
df['F_coherence'] = (
    (1 - df['sx_error'])**20 *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-t_coh / T1_ns) * np.exp(-t_coh / T2_ns))
).clip(0, 1)

t_gate = 20.0 * df['t_x_ns']
df['F_gate'] = (
    (1 - df['x_error'])**20 *
    (1 - df['readout_error']) *
    np.exp(-t_gate / T1_ns) *
    np.exp(-t_gate / T2_ns)
).clip(0, 1)

# ── 7. EXTRACT EXTREME IBM STABLE ROWS ONLY ───────────────────────────────────
# Only stable rows are used in training.
# Drifted IBM rows are saved for reference/diagnostics but NOT added to
# the training drifted class — their fidelity values overlap too heavily
# with stable rows to help the MLP learn the drift boundary.

N_IBM_STABLE = 1500

ibm_stable_df  = df[df['zone'] == 'extreme_stable'].copy()
ibm_stable_sel = ibm_stable_df.sort_values(
    'pct_T1_dev', ascending=True).head(N_IBM_STABLE).copy()
ibm_stable_sel['drifted'] = 0

print(f"\nExtreme IBM stable rows selected: {len(ibm_stable_sel):,}")
print(f"  IBM stable fidelity means:")
print(ibm_stable_sel[['F_bell','F_coherence','F_gate']].mean().round(4))
print(f"\n  Note: IBM drifted rows NOT included in training drifted class.")
print(f"  Reason: IBM T1 baseline so high that even 55% T1 drops produce")
print(f"  fidelity values indistinguishable from stable hardware.")
print(f"  Simulation drifted data covers this regime accurately.")

# Save full labeled IBM file for reference
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
# Stable class:  sim stable (6,000) + IBM stable (1,500) = 7,500
# Drifted class: sim drifted (6,000) only
# Total: 13,500 rows

ibm_feat     = ['F_bell', 'F_coherence', 'F_gate', 'drifted']
combined     = pd.concat(
    [sim_clean, ibm_stable_sel[ibm_feat]],
    ignore_index=True
)
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

n_stable  = (combined['drifted'] == 0).sum()
n_drifted = (combined['drifted'] == 1).sum()
drift_rate = round(combined['drifted'].mean() * 100, 1)

print(f"\nCombined dataset:")
print(f"  Sim stable    : 6,000  (simulation)")
print(f"  IBM stable    : {len(ibm_stable_sel):,}  (real hardware anchor)")
print(f"  Sim drifted   : 6,000  (simulation — IBM drifted excluded)")
print(f"  Total         : {len(combined):,}")
print(f"  Stable  (0)   : {n_stable:,}")
print(f"  Drifted (1)   : {n_drifted:,}")
print(f"  Drift rate    : {drift_rate}%")

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

min_sep = min(sep_bell, sep_coh, sep_gate)
if min_sep >= 0.05:
    print(f"\n  GOOD: all features show >=0.05 separation. Proceed.")
else:
    print(f"\n  WARNING: separation below 0.05 on some features.")

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
               label=f'Extreme drift events (n={len(drift_dates)})')
if len(stable_dates) > 0:
    ax.axvline(stable_dates.iloc[0], color='steelblue', alpha=0.4, lw=0.8,
               label=f'Extreme stable events (n={len(stable_dates)})')
ax.set_title(f'Qubit stability — {fig1_backend} qubit {fig1_qubit}\n'
             f'IBM stable rows anchor training · grey zone discarded',
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
    all_vals = pd.concat([s_sim[feat], d_sim[feat], ibm_stable_sel[feat]])
    bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 55)
    ax.hist(s_sim[feat], bins=bins, alpha=0.5, color='steelblue',
            label=f'Sim stable (n={len(s_sim):,})', density=True)
    ax.hist(d_sim[feat], bins=bins, alpha=0.5, color='crimson',
            label=f'Sim drifted (n={len(d_sim):,})', density=True)
    ax.hist(ibm_stable_sel[feat], bins=bins, alpha=0.6, color='navy',
            label=f'IBM stable anchor (n={len(ibm_stable_sel):,})',
            density=True, histtype='step', linewidth=2)
    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Training data fidelity distributions\n'
             'Sim drifted (red) vs sim+IBM stable (blue) · IBM anchors real hardware behavior',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  LABELING COMPLETE")
print(f"{'='*60}")
print(f"  Sim stable         : 6,000")
print(f"  IBM stable anchor  : {len(ibm_stable_sel):,}  ← real hardware grounding")
print(f"  Sim drifted        : 6,000  ← IBM drifted excluded (no boundary signal)")
print(f"  Grey zone          : {zone_counts.get('grey', 0):,}  discarded")
print(f"  Total              : {len(combined):,}")
print(f"  Drift rate         : {drift_rate}%")
print(f"\n  Class separation:")
print(f"    F_bell      : {sep_bell:.4f}")
print(f"    F_coherence : {sep_coh:.4f}")
print(f"    F_gate      : {sep_gate:.4f}")
print(f"\n  Labeling physics:")
print(f"    Stable  : T1 within +-8% AND CZ/readout within +20% of 30d median")
print(f"    Drifted : T1 drop >=35% (TLS) OR CZ >=100% OR readout >=50%")
print(f"    Refs    : Carroll et al. 2022 · ISCA 2025 · Escofet et al. 2025")

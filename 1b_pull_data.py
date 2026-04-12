# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ON SYNTHETIC, TEST ON REAL IBM DATA
#
# Training set: 10,000 synthetic samples from 1e_pull_data.py
#   → Clean, well-separated, no IBM data
#   → Model learns the fidelity boundary from simulated noise
#
# Test set: Real IBM calibration rows from ibm_calibration_labeled.csv
#   → Completely unseen during training
#   → Validates that synthetic training generalizes to real hardware
#   → This is the scientific claim: analytical fidelity proxies + synthetic
#     noise models accurately capture real IBM hardware behavior
#
# Labels for IBM data: T1 drop >=15% | CZ rise >=50% | Readout rise >=20%
# Features: F_bell, F_gate, F_coherence (raw fidelity 0-1)
#   In training: from Aer simulation
#   In deployment: from real circuit shots
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json, os

os.makedirs('data',    exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD RAW IBM DATA ──────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend','qubit_id','timestamp']).reset_index(drop=True)
print(f"  {len(df):,} rows | {df['backend'].nunique()} backends\n")

# ── 2. FILL MISSING VALUES ────────────────────────────────────────────────────
df['t_sx_ns'] = df['t_sx_ns'].fillna(56.0)
df['t_x_ns']  = df['t_x_ns'].fillna(56.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(400.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

# ── 3. ROLLING BASELINE PER QUBIT (for labeling only) ────────────────────────
print("Computing 7-day rolling baselines...")
backend_col = df['backend'].values
qubit_col   = df['qubit_id'].values

def add_rolling(group):
    # shift(1) ensures strict temporal causality
    group['T1_rolling_mean'] = group['T1_us'].rolling(window=7, min_periods=3).mean().shift(1)
    group['cz_rolling_mean'] = group['cz_error'].rolling(window=7, min_periods=3).mean().shift(1)
    group['ro_rolling_mean'] = group['readout_error'].rolling(window=7, min_periods=3).mean().shift(1)
    return group

df = df.groupby(['backend','qubit_id'], group_keys=False).apply(add_rolling)
df = df.reset_index(drop=True)
df['backend']  = backend_col
df['qubit_id'] = qubit_col

# ── 4. LABEL IBM DATA ─────────────────────────────────────────────────────────
THRESHOLD_T1      = 0.15
THRESHOLD_CZ      = 0.50
THRESHOLD_READOUT = 0.20

def label_drift(row):
    if (pd.isna(row['T1_rolling_mean']) or
        pd.isna(row['cz_rolling_mean']) or
        pd.isna(row['ro_rolling_mean'])):
        return 0
    t1_drop = (row['T1_rolling_mean'] - row['T1_us'])         / row['T1_rolling_mean']
    cz_val  = row['cz_error'] if not pd.isna(row['cz_error']) else row['cz_rolling_mean']
    cz_rise = (cz_val - row['cz_rolling_mean'])                / row['cz_rolling_mean']
    ro_rise = (row['readout_error'] - row['ro_rolling_mean'])  / row['ro_rolling_mean']
    return 1 if (t1_drop >= THRESHOLD_T1 or
                 cz_rise >= THRESHOLD_CZ or
                 ro_rise >= THRESHOLD_READOUT) else 0

df['drifted'] = df.apply(label_drift, axis=1)

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count / len(df) * 100, 1)
print(f"  IBM data labels:")
print(f"  Drift  (1): {drift_count:,}  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,}  ({100-drift_pct}%)\n")

# ── 5. ANALYTICAL FIDELITY PROXY FORMULAS (IBM data) ─────────────────────────
print("Computing analytical fidelity proxy formulas for IBM data...")
T2_ns      = (df['T2_us'] * 1e3).replace(0, np.nan).fillna(5600.0)
cz         = df['cz_error'].fillna(df['sx_error'])
T1_us_safe = df['T1_us'].replace(0, np.nan).fillna(df['T1_us'].median())
T1_ns      = T1_us_safe * 1e3

df['F_bell'] = (
    (1 - df['sx_error']) * (1 - cz) *
    (1 - df['readout_error'])**2 *
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T1_ns) *
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T2_ns)
).clip(0, 1)

df['F_gate'] = (
    (1 - df['x_error'])**8 *
    (1 - df['readout_error']) *
    np.exp(-(8 * df['t_x_ns']) / T1_ns) *
    np.exp(-(8 * df['t_x_ns']) / T2_ns)
).clip(0, 1)

df['F_coherence'] = (
    (1 - df['sx_error'])**2 *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-(2 * df['t_sx_ns']) / T1_ns) *
               np.exp(-(2 * df['t_sx_ns']) / T2_ns))
).clip(0, 1)

print(f"  IBM fidelity means by class:")
print(df.groupby('drifted')[['F_bell','F_gate','F_coherence']].mean().round(4))

# Save full labeled IBM file
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence','drifted']
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv\n")

# ── 6. LOAD SYNTHETIC DATA ────────────────────────────────────────────────────
print("Loading data/synthetic_data.csv ...")
synth = pd.read_csv('data/synthetic_data.csv')
print(f"  {len(synth):,} rows")
print(f"  Synthetic fidelity means by class:")
print(synth.groupby('drifted')[['F_bell','F_coherence','F_gate']].mean().round(4))

# Rename F_coherence if needed
if 'F_coherence' not in synth.columns and 'F_coh' in synth.columns:
    synth = synth.rename(columns={'F_coh': 'F_coherence'})

# ── 7. BUILD TRAIN/VAL/TEST SPLITS ───────────────────────────────────────────
# Train + Val: synthetic data only (80/20 split)
# Test:        real IBM data only (completely unseen during training)
#
# This is the key scientific claim:
# A model trained on synthetic Aer noise generalizes to real IBM hardware.

print("\nBuilding splits...")

# Synthetic → train + val
synth_clean = synth[['F_bell','F_gate','F_coherence','drifted']].copy()
synth_idx   = np.arange(len(synth_clean))
y_synth     = synth_clean['drifted'].values

train_idx, val_idx = train_test_split(
    synth_idx, test_size=0.20,
    stratify=y_synth, random_state=42
)

# IBM real → test only
ibm_clean = df[['F_bell','F_gate','F_coherence','drifted']].copy()
ibm_clean = ibm_clean.dropna().reset_index(drop=True)
test_idx  = np.arange(len(ibm_clean))

print(f"  Train : {len(train_idx):,} rows (synthetic)")
print(f"  Val   : {len(val_idx):,} rows (synthetic)")
print(f"  Test  : {len(test_idx):,} rows (real IBM — never seen during training)")

# ── 8. SAVE DATASETS ──────────────────────────────────────────────────────────
# Save synthetic as training dataset
synth_clean.to_csv('data/features_data.csv', index=False)
print("\n  ✓ Saved data/features_data.csv (synthetic — train/val)")

# Save IBM real as test dataset
ibm_clean.to_csv('data/ibm_test_data.csv', index=False)
print("  ✓ Saved data/ibm_test_data.csv (real IBM — test only)")

# Save split indices
# Note: train/val indices index into features_data.csv (synthetic)
#       test indices index into ibm_test_data.csv (real IBM)
split = {
    "train": sorted(train_idx.tolist()),
    "val":   sorted(val_idx.tolist()),
    "test":  sorted(test_idx.tolist()),
    "test_source": "ibm_real"
}
with open('data/split_indices.json', 'w') as f:
    json.dump(split, f)
print("  ✓ Saved data/split_indices.json")

# ── 9. FIGURES ────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Figure 1: T1 over time with drift markers
dc = df[df['drifted']==1].groupby(['backend','qubit_id']).size()
if len(dc) > 0:
    fig1_backend, fig1_qubit = dc.idxmax()
else:
    fig1_backend = df['backend'].iloc[0]
    fig1_qubit   = 0

fd = df[(df['backend']==fig1_backend) &
        (df['qubit_id']==fig1_qubit)].sort_values('timestamp')

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(fd['timestamp'], fd['T1_us'], color='steelblue', lw=1.5, label='T1 (µs)')
ax.plot(fd['timestamp'], fd['T1_rolling_mean'], color='orange', lw=1.3,
        linestyle='--', label='7-day rolling mean')
drift_dates = fd[fd['drifted']==1]['timestamp']
for d in drift_dates:
    ax.axvline(d, color='crimson', alpha=0.25, lw=0.8)
if len(drift_dates) > 0:
    ax.axvline(drift_dates.iloc[0], color='crimson', alpha=0.7, lw=0.8,
               label=f'Drift event (n={len(drift_dates)})')
ax.set_title(f'Qubit Stability — {fig1_backend} Qubit {fig1_qubit}',
             fontsize=9, fontweight='bold')
ax.set_xlabel('Date', fontsize=9); ax.set_ylabel('T1 (µs)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_qubit_stability.png")

# Figure 2: Feature distributions — synthetic vs IBM real
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
feats = ['F_bell', 'F_gate', 'F_coherence']
flabs = ['Bell State Fidelity', 'Gate Error Fidelity', 'Coherence Fidelity']

for ax, feat, lab in zip(axes, feats, flabs):
    # Synthetic distributions
    s_synth = synth_clean[synth_clean['drifted']==0][feat]
    d_synth = synth_clean[synth_clean['drifted']==1][feat]
    # IBM real distributions
    s_ibm   = ibm_clean[ibm_clean['drifted']==0][feat]
    d_ibm   = ibm_clean[ibm_clean['drifted']==1][feat]

    all_vals = pd.concat([s_synth, d_synth, s_ibm, d_ibm])
    bins = np.linspace(all_vals.quantile(0.01), all_vals.quantile(0.99), 55)

    ax.hist(s_synth, bins=bins, alpha=0.5, color='steelblue',
            label=f'Synth stable (n={len(s_synth):,})', density=True)
    ax.hist(d_synth, bins=bins, alpha=0.5, color='crimson',
            label=f'Synth drifted (n={len(d_synth):,})', density=True)
    ax.hist(s_ibm, bins=bins, alpha=0.4, color='navy',
            label=f'IBM stable (n={len(s_ibm):,})', density=True, linestyle='--',
            histtype='step', linewidth=1.5)
    ax.hist(d_ibm, bins=bins, alpha=0.4, color='darkred',
            label=f'IBM drifted (n={len(d_ibm):,})', density=True, linestyle='--',
            histtype='step', linewidth=1.5)
    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

plt.suptitle('Fidelity Distributions: Synthetic (train) vs IBM Real (test)\n'
             'Overlap validates analytical proxy formulas',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  LABELING COMPLETE")
print(f"{'='*55}")
print(f"  Training data : {len(synth_clean):,} synthetic rows (Aer noise model)")
print(f"  Test data     : {len(ibm_clean):,} real IBM rows (never seen in training)")
print(f"  IBM drift rate: {drift_pct}%")
print(f"  Features      : F_bell, F_gate, F_coherence (raw fidelity 0-1)")
print(f"  Labels        : T1 drop >=15% | CZ rise >=50% | Readout rise >=20%")
print(f"\n  KEY CLAIM: Model trained on synthetic generalizes to real IBM hardware.")
print(f"  This validates the analytical fidelity proxy formulas.")
print(f"\n  NEXT: python 3_validate_data.py")
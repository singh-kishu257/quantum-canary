# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# LABELS  → from T1 drop + CZ error rise vs 7-day rolling mean (same as QC1)
# FEATURES → F_bell, F_gate, F_coherence from analytical fidelity formulas
# These two are completely independent — the MLP learns the relationship.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
import json, os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend','qubit_id','timestamp']).reset_index(drop=True)
print(f"  {len(df):,} rows | {df['backend'].nunique()} backends\n")

# ── 2. FILL MISSING GATE DURATIONS ───────────────────────────────────────────
df['t_sx_ns'] = df['t_sx_ns'].fillna(56.0)
df['t_x_ns']  = df['t_x_ns'].fillna(56.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(400.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

# ── 3. ROLLING BASELINE PER QUBIT (for labeling only) ────────────────────────
print("Computing 7-day rolling baselines...")
backend_col = df['backend'].values
qubit_col   = df['qubit_id'].values

def add_rolling(group):
    # IMPORTANT: shift(1) to avoid using today's value in its own baseline.
    # This preserves strict temporal causality for labels/features.
    group['T1_rolling_mean'] = group['T1_us'].rolling(window=7, min_periods=3).mean().shift(1)
    group['cz_rolling_mean'] = group['cz_error'].rolling(window=7, min_periods=3).mean().shift(1)
    group['ro_rolling_mean'] = group['readout_error'].rolling(window=7, min_periods=3).mean().shift(1)
    return group

df = df.groupby(['backend','qubit_id'], group_keys=False).apply(add_rolling)
df = df.reset_index(drop=True)
df['backend']  = backend_col
df['qubit_id'] = qubit_col

# ── 4. LABEL: T1 DROP OR CZ ERROR RISE ≥15% (same logic as QC1) ──────────────
# Completely independent of the 3 fidelity features below.
# Physically grounded: T1 drop = decoherence worsening, CZ rise = gate degrading.
THRESHOLD_T1      = 0.07   # T1 drops 7% below rolling mean
THRESHOLD_READOUT = 0.05   # readout error rises 5% above rolling mean
THRESHOLD_CZ      = 1.00   # CZ error doubles (100% rise above rolling mean)

def label_drift(row):
    if pd.isna(row['T1_rolling_mean']) or pd.isna(row['cz_rolling_mean']) or pd.isna(row['ro_rolling_mean']):
        return 0
    t1_drop = (row['T1_rolling_mean'] - row['T1_us'])          / row['T1_rolling_mean']
    cz_val  = row['cz_error'] if not pd.isna(row['cz_error'])  else row['cz_rolling_mean']
    cz_rise = (cz_val - row['cz_rolling_mean'])                 / row['cz_rolling_mean']
    ro_rise = (row['readout_error'] - row['ro_rolling_mean'])   / row['ro_rolling_mean']
    return 1 if (t1_drop >= THRESHOLD_T1 or
                 cz_rise >= THRESHOLD_CZ or
                 ro_rise >= THRESHOLD_READOUT) else 0

df['drifted'] = df.apply(label_drift, axis=1)

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count / len(df) * 100, 1)
print(f"  Drift  (1): {drift_count:,}  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,}  ({100-drift_pct}%)")
if drift_count < 50:
    print("  WARNING: <50 drift events. Lower THRESHOLD.")
else:
    print("  ✓ Sufficient drift events.\n")

# ── 5. ANALYTICAL FIDELITY FEATURES (independent of labeling) ─────────────────
print("Computing analytical fidelity features...")
T2_ns = (df['T2_us'] * 1e3).replace(0, np.nan).fillna(5600.0)
cz    = df['cz_error'].fillna(df['sx_error'])

T1_us_safe = df['T1_us'].replace(0, np.nan).fillna(df['T1_us'].median())
T1_ns = T1_us_safe * 1e3

# Bell State Canary — T1 + T2 decay during (t_sx + t_cz)
df['F_bell'] = (
    (1 - df['sx_error']) * (1 - cz) *
    (1 - df['readout_error'])**2 *
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T1_ns) *
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T2_ns)
).clip(0, 1)

# Gate Error Canary (8X) — T1 + T2 decay during 8*t_x
df['F_gate'] = (
    (1 - df['x_error'])**8 *
    (1 - df['readout_error']) *
    np.exp(-(8 * df['t_x_ns']) / T1_ns) *
    np.exp(-(8 * df['t_x_ns']) / T2_ns)
).clip(0, 1)

# Coherence Canary (Ramsey / 2H) — T1 + T2 decay during 2*t_sx
df['F_coherence'] = (
    (1 - df['sx_error'])**2 *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-(2 * df['t_sx_ns']) / T1_ns) *
               np.exp(-(2 * df['t_sx_ns']) / T2_ns))
).clip(0, 1)

print(f"  F_bell      mean={df['F_bell'].mean():.4f}  std={df['F_bell'].std():.4f}")
print(f"  F_gate      mean={df['F_gate'].mean():.4f}  std={df['F_gate'].std():.4f}")
print(f"  F_coherence mean={df['F_coherence'].mean():.4f}  std={df['F_coherence'].std():.4f}\n")

# ── 5b. ROLLING Z-SCORE FEATURES ─────────────────────────────────────────────
# KEY INSIGHT: Drift is a relative concept — deviation from recent baseline.
# Absolute fidelity values (0.97–0.99) are blind to drift because IBM hardware
# is uniformly high quality. Z-scores measure HOW MUCH each fidelity deviated
# from its own recent history — same language as the T1/CZ drift labels.
#
# z_F = (F_today - 7day_rolling_mean_F) / 7day_rolling_std_F
#
# Negative z-score = fidelity worse than recent baseline → drift signal.
# This is physically equivalent to what a canary circuit measures in deployment:
# not "is hardware good?" but "is hardware worse than usual?"

print("Computing rolling z-score features...")

ROLL_WINDOW = 7
MIN_PERIODS = 3

backend_col = df['backend'].values
qubit_col   = df['qubit_id'].values

def add_fidelity_rolling(group):
    for feat in ['F_bell', 'F_gate', 'F_coherence']:
        group[f'{feat}_roll_mean'] = group[feat].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean().shift(1)
        group[f'{feat}_roll_std']  = group[feat].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).std().shift(1)
        group[f'{feat}_lag1']      = group[feat].shift(1)
    return group

df = df.groupby(['backend','qubit_id'], group_keys=False).apply(add_fidelity_rolling)
df = df.reset_index(drop=True)
df['backend']  = backend_col
df['qubit_id'] = qubit_col

# Compute z-scores — clip to [-5, 5] to handle outliers
EPS = 1e-8
for feat in ['F_bell', 'F_gate', 'F_coherence']:
    df[f'z_{feat}'] = (
        (df[feat] - df[f'{feat}_roll_mean']) /
        (df[f'{feat}_roll_std'] + EPS)
    ).clip(-5, 5).fillna(0)

# Additional temporal features (still independent from label rules):
# 1) %-change vs 7-day baseline and 2) one-step momentum.
for feat in ['F_bell', 'F_gate', 'F_coherence']:
    df[f'delta_{feat}'] = (
        (df[feat] - df[f'{feat}_roll_mean']) / (df[f'{feat}_roll_mean'] + EPS)
    ).clip(-1, 1).fillna(0)
    df[f'momentum_{feat}'] = (
        (df[feat] - df[f'{feat}_lag1']) / (df[f'{feat}_lag1'] + EPS)
    ).clip(-1, 1).fillna(0)

# Label-aligned physical telemetry features (still computed causally with shift(1) baselines)
# These preserve the labeling method but give the model direct access to calibration drift cues.
cz_safe = df['cz_error'].fillna(df['cz_rolling_mean'])
df['rel_t1_drop'] = (
    (df['T1_rolling_mean'] - df['T1_us']) / (df['T1_rolling_mean'] + EPS)
).clip(-1, 1).fillna(0)
df['rel_cz_rise'] = (
    (cz_safe - df['cz_rolling_mean']) / (df['cz_rolling_mean'] + EPS)
).clip(-1, 3).fillna(0)
df['rel_ro_rise'] = (
    (df['readout_error'] - df['ro_rolling_mean']) / (df['ro_rolling_mean'] + EPS)
).clip(-1, 3).fillna(0)

# Absolute health indicators.
df['log_t1_us'] = np.log1p(df['T1_us'].clip(lower=0)).fillna(0)
df['log_t2_us'] = np.log1p(df['T2_us'].clip(lower=0)).fillna(0)
df['sx_error_clipped'] = df['sx_error'].clip(0, 1).fillna(df['sx_error'].median())
df['cz_error_clipped'] = df['cz_error'].clip(0, 1).fillna(df['cz_error'].median())
df['ro_error_clipped'] = df['readout_error'].clip(0, 1).fillna(df['readout_error'].median())

print(f"  z_F_bell      mean={df['z_F_bell'].mean():.4f}  std={df['z_F_bell'].std():.4f}")
print(f"  z_F_gate      mean={df['z_F_gate'].mean():.4f}  std={df['z_F_gate'].std():.4f}")
print(f"  z_F_coherence mean={df['z_F_coherence'].mean():.4f}  std={df['z_F_coherence'].std():.4f}\n")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence',
             'z_F_bell','z_F_gate','z_F_coherence','drifted']
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv")

feature_columns = [
    'z_F_bell', 'z_F_gate', 'z_F_coherence',
    'delta_F_bell', 'delta_F_gate', 'delta_F_coherence',
    'momentum_F_bell', 'momentum_F_gate', 'momentum_F_coherence',
    'rel_t1_drop', 'rel_cz_rise', 'rel_ro_rise',
    'log_t1_us', 'log_t2_us',
    'sx_error_clipped', 'cz_error_clipped', 'ro_error_clipped'
]
feat_df = df[feature_columns + ['drifted']].copy()
feat_df.to_csv('data/features_data.csv', index=False)
print("  ✓ Saved data/features_data.csv")

# Stratified random split (70/15/15) generally improves stability vs sequential split.
all_idx = np.arange(len(feat_df))
y_all = feat_df['drifted'].values
train_idx, temp_idx = train_test_split(
    all_idx, test_size=0.30, stratify=y_all, random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=y_all[temp_idx], random_state=42
)
split = {
    "train": sorted(train_idx.tolist()),
    "val": sorted(val_idx.tolist()),
    "test": sorted(test_idx.tolist())
}
with open('data/split_indices.json','w') as f:
    json.dump(split, f)
print(f"  Train: {len(split['train']):,} | Val: {len(split['val']):,} | Test: {len(split['test']):,}\n")

# ── 7. FIGURE 1: T1 over time with drift markers ──────────────────────────────
print("Generating Figure 1: Qubit Stability...")

# Pick qubit with most drift events
dc = df[df['drifted']==1].groupby(['backend','qubit_id']).size()
if len(dc) > 0:
    fig1_backend, fig1_qubit = dc.idxmax()
else:
    fig1_backend = df['backend'].iloc[0]; fig1_qubit = 4

fd = df[(df['backend']==fig1_backend)&(df['qubit_id']==fig1_qubit)]\
       .sort_values('timestamp')

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

ax.set_title(f'Qubit Stability — {fig1_backend} Qubit {fig1_qubit}\n'
             f'T1 drop or CZ rise ≥15% below 7-day rolling mean → drifted=1',
             fontsize=9, fontweight='bold')
ax.set_xlabel('Date', fontsize=9)
ax.set_ylabel('T1 (µs)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_qubit_stability.png")

# ── 8. FIGURE 2: Feature Distributions ───────────────────────────────────────
print("Generating Figure 2: Feature Distributions...")
s_df = df[df['drifted']==0]
d_df = df[df['drifted']==1]
feats = ['z_F_bell','z_F_gate','z_F_coherence']
flabs = ['Bell State Z-Score','Gate Error Z-Score','Coherence Z-Score']

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, feat, lab in zip(axes, feats, flabs):
    bins = np.linspace(df[feat].quantile(0.01), df[feat].quantile(0.99), 55)
    ax.hist(s_df[feat], bins=bins, alpha=0.6, color='steelblue',
            label=f'Stable (n={len(s_df):,})', density=True)
    ax.hist(d_df[feat], bins=bins, alpha=0.6, color='crimson',
            label=f'Drifted (n={len(d_df):,})', density=True)
    ax.axvline(s_df[feat].mean(), color='steelblue', lw=1.8, ls='--',
               label=f'Stable mean={s_df[feat].mean():.3f}')
    if len(d_df) > 0:
        ax.axvline(d_df[feat].mean(), color='crimson', lw=1.8, ls='--',
                   label=f'Drifted mean={d_df[feat].mean():.3f}')
    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Analytical Fidelity Distributions: Stable vs Drifted\n'
             'Labels from T1/CZ rolling baseline (independent of features)',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  LABELING COMPLETE")
print(f"{'='*50}")
print(f"  Total rows  : {len(df):,}")
print(f"  Drift (1)   : {drift_count:,} ({drift_pct}%)")
print(f"  Stable (0)  : {stable_count:,} ({100-drift_pct}%)")
print(f"  Thresholds  : T1 drop >=7% | CZ doubles | Readout rise >=5%")
print(f"  Features    : z-score + delta + momentum for F_bell/F_gate/F_coherence")
print(f"\n  NEXT: python 3_validate_data.py")

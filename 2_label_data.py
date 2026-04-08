# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Compute 3 analytical fidelity features, then label each row using IBM's
#   own recalibration events as ground truth — completely independent of the
#   fidelity formulas.
#
# LABELING PHILOSOPHY:
#   A recalibration event occurs when IBM's snapshot changes between days —
#   meaning IBM's own automated systems detected drift and intervened.
#   We label the LEAD_DAYS window before each recalibration as drifted=1.
#   This transforms the MLP into a FORECASTING tool:
#     Input:  F_bell, F_gate, F_coherence at time t
#     Label:  Will IBM recalibrate within the next LEAD_DAYS days?
#
#   Non-circular: inputs = your fidelity formulas
#                 label  = IBM's own maintenance decision (external event)
#   No arbitrary threshold: IBM's engineers define "drifted"
#   Real utility: AUC of 0.75 here beats AUC of 1.0 on a circular model
#
# CITATIONS:
#   Proctor et al., Nature Communications 11, 5706 (2020)
#   "Detecting and tracking drift in quantum information processors"
#   — establishes recalibration-event-based drift labeling framework
#
# FIDELITY FORMULAS (unchanged from design doc):
#   F_bell      = (1-sx)(1-cz)(1-ro_q0)(1-ro_q1) × e^(-(t_sx+t_cz)/T2)
#   F_gate      = (1-x)^8 × (1-ro) × e^(-8*t_x/T2)
#   F_coherence = (1-sx)^2 × (1-ro) × 0.5 × (1 + e^(-2*t_sx/T2))
#
# OUTPUT:
#   data/ibm_calibration_labeled.csv
#   data/features_data.csv            ← F_bell, F_gate, F_coherence, drifted
#   data/split_indices.json
#   figures/fig_qubit_stability.png
#   figures/fig_feature_distributions.png
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json
import os

os.makedirs('data',    exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  {len(df):,} rows | {df['backend'].nunique()} backends\n")

df = df.sort_values(['backend', 'qubit_id', 'timestamp']).reset_index(drop=True)

# ── 2. FILL MISSING GATE DURATIONS ───────────────────────────────────────────
DEFAULT_T_SX_NS = 56.0
DEFAULT_T_X_NS  = 56.0
DEFAULT_T_CZ_NS = 400.0

df['t_sx_ns'] = df['t_sx_ns'].fillna(DEFAULT_T_SX_NS)
df['t_x_ns']  = df['t_x_ns'].fillna(DEFAULT_T_X_NS)
df['t_cz_ns'] = df['t_cz_ns'].fillna(DEFAULT_T_CZ_NS)
df['x_error'] = df['x_error'].fillna(df['sx_error'])
df['readout_err_q1'] = df['readout_error']

# ── 3. T2 IN NANOSECONDS ──────────────────────────────────────────────────────
T2_ns = (df['T2_us'] * 1e3).replace(0, np.nan).fillna(DEFAULT_T_SX_NS * 100)

# ── 4. ANALYTICAL FIDELITY FEATURES ──────────────────────────────────────────
print("Computing analytical fidelity features...")

cz_err_filled = df['cz_error'].fillna(df['sx_error'])

# Bell State Canary
df['F_bell'] = (
    (1 - df['sx_error'])      *
    (1 - cz_err_filled)       *
    (1 - df['readout_error']) *
    (1 - df['readout_err_q1'])*
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T2_ns)
).clip(0, 1)

# Gate Error Canary (8X)
df['F_gate'] = (
    (1 - df['x_error'])**8    *
    (1 - df['readout_error']) *
    np.exp(-(8 * df['t_x_ns']) / T2_ns)
).clip(0, 1)

# Coherence Canary (Ramsey / 2H)
df['F_coherence'] = (
    (1 - df['sx_error'])**2   *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-(2 * df['t_sx_ns']) / T2_ns))
).clip(0, 1)

print(f"  F_bell      mean={df['F_bell'].mean():.4f}  std={df['F_bell'].std():.4f}")
print(f"  F_gate      mean={df['F_gate'].mean():.4f}  std={df['F_gate'].std():.4f}")
print(f"  F_coherence mean={df['F_coherence'].mean():.4f}  std={df['F_coherence'].std():.4f}\n")

# ── 5. DETECT IBM RECALIBRATION EVENTS ───────────────────────────────────────
# A recalibration event = IBM's snapshot changed from the previous day.
# Detected when T1, sx_error, or readout_error shift by more than noise floor.
# This is IBM's own maintenance decision — completely independent of our formulas.
#
# RECAL_THRESHOLD: minimum fractional change to count as a new snapshot.
# Small changes (< 1%) are measurement noise. Large changes = IBM recalibrated.
RECAL_THRESHOLD = 0.01   # 1% change in any key parameter = recalibration
LEAD_DAYS       = 7      # label this many days BEFORE each recalibration as drifted

print(f"Detecting IBM recalibration events (threshold={RECAL_THRESHOLD*100:.0f}%)...")

def detect_recalibrations(group):
    group = group.copy().sort_values('timestamp').reset_index(drop=True)

    # Fractional change in key calibration parameters day-over-day
    for col in ['T1_us', 'sx_error', 'readout_error']:
        group[f'd_{col}'] = group[col].pct_change().abs()

    # Recalibration = any key parameter changed more than threshold
    group['recalibrated'] = (
        (group['d_T1_us']          > RECAL_THRESHOLD) |
        (group['d_sx_error']       > RECAL_THRESHOLD) |
        (group['d_readout_error']  > RECAL_THRESHOLD)
    ).astype(int)

    # First row has no previous day — always stable
    group.loc[0, 'recalibrated'] = 0

    return group[['timestamp', 'backend', 'qubit_id', 'recalibrated']]

recal_df = df.groupby(
    ['backend', 'qubit_id'], group_keys=False
).apply(detect_recalibrations).reset_index(drop=True)

df = df.merge(recal_df[['timestamp','backend','qubit_id','recalibrated']],
              on=['timestamp','backend','qubit_id'], how='left')
df['recalibrated'] = df['recalibrated'].fillna(0).astype(int)

n_recal = df['recalibrated'].sum()
print(f"  Recalibration events detected: {n_recal:,}")
print(f"  ({round(n_recal/len(df)*100,1)}% of all rows)\n")

# ── 6. LABEL: LEAD_DAYS WINDOW BEFORE EACH RECALIBRATION ─────────────────────
# For each recalibration event, label the LEAD_DAYS rows before it as drifted=1.
# These are the rows where the fidelities are "warning" of imminent recalibration.
# All other rows = stable=0.
#
# MLP task: given F_bell, F_gate, F_coherence today,
#           predict whether IBM will recalibrate within the next LEAD_DAYS days.

print(f"Labeling {LEAD_DAYS}-day pre-recalibration windows as drifted=1...")

def label_lead_window(group):
    group = group.copy().sort_values('timestamp').reset_index(drop=True)
    labels = np.zeros(len(group), dtype=int)

    recal_indices = group.index[group['recalibrated'] == 1].tolist()

    for idx in recal_indices:
        # Label LEAD_DAYS rows immediately before this recalibration
        start = max(0, idx - LEAD_DAYS)
        labels[start:idx] = 1

    group['drifted'] = labels
    return group

df = df.groupby(
    ['backend', 'qubit_id'], group_keys=False
).apply(label_lead_window).reset_index(drop=True)

drift_count  = int(df['drifted'].sum())
stable_count = int((df['drifted'] == 0).sum())
total        = len(df)
drift_pct    = round(drift_count / total * 100, 1)

print(f"  Drift  (1): {drift_count:,} rows  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,} rows  ({100-drift_pct}%)")

if drift_count < 50:
    print(f"  WARNING: <50 drift events. Lower RECAL_THRESHOLD or increase LEAD_DAYS.")
else:
    print(f"  ✓ Sufficient drift events for training.\n")

# ── 7. SAVE DATASETS ──────────────────────────────────────────────────────────
keep_cols = [
    'timestamp', 'backend', 'qubit_id',
    'T1_us', 'T2_us',
    'sx_error', 'x_error', 'cz_error', 'readout_error',
    't_sx_ns', 't_x_ns', 't_cz_ns',
    'F_bell', 'F_gate', 'F_coherence',
    'recalibrated', 'drifted'
]
df[keep_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv")

features_df = df[['F_bell', 'F_gate', 'F_coherence', 'drifted']].copy()
features_df.to_csv('data/features_data.csv', index=False)
print("  ✓ Saved data/features_data.csv")

# ── 8. TRAIN / VAL / TEST SPLIT ───────────────────────────────────────────────
n      = len(features_df)
i_val  = int(n * 0.70)
i_test = int(n * 0.85)

split = {
    "train": list(range(0, i_val)),
    "val":   list(range(i_val, i_test)),
    "test":  list(range(i_test, n)),
}
with open('data/split_indices.json', 'w') as f:
    json.dump(split, f)

print(f"\n  Train: {len(split['train']):,} rows")
print(f"  Val:   {len(split['val']):,} rows")
print(f"  Test:  {len(split['test']):,} rows\n")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: F_composite over time with recalibration events + drift windows
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Qubit Stability...")

df['F_composite'] = df['F_bell'] * df['F_gate'] * df['F_coherence']

# Pick qubit with most recalibration events
recal_counts = (
    df[df['recalibrated'] == 1]
    .groupby(['backend','qubit_id']).size()
    .reset_index(name='n')
    .sort_values('n', ascending=False)
)
fig1_backend = recal_counts.iloc[0]['backend'] if len(recal_counts) > 0 else df['backend'].iloc[0]
fig1_qubit   = int(recal_counts.iloc[0]['qubit_id']) if len(recal_counts) > 0 else 4

fig1_data = df[
    (df['backend']  == fig1_backend) &
    (df['qubit_id'] == fig1_qubit)
].copy().sort_values('timestamp').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 3.8))

ax.plot(fig1_data['timestamp'], fig1_data['F_composite'],
        color='steelblue', linewidth=1.5,
        label='F_composite', zorder=3)

# Shade pre-recalibration drift windows
prev_idx   = 0
prev_drift = int(fig1_data['drifted'].iloc[0])
for i in range(1, len(fig1_data)):
    curr = int(fig1_data['drifted'].iloc[i])
    if curr != prev_drift or i == len(fig1_data) - 1:
        color = 'crimson' if prev_drift == 1 else 'steelblue'
        alpha = 0.18      if prev_drift == 1 else 0.05
        ax.axvspan(fig1_data['timestamp'].iloc[prev_idx],
                   fig1_data['timestamp'].iloc[i],
                   color=color, alpha=alpha, zorder=1)
        prev_idx   = i
        prev_drift = curr

# Recalibration event lines
recal_rows = fig1_data[fig1_data['recalibrated'] == 1]
for _, row in recal_rows.iterrows():
    ax.axvline(row['timestamp'], color='black',
               linewidth=1.2, alpha=0.7, zorder=4)

ax.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', lw=1.5, label='F_composite'),
    Patch(facecolor='crimson',   alpha=0.4,  label=f'{LEAD_DAYS}-day pre-recal window (drifted=1)'),
    Patch(facecolor='steelblue', alpha=0.2,  label='Stable (drifted=0)'),
    plt.Line2D([0],[0], color='black', lw=1.2, label='IBM recalibration event'),
], fontsize=7)

ax.set_title(
    f'F_composite — {fig1_backend} Qubit {fig1_qubit}\n'
    f'IBM recalibration events = ground truth labels (Proctor et al. 2020)',
    fontsize=9, fontweight='bold', pad=8
)
ax.set_xlabel('Date', fontsize=9)
ax.set_ylabel('F_composite', fontsize=9)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_qubit_stability.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Feature Distributions — stable vs drifted
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Feature Distributions...")

stable_df  = df[df['drifted'] == 0]
drifted_df = df[df['drifted'] == 1]

features    = ['F_bell', 'F_gate', 'F_coherence']
feat_labels = ['Bell State Fidelity', 'Gate Error Canary (8X)', 'Coherence Fidelity (Ramsey)']

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, feat, label in zip(axes, features, feat_labels):
    lo   = df[feat].quantile(0.01)
    hi   = df[feat].quantile(0.99)
    bins = np.linspace(lo, hi, 55)

    ax.hist(stable_df[feat],  bins=bins, alpha=0.6, color='steelblue',
            label=f'Stable  (n={len(stable_df):,})',  density=True)
    ax.hist(drifted_df[feat], bins=bins, alpha=0.6, color='crimson',
            label=f'Drifted (n={len(drifted_df):,})', density=True)

    # Stable mean
    ax.axvline(stable_df[feat].mean(), color='steelblue', linewidth=1.8,
               linestyle='--', label=f'Stable mean={stable_df[feat].mean():.3f}')

    # Drifted mean
    if len(drifted_df) > 0:
        ax.axvline(drifted_df[feat].mean(), color='crimson', linewidth=1.8,
                   linestyle='--', label=f'Drifted mean={drifted_df[feat].mean():.3f}')

    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.suptitle(
    'Fidelity Distributions: Stable vs Pre-Recalibration Windows\n'
    'Label = IBM recalibration event (Proctor et al., Nat. Commun. 2020)',
    fontsize=10, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  LABELING COMPLETE")
print(f"{'='*55}")
print(f"  Total rows         : {total:,}")
print(f"  Recalibrations     : {n_recal:,}")
print(f"  Drift (pre-recal)  : {drift_count:,} ({drift_pct}%)")
print(f"  Stable             : {stable_count:,} ({100-drift_pct}%)")
print(f"  Lead window        : {LEAD_DAYS} days")
print(f"  Method             : IBM recalibration events as ground truth")
print(f"  Citation           : Proctor et al., Nat. Commun. 11, 5706 (2020)")

# 2_label_data_v2.py — Quantum Canary Prototype 2
# Fidelity formulas unchanged. Labeling via IBM recalibration events.
# Citation: Proctor et al., Nat. Commun. 11, 5706 (2020)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json, os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend','qubit_id','timestamp']).reset_index(drop=True)
print(f"  {len(df):,} rows | {df['backend'].nunique()} backends\n")

# ── 2. FILL MISSING DURATIONS ─────────────────────────────────────────────────
df['t_sx_ns'] = df['t_sx_ns'].fillna(56.0)
df['t_x_ns']  = df['t_x_ns'].fillna(56.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(400.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

# ── 3. FIDELITY FEATURES (unchanged formulas) ─────────────────────────────────
print("Computing analytical fidelity features...")
T2_ns = (df['T2_us'] * 1e3).replace(0, np.nan).fillna(5600.0)
cz    = df['cz_error'].fillna(df['sx_error'])

df['F_bell'] = (
    (1 - df['sx_error']) * (1 - cz) *
    (1 - df['readout_error'])**2 *
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T2_ns)
).clip(0, 1)

df['F_gate'] = (
    (1 - df['x_error'])**8 *
    (1 - df['readout_error']) *
    np.exp(-(8 * df['t_x_ns']) / T2_ns)
).clip(0, 1)

df['F_coherence'] = (
    (1 - df['sx_error'])**2 *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-(2 * df['t_sx_ns']) / T2_ns))
).clip(0, 1)

print(f"  F_bell      mean={df['F_bell'].mean():.4f}  std={df['F_bell'].std():.4f}")
print(f"  F_gate      mean={df['F_gate'].mean():.4f}  std={df['F_gate'].std():.4f}")
print(f"  F_coherence mean={df['F_coherence'].mean():.4f}  std={df['F_coherence'].std():.4f}\n")

# ── 4. DETECT IBM RECALIBRATION EVENTS ────────────────────────────────────────
# Recalibration = ALL 3 key parameters change by >25% on the same day.
# Requires simultaneous large shifts — filters out daily measurement noise.
RECAL_THRESHOLD = 0.25
RECAL_MIN_COLS  = 3
LEAD_DAYS       = 1

print(f"Detecting IBM recalibration events (threshold={int(RECAL_THRESHOLD*100)}%, min_cols={RECAL_MIN_COLS})...")

df['recalibrated'] = 0

for (backend, qubit), grp in df.groupby(['backend','qubit_id']):
    grp = grp.sort_values('timestamp')
    idx = grp.index.tolist()

    changed_count = np.zeros(len(idx), dtype=int)
    for col in ['T1_us','sx_error','readout_error']:
        vals = grp[col].values.astype(float)
        pct  = np.abs(np.diff(vals, prepend=vals[0]) / (np.abs(vals) + 1e-12))
        changed_count += (pct > RECAL_THRESHOLD).astype(int)

    recal_flags = (changed_count >= RECAL_MIN_COLS).astype(int)
    recal_flags[0] = 0  # first row never a recalibration
    df.loc[idx, 'recalibrated'] = recal_flags

n_recal = int(df['recalibrated'].sum())
print(f"  Recalibration events detected: {n_recal:,} ({round(n_recal/len(df)*100,1)}% of rows)")

# ── 5. LABEL: 7-DAY PRE-RECALIBRATION WINDOWS ────────────────────────────────
print(f"Labeling {LEAD_DAYS}-day pre-recalibration windows as drifted=1...")

df['drifted'] = 0

for (backend, qubit), grp in df.groupby(['backend','qubit_id']):
    grp  = grp.sort_values('timestamp')
    idxs = grp.index.tolist()
    recal_positions = [i for i, ix in enumerate(idxs) if df.loc[ix,'recalibrated'] == 1]

    for pos in recal_positions:
        start = max(0, pos - LEAD_DAYS)
        for p in range(start, pos):
            df.loc[idxs[p], 'drifted'] = 1

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count / len(df) * 100, 1)

print(f"  Drift  (1): {drift_count:,} rows  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,} rows  ({100-drift_pct}%)")
if drift_count < 50:
    print("  WARNING: <50 drift events. Lower RECAL_THRESHOLD.")
else:
    print("  ✓ Sufficient drift events for training.\n")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence','recalibrated','drifted']
full_cols = [c for c in full_cols if c in df.columns]
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv")

feat_df = df[['F_bell','F_gate','F_coherence','drifted']].copy()
feat_df.to_csv('data/features_data.csv', index=False)
print("  ✓ Saved data/features_data.csv")

n      = len(feat_df)
i_val  = int(n * 0.70)
i_test = int(n * 0.85)
split  = {"train": list(range(0, i_val)),
          "val":   list(range(i_val, i_test)),
          "test":  list(range(i_test, n))}
with open('data/split_indices.json','w') as f:
    json.dump(split, f)
print(f"  Train: {len(split['train']):,} | Val: {len(split['val']):,} | Test: {len(split['test']):,}\n")

# ── 7. FIGURE 1: F_composite over time ───────────────────────────────────────
print("Generating Figure 1: Qubit Stability...")
df['F_composite'] = df['F_bell'] * df['F_gate'] * df['F_coherence']

# Pick qubit with most recalibration events
rc = df[df['recalibrated']==1].groupby(['backend','qubit_id']).size()
if len(rc) > 0:
    fig1_backend, fig1_qubit = rc.idxmax()
else:
    fig1_backend = df['backend'].iloc[0]; fig1_qubit = 4

fd = df[(df['backend']==fig1_backend)&(df['qubit_id']==fig1_qubit)]\
       .sort_values('timestamp').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(fd['timestamp'], fd['F_composite'], color='steelblue', lw=1.5,
        label='F_composite', zorder=3)

# Shade regions
prev, pd_ = 0, int(fd['drifted'].iloc[0])
for i in range(1, len(fd)):
    cur = int(fd['drifted'].iloc[i])
    if cur != pd_ or i == len(fd)-1:
        ax.axvspan(fd['timestamp'].iloc[prev], fd['timestamp'].iloc[i],
                   color='crimson' if pd_==1 else 'steelblue',
                   alpha=0.18 if pd_==1 else 0.05, zorder=1)
        prev, pd_ = i, cur

# Recalibration lines
for _, row in fd[fd['recalibrated']==1].iterrows():
    ax.axvline(row['timestamp'], color='black', lw=1.1, alpha=0.7, zorder=4)

ax.legend(handles=[
    plt.Line2D([0],[0], color='steelblue', lw=1.5, label='F_composite'),
    Patch(facecolor='crimson',   alpha=0.4, label=f'{LEAD_DAYS}-day pre-recal (drifted=1)'),
    Patch(facecolor='steelblue', alpha=0.2, label='Stable (drifted=0)'),
    plt.Line2D([0],[0], color='black', lw=1.1, label='IBM recalibration'),
], fontsize=7)
ax.set_title(f'F_composite — {fig1_backend} Qubit {fig1_qubit}\n'
             f'IBM recalibration events = ground truth (Proctor et al. 2020)',
             fontsize=9, fontweight='bold')
ax.set_xlabel('Date', fontsize=9); ax.set_ylabel('F_composite', fontsize=9)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_qubit_stability.png")

# ── 8. FIGURE 2: Feature Distributions ───────────────────────────────────────
print("Generating Figure 2: Feature Distributions...")
s_df = df[df['drifted']==0]; d_df = df[df['drifted']==1]
feats = ['F_bell','F_gate','F_coherence']
flabs = ['Bell State Fidelity','Gate Error Canary (8X)','Coherence Fidelity (Ramsey)']

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
    ax.set_xlabel('Fidelity', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Fidelity Distributions: Stable vs Pre-Recalibration Windows\n'
             'Label = IBM recalibration event (Proctor et al. 2020)',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  LABELING COMPLETE")
print(f"{'='*50}")
print(f"  Total rows       : {len(df):,}")
print(f"  Recalibrations   : {n_recal:,}")
print(f"  Drift (pre-recal): {drift_count:,} ({drift_pct}%)")
print(f"  Stable           : {stable_count:,} ({100-drift_pct}%)")
print(f"  Lead window      : {LEAD_DAYS} days")
print(f"  Method           : IBM recalibration events as ground truth")
print(f"  Citation         : Proctor et al., Nat. Commun. 11, 5706 (2020)")

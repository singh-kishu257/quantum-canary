# 2_label_data.py — Quantum Canary Prototype 2
# Labels: T1 drop ≥7% OR readout rise ≥5% OR CZ rise ≥100% vs 7-day rolling mean
# Features: z_F_bell, z_F_gate, z_F_coherence (rolling z-scores)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend','qubit_id','timestamp']).reset_index(drop=True)
print(f"  {len(df):,} rows | {df['backend'].nunique()} backends\n")

# ── 2. FILL MISSING ───────────────────────────────────────────────────────────
df['t_sx_ns'] = df['t_sx_ns'].fillna(56.0)
df['t_x_ns']  = df['t_x_ns'].fillna(56.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(400.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

# ── 3. ALL COMPUTATION PER QUBIT IN ONE LOOP ──────────────────────────────────
print("Processing per qubit (rolling baselines, fidelities, z-scores, labels)...")

THRESHOLD_T1      = 0.07   # T1 drops 7% below 7-day rolling mean
THRESHOLD_READOUT = 0.05   # readout rises 5% above 7-day rolling mean
THRESHOLD_CZ      = 1.00   # CZ rises 100% (doubles) above 7-day rolling mean
ROLL_WINDOW       = 7
MIN_PERIODS       = 3
T_WAIT_NS         = 1000.0
EPS               = 1e-8

all_rows = []

for (backend, qubit), grp in df.groupby(['backend','qubit_id']):
    grp = grp.copy().sort_values('timestamp').reset_index(drop=True)

    # Rolling baselines for labeling
    grp['T1_roll']  = grp['T1_us'].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()
    grp['cz_roll']  = grp['cz_error'].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()
    grp['ro_roll']  = grp['readout_error'].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()

    # Labels — all three compared to rolling mean (relative)
    grp['drifted'] = 0
    for pos in range(len(grp)):
        t1_roll = grp.loc[pos, 'T1_roll']
        cz_roll = grp.loc[pos, 'cz_roll']
        ro_roll = grp.loc[pos, 'ro_roll']
        if pd.isna(t1_roll) or pd.isna(cz_roll) or pd.isna(ro_roll): continue

        t1_val  = grp.loc[pos, 'T1_us']
        ro_val  = grp.loc[pos, 'readout_error']
        cz_val  = grp.loc[pos, 'cz_error'] if not pd.isna(grp.loc[pos,'cz_error']) else cz_roll

        t1_drop  = (t1_roll - t1_val) / t1_roll if t1_roll > 0 else 0
        cz_rise  = (cz_val - cz_roll) / cz_roll if cz_roll > 0 else 0
        ro_rise  = (ro_val - ro_roll) / ro_roll if ro_roll > 0 else 0

        if t1_drop >= THRESHOLD_T1 or cz_rise >= THRESHOLD_CZ or ro_rise >= THRESHOLD_READOUT:
            grp.loc[pos, 'drifted'] = 1

    # Fidelity features
    T2_ns = (grp['T2_us'] * 1e3).replace(0, np.nan).fillna(5600.0)
    T1_ns = grp['T1_us'].replace(0, np.nan).fillna(grp['T1_us'].median()) * 1e3
    cz    = grp['cz_error'].fillna(grp['sx_error'])

    grp['F_bell'] = (
        (1 - grp['sx_error']) * (1 - cz) *
        (1 - grp['readout_error'])**2 *
        np.exp(-(grp['t_sx_ns'] + grp['t_cz_ns']) / T1_ns) *
        np.exp(-(grp['t_sx_ns'] + grp['t_cz_ns']) / T2_ns)
    ).clip(0, 1)

    grp['F_gate'] = (
        (1 - grp['x_error'])**8 *
        (1 - grp['readout_error']) *
        np.exp(-(8 * grp['t_x_ns']) / T1_ns) *
        np.exp(-(8 * grp['t_x_ns']) / T2_ns)
    ).clip(0, 1)

    grp['F_coherence'] = (
        (1 - grp['sx_error'])**2 *
        (1 - grp['readout_error']) *
        0.5 * (1 + np.exp(-T_WAIT_NS / T2_ns))
    ).clip(0, 1)

    # Z-scores
    for feat in ['F_bell','F_gate','F_coherence']:
        rm = grp[feat].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()
        rs = grp[feat].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).std()
        grp[f'z_{feat}'] = ((grp[feat] - rm) / (rs + EPS)).clip(-5, 5).fillna(0)

    all_rows.append(grp)

df = pd.concat(all_rows, ignore_index=True)

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count / len(df) * 100, 1)
print(f"  Drift  (1): {drift_count:,}  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,}  ({100-drift_pct}%)\n")

# ── 4. SAVE ───────────────────────────────────────────────────────────────────
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence',
             'z_F_bell','z_F_gate','z_F_coherence','drifted']
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv")

feat_df = df[['z_F_bell','z_F_gate','z_F_coherence','drifted']].copy()
feat_df.to_csv('data/features_data.csv', index=False)
print(f"  ✓ Saved data/features_data.csv ({len(feat_df):,} rows)")

n      = len(feat_df)
i_val  = int(n * 0.70)
i_test = int(n * 0.85)
split  = {"train": list(range(0, i_val)),
          "val":   list(range(i_val, i_test)),
          "test":  list(range(i_test, n))}
with open('data/split_indices.json','w') as f:
    json.dump(split, f)
print(f"  Train: {i_val:,} | Val: {i_test-i_val:,} | Test: {n-i_test:,}\n")

# ── 5. FIGURE 1 ───────────────────────────────────────────────────────────────
print("Generating figures...")
dc = df[df['drifted']==1].groupby(['backend','qubit_id']).size()
fig1_backend = dc.idxmax()[0] if len(dc) > 0 else df['backend'].iloc[0]
fig1_qubit   = dc.idxmax()[1] if len(dc) > 0 else 4
fd = df[(df['backend']==fig1_backend)&(df['qubit_id']==fig1_qubit)].sort_values('timestamp')

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(fd['timestamp'], fd['T1_us'], color='steelblue', lw=1.5, label='T1 (µs)')
ax.plot(fd['timestamp'], fd['T1_roll'], color='orange', lw=1.3, ls='--', label='7-day rolling mean')
drift_dates = fd[fd['drifted']==1]['timestamp']
for d in drift_dates:
    ax.axvline(d, color='crimson', alpha=0.25, lw=0.8)
if len(drift_dates) > 0:
    ax.axvline(drift_dates.iloc[0], color='crimson', alpha=0.7, lw=0.8,
               label=f'Drift event (n={len(drift_dates)})')
ax.set_title(f'Qubit Stability — {fig1_backend} Qubit {fig1_qubit}\n'
             f'T1 drop ≥7% | Readout rise ≥5% | CZ rise ≥100% vs rolling mean → drifted=1',
             fontsize=9, fontweight='bold')
ax.set_xlabel('Date', fontsize=9); ax.set_ylabel('T1 (µs)', fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_qubit_stability.png")

# ── 6. FIGURE 2 ───────────────────────────────────────────────────────────────
s_df = df[df['drifted']==0]; d_df = df[df['drifted']==1]
feats = ['z_F_bell','z_F_gate','z_F_coherence']
flabs = ['Bell State Z-Score','Gate Error Z-Score','Coherence Z-Score']
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, feat, lab in zip(axes, feats, flabs):
    bins = np.linspace(-3, 3, 55)
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
    ax.set_xlabel('Z-Score', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle('Analytical Fidelity Z-Score Distributions: Stable vs Drifted\n'
             'Labels from T1/CZ/Readout rolling baseline (independent of features)',
             fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

print(f"\n{'='*50}")
print(f"  LABELING COMPLETE")
print(f"{'='*50}")
print(f"  Total rows  : {len(df):,}")
print(f"  Drift  (1)  : {drift_count:,} ({drift_pct}%)")
print(f"  Stable (0)  : {stable_count:,} ({100-drift_pct}%)")
print(f"  Features    : z_F_bell, z_F_gate, z_F_coherence")
print(f"\n  NEXT: python 3_validate_data.py")
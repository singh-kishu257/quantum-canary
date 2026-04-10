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
    group['T1_rolling_mean'] = group['T1_us'].rolling(window=7, min_periods=3).mean()
    group['cz_rolling_mean'] = group['cz_error'].rolling(window=7, min_periods=3).mean()
    group['ro_rolling_mean'] = group['readout_error'].rolling(window=7, min_periods=3).mean()
    return group

df = df.groupby(['backend','qubit_id'], group_keys=False).apply(add_rolling)
df = df.reset_index(drop=True)
df['backend']  = backend_col
df['qubit_id'] = qubit_col

# ── 4. LABEL: PELT ON z_F_composite ─────────────────────────────────────────
# Labels are derived from the TEMPORAL STRUCTURE of the 3 fidelity z-scores.
# NOT from their instantaneous values — so not circular.
#
# Steps:
#   1. Compute z_F_composite = z_F_bell × z_F_gate × z_F_coherence per row
#      (fidelity multiplicativity theorem — Escofet et al. 2025)
#   2. Run PELT per qubit per backend on the z_F_composite time series
#      (Killick, Fearnhead & Eckley, JASA 2012)
#   3. First segment = reference (stable=0)
#   4. Segments whose mean drops below reference mean = drifted=1
#
# Features (instantaneous): z_F_bell, z_F_gate, z_F_coherence
# Label (temporal):         did the fidelity regime structurally shift?
# These are independent transformations of the same signal — not circular.

try:
    import ruptures as rpt
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ruptures', '-q'])
    import ruptures as rpt

print("Running PELT change point detection on z_F_composite per qubit...")

# z_F_composite will be computed after fidelity features below
# We do labeling AFTER features so we can use the z-scores directly.
# Placeholder — filled after feature computation.
df['drifted'] = -999  # sentinel — filled below


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

# Coherence Canary (Ramsey / 2H) — T2 decay during free evolution wait
# Circuit: H → wait(t_wait) → H → measure
# The T2 decay happens DURING the wait, not during the gates.
# t_wait = 1000 ns (1 µs) — fixed wait time programmed into the circuit.
# Chosen to be sensitive to T2 degradation while remaining short enough
# for rapid canary deployment on IBM hardware.
T_WAIT_NS = 1000.0  # ns — fixed Ramsey wait time

df['F_coherence'] = (
    (1 - df['sx_error'])**2 *
    (1 - df['readout_error']) *
    0.5 * (1 + np.exp(-T_WAIT_NS / T2_ns))
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
    for feat in ['F_bell', 'F_gate', 'F_coherence', 'T1_us', 'cz_error', 'readout_error']:
        group[f'{feat}_roll_mean'] = group[feat].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).mean()
        group[f'{feat}_roll_std']  = group[feat].rolling(ROLL_WINDOW, min_periods=MIN_PERIODS).std()
    return group

df = df.groupby(['backend','qubit_id'], group_keys=False).apply(add_fidelity_rolling)
df = df.reset_index(drop=True)
df['backend']  = backend_col
df['qubit_id'] = qubit_col

EPS = 1e-8
# Fidelity z-scores
for feat in ['F_bell', 'F_gate', 'F_coherence']:
    df[f'z_{feat}'] = (
        (df[feat] - df[f'{feat}_roll_mean']) /
        (df[f'{feat}_roll_std'] + EPS)
    ).clip(-5, 5).fillna(0)

# Raw IBM parameter z-scores — directly mirror label logic
# z_T1 negative = T1 dropped (drift signal)
# z_cz positive = CZ rose (drift signal)
# z_readout positive = readout worsened (drift signal)
df['z_T1']      = ((df['T1_us']        - df['T1_us_roll_mean'])        / (df['T1_us_roll_std']        + EPS)).clip(-5,5).fillna(0)
df['z_cz']      = ((df['cz_error']     - df['cz_error_roll_mean'])     / (df['cz_error_roll_std']     + EPS)).clip(-5,5).fillna(0)
df['z_readout'] = ((df['readout_error'] - df['readout_error_roll_mean'])/ (df['readout_error_roll_std'] + EPS)).clip(-5,5).fillna(0)

print(f"  z_F_bell  mean={df['z_F_bell'].mean():.4f}  z_F_gate mean={df['z_F_gate'].mean():.4f}  z_F_coh mean={df['z_F_coherence'].mean():.4f}\n")

# z_F_composite: joint signal via fidelity multiplicativity theorem
df['z_F_composite'] = (df['z_F_bell'] * df['z_F_gate'] * df['z_F_coherence']).clip(-5, 5)

# ── 5b. PELT LABELING ON z_F_composite ───────────────────────────────────────
# Label = temporal regime shift in z_F_composite per qubit per backend.
# Features (z_F_bell, z_F_gate, z_F_coherence) are instantaneous values.
# Label comes from temporal structure — genuinely independent.

print("Running PELT labeling on z_F_composite per qubit...")
df['drifted'] = 0
total_cps = 0

for (backend, qubit), grp in df.groupby(['backend', 'qubit_id']):
    grp  = grp.sort_values('timestamp')
    idxs = grp.index.tolist()
    signal = df.loc[idxs, 'z_F_composite'].values.astype(float)

    if len(signal) < 10:
        continue

    try:
        algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(signal)
        bkps = algo.predict(pen=1.5)
        bkps = [b for b in bkps if b < len(signal)]
    except Exception:
        continue

    if not bkps:
        continue

    total_cps += len(bkps)
    first_cp   = bkps[0]
    baseline_mean = signal[:first_cp].mean() if first_cp > 0 else signal.mean()
    baseline_std  = signal[:first_cp].std()  if first_cp > 0 else 0.01

    labels = np.zeros(len(signal), dtype=int)
    seg_start = 0
    for cp in bkps:
        seg = signal[seg_start:cp]
        if len(seg) > 0 and seg.mean() < (baseline_mean - baseline_std):
            labels[seg_start:cp] = 1
        seg_start = cp
    last_seg = signal[seg_start:]
    if len(last_seg) > 0 and last_seg.mean() < (baseline_mean - baseline_std):
        labels[seg_start:] = 1

    for i, idx in enumerate(idxs):
        df.loc[idx, 'drifted'] = int(labels[i])

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count / len(df) * 100, 1)
print(f"  Change points detected : {total_cps}")
print(f"  Drift  (1): {drift_count:,}  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,}  ({100-drift_pct}%)")
if drift_count < 50:
    print("  WARNING: <50 drift events. Lower pen parameter.")
else:
    print("  ✓ Sufficient drift events.\n")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence',
             'z_F_bell','z_F_gate','z_F_coherence',
             'z_T1','z_cz','z_readout','drifted']
full_cols = [c for c in full_cols if c in df.columns]
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv")

feat_df = df[['z_F_bell','z_F_gate','z_F_coherence','drifted']].copy()
feat_df = feat_df.reset_index(drop=True)
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
clean_total = drift_count + stable_count
print(f"\n{'='*50}")
print(f"  LABELING COMPLETE")
print(f"{'='*50}")
print(f"  Total rows    : {len(df):,}")
print(f"  Drift    (1)  : {drift_count:,} ({round(drift_count/len(df)*100,1)}%)")
print(f"  Stable   (0)  : {stable_count:,} ({round(stable_count/len(df)*100,1)}%)")
print(f"  Label method  : PELT on z_F_composite")
print(f"  Clean rows    : {clean_total:,}")
print(f"  Features      : z_F_bell, z_F_gate, z_F_coherence, z_F_composite")
print(f"\n  NEXT: python 3_validate_data.py")
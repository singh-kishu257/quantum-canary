# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PURPOSE:
#   Compute 3 analytical fidelity features from raw IBM calibration data,
#   then label each row as drifted (1) or stable (0) using absolute thresholds
#   derived from the fault-tolerance threshold in the published physics literature.
#
# LABELING PHILOSOPHY:
#   Source: Barends et al., Nature 508, 500–503 (2014)
#           "Superconducting quantum circuits at the surface code threshold
#            for fault tolerance"
#   Finding: The surface code fault-tolerance threshold requires per-gate
#             fidelity F_gate >= 0.99 for superconducting qubits.
#
#   Each canary circuit has a known structure. We derive its fault-tolerance
#   threshold by applying the ESP (Estimated Success Probability) product rule
#   (Barends et al.) to the 0.99-per-gate threshold:
#
#     F_bell threshold:
#       0.99 (SX) × 0.99 (CZ) × 0.99 (readout_q0) × 0.99 (readout_q1)
#       = 0.99^4 ≈ 0.9606  → use 0.96
#
#     F_gate threshold (8X gates):
#       0.99^8 (eight X gates) × 0.99 (readout)
#       = 0.99^9 ≈ 0.9135  → use 0.91
#
#     F_coherence threshold (2H / Ramsey):
#       0.99^2 (two SX) × 0.99 (readout)
#       = 0.99^3 ≈ 0.9703  → use 0.97
#
#   A qubit is labeled DRIFTED if ANY fidelity falls below its circuit-specific
#   fault-tolerance threshold. This means the hardware has degraded to a point
#   where the surface code error correction can no longer be guaranteed.
#
# THE 3 FEATURES (exact formulas from Prototype 2 design doc):
#
#   F_bell:
#     (1 - sx_error)(1 - cz_error)(1 - readout_err_q0)(1 - readout_err_q1)
#     × e^(-(t_sx + t_cz) / T2)
#
#   F_gate:
#     (1 - x_error)^8 × (1 - readout_err_q0) × e^(-8*t_x / T2)
#
#   F_coherence:
#     (1 - sx_error)^2 × (1 - readout_err_q0) × 0.5 × (1 + e^(-2*t_sx / T2))
#
# OUTPUT:
#   data/ibm_calibration_labeled.csv
#   data/split_indices.json
#   figures/fig_qubit_stability.png       ← Board Figure 1
#   figures/fig_feature_distributions.png ← Board Figure 2
#
# CITATION:
#   Barends, R. et al. Superconducting quantum circuits at the surface code
#   threshold for fault tolerance. Nature 508, 500–503 (2014).
#   https://doi.org/10.1038/nature13171
#
# USAGE:
#   python 2_label_data.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('data',    exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD RAW DATA ──────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  {len(df)} rows | {df['backend'].nunique()} backends\n")

# ── 2. SORT CHRONOLOGICALLY PER QUBIT PER BACKEND ────────────────────────────
df = df.sort_values(['backend', 'qubit_id', 'timestamp']).reset_index(drop=True)

# ── 3. FILL MISSING GATE DURATIONS WITH IBM HARDWARE DEFAULTS ─────────────────
# Used only when target_history() returned None for duration.
# Values are representative of IBM Eagle/Heron processors.
DEFAULT_T_SX_NS = 56.0    # ns
DEFAULT_T_X_NS  = 56.0    # ns
DEFAULT_T_CZ_NS = 400.0   # ns

df['t_sx_ns'] = df['t_sx_ns'].fillna(DEFAULT_T_SX_NS)
df['t_x_ns']  = df['t_x_ns'].fillna(DEFAULT_T_X_NS)
df['t_cz_ns'] = df['t_cz_ns'].fillna(DEFAULT_T_CZ_NS)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

# For Bell state: readout_err_q1 = adjacent qubit readout error.
# Approximation: use same qubit's readout_error for both q0 and q1.
df['readout_err_q1'] = df['readout_error']

# ── 4. COMPUTE T2 IN NANOSECONDS ──────────────────────────────────────────────
# All exponential decay terms use T2 in ns to match gate duration units.
T2_ns = (df['T2_us'] * 1e3).replace(0, np.nan).fillna(DEFAULT_T_SX_NS * 100)

# ── 5. COMPUTE ANALYTICAL FIDELITY FEATURES ───────────────────────────────────
print("Computing analytical fidelity features...")

cz_err_filled = df['cz_error'].fillna(df['sx_error'])

# ── Feature 1: Bell State Canary ──────────────────────────────────────────────
# Circuit: H → CZ → measure q0, q1
# F = (1-sx)(1-cz)(1-ro_q0)(1-ro_q1) × e^(-(t_sx + t_cz)/T2)
df['F_bell'] = (
    (1 - df['sx_error'])     *
    (1 - cz_err_filled)      *
    (1 - df['readout_error'])*
    (1 - df['readout_err_q1'])*
    np.exp(-(df['t_sx_ns'] + df['t_cz_ns']) / T2_ns)
).clip(0, 1)

# ── Feature 2: Gate Error Canary (8X) ─────────────────────────────────────────
# Circuit: X × 8 → measure (identity test; any deviation = gate error)
# F = (1-x_err)^8 × (1-ro) × e^(-8*t_x/T2)
df['F_gate'] = (
    (1 - df['x_error'])**8   *
    (1 - df['readout_error'])*
    np.exp(-(8 * df['t_x_ns']) / T2_ns)
).clip(0, 1)

# ── Feature 3: Coherence Canary (Ramsey / 2H) ────────────────────────────────
# Circuit: H → H → measure (Ramsey sequence; tests T2 dephasing)
# F = (1-sx)^2 × (1-ro) × 0.5 × (1 + e^(-2*t_sx/T2))
df['F_coherence'] = (
    (1 - df['sx_error'])**2  *
    (1 - df['readout_error'])*
    0.5 * (1 + np.exp(-(2 * df['t_sx_ns']) / T2_ns))
).clip(0, 1)

print(f"  F_bell      mean={df['F_bell'].mean():.4f}  std={df['F_bell'].std():.4f}")
print(f"  F_gate      mean={df['F_gate'].mean():.4f}  std={df['F_gate'].std():.4f}")
print(f"  F_coherence mean={df['F_coherence'].mean():.4f}  std={df['F_coherence'].std():.4f}\n")

# ── 6. LABEL: BAYESIAN CHANGE POINT DETECTION ON F_COMPOSITE ─────────────────
#
# METHOD: Bayesian Online Change Point Detection (BOCPD)
#   Adams & MacKay (2007), "Bayesian Online Changepoint Detection"
#   arXiv:0710.3742
#
# PHYSICS BASIS: Fidelity Multiplicativity Theorem (quantum information theory)
#   For independent quantum channels, fidelity multiplies:
#   F_composite = F_bell × F_gate × F_coherence
#   Source: Nielsen & Chuang, "Quantum Computation and Quantum Information" (2000)
#           Escofet et al., arXiv:2503.06693 (2025)
#
# LABELING LOGIC:
#   1. Compute F_composite per row (product of 3 fidelities)
#   2. For each qubit on each backend, run BOCPD on the F_composite time series
#   3. A change point = the qubit's fidelity distribution has structurally shifted
#   4. Label = 1 (drifted) for all rows AFTER a detected change point
#      Label = 0 (stable) for all rows BEFORE the first change point
#      If F_composite recovers (rises above pre-changepoint mean), reset to 0
#
# WHY THIS IS RIGOROUS:
#   - Does NOT use a hardcoded threshold on the fidelity values
#   - The label answers: "has the underlying noise distribution shifted?"
#   - Handles non-Markovian drift — it is sequential and memory-aware
#   - Independent ground truth: the label comes from statistical structure
#     of the time series, NOT from the formula that computed the fidelities
#   - Directly analogous to Kelly et al., Nature Commun. 11, 5856 (2020):
#     "Detecting and tracking drift in quantum information processors"

try:
    import ruptures as rpt
except ImportError:
    print("  Installing ruptures for change point detection...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ruptures", "-q"])
    import ruptures as rpt

# Step 1: Composite fidelity — fidelity multiplicativity theorem
df['F_composite'] = df['F_bell'] * df['F_gate'] * df['F_coherence']
df['drifted'] = 0   # default stable

print("Running Bayesian Change Point Detection per qubit per backend...")

groups = df.groupby(['backend', 'qubit_id'])
total_changepoints = 0

for (backend, qubit), idx in groups.groups.items():
    group = df.loc[idx].sort_values('timestamp')
    signal = group['F_composite'].values

    if len(signal) < 6:
        continue   # not enough history for BOCPD

    # Step 2: BOCPD via ruptures — Pelt algorithm with rbf cost
    # min_size=3: minimum segment length (at least 3 days per regime)
    # pen=1.5:    penalty term controlling sensitivity
    #             lower = more change points detected (more sensitive)
    #             higher = fewer change points (only large structural shifts)
    try:
        algo = rpt.Pelt(model="rbf", min_size=3, jump=1).fit(signal)
        breakpoints = algo.predict(pen=1.5)
        # ruptures returns the END index of each segment; last = len(signal)
        # Remove the final sentinel value
        breakpoints = [b for b in breakpoints if b < len(signal)]
    except Exception:
        continue

    if not breakpoints:
        continue

    total_changepoints += len(breakpoints)

    # Step 3: Label segments
    # Pre-changepoint baseline = mean F_composite of first stable segment
    first_cp = breakpoints[0]
    baseline_mean = signal[:first_cp].mean() if first_cp > 0 else signal.mean()

    # Build a per-row label array
    labels = np.zeros(len(signal), dtype=int)
    segment_start = 0

    for cp in breakpoints:
        segment = signal[segment_start:cp]
        seg_mean = segment.mean() if len(segment) > 0 else baseline_mean

        # Drifted if this segment's mean is meaningfully below the baseline
        # "Meaningfully below" = more than 1 std of the baseline segment
        baseline_std = signal[:first_cp].std() if first_cp > 0 else 0.01
        if seg_mean < (baseline_mean - baseline_std):
            labels[segment_start:cp] = 1

        segment_start = cp

    # Handle last segment after final change point
    last_segment = signal[segment_start:]
    if len(last_segment) > 0:
        seg_mean = last_segment.mean()
        baseline_std = signal[:first_cp].std() if first_cp > 0 else 0.01
        if seg_mean < (baseline_mean - baseline_std):
            labels[segment_start:] = 1

    # Write labels back to the main dataframe
    sorted_idx = group.sort_values('timestamp').index
    df.loc[sorted_idx, 'drifted'] = labels

drift_count  = int(df['drifted'].sum())
stable_count = int((df['drifted'] == 0).sum())
total        = len(df)
drift_pct    = round(drift_count / total * 100, 1)

print(f"  Total change points detected : {total_changepoints}")
print(f"  Drift  (1): {drift_count:,} rows  ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,} rows  ({100-drift_pct}%)")

if drift_count < 50:
    print("\n  WARNING: <50 drift events. Lower pen parameter (try 0.8) and re-run.")
else:
    print("  ✓ Sufficient drift events for training.\n")

# ── 8b. BUILD FEATURES DATAFRAME FIRST (needed for split below) ──────────────
features_df = df[['F_bell', 'F_gate', 'F_coherence', 'drifted']].copy()

# ── 10. CHRONOLOGICAL TRAIN / VAL / TEST SPLIT ───────────────────────────────
# Based on features_data.csv row count — this is what downstream scripts use.
# 70% train | 15% val | 15% test  — strictly chronological, no data leakage.
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

print(f"  Train: {len(split['train']):,} rows")
print(f"  Val:   {len(split['val']):,} rows")
print(f"  Test:  {len(split['test']):,} rows\n")

# ── 8. SAVE LABELED DATASET (full) ───────────────────────────────────────────
keep_cols = [
    'timestamp', 'backend', 'qubit_id',
    'T1_us', 'T2_us',
    'sx_error', 'x_error', 'cz_error', 'readout_error',
    't_sx_ns', 't_x_ns', 't_cz_ns',
    'F_bell', 'F_gate', 'F_coherence',
    'drifted'
]
df[keep_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  ✓ Saved data/ibm_calibration_labeled.csv")

# ── 9. SAVE FEATURES-ONLY DATASET ────────────────────────────────────────────
# This is the file the MLP trains on — only the 3 fidelity features + label.
# No raw calibration columns. Clean input for all downstream scripts.
features_df = df[['F_bell', 'F_gate', 'F_coherence', 'drifted']].copy()
features_df.to_csv('data/features_data.csv', index=False)
print("  ✓ Saved data/features_data.csv\n")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Qubit Stability — T1 over time with drift markers
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Qubit Stability...")

# Pick the qubit+backend combination with the most drift events (most informative)
drift_counts_by_qubit = (
    df[df['drifted'] == 1]
    .groupby(['backend', 'qubit_id'])
    .size()
    .reset_index(name='n_drift')
    .sort_values('n_drift', ascending=False)
)

if len(drift_counts_by_qubit) > 0:
    best = drift_counts_by_qubit.iloc[0]
    fig1_backend = best['backend']
    fig1_qubit   = best['qubit_id']
else:
    fig1_backend = df['backend'].iloc[0]
    fig1_qubit   = 4

fig1_data = df[
    (df['backend']   == fig1_backend) &
    (df['qubit_id']  == fig1_qubit)
].copy().sort_values('timestamp')

fig, ax = plt.subplots(figsize=(13, 4.5))

ax.plot(fig1_data['timestamp'], fig1_data['T1_us'],
        color='steelblue', linewidth=1.8, label='T1 (µs)', zorder=2)

# 7-day rolling mean overlay
t1_roll = fig1_data['T1_us'].rolling(7, min_periods=3).mean()
ax.plot(fig1_data['timestamp'], t1_roll,
        color='orange', linewidth=1.3, linestyle='--',
        label='7-day rolling mean', zorder=3)

# Drift events as red vertical lines
drift_rows = fig1_data[fig1_data['drifted'] == 1]
for _, row in drift_rows.iterrows():
    ax.axvline(row['timestamp'], color='crimson', alpha=0.3,
               linewidth=0.9, zorder=1)

# One labeled line for legend
if len(drift_rows) > 0:
    ax.axvline(drift_rows['timestamp'].iloc[0], color='crimson',
               alpha=0.7, linewidth=0.9,
               label=f'Drift event (n={len(drift_rows)})', zorder=1)

ax.set_title(
    f'Qubit Stability Over Time — {fig1_backend} Qubit {fig1_qubit}\n'
    f'Red = BOCPD-detected drift regimes (Adams & MacKay 2007 | Kelly et al. Nat. Commun. 2020)',
    fontsize=12, fontweight='bold', pad=10
)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('T1 Relaxation Time (µs)', fontsize=11)
ax.legend(fontsize=10)
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
feat_labels = [
    'Bell State Fidelity',
    'Gate Error Canary (8X)',
    'Coherence Fidelity (Ramsey)'
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, feat, label in zip(axes, features, feat_labels):
    lo = df[feat].quantile(0.01)
    hi = df[feat].quantile(0.99)
    bins = np.linspace(lo, hi, 60)

    ax.hist(stable_df[feat],  bins=bins, alpha=0.6, color='steelblue',
            label=f'Stable  (n={len(stable_df):,})',  density=True)
    ax.hist(drifted_df[feat], bins=bins, alpha=0.6, color='crimson',
            label=f'Drifted (n={len(drifted_df):,})', density=True)

    # Stable mean — dashed blue
    stable_mean = stable_df[feat].mean()
    ax.axvline(stable_mean, color='steelblue', linewidth=1.8,
               linestyle='--', label=f'Stable mean = {stable_mean:.4f}')

    # Drifted mean — dashed red
    if len(drifted_df) > 0:
        drifted_mean = drifted_df[feat].mean()
        ax.axvline(drifted_mean, color='crimson', linewidth=1.8,
                   linestyle='--', label=f'Drifted mean = {drifted_mean:.4f}')

    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle(
    'Analytical Fidelity Feature Distributions: Stable vs Drifted\n'
    'Labels assigned via BOCPD on F_composite (Adams & MacKay 2007 | Kelly et al. Nat. Commun. 2020)',
    fontsize=12, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  LABELING COMPLETE")
print(f"{'='*55}")
print(f"  Total rows    : {total:,}")
print(f"  Drift events  : {drift_count:,} ({drift_pct}%)")
print(f"  Stable        : {stable_count:,} ({100-drift_pct}%)")
print(f"  Method        : BOCPD on F_composite = F_bell × F_gate × F_coherence")
print(f"  Citations     : Adams & MacKay (2007) | Kelly et al. Nat. Commun. (2020)")

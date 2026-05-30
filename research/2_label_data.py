# 2_label_data.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS-INFORMED HYBRID DATASET CONSTRUCTION
#
# Stable class  (label=0): sim stable   + IBM extreme stable anchors
# Drifted class (label=1): sim drifted  + IBM extreme drifted anchors
#
# Sim data: 15,000 rows (5k stable, 5k borderline, 5k drifted) from 1e_pull_data.py
# IBM anchors: up to 3,000 stable + 3,000 drifted from all_backends_raw.csv
# Total: ~21,000 rows
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ── 1. LOAD AND SORT ──────────────────────────────────────────────────────────
print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv(DATA_DIR / 'all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend', 'qubit_id', 'timestamp']).reset_index(drop=True)

# ── 2. FILL MISSING GATE DURATIONS ───────────────────────────────────────────
df['t_sx_ns'] = df['t_sx_ns'].fillna(30.0)
df['t_x_ns']  = df['t_x_ns'].fillna(30.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(75.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])
df.loc[df['cz_error'] > 0.5, 'cz_error'] = np.nan

# ── 3. 30-DAY ROLLING BASELINE PER QUBIT ─────────────────────────────────────
print("Computing 30-day rolling baselines...")
groupby_obj = df.groupby(['backend', 'qubit_id'])

df['T1_roll_med'] = groupby_obj['T1_us'].transform(
    lambda x: x.rolling(window=30, min_periods=14).median().shift(1)
)
df['CZ_roll_med'] = groupby_obj['cz_error'].transform(
    lambda x: x.rolling(window=30, min_periods=14).median().shift(1)
)
df['RO_roll_med'] = groupby_obj['readout_error'].transform(
    lambda x: x.rolling(window=30, min_periods=14).median().shift(1)
)

df = df.dropna(subset=['T1_roll_med', 'RO_roll_med']).reset_index(drop=True)

# ── 4. PERCENTAGE DEVIATIONS ──────────────────────────────────────────────────
EPS = 1e-8
df['pct_T1_drop'] = ((df['T1_roll_med'] - df['T1_us']) / (df['T1_roll_med'] + EPS)).clip(-2, 2)
df['pct_T1_dev']  = ((df['T1_us'] - df['T1_roll_med']).abs() / (df['T1_roll_med'] + EPS)).clip(0, 2)
cz_v = df['cz_error'].notna() & df['CZ_roll_med'].notna()
df['pct_CZ_rise'] = np.nan
df.loc[cz_v, 'pct_CZ_rise'] = ((df.loc[cz_v, 'cz_error'] - df.loc[cz_v, 'CZ_roll_med']) / (df.loc[cz_v, 'CZ_roll_med'] + EPS)).clip(-1, 10)
df['pct_RO_rise'] = ((df['readout_error'] - df['RO_roll_med']) / (df['RO_roll_med'] + EPS)).clip(-1, 10)

# ── 5. CLASSIFY IBM ZONES ────────────────────────────────────────────────────
def classify_zone(row):
    if row['pct_T1_drop'] >= 0.35 or (not pd.isna(row['pct_CZ_rise']) and row['pct_CZ_rise'] >= 1.00) or row['pct_RO_rise'] >= 0.50:
        return 'extreme_drifted'
    if row['pct_T1_dev'] <= 0.08 and row['pct_RO_rise'] <= 0.20 and (pd.isna(row['pct_CZ_rise']) or row['pct_CZ_rise'] <= 0.20):
        return 'extreme_stable'
    return 'grey'

df['zone'] = df.apply(classify_zone, axis=1)

# ── 6. ANALYTICAL FIDELITY PROXY FORMULAS ────────────────────────────────────
T1_ns = df['T1_us'] * 1e3
T2_ns = (df['T2_us'] * 1e3).fillna(T1_ns * 0.9).clip(upper=2.0 * T1_ns)
cz    = df['cz_error'].fillna(df['sx_error'])

t_bell = df['t_sx_ns'] + df['t_cz_ns']
df['F_bell'] = ((1-df['sx_error']) * (1-cz) * (1-df['readout_error'])**2 * np.exp(-t_bell/T1_ns) * np.exp(-t_bell/T2_ns)).clip(0,1)

t_coh = 20.0 * df['t_sx_ns']
df['F_coherence'] = ((1-df['sx_error'])**20 * (1-df['readout_error']) * 0.5 * (1 + np.exp(-t_coh/T1_ns) * np.exp(-t_coh/T2_ns))).clip(0,1)

t_gate = 20.0 * df['t_x_ns']
df['F_gate'] = ((1-df['x_error'])**20 * (1-df['readout_error']) * np.exp(-t_gate/T1_ns) * np.exp(-t_gate/T2_ns)).clip(0,1)

# ── 7. EXTRACT ANCHORS ────────────────────────────────────────────────────────
N_ANCHORS = 3000

ibm_stable_pool  = df[df['zone']=='extreme_stable']
n_s              = min(N_ANCHORS, len(ibm_stable_pool))
ibm_stable_sel   = ibm_stable_pool.sort_values('pct_T1_dev').head(n_s).copy()
ibm_stable_sel['drifted'] = 0

ibm_drifted_pool = df[df['zone']=='extreme_drifted']
n_d              = min(N_ANCHORS, len(ibm_drifted_pool))
ibm_drifted_sel  = ibm_drifted_pool.sort_values('pct_T1_drop', ascending=False).head(n_d).copy()
ibm_drifted_sel['drifted'] = 1

# ── 8. LOAD SIMULATION DATA ───────────────────────────────────────────────────
sim   = pd.read_csv(DATA_DIR / 'sim_data.csv')
s_sim = sim[sim['drifted']==0].copy()
d_sim = sim[sim['drifted']==1].copy()

# ── 9. COMBINE (HYBRID DATASET) ───────────────────────────────────────────────
ibm_feat = ['F_bell', 'F_coherence', 'F_gate', 'drifted']
combined = pd.concat([
    s_sim[ibm_feat],
    ibm_stable_sel[ibm_feat],
    d_sim[ibm_feat],
    ibm_drifted_sel[ibm_feat]
], ignore_index=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# ── 10. SAVE ──────────────────────────────────────────────────────────────────
combined.to_csv(DATA_DIR / 'features_data.csv', index=False)
df.to_csv(DATA_DIR / 'ibm_calibration_labeled.csv', index=False)

train_idx, temp_idx = train_test_split(
    np.arange(len(combined)), test_size=0.3,
    stratify=combined['drifted'], random_state=42
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.5,
    stratify=combined.iloc[temp_idx]['drifted'], random_state=42
)
with open(DATA_DIR / 'split_indices.json', 'w') as f:
    json.dump({"train": train_idx.tolist(),
               "val":   val_idx.tolist(),
               "test":  test_idx.tolist()}, f)

# ── 11. FIGURES ───────────────────────────────────────────────────────────────
print("Generating figures...")

# Merge sim + IBM anchors into unified stable/drifted pools
all_stable  = pd.concat([s_sim[ibm_feat],       ibm_stable_sel[ibm_feat]],  ignore_index=True)
all_drifted = pd.concat([d_sim[ibm_feat],        ibm_drifted_sel[ibm_feat]], ignore_index=True)
all_stable  = all_stable[all_stable['drifted']  == 0]
all_drifted = all_drifted[all_drifted['drifted'] == 1]

n_stable  = len(all_stable)
n_drifted = len(all_drifted)
n_total   = len(combined)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
feats = ['F_bell', 'F_coherence', 'F_gate']
flabs = ['Bell State Fidelity', 'Coherence Fidelity', 'Gate Error Fidelity']

for ax, feat, lab in zip(axes, feats, flabs):
    bins = np.linspace(combined[feat].quantile(0.01),
                       combined[feat].quantile(0.99), 55)

    # Stable — light blue filled
    ax.hist(all_stable[feat],  bins=bins, alpha=0.5, color='steelblue',
            density=True, label=f'Stable (n={n_stable:,})')

    # Drifted — light red filled
    ax.hist(all_drifted[feat], bins=bins, alpha=0.5, color='crimson',
            density=True, label=f'Drifted (n={n_drifted:,})')

    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9)
    ax.set_ylabel('Density',  fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

n_sim_total = len(sim)
n_ibm_total = n_s + n_d
plt.suptitle(
    f'Training Data Fidelity Distributions\n'
    f'Hybrid Dataset (n={n_total:,}) · {n_sim_total:,} Sim Samples + {n_ibm_total:,} IBM Anchors '
    f'· Overlap regions shown',
    fontsize=10, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_feature_distributions.png")

print(f"\nLabeling Complete.")
print(f"  Stable  : {len(s_sim):,} Sim + {len(ibm_stable_sel):,} IBM = {n_stable:,}")
print(f"  Drifted : {len(d_sim):,} Sim + {len(ibm_drifted_sel):,} IBM = {n_drifted:,}")
print(f"  Total   : {n_total:,}")
print(f"\n  NEXT: python 3_validate_data.py")
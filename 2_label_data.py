# 2_label_data.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS-INFORMED HYBRID DATASET CONSTRUCTION
#
# DATASET DESIGN RATIONALE:
#
#   Stable class  (label=0): 6,000 sim stable + 3,000 IBM extreme stable
#   Drifted class (label=1): 6,000 sim drifted + 3,000 IBM extreme drifted
#
#   Total Dataset: 18,000 rows.
#
#   STRATEGY: 
#     We keep the original 6k/6k simulation samples from 1e_pull_data.py.
#     We then add 3k IBM snapshots for each class to ground the model.
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

# ── 7. EXTRACT ANCHORS (3,000 STABLE, 3,000 DRIFTED) ──────────────────────────
N_ANCHORS = 3000

# Get IBM Stable
ibm_stable_pool = df[df['zone']=='extreme_stable']
n_s = min(N_ANCHORS, len(ibm_stable_pool))
ibm_stable_sel = ibm_stable_pool.sort_values('pct_T1_dev').head(n_s).copy()
ibm_stable_sel['drifted'] = 0

# Get IBM Drifted
ibm_drifted_pool = df[df['zone']=='extreme_drifted']
n_d = min(N_ANCHORS, len(ibm_drifted_pool))
ibm_drifted_sel = ibm_drifted_pool.sort_values('pct_T1_drop', ascending=False).head(n_d).copy()
ibm_drifted_sel['drifted'] = 1

# ── 8. LOAD SIMULATION DATA (6,000 EACH - AS GENERATED) ──────────────────────
sim = pd.read_csv('data/sim_data.csv')
s_sim = sim[sim['drifted']==0].copy()
d_sim = sim[sim['drifted']==1].copy()

# ── 9. COMBINE (HYBRID DATASET) ───────────────────────────────────────────────
ibm_feat = ['F_bell', 'F_coherence', 'F_gate', 'drifted']
combined = pd.concat([
    s_sim[ibm_feat],             # 6,000 sim stable
    ibm_stable_sel[ibm_feat],    # 3,000 IBM stable
    d_sim[ibm_feat],             # 6,000 sim drifted
    ibm_drifted_sel[ibm_feat]    # 3,000 IBM drifted
], ignore_index=True)

combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# ── 10. SAVE ──────────────────────────────────────────────────────────────────
combined.to_csv('data/features_data.csv', index=False)
df.to_csv('data/ibm_calibration_labeled.csv', index=False)

train_idx, temp_idx = train_test_split(np.arange(len(combined)), test_size=0.3, stratify=combined['drifted'], random_state=42)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.5, stratify=combined.iloc[temp_idx]['drifted'], random_state=42)
with open('data/split_indices.json', 'w') as f:
    json.dump({"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}, f)

# ── 11. FIGURES ───────────────────────────────────────────────────────────────
print("Generating figures...")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
feats = ['F_bell', 'F_coherence', 'F_gate']
flabs = ['Bell state fidelity', 'Coherence fidelity', 'Gate error fidelity']

for ax, feat, lab in zip(axes, feats, flabs):
    bins = np.linspace(combined[feat].quantile(0.01), combined[feat].quantile(0.99), 55)
    
    ax.hist(s_sim[feat], bins=bins, alpha=0.4, color='steelblue', label='Sim stable', density=True)
    ax.hist(d_sim[feat], bins=bins, alpha=0.4, color='crimson', label='Sim drifted', density=True)
    
    ax.hist(ibm_stable_sel[feat], bins=bins, color='navy', label=f'IBM stable (n={len(ibm_stable_sel)})', 
            density=True, histtype='step', linewidth=1.5)
    ax.hist(ibm_drifted_sel[feat], bins=bins, color='darkred', label=f'IBM drifted (n={len(ibm_drifted_sel)})', 
            density=True, histtype='step', linewidth=1.5, linestyle='--')

    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.set_xlabel('Fidelity', fontsize=9); ax.set_ylabel('Density', fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.suptitle('Training data fidelity distributions\nHybrid Dataset (n=18,000) · 6k Sim Samples + 6k IBM Anchors', fontsize=10, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nLabeling Complete.")
print(f"  Stable  : {len(s_sim)} Sim + {len(ibm_stable_sel)} IBM")
print(f"  Drifted : {len(d_sim)} Sim + {len(ibm_drifted_sel)} IBM")
print(f"  Total   : {len(combined):,}")
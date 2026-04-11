# 2_label_data.py — Quantum Canary Prototype 2
# Labels: T1 drop >=7% OR readout rise >=5% OR CZ doubles vs 7-day rolling mean
# Features: z_F_bell, z_F_gate, z_F_coherence (rolling z-scores)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, os

os.makedirs('data', exist_ok=True)
os.makedirs('figures', exist_ok=True)

print("Loading data/all_backends_raw.csv ...")
df = pd.read_csv('data/all_backends_raw.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['backend','qubit_id','timestamp']).reset_index(drop=True)
print(f"  {len(df):,} rows | {df['backend'].nunique()} backends\n")

df['t_sx_ns'] = df['t_sx_ns'].fillna(56.0)
df['t_x_ns']  = df['t_x_ns'].fillna(56.0)
df['t_cz_ns'] = df['t_cz_ns'].fillna(400.0)
df['x_error'] = df['x_error'].fillna(df['sx_error'])

T_WAIT_NS  = 1000.0
ROLL       = 7
MIN_P      = 3
EPS        = 1e-8
THR_T1     = 0.07
THR_RO     = 0.05
THR_CZ     = 1.00  # 100% rise = doubles

print("Processing per qubit...")
all_rows = []

for (backend, qubit), grp in df.groupby(['backend','qubit_id']):
    grp = grp.copy().sort_values('timestamp').reset_index(drop=True)

    T2_ns = (grp['T2_us'] * 1e3).replace(0, np.nan).fillna(5600.0)
    T1_ns = grp['T1_us'].replace(0, np.nan).fillna(grp['T1_us'].median()) * 1e3
    cz    = grp['cz_error'].fillna(grp['sx_error'])

    # Rolling baselines for labeling
    grp['T1_roll'] = grp['T1_us'].rolling(ROLL, min_periods=MIN_P).mean()
    grp['cz_roll'] = grp['cz_error'].rolling(ROLL, min_periods=MIN_P).mean()
    grp['ro_roll'] = grp['readout_error'].rolling(ROLL, min_periods=MIN_P).mean()

    # Fidelity features (Prototype 2 image formulas)
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

    # Rolling z-scores of fidelities
    for feat in ['F_bell','F_gate','F_coherence']:
        rm = grp[feat].rolling(ROLL, min_periods=MIN_P).mean()
        rs = grp[feat].rolling(ROLL, min_periods=MIN_P).std()
        grp[f'z_{feat}'] = ((grp[feat] - rm) / (rs + EPS)).clip(-5, 5).fillna(0)

    # Labels: T1 drop, readout rise, CZ rise vs rolling mean
    grp['drifted'] = 0
    for pos in range(len(grp)):
        t1r = grp.loc[pos, 'T1_roll']
        czr = grp.loc[pos, 'cz_roll']
        ror = grp.loc[pos, 'ro_roll']
        if pd.isna(t1r) or pd.isna(czr) or pd.isna(ror): continue
        t1v = grp.loc[pos, 'T1_us']
        rov = grp.loc[pos, 'readout_error']
        czv = grp.loc[pos, 'cz_error'] if not pd.isna(grp.loc[pos,'cz_error']) else czr
        t1_drop = (t1r - t1v) / t1r if t1r > 0 else 0
        cz_rise = (czv - czr) / czr if czr > 0 else 0
        ro_rise = (rov - ror) / ror if ror > 0 else 0
        if t1_drop >= THR_T1 or cz_rise >= THR_CZ or ro_rise >= THR_RO:
            grp.loc[pos, 'drifted'] = 1

    all_rows.append(grp)

df = pd.concat(all_rows, ignore_index=True)

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count/len(df)*100, 1)
print(f"  Drift  (1): {drift_count:,} ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,} ({100-drift_pct}%)\n")

# SAVE
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence',
             'z_F_bell','z_F_gate','z_F_coherence','drifted']
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  Saved data/ibm_calibration_labeled.csv")

feat_df = df[['z_F_bell','z_F_gate','z_F_coherence','drifted']].copy()
feat_df.to_csv('data/features_data.csv', index=False)
print(f"  Saved data/features_data.csv ({len(feat_df):,} rows)")

from sklearn.model_selection import train_test_split

n = len(feat_df)
indices = np.arange(n)
y_all   = feat_df['drifted'].values

# Stratified split — same drift rate in train/val/test
train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, random_state=42, stratify=y_all)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=y_all[temp_idx])

split = {"train": train_idx.tolist(),
         "val":   val_idx.tolist(),
         "test":  test_idx.tolist()}
with open('data/split_indices.json','w') as f: json.dump(split,f)

print(f"  Train:{len(train_idx):,} drift={round(y_all[train_idx].mean()*100,1)}%")
print(f"  Val:  {len(val_idx):,} drift={round(y_all[val_idx].mean()*100,1)}%")
print(f"  Test: {len(test_idx):,} drift={round(y_all[test_idx].mean()*100,1)}%\n")

# FIGURE 1
dc = df[df['drifted']==1].groupby(['backend','qubit_id']).size()
b1,q1 = dc.idxmax() if len(dc)>0 else (df['backend'].iloc[0],4)
fd = df[(df['backend']==b1)&(df['qubit_id']==q1)].sort_values('timestamp')

fig,ax = plt.subplots(figsize=(7,3.8))
ax.plot(fd['timestamp'],fd['T1_us'],color='steelblue',lw=1.5,label='T1 (us)')
ax.plot(fd['timestamp'],fd['T1_roll'],color='orange',lw=1.3,ls='--',label='7-day rolling mean')
for d in fd[fd['drifted']==1]['timestamp']:
    ax.axvline(d,color='crimson',alpha=0.25,lw=0.8)
if fd['drifted'].sum()>0:
    ax.axvline(fd[fd['drifted']==1]['timestamp'].iloc[0],color='crimson',alpha=0.7,lw=0.8,
               label=f"Drift (n={fd['drifted'].sum()})")
ax.set_title(f'Qubit Stability — {b1} Qubit {q1}',fontsize=9,fontweight='bold')
ax.set_xlabel('Date',fontsize=9); ax.set_ylabel('T1 (us)',fontsize=9)
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png',dpi=300,bbox_inches='tight')
plt.close()

# FIGURE 2
s_df=df[df['drifted']==0]; d_df=df[df['drifted']==1]
feats=['z_F_bell','z_F_gate','z_F_coherence']
flabs=['Bell State Z-Score','Gate Error Z-Score','Coherence Z-Score']
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,feat,lab in zip(axes,feats,flabs):
    bins=np.linspace(-3,3,55)
    ax.hist(s_df[feat],bins=bins,alpha=0.6,color='steelblue',
            label=f'Stable (n={len(s_df):,})',density=True)
    ax.hist(d_df[feat],bins=bins,alpha=0.6,color='crimson',
            label=f'Drifted (n={len(d_df):,})',density=True)
    ax.axvline(s_df[feat].mean(),color='steelblue',lw=1.8,ls='--',
               label=f'Stable mean={s_df[feat].mean():.3f}')
    ax.axvline(d_df[feat].mean(),color='crimson',lw=1.8,ls='--',
               label=f'Drifted mean={d_df[feat].mean():.3f}')
    ax.set_title(lab,fontsize=10,fontweight='bold')
    ax.set_xlabel('Z-Score',fontsize=9); ax.set_ylabel('Density',fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle('Fidelity Z-Score Distributions: Stable vs Drifted\nLabels from T1/CZ/Readout rolling baseline',
             fontsize=10,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png',dpi=300,bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print(f"  LABELING COMPLETE")
print(f"  Drift:{drift_count:,} ({drift_pct}%)  Stable:{stable_count:,}")
print(f"  Features: z_F_bell, z_F_gate, z_F_coherence")
print(f"\n  NEXT: python 3_validate_data.py")
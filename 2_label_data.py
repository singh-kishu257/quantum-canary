# 2_label_data.py — Quantum Canary Prototype 2
# Labels via GMM on F_bell, F_gate, F_coherence
# Features: F_bell, F_gate, F_coherence (raw fidelity 0-1)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
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

T_WAIT_NS = 1000.0

print("Computing fidelity features per qubit...")
all_rows = []

for (backend, qubit), grp in df.groupby(['backend','qubit_id']):
    grp = grp.copy().sort_values('timestamp').reset_index(drop=True)

    T2_ns = (grp['T2_us'] * 1e3).replace(0, np.nan).fillna(5600.0)
    T1_ns = grp['T1_us'].replace(0, np.nan).fillna(grp['T1_us'].median()) * 1e3
    cz    = grp['cz_error'].fillna(grp['sx_error'])

    # Bell State Canary: H -> CZ -> measure q0, q1
    grp['F_bell'] = (
        (1 - grp['sx_error']) * (1 - cz) *
        (1 - grp['readout_error'])**2 *
        np.exp(-(grp['t_sx_ns'] + grp['t_cz_ns']) / T1_ns) *
        np.exp(-(grp['t_sx_ns'] + grp['t_cz_ns']) / T2_ns)
    ).clip(1e-9, 1 - 1e-9)

    # Gate Error Canary: X x8 -> measure
    grp['F_gate'] = (
        (1 - grp['x_error'])**8 *
        (1 - grp['readout_error']) *
        np.exp(-(8 * grp['t_x_ns']) / T1_ns) *
        np.exp(-(8 * grp['t_x_ns']) / T2_ns)
    ).clip(1e-9, 1 - 1e-9)

    # Coherence Canary: H -> wait(1000ns) -> H -> measure
    grp['F_coherence'] = (
        (1 - grp['sx_error'])**2 *
        (1 - grp['readout_error']) *
        0.5 * (1 + np.exp(-T_WAIT_NS / T2_ns))
    ).clip(1e-9, 1 - 1e-9)

    all_rows.append(grp)

df = pd.concat(all_rows, ignore_index=True)
print(f"  F_bell      mean={df['F_bell'].mean():.4f}  std={df['F_bell'].std():.4f}")
print(f"  F_gate      mean={df['F_gate'].mean():.4f}  std={df['F_gate'].std():.4f}")
print(f"  F_coherence mean={df['F_coherence'].mean():.4f}  std={df['F_coherence'].std():.4f}\n")

# GMM LABELING
print("Fitting GMM (2 components) on F_bell, F_gate, F_coherence...")
X_gmm = df[['F_bell','F_gate','F_coherence']].values
valid  = ~(np.isnan(X_gmm).any(axis=1) | np.isinf(X_gmm).any(axis=1))

gmm = GaussianMixture(n_components=2, covariance_type='full',
                      random_state=42, max_iter=200, n_init=5)
gmm.fit(X_gmm[valid])

means = gmm.means_.mean(axis=1)
drift_comp = int(np.argmin(means))
print(f"  Component 0 mean fidelity: {means[0]:.4f}")
print(f"  Component 1 mean fidelity: {means[1]:.4f}")
print(f"  Drift component: {drift_comp}")

labels = np.zeros(len(df), dtype=int)
gmm_pred = gmm.predict(X_gmm[valid])
labels[valid] = (gmm_pred == drift_comp).astype(int)
df['drifted'] = labels

drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count/len(df)*100, 1)
print(f"\n  Drift  (1): {drift_count:,} ({drift_pct}%)")
print(f"  Stable (0): {stable_count:,} ({100-drift_pct}%)\n")

# SAVE
full_cols = ['timestamp','backend','qubit_id','T1_us','T2_us',
             'sx_error','x_error','cz_error','readout_error',
             't_sx_ns','t_x_ns','t_cz_ns',
             'F_bell','F_gate','F_coherence','drifted']
df[full_cols].to_csv('data/ibm_calibration_labeled.csv', index=False)
print("  Saved data/ibm_calibration_labeled.csv")

feat_df = df[['F_bell','F_gate','F_coherence','drifted']].copy()
feat_df.to_csv('data/features_data.csv', index=False)
print(f"  Saved data/features_data.csv ({len(feat_df):,} rows)")

n=len(feat_df); i_val=int(n*0.70); i_test=int(n*0.85)
split={"train":list(range(0,i_val)),"val":list(range(i_val,i_test)),"test":list(range(i_test,n))}
with open('data/split_indices.json','w') as f: json.dump(split,f)
print(f"  Train:{i_val:,} Val:{i_test-i_val:,} Test:{n-i_test:,}\n")

# FIGURE 1
print("Generating Figure 1...")
dc = df[df['drifted']==1].groupby(['backend','qubit_id']).size()
b1,q1 = dc.idxmax() if len(dc)>0 else (df['backend'].iloc[0],4)
fd = df[(df['backend']==b1)&(df['qubit_id']==q1)].sort_values('timestamp')

fig,ax = plt.subplots(figsize=(7,3.8))
ax.plot(fd['timestamp'],fd['F_bell'],color='steelblue',lw=1.3,label='F_bell')
ax.plot(fd['timestamp'],fd['F_gate'],color='darkorange',lw=1.3,label='F_gate')
ax.plot(fd['timestamp'],fd['F_coherence'],color='green',lw=1.3,label='F_coherence')
for d in fd[fd['drifted']==1]['timestamp']:
    ax.axvline(d,color='crimson',alpha=0.2,lw=0.8)
ax.set_title(f'Fidelity Over Time — {b1} Qubit {q1}\nRed = GMM-labeled drift',fontsize=9,fontweight='bold')
ax.set_xlabel('Date',fontsize=9); ax.set_ylabel('Fidelity (0-1)',fontsize=9)
ax.set_ylim(0,1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_qubit_stability.png',dpi=300,bbox_inches='tight')
plt.close()
print("  Saved figures/fig_qubit_stability.png")

# FIGURE 2
print("Generating Figure 2...")
s_df=df[df['drifted']==0]; d_df=df[df['drifted']==1]
feats=['F_bell','F_gate','F_coherence']
flabs=['Bell State Fidelity\n(H -> CZ -> measure)',
       'Gate Error Fidelity\n(X x8 -> measure)',
       'Coherence Fidelity\n(H -> wait -> H -> measure)']

fig,axes=plt.subplots(1,3,figsize=(14,4.5))
for ax,feat,lab in zip(axes,feats,flabs):
    bins=np.linspace(0,1,60)
    ax.hist(s_df[feat],bins=bins,alpha=0.6,color='steelblue',
            label=f'Stable (n={len(s_df):,})',density=False)
    ax.hist(d_df[feat],bins=bins,alpha=0.6,color='crimson',
            label=f'Drifted (n={len(d_df):,})',density=False)
    ax.axvline(s_df[feat].mean(),color='steelblue',lw=2,ls='--',
               label=f'Stable mean={s_df[feat].mean():.4f}')
    ax.axvline(d_df[feat].mean(),color='crimson',lw=2,ls='--',
               label=f'Drifted mean={d_df[feat].mean():.4f}')
    ax.set_title(lab,fontsize=10,fontweight='bold')
    ax.set_xlabel('Fidelity',fontsize=9); ax.set_ylabel('Count',fontsize=9)
    ax.set_xlim(0,1); ax.legend(fontsize=7); ax.grid(alpha=0.3)
plt.suptitle('Analytical Fidelity Distributions: Stable vs Drifted\nLabels from GMM 2-component clustering',
             fontsize=10,fontweight='bold',y=1.02)
plt.tight_layout()
plt.savefig('figures/fig_feature_distributions.png',dpi=300,bbox_inches='tight')
plt.close()
print("  Saved figures/fig_feature_distributions.png")

print(f"\n{'='*50}")
print(f"  LABELING COMPLETE")
print(f"  Total: {len(df):,}  Drift: {drift_count:,} ({drift_pct}%)  Stable: {stable_count:,}")
print(f"  Features: F_bell, F_gate, F_coherence")
print(f"  Method: GMM 2-component maximum likelihood")
print(f"\n  NEXT: python 3_validate_data.py")



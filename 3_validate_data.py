# 3_validate_data.py — Quantum Canary Prototype 2
# Validates the labeled dataset before training.
# Checks class balance, feature statistics, and separation.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data/features_data.csv ...")
df = pd.read_csv('data/features_data.csv')
print(f"  {len(df):,} rows\n")

FEATURES = [
    'z_F_bell', 'z_F_gate', 'z_F_coherence',
    'delta_F_bell', 'delta_F_gate', 'delta_F_coherence',
    'momentum_F_bell', 'momentum_F_gate', 'momentum_F_coherence',
    'rel_t1_drop', 'rel_cz_rise', 'rel_ro_rise',
    'log_t1_us', 'log_t2_us',
    'sx_error_clipped', 'cz_error_clipped', 'ro_error_clipped'
]

# ── 2. CLASS BALANCE ──────────────────────────────────────────────────────────
drift_count  = int(df['drifted'].sum())
stable_count = len(df) - drift_count
drift_pct    = round(drift_count / len(df) * 100, 1)

print("── Class Balance ──")
print(f"  Stable (0): {stable_count:,}  ({100-drift_pct}%)")
print(f"  Drift  (1): {drift_count:,}  ({drift_pct}%)")
if drift_pct < 10:
    print("  NOTE: Imbalanced — class_weight='balanced' will be used in training.")
else:
    print("  ✓ Class balance acceptable.\n")

# ── 3. FEATURE STATISTICS ─────────────────────────────────────────────────────
print("── Feature Statistics ──")
print(df[FEATURES].describe().round(6).to_string())

print("\n── Feature Means: Stable vs Drifted ──")
print(df.groupby('drifted')[FEATURES].mean().round(6).to_string())

print("\n── Feature Std: Stable vs Drifted ──")
print(df.groupby('drifted')[FEATURES].std().round(6).to_string())

# ── 4. MISSING VALUES ─────────────────────────────────────────────────────────
print("\n── Missing Values ──")
missing = df[FEATURES].isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values.")
else:
    print(missing)

# ── 5. RANGE CHECK ────────────────────────────────────────────────────────────
print("\n── Range Check (feature-family specific bounds) ──")
for f in FEATURES:
    lo = df[f].min()
    hi = df[f].max()
    if f.startswith('z_'):
        ok = "✓" if lo >= -5 and hi <= 5 else "WARNING"
    elif f.startswith('rel_'):
        ok = "✓" if lo >= -1 and hi <= 3 else "WARNING"
    elif f.startswith('log_'):
        ok = "✓" if lo >= 0 else "WARNING"
    elif f.endswith('_clipped'):
        ok = "✓" if lo >= 0 and hi <= 1 else "WARNING"
    else:
        ok = "✓" if lo >= -1 and hi <= 1 else "WARNING"
    print(f"  {f}: [{lo:.6f}, {hi:.6f}]  {ok}")

# ── 6. SPLIT VERIFICATION ─────────────────────────────────────────────────────
print("\n── Split Verification ──")
with open('data/split_indices.json') as f:
    split = json.load(f)

for name, idx in split.items():
    subset     = df.iloc[idx]
    d_pct      = round(subset['drifted'].mean() * 100, 1)
    print(f"  {name:5s}: {len(idx):,} rows | drift={d_pct}%")

# ── 7. FIGURE: CORRELATION MATRIX ────────────────────────────────────────────
print("\nGenerating Figure: Feature Correlation Matrix...")
corr_cols = FEATURES + ['drifted']
corr = df[corr_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
labels = corr_cols
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}',
                ha='center', va='center', fontsize=7,
                color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black')
ax.set_title('Feature Correlation Matrix', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_correlation_matrix.png")

print(f"\n{'='*45}")
print(f"  VALIDATION COMPLETE")
print(f"{'='*45}")
print(f"  NEXT: python 4_baselines.py")

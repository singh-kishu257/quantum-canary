# 4c_hotelling_baseline.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# HOTELLING'S T² BASELINE
#
# The multivariate generalization of the univariate t-test / CUSUM approach.
# Rigorously answers: "How far, statistically, is this measurement from the
# healthy qubit distribution across all three fidelity features jointly?"
#
# WHY T²:
#   CUSUM and the simple threshold collapse the three canary features into
#   a scalar (mean fidelity), discarding information about how the features
#   correlate. Hotelling's T² preserves the full covariance structure —
#   it's the correct multivariate statistical process control method.
#
# HOW IT WORKS:
#   1. Estimate the stable class's mean vector μ and covariance matrix Σ
#      from training data.
#   2. For each test sample x, compute Mahalanobis-like distance:
#         T²(x) = (x - μ)ᵀ · Σ⁻¹ · (x - μ)
#   3. Large T² means the point is far from the stable distribution in
#      covariance-adjusted space → drift.
#
# This is a direct three-way comparison point for the MLP: both methods
# operate on the same three features, but T² is linear / Gaussian-assumption
# based, while the MLP learns nonlinear decision boundaries.
#
# REFERENCE:
#   Hotelling, H. (1947). Multivariate Quality Control. In Techniques of
#   Statistical Analysis. McGraw-Hill.
#
# OUTPUT:
#   models/hotelling_params.json   — μ, Σ⁻¹, tuned decision threshold
#   figures/fig_hotelling.png      — T² diagnostic figure
#
# RUNTIME: ~3 seconds
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score,
                              accuracy_score, roc_curve)

os.makedirs('models',  exist_ok=True)
os.makedirs('figures', exist_ok=True)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  HOTELLING'S T² BASELINE — QUANTUM CANARY 2")
print("=" * 60)

df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

X = df[FEATURES].values
y = df['drifted'].values

X_tr,   y_tr   = X[split['train']], y[split['train']]
X_val,  y_val  = X[split['val']],   y[split['val']]
X_test, y_test = X[split['test']],  y[split['test']]

print(f"\n  Train : {len(X_tr):,} samples")
print(f"  Val   : {len(X_val):,} samples")
print(f"  Test  : {len(X_test):,} samples")

# ── FIT STABLE DISTRIBUTION ───────────────────────────────────────────────────
# Estimate mean vector and covariance matrix from the STABLE class only.
# This defines the "healthy manifold" in 3D fidelity space.

stable_train = X_tr[y_tr == 0]
mu_vec       = stable_train.mean(axis=0)
cov_matrix   = np.cov(stable_train, rowvar=False)

# Regularize the covariance matrix slightly to avoid singular inversion
# (happens when features are highly correlated, e.g. F_gate ~ F_coherence).
epsilon      = 1e-6
cov_reg      = cov_matrix + epsilon * np.eye(cov_matrix.shape[0])
cov_inv      = np.linalg.inv(cov_reg)

print(f"\n  Healthy mean vector μ:")
for name, val in zip(FEATURES, mu_vec):
    print(f"    {name}: {val:.5f}")

print(f"\n  Healthy covariance Σ:")
for i, row in enumerate(cov_matrix):
    row_str = "    "
    for val in row:
        row_str += f"{val:+.2e}  "
    print(row_str)

# ── T² SCORE FUNCTION ─────────────────────────────────────────────────────────
def hotelling_t2(X, mu, cov_inv):
    """Compute Hotelling's T² statistic for each row of X."""
    diff = X - mu
    # T² = (x-μ)ᵀ · Σ⁻¹ · (x-μ)
    return np.einsum('ij,jk,ik->i', diff, cov_inv, diff)

# ── VALIDATE + TUNE DECISION THRESHOLD ────────────────────────────────────────
val_scores = hotelling_t2(X_val, mu_vec, cov_inv)
val_auc    = roc_auc_score(y_val, val_scores)

print(f"\n  Validation AUC: {val_auc:.4f}")

# Grid search decision threshold on validation F1
best_f1, h_opt = -1.0, None
for h in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores > h).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, h_opt = f1, float(h)

print(f"  Best threshold h: {h_opt:.4f}  (validation F1 = {best_f1:.4f})")

# ── TEST EVALUATION ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST SET EVALUATION")
print("=" * 60)

test_scores = hotelling_t2(X_test, mu_vec, cov_inv)
test_preds  = (test_scores > h_opt).astype(int)

test_auc = roc_auc_score(y_test, test_scores)
test_f1  = f1_score(y_test, test_preds, zero_division=0)
test_acc = accuracy_score(y_test, test_preds)

print(f"\n  Hotelling T² Test AUC : {test_auc:.4f}")
print(f"  Hotelling T² Test F1  : {test_f1:.4f}")
print(f"  Hotelling T² Test Acc : {test_acc:.4f}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
hotelling_params = {
    'mu_vec':       mu_vec.tolist(),
    'cov_matrix':   cov_matrix.tolist(),
    'cov_inv':      cov_inv.tolist(),
    'h':            h_opt,
    'val_auc':      float(val_auc),
    'val_f1':       float(best_f1),
    'test_auc':     float(test_auc),
    'test_f1':      float(test_f1),
    'test_acc':     float(test_acc),
    'epsilon':      epsilon,
    'reference':    'Hotelling, H. (1947). Multivariate Quality Control.',
}
with open('models/hotelling_params.json', 'w') as f:
    json.dump(hotelling_params, f, indent=2)

print(f"\n  ✓ Saved: models/hotelling_params.json")

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

# Panel 1: T² scores across test set, color by true label
ax1 = axes[0]
colors_test = ['#3fb950' if label == 0 else '#e63946' for label in y_test]
ax1.scatter(range(len(test_scores)), test_scores,
            c=colors_test, s=6, alpha=0.5)
ax1.axhline(h_opt, color='#1c5d8c', lw=1.5, linestyle='--',
            label=f'Decision threshold h = {h_opt:.3f}')
ax1.set_ylabel('Hotelling T² statistic', fontsize=11)
ax1.set_xlabel('Test sample index', fontsize=11)
ax1.set_title(f"T² scores — Test AUC = {test_auc:.4f}",
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(alpha=0.2)
ax1.set_yscale('log')

# Panel 2: ROC curve
ax2 = axes[1]
fpr, tpr, _ = roc_curve(y_test, test_scores)
ax2.plot(fpr, tpr, color='#c1272d', lw=2.2,
         label=f'Hotelling T² (AUC = {test_auc:.3f})')
ax2.plot([0, 1], [0, 1], color='#888', lw=1, linestyle='--',
         label='Random (AUC = 0.5)')
ax2.set_xlabel('False positive rate', fontsize=11)
ax2.set_ylabel('True positive rate', fontsize=11)
ax2.set_title('Hotelling T² ROC curve',
              fontsize=11, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('figures/fig_hotelling.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: figures/fig_hotelling.png")

print("\n" + "=" * 60)
print("  HOTELLING'S T² BASELINE COMPLETE")
print("=" * 60)
print(f"\n  NEXT: update 6_evaluate.py for four-way comparison")
print(f"        (MLP vs CUSUM vs Hotelling T² vs Threshold)")
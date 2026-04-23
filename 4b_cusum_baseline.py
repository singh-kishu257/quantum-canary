# 4b_cusum_baseline.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# CUSUM (Cumulative Sum) control chart baseline.
#
# WHY CUSUM:
#   CUSUM is the standard statistical process control method for detecting
#   drift in time series data. It exploits temporal correlation by tracking
#   the cumulative deviation from a healthy baseline. Unlike the naive
#   threshold classifier, CUSUM is sensitive to small sustained shifts that
#   never cross a simple cutoff.
#
# REFERENCE:
#   Page, E. S. (1954). Continuous Inspection Schemes. Biometrika, 41(1/2),
#   100-115. https://doi.org/10.2307/2333009
#
# HOW IT WORKS:
#   At each time step, we compute the deviation of the mean fidelity from
#   a healthy target value μ₀. We maintain a cumulative sum of downward
#   deviations (since drift = lower fidelity):
#
#     S_t = max(0, S_{t-1} + (μ₀ - x_t) - k)
#
#   Where:
#     x_t  = observed mean fidelity at time t
#     μ₀   = target healthy fidelity (estimated from training stable class)
#     k    = allowance parameter (small tolerance for normal noise)
#     h    = decision threshold — alarm if S_t > h
#
#   The MLP ensemble score provides a "continuous probability" which we
#   approximate for CUSUM by using S_t / h as its own drift score.
#
# OUTPUT:
#   models/cusum_params.json  — tuned k, h, μ₀ values
#   figures/fig_cusum.png     — CUSUM trace on validation data
#
# RUNTIME: ~5 seconds
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
print("  CUSUM BASELINE — QUANTUM CANARY 2")
print("=" * 60)

df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

X     = df[FEATURES].values
y     = df['drifted'].values
mean  = X.mean(axis=1)     # CUSUM operates on scalar — use mean fidelity

X_tr,   y_tr   = mean[split['train']], y[split['train']]
X_val,  y_val  = mean[split['val']],   y[split['val']]
X_test, y_test = mean[split['test']],  y[split['test']]

print(f"\n  Train : {len(X_tr):,} samples")
print(f"  Val   : {len(X_val):,} samples")
print(f"  Test  : {len(X_test):,} samples")

# ── ESTIMATE HEALTHY BASELINE μ₀ ──────────────────────────────────────────────
# Use the mean + std of STABLE class in training data
stable_train = X_tr[y_tr == 0]
mu_0    = float(np.mean(stable_train))
sigma_0 = float(np.std(stable_train))

print(f"\n  Healthy baseline μ₀: {mu_0:.5f}")
print(f"  Healthy std     σ₀: {sigma_0:.5f}")

# ── CUSUM SCORE FUNCTION ──────────────────────────────────────────────────────
def cusum_scores(values, mu_0, k):
    """
    Compute one-sided downward CUSUM scores for a sequence of values.
    Returns raw cumulative sums S_t (before thresholding).

    Since drift manifests as lower fidelity, we track negative deviations.
    """
    S = np.zeros(len(values))
    S[0] = max(0.0, mu_0 - values[0] - k)
    for t in range(1, len(values)):
        S[t] = max(0.0, S[t-1] + (mu_0 - values[t]) - k)
    return S

# ── TUNE k AND h ON VALIDATION SET ────────────────────────────────────────────
# Standard CUSUM heuristic: k = σ/2 to detect shifts of size σ.
# We grid-search over a physically reasonable range to maximise val AUC.

print("\n  Grid-searching CUSUM parameters on validation set...")

best = {'auc': -1, 'k': None, 'h': None}
k_grid = np.linspace(0.1 * sigma_0, 3.0 * sigma_0, 30)

# For AUC, we don't need h — CUSUM scores themselves are the ranking.
# But we do need k.
for k in k_grid:
    scores = cusum_scores(X_val, mu_0, k)
    # Skip if all scores are zero (k too large)
    if scores.max() == 0:
        continue
    try:
        auc = roc_auc_score(y_val, scores)
    except ValueError:
        continue
    if auc > best['auc']:
        best['auc'] = auc
        best['k']   = float(k)

k_opt = best['k']
print(f"  Best k = {k_opt:.5f}  (validation AUC = {best['auc']:.4f})")

# Tune h for the best F1 at the operating point
val_scores = cusum_scores(X_val, mu_0, k_opt)
best_f1, h_opt = -1.0, None
for h in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores > h).astype(int)
    if len(np.unique(preds)) < 2:
        continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, h_opt = f1, float(h)

print(f"  Best h = {h_opt:.5f}  (validation F1 = {best_f1:.4f})")

# ── EVALUATE ON TEST SET ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TEST SET EVALUATION")
print("=" * 60)

test_scores = cusum_scores(X_test, mu_0, k_opt)
test_preds  = (test_scores > h_opt).astype(int)

# Handle edge case where CUSUM produces no positive scores
if test_scores.max() > 0:
    test_auc = roc_auc_score(y_test, test_scores)
else:
    test_auc = 0.5
    print("  ⚠ Warning: all CUSUM scores are zero on test set")

test_acc = accuracy_score(y_test, test_preds)
test_f1  = f1_score(y_test, test_preds, zero_division=0)

print(f"\n  CUSUM Test AUC      : {test_auc:.4f}")
print(f"  CUSUM Test F1       : {test_f1:.4f}")
print(f"  CUSUM Test Accuracy : {test_acc:.4f}")

# ── SAVE PARAMETERS ───────────────────────────────────────────────────────────
cusum_params = {
    'mu_0':      mu_0,
    'sigma_0':   sigma_0,
    'k':         k_opt,
    'h':         h_opt,
    'test_auc':  float(test_auc),
    'test_f1':   float(test_f1),
    'test_acc':  float(test_acc),
    'val_auc':   float(best['auc']),
    'val_f1':    float(best_f1),
    'reference': 'Page, E.S. (1954) Biometrika 41(1/2), 100-115',
}
with open('models/cusum_params.json', 'w') as f:
    json.dump(cusum_params, f, indent=2)

print(f"\n  ✓ Saved: models/cusum_params.json")

# ── VISUALIZATION ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 8),
                         gridspec_kw={'height_ratios': [1.4, 1]})
fig.patch.set_facecolor('white')

# Panel 1: raw mean-fidelity on test set
ax1 = axes[0]
colors_test = ['#3fb950' if label == 0 else '#e63946' for label in y_test]
ax1.scatter(range(len(X_test)), X_test, c=colors_test, s=8, alpha=0.5)
ax1.axhline(mu_0, color='#1c5d8c', lw=1.5, linestyle='--',
            label=f'Healthy baseline μ₀ = {mu_0:.4f}')
ax1.set_ylabel('Mean fidelity', fontsize=11)
ax1.set_title('Test Set — mean fidelity (green=stable, red=drifted)',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=9, loc='lower left')
ax1.grid(alpha=0.2)

# Panel 2: CUSUM score
ax2 = axes[1]
ax2.plot(range(len(test_scores)), test_scores, color='#c1272d', lw=1.2)
ax2.fill_between(range(len(test_scores)), test_scores,
                 alpha=0.18, color='#c1272d')
ax2.axhline(h_opt, color='#1c5d8c', lw=1.5, linestyle='--',
            label=f'Decision threshold h = {h_opt:.4f}')
ax2.set_ylabel('CUSUM statistic $S_t$', fontsize=11)
ax2.set_xlabel('Sample index (test set)', fontsize=11)
ax2.set_title(
    f'CUSUM statistic (k = {k_opt:.4f})  ·  Test AUC = {test_auc:.4f}  ·  '
    f'F1 = {test_f1:.4f}',
    fontsize=11, fontweight='bold'
)
ax2.legend(fontsize=9, loc='upper left')
ax2.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('figures/fig_cusum.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✓ Saved: figures/fig_cusum.png")

print("\n" + "=" * 60)
print("  CUSUM BASELINE COMPLETE")
print("=" * 60)
print(f"\n  NEXT: re-run 6_evaluate.py for the three-way comparison.")
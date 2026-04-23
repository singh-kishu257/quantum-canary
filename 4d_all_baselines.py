# 4d_all_baselines.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# COMPREHENSIVE STATISTICAL BASELINE COMPARISON
#
# Evaluates all non-ML drift detection methods considered for this project
# on the same held-out test data, providing a complete historical survey
# of statistical process control approaches from 1924 to present.
#
# METHODS EVALUATED:
#   1. Naive mean-fidelity threshold       — baseline reference
#   2. Shewhart 3-sigma control chart      — Shewhart (1924)
#   3. CUSUM cumulative sum chart          — Page (1954)
#   4. Hotelling's T² multivariate test    — Hotelling (1947)
#   5. Mahalanobis distance (MCD)          — Rousseeuw (1984)
#
# Each method is tuned on validation data to maximize F1 score at its
# operating point, then evaluated on the held-out test split.
#
# OUTPUT:
#   models/all_baselines_results.json  — all AUCs, F1s, accuracies, params
#   figures/fig_all_baselines.png      — side-by-side ROC + bar chart
#
# RUNTIME: ~15 seconds
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, f1_score,
                              accuracy_score, roc_curve)
from sklearn.covariance import MinCovDet

os.makedirs('models',  exist_ok=True)
os.makedirs('figures', exist_ok=True)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  COMPREHENSIVE STATISTICAL BASELINE COMPARISON")
print("=" * 70)

df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

X = df[FEATURES].values
y = df['drifted'].values

X_tr,   y_tr   = X[split['train']], y[split['train']]
X_val,  y_val  = X[split['val']],   y[split['val']]
X_test, y_test = X[split['test']],  y[split['test']]

X_tr_mean   = X_tr.mean(axis=1)
X_val_mean  = X_val.mean(axis=1)
X_test_mean = X_test.mean(axis=1)

stable_train = X_tr[y_tr == 0]

print(f"\n  Train: {len(X_tr):,}   Val: {len(X_val):,}   Test: {len(X_test):,}")
print(f"  Stable training samples for fitting healthy distributions: "
      f"{len(stable_train):,}\n")

results = {}

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 1: NAIVE MEAN-FIDELITY THRESHOLD
# ═════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("  METHOD 1 — Naive Mean-Fidelity Threshold")
print("─" * 70)

# Tune threshold on validation F1. Drift score = 1 / mean_fidelity (so higher
# means more drift) for consistent AUC computation.
best_f1, t_opt = -1.0, None
for t in np.linspace(X_val_mean.min(), X_val_mean.max(), 500):
    preds = (X_val_mean < t).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, t_opt = f1, float(t)

# For AUC we use -mean_fidelity as the score (higher = more drift)
thresh_test_scores = -X_test_mean
thresh_test_preds  = (X_test_mean < t_opt).astype(int)

thresh_auc = roc_auc_score(y_test, thresh_test_scores)
thresh_f1  = f1_score(y_test, thresh_test_preds, zero_division=0)
thresh_acc = accuracy_score(y_test, thresh_test_preds)

print(f"  Threshold: mean fidelity < {t_opt:.4f}")
print(f"  Test AUC: {thresh_auc:.4f}   F1: {thresh_f1:.4f}   Acc: {thresh_acc:.4f}")

results['threshold'] = {
    'method':  'Naive mean-fidelity threshold',
    'reference': 'Reference baseline',
    'params':  {'cutoff': t_opt},
    'test_auc': float(thresh_auc),
    'test_f1':  float(thresh_f1),
    'test_acc': float(thresh_acc),
    'fpr_tpr':  [a.tolist() for a in roc_curve(y_test, thresh_test_scores)[:2]],
}

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 2: SHEWHART 3-SIGMA CONTROL CHART
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  METHOD 2 — Shewhart 3-Sigma Control Chart  [Shewhart 1924]")
print("─" * 70)

# Per-feature mean and std from stable training distribution
mu_per_feat    = stable_train.mean(axis=0)
sigma_per_feat = stable_train.std(axis=0)

# Score = max number of sigmas below mean across any feature
# (a qubit is drifted if ANY feature falls far below its healthy mean)
def shewhart_score(X):
    deviations = (mu_per_feat - X) / sigma_per_feat   # positive = below mean
    return deviations.max(axis=1)                     # worst-offending feature

val_scores = shewhart_score(X_val)

# Tune k-sigma threshold (typically 3.0, but we let validation pick)
best_f1, k_opt = -1.0, None
for k_sigma in np.linspace(0.5, 6.0, 200):
    preds = (val_scores > k_sigma).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, k_opt = f1, float(k_sigma)

test_scores = shewhart_score(X_test)
test_preds  = (test_scores > k_opt).astype(int)

shewhart_auc = roc_auc_score(y_test, test_scores)
shewhart_f1  = f1_score(y_test, test_preds, zero_division=0)
shewhart_acc = accuracy_score(y_test, test_preds)

print(f"  Tuned sigma threshold: {k_opt:.3f}")
print(f"  Per-feature μ: {mu_per_feat.round(4)}")
print(f"  Per-feature σ: {sigma_per_feat.round(4)}")
print(f"  Test AUC: {shewhart_auc:.4f}   F1: {shewhart_f1:.4f}   Acc: {shewhart_acc:.4f}")

results['shewhart'] = {
    'method': 'Shewhart 3-sigma control chart',
    'reference': 'Shewhart, W. A. (1924/1931). Economic Control of Quality of '
                 'Manufactured Product.',
    'params': {'k_sigma': k_opt,
               'mu_per_feat':    mu_per_feat.tolist(),
               'sigma_per_feat': sigma_per_feat.tolist()},
    'test_auc': float(shewhart_auc),
    'test_f1':  float(shewhart_f1),
    'test_acc': float(shewhart_acc),
    'fpr_tpr':  [a.tolist() for a in roc_curve(y_test, test_scores)[:2]],
}

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 3: CUSUM CUMULATIVE SUM CHART
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  METHOD 3 — CUSUM Control Chart  [Page 1954]")
print("─" * 70)

stable_train_mean = X_tr_mean[y_tr == 0]
mu_cusum    = float(stable_train_mean.mean())
sigma_cusum = float(stable_train_mean.std())

def cusum_scores(values, mu_0, k):
    S = np.zeros(len(values))
    S[0] = max(0.0, mu_0 - values[0] - k)
    for t in range(1, len(values)):
        S[t] = max(0.0, S[t-1] + (mu_0 - values[t]) - k)
    return S

# Grid-search k on validation AUC
best_auc, k_opt_cusum = -1.0, None
for k in np.linspace(0.1 * sigma_cusum, 3.0 * sigma_cusum, 30):
    scores = cusum_scores(X_val_mean, mu_cusum, k)
    if scores.max() == 0: continue
    try:
        auc = roc_auc_score(y_val, scores)
    except ValueError: continue
    if auc > best_auc:
        best_auc, k_opt_cusum = auc, float(k)

val_scores = cusum_scores(X_val_mean, mu_cusum, k_opt_cusum)
best_f1, h_opt_cusum = -1.0, None
for h in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores > h).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, h_opt_cusum = f1, float(h)

test_scores_cusum = cusum_scores(X_test_mean, mu_cusum, k_opt_cusum)
test_preds        = (test_scores_cusum > h_opt_cusum).astype(int)

cusum_auc = roc_auc_score(y_test, test_scores_cusum) \
            if test_scores_cusum.max() > 0 else 0.5
cusum_f1  = f1_score(y_test, test_preds, zero_division=0)
cusum_acc = accuracy_score(y_test, test_preds)

print(f"  Tuned k: {k_opt_cusum:.4f}   h: {h_opt_cusum:.4f}")
print(f"  Test AUC: {cusum_auc:.4f}   F1: {cusum_f1:.4f}   Acc: {cusum_acc:.4f}")

results['cusum'] = {
    'method': 'CUSUM cumulative sum control chart',
    'reference': 'Page, E. S. (1954). Continuous Inspection Schemes. '
                 'Biometrika 41(1/2), 100-115.',
    'params': {'mu_0': mu_cusum, 'sigma_0': sigma_cusum,
               'k': k_opt_cusum, 'h': h_opt_cusum},
    'test_auc': float(cusum_auc),
    'test_f1':  float(cusum_f1),
    'test_acc': float(cusum_acc),
    'fpr_tpr':  [a.tolist() for a in roc_curve(y_test, test_scores_cusum)[:2]],
}

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 4: HOTELLING'S T² MULTIVARIATE STATISTICAL TEST
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  METHOD 4 — Hotelling's T²  [Hotelling 1947]")
print("─" * 70)

mu_vec       = stable_train.mean(axis=0)
cov_matrix   = np.cov(stable_train, rowvar=False)
cov_reg      = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])
cov_inv      = np.linalg.inv(cov_reg)

def hotelling_t2(X, mu, inv_cov):
    d = X - mu
    return np.einsum('ij,jk,ik->i', d, inv_cov, d)

val_scores = hotelling_t2(X_val, mu_vec, cov_inv)
best_f1, h_opt_t2 = -1.0, None
for h in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores > h).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, h_opt_t2 = f1, float(h)

test_scores_t2 = hotelling_t2(X_test, mu_vec, cov_inv)
test_preds     = (test_scores_t2 > h_opt_t2).astype(int)

t2_auc = roc_auc_score(y_test, test_scores_t2)
t2_f1  = f1_score(y_test, test_preds, zero_division=0)
t2_acc = accuracy_score(y_test, test_preds)

print(f"  Tuned h: {h_opt_t2:.4f}")
print(f"  Test AUC: {t2_auc:.4f}   F1: {t2_f1:.4f}   Acc: {t2_acc:.4f}")

results['hotelling'] = {
    'method': "Hotelling's T² multivariate statistical test",
    'reference': 'Hotelling, H. (1947). Multivariate Quality Control. '
                 'In Techniques of Statistical Analysis, McGraw-Hill.',
    'params': {'mu': mu_vec.tolist(),
               'cov_matrix': cov_matrix.tolist(),
               'h': h_opt_t2},
    'test_auc': float(t2_auc),
    'test_f1':  float(t2_f1),
    'test_acc': float(t2_acc),
    'fpr_tpr':  [a.tolist() for a in roc_curve(y_test, test_scores_t2)[:2]],
}

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 5: ROBUST MAHALANOBIS DISTANCE (MCD)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("  METHOD 5 — Robust Mahalanobis Distance (MCD)  [Rousseeuw 1984]")
print("─" * 70)

try:
    mcd = MinCovDet(support_fraction=0.75, random_state=42)
    mcd.fit(stable_train)
    mcd_mu  = mcd.location_
    mcd_inv = np.linalg.inv(
        mcd.covariance_ + 1e-6 * np.eye(mcd.covariance_.shape[0])
    )

    def mahalanobis_robust(X):
        d = X - mcd_mu
        return np.einsum('ij,jk,ik->i', d, mcd_inv, d)

    val_scores = mahalanobis_robust(X_val)
    best_f1, h_opt_mcd = -1.0, None
    for h in np.linspace(val_scores.min(), val_scores.max(), 500):
        preds = (val_scores > h).astype(int)
        if len(np.unique(preds)) < 2: continue
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, h_opt_mcd = f1, float(h)

    test_scores_mcd = mahalanobis_robust(X_test)
    test_preds      = (test_scores_mcd > h_opt_mcd).astype(int)

    mcd_auc = roc_auc_score(y_test, test_scores_mcd)
    mcd_f1  = f1_score(y_test, test_preds, zero_division=0)
    mcd_acc = accuracy_score(y_test, test_preds)

    print(f"  MCD support fraction: 0.75")
    print(f"  Test AUC: {mcd_auc:.4f}   F1: {mcd_f1:.4f}   Acc: {mcd_acc:.4f}")

    results['mahalanobis_mcd'] = {
        'method': 'Robust Mahalanobis distance (Minimum Covariance Determinant)',
        'reference': 'Rousseeuw, P. J. (1984). Least Median of Squares Regression. '
                     'Journal of the American Statistical Association, 79(388), 871-880.',
        'params': {'mu': mcd_mu.tolist(),
                   'h': h_opt_mcd,
                   'support_fraction': 0.75},
        'test_auc': float(mcd_auc),
        'test_f1':  float(mcd_f1),
        'test_acc': float(mcd_acc),
        'fpr_tpr':  [a.tolist() for a in roc_curve(y_test, test_scores_mcd)[:2]],
    }
except Exception as e:
    print(f"  MCD failed: {e}")
    results['mahalanobis_mcd'] = {'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with open('models/all_baselines_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 70)
print("  SUMMARY — FIVE NON-ML STATISTICAL METHODS ON TEST SET")
print("=" * 70)
print(f"  {'Method':45s}  {'AUC':>7s}  {'F1':>7s}  {'Acc':>7s}")
print("  " + "─" * 68)

order = ['threshold', 'shewhart', 'cusum', 'hotelling', 'mahalanobis_mcd']
for key in order:
    if 'error' in results.get(key, {}): continue
    r = results[key]
    name = r['method']
    if len(name) > 43: name = name[:40] + '...'
    print(f"  {name:45s}  {r['test_auc']:>7.4f}  "
          f"{r['test_f1']:>7.4f}  {r['test_acc']:>7.4f}")

print("=" * 70)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: ROC CURVES + AUC BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.patch.set_facecolor('white')

COLORS = {
    'threshold':       '#888888',
    'shewhart':        '#e89c2f',
    'cusum':           '#c1272d',
    'hotelling':       '#1c5d8c',
    'mahalanobis_mcd': '#3d9970',
}
LABELS = {
    'threshold':       'Threshold (naive)',
    'shewhart':        'Shewhart 3σ (1924)',
    'cusum':           'CUSUM (1954)',
    'hotelling':       "Hotelling T² (1947)",
    'mahalanobis_mcd': 'Mahalanobis MCD (1984)',
}

# ROC panel
ax1 = axes[0]
for key in order:
    if 'error' in results.get(key, {}): continue
    r = results[key]
    fpr, tpr = r['fpr_tpr']
    ax1.plot(fpr, tpr, lw=1.8, color=COLORS[key],
             label=f"{LABELS[key]} ({r['test_auc']:.3f})")

ax1.plot([0, 1], [0, 1], lw=1, color='#ccc', linestyle='--',
         label='Random (0.500)')
ax1.set_xlabel('False positive rate', fontsize=11)
ax1.set_ylabel('True positive rate',  fontsize=11)
ax1.set_title('ROC — Non-ML statistical baselines',
              fontsize=11, fontweight='bold')
ax1.legend(fontsize=8.5, loc='lower right')
ax1.grid(alpha=0.2)

# Bar chart panel
ax2 = axes[1]
labels = [LABELS[k] for k in order if 'error' not in results.get(k, {})]
aucs   = [results[k]['test_auc'] for k in order if 'error' not in results.get(k, {})]
colors = [COLORS[k] for k in order if 'error' not in results.get(k, {})]

bars = ax2.bar(range(len(aucs)), aucs, color=colors, edgecolor='#222', lw=1)
for bar, auc in zip(bars, aucs):
    ax2.text(bar.get_x() + bar.get_width()/2, auc + 0.01,
             f'{auc:.3f}', ha='center', fontsize=9, fontweight='bold')

ax2.axhline(0.5, color='#888', lw=1, linestyle='--', label='Random')
ax2.set_xticks(range(len(labels)))
ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=8.5)
ax2.set_ylabel('Test AUC', fontsize=11)
ax2.set_ylim(0, 1.0)
ax2.set_title('Statistical baselines — test AUC comparison',
              fontsize=11, fontweight='bold')
ax2.grid(alpha=0.2, axis='y')

plt.tight_layout()
plt.savefig('figures/fig_all_baselines.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n  ✓ Saved: models/all_baselines_results.json")
print(f"  ✓ Saved: figures/fig_all_baselines.png")
print(f"\n  NEXT: fold these results into 6_evaluate.py alongside the MLP ensemble.\n")
# 4_standard_baseline.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# Per-feature majority-vote threshold classifier.
#
# RULE:
#   For each feature, compute the mean of the TRAINING distribution.
#   For each test sample, each feature votes:
#     - feature < training mean → vote "drifted"
#     - feature ≥ training mean → vote "stable"
#   Final prediction: majority vote across the three features (≥2 of 3).
#
# This is a more rigorous extension of the naive single-mean-fidelity
# threshold: it preserves per-feature information and treats each canary
# circuit as an independent voter, rather than collapsing them into one
# scalar via averaging.
#
# IMPORTANT — NO DATA LEAKAGE:
#   Thresholds are computed on the TRAINING split only. Test data is never
#   touched until evaluation. This is required for honest reporting.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_train = X[split['train']]; y_train = y[split['train']]
X_val   = X[split['val']];   y_val   = y[split['val']]
X_test  = X[split['test']];  y_test  = y[split['test']]

print(f"Train : {len(X_train):,} rows | drift={round(y_train.mean()*100,1)}%")
print(f"Val   : {len(X_val):,} rows | drift={round(y_val.mean()*100,1)}%")
print(f"Test  : {len(X_test):,} rows | drift={round(y_test.mean()*100,1)}%\n")

# ── COMPUTE THRESHOLDS — TRAINING ONLY (no leakage) ───────────────────────────
feat_means = X_train.mean(axis=0)
print(f"Per-feature training means (used as decision thresholds):")
for name, mean in zip(FEATURES, feat_means):
    print(f"  {name:14s}: {mean:.6f}")
print()

# ── CLASSIFY ──────────────────────────────────────────────────────────────────
# Each feature votes: 1 = drifted (below training mean), 0 = stable.
# Final prediction: majority vote (>=2 of 3 features say drifted).
test_votes  = (X_test < feat_means).astype(int)
thresh_preds = (test_votes.sum(axis=1) >= 2).astype(int)

# AUC: rank by total drift votes (0, 1, 2, or 3). Higher score = more drifted.
thresh_scores_for_auc = test_votes.sum(axis=1)

auc       = round(roc_auc_score(y_test, thresh_scores_for_auc),           4)
accuracy  = round(accuracy_score(y_test, thresh_preds),                   4)
precision = round(precision_score(y_test, thresh_preds, zero_division=0), 4)
recall    = round(recall_score(y_test, thresh_preds, zero_division=0),    4)
f1        = round(f1_score(y_test, thresh_preds, zero_division=0),        4)

print("── Per-Feature Majority-Vote Threshold (Test Set) ──")
print(f"  AUC       : {auc}")
print(f"  Accuracy  : {accuracy}")
print(f"  Precision : {precision}")
print(f"  Recall    : {recall}")
print(f"  F1        : {f1}")

# ── VOTE DISTRIBUTION DIAGNOSTIC ──────────────────────────────────────────────
print(f"\n  Vote distribution on test set:")
for n_votes in range(4):
    mask = thresh_scores_for_auc == n_votes
    n    = mask.sum()
    if n == 0: continue
    drift_rate = y_test[mask].mean() if n > 0 else 0
    print(f"    {n_votes} drift votes: {n:>5,} samples  "
          f"({round(drift_rate*100, 1)}% actually drifted)")

# ── SAVE ──────────────────────────────────────────────────────────────────────
with open('results/baseline_standard_results.json', 'w') as f:
    json.dump({
        'standard_threshold_classifier': {
            'method':     'Per-feature majority-vote threshold',
            'auc':        auc,
            'accuracy':   accuracy,
            'precision':  precision,
            'recall':     recall,
            'f1':         f1,
            'thresholds': {
                name: round(float(mean), 6)
                for name, mean in zip(FEATURES, feat_means)
            },
            'rule':       'majority vote: feature < training_mean → drifted (>=2 of 3)',
            'training_only_thresholds': True,
        }
    }, f, indent=2)
print("\n  ✓ Saved results/baseline_standard_results.json")

# ── ROC FIGURE ────────────────────────────────────────────────────────────────
# Only 4 possible scores (0,1,2,3) → ROC is a step function with 4 points
fpr, tpr, _ = roc_curve(y_test, thresh_scores_for_auc)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, color='steelblue', lw=2, marker='o', markersize=6,
        label=f'Per-Feature Majority Vote (AUC={auc})')
ax.plot([0,1], [0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title('Standard Baseline ROC Curve\nPer-Feature Majority-Vote Threshold',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_baseline_standard_roc.png',
            dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_baseline_standard_roc.png")

print(f"\n{'='*50}")
print(f"  STANDARD BASELINE COMPLETE")
print(f"  AUC : {auc}")
print(f"{'='*50}")
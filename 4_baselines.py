# 4_baselines_naive.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# Naive baseline: threshold = global dataset mean per feature.
# No tuning, no scaling. Each feature votes independently, majority wins.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)

os.makedirs('results', exist_ok=True)
os.makedirs('figures',  exist_ok=True)

# ── LOAD ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_val  = X[split['val']];  y_val  = y[split['val']]
X_test = X[split['test']]; y_test = y[split['test']]

print(f"Val  : {len(X_val):,} rows | drift={round(y_val.mean()*100,1)}%")
print(f"Test : {len(X_test):,} rows | drift={round(y_test.mean()*100,1)}%\n")

# ── NAIVE THRESHOLD CLASSIFIER ────────────────────────────────────────────────
# Threshold for each feature = its mean across the ENTIRE dataset.
# No val tuning — the mean is computed once and applied directly.
# Each feature votes: below its mean = drifted.
# Final prediction: majority vote (>=2 of 3 features say drifted).

feat_means = X.mean(axis=0)
print(f"Global means:")
for name, mean in zip(FEATURES, feat_means):
    print(f"  {name}: {mean:.4f}")
print()

# Votes: 1 = drifted, 0 = stable
val_votes  = (X_val  < feat_means).astype(int)
test_votes = (X_test < feat_means).astype(int)

# Majority vote
thresh_preds = (test_votes.sum(axis=1) >= 2).astype(int)

# AUC score = number of "drifted" votes (0, 1, 2, or 3)
thresh_scores_for_auc = test_votes.sum(axis=1)

auc       = round(roc_auc_score(y_test, thresh_scores_for_auc),           4)
accuracy  = round(accuracy_score(y_test, thresh_preds),                   4)
precision = round(precision_score(y_test, thresh_preds, zero_division=0), 4)
recall    = round(recall_score(y_test, thresh_preds, zero_division=0),    4)
f1        = round(f1_score(y_test, thresh_preds, zero_division=0),        4)

print("── Naive Threshold Classifier (Test Set) ──")
print(f"  AUC       : {auc}")
print(f"  Accuracy  : {accuracy}")
print(f"  Precision : {precision}")
print(f"  Recall    : {recall}")
print(f"  F1        : {f1}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
with open('results/baseline_naive_results.json', 'w') as f:
    json.dump({
        'naive_threshold_classifier': {
            'auc':       auc,
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'thresholds': {
                name: round(float(mean), 6)
                for name, mean in zip(FEATURES, feat_means)
            },
            'rule': 'majority vote: feature < global_mean → drifted (>=2 of 3)'
        }
    }, f, indent=2)
print("\n  ✓ Saved results/baseline_naive_results.json")

# ── ROC FIGURE ────────────────────────────────────────────────────────────────
# Note: only 4 possible score values (0,1,2,3) so ROC is a step function
fpr, tpr, _ = roc_curve(y_test, thresh_scores_for_auc)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, color='steelblue', lw=2,
        label=f'Naive Threshold (AUC={auc})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title('Naive Baseline ROC Curve', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_baseline_naive_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_baseline_naive_roc.png")

print(f"\n{'='*45}")
print(f"  NAIVE BASELINE COMPLETE")
print(f"  AUC : {auc}")
print(f"{'='*45}")
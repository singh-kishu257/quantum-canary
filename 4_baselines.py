# 4_baselines.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# Simplest possible baseline: if mean fidelity drops below threshold → drifted.
# No fitting, no scaling, no learning. Pure rule-based detection.
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

# ── THRESHOLD CLASSIFIER ──────────────────────────────────────────────────────
# Score = mean of raw fidelity features (no scaling).
# Low mean fidelity = bad qubit health = drifted.
# Sweep thresholds on val, pick best F1, apply to test.

val_scores  = X_val.mean(axis=1)
test_scores = X_test.mean(axis=1)

best_thresh, best_f1 = 0.0, -1.0
for t in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores < t).astype(int)   # below threshold = drifted
    if len(np.unique(preds)) < 2:
        continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1     = f1
        best_thresh = t

print(f"Best threshold (val F1={round(best_f1,4)}): mean_fidelity < {round(best_thresh,4)} → drifted\n")

# Apply to test
thresh_preds  = (test_scores < best_thresh).astype(int)
thresh_scores_for_auc = -test_scores   # negate so higher score = more drifted

auc       = round(roc_auc_score(y_test, thresh_scores_for_auc),                    4)
accuracy  = round(accuracy_score(y_test, thresh_preds),                            4)
precision = round(precision_score(y_test, thresh_preds, zero_division=0),          4)
recall    = round(recall_score(y_test, thresh_preds, zero_division=0),             4)
f1        = round(f1_score(y_test, thresh_preds, zero_division=0),                 4)

print("── Threshold Classifier (Test Set) ──")
print(f"  AUC       : {auc}")
print(f"  Accuracy  : {accuracy}")
print(f"  Precision : {precision}")
print(f"  Recall    : {recall}")
print(f"  F1        : {f1}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
with open('results/baseline_results.json', 'w') as f:
    json.dump({
        'threshold_classifier': {
            'auc':       auc,
            'accuracy':  accuracy,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
            'threshold': round(best_thresh, 6),
            'rule':      'mean(F_bell, F_gate, F_coherence) < threshold → drifted'
        }
    }, f, indent=2)
print("\n  ✓ Saved results/baseline_results.json")

# ── ROC FIGURE ────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, thresh_scores_for_auc)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'Threshold Classifier (AUC={auc})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title('Baseline ROC Curve', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_baseline_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_baseline_roc.png")

print(f"\n{'='*45}")
print(f"  BASELINE COMPLETE")
print(f"  Threshold AUC : {auc}")
print(f"  Rule          : mean fidelity < {round(best_thresh,4)} → drifted")
print(f"  MLP must beat this.")
print(f"{'='*45}")
print(f"\n  NEXT: python 5_train_mlp.py")
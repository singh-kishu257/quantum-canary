# 4_baselines.py — Quantum Canary Prototype 2
# Evaluates the threshold classifier baseline.
# This is the model the MLP must beat.

import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)

FEATURES = ['z_F_bell', 'z_F_gate', 'z_F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_train = X[split['train']]; y_train = y[split['train']]
X_val   = X[split['val']];   y_val   = y[split['val']]
X_test  = X[split['test']];  y_test  = y[split['test']]

print(f"Test set: {len(X_test):,} rows | drift={round(y_test.mean()*100,1)}%\n")

# ── 2. SCALE ──────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
np.save('models/scaler_mean.npy', scaler.mean_)
np.save('models/scaler_scale.npy', scaler.scale_)
print("Scaler saved.\n")

# ── 3. THRESHOLD CLASSIFIER ───────────────────────────────────────────────────
# Predicts drifted=1 if the mean of the 3 standardized features < 0
# (i.e. below the training mean). Simple, interpretable baseline.
# Score = negative mean of features (higher = more likely drifted)
threshold_scores = -X_test.mean(axis=1)

# Find best threshold on validation set
val_scores = -X_val.mean(axis=1)
best_thresh, best_auc = 0.0, 0.0
for t in np.linspace(val_scores.min(), val_scores.max(), 200):
    preds = (val_scores >= t).astype(int)
    if len(np.unique(preds)) < 2:
        continue
    auc = roc_auc_score(y_val, val_scores)
    if auc > best_auc:
        best_auc = auc
        best_thresh = t

thresh_preds = (threshold_scores >= best_thresh).astype(int)

def metrics(y_true, y_pred, y_score):
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred),                    4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0),  4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0),      4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0),          4),
        'auc':       round(roc_auc_score(y_true, y_score),                     4),
    }

thresh_metrics = metrics(y_test, thresh_preds, threshold_scores)

print("── Threshold Classifier (Test Set) ──")
for k, v in thresh_metrics.items():
    print(f"  {k:10s}: {v}")

# Save results
with open('results/baseline_results.json', 'w') as f:
    json.dump({'threshold_classifier': thresh_metrics}, f, indent=2)
print("\n  ✓ Saved results/baseline_results.json")

# ── 4. FIGURE: ROC CURVE ──────────────────────────────────────────────────────
print("\nGenerating Baseline ROC curve...")
fpr, tpr, _ = roc_curve(y_test, threshold_scores)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr, tpr, color='darkorange', lw=2,
        label=f'Threshold Classifier (AUC={thresh_metrics["auc"]})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('Baseline ROC Curve\n(Threshold Classifier vs Random)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_baseline_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_baseline_roc.png")

print(f"\n{'='*45}")
print(f"  BASELINE COMPLETE")
print(f"  Threshold AUC: {thresh_metrics['auc']}")
print(f"  MLP must beat this.")
print(f"{'='*45}")

# 4_baselines.py — Quantum Canary Prototype 2
import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('models',  exist_ok=True)

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

print(f"Test set: {len(X_test):,} rows | drift={round(y_test.mean()*100,1)}%\n")

# ── SCALE ─────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

np.save('models/scaler_mean.npy',  scaler.mean_)
np.save('models/scaler_scale.npy', scaler.scale_)
print("Scaler saved.\n")

# ── THRESHOLD CLASSIFIER ──────────────────────────────────────────────────────
# Lower mean fidelity = more likely drifted
# Score = negative mean of standardized features
threshold_scores = -X_test_s.mean(axis=1)
val_scores       = -X_val_s.mean(axis=1)

best_thresh, best_f1 = 0.0, -1.0
for t in np.linspace(val_scores.min(), val_scores.max(), 200):
    preds = (val_scores >= t).astype(int)
    if len(np.unique(preds)) < 2:
        continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1    = f1
        best_thresh = t

thresh_preds = (threshold_scores >= best_thresh).astype(int)

def metrics(y_true, y_pred, y_score):
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred),                   4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0),    4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0),        4),
        'auc':       round(roc_auc_score(y_true, y_score),                   4),
    }

thresh_metrics = metrics(y_test, thresh_preds, threshold_scores)
print("── Threshold Classifier (Test Set) ──")
for k, v in thresh_metrics.items():
    print(f"  {k:10s}: {v}")

# ── LOGISTIC REGRESSION ───────────────────────────────────────────────────────
logreg = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
logreg.fit(X_train_s, y_train)
logreg_scores = logreg.predict_proba(X_test_s)[:, 1]
logreg_preds  = (logreg_scores >= 0.5).astype(int)
logreg_metrics = metrics(y_test, logreg_preds, logreg_scores)

print("\n── Logistic Regression (Test Set) ──")
for k, v in logreg_metrics.items():
    print(f"  {k:10s}: {v}")

with open('results/baseline_results.json', 'w') as f:
    json.dump({'threshold_classifier': thresh_metrics,
               'logistic_regression':  logreg_metrics}, f, indent=2)
print("\n  ✓ Saved results/baseline_results.json")

# ── ROC FIGURE ────────────────────────────────────────────────────────────────
fpr_th, tpr_th, _   = roc_curve(y_test, threshold_scores)
fpr_lr, tpr_lr, _   = roc_curve(y_test, logreg_scores)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(fpr_th, tpr_th, color='darkorange', lw=2,
        label=f'Threshold Classifier (AUC={thresh_metrics["auc"]})')
ax.plot(fpr_lr, tpr_lr, color='steelblue',  lw=2,
        label=f'Logistic Regression (AUC={logreg_metrics["auc"]})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title('Baseline ROC Curve', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_baseline_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_baseline_roc.png")

print(f"\n{'='*45}")
print(f"  BASELINE COMPLETE")
print(f"  Threshold AUC : {thresh_metrics['auc']}")
print(f"  LogReg AUC    : {logreg_metrics['auc']}")
print(f"  MLP must beat this.")
print(f"{'='*45}")
print(f"\n  NEXT: python 5_train_mlp.py")
# 6_evaluate.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# Direct MLP vs Hotelling's T² comparison.
#
#
# REFERENCE:
#   Hotelling, H. (1947). Multivariate Quality Control. In Techniques of
#   Statistical Analysis. McGraw-Hill.
#
# OUTPUT:
#   results/evaluation_hotelling.json     — side-by-side metrics
#   figures/fig_roc_vs_hotelling.png      — ROC comparison
#   figures/fig_auc_vs_hotelling.png      — AUC bar chart
#   figures/fig_confusion_hotelling.png   — T² confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json', encoding='utf-8') as f:
    split = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_train = X[split['train']]; y_train = y[split['train']]
X_val   = X[split['val']];   y_val   = y[split['val']]
X_test  = X[split['test']];  y_test  = y[split['test']]

print(f"Test set : {len(X_test):,} rows | drift={round(y_test.mean()*100,1)}%\n")

# ── 2. HOTELLING'S T² BASELINE ────────────────────────────────────────────────
# Multivariate statistical process control (Hotelling 1947).
# Fits the stable training distribution's mean vector μ and covariance Σ,
# then scores each test sample by its Mahalanobis distance from μ:
#     T²(x) = (x - μ)ᵀ · Σ⁻¹ · (x - μ)
# High T² means the sample is statistically far from healthy hardware.

stable_train = X_train[y_train == 0]
mu_vec       = stable_train.mean(axis=0)
cov_matrix   = np.cov(stable_train, rowvar=False)
# Regularize to prevent singular inversion when features are correlated
cov_reg      = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])
cov_inv      = np.linalg.inv(cov_reg)

def hotelling_t2(X, mu, inv_cov):
    """Compute Hotelling's T² statistic for each row of X."""
    d = X - mu
    return np.einsum('ij,jk,ik->i', d, inv_cov, d)

# Tune decision threshold on validation F1
val_scores  = hotelling_t2(X_val,  mu_vec, cov_inv)
test_scores = hotelling_t2(X_test, mu_vec, cov_inv)

best_h, best_f1 = 0.0, -1.0
for h in np.linspace(val_scores.min(), val_scores.max(), 500):
    preds = (val_scores > h).astype(int)
    if len(np.unique(preds)) < 2:
        continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_h  = h

hotelling_preds = (test_scores > best_h).astype(int)
hotelling_auc   = round(roc_auc_score(y_test, test_scores), 4)

print(f"Hotelling's T² — AUC: {hotelling_auc}")
print(f"Healthy mean μ: {mu_vec.round(5)}")
print(f"Decision threshold h: {round(best_h, 4)}\n")

# ── 3. SCALE FOR MLP ─────────────────────────────────────────────────────────
scaler   = StandardScaler()
scaler.fit(X_train)
X_test_s = scaler.transform(X_test)

# ── 4. MLP ENSEMBLE PREDICTIONS ───────────────────────────────────────────────
print("Loading 10 MLP models...")
all_test_preds = []
for seed in range(10):
    model = keras.models.load_model(f'models/mlp_seed_{seed}.keras')
    preds = model.predict(X_test_s, verbose=0).flatten()
    all_test_preds.append(preds)

ensemble_preds       = np.mean(all_test_preds, axis=0)
ensemble_uncertainty = np.std(all_test_preds,  axis=0)
ensemble_labels      = (ensemble_preds >= 0.5).astype(int)

# ── 5. METRICS ────────────────────────────────────────────────────────────────
auc         = round(roc_auc_score(y_test, ensemble_preds),                     4)
accuracy    = round(accuracy_score(y_test, ensemble_labels),                   4)
precision   = round(precision_score(y_test, ensemble_labels, zero_division=0), 4)
recall      = round(recall_score(y_test, ensemble_labels, zero_division=0),    4)
f1          = round(f1_score(y_test, ensemble_labels, zero_division=0),        4)
cm          = confusion_matrix(y_test, ensemble_labels)
tn, fp, fn, tp = cm.ravel()
specificity = round(tn / (tn + fp), 4)
improvement = round((auc - hotelling_auc) / abs(hotelling_auc) * 100, 1)

# Hotelling metrics for side-by-side reporting
h_accuracy  = round(accuracy_score(y_test, hotelling_preds),                   4)
h_precision = round(precision_score(y_test, hotelling_preds, zero_division=0), 4)
h_recall    = round(recall_score(y_test, hotelling_preds, zero_division=0),    4)
h_f1        = round(f1_score(y_test, hotelling_preds, zero_division=0),        4)
h_cm        = confusion_matrix(y_test, hotelling_preds)
h_tn, h_fp, h_fn, h_tp = h_cm.ravel()
h_specificity = round(h_tn / (h_tn + h_fp), 4)

print(f"── MLP Ensemble vs Hotelling's T² — Test Set ──")
print(f"                  {'MLP':>10s}  {'Hotelling T²':>14s}")
print(f"  AUC           : {auc:>10}  {hotelling_auc:>14}")
print(f"  Accuracy      : {accuracy:>10}  {h_accuracy:>14}")
print(f"  Precision     : {precision:>10}  {h_precision:>14}")
print(f"  Recall        : {recall:>10}  {h_recall:>14}")
print(f"  Specificity   : {specificity:>10}  {h_specificity:>14}")
print(f"  F1            : {f1:>10}  {h_f1:>14}")
print(f"\n  MLP confusion   : TP={tp}  FP={fp}  TN={tn}  FN={fn}")
print(f"  Hotelling conf. : TP={h_tp}  FP={h_fp}  TN={h_tn}  FN={h_fn}")
print(f"\n  MLP improvement over Hotelling T²: {improvement}%")
print(f"  AUC > 0.93?      : {'YES' if auc > 0.93 else 'NO'}")
print(f"  Improvement>20%? : {'YES' if improvement >= 20 else 'NO'}")

with open('results/evaluation_hotelling.json', 'w') as f:
    json.dump({
        'mlp_ensemble': {
            'auc':         auc,
            'accuracy':    accuracy,
            'precision':   precision,
            'recall':      recall,
            'specificity': specificity,
            'f1':          f1,
        },
        'hotelling_t2': {
            'auc':         hotelling_auc,
            'accuracy':    h_accuracy,
            'precision':   h_precision,
            'recall':      h_recall,
            'specificity': h_specificity,
            'f1':          h_f1,
            'decision_threshold': round(best_h, 4),
            'reference':   'Hotelling (1947)',
        },
        'comparison': {
            'improvement_over_hotelling': improvement,
            'n_test_rows':                len(X_test),
            'drift_rate':                 round(y_test.mean()*100, 1),
        }
    }, f, indent=2)
print("\n  Saved results/evaluation_hotelling.json")

# ── 6. ROC CURVE ──────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, ensemble_preds)
fpr_t2,  tpr_t2,  _ = roc_curve(y_test, test_scores)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.fill_between(fpr_mlp, tpr_mlp, alpha=0.15, color='crimson')
ax.plot(fpr_mlp, tpr_mlp, color='crimson', lw=2.5,
        label=f'Quantum Canary MLP (AUC={auc})')
ax.plot(fpr_t2,  tpr_t2,  color='#1c5d8c', lw=2,
        label=f"Hotelling's T² (AUC={hotelling_auc})")
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title("ROC — Quantum Canary vs Hotelling's T²\nHeld-out Test Set",
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_roc_vs_hotelling.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_roc_vs_hotelling.png")

# ── 7. HOTELLING CONFUSION MATRIX ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4))
disp = ConfusionMatrixDisplay(h_cm, display_labels=['Stable', 'Drifted'])
disp.plot(ax=ax, colorbar=False, cmap='Oranges')
ax.set_title("Hotelling's T² Confusion Matrix\nHeld-out Test Set",
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_confusion_hotelling.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_confusion_hotelling.png")

# ── 8. AUC COMPARISON ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
models_ = ["Hotelling's T²\n(Hotelling 1947)", 'Quantum Canary\nMLP Ensemble']
aucs_   = [hotelling_auc, auc]
colors_ = ['#1c5d8c', 'crimson']
bars    = ax.bar(models_, aucs_, color=colors_,
                 width=0.45, edgecolor='black', linewidth=0.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel('AUC', fontsize=11)
ax.set_title('AUC Comparison — Held-out Test Set',
             fontsize=11, fontweight='bold')
for bar, val in zip(bars, aucs_):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.annotate(f'+{improvement}% improvement',
            xy=(1, auc), xytext=(0.4, auc + 0.07),
            fontsize=10, fontweight='bold', color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_auc_vs_hotelling.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_auc_vs_hotelling.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  MLP vs HOTELLING'S T² EVALUATION COMPLETE")
print(f"{'='*60}")
print(f"  MLP Test AUC       : {auc}")
print(f"  Hotelling T² AUC   : {hotelling_auc}")
print(f"  Improvement        : {improvement}%")
print(f"  AUC > 0.93?        : {'YES' if auc > 0.93 else 'NO'}")
print(f"  Improvement > 20%? : {'YES' if improvement >= 20 else 'NO'}")
print(f"{'='*60}")
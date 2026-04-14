# 6_evaluate.py — Quantum Canary 2
# Evaluates the 10-seed MLP ensemble on the held-out test split from
# data/features_data.csv (18,000-row hybrid dataset: 12,000 physics-grounded
# synthetic + 6,000 IBM extreme anchor samples).

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
from tensorflow import keras

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json', encoding='utf-8') as f:
    split = json.load(f)
with open('results/baseline_results.json', encoding='utf-8') as f:
    baseline = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_test = X[split['test']]
y_test = y[split['test']]

print(f"Test set : {len(X_test):,} rows")
print(f"Drift rate: {round(y_test.mean()*100, 1)}%\n")

# ── 2. SCALE ──────────────────────────────────────────────────────────────────
scaler_mean  = np.load('models/scaler_mean.npy')
scaler_scale = np.load('models/scaler_scale.npy')
X_test_s     = (X_test - scaler_mean) / scaler_scale

# ── 3. ENSEMBLE PREDICTIONS ───────────────────────────────────────────────────
print("Loading 10 models...")
all_test_preds = []
for seed in range(10):
    model = keras.models.load_model(f'models/mlp_seed_{seed}.keras')
    preds = model.predict(X_test_s, verbose=0).flatten()
    all_test_preds.append(preds)

ensemble_preds       = np.mean(all_test_preds, axis=0)
ensemble_uncertainty = np.std(all_test_preds,  axis=0)
ensemble_labels      = (ensemble_preds >= 0.5).astype(int)

# ── 4. METRICS ────────────────────────────────────────────────────────────────
auc         = round(roc_auc_score(y_test, ensemble_preds),                    4)
accuracy    = round(accuracy_score(y_test, ensemble_labels),                  4)
precision   = round(precision_score(y_test, ensemble_labels, zero_division=0),4)
recall      = round(recall_score(y_test, ensemble_labels, zero_division=0),   4)
f1          = round(f1_score(y_test, ensemble_labels, zero_division=0),       4)
cm          = confusion_matrix(y_test, ensemble_labels)
tn, fp, fn, tp = cm.ravel()
specificity = round(tn / (tn + fp), 4)

threshold_auc   = baseline['threshold_classifier']['auc']
logreg_auc      = baseline['logistic_regression']['auc']
traditional_auc = max(threshold_auc, logreg_auc)
improvement     = round((auc - traditional_auc) / abs(traditional_auc) * 100, 1)

print(f"\n── MLP Ensemble — Test Set Results ──")
print(f"  AUC              : {auc}")
print(f"  Accuracy         : {accuracy}")
print(f"  Precision        : {precision}")
print(f"  Recall           : {recall}")
print(f"  Specificity      : {specificity}")
print(f"  F1               : {f1}")
print(f"  TP: {tp}  FP: {fp}  TN: {tn}  FN: {fn}")
print(f"  Threshold AUC    : {threshold_auc}")
print(f"  LogReg AUC       : {logreg_auc}")
print(f"  Traditional AUC  : {traditional_auc}")
print(f"  Improvement      : {improvement}%")
print(f"  AUC > 0.93?      : {'YES ✓' if auc > 0.93 else 'NO'}")
print(f"  Improvement>20%? : {'YES ✓' if improvement >= 20 else 'NO'}")

with open('results/evaluation_results.json', 'w') as f:
    json.dump({
        'mlp_ensemble': {
            'auc':                        auc,
            'accuracy':                   accuracy,
            'precision':                  precision,
            'recall':                     recall,
            'specificity':                specificity,
            'f1':                         f1,
            'improvement_over_traditional': improvement,
            'n_test_rows':                len(X_test),
            'drift_rate':                 round(y_test.mean()*100, 1),
        }
    }, f, indent=2)
print("\n  ✓ Saved results/evaluation_results.json")

# ── 5. ROC CURVE ──────────────────────────────────────────────────────────────
print("\nGenerating figures...")
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, ensemble_preds)
thresh_scores        = -X_test_s.mean(axis=1)
fpr_th,  tpr_th,  _ = roc_curve(y_test, thresh_scores)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr_mlp, tpr_mlp, color='crimson',    lw=2.5,
        label=f'Neural Canary MLP (AUC={auc})')
ax.plot(fpr_th,  tpr_th,  color='darkorange', lw=2,
        label=f'Threshold Classifier (AUC={threshold_auc})')
ax.plot([0,1],[0,1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate',  fontsize=10)
ax.set_title('Model ROC Curves — Held-out Test Set',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_roc_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_roc_test.png")

# ── 6. CONFUSION MATRIX ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(4.5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Stable', 'Drifted'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('MLP Ensemble Confusion Matrix\nHeld-out Test Set',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_confusion_matrix.png")

# ── 7. AUC COMPARISON ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
models_ = ['Threshold', 'Logistic\nRegression', 'Neural Canary\nMLP Ensemble']
aucs_   = [threshold_auc, logreg_auc, auc]
colors_ = ['darkorange', 'steelblue', 'crimson']
bars    = ax.bar(models_, aucs_, color=colors_,
                 width=0.45, edgecolor='black', linewidth=0.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel('AUC', fontsize=11)
ax.set_title('AUC Comparison — Held-out Test Set',
             fontsize=11, fontweight='bold')
for bar, val in zip(bars, aucs_):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.annotate(f'+{improvement}% vs best traditional',
            xy=(2, auc), xytext=(1.0, auc + 0.07),
            fontsize=10, fontweight='bold', color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_auc_comparison.png")

# ── 8. ABLATION STUDY ─────────────────────────────────────────────────────────
print("\nRunning ablation study...")
ablation = {}
for drop_feat in FEATURES:
    preds_abl = []
    for seed in range(10):
        model    = keras.models.load_model(f'models/mlp_seed_{seed}.keras')
        X_zeroed = X_test_s.copy()
        X_zeroed[:, FEATURES.index(drop_feat)] = 0
        preds_abl.append(model.predict(X_zeroed, verbose=0).flatten())
    abl_auc = round(roc_auc_score(y_test, np.mean(preds_abl, axis=0)), 4)
    ablation[drop_feat] = abl_auc
    print(f"  Without {drop_feat:15s}: AUC={abl_auc}  (drop={round(auc - abl_auc, 4)})")

fig, ax = plt.subplots(figsize=(6, 4))
abl_aucs = [ablation[f] for f in FEATURES]
drops    = [round(auc - a, 4) for a in abl_aucs]
bars     = ax.bar(FEATURES, abl_aucs, color='steelblue',
                  width=0.4, edgecolor='black', linewidth=0.8)
ax.axhline(auc, color='crimson', lw=2, linestyle='--',
           label=f'Full model AUC={auc}')
ax.set_ylim(max(0, min(abl_aucs) - 0.1), 1.0)
ax.set_ylabel('AUC', fontsize=10)
ax.set_title('Ablation Study — Feature Importance\nHeld-out Test Set',
             fontsize=11, fontweight='bold')
for bar, val, drop in zip(bars, abl_aucs, drops):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val}\n(−{drop})', ha='center', va='bottom', fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_ablation.png")

# ── 9. CONFIDENCE DISTRIBUTION ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ensemble_preds[y_test==0], bins=50, alpha=0.6, color='steelblue',
        density=True, label=f'Stable (n={int((y_test==0).sum()):,})')
ax.hist(ensemble_preds[y_test==1], bins=50, alpha=0.6, color='crimson',
        density=True, label=f'Drifted (n={int((y_test==1).sum()):,})')
ax.axvline(0.5, color='black', lw=1.5, linestyle='--',
           label='Decision threshold (0.5)')
ax.set_xlabel('Ensemble Drift Probability', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Model Confidence Distribution — Held-out Test Set',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_confidence_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_confidence_test.png")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  EVALUATION COMPLETE")
print(f"{'='*55}")
print(f"  MLP Test AUC    : {auc}")
print(f"  Traditional AUC : {traditional_auc}")
print(f"  Improvement     : {improvement}%")
print(f"  Recall          : {recall}")
print(f"  Specificity     : {specificity}")
print(f"  AUC > 0.93?     : {'YES ✓' if auc > 0.93 else 'NO'}")
print(f"  Improve > 20%?  : {'YES ✓' if improvement >= 20 else 'NO'}")
print(f"{'='*55}")
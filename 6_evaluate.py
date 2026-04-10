# 6_evaluate.py — Quantum Canary Prototype 2
# Final evaluation on held-out test set.
# Generates all board figures: ROC, confusion matrix, AUC comparison, ablation.

import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
from tensorflow import keras

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')
with open('data/split_indices.json') as f:
    split = json.load(f)
with open('results/baseline_results.json') as f:
    baseline = json.load(f)

FEATURES = ['z_F_bell', 'z_F_gate', 'z_F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_test = X[split['test']]
y_test = y[split['test']]

scaler_mean  = np.load('models/scaler_mean.npy')
scaler_scale = np.load('models/scaler_scale.npy')
X_test_scaled = (X_test - scaler_mean) / scaler_scale

# ── 2. ENSEMBLE PREDICTIONS ON TEST SET ──────────────────────────────────────
print("Loading 10 models and predicting on test set...")
all_test_preds = []
for seed in range(10):
    model = keras.models.load_model(f'models/mlp_seed_{seed}.keras')
    preds = model.predict(X_test_scaled, verbose=0).flatten()
    all_test_preds.append(preds)

ensemble_preds       = np.mean(all_test_preds, axis=0)
ensemble_uncertainty = np.std(all_test_preds,  axis=0)
ensemble_labels      = (ensemble_preds >= 0.5).astype(int)

# ── 3. METRICS ────────────────────────────────────────────────────────────────
auc       = round(roc_auc_score(y_test, ensemble_preds), 4)
accuracy  = round(accuracy_score(y_test, ensemble_labels), 4)
precision = round(precision_score(y_test, ensemble_labels, zero_division=0), 4)
recall    = round(recall_score(y_test, ensemble_labels, zero_division=0), 4)
f1        = round(f1_score(y_test, ensemble_labels, zero_division=0), 4)

thresh_auc = baseline['threshold_classifier']['auc']
improvement = round((auc - thresh_auc) / abs(thresh_auc) * 100, 1)

print(f"\n── MLP Ensemble Test Set Results ──")
print(f"  AUC       : {auc}")
print(f"  Accuracy  : {accuracy}")
print(f"  Precision : {precision}")
print(f"  Recall    : {recall}")
print(f"  F1        : {f1}")
print(f"  Threshold AUC : {thresh_auc}")
print(f"  Improvement   : {improvement}%")

results = {
    'mlp_ensemble': {
        'auc': auc, 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'f1': f1
    }
}
with open('results/evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── 4. FIGURE: ROC CURVE ──────────────────────────────────────────────────────
print("\nGenerating figures...")
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, ensemble_preds)
thresh_scores = -X_test_scaled.mean(axis=1)
fpr_th, tpr_th, _   = roc_curve(y_test, thresh_scores)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.plot(fpr_mlp, tpr_mlp, color='crimson', lw=2.5,
        label=f'Neural Canary MLP (AUC={auc})')
ax.plot(fpr_th, tpr_th, color='darkorange', lw=2,
        label=f'Threshold Classifier (AUC={thresh_auc})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('Model ROC Curves — Test Set', fontsize=11, fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_roc_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_roc_test.png")

# ── 5. FIGURE: CONFUSION MATRIX ───────────────────────────────────────────────
cm = confusion_matrix(y_test, ensemble_labels)
fig, ax = plt.subplots(figsize=(4.5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Stable','Drifted'])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('MLP Ensemble Confusion Matrix\nTest Set', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_confusion_matrix.png")

# ── 6. FIGURE: AUC COMPARISON BAR CHART ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
models  = ['Threshold\nClassifier', 'Neural Canary\nMLP Ensemble']
aucs    = [thresh_auc, auc]
colors  = ['darkorange', 'crimson']
bars    = ax.bar(models, aucs, color=colors, width=0.45, edgecolor='black', linewidth=0.8)
ax.set_ylim(0, 1.0)
ax.set_ylabel('AUC', fontsize=11)
ax.set_title('AUC Comparison — Test Set', fontsize=11, fontweight='bold')
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.annotate(f'+{improvement}% AUC', xy=(1, auc), xytext=(0.5, auc + 0.07),
            fontsize=11, fontweight='bold', color='crimson',
            arrowprops=dict(arrowstyle='->', color='crimson'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_auc_comparison.png")

# ── 7. ABLATION STUDY ─────────────────────────────────────────────────────────
print("\nRunning ablation study...")
ablation = {}
for drop_feat in FEATURES:
    remaining = [f for f in FEATURES if f != drop_feat]
    X_abl = df[remaining].values[split['test']]
    X_abl = (X_abl - scaler_mean[[FEATURES.index(f) for f in remaining]]) / \
             scaler_scale[[FEATURES.index(f) for f in remaining]]
    preds_abl = []
    for seed in range(10):
        model = keras.models.load_model(f'models/mlp_seed_{seed}.keras')
        # Use only the remaining features — retrain would be ideal but
        # for ablation we zero out the dropped feature
        full = np.zeros((len(X_abl), 3))
        for i, f in enumerate(remaining):
            full[:, FEATURES.index(f)] = X_abl[:, i]
        preds_abl.append(model.predict(full, verbose=0).flatten())
    abl_auc = round(roc_auc_score(y_test, np.mean(preds_abl, axis=0)), 4)
    ablation[drop_feat] = abl_auc
    print(f"  Without {drop_feat}: AUC={abl_auc}  (drop={round(auc-abl_auc,4)})")

# Ablation figure
fig, ax = plt.subplots(figsize=(6, 4))
feat_labels = ['Without\nz_F_bell', 'Without\nz_F_gate', 'Without\nz_F_coherence']
abl_aucs    = [ablation[f] for f in FEATURES]
drops       = [round(auc - a, 4) for a in abl_aucs]
bars = ax.bar(feat_labels, abl_aucs, color='steelblue',
              width=0.45, edgecolor='black', linewidth=0.8)
ax.axhline(auc, color='crimson', lw=2, linestyle='--',
           label=f'Full model AUC={auc}')
ax.set_ylim(min(abl_aucs) - 0.05, 1.0)
ax.set_ylabel('AUC', fontsize=10)
ax.set_title('Ablation Study — Feature Importance\nAUC when each feature removed',
             fontsize=11, fontweight='bold')
for bar, val, drop in zip(bars, abl_aucs, drops):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val}\n(−{drop})', ha='center', va='bottom', fontsize=8)
ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_ablation.png")

print(f"\n{'='*45}")
print(f"  EVALUATION COMPLETE")
print(f"  MLP Test AUC    : {auc}")
print(f"  Threshold AUC   : {thresh_auc}")
print(f"  Improvement     : {improvement}%")
print(f"  Recall          : {recall}")
print(f"  Precision       : {precision}")
print(f"{'='*45}")
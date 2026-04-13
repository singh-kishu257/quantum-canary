# 6_evaluate.py — Quantum Canary Prototype 2
# ─────────────────────────────────────────────────────────────────────────────
# FINAL EVALUATION & BENCHMARKING
#
# Tests the 10-model MLP ensemble on the held-out TEST split.
# Compares performance against traditional threshold-based baselines.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
from tensorflow import keras

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print("Loading test data and models...")
df = pd.read_csv("data/features_data.csv")
with open("data/split_indices.json", encoding="utf-8") as f:
    split = json.load(f)

# Load baseline results to get traditional benchmarks
with open("results/baseline_results.json", encoding="utf-8") as f:
    baseline = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X = df[FEATURES].values
y = df["drifted"].values

X_test = X[split["test"]]
y_test = y[split["test"]]

# Load normalization params if they exist
try:
    scaler_mean = np.load("models/scaler_mean.npy")
    scaler_scale = np.load("models/scaler_scale.npy")
    X_test_processed = (X_test - scaler_mean) / scaler_scale
    print("  ✓ Scaler found and applied.")
except FileNotFoundError:
    X_test_processed = X_test
    print("  ! No scaler found. Using raw features.")

print(f"  Test set size: {len(X_test):,} rows")
print(f"  Drift rate   : {round(y_test.mean() * 100, 1)}%\n")

# ── 2. ENSEMBLE PREDICTIONS ───────────────────────────────────────────────────
# Fix: Using the correct filename 'mlp_seed_{seed}.keras' as specified
print("Predicting with 10-model ensemble...")
all_test_preds = []
for seed in range(10):
    model_path = f"models/mlp_seed_{seed}.keras"
    if not os.path.exists(model_path):
        # Fallback check for the 'canary_' prefix just in case
        model_path = f"models/canary_mlp_seed_{seed}.keras"
    
    model = keras.models.load_model(model_path)
    preds = model.predict(X_test_processed, verbose=0).flatten()
    all_test_preds.append(preds)

ensemble_preds = np.mean(all_test_preds, axis=0)
ensemble_labels = (ensemble_preds >= 0.5).astype(int)

# ── 3. METRICS CALCULATION ────────────────────────────────────────────────────
auc = round(roc_auc_score(y_test, ensemble_preds), 4)
accuracy = round(accuracy_score(y_test, ensemble_labels), 4)
precision = round(precision_score(y_test, ensemble_labels, zero_division=0), 4)
recall = round(recall_score(y_test, ensemble_labels, zero_division=0), 4)
f1 = round(f1_score(y_test, ensemble_labels, zero_division=0), 4)

cm = confusion_matrix(y_test, ensemble_labels)
tn, fp, fn, tp = cm.ravel()
specificity = round(tn / (tn + fp), 4)

# Benchmark comparison
thresh_auc = baseline['threshold_classifier']['auc']
logreg_auc = baseline['logistic_regression']['auc']
best_traditional_auc = max(thresh_auc, logreg_auc)
improvement = round((auc - best_traditional_auc) / abs(best_traditional_auc) * 100, 1)

print(f"\n── MLP Ensemble — Performance Report ──")
print(f"  AUC          : {auc}")
print(f"  Accuracy     : {accuracy}")
print(f"  Precision    : {precision}")
print(f"  Recall       : {recall}")
print(f"  Specificity  : {specificity}")
print(f"  F1-Score     : {f1}")
print(f"  Improvement  : {improvement}% vs best traditional baseline")

# ── 4. ROC CURVE VISUALIZATION ────────────────────────────────────────────────
print("\nGenerating Figures...")
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, ensemble_preds)
# For the plot, we re-calculate threshold ROC directly on test set values
thresh_scores = 1.0 - np.mean(X_test, axis=1) 
fpr_th, tpr_th, _ = roc_curve(y_test, thresh_scores)

plt.figure(figsize=(6, 6))
plt.plot(fpr_th, tpr_th, color='darkorange', lw=2, label=f'Threshold Baseline (AUC={thresh_auc})')
plt.plot(fpr_mlp, tpr_mlp, color='crimson', lw=3, label=f'Neural Canary MLP (AUC={auc})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison - IBM Hardware Test Set', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_roc_ibm_test.png', dpi=300)
plt.close()

# ── 5. CONFUSION MATRIX ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable', 'Drifted'])
disp.plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title('MLP Ensemble Confusion Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig('figures/fig_confusion_matrix_ibm.png', dpi=300)
plt.close()

# ── 6. ABLATION STUDY ─────────────────────────────────────────────────────────
print("Running ablation study...")
ablation = {}
for drop_feat in FEATURES:
    preds_abl = []
    feat_idx = FEATURES.index(drop_feat)
    X_abl = X_test_processed.copy()
    X_abl[:, feat_idx] = 0 # Zero out the feature
    
    for seed in range(10):
        model_path = f"models/mlp_seed_{seed}.keras"
        if not os.path.exists(model_path):
            model_path = f"models/canary_mlp_seed_{seed}.keras"
        model = keras.models.load_model(model_path)
        preds_abl.append(model.predict(X_abl, verbose=0).flatten())
    
    abl_auc = round(roc_auc_score(y_test, np.mean(preds_abl, axis=0)), 4)
    ablation[drop_feat] = abl_auc
    print(f"  Without {drop_feat:12s}: AUC={abl_auc} (drop={round(auc - abl_auc, 4)})")

# ── 7. SAVE RESULTS ───────────────────────────────────────────────────────────
with open('results/evaluation_results_ibm.json', 'w') as f:
    json.dump({
        "mlp_ensemble": {
            "auc": auc, "accuracy": accuracy, "precision": precision,
            "recall": recall, "specificity": specificity, "f1": f1,
            "improvement": improvement
        },
        "ablation": ablation
    }, f, indent=2)

print(f"\nEvaluation Complete.")
print("  ✓ Saved results/evaluation_results_ibm.json")
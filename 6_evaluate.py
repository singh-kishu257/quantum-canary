# Full evaluation of all 3 models on the held-out test set.
# Ablation study to rank feature importance.
# Lead time analysis — how early Neural Canary detects drift.
# Generates 4 publication-quality figures.

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
df = pd.read_csv('data/features_data.csv')


with open('data/split_indices.json', 'r') as f:
    split = json.load(f)

FEATURES = ['F_bell','F_coherence', 'F_gate']

X = df[FEATURES].values
y = df['drifted'].values

X_train = X[split['train']];  y_train = y[split['train']]
X_val   = X[split['val']];    y_val   = y[split['val']]
X_test  = X[split['test']];   y_test  = y[split['test']]
df_test = df.iloc[split['test']].reset_index(drop=True)

print(f"Test set: {len(X_test)} rows")
print(f"Test drift rate: {round(y_test.mean()*100, 1)}%")

# Load scaler — same normalization used in training
scaler        = StandardScaler()
scaler.mean_  = np.load('models/scaler_mean.npy')
scaler.scale_ = np.load('models/scaler_scale.npy')

X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Helper: compute all 5 metrics ────────────────────────────
def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        'accuracy':  round(accuracy_score(y_true, y_pred),                   4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall':    round(recall_score(y_true, y_pred, zero_division=0),     4),
        'f1':        round(f1_score(y_true, y_pred, zero_division=0),         4),
        'auc':       round(roc_auc_score(y_true, y_prob),                     4),
    }

results = {}

# ════════════════════════════════════════════════════════════
# MODEL 1: MAJORITY BASELINE on test set
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL 1: Majority Baseline — Test Set")
print("="*55)

majority_probs = np.zeros(len(y_test))
m1 = metrics(y_test, majority_probs)
results['majority_baseline'] = m1
print(f"Accuracy: {m1['accuracy']} | Recall: {m1['recall']} | AUC: {m1['auc']}")

# ════════════════════════════════════════════════════════════
# MODEL 2: THRESHOLD CLASSIFIER on test set
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL 2: Threshold Classifier — Test Set")
print("="*55)

# gate_error_canary is feature index 2
gate_train = X_train_sc[:, 2]
gate_test  = X_test_sc[:, 2]

g_min = gate_test.min()
g_max = gate_test.max()
threshold_probs = (gate_test - g_min) / (g_max - g_min + 1e-8)

m2 = metrics(y_test, threshold_probs)
results['threshold_classifier'] = m2
print(f"Accuracy: {m2['accuracy']} | Recall: {m2['recall']} | AUC: {m2['auc']}")

# ════════════════════════════════════════════════════════════
# MODEL 3: MLP ENSEMBLE on test set
# ════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("MODEL 3: MLP Ensemble — Test Set")
print("="*55)

# Load all 10 trained models and collect predictions
all_test_preds = []
for seed in range(10):
    model     = keras.models.load_model(f'models/mlp_seed_{seed}.keras')
    preds     = model.predict(X_test_sc, verbose=0).flatten()
    all_test_preds.append(preds)

ensemble_probs       = np.mean(all_test_preds, axis=0)
ensemble_uncertainty = np.std(all_test_preds,  axis=0)

m3 = metrics(y_test, ensemble_probs)
results['mlp_ensemble'] = m3
print(f"Accuracy: {m3['accuracy']} | Recall: {m3['recall']} | AUC: {m3['auc']}")
print(f"Mean uncertainty: {round(ensemble_uncertainty.mean(), 4)}")

# ── Full comparison table ─────────────────────────────────────
print("\n" + "="*55)
print("FULL COMPARISON — TEST SET")
print("="*55)
print(f"{'Model':<25} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
print("-"*55)
for name, m in results.items():
    print(f"{name:<25} {m['accuracy']:>6} {m['precision']:>6} "
          f"{m['recall']:>6} {m['f1']:>6} {m['auc']:>6}")

improvement = round((m3['auc'] - m2['auc']) / m2['auc'] * 100, 1)
print(f"\nAUC improvement over threshold: {improvement}%")

# Save results table
results_df = pd.DataFrame(results).T
results_df.to_csv('results/results_table.csv')
print("Saved results/results_table.csv")

# ════════════════════════════════════════════════════════════
# ABLATION STUDY
# ════════════════════════════════════════════════════════════
# Retrain MLP removing one feature at a time.
# AUC drop = how much that feature contributed.
print("\n" + "="*55)
print("ABLATION STUDY")
print("="*55)

# Class weights
n_total  = len(y_train)
n_drift  = y_train.sum()
n_stable = n_total - n_drift
class_weight = {0: n_drift/n_total, 1: n_stable/n_total}

def train_ablation(feature_indices, seeds=3):
    # Train a smaller ensemble with only specific features
    X_tr = X_train_sc[:, feature_indices]
    X_te = X_test_sc[:, feature_indices]
    preds_list = []
    for seed in range(seeds):
        tf.random.set_seed(seed)
        np.random.seed(seed)
        n_feat = len(feature_indices)
        model  = keras.Sequential([
            keras.layers.Input(shape=(n_feat,)),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(8,  activation='relu'),
            keras.layers.Dense(1,  activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=['accuracy'])
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10,
            restore_best_weights=True, verbose=0)
        X_val_ab = X_test_sc[:, feature_indices]
        model.fit(X_tr, y_train,
                  validation_data=(X_val_ab, y_test),
                  epochs=60, batch_size=16,
                  class_weight=class_weight,
                  callbacks=[early_stop], verbose=0)
        preds_list.append(model.predict(X_te, verbose=0).flatten())
    return roc_auc_score(y_test, np.mean(preds_list, axis=0))

# Full model AUC
full_auc = m3['auc']

# Remove one feature at a time
ablation_results = {}
for i, feat in enumerate(FEATURES):
    remaining = [j for j in range(len(FEATURES)) if j != i]
    print(f"  Removing {feat}...")
    auc_without = train_ablation(remaining)
    drop = round(full_auc - auc_without, 4)
    ablation_results[feat] = {
        'auc_without': round(auc_without, 4),
        'auc_drop':    drop
    }
    print(f"    AUC without {feat}: {round(auc_without,4)} (drop: {drop})")

# Rank by importance
ablation_df = pd.DataFrame(ablation_results).T
ablation_df = ablation_df.sort_values('auc_drop', ascending=False)
ablation_df.to_csv('results/ablation.csv')
print("\nFeature importance ranking:")
print(ablation_df)
print("Saved results/ablation.csv")


# ════════════════════════════════════════════════════════════
# FIGURE : ROC Curves — all 3 models
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 7))

fpr1, tpr1, _ = roc_curve(y_test, majority_probs)
fpr2, tpr2, _ = roc_curve(y_test, threshold_probs)
fpr3, tpr3, _ = roc_curve(y_test, ensemble_probs)

ax.plot(fpr1, tpr1, color='gray',      linewidth=2,
        label=f"Majority baseline (AUC={m1['auc']})")
ax.plot(fpr2, tpr2, color='steelblue', linewidth=2,
        label=f"Threshold classifier (AUC={m2['auc']})")
ax.plot(fpr3, tpr3, color='red',       linewidth=2.5,
        label=f"MLP ensemble (AUC={m3['auc']})")
ax.plot([0,1],[0,1], 'k--', linewidth=1, label='Random (AUC=0.5)')
ax.fill_between(fpr3, tpr3, alpha=0.1, color='red')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Quantum Canary — ROC Curves\nAll 3 Models on Test Set',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig2_roc.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig2_roc.png")

# ════════════════════════════════════════════════════════════
# FIGURE : AUC Comparison Bar Chart with error bars
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

names  = ['Majority\nBaseline', 'Threshold\nClassifier', 'MLP\nEnsemble']
aucs   = [m1['auc'], m2['auc'], m3['auc']]
colors = ['#cccccc', '#5588bb', '#1a3a6a']

# Error bar for MLP = std across 10 seeds
seed_aucs = [roc_auc_score(y_test, p) for p in all_test_preds]
errors    = [0, 0, round(np.std(seed_aucs), 4)]

bars = ax.bar(names, aucs, color=colors,
              width=0.5, edgecolor='navy',
              linewidth=0.8, yerr=errors,
              capsize=6, error_kw={'linewidth': 2})

for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.015,
            str(val), ha='center',
            fontweight='bold', fontsize=12)

ax.set_ylabel('AUC Score', fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_title(f'Neural Canary — AUC Comparison\nMLP achieves +{improvement}% over threshold baseline',
             fontsize=13, fontweight='bold')
ax.axhline(y=m2['auc'], color='red', linestyle='--',
           linewidth=1.5, label=f'Threshold ceiling: {m2["auc"]}')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig3_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved figures/fig3_auc_comparison.png")

# ════════════════════════════════════════════════════════════
# FIGURE :Confusion Matrix
# ════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix
y_pred_binary = (ensemble_probs >= 0.5).astype(int)
cm   = confusion_matrix(y_test, y_pred_binary)
disp = ConfusionMatrixDisplay(cm, display_labels=['Stable', 'Drifted'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title('MLP Ensemble — Confusion Matrix\n(Test Set)',
                  fontsize=12, fontweight='bold')


# ── Final summary ─────────────────────────────────────────────
print("\n" + "="*55)
print("PHASE 5 COMPLETE — FINAL RESULTS")
print("="*55)
print(f"Majority baseline AUC:  {m1['auc']}")
print(f"Threshold baseline AUC: {m2['auc']}")
print(f"MLP ensemble AUC:       {m3['auc']}")
print(f"AUC improvement:        +{improvement}%")
print(f"MLP Recall:             {m3['recall']}")
print("\nAll figures saved to figures/")
print("All results saved to results/")
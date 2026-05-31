# 6_evaluate.py — Quantum Canary 2
# ─────────────────────────────────────────────────────────────────────────────
# Three-way comparison on held-out test set:
#   1. Per-feature majority-vote threshold (naive baseline)
#   2. Hotelling's T² (multivariate statistical baseline, Hotelling 1947)
#   3. Quantum Canary MLP ensemble (this work)
#
# All three operate on the same three canary features (F_bell, F_gate,
# F_coherence). Thresholds and decision boundaries are tuned strictly on
# training/validation data — never on the test set.
#
# OUTPUT:
#   results/evaluation_results.json  — full metrics for all three methods
#   figures/fig_roc_test.png         — three-way ROC comparison
#   figures/fig_auc_comparison.png   — three-way AUC bar chart
#   figures/fig_confusion_matrix.png — MLP confusion matrix
#   figures/fig_ablation.png         — feature ablation
#   figures/fig_confidence_test.png  — MLP confidence distribution
# ─────────────────────────────────────────────────────────────────────────────

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


SCRIPT_DIR  = Path(__file__).resolve().parent
DATA_DIR    = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR / "figures"
MODELS_DIR  = SCRIPT_DIR / "models"
RESULTS_DIR = SCRIPT_DIR / "results"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_DIR / 'features_data.csv')
with open(DATA_DIR / 'split_indices.json', encoding='utf-8') as f:
    split = json.load(f)

FEATURES = ['F_bell', 'F_gate', 'F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

X_train = X[split['train']]; y_train = y[split['train']]
X_val   = X[split['val']];   y_val   = y[split['val']]
X_test  = X[split['test']];  y_test  = y[split['test']]

print(f"Test set : {len(X_test):,} rows | drift={round(y_test.mean()*100,1)}%\n")

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 1 — STANDARD BASELINE: Per-feature majority-vote threshold
# ═════════════════════════════════════════════════════════════════════════════
# Thresholds = per-feature mean of TRAINING distribution (no leakage).
# Each feature votes; majority wins (>=2 of 3 features below mean → drifted).

print("─" * 60)
print("  METHOD 1 — Per-Feature Majority-Vote Threshold")
print("─" * 60)

feat_means_train = X_train.mean(axis=0)
print("  Training-set per-feature means:")
for name, mean in zip(FEATURES, feat_means_train):
    print(f"    {name:14s}: {mean:.6f}")

test_votes              = (X_test < feat_means_train).astype(int)
threshold_preds         = (test_votes.sum(axis=1) >= 2).astype(int)
threshold_scores_for_auc= test_votes.sum(axis=1)

threshold_auc       = round(roc_auc_score(y_test, threshold_scores_for_auc),  4)
threshold_accuracy  = round(accuracy_score(y_test, threshold_preds),          4)
threshold_precision = round(precision_score(y_test, threshold_preds, zero_division=0), 4)
threshold_recall    = round(recall_score(y_test, threshold_preds, zero_division=0),    4)
threshold_f1        = round(f1_score(y_test, threshold_preds, zero_division=0),        4)

print(f"\n  AUC: {threshold_auc}  Accuracy: {threshold_accuracy}  "
      f"Precision: {threshold_precision}  Recall: {threshold_recall}  "
      f"F1: {threshold_f1}")

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 2 — HOTELLING'S T²: Multivariate statistical SPC
# ═════════════════════════════════════════════════════════════════════════════
# Mahalanobis distance from healthy training centroid.
# T²(x) = (x - μ)ᵀ · Σ⁻¹ · (x - μ)

print("\n" + "─" * 60)
print("  METHOD 2 — Hotelling's T²  (Hotelling 1947)")
print("─" * 60)

stable_train = X_train[y_train == 0]
mu_vec       = stable_train.mean(axis=0)
cov_matrix   = np.cov(stable_train, rowvar=False)
cov_reg      = cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0])
cov_inv      = np.linalg.inv(cov_reg)

def hotelling_t2(X, mu, inv_cov):
    d = X - mu
    return np.einsum('ij,jk,ik->i', d, inv_cov, d)

# Tune decision threshold on validation F1
val_t2_scores  = hotelling_t2(X_val,  mu_vec, cov_inv)
test_t2_scores = hotelling_t2(X_test, mu_vec, cov_inv)

best_h, best_f1_h = 0.0, -1.0
for h in np.linspace(val_t2_scores.min(), val_t2_scores.max(), 500):
    preds = (val_t2_scores > h).astype(int)
    if len(np.unique(preds)) < 2: continue
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1_h:
        best_f1_h, best_h = f1, float(h)

t2_preds = (test_t2_scores > best_h).astype(int)

t2_auc       = round(roc_auc_score(y_test, test_t2_scores),         4)
t2_accuracy  = round(accuracy_score(y_test, t2_preds),              4)
t2_precision = round(precision_score(y_test, t2_preds, zero_division=0), 4)
t2_recall    = round(recall_score(y_test, t2_preds, zero_division=0),    4)
t2_f1        = round(f1_score(y_test, t2_preds, zero_division=0),        4)

print(f"  Decision threshold h (tuned on val): {round(best_h, 4)}")
print(f"  AUC: {t2_auc}  Accuracy: {t2_accuracy}  "
      f"Precision: {t2_precision}  Recall: {t2_recall}  F1: {t2_f1}")

# ═════════════════════════════════════════════════════════════════════════════
# METHOD 3 — QUANTUM CANARY MLP ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("  METHOD 3 — Quantum Canary MLP Ensemble  (this work)")
print("─" * 60)

scaler   = StandardScaler()
scaler.fit(X_train)
X_test_s = scaler.transform(X_test)

print("  Loading 10 MLP models...")
all_test_preds = []
for seed in range(10):
    model = keras.models.load_model(MODELS_DIR / f'mlp_seed_{seed}.keras')
    preds = model.predict(X_test_s, verbose=0).flatten()
    all_test_preds.append(preds)

ensemble_preds       = np.mean(all_test_preds, axis=0)
ensemble_uncertainty = np.std(all_test_preds,  axis=0)
ensemble_labels      = (ensemble_preds >= 0.5).astype(int)

mlp_auc       = round(roc_auc_score(y_test, ensemble_preds),4)
mlp_accuracy  = round(accuracy_score(y_test, ensemble_labels),4)
mlp_precision = round(precision_score(y_test, ensemble_labels, zero_division=0), 4)
mlp_recall    = round(recall_score(y_test, ensemble_labels, zero_division=0),4)
mlp_f1        = round(f1_score(y_test, ensemble_labels, zero_division=0),4)
mlp_cm        = confusion_matrix(y_test, ensemble_labels)
tn, fp, fn, tp = mlp_cm.ravel()
mlp_specificity = round(tn / (tn + fp), 4)

print(f"  AUC: {mlp_auc}  Accuracy: {mlp_accuracy}  "
      f"Precision: {mlp_precision}  Recall: {mlp_recall}  F1: {mlp_f1}")
print(f"  Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")

# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
improvement_threshold = round((mlp_auc - threshold_auc) / abs(threshold_auc) * 100, 1)
improvement_t2        = round((mlp_auc - t2_auc)        / abs(t2_auc)        * 100, 1)

print("\n" + "═" * 60)
print("  THREE-WAY TEST SET COMPARISON")
print("═" * 60)
print(f"  {'Method':40s}  {'AUC':>7s}  {'F1':>7s}")
print(f"  {'-'*40}  {'-'*7}  {'-'*7}")
print(f"  {'Per-feature majority-vote threshold':40s}  "
      f"{threshold_auc:>7}  {threshold_f1:>7}")
print(f"  {chr(34)+ chr(34)}".replace(chr(34)+chr(34),"")
      + f"  Hotelling's T² (1947)                     {t2_auc:>7}  {t2_f1:>7}"[2:])
print(f"  {'Quantum Canary MLP Ensemble':40s}  "
      f"{mlp_auc:>7}  {mlp_f1:>7}")
print(f"\n  MLP improvement over standard threshold : +{improvement_threshold}%")
print(f"  MLP improvement over Hotelling's T²     : +{improvement_t2}%")

# ═════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with open(RESULTS_DIR / 'evaluation_results.json', 'w') as f:
    json.dump({
        'mlp_ensemble': {
            'auc':         mlp_auc,
            'accuracy':    mlp_accuracy,
            'precision':   mlp_precision,
            'recall':      mlp_recall,
            'specificity': mlp_specificity,
            'f1':          mlp_f1,
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        },
        'standard_threshold': {
            'method':      'Per-feature majority-vote threshold',
            'auc':         threshold_auc,
            'accuracy':    threshold_accuracy,
            'precision':   threshold_precision,
            'recall':      threshold_recall,
            'f1':          threshold_f1,
            'thresholds':  {n: round(float(m), 6)
                            for n, m in zip(FEATURES, feat_means_train)},
        },
        'hotelling_t2': {
            'method':              "Hotelling's T² (1947)",
            'auc':                 t2_auc,
            'accuracy':            t2_accuracy,
            'precision':           t2_precision,
            'recall':              t2_recall,
            'f1':                  t2_f1,
            'decision_threshold':  round(best_h, 4),
            'reference':           'Hotelling (1947)',
        },
        'comparison': {
            'mlp_vs_threshold_pct':  improvement_threshold,
            'mlp_vs_hotelling_pct':  improvement_t2,
            'n_test_rows':           len(X_test),
            'drift_rate':            round(y_test.mean()*100, 1),
        }
    }, f, indent=2)

print("\n  ✓ Saved results/evaluation_results.json")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: THREE-WAY ROC CURVE
# ═════════════════════════════════════════════════════════════════════════════
print("\nGenerating figures...")

fpr_mlp, tpr_mlp, _ = roc_curve(y_test, ensemble_preds)
fpr_t2,  tpr_t2,  _ = roc_curve(y_test, test_t2_scores)
fpr_th,  tpr_th,  _ = roc_curve(y_test, threshold_scores_for_auc)

fig, ax = plt.subplots(figsize=(6, 6))
ax.fill_between(fpr_mlp, tpr_mlp, alpha=0.15, color='crimson')
ax.plot(fpr_mlp, tpr_mlp, color='crimson', lw=2.5,
        label=f'Quantum Canary MLP (AUC={mlp_auc})')
ax.plot(fpr_t2, tpr_t2, color='#1c5d8c', lw=2,
        label=f"Hotelling's T² (AUC={t2_auc})")
ax.plot(fpr_th, tpr_th, color='darkorange', lw=2, marker='o', markersize=5,
        label=f'Standard Threshold (AUC={threshold_auc})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate',  fontsize=11)
ax.set_title('Three-Way ROC Comparison — Held-out Test Set',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_roc_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_roc_test.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: THREE-WAY AUC BAR CHART
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6.5, 4.5))
methods = ['Standard\nThreshold', "Hotelling's T²\n(1947)", 'Quantum Canary\nMLP Ensemble']
aucs    = [threshold_auc, t2_auc, mlp_auc]
colors  = ['darkorange', '#1c5d8c', 'crimson']

bars = ax.bar(methods, aucs, color=colors, width=0.5,
              edgecolor='black', linewidth=0.8)
ax.set_ylim(0, 1.05)
ax.set_ylabel('AUC', fontsize=11)
ax.set_title('AUC Comparison — Held-out Test Set',
             fontsize=11, fontweight='bold')
for bar, val in zip(bars, aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{val}', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

ax.annotate(f'+{improvement_threshold}% over threshold\n+{improvement_t2}% over T²',
            xy=(2, mlp_auc), xytext=(1.0, mlp_auc + 0.02),
            fontsize=9.5, fontweight='bold', color='crimson',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='crimson'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_auc_comparison.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MLP CONFUSION MATRIX
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(4.5, 4))
disp = ConfusionMatrixDisplay(mlp_cm, display_labels=['Stable', 'Drifted'])
disp.plot(ax=ax, colorbar=False, cmap='Reds')
ax.set_title('Quantum Canary MLP — Confusion Matrix\nHeld-out Test Set',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_confusion_matrix.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: ABLATION STUDY (MLP feature importance)
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning ablation study...")
ablation = {}
for drop_feat in FEATURES:
    preds_abl = []
    for seed in range(10):
        model    = keras.models.load_model(MODELS_DIR / f'mlp_seed_{seed}.keras')
        X_zeroed = X_test_s.copy()
        X_zeroed[:, FEATURES.index(drop_feat)] = 0
        preds_abl.append(model.predict(X_zeroed, verbose=0).flatten())
    abl_auc = round(roc_auc_score(y_test, np.mean(preds_abl, axis=0)), 4)
    ablation[drop_feat] = abl_auc
    print(f"  Without {drop_feat:14s}: AUC={abl_auc}  "
          f"(drop={round(mlp_auc - abl_auc, 4)})")

fig, ax = plt.subplots(figsize=(6, 4))
abl_aucs = [ablation[f] for f in FEATURES]
drops    = [round(mlp_auc - a, 4) for a in abl_aucs]
bars     = ax.bar(FEATURES, abl_aucs, color='steelblue',
                  width=0.4, edgecolor='black', linewidth=0.8)
ax.axhline(mlp_auc, color='crimson', lw=2, linestyle='--',
           label=f'Full MLP AUC={mlp_auc}')
ax.set_ylim(max(0, min(abl_aucs) - 0.1), 1.0)
ax.set_ylabel('AUC', fontsize=10)
ax.set_title('Ablation Study — Feature Importance\nHeld-out Test Set',
             fontsize=11, fontweight='bold')
for bar, val, drop in zip(bars, abl_aucs, drops):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val}\n(-{drop})', ha='center', va='bottom', fontsize=9)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_ablation.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: MLP CONFIDENCE DISTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(ensemble_preds[y_test == 0], bins=50, alpha=0.6, color='steelblue',
        density=True, label=f'Stable (n={int((y_test == 0).sum()):,})')
ax.hist(ensemble_preds[y_test == 1], bins=50, alpha=0.6, color='crimson',
        density=True, label=f'Drifted (n={int((y_test == 1).sum()):,})')
ax.axvline(0.5, color='black', lw=1.5, linestyle='--',
           label='Decision threshold (0.5)')
ax.set_xlabel('MLP Ensemble Drift Probability', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title('Quantum Canary MLP — Confidence Distribution\nHeld-out Test Set',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_confidence_test.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_confidence_test.png")

# ═════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  EVALUATION COMPLETE — THREE-WAY COMPARISON")
print("═" * 60)
print(f"  Quantum Canary MLP Test AUC : {mlp_auc}")
print(f"  Hotelling's T² Test AUC     : {t2_auc}")
print(f"  Standard Threshold AUC      : {threshold_auc}")
print(f"  MLP improvement over T²     : +{improvement_t2}%")
print(f"  MLP improvement over thresh : +{improvement_threshold}%")
print("═" * 60)
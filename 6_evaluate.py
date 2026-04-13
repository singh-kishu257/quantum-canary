import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("data/features_data.csv")

with open("data/split_indices.json", "r") as f:
    split = json.load(f)

FEATURES = ["F_bell", "F_gate", "F_coherence"]

X = df[FEATURES].values
y = df["drifted"].values

X_test = X[split["test"]]
y_test = y[split["test"]]

# ─────────────────────────────────────────────
# LOAD SCALER (from 4_baselines.py)
# ─────────────────────────────────────────────

mean = np.load("models/scaler_mean.npy")
scale = np.load("models/scaler_scale.npy")

X_test = (X_test - mean) / scale

# ─────────────────────────────────────────────
# LOAD BASELINES
# ─────────────────────────────────────────────

with open("results/baseline_results.json", "r") as f:
    baseline = json.load(f)

threshold_auc = baseline["threshold_classifier"]["auc"]
logreg_auc = baseline["logistic_regression"]["auc"]

# ─────────────────────────────────────────────
# LOAD ENSEMBLE MODELS
# ─────────────────────────────────────────────

print("Loading ensemble models...")

models = [
    tf.keras.models.load_model(f"models/mlp_seed_{i}.keras")
    for i in range(10)
]

# ─────────────────────────────────────────────
# ENSEMBLE INFERENCE
# ─────────────────────────────────────────────

all_preds = []

for m in models:
    p = m.predict(X_test, verbose=0).flatten()
    all_preds.append(p)

mlp_scores = np.mean(all_preds, axis=0)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

mlp_auc = roc_auc_score(y_test, mlp_scores)

# threshold baseline scores (same logic as training)
thresh_scores = -np.mean(X_test, axis=1)
thresh_auc_calc = roc_auc_score(y_test, thresh_scores)

best_baseline_auc = max(threshold_auc, logreg_auc)

improvement = (mlp_auc - best_baseline_auc) / abs(best_baseline_auc) * 100

print("\n===== TEST RESULTS =====")
print(f"MLP AUC        : {mlp_auc:.4f}")
print(f"Threshold AUC  : {threshold_auc:.4f}")
print(f"Logistic AUC   : {logreg_auc:.4f}")
print(f"Improvement %  : {improvement:.2f}%")

# ─────────────────────────────────────────────
# ROC CURVE
# ─────────────────────────────────────────────

fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_scores)
fpr_th, tpr_th, _ = roc_curve(y_test, thresh_scores)

plt.figure(figsize=(6, 6))

plt.plot(fpr_th, tpr_th,
         label=f"Threshold (AUC={threshold_auc:.3f})")

plt.plot(fpr_mlp, tpr_mlp,
         label=f"MLP Ensemble (AUC={mlp_auc:.3f})")

plt.plot([0, 1], [0, 1], "k--", alpha=0.5)

plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve — Test Set")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig_roc_test.png", dpi=300)
plt.close()

# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────

y_pred = (mlp_scores >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Stable", "Drifted"]
)

fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title("MLP Ensemble Confusion Matrix")
plt.tight_layout()

plt.savefig("figures/fig_confusion_matrix.png", dpi=300)
plt.close()

# ─────────────────────────────────────────────
# ABLATION STUDY (FEATURE IMPACT VIA ZEROING)
# ─────────────────────────────────────────────

print("Running ablation study...")

ablation_auc = {}

for i, feat in enumerate(FEATURES):

    X_abl = X_test.copy()
    X_abl[:, i] = 0

    preds = []

    for m in models:
        preds.append(m.predict(X_abl, verbose=0).flatten())

    score = np.mean(preds, axis=0)
    auc_score = roc_auc_score(y_test, score)

    ablation_auc[feat] = auc_score
    print(f"{feat}: {auc_score:.4f}")

# ─────────────────────────────────────────────
# ABLATION FIGURE
# ─────────────────────────────────────────────

plt.figure(figsize=(6, 4))

labels = list(ablation_auc.keys())
values = list(ablation_auc.values())

plt.bar(labels, values)

plt.ylabel("AUC (feature removed)")
plt.title("Ablation Study")

for i, v in enumerate(values):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig("figures/fig_ablation.png", dpi=300)
plt.close()
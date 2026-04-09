# 5_train_mlp.py — Quantum Canary Prototype 2
# Trains a 10-model MLP ensemble on F_bell, F_gate, F_coherence.
# Architecture: 3 → 16 → 8 → 1 (sigmoid), 10 seeds, early stopping.

import pandas as pd
import numpy as np
import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import tensorflow as tf
from tensorflow import keras

os.makedirs('models', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

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

# ── 2. SCALE ──────────────────────────────────────────────────────────────────
scaler_mean  = np.load('models/scaler_mean.npy')
scaler_scale = np.load('models/scaler_scale.npy')
X_train = (X_train - scaler_mean) / scaler_scale
X_val   = (X_val   - scaler_mean) / scaler_scale
X_test  = (X_test  - scaler_mean) / scaler_scale

# ── 3. CLASS WEIGHT ───────────────────────────────────────────────────────────
n_total  = len(y_train)
n_drift  = y_train.sum()
n_stable = n_total - n_drift
class_weight = {0: n_total / (2 * n_stable), 1: n_total / (2 * n_drift)}
print(f"Class weights: stable={class_weight[0]:.3f}, drift={class_weight[1]:.3f}\n")

# ── 4. BUILD MLP ──────────────────────────────────────────────────────────────
def build_mlp(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1,  activation='sigmoid'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005),
                  loss='binary_crossentropy',
                  metrics=['accuracy', keras.metrics.AUC(name='auc')])
    return model

# ── 5. TRAIN 10 SEEDS ─────────────────────────────────────────────────────────
val_aucs       = []
all_val_preds  = []
all_train_acc  = []
all_val_acc    = []
all_train_loss = []
all_val_loss   = []

print("Training 10 MLP models...\n")
for seed in range(10):
    model = build_mlp(seed)
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=25,
        restore_best_weights=True, verbose=0)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=300, batch_size=64,
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0
    )

    val_preds = model.predict(X_val, verbose=0).flatten()
    val_auc   = roc_auc_score(y_val, val_preds)
    val_aucs.append(val_auc)
    all_val_preds.append(val_preds)

    all_train_acc.append(history.history['accuracy'])
    all_val_acc.append(history.history['val_accuracy'])
    all_train_loss.append(history.history['loss'])
    all_val_loss.append(history.history['val_loss'])

    model.save(f'models/mlp_seed_{seed}.keras')
    print(f"  Seed {seed}: Val AUC={round(val_auc,4)} | Epochs={len(history.history['loss'])}")

# ── 6. ENSEMBLE ───────────────────────────────────────────────────────────────
ensemble_val_preds = np.mean(all_val_preds, axis=0)
ensemble_val_auc   = round(roc_auc_score(y_val, ensemble_val_preds), 4)
print(f"\nEnsemble Val AUC : {ensemble_val_auc}")
print(f"Mean seed AUC    : {round(np.mean(val_aucs), 4)}")
print(f"Std seed AUC     : {round(np.std(val_aucs),  4)}")

# Load baseline
with open('results/baseline_results.json') as f:
    baseline = json.load(f)
thresh_auc  = baseline['threshold_classifier']['auc']
improvement = round((ensemble_val_auc - thresh_auc) / abs(thresh_auc) * 100, 1)
print(f"Threshold AUC    : {thresh_auc}")
print(f"Improvement      : {improvement}%")

# ── 7. FIGURE 1: Val Loss all 10 seeds overlaid on one plot ──────────────────
print("\nGenerating figures...")
fig, ax = plt.subplots(figsize=(8, 5))
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(10):
    ax.plot(all_val_loss[i], color=colors[i], lw=1.4, alpha=0.8,
            label=f'Seed {i} (AUC={round(val_aucs[i],3)})')
ax.set_xlabel('Epoch', fontsize=10)
ax.set_ylabel('Validation Loss', fontsize=10)
ax.set_title('MLP Validation Loss Over Epochs — All 10 Seeds',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='upper right', ncol=2)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_train_val_loss.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_train_val_loss.png")

# ── 8. FIGURE 2: ROC curves ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
colors = plt.cm.Blues(np.linspace(0.4, 0.9, 10))
for i, preds in enumerate(all_val_preds):
    fpr, tpr, _ = roc_curve(y_val, preds)
    ax.plot(fpr, tpr, color=colors[i], lw=1, alpha=0.6,
            label=f'Seed {i} (AUC={round(val_aucs[i],3)})')
fpr_ens, tpr_ens, _ = roc_curve(y_val, ensemble_val_preds)
ax.plot(fpr_ens, tpr_ens, color='crimson', lw=2.5,
        label=f'Ensemble (AUC={ensemble_val_auc})')
ax.plot([0,1],[0,1],'k--', lw=1, label='Random')
ax.set_xlabel('False Positive Rate', fontsize=10)
ax.set_ylabel('True Positive Rate', fontsize=10)
ax.set_title('ROC Curves — All 10 Seeds + Ensemble\nValidation Set', fontsize=11, fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/fig_roc_val.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ figures/fig_roc_val.png")

# Save ensemble val predictions for evaluation script
np.save('results/ensemble_val_preds.npy', ensemble_val_preds)
np.save('results/val_aucs.npy', np.array(val_aucs))

print(f"\n{'='*45}")
print(f"  TRAINING COMPLETE")
print(f"  Ensemble Val AUC : {ensemble_val_auc}")
print(f"  Threshold AUC    : {thresh_auc}")
print(f"  Improvement      : {improvement}%")
print(f"{'='*45}")
print(f"\n  NEXT: python 6_evaluate.py")
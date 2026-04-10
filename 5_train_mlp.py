# 5_train_mlp_optimized.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# Load the fixed data
df = pd.read_csv('data/features_data_fixed.csv')
FEATURES = ['z_F_bell', 'z_F_gate', 'z_F_coherence']
X = df[FEATURES].values
y = df['drifted'].values

# Split and Scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Calculate Class Weights to handle imbalance
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: (neg / pos) * 1.5} # Extra emphasis on detecting drift

# Optimized Architecture for 3-feature input
model = tf.keras.Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(12, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='auc')]
)

# Train with Early Stopping
es = callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=15, restore_best_weights=True)

print("Training MLP...")
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=150,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[es],
    verbose=0
)

# Evaluate
y_pred = model.predict(X_test)
final_auc = roc_auc_score(y_test, y_pred)
print(f"\nTarget AUC: > 0.90")
print(f"Resulting AUC: {final_auc:.4f}")

model.save('models/canary_mlp_v2.h5')
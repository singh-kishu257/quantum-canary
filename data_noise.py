import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/ibm_calibration_labeled.csv')
X = df[['F_bell', 'F_gate']].values
y = df['drifted'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

mlp = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
auc = roc_auc_score(y_test, mlp.predict_proba(X_test)[:,1])
print(f"AUC: {auc:.4f}")
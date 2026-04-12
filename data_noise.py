import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

df    = pd.read_csv('data/features_data.csv')
y     = df['drifted'].values
idx   = np.arange(len(df))

train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx], random_state=42)

split = {
    "train": sorted(train_idx.tolist()),
    "val":   sorted(val_idx.tolist()),
    "test":  sorted(test_idx.tolist())
}
with open('data/split_indices.json', 'w') as f:
    json.dump(split, f)

print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}")
print(f"Dataset size: {len(df):,}")
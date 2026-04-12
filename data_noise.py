import pandas as pd
df = pd.read_csv('data/features_data.csv')
print(df.drop(columns='drifted').corr().round(2))
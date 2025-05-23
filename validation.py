import pandas as pd
df = pd.read_parquet('data/monaco/monaco_2023.parquet')
df2 = pd.read_parquet('data/monaco/monaco_2024.parquet')
print (df.head())
print (df2.head())
import pandas as pd

FILE = "data/twcs.csv"

print("Loading dataset...")

df = pd.read_csv(FILE)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nInbound distribution:")
print(df["inbound"].value_counts())

print("\nNumber of unique authors:")
print(df["author_id"].nunique())

print("\nTop authors:")
print(df["author_id"].value_counts().head(30))
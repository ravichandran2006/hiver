import pandas as pd

INPUT_FILE = "data/amazon_pairs.csv"

TRAIN_FILE = "data/intent_train.csv"
GOLD_FILE = "data/intent_gold.csv"

df = pd.read_csv(INPUT_FILE)

df = df.drop_duplicates(subset=["customer_text"])
df = df.reset_index(drop=True)

gold = df.sample(n=200, random_state=42)

remaining = df.drop(gold.index)

train = remaining.sample(n=800, random_state=42)

train = train.reset_index(drop=True)
gold = gold.reset_index(drop=True)

train.to_csv(TRAIN_FILE, index=False)
gold.to_csv(GOLD_FILE, index=False)

print("Total dataset:", len(df))
print("Training examples:", len(train))
print("Gold examples:", len(gold))

print("\nTraining file:", TRAIN_FILE)
print("Gold file:", GOLD_FILE)
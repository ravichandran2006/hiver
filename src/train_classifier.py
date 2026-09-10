import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from joblib import dump

TRAIN_FILE = "data/intent_train_labeled.csv"
MODEL_FILE = "results/intent_classifier.joblib"

df = pd.read_csv(TRAIN_FILE)

df = df.dropna(
    subset=["customer_text", "intent"]
)

print("Training examples:", len(df))

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

model.fit(
    df["customer_text"],
    df["intent"]
)

os.makedirs(
    "results",
    exist_ok=True
)

dump(
    model,
    MODEL_FILE
)

print("Training completed.")
print("Model saved:", MODEL_FILE)
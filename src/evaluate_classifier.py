import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

TRAIN_FILE = "data/intent_train_labeled.csv"

MODEL_FILE = "results/intent_classifier.joblib"

RESULT_FILE = "results/intent_evaluation.csv"

INTENTS = [
    "DELIVERY_TRACKING",
    "ORDER_MANAGEMENT",
    "REFUND_RETURN",
    "PAYMENT_BILLING",
    "ACCOUNT_SECURITY",
    "PRODUCT_TECHNICAL",
    "PRIME_DIGITAL",
    "GENERAL_COMPLAINT"
]


df = pd.read_csv(TRAIN_FILE)

df = df.dropna(
    subset=["customer_text", "intent"]
)

df = df[
    df["intent"].isin(INTENTS)
]

print("Total labeled examples:", len(df))


X = df["customer_text"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training examples:", len(X_train))
print("Test examples:", len(X_test))


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
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)


print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Macro-F1: {macro_f1:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=INTENTS,
        digits=4,
        zero_division=0
    )
)


matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=INTENTS
)

matrix_df = pd.DataFrame(
    matrix,
    index=INTENTS,
    columns=INTENTS
)


print("\nConfusion Matrix:")
print(matrix_df)


results = pd.DataFrame({
    "customer_text": X_test.values,
    "true_intent": y_test.values,
    "predicted_intent": y_pred
})


os.makedirs(
    "results",
    exist_ok=True
)


results.to_csv(
    RESULT_FILE,
    index=False
)


matrix_df.to_csv(
    "results/confusion_matrix.csv"
)


with open(
    "results/metrics.txt",
    "w"
) as f:

    f.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    f.write(
        f"Macro-F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Training examples: {len(X_train)}\n"
    )

    f.write(
        f"Test examples: {len(X_test)}\n"
    )


print("\nResults saved:")
print(RESULT_FILE)
print("results/confusion_matrix.csv")
print("results/metrics.txt")
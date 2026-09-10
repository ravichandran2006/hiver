import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


DATA_PATH = "data/intent_train_labeled.csv"


df = pd.read_csv(DATA_PATH)

df = df.dropna(
    subset=["customer_text", "intent"]
)

valid_intents = [
    "DELIVERY_TRACKING",
    "ORDER_MANAGEMENT",
    "REFUND_RETURN",
    "PAYMENT_BILLING",
    "ACCOUNT_SECURITY",
    "PRODUCT_TECHNICAL",
    "PRIME_DIGITAL",
    "GENERAL_COMPLAINT"
]

df = df[df["intent"].isin(valid_intents)]

print(f"Total labeled examples: {len(df)}")


X = df["customer_text"]
y = df["intent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")


# ---------------------------------------------------------
# BASELINE 1: MAJORITY CLASS
# ---------------------------------------------------------

majority_class = y_train.value_counts().idxmax()

baseline_predictions = [
    majority_class
] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("BASELINE: MAJORITY CLASS")
print("=" * 60)

print(f"Majority intent: {majority_class}")
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"Macro-F1: {baseline_f1:.4f}")


# ---------------------------------------------------------
# OUR MODEL
# ---------------------------------------------------------

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


model.fit(X_train, y_train)

predictions = model.predict(X_test)


model_accuracy = accuracy_score(
    y_test,
    predictions
)

model_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)


print("\n" + "=" * 60)
print("OUR MODEL")
print("=" * 60)

print(f"Accuracy: {model_accuracy:.4f}")
print(f"Macro-F1: {model_f1:.4f}")


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)


# ---------------------------------------------------------
# COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BASELINE vs OUR MODEL")
print("=" * 60)

print(
    f"Baseline Accuracy : {baseline_accuracy:.4f}"
)

print(
    f"Model Accuracy    : {model_accuracy:.4f}"
)

print(
    f"Baseline Macro-F1 : {baseline_f1:.4f}"
)

print(
    f"Model Macro-F1    : {model_f1:.4f}"
)

print(
    f"\nAccuracy improvement: "
    f"{model_accuracy - baseline_accuracy:.4f}"
)

print(
    f"Macro-F1 improvement: "
    f"{model_f1 - baseline_f1:.4f}"
)
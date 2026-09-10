import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = "data/amazon_pairs.csv"


class NearestNeighborBaseline:

    def __init__(self):

        print("Loading Amazon conversations...")

        self.df = pd.read_csv(DATA_PATH)

        self.df = self.df.dropna(
            subset=["customer_text", "brand_response"]
        ).reset_index(drop=True)

        print(f"Loaded {len(self.df)} conversations.")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )

        print("Building TF-IDF index...")

        self.vectors = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

        print("Baseline ready.")

    def generate_response(self, message):

        query_vector = self.vectorizer.transform([message])

        similarities = cosine_similarity(
            query_vector,
            self.vectors
        ).flatten()

        best_index = similarities.argmax()

        return {
            "customer_message": message,
            "response": self.df.iloc[best_index]["brand_response"],
            "matched_customer": self.df.iloc[best_index]["customer_text"],
            "similarity": float(similarities[best_index])
        }


if __name__ == "__main__":

    baseline = NearestNeighborBaseline()

    test_messages = [
        "My package says delivered but I never received it.",
        "I returned my order but I still haven't received my refund.",
        "I want to cancel my order before it ships.",
        "Someone hacked my Amazon account and changed my password.",
        "I was charged twice for the same order.",
        "My Kindle is not connecting to WiFi."
    ]

    print("\n" + "=" * 70)
    print("NEAREST-NEIGHBOR BASELINE")
    print("=" * 70)

    for i, message in enumerate(test_messages, 1):

        result = baseline.generate_response(message)

        print(f"\nTEST {i}")
        print("-" * 70)

        print("Customer:")
        print(message)

        print("\nMatched historical customer:")
        print(result["matched_customer"])

        print("\nHistorical Amazon response:")
        print(result["response"])

        print(
            f"\nSimilarity: {result['similarity']:.4f}"
        )
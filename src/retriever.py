import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HistoricalRetriever:

    def __init__(self, data_path="data/amazon_pairs.csv"):
        print("Loading Amazon historical conversations...")

        self.df = pd.read_csv(data_path)

        self.df = self.df.dropna(
            subset=["customer_text", "brand_response"]
        ).reset_index(drop=True)

        print(f"Historical conversations loaded: {len(self.df)}")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000
        )

        print("Building TF-IDF index...")

        self.customer_vectors = self.vectorizer.fit_transform(
            self.df["customer_text"]
        )

        print("Retriever ready.")

    def retrieve(self, query, top_k=3):

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.customer_vectors
        ).flatten()

        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []

        for index in top_indices:

            results.append({
                "customer_text": self.df.iloc[index]["customer_text"],
                "brand_response": self.df.iloc[index]["brand_response"],
                "similarity": float(similarities[index])
            })

        return results


if __name__ == "__main__":

    retriever = HistoricalRetriever()

    query = "My package says delivered but I never received it."

    results = retriever.retrieve(query, top_k=3)

    print("\n" + "=" * 70)
    print("TEST QUERY")
    print("=" * 70)

    print(query)

    print("\n" + "=" * 70)
    print("TOP HISTORICAL CASES")
    print("=" * 70)

    for i, result in enumerate(results, 1):

        print(f"\nCASE {i}")
        print("-" * 50)

        print("Customer:")
        print(result["customer_text"])

        print("\nAmazon Response:")
        print(result["brand_response"])

        print(f"\nSimilarity: {result['similarity']:.4f}")
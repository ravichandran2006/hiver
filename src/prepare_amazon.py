import pandas as pd
import re

INPUT_FILE = "data/twcs.csv"
OUTPUT_FILE = "data/amazon_pairs.csv"

BRAND = "AmazonHelp"
MAX_PAIRS = 10000
RANDOM_SEED = 42

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded.")
print("Total rows:", len(df))

# Find AmazonHelp responses
amazon_responses = df[
    (df["author_id"] == BRAND) &
    (df["inbound"] == False) &
    (df["in_response_to_tweet_id"].notna())
].copy()

print("AmazonHelp responses:", len(amazon_responses))

# Get customer tweets
customer_tweets = df[
    df["inbound"] == True
][["tweet_id", "author_id", "text"]].copy()

customer_tweets.columns = [
    "customer_tweet_id",
    "customer_id",
    "customer_text"
]

# Make IDs compatible
amazon_responses["in_response_to_tweet_id"] = (
    amazon_responses["in_response_to_tweet_id"].astype("Int64")
)

customer_tweets["customer_tweet_id"] = (
    customer_tweets["customer_tweet_id"].astype("Int64")
)

# Connect customer message with AmazonHelp response
pairs = amazon_responses.merge(
    customer_tweets,
    left_on="in_response_to_tweet_id",
    right_on="customer_tweet_id",
    how="inner"
)

pairs = pairs[
    [
        "customer_tweet_id",
        "customer_id",
        "customer_text",
        "tweet_id",
        "text"
    ]
]

pairs.columns = [
    "customer_tweet_id",
    "customer_id",
    "customer_text",
    "response_tweet_id",
    "brand_response"
]

print("Valid customer-response pairs:", len(pairs))


def clean_text(text):
    text = str(text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "[LINK]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


pairs["customer_text"] = pairs["customer_text"].apply(clean_text)
pairs["brand_response"] = pairs["brand_response"].apply(clean_text)

# Remove empty examples
pairs = pairs[
    (pairs["customer_text"].str.len() > 5) &
    (pairs["brand_response"].str.len() > 5)
]

# Remove duplicates
pairs = pairs.drop_duplicates(
    subset=["customer_text", "brand_response"]
)

print("Pairs after cleaning:", len(pairs))

# Take reproducible sample
if len(pairs) > MAX_PAIRS:
    pairs = pairs.sample(
        n=MAX_PAIRS,
        random_state=RANDOM_SEED
    )

pairs = pairs.reset_index(drop=True)

# Save
pairs.to_csv(
    OUTPUT_FILE,
    index=False
)

print("=" * 70)
print("AMAZONHELP DATA PREPARATION COMPLETE")
print("=" * 70)
print("Final number of pairs:", len(pairs))
print("Output file:", OUTPUT_FILE)

# Show sample conversations
print("\nSAMPLE CONVERSATIONS")
print("=" * 70)

for i in range(min(10, len(pairs))):

    print("\nCUSTOMER:")
    print(pairs.iloc[i]["customer_text"])

    print("\nAMAZONHELP:")
    print(pairs.iloc[i]["brand_response"])

    print("-" * 70)
import pandas as pd

FILE = "data/twcs.csv"

df = pd.read_csv(FILE)

# Known support/brand accounts from the dataset
brands = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "SouthwestAir",
    "VirginTrains",
    "Ask_Spectrum",
    "XboxSupport",
    "sprintcare",
    "hulu_support",
    "sainsburys",
    "GWRHelp",
    "AskPlayStation",
    "ChipotleTweets"
]

results = []

for brand in brands:

    brand_tweets = df[df["author_id"] == brand]

    # Brand responses
    brand_responses = brand_tweets[brand_tweets["inbound"] == False]

    # Customer messages directed to this brand
    customer_messages = df[
        (df["inbound"] == True) &
        (df["text"].str.contains("@" + brand, case=False, na=False))
    ]

    results.append({
        "brand": brand,
        "total_tweets": len(brand_tweets),
        "brand_responses": len(brand_responses),
        "customer_messages": len(customer_messages)
    })

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    by="customer_messages",
    ascending=False
)

print("\nBrand Analysis")
print("=" * 70)
print(result_df.to_string(index=False))
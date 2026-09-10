import pandas as pd


INPUT_PATH = "data/intent_gold.csv"
OUTPUT_PATH = "data/intent_gold_labeled.csv"


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


def show_intents():

    print("\nAvailable intents:")

    for i, intent in enumerate(INTENTS, 1):
        print(f"{i}. {intent}")


def main():

    df = pd.read_csv(INPUT_PATH)

    if "intent" not in df.columns:
        df["intent"] = ""

    # Resume previous labeling if the output already exists
    try:
        existing = pd.read_csv(OUTPUT_PATH)

        if len(existing) == len(df):
            df = existing
            print("Resuming previous labeling session.")

    except FileNotFoundError:
        pass

    print("=" * 70)
    print("HUMAN GOLD SET LABELING")
    print("=" * 70)

    print(f"Total examples: {len(df)}")

    print("\nType:")
    print("1-8  = select intent")
    print("s    = skip")
    print("q    = save and quit")

    for index in range(len(df)):

        current_label = str(df.loc[index, "intent"])

        if current_label in INTENTS:
            continue

        message = str(
            df.loc[index, "customer_text"]
        )

        print("\n" + "=" * 70)
        print(f"Example {index + 1}/{len(df)}")
        print("=" * 70)

        print("\nCustomer message:")
        print(message)

        show_intents()

        while True:

            choice = input("\nYour label: ").strip().lower()

            if choice == "q":

                df.to_csv(
                    OUTPUT_PATH,
                    index=False
                )

                print(
                    f"\nProgress saved to {OUTPUT_PATH}"
                )

                return

            if choice == "s":

                print("Skipped.")
                break

            if choice.isdigit():

                number = int(choice)

                if 1 <= number <= len(INTENTS):

                    df.loc[index, "intent"] = INTENTS[number - 1]

                    print(
                        f"Labeled as: {INTENTS[number - 1]}"
                    )

                    break

            print(
                "Invalid input. Enter 1-8, s, or q."
            )

        # Save after every example
        df.to_csv(
            OUTPUT_PATH,
            index=False
        )

    print("\n" + "=" * 70)
    print("LABELING COMPLETE")
    print("=" * 70)

    labeled = df[
        df["intent"].isin(INTENTS)
    ]

    print(
        f"Labeled examples: {len(labeled)}"
    )

    print("\nIntent distribution:")

    print(
        labeled["intent"].value_counts()
    )

    print(
        f"\nSaved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
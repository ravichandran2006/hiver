import os
import json
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from agent import generate_response


load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


DATA_PATH = "data/intent_train_labeled.csv"
OUTPUT_PATH = "results/response_evaluation.csv"

SAMPLE_SIZE = 20


def judge_response(customer_message, reply, historical_cases):

    evidence = ""

    for i, case in enumerate(historical_cases, 1):
        evidence += f"""
CASE {i}

Customer:
{case["customer_text"]}

Historical Amazon response:
{case["brand_response"]}
"""

    prompt = f"""
You are evaluating an AI customer-support response.

Customer message:
{customer_message}

AI-generated response:
{reply}

Historical evidence used by the AI:
{evidence}

Score the AI response from 1 to 5 on these three dimensions.

RELEVANCE:
Does the response directly address the customer's problem?

GROUNDEDNESS:
Is the response supported by the historical Amazon examples?
Does it avoid unsupported claims?

USEFULNESS:
Would this response provide a useful next step to the customer?

Scoring:

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent

Return ONLY a JSON object with:

{
    "relevance": integer,
    "groundedness": integer,
    "usefulness": integer,
    "reason": "short explanation"
}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1,

        max_completion_tokens=800,

        reasoning_effort="low",

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "response_evaluation",
                "strict": True,
                "schema": {
                    "type": "object",

                    "properties": {
                        "relevance": {
                            "type": "integer"
                        },
                        "groundedness": {
                            "type": "integer"
                        },
                        "usefulness": {
                            "type": "integer"
                        },
                        "reason": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "relevance",
                        "groundedness",
                        "usefulness",
                        "reason"
                    ],

                    "additionalProperties": False
                }
            }
        }
    )

    return json.loads(
        response.choices[0].message.content
    )


def main():

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=["customer_text"]
    )

    # Use a fixed sample for reproducibility
    sample = df.sample(
        n=SAMPLE_SIZE,
        random_state=42
    )

    results = []

    print("=" * 70)
    print("RESPONSE QUALITY EVALUATION")
    print("=" * 70)

    for i, (_, row) in enumerate(sample.iterrows(), 1):

        customer_message = row["customer_text"]

        print(f"\nEvaluating {i}/{SAMPLE_SIZE}")
        print(f"Customer: {customer_message}")

        try:

            agent_result = generate_response(
                customer_message
            )

            judgment = judge_response(
                customer_message,
                agent_result["reply"],
                agent_result["historical_cases"]
            )

            result = {
                "customer_message": customer_message,
                "intent": agent_result["intent"],
                "confidence": agent_result["confidence"],
                "reply": agent_result["reply"],
                "escalate": agent_result["escalate"],
                "reason": agent_result["reason"],
                "relevance": judgment["relevance"],
                "groundedness": judgment["groundedness"],
                "usefulness": judgment["usefulness"],
                "judge_reason": judgment["reason"]
            }

            results.append(result)

            print(
                f"Scores: "
                f"R={judgment['relevance']} "
                f"G={judgment['groundedness']} "
                f"U={judgment['usefulness']}"
            )

        except Exception as e:

            print(f"ERROR: {e}")

    if not results:
        print("\nNo evaluation results were produced.")
        return

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Examples evaluated: {len(results_df)}"
    )

    print(
        f"Average Relevance: "
        f"{results_df['relevance'].mean():.2f}"
    )

    print(
        f"Average Groundedness: "
        f"{results_df['groundedness'].mean():.2f}"
    )

    print(
        f"Average Usefulness: "
        f"{results_df['usefulness'].mean():.2f}"
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
import os
import json
import joblib
from dotenv import load_dotenv
from groq import Groq

from retriever import HistoricalRetriever


load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

classifier = joblib.load(
    "results/intent_classifier.joblib"
)

retriever = HistoricalRetriever(
    "data/amazon_pairs.csv"
)


def classify_intent(message):

    intent = classifier.predict([message])[0]

    probabilities = classifier.predict_proba([message])[0]

    confidence = max(probabilities)

    return intent, confidence


def generate_response(message):

    intent, confidence = classify_intent(message)

    historical_cases = retriever.retrieve(
        message,
        top_k=3
    )

    evidence = ""

    for i, case in enumerate(historical_cases, 1):

        evidence += f"""
CASE {i}

Customer:
{case["customer_text"]}

Amazon response:
{case["brand_response"]}

Similarity:
{case["similarity"]:.4f}
"""

    prompt = f"""
You are an AI customer support assistant for Amazon.

Customer message:
{message}

Predicted intent:
{intent}

Classifier confidence:
{confidence:.4f}

Historical Amazon support examples:

{evidence}

Task:

Write a concise customer support reply based on the historical
Amazon responses.

Do not invent policies, refunds, compensation, delivery promises,
or actions.

Do not claim that you personally performed an action.

Escalate the conversation if the historical examples are not
sufficient to safely answer the customer.

Otherwise, do not escalate.

Return the answer using the required JSON schema.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2,

        max_completion_tokens=1000,
        reasoning_effort="low",

        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "support_response",
                "strict": True,
                "schema": {
                    "type": "object",

                    "properties": {
                        "reply": {
                            "type": "string"
                        },

                        "escalate": {
                            "type": "boolean"
                        },

                        "reason": {
                            "type": "string"
                        }
                    },

                    "required": [
                        "reply",
                        "escalate",
                        "reason"
                    ],

                    "additionalProperties": False
                }
            }
        }
    )

    content = response.choices[0].message.content

    result = json.loads(content)

    return {
        "customer_message": message,
        "intent": intent,
        "confidence": confidence,
        "historical_cases": historical_cases,
        "reply": result["reply"],
        "escalate": result["escalate"],
        "reason": result["reason"]
    }


if __name__ == "__main__":

    message = input("\nCustomer message: ")

    result = generate_response(message)

    print("\n" + "=" * 70)
    print("AI SUPPORT AGENT")
    print("=" * 70)

    print("\nIntent:")
    print(result["intent"])

    print("\nConfidence:")
    print(f"{result['confidence']:.4f}")

    print("\nDraft Reply:")
    print(result["reply"])

    print("\nEscalate:")
    print(result["escalate"])

    print("\nReason:")
    print(result["reason"])

    print("\n" + "=" * 70)
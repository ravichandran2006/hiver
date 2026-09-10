# Amazon Customer Support AI Agent

## Overview

This project implements an AI-powered customer support agent for Amazon customer conversations on Twitter.

The system takes a customer message and:

1. Classifies the customer's intent.
2. Retrieves similar historical Amazon support conversations.
3. Generates a response grounded in those conversations.
4. Decides whether the case should be handled automatically or escalated to a human.

The project focuses on the **AmazonHelp** support account from the Customer Support on Twitter dataset.

## Dataset

The project uses the [`thoughtvector/customer-support-on-twitter`](https://huggingface.co/datasets/thoughtvector/customer-support-on-twitter) dataset, which contains approximately 2.8 million tweets from different customer-support accounts.

The main fields used are:

- `tweet_id`
- `author_id`
- `inbound`
- `created_at`
- `text`
- `response_tweet_id`
- `in_response_to_tweet_id`

AmazonHelp was selected because it provides a large number of real customer-support conversations.

The original dataset is not included in the repository. After downloading it, place it at:

```text
data/twcs.csv
```

The data preparation process is implemented in [`src/prepare_amazon.py`](src/prepare_amazon.py).

The pipeline extracts AmazonHelp conversations, connects customer messages with Amazon responses, removes empty and duplicate records, cleans the text, and creates a reproducible sample of 10,000 conversation pairs.

```text
Twitter Dataset
      |
      v
AmazonHelp Conversations
      |
      v
Customer-Response Pairs
      |
      v
Cleaning and Deduplication
      |
      v
10,000 Working Examples
```

## Intent Classification

After analysing the Amazon conversations, eight support intents were defined:

| Intent              | Description                                          |
| ------------------- | ---------------------------------------------------- |
| `DELIVERY_TRACKING` | Delivery, tracking and missing package issues        |
| `ORDER_MANAGEMENT`  | Order cancellation, changes and order problems       |
| `REFUND_RETURN`     | Returns and refund issues                            |
| `PAYMENT_BILLING`   | Payment and billing problems                         |
| `ACCOUNT_SECURITY`  | Hacked or unauthorized account issues                |
| `PRODUCT_TECHNICAL` | Product, application and technical problems          |
| `PRIME_DIGITAL`     | Prime, Kindle, Music and digital-service issues      |
| `GENERAL_COMPLAINT` | General, vague or unclear complaints                 |

The training data contains 700 labeled examples used to train a TF-IDF + Logistic Regression classifier.

A separate golden evaluation set contains 200 examples.

The classifier is implemented in:

- [`src/train_classifier.py`](src/train_classifier.py)

and evaluated using:

- [`src/evaluate_classifier.py`](src/evaluate_classifier.py)

## System Architecture

The complete workflow is:

```text
                 Customer Message
                        |
                        v
                Intent Classifier
                        |
                        v
                Intent + Confidence
                        |
                        v
             Historical Retriever
                        |
                        v
              Top 3 Similar Cases
                        |
                        v
               LLM Response Generator
                        |
             +----------+----------+
             |                     |
             v                     v
        Draft Reply          Escalation Decision
                                   |
                                   v
                            Escalation Reason
```

The system combines a lightweight ML classifier with historical retrieval and an LLM instead of relying only on the language model.

## Historical Retrieval

The retrieval component is implemented in [`src/retriever.py`](src/retriever.py).

TF-IDF and cosine similarity are used to find the three most similar historical customer conversations.

The retrieved conversations contain both the original customer message and the historical Amazon response.

These examples are provided to the LLM as evidence for generating the new response.

## Response Generation

The response-generation and escalation logic are implemented in [`src/agent.py`](src/agent.py).

The LLM receives the customer message, predicted intent, and retrieved historical conversations.

It produces:

- `reply`
- `escalate`
- `reason`

The system only drafts responses and does not directly perform actions such as refunds, order cancellation, account modification, or payment changes.

## Evaluation

The project evaluates the classifier against a trivial majority-class baseline and uses a nearest-neighbor historical response as a simple response-generation baseline.

Current intent-classification results are:

| Metric    | Proposed Model | Majority Baseline |
| --------- | -------------- | ----------------- |
| Accuracy  | 50.14%         | 33.57%            |
| Macro F1  | 45.72%         | 6.28%             |

Response quality is evaluated using an LLM-as-judge rubric covering:

- Relevance
- Groundedness
- Usefulness

A human-reviewed subset is used to compare the automated judge with human evaluation.

The evaluation scripts are located in the `src` directory and generated results are stored in `results`.

## Installation

The project requires Python 3.10 or later.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install pandas groq scikit-learn python-dotenv joblib
```

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

## Running the Project

Place the dataset at:

```text
data/twcs.csv
```

Then run:

```bash
python src/prepare_amazon.py
python src/create_split.py
python src/train_classifier.py
python src/evaluate_classifier.py
python src/evaluate_baselines.py
python src/retriever.py
python src/nearest_neighbor_baseline.py
python src/agent.py
python src/evaluate_responses.py
```

The main results are saved under:

```text
results/
```

## Project Structure

```text
hiver-sde-assignment/
|
├── data/
│   ├── amazon_pairs.csv
│   ├── intent_train.csv
│   ├── intent_train_labeled.csv
│   └── intent_gold.csv
|
├── src/
│   ├── prepare_amazon.py
│   ├── create_split.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── evaluate_baselines.py
│   ├── retriever.py
│   ├── nearest_neighbor_baseline.py
│   ├── agent.py
│   └── evaluate_responses.py
|
├── results/
├── report/
├── README.md
├── .env.example
└── .gitignore
```

## Limitations and Future Work

The current system can struggle with ambiguous messages, closely related intents, multilingual text, and weak historical matches.

The classifier also has class imbalance, so accuracy alone does not fully represent performance.

Future improvements would include better embedding-based retrieval, retrieval reranking, more human-labeled data for rare intents, confidence calibration, stronger escalation evaluation, and larger-scale human evaluation.

## Conclusion

The project demonstrates a practical customer-support workflow that combines:

```text
Intent Classification
        +
Historical Evidence Retrieval
        +
LLM Response Generation
        +
Escalation Decision
        +
Evaluation
```
The main objective is to build a support assistant that can understand customer issues, learn from historical support interactions, generate useful evidence-grounded responses, and identify cases where human support is more appropriate.

# Evaluation & System Report: AI Customer Support Agent

This report details the system design, dataset construction, baseline comparisons, evaluation metrics, failure modes, and architectural decision logs for the prototype AI Customer Support Agent built on the `AmazonHelp` dataset.

---

## 1. System Overview & Scope

The system is designed as a decision-support pipeline for customer support agents rather than an autonomous execution engine.

### System Pipeline

## 2. Golden Evaluation Set & Dataset Split

A separate, reproducible 200-example golden evaluation set was created from the prepared AmazonHelp dataset using a fixed random seed.

### Dataset Split Overview
| Dataset Partition | Size | Usage |
| :--- | :--- | :--- |
| **Training Set** | 800 examples | Classifier training |
| **Golden Test Set** | 200 examples | Reproducible system evaluation only (never used in training) |
| **Total** | **1,000 examples** | Combined annotated corpus |

### Golden Set Properties
* **Source:** AmazonHelp customer-support conversations
* **Size:** 200 examples
* **Label Type:** Human-reviewed intent (exactly 1 of 8 predefined categories per message)
* **Labeling Rule:** When messages expressed multiple concerns, labels were assigned based on the **primary** customer issue.

---

## 3. Predefined Intent Categories

| Intent Category | Description & Scope |
| :--- | :--- |
| `DELIVERY_TRACKING` | Delivery delays, missing packages, tracking status, carrier issues |
| `ORDER_MANAGEMENT` | Order cancellations, order modifications, general order placement issues |
| `REFUND_RETURN` | Return requests, drop-off processes, refund status tracking |
| `PAYMENT_BILLING` | Payment failures, double charges, billing inquiries |
| `ACCOUNT_SECURITY` | Compromised accounts, unauthorized access, security alerts |
| `PRODUCT_TECHNICAL` | Product defects, hardware/software troubleshooting |
| `PRIME_DIGITAL` | Prime memberships, Kindle, Amazon Music, digital media purchases |
| `GENERAL_COMPLAINT` | Expressed dissatisfaction without a clear actionable primary category |

---

## 4. Evaluation & Baseline Comparisons

### 4.1 Intent Classification Metrics
The proposed classifier (TF-IDF + Logistic Regression) was evaluated against a Majority Class Baseline (which trivially predicts `DELIVERY_TRACKING` for all instances).

| Metric | Majority Baseline | Proposed Model | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 33.57% | **50.00%** | +16.43% |
| **Macro F1** | 6.28% | **40.00%** | +33.72% |

> **Key Takeaway:** The significant leap in **Macro F1 (+33.72%)** highlights that the proposed model handles imbalanced minority classes far better than the baseline, which scores poorly due to class weighting.

### 4.2 Component & System Comparison

| System Component | Nearest-Neighbor Baseline | Proposed Agent |
| :--- | :--- | :--- |
| **Intent Classification** | No | Yes (8-Class Intent Model) |
| **Historical Retrieval** | Top 1 case | Top 3 cases |
| **Response Generation** | Exact copy of historical response | LLM-synthesized context draft |
| **Evidence Synthesis** | No | Yes |
| **Escalation Decision** | No | Yes (`escalate`: boolean) |
| **Escalation Rationale** | No | Yes (`reason`: string) |

### 4.3 Response Quality Rubric & Status
Responses are evaluated via an **LLM-as-a-Judge** framework using a 1–5 scoring scale across three dimensions:
* **Relevance (1–5):** Does the response address the customer's core issue?
* **Groundedness (1–5):** Is the response directly backed by retrieved historical cases?
* **Usefulness (1–5):** Does it provide clear, actionable next steps?

> [!WARNING]
> **Evaluation Status Note:** Full quantitative LLM-as-a-Judge and Human-vs-LLM agreement scoring runs were interrupted due to daily API rate limits. To maintain scientific integrity, unverified agreement scores are omitted.

---

## 5. Failure Analysis

1. **Delivery vs. Order Management Confusion**
   * *Issue:* Overlap in terms like `"order"`, `"package"`, and `"status"` causes misclassification between `DELIVERY_TRACKING` and `ORDER_MANAGEMENT`.
   * *Mitigation:* Transition from lexical TF-IDF features to dense semantic embeddings.
2. **Poor Performance on Minority Intents**
   * *Issue:* Low recall on `PAYMENT_BILLING`, `PRIME_DIGITAL`, and `REFUND_RETURN` due to insufficient training representation in the 800-example set.
   * *Mitigation:* Expand target labeling for minority classes or apply class-reweighting strategies.
3. **`GENERAL_COMPLAINT` Overlap**
   * *Issue:* Highly frustrated customers mentioning specific issues (e.g., late refunds) get misclassified under `GENERAL_COMPLAINT`.
   * *Mitigation:* Enforce a hierarchy rule: classify by actionable technical/operational issue first; fall back to `GENERAL_COMPLAINT` only if no actionable issue exists.
4. **Lexical Retrieval Mismatches**
   * *Issue:* TF-IDF cosine similarity fails when sentence structures differ despite identical intent (e.g., *"Where is my package?"* vs. *"My delivery hasn't arrived"*).
   * *Mitigation:* Implement dense vector retrieval (e.g., sentence-transformers) followed by a cross-encoder reranker.
5. **Over-Specific Nearest-Neighbor Responses**
   * *Issue:* Baseline directly copying single historical responses often introduces irrelevant specifics (e.g., answering a Kindle issue with phone support advice).
   * *Mitigation:* Synthesizing multi-case context using an LLM prevents single-source hallucination or over-specificity.

---

## 6. Interpretation of Headline Metrics

> [!IMPORTANT]
> The headline accuracy of **50.00%** measures **intent classification precision only** on the held-out test set.

It **does not** represent:
* Overall end-to-end task resolution rate
* Response helpfulness or groundedness
* Escalation decision correctness

It serves strictly as an internal benchmark for intent routing efficiency.

---

## 7. Development Decision Log

| # | Decision | Context & Justification |
| :---: | :--- | :--- |
| **1** | **Target Brand Selection** | Selected `AmazonHelp` from the TWCS dataset due to its high volume of real customer interactions across diverse domain issues. |
| **2** | **10,000-Pair Working Subset** | Sampled a fixed 10,000-pair dataset to maintain fast, reproducible iteration cycles while avoiding resource exhaustion. |
| **3** | **8 Intent Scope** | Defined 8 primary classes to balance broad coverage against multi-class ambiguity and boundary overlap. |
| **4** | **800-Example Training Split** | Set 800 examples for training to establish a lightweight, human-annotated baseline. |
| **5** | **Dedicated 200 Golden Set** | Formed a distinct 200-example set reserved strictly for final testing and evaluation. |
| **6** | **Fixed Random Seeds** | Enforced deterministic random seeds across all scripts for full experimental reproducibility. |
| **7** | **TF-IDF Feature Extraction** | Chosen for classification to establish a fast, interpretable, and computationally light baseline. |
| **8** | **Logistic Regression Classifier** | Paired with sparse TF-IDF matrices for efficient training and robust baseline performance. |
| **9** | **Macro F1 Reporting** | Adopted Macro F1 alongside accuracy to accurately capture performance on imbalanced minority intent classes. |
| **10** | **TF-IDF Vector Retrieval** | Utilized cosine similarity over historical customer messages to ground generations in real support context. |
| **11** | **Top-3 Case Context Window** | Retrieved 3 cases per query to provide sufficient evidence diversity without exceeding prompt context limits. |
| **12** | **LLM Response Synthesis** | Configured the LLM to generate novel responses informed by retrieved context rather than outputting exact baseline text. |
| **13** | **Structured JSON Schema** | Enforced output structure (`reply`, `escalate`, `reason`) for automated evaluation and predictable execution. |
| **14** | **Restricted Account Execution** | Limited the agent to response generation and escalation tagging to avoid risk in production account management. |
| **15** | **Dual Baseline Strategy** | Implemented Majority Class and Nearest-Neighbor baselines to clearly isolate and measure system value additions. |

---

## 8. Reproduction Steps

Execute the full pipeline within the target reproduction environment:

```bash
# 1. Prepare AmazonHelp dataset
python src/prepare_amazon.py

# 2. Create train/eval splits
python src/create_split.py

# 3. Train classifier
python src/train_classifier.py

# 4. Evaluate proposed classifier
python src/evaluate_classifier.py

# 5. Run baseline comparisons
python src/evaluate_baselines.py
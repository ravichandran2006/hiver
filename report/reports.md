Evaluation Report

1. Golden Evaluation Set

A separate 200-example golden evaluation set was created from the prepared AmazonHelp dataset.

The examples were sampled using a fixed random seed to make the evaluation reproducible. The golden examples were kept separate from the classifier training data.

Each example was manually reviewed and assigned exactly one of the eight predefined intent categories.

When a message contained multiple issues, the label was assigned based on the primary customer problem expressed in the message.

The golden set was used only for evaluation and was not used to train the classifier.

Golden Set Summary

Property

Value

Source

AmazonHelp customer-support conversations

Size

200 examples

Label type

Intent

Number of intents

8

Labeling

Human-reviewed

Used for training

No

Used for evaluation

Yes

2. Problem Framing

What Does Good Mean?

For Amazon customer support, a good response should:

Correctly understand the customer's primary issue.

Address the customer's actual problem.

Be consistent with how similar Amazon cases were historically handled.

Provide a useful next step.

Avoid making unsupported claims.

Recognize cases that require human intervention.

The system therefore treats customer-support quality as a combination of:

Intent Understanding
        +
Historical Evidence
        +
Response Quality
        +
Escalation Quality

What Was Not Built

The prototype intentionally does not directly execute customer-account actions.

The system does not perform:

Refunds

Order cancellations

Payment changes

Account modifications

Password changes

Other customer-account actions

Instead, the system focuses on decision support:

Classify the customer's message.

Retrieve relevant historical support cases.

Generate a grounded response draft.

Decide whether the case should be escalated.

Provide a reason for the escalation decision.

3. System Overview

The proposed system follows the following pipeline:

Customer Message
       |
       v
Intent Classification
       |
       v
Historical Case Retrieval
       |
       v
Evidence Collection
       |
       v
LLM Response Generation
       |
       v
Escalation Decision
       |
       v
Final Support Response

Intent Categories

The system uses eight intents:

Intent

Description

DELIVERY_TRACKING

Delivery delays, missing packages, tracking and delivery status

ORDER_MANAGEMENT

Order cancellation, modification and order-related issues

REFUND_RETURN

Refunds, returns and return-related issues

PAYMENT_BILLING

Payment, billing and charge-related issues

ACCOUNT_SECURITY

Hacked accounts, unauthorized access and account security

PRODUCT_TECHNICAL

Product defects and technical problems

PRIME_DIGITAL

Prime, Kindle, Amazon Music and digital services

GENERAL_COMPLAINT

General complaints or issues that do not clearly fit another category

Intent Classification

The first stage predicts one of the eight predefined customer-support intents.

Historical Retrieval

After identifying the intent, the system searches the prepared AmazonHelp conversation dataset for historically similar customer messages.

TF-IDF vectorization and cosine similarity are used to retrieve the top three historical cases.

The retrieved cases contain:

Historical customer message

Historical Amazon response

Similarity score

These cases provide evidence for the response-generation stage.

Response Generation

The LLM receives:

The customer's current message

The predicted intent

The retrieved historical support cases

It then generates a new response rather than directly copying a historical response.

The generated output contains:

{
  "reply": "Draft response to the customer",
  "escalate": false,
  "reason": "Reason for the escalation decision"
}

Escalation

The agent also determines whether the issue should be handled automatically or escalated to a human support representative.

The escalation decision is returned together with a reason so that the decision is interpretable.

4. Reproducibility and Runtime

The repository contains the scripts required to reproduce the classifier headline result.

After placing the original dataset at:

data/twcs.csv

the main evaluation pipeline can be executed with:

python src/prepare_amazon.py
python src/create_split.py
python src/train_classifier.py
python src/evaluate_classifier.py
python src/evaluate_baselines.py

The headline classifier metrics are written to:

results/metrics.txt
results/intent_evaluation.csv
results/confusion_matrix.csv

The pipeline uses a fixed random seed for sampling and splitting, making the reported evaluation reproducible.

The classifier evaluation is designed to run within the assignment's 15-minute reproduction target on a normal development machine.

LLM-based response generation and evaluation are separate API-dependent steps.

5. Intent Classification Evaluation

The labeled dataset was divided into:

Dataset

Examples

Training

800

Testing

200

Total

1,000

The classifier achieved:

Metric

Proposed Model

Majority Baseline

Accuracy

50.00%

33.57%

Macro F1

40.00%

6.28%

The proposed classifier improves over the majority baseline by:

16.43 percentage points in Accuracy

33.72 percentage points in Macro F1

The improvement in Macro F1 is particularly important because the intent classes are imbalanced.

Accuracy alone can hide poor performance on minority classes, while Macro F1 gives every intent equal importance.

6. Baseline Comparison

6.1 Trivial Baseline — Majority Class

The majority baseline always predicts:

DELIVERY_TRACKING

This provides a simple lower-bound reference for the classifier.

Results

Metric

Majority Baseline

Proposed Model

Accuracy

33.57%

50.00%

Macro F1

6.28%

40.00%

The proposed classifier substantially outperforms the majority-class baseline.

This indicates that the classifier is learning useful patterns from customer messages rather than simply predicting the most common intent.

6.2 Simple Baseline — Nearest Historical Response

The second baseline retrieves the single most similar historical customer message using TF-IDF cosine similarity and directly returns the associated Amazon response.

Implementation:

src/nearest_neighbor_baseline.py

This baseline was tested on representative customer-support cases.

It can produce useful responses when the retrieved example is highly similar.

However, it can also return an inappropriate response when lexical similarity does not correspond to semantic similarity.

For example, a Kindle-related technical question retrieved a historical response concerning a telephone-number issue. Although there was some lexical similarity, the historical response did not address the actual underlying problem.

The proposed system addresses this limitation by retrieving multiple historical cases and using an LLM to generate a new response rather than directly copying one historical response.

Component Comparison

Component

Nearest-Neighbor Baseline

Proposed Agent

Intent classification

No

Yes

Historical retrieval

1 case

Top 3 cases

Response generation

Copies historical response

LLM generates response

Evidence synthesis

No

Yes

Escalation decision

No

Yes

Escalation reason

No

Yes

A quantitative response-quality score for this baseline has not been claimed because the full human/LLM response evaluation was not completed due to the LLM API quota limitation.

7. Response Quality Evaluation

Generated responses are evaluated using an LLM-as-judge rubric.

Each response is scored from 1 to 5 on three dimensions.

Dimension

Evaluation Question

Relevance

Does the response address the customer's actual problem?

Groundedness

Is the response supported by the retrieved historical evidence?

Usefulness

Does the response provide a useful next step?

Relevance

Score

Description

1

Does not address the customer's issue

2

Mostly irrelevant

3

Partially addresses the issue

4

Relevant with minor weaknesses

5

Directly addresses the issue

Groundedness

Score

Description

1

Unsupported by evidence

2

Mostly unsupported

3

Partially supported

4

Mostly supported

5

Strongly supported by historical evidence

Usefulness

Score

Description

1

Provides no useful help

2

Provides very limited help

3

Provides a reasonable next step

4

Provides a useful actionable response

5

Provides a clear and highly useful next step

The evaluation harness is implemented in:

src/evaluate_responses.py

8. Human vs LLM Judge Agreement

The automated LLM judge is intended to be validated against human ratings.

The evaluation process is:

Generated Response
        |
        +----------------------+
        |                      |
        v                      v
  Human Rating           LLM Rating
        |                      |
        +----------+-----------+
                   |
                   v
           Agreement Analysis

The same generated responses are rated by both the human evaluator and the LLM judge using:

Relevance

Groundedness

Usefulness

Current Evaluation Status

The response-quality evaluation harness was implemented, but the initial evaluation run was interrupted by the daily LLM API token limit.

Therefore, this report does not claim a numerical human-vs-LLM agreement score that was not actually measured.

This is reported as an evaluation limitation rather than replacing the missing evaluation with unsupported numbers.

Metric

Result

Number of evaluated responses

Not measured

Relevance agreement

Not measured

Groundedness agreement

Not measured

Usefulness agreement

Not measured

Overall agreement

Not measured

9. Failure Analysis

Failure 1 — Delivery vs Order Confusion

Customer messages containing words such as:

order
package
delivery
status

can be confused between:

DELIVERY_TRACKING

and:

ORDER_MANAGEMENT

Hypothesis

The TF-IDF classifier relies heavily on lexical features and does not fully capture semantic differences.

Improvement

Use sentence embeddings or a stronger semantic classifier.

Failure 2 — Poor Performance on Rare Intents

The classifier performs poorly on less represented categories, particularly:

PAYMENT_BILLING

PRIME_DIGITAL

REFUND_RETURN

Hypothesis

There are not enough labeled examples for the classifier to learn robust patterns.

Improvement

Increase human-labeled training examples for minority intents.

Failure 3 — General Complaint Overlap

Customers can mention a specific issue while primarily expressing dissatisfaction.

For example, a refund complaint may also contain strong general dissatisfaction with Amazon.

Hypothesis

GENERAL_COMPLAINT overlaps with several actionable intents.

Improvement

Use a labeling rule that prioritizes the actionable support issue when it is clearly identifiable.

For example:

If a concrete support issue exists:
    assign the actionable intent

Otherwise:
    assign GENERAL_COMPLAINT

Failure 4 — Lexical Retrieval Misses Semantic Similarity

TF-IDF retrieval depends on word overlap.

Two messages describing the same issue with different wording can receive a relatively low similarity score.

For example:

"My package hasn't arrived yet."

"Where is my delivery? It was supposed to come yesterday."

These messages have similar meaning but different wording.

Hypothesis

TF-IDF cannot capture semantic equivalence effectively.

Improvement

Use sentence embeddings followed by reranking.

Failure 5 — Nearest-Neighbor Response Can Be Over-Specific

The nearest-neighbor baseline directly copies the response from the most similar historical case.

A retrieved case may share words with the new customer message but describe a different underlying issue.

Example

A Kindle-related technical question retrieved a historical response related to a telephone-number issue.

The response therefore contained information that was not relevant to the customer's actual problem.

Hypothesis

Single-example lexical retrieval is too sensitive to surface-level word overlap.

Improvement

Retrieve multiple examples and generate a response from the common evidence rather than copying one historical response.

10. What Is Misleading About My Headline Number?

The headline classifier accuracy of 50.00% should not be interpreted as the overall performance of the customer-support agent.

It measures only intent classification accuracy on the held-out testing set.

It does not measure:

Response relevance

Response groundedness

Response usefulness

Escalation accuracy

Customer satisfaction

Actual issue resolution

Business impact

The result is also affected by class imbalance.

Therefore, Macro F1 is reported alongside accuracy.

Headline Metrics

Metric

Result

Accuracy

50.00%

Macro F1

40.00%

The correct interpretation is:

The classifier correctly predicts the predefined support intent for 50.00% of the held-out test examples.

It should not be interpreted as:

The AI agent successfully resolves 50.00% of customer problems.

The headline number is therefore useful as a classifier metric, but it is not a complete measure of customer-support quality.

11. Decision Log

The following table summarizes the major non-obvious decisions made during development.

#

Decision

Why?

1

Selected AmazonHelp as the target brand

AmazonHelp provides a large number of real customer-support conversations, making it suitable for experimentation and evaluation.

2

Used a 10,000-pair working dataset

The complete dataset is very large, so a fixed working sample keeps experimentation practical while remaining reproducible.

3

Defined eight intents

Eight categories provide useful support coverage without creating excessive overlap between classes.

4

Used 800 labeled examples for training

This provides a manageable amount of manually labeled data for the initial classifier.

5

Created a separate 200-example golden set

A separate evaluation set prevents evaluation examples from being used to train the classifier.

6

Used a fixed random seed

Fixed sampling and splitting make the experiments reproducible.

7

Used TF-IDF for classification

TF-IDF is lightweight, fast and easy to interpret for a prototype.

8

Used Logistic Regression

Logistic Regression works efficiently with sparse TF-IDF features.

9

Reported Macro F1 alongside accuracy

Macro F1 gives equal importance to each intent and is more informative when classes are imbalanced.

10

Used TF-IDF cosine similarity for retrieval

It provides a simple and reproducible way to retrieve historically similar customer messages.

11

Retrieved three historical cases

Multiple examples provide more evidence than relying on a single historical case.

12

Used an LLM for response generation

The LLM can synthesize information from multiple historical cases instead of copying one response.

13

Used structured output

Returning reply, escalate, and reason makes the agent output consistent and easier to evaluate.

14

Restricted direct customer-account actions

The prototype does not have production integrations, so it only drafts responses and recommends escalation.

15

Added majority and nearest-neighbor baselines

These provide simple reference points for measuring whether the proposed approach adds value.
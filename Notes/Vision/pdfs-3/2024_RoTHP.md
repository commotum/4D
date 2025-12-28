# RoTHP (2024) Survey Answers

## 1. Basic Metadata
- Title: ROTHP: ROTARY POSITION EMBEDDING-BASED TRANSFORMER HAWKES PROCESS
- Authors: Anningzhe Gao; Shan Dai
- Year: 2024
- Venue: arXiv (arXiv:2405.06985v1 [cs.LG])
Evidence:
- "ROTHP: ROTARY P OSITION E MBEDDING - BASED T RANSFORMER H AWKES P ROCESS" (Title page)
- "arXiv:2405.06985v1 [cs.LG] 11 May 2024" (Title page)

## 2. One-Sentence Contribution Summary
- The paper proposes a Rotary Position Embedding-based Transformer Hawkes Process (RoTHP) to address timestamp-noise sensitivity and sequence prediction issues in attention-based Hawkes processes by introducing rotary positional embeddings with translation-invariance and improved sequence prediction.
Evidence:
- "To deal with the problems, we propose a new Rotary Position Embedding-based THP (RoTHP) architecture in this paper." (Abstract)
- "Notably, we show the translation invariance property and sequence prediction flexibility of our RoTHP induced by the relative time embeddings when coupled with Hawkes process theoretically." (Abstract)

## 3. Tasks Evaluated

Task 1: Next-event prediction (event type + timestamp)
- Task type: Classification + Other (time prediction / regression)
- Dataset(s) used: Synthetic; Financial Transactions; StackOverFlow; Retweet; Memetrack; Mimic-II
- Domain: temporal event sequences (synthetic Hawkes process; financial transactions; social Q&A activity; social media retweets; meme diffusion; medical ICU admissions)
- Evidence (task definition):
  - "For the prediction of next event type and timestamp, we train two linear layers W e , W t" (Section 3.1.2 Training)
  - "By definition, Levent measures the accuracy of the event type prediction and Ltime measures the mean square loss the of time prediction." (Section 3.1.2 Training)
- Evidence (datasets/domain):
  - "Synthetic This dataset was generated using Python, following the methodology outlined in [16]. It is a product of a Hawkes process, making it a suitable case for our study." (Section 4.1 Dataset)
  - "Financial Transactions [5] This dataset comprises stock transaction records from a single trading day." (Section 4.1 Dataset)
  - "StackOverFlow [29] This dataset is a collection of data from the question-answer website, Stacoverflow." (Section 4.1 Dataset)
  - "Retweet [30] The data set for Retweets compiles a variety of tweet chains. Every chain comprises an original tweet initiated by a user, accompanied by subsequent response tweets." (Section 4.1 Dataset)
  - "Memetrack [29] This dataset comprises references to 42,000 distinct memes over a period of ten months." (Section 4.1 Dataset)
  - "Mimic-II [31] The MIMIC-II medical dataset compiles data from patients’ admissions to an ICU over a span of seven years." (Section 4.1 Dataset)

Task 2: Future event prediction (train on past, predict future)
- Task type: Other (future event prediction)
- Dataset(s) used: Financial Transactions; Synthetic; StackOverFlow
- Domain: temporal event sequences in finance, synthetic Hawkes process, social Q&A activity
- Evidence (task definition):
  - "In this subsection, we consider the case where we use the previous information to predict the future ones." (Section 4.6 Predict the future features)
  - "We do the test on financial transaction, synthetic and StackOverflow dataset, and Table 6 shows the result." (Section 4.6 Predict the future features)

## 4. Domain and Modality Scope
- Single domain vs multiple domains: Multiple domains within the same modality (temporal event sequences).
- Multiple modalities: Not specified in the paper.
- Domain generalization / cross-domain transfer: Not claimed; the paper focuses on robustness to timestamp translation/noise and sequence prediction.
Evidence:
- "Temporal Point Processes (TPPs), especially Hawkes Process are commonly used for modeling asynchronous event sequences data such as financial transactions and user behaviors in social networks." (Abstract)
- "Financial Transactions [5] This dataset comprises stock transaction records from a single trading day." (Section 4.1 Dataset)
- "StackOverFlow [29] This dataset is a collection of data from the question-answer website, Stacoverflow." (Section 4.1 Dataset)
- "Retweet [30] The data set for Retweets compiles a variety of tweet chains." (Section 4.1 Dataset)
- "Memetrack [29] This dataset comprises references to 42,000 distinct memes over a period of ten months." (Section 4.1 Dataset)
- "Mimic-II [31] The MIMIC-II medical dataset compiles data from patients’ admissions to an ICU over a span of seven years." (Section 4.1 Dataset)
- "we demonstrate empirically that our RoTHP can be better generalized in sequence data scenarios with translation or noise in timestamps and sequence prediction tasks." (Abstract)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Next-event prediction (event type + timestamp) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Future event prediction | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## 6. Input and Representation Constraints
- Event representation (timestamp + type): "Let S = {(ti , ki )}ni=1 be a sequence of Hawkes process." (Section 3.1.1 Model architecture)
- Event type encoding: "We use X to denote the matrix representing the one-hot vector corresponding to the event sequence. X ∈ RK×L , the ith column of X is a one-hot vector where the jth entry is non-zero if and only if ki = j." (Section 3.1.1 Model architecture)
- Sequence length: variable in datasets (e.g., "Our synthetic dataset admits 5 event types, with average length 60. The minimal length is 20 and the maximal length is 100." (Section 4.1 Dataset); "The average length of the sequences in the dataset is 72, with minimum 41 and maximum 736" (Section 4.1 Dataset))
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper (datasets show variable lengths).
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
- Padding/resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Variable in datasets (e.g., "The minimal length is 20 and the maximal length is 100." (Section 4.1 Dataset); "The average length of the sequences in the dataset is 72, with minimum 41 and maximum 736" (Section 4.1 Dataset)).
- Attention type: Self-attention (Transformer); no windowed/hierarchical/sparse attention is described.
- Mechanisms to manage computational cost: Not specified in the paper.
Evidence:
- "Recent developments in natural language processing (NLP) have led to an increasing interest in the self-attention mechanism. [16] present self-attention Hawkes process, furthermore, [17] propose transformer Hawkes process based on the attention mechanism and encoder structure in transformer." (Introduction)

## 8. Positional Encoding (Critical Section)
- Mechanism: Rotary Position Embedding / Rotary Temporal Positional Embedding (RoTPE) in RoTHP.
- Where applied: Not specified in the paper beyond being the positional embedding used in the transformer model.
- Fixed across experiments vs modified per task: Not specified in the paper.
- Ablated/compared against alternatives: Compared against THP with absolute positional encoding.
Evidence:
- "The key idea in our model design is to apply the rotary position embedding method [22] into temporal process." (Section 3.1.1 Model architecture)
- "To deal with the problems, we propose a new Rotary Position Embedding-based THP (RoTHP) architecture in this paper." (Abstract)
- "the Transformer Hawkes process (THP) as operatized in [17], applies the transformer structure to the Hawkes process and incorporates absolute positional encoding in the model construction" (Section 4.2 Baselines)

## 9. Positional Encoding as a Variable
- Core research variable vs fixed assumption: Core research variable (the paper proposes RoTHP to address positional encoding issues).
- Multiple positional encodings compared: Yes, RoTHP is compared to THP with absolute positional encoding (baseline).
- Claim that PE choice is not critical/secondary: Not specified in the paper.
Evidence:
- "To deal with the problems, we propose a new Rotary Position Embedding-based THP (RoTHP) architecture in this paper." (Abstract)
- "the Transformer Hawkes process (THP) as operatized in [17], applies the transformer structure to the Hawkes process and incorporates absolute positional encoding in the model construction" (Section 4.2 Baselines)

## 10. Evidence of Constraint Masking
- Model size(s): Not specified in the paper.
- Dataset size(s) / scale:
  - "Our synthetic dataset admits 5 event types, with average length 60. The minimal length is 20 and the maximal length is 100." (Section 4.1 Dataset)
  - "The average length of the dataset is 2074 and is appropriate to our experiment." (Section 4.1 Dataset)
  - "This dataset comprises references to 42,000 distinct memes over a period of ten months. It encompasses data from more than 1.5 million documents, including blogs and web articles, sourced from over 5,000 websites." (Section 4.1 Dataset)
- Performance gains attributed to: Architectural change (rotary positional embedding), not model/data scaling.
Evidence:
- "We can see that the RoTHP outperforms all other models for these three datasets for the log-likelihood, see Table 2" (Section 4.4 Result)
- "For the training durations, RoTHP consistently outperforms THP, underscoring the benefits of rotary embedding in the Hawkes process." (Section 4.4 Result)

## 11. Architectural Workarounds
- Rotary positional embedding (RoTHP/RoTPE) to address timestamp noise sensitivity and sequence prediction issues.
  - "To deal with the problems, we propose a new Rotary Position Embedding-based THP (RoTHP) architecture in this paper." (Abstract)
  - "The key idea in our model design is to apply the rotary position embedding method [22] into temporal process." (Section 3.1.1 Model architecture)
- Using time differences directly in the intensity function (no normalization by tj) to support translation invariance.
  - "Different from the conditional intensity function setting in the THP [17], here we directly adopt the time difference t − tj without the normalization by tj." (Section 3.1.2 Training)

## 12. Explicit Limitations and Non-Claims
- Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Multiple domains (finance, social Q&A, social media retweets, meme diffusion, medical ICU admissions, synthetic Hawkes process) within a single modality of temporal event sequences.
- Task structure: Predicting next event type and timestamp (classification + time prediction), plus a future-prediction split; evaluation uses log-likelihood/accuracy/RMSE on temporal sequences.
- Representation rigidity: Event sequences represented as (timestamp, type) with one-hot event type encoding; sequence lengths vary by dataset.
- Model sharing vs specialization: Not specified whether weights are shared or fine-tuned per task/dataset.
- Role of positional encoding: Central research variable; RoTHP introduces rotary positional embedding and compares against absolute positional encoding baselines.

## 14. Final Classification
- Classification: Multi-task, multi-domain (constrained).
- Justification: The paper evaluates multiple domains/datasets (e.g., "stock transaction records," "question-answer website," "tweet chains," "memes," and a medical ICU dataset) under temporal point process modeling, and it explicitly defines prediction of "next event type and timestamp" as a core task; it also evaluates "predict[ing] the future ones" on multiple datasets. These tasks remain constrained to a single modality of temporal event sequences rather than unrestrained multi-domain/multi-modal learning.
Evidence:
- "Financial Transactions [5] This dataset comprises stock transaction records from a single trading day." (Section 4.1 Dataset)
- "StackOverFlow [29] This dataset is a collection of data from the question-answer website, Stacoverflow." (Section 4.1 Dataset)
- "Retweet [30] The data set for Retweets compiles a variety of tweet chains." (Section 4.1 Dataset)
- "Memetrack [29] This dataset comprises references to 42,000 distinct memes over a period of ten months." (Section 4.1 Dataset)
- "Mimic-II [31] The MIMIC-II medical dataset compiles data from patients’ admissions to an ICU over a span of seven years." (Section 4.1 Dataset)
- "For the prediction of next event type and timestamp, we train two linear layers W e , W t" (Section 3.1.2 Training)
- "In this subsection, we consider the case where we use the previous information to predict the future ones." (Section 4.6 Predict the future features)

## 1. Basic Metadata

- **Title**: "T RAIN S HORT, T EST L ONG : ATTENTION WITH L INEAR B IASES E NABLES I NPUT L ENGTH E XTRAPOLATION" (p.1)
- **Authors**: "Ofir Press1,2        Noah A. Smith1,3       Mike Lewis2" (p.1)
- **Year**: 2022 ("Published as a conference paper at ICLR 2022" and "arXiv:2108.12409v2 [cs.CL] 22 Apr 2022" on p.1)
- **Venue**: ICLR 2022 ("Published as a conference paper at ICLR 2022", p.1); arXiv preprint listed on p.1

---

## 2. One-Sentence Contribution Summary

Introduces ALiBi, a position method that linearly biases attention scores by distance to enable transformer language models trained on short sequences to extrapolate to longer input lengths efficiently.

---

## 3. Tasks Evaluated

### Task 1: Language Modeling (Next-token prediction) on WikiText-103
- **Task type**: Generation (language modeling / next-token prediction)
- **Dataset(s)**: WikiText-103
- **Domain**: Natural language text (English Wikipedia)
- **Quotes**:
  - Task definition: "A transformer LM receives a list of tokens and outputs a probability distribution representing its prediction for the next token." (§2.1, p.2)
  - Dataset usage: "We first test the extrapolation abilities of various position methods on the WikiText-103 corpus (Merity et al., 2016)" (§2.1, p.2)
  - Domain: "The training set is about 103 million tokens from English Wikipedia (half a gigabyte)." (§2.1, p.2)

### Task 2: Language Modeling (Next-token prediction) on Toronto BookCorpus
- **Task type**: Generation (language modeling / next-token prediction)
- **Dataset(s)**: Toronto BooksCorpus
- **Domain**: Natural language text (books)
- **Quotes**:
  - Different domain evaluation: "After developing our method on WikiText-103, in Appendix Section A.3, we run one set of experiments on a different domain (books)" (§4.1, p.6)
  - Dataset usage: "Specifically, we use the Toronto BooksCorpus (Zhu et al., 2015)" (Appendix A.3, p.20)
  - Dataset size (text domain): "The corpus is about 700M tokens (2.9 GB)." (Appendix A.3, p.20)

### Task 3: Language Modeling (Next-token prediction) on CC100+RoBERTa corpus
- **Task type**: Generation (language modeling / next-token prediction)
- **Dataset(s)**: CC100+RoBERTa corpus (RoBERTa training corpora + CC-100 English)
- **Domain**: Natural language text (multi-source English corpora)
- **Quotes**:
  - Dataset composition: "The dataset we choose is a combination of the datasets used to train the RoBERTa (Liu et al., 2019) implementation of BERT (Devlin et al., 2019) and the English part of the CC-100 corpus introduced in Conneau et al. (2020), for a total of 461 GB." (§4.2, p.7)
  - Domain sources: "The RoBERTa training corpus—i.e., the Toronto Book Corpus (Zhu et al., 2015), English Wikipedia, CC-News (Nagel, 2016), OpenWeb-Text (Gokaslan & Cohen, 2019) and Stories (Trinh & Le, 2018))—is 161 gigabytes, and the English part of the CC-100 corpus is 300 gigabytes." (§4.2, p.7)

---

## 4. Domain and Modality Scope

- **Single domain or multiple domains?** Multiple domains within the same modality (text). Evidence:
  - "We first test the extrapolation abilities of various position methods on the WikiText-103 corpus" (§2.1, p.2)
  - "After developing our method on WikiText-103, in Appendix Section A.3, we run one set of experiments on a different domain (books)" (§4.1, p.6)
  - "The dataset we choose is a combination of the datasets used to train the RoBERTa ... and the English part of the CC-100 corpus" (§4.2, p.7)
- **Multiple modalities?** No; only text modality is described (LM on token sequences). Evidence: "A transformer LM receives a list of tokens and outputs a probability distribution representing its prediction for the next token." (§2.1, p.2)
- **Domain generalization / cross-domain transfer claims?** Limited claim of transfer across text domains: "This result establishes the generality of ALiBi and the particular set of slopes we found and suggests that they may be used on different text domains without further hyperparameter tuning." (Appendix A.3, p.20)

---

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling on WikiText-103 | No (separate training per dataset/setting) | No fine-tuning mentioned | Not specified in the paper | "We first develop our method on the WikiText-103 corpus" (§4.1, p.6) and "We first test the extrapolation abilities of various position methods on the WikiText-103 corpus" (§2.1, p.2) indicate training/evaluation on that dataset. |
| Language modeling on Toronto BookCorpus | No (separate training on a different domain) | No fine-tuning mentioned | Not specified in the paper | "After developing our method on WikiText-103, in Appendix Section A.3, we run one set of experiments on a different domain (books)" (§4.1, p.6); "Specifically, we use the Toronto BooksCorpus" (Appendix A.3, p.20). |
| Language modeling on CC100+RoBERTa | No (separate training on larger combined corpus) | No fine-tuning mentioned | Not specified in the paper | "The dataset we choose is a combination of the datasets used to train the RoBERTa ... and the English part of the CC-100 corpus" (§4.2, p.7); "Our models for this dataset have 25 transformer layers..." (§4.2, p.7) indicates separate training setup. |

Additional note on evaluation across lengths (same weights used across Lvalid for a given model): "When L differs between inference and training, we use L to refer to the length of subsequences during training and Lvalid to refer to their length at validation." (§2.1, p.2)

---

## 6. Input and Representation Constraints

- **Fixed or variable input resolution?** Not specified in the paper (text LM; no image resolution).
- **Fixed patch size?** Not specified in the paper.
- **Fixed number of tokens?** Training uses fixed subsequence length L; evaluation can vary Lvalid.
  - "Let L be the length of each input subsequence during training; it includes L predictions" (§2.1, p.2)
  - "When L differs between inference and training, we use L to refer to the length of subsequences during training and Lvalid to refer to their length at validation." (§2.1, p.2)
- **Fixed dimensionality (e.g., strictly 2D)?** Not specified in the paper.
- **Padding or resizing requirements?** Not specified in the paper.
- **Tokenization / representation details (explicit):** "their tokenization, which uses BERT’s vocabulary of 29K byte-pair encodings." (Appendix A.3, p.20)

---

## 7. Context Window and Attention Structure

- **Maximum sequence length**:
  - "we then run inference with it on the validation set on L + k tokens, with k ranging from 0 to 15,000." (§2.2, p.3)
  - "ALiBi maintains strong performance even on sequences of length 10,000." (Introduction, p.1)
- **Fixed or variable sequence length?** Fixed during training (L), variable during evaluation (Lvalid > L).
  - "Let L be the length of each input subsequence during training" (§2.1, p.2)
  - "To explore a model’s extrapolation abilities, we are interested in cases where sequences of length Lvalid > L are considered at evaluation time." (§2.1, p.2)
- **Attention type**: Global causal self-attention with a causal mask, plus ALiBi’s linear bias.
  - "a 'causal mask' that ensures each position’s prediction is influenced only by tokens to its left." (§2.1, p.2)
  - "The only modification we apply is after the query-key dot product, where we add a static, non-learned bias" (§3, p.5)
- **Mechanisms introduced to manage computational cost**:
  - Training on shorter sequences with ALiBi: "Using ALiBi, a transformer LM can be trained on short-L sequences and therefore at much lower cost, and it can still be reliably applied to long sequences at runtime." (Introduction, p.1)
  - Nonoverlapping inference to handle long sequences cheaply: "it is typical to segment the sequence into L-length subsequences and train on or evaluate them independently. Unless otherwise stated, we use nonoverlapping inference" (§2, p.2)
  - Sliding window evaluation is discussed but noted as slow: "the sliding window approach is much slower" (Appendix B.1, p.23)

---

## 8. Positional Encoding (Critical Section)

- **Mechanism used**: Bias-based relative position method (ALiBi) with linear distance penalty.
  - "ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance." (Abstract, p.1)
- **Where applied**: After query-key dot product in attention; position info added at every layer to keys and queries (not values).
  - "When using ALiBi, we do not add position embeddings at any point in the network. The only modification we apply is after the query-key dot product, where we add a static, non-learned bias" (§3, p.5)
  - "Since ALiBi is a relative position method, we add position information at every layer to the keys and queries but not to the values" (§3, p.5)
- **Fixed across experiments vs modified per task**:
  - Fixed slopes reused across domains: "We emphasize that our set of slopes was chosen by running experiments on the WikiText-103 corpus, and here we apply that set of slopes to a model trained on a very different text domain." (Appendix A.3, p.20)
  - "we do not believe that it is necessary to tune these slope values every time a new model is trained on a new dataset." (§3, p.5)

---

## 9. Positional Encoding as a Variable

- **Core research variable?** Yes; position method is central to the paper’s claim about extrapolation.
  - "We demonstrate that this failure to extrapolate is caused by the position embedding method." (Introduction, p.1)
  - "we conclude that extrapolation ability depends heavily on the position embedding." (§2, p.2)
- **Multiple positional encodings compared?** Yes; sinusoidal, rotary, T5 bias, and ALiBi are compared.
  - "recent alternatives to the original sinusoidal position method (Su et al., 2021; Raffel et al., 2020) have improved extrapolation" (Introduction, p.1)
  - "the T5 bias method leads to better extrapolation than either of these" (§2, p.2)
- **PE choice claimed as not critical / secondary?** No; the paper explicitly states it is critical: "extrapolation ability depends heavily on the position embedding." (§2, p.2)

---

## 10. Evidence of Constraint Masking (Scale vs Structure)

- **Model sizes**:
  - "a 1.3 billion parameter model" (Abstract, p.1)
  - WikiText-103 model config: "The model has 16 transformer layers of dimension 1024, with 8 heads, and a feedforward inner dimension of 4096." (§2.1, p.2)
  - CC100+RoBERTa model config: "Our models for this dataset have 25 transformer layers with 16 heads and a dimension of 2048, with an 8192 hidden dimension... These models have 1.3B parameters." (§4.2, p.7)
- **Dataset sizes**:
  - WikiText-103: "The training set is about 103 million tokens from English Wikipedia (half a gigabyte)." (§2.1, p.2)
  - Toronto BookCorpus: "The corpus is about 700M tokens (2.9 GB)." (Appendix A.3, p.20)
  - CC100+RoBERTa: "for a total of 461 GB" (RoBERTa corpus + CC-100 English) (§4.2, p.7)
- **Attribution of performance gains**:
  - The paper attributes extrapolation improvements primarily to positional encoding choice: "We demonstrate that this failure to extrapolate is caused by the position embedding method." (Introduction, p.1)
  - Efficiency gains attributed to shorter training sequences enabled by ALiBi: "Using ALiBi, a transformer LM can be trained on short-L sequences and therefore at much lower cost" (Introduction, p.1)
- **Scaling model size/data as primary cause?** Not claimed; the paper emphasizes positional encoding and sequence length, not scaling alone. No explicit statement that scaling model size or data is the primary driver of gains.

---

## 11. Architectural Workarounds

- **ALiBi linear bias in attention scores** (purpose: enable extrapolation without positional embeddings and keep runtime/memory low):
  - "ALiBi does not add positional embeddings to word embeddings; instead, it biases query-key attention scores with a penalty that is proportional to their distance." (Abstract, p.1)
  - "The only modification we apply is after the query-key dot product, where we add a static, non-learned bias" (§3, p.5)
  - Implementation detail: "We implement it by modifying the mask matrix by adding the linear biases to it... there is no runtime penalty" (§3, p.5)
- **Causal masking** (standard LM constraint): "a 'causal mask' that ensures each position’s prediction is influenced only by tokens to its left." (§2.1, p.2)
- **Evaluation workarounds for long context** (nonoverlapping or sliding window evaluation):
  - "segment the sequence into L-length subsequences and train on or evaluate them independently" (§2, p.2)
  - "the sliding window approach is much slower" (Appendix B.1, p.23)
- **No windowed attention, hierarchical stages, token pooling/merging, or task-specific heads are described**; not specified in the paper.

---

## 12. Explicit Limitations and Non-Claims

- **Limitations / future work**:
  - "Our analysis reveals that when Lvalid > L, ALiBi might not be using contexts longer than the ones it was trained on. This highlights a research direction that could be pursued in future work." (Appendix B.2, p.24)
  - "We hypothesize that future work building on ALiBi might achieve further gains by more efficiently exploiting longer histories." (§4.2, p.6-7)
  - Sliding window evaluation is slow: "we note that it is normally prohibitively slow in practice" (Appendix B.1, p.23)
- **Explicit statements of what the model does not attempt to do** (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified in the paper.

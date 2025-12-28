### 1. Basic Metadata
Title: Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding.
Evidence: "Length Extrapolation of Transformers:
A Survey from the Perspective of Positional Encoding" (page 9959)

Authors: Liang Zhao; Xiachong Feng; Xiaocheng Feng; Weihong Zhong; Dongliang Xu; Qing Yang; Hongtao Liu; Bing Qin; Ting Liu.
Evidence: "Liang Zhao1 , Xiachong Feng2 , Xiaocheng Feng1,3 * , Weihong Zhong1 ,
Dongliang Xu4 , Qing Yang4 , Hongtao Liu4 , Bing Qin1,3 , Ting Liu1" (page 9959)

Year: 2024.
Evidence: "Findings of the Association for Computational Linguistics: EMNLP 2024" (page 9959)

Venue: Findings of the Association for Computational Linguistics: EMNLP 2024.
Evidence: "Findings of the Association for Computational Linguistics: EMNLP 2024" (page 9959)

### 2. One-Sentence Contribution Summary
This paper is a survey that organizes and analyzes methods for Transformer length extrapolation from the perspective of positional encoding, including extrapolatable PEs, position interpolation, randomized PEs, and challenges/future directions.

### 3. Tasks Evaluated

Task: QA (LongBench-E)
Task type: Generation (question answering).
Datasets: 2WikiMQA; HotpotQA; MultiFieldQA-en.
Domain: NLP / natural language text.
Evidence (Appendix A.1 / Table 2):
"LongBench-E (Bai et al., 2023b) as our testbed
and choose three trending LLMs with different con-
text window sizes to evaluate their performance on
various generation tasks and different evaluation
length ranges." (Appendix A.1, page 9974)
"QA
 2WikiMQA
 HotpotQA
 MultiFieldQA-en" (Table 2, page 9975)

Task: Summarization
Task type: Generation.
Datasets: MultiNews; GovReport.
Domain: NLP / natural language text.
Evidence (Table 2):
"Summarization
 MultiNews
 GovReport" (Table 2, page 9975)

Task: Code Completion
Task type: Generation.
Datasets: LCC.
Domain: Code (as labeled).
Evidence (Table 2):
"Code Completion
 LCC" (Table 2, page 9975)

Task: Language Modeling
Task type: Generation (language modeling).
Datasets: WikiText-103; OpenWebText2; ArXiv.
Domain: NLP / natural language text.
Evidence (Appendix A.2 / Table 3):
"A.2 Results on Language Modeling" (Appendix A.2, page 9975)
"Dataset                                WikiText-103                  OpenWebText2              ArXiv" (Table 3, page 9976)

### 4. Domain and Modality Scope
Single domain or multiple domains within the same modality: Multiple tasks within the same modality (NLP / language).
Evidence: "though Transformer-based large language models
(LLMs) (Touvron et al., 2023a; OpenAI, 2023)
have drastically advanced the NLP field." (page 9959)
Multiple modalities: Not specified in the paper.
Domain generalization / cross-domain transfer: Not claimed.

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| QA (LongBench-E) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Summarization | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Code Completion | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Language Modeling | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

### 6. Input and Representation Constraints
- Maximum length in training: "Transformer-based models are trained on se-
quences with a maximum length (Raffel et al.,
2020; Zhang et al., 2020; Brown et al., 2020), as a
result of the quadratic memory and computational
complexity with regard to input length." (page 9959)
- Input representation: "Given an input matrix X ∈ Rn×d as a sequence of
n embeddings with dimension d, an encoder layer" (Section 2 Preliminary, page 9960)
- Variable length in theory: "we have not
imposed any limit on input length n, which means
the Transformer is naturally equipped with a notion
of length extrapolation." (Section 2 Preliminary, page 9960)
- Fixed/variable input resolution, fixed patch size, fixed number of tokens, padding/resizing requirements: Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length / context window: "Transformer-based models are trained on se-
quences with a maximum length (Raffel et al.,
2020; Zhang et al., 2020; Brown et al., 2020)" (page 9959). Evaluation uses model context windows like "Llama2-7B-Chat (4K) ChatGLM3-6B (8K) Vicuna-v1.5-7b-16k" (Table 2, page 9975). The paper also notes that "LLMs are claimed to be capable of processing se-
quences with up to 128k tokens" (Section 5, page 9966).
- Fixed or variable sequence length: "we have not
imposed any limit on input length n, which means
the Transformer is naturally equipped with a notion
of length extrapolation." (Section 2 Preliminary, page 9960)
- Attention type: Standard full self-attention is implied by the dot-product attention definition and softmax over compatibility scores. Evidence: "the
row-wise softmax function converts compatibility
scores into weights, and the weighted sum of the
values is the output of the attention layer" (Section 2 Preliminary, page 9960)
- Mechanisms to manage computational cost: "quadratic memory and computational
complexity with regard to input length" (page 9959); "efficient Transformer variants
(Tay et al., 2022; Fournier et al., 2023) mainly aim
at improving the quadratic complexity of attention
mechanism" and "Flash Attention (Dao et al.,
2022; Dao, 2023) greatly improves both training
and inference efficiency of Transformers with little
to no overhead" (Section 6.1, page 9966).

### 8. Positional Encoding (Critical Section)
The paper surveys multiple positional encodings rather than using a single PE. Table 1 lists the mechanisms, their integration, and the injection layer.
Evidence (Table 1 examples):
"Sinusoidal (Vaswani et al., 2017)    Embedding       ✕         Add         Initial" (Table 1, page 9960)
"SHAPE (Kiyono et al., 2021)          Embedding       ✕         Add         Initial" (Table 1, page 9960)
"Shaw et al. (2018)                   Embedding       ✓        Add          Every" (Table 1, page 9960)
"RoPE (Su et al., 2024)               Embedding       ✕       Multiply      Every" (Table 1, page 9960)
"xPOS (Sun et al., 2023)              Embedding       ✕       Multiply      Every" (Table 1, page 9960)
Where applied: "Injection
Layer shows the injecting position PE." (Table 1 caption, page 9960)
Fixed across experiments / modified per task / ablations: Not specified in the paper.

### 9. Positional Encoding as a Variable
- PE is a core research variable: "from
the perspective of positional encoding (PE), as
it has been considered the primary factor on
length extrapolation." (Abstract, page 9959)
- Multiple PEs compared: "Table 1: A list of extrapolatable PEs." (page 9960)
- Claim that PE choice is not critical: Not claimed.

### 10. Evidence of Constraint Masking
- Model sizes reported: "Llama2-7B-Chat (4K) ChatGLM3-6B (8K) Vicuna-v1.5-7b-16k" (Table 2, page 9975).
- Dataset sizes: Not specified in the paper.
- Scaling vs. training tricks: "Anil et al. (2022) find that
fine-tuning regime, scaling data, model sizes, and
compute does not improve length generalization,
while scratchpad (Nye et al., 2022) or chain-of-
thought (Wei et al., 2022) in the in-context learning
regime do." (Section 6.2, page 9966)

### 11. Architectural Workarounds
- Recurrent/efficient variants and Flash Attention to reduce cost: "recurrent Transformer variances integrate re-
currence with attention ... efficient Transformer variants
(Tay et al., 2022; Fournier et al., 2023) mainly aim
at improving the quadratic complexity of attention
mechanism" and "Flash Attention (Dao et al.,
2022; Dao, 2023) greatly improves both training
and inference efficiency of Transformers with little
to no overhead" (Section 6.1, page 9966).
- Alternatives that abandon attention complexity: "research efforts that attempt to abandon attention and
its quadratic complexity with regard to sequence
length completely, such as S4 (Gu et al., 2022),
RWKV (Peng et al., 2023a), and Hyena (Poli
et al., 2023)." (Section 6.1, page 9967)

### 12. Explicit Limitations and Non-Claims
- Limitations on evaluation granularity: "due to the lack of
standardized benchmark and evaluation methods,
we primarily focus on high-level comparisons and
distinctions in principle of different approaches,
rather than fine-grained empirical analysis." (Limitation, page 9967)
- Scope limitation: "we focus on length extrapola-
tion studies aimed at extending the context window
of LLMs in real-world scenarios." (Limitation, page 9967)
- Non-claim / limited coverage of synthetic tasks: "Although we
acknowledge the importance of studies analyzing
length generalization in synthetic tasks within a
small context window as well, we provide only a
brief discussion on them due to the page limitation." (Limitation, page 9967)

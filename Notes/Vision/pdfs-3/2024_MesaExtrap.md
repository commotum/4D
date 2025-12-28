## 1. Basic Metadata
- Title: "Mesa-Extrapolation: A Weave Position Encoding Method for Enhanced Extrapolation in LLMs" (p. 1)
- Authors: "Xin Ma", "Yang Liu", "Jingjing Liu", "Xiaoxu Ma" (p. 1)
- Year: 2024 ("38th Conference on Neural Information Processing Systems (NeurIPS 2024).", p. 1)
- Venue: "38th Conference on Neural Information Processing Systems (NeurIPS 2024)." (p. 1)

## 2. One-Sentence Contribution Summary
The paper proposes Mesa-Extrapolation, a weave positional encoding plug-in, to address the "extrapolation problem" where "the inference ability of LLMs sharply declines beyond their max training lengths" by using a chunk-based triangular attention matrix with Stair PE for improved length extrapolation without extra training (p. 1).

## 3. Tasks Evaluated

### Task: Passkey Retrieval
- Task type: Other (passkey retrieval / information retrieval)
- Dataset(s): Generated passkey dataset
- Domain: Synthetic text
- Evidence: "We assess the accuracy of Mesa-Extrapolation using the generated passkey dataset. This dataset comprises samples of varying lengths, each storing a random password at a random position." (p. 7) "The LLM is required to find the correct password from the sample." (p. 15)

### Task: Language Modeling (Perplexity / NLL)
- Task type: Generation
- Dataset(s): Pile
- Domain: Natural language text (diverse text)
- Evidence: "We further assess the fluency of Mesa-Extrapolation utilizing the perplexity metric. Results evaluated on the Pile dataset are presented in Fig.4." (p. 8) "The pile: An 800gb dataset of diverse text for language modeling." (p. 11)

### Task: Summarization
- Task type: Generation
- Dataset(s): GovReport
- Domain: Long-form reports / documents
- Evidence: "We conduct a summary task using the GovReport dataset and employ ROUGE ROUGE (2004) (ROUGE-1/2/L) as evaluation metrics." (p. 9) "In this experiment, task is to generate a summary for texts of varying lengths, limited to 1000 tokens." (p. 22)

### Task: LongEval Lines Task
- Task type: Other (lines task / long-text evaluation)
- Dataset(s): LongEval
- Domain: Long-form text
- Evidence: "We conduct additional testing on LongEval Krishna et al. (2023) lines task, a recently prominent evaluation task for long texts." (p. 19)

### Task: Single-Document QA (LongBench)
- Task type: Other (QA)
- Dataset(s): qasper (LongBench)
- Domain: Natural language text
- Evidence: "We select LongBench Bai et al. (2023) dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (p. 20)

### Task: Multi-Document QA (LongBench)
- Task type: Other (QA)
- Dataset(s): hotpotqa (LongBench)
- Domain: Natural language text
- Evidence: "We select LongBench Bai et al. (2023) dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (p. 20)

### Task: Few-shot Learning (LongBench)
- Task type: Other (few-shot learning)
- Dataset(s): samsum (LongBench)
- Domain: Natural language text
- Evidence: "We select LongBench Bai et al. (2023) dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (p. 20)

### Task: Synthesis Tasks (LongBench)
- Task type: Generation
- Dataset(s): passage-retrieval-en (LongBench)
- Domain: Natural language text
- Evidence: "We select LongBench Bai et al. (2023) dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (p. 20)

### Task: Code Completion (LongBench)
- Task type: Generation
- Dataset(s): repobench-p (LongBench)
- Domain: Code text
- Evidence: "We select LongBench Bai et al. (2023) dataset and use 5 major categories of tasks, including Single-Document QA, Multi-Document QA, Few-shot Learning, Synthesis Tasks and Code Completion. Among them, each task selects a dataset, namely qasper, hotpotqa, samsum, passage-retrieval-en, and repobench-p." (p. 20)

### Task: Needle-in-a-Haystack (NIAH) Retrieval
- Task type: Other (needle-in-a-haystack retrieval)
- Dataset(s): Ruler datasets (single-keys NIAH)
- Domain: Long-text retrieval
- Evidence: "We further conducted experimental validation on the Ruler datasets Hsieh et al. (2024), focusing on the single-keys NIAH task. The needle-in-a-haystack (NIAH) test assesses the ability to retrieve a specific piece of information (the “needle”) from long distractor texts (the “haystack”)." (p. 28)

## 4. Domain and Modality Scope
- Modality scope: Single modality (text). Evidence of multiple text datasets: "We choose GovReport Huang et al. (2021), Pile Gao et al. (2020), LongBench Bai et al. (2023), and LongEval Krishna et al. (2023) datasets" (p. 7); "We further conducted experimental validation on the Ruler datasets" (p. 28).
- Domain scope: Multiple domains within the same modality (multiple text datasets and tasks as above). Evidence: "We choose GovReport... Pile... LongBench... and LongEval... datasets" (p. 7) and "Ruler datasets" (p. 28).
- Domain generalization / cross-domain transfer: Not claimed in the paper.

## 5. Model Sharing Across Tasks
Evidence for training/fine-tuning status: "Since our method is completely free plug-in and does not require fine-tuning" (p. 7); "Mesa-Extrapolation is a plug-and-play method that does not require additional fine-tuning." (p. 10)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Passkey Retrieval | Not specified in the paper | No (does not require fine-tuning) | Not specified in the paper | "completely free plug-in and does not require fine-tuning" (p. 7) |
| Language Modeling (PPL/NLL) | Not specified in the paper | No (does not require fine-tuning) | Not specified in the paper | "does not require fine-tuning" (p. 7) |
| Summarization (GovReport) | Not specified in the paper | No (does not require fine-tuning) | Not specified in the paper | "does not require fine-tuning" (p. 7) |
| LongEval Lines Task | Not specified in the paper | No (does not require fine-tuning) | Not specified in the paper | "does not require fine-tuning" (p. 7) |
| LongBench Tasks (S-Doc QA, M-Doc QA, Few-shot, Synthesis, Code Completion) | Not specified in the paper | No (does not require fine-tuning) | Not specified in the paper | "does not require fine-tuning" (p. 7) |
| NIAH (Ruler) | Not specified in the paper | No (does not require fine-tuning) | Not specified in the paper | "does not require fine-tuning" (p. 7) |

## 6. Input and Representation Constraints
- Max window length constraint is explicit: "Let M be the max window length for LLM." (p. 4)
- Input length is variable (token sequences): "Input: s[0 : T − 1] (input tokens with length T)" (p. 6); "The sample length initiates at 1024 and increments by 1024." (p. 7)
- Chunking and sequence segmentation constraint: "We segment the input sequence into several sub-sequences according to DynamicSplit function... The length of each sub-sequence is determined by both the input token length and the max training length." (p. 6)
- Fixed patch size / fixed number of tokens / padding or resizing: Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper (the paper defines token sequences and Transformer notation rather than spatial dimensionality).

## 7. Context Window and Attention Structure
- Maximum sequence length (training window) examples:
  - LLaMA2-7B-Chat: "The vertical black dashed line indicate the position of maximum training length of the model. In this case, it is 4k for LLaMA2-7B-Chat model." (p. 5)
  - MPT-7B: "The white dashed line represents MPT-7B’s maximum training length at 2k." (p. 18)
  - Phi-3-mini-128k-instruct: "officially claimed to support extrapolation lengths of up to 128k tokens." (p. 28)
- Sequence length fixed or variable: Variable. Evidence: "Input: s[0 : T − 1] (input tokens with length T)" (p. 6) and "The sample length initiates at 1024 and increments by 1024." (p. 7)
- Attention type: Other (chunk-based triangular attention matrix). Evidence: "We design a chunk-based triangular attention matrix" (p. 6).
- Mechanisms to manage computational cost: "To achieve approximate linear memory consumption and computational speed, we further split the triangular attention matrix into several chunks and concatenate these chunks." (p. 6)

## 8. Positional Encoding (Critical Section)
- Mechanism: Relative PE (RoPE / ALiBi) with weave PE (Stair PE).
  - "RoPE... rotates the query and key vectors with an angle proportional to their absolute positions, so the attention dot production only depends on relative distance between tokens" (p. 3)
  - "ALiBi... subtracts a scalar bias from the attention score" (p. 3)
  - "we define a novel weave PE method, namely Stair PE" (p. 5)
  - "Stair PE can be applied to existing relative PEs such as RoPE and ALiBi." (p. 6)
- Where applied: The paper defines PE in the attention dot product (query-key) (e.g., "⟨qt , ki ⟩ := fPE (qt , ki , t − i)" (p. 3)). Application to input-only vs every layer is not specified in the paper.
- Fixed vs modified across experiments: Multiple PEs are compared/ablated.
  - "Since our method is completely free plug-in and does not require fine-tuning, we choose methods of this type for comparison, including: model self (Origin), ReRoPE... Leaky-ReRoPE... Dynamic-NTK... LM-Infinite... Streaming-LLM" (p. 7)
  - "We design ablation experiment about weave PE based methods... applying different weaving PE methods" (p. 21)
- Mesa-Extrapolation usage detail: "regular PE (such as RoPE or ALiBi) is applied to all chunks except for the last chunk, for which Stair PE is applied." (p. 6)

## 9. Positional Encoding as a Variable
- PE is a core research variable: The paper centers on weave PE and introduces a new PE method.
  - "we define a novel weave PE method, namely Stair PE" (p. 5)
  - "we choose methods... including... ReRoPE... Leaky-ReRoPE... Dynamic-NTK..." (p. 7)
- Multiple positional encodings compared: Yes (Origin, ReRoPE, Leaky-ReRoPE, Dynamic-NTK, etc.). Evidence: "we choose methods... including: model self (Origin), ReRoPE... Leaky-ReRoPE... Dynamic-NTK..." (p. 7)
- Claim that PE is “not critical”: Not claimed in the paper.

## 10. Evidence of Constraint Masking (Scale vs. Structure)
- Model sizes used (evidence of scale): "LLaMA-3B... LLaMA2-7B-Chat, and Vicuna-13B-V1.3... MPT-7B... PyThia-6.9B and PyThia-12B." (p. 7)
- Dataset size details: The Pile is cited as "An 800gb dataset of diverse text for language modeling." (p. 11) Passkey dataset size by sampling: "100 samples are randomly generated for each length." (p. 7)
- Performance gains attributed to architectural changes (weave PE + chunking), not scaling data/model:
  - "Our theorems establish that LLMs equipped with weave PE can achieve improved extrapolation performance without additional cost." (p. 1)
  - "we introduce a novel weave PE method, Mesa-Extrapolation, which utilizes a chunk-based triangular attention matrix and applies Stair PE" (p. 1)
  - "Mesa-Extrapolation... does not require additional training" (p. 2)
- Claims that gains are primarily from scaling model size or data: Not specified in the paper.

## 11. Architectural Workarounds
- Chunk-based triangular attention matrix to manage memory/speed: "We design a chunk-based triangular attention matrix... To achieve approximate linear memory consumption and computational speed, we further split the triangular attention matrix into several chunks and concatenate these chunks." (p. 6)
- Dynamic splitting of sequences into chunks: "We segment the input sequence into several sub-sequences according to DynamicSplit function" (p. 6)
- Stair PE to extend effective window for the last chunk: "Stair PE is applied" to the last chunk to "rearrange relative positional encoding to achieve extrapolation beyond the effective window length." (p. 6)
- Regular PE for earlier chunks, Stair PE for last chunk: "regular PE (such as RoPE or ALiBi) is applied to all chunks except for the last chunk, for which Stair PE is applied." (p. 6)

## 12. Explicit Limitations and Non-Claims
- Limitations / future work: "Mesa-Extrapolation is a plug-and-play method that does not require additional fine-tuning. However, previous work... shows that applying further fine-tuning... is possible. Therefore exploring fine-tuning based on Mesa-Extrapolation can be an interesting next step. Due to limitations of resources, we have not yet validated our method at longer lengths." (p. 10)
- Explicit non-claims about open-world / unrestrained multi-task learning: Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Multiple datasets within a single modality (text), e.g., "GovReport... Pile... LongBench... LongEval..." and "Ruler datasets" (p. 7, p. 28).
- Task structure: Multiple long-context text tasks (retrieval, summarization, QA, code completion) rather than a single task.
- Representation rigidity: Token sequences with a max window length M ("Let M be the max window length for LLM." p. 4); variable lengths via chunking (p. 6–7).
- Model sharing vs specialization: Method is a plug-in without fine-tuning, but task-specific heads are not described ("does not require fine-tuning" p. 7).
- Positional encoding role: Central research variable with new weave PE (Stair PE) and comparisons against other PE schemes (p. 5–7, p. 21).

## 14. Final Classification
**Multi-task, single-domain.** The paper evaluates multiple text tasks, including passkey retrieval, language modeling on Pile, summarization on GovReport, LongEval lines, LongBench QA/few-shot/synthesis/code completion, and NIAH retrieval (p. 7, p. 19–20, p. 28). All evaluations are within the text modality (single domain of language data), and no multi-modality or cross-domain transfer is claimed.

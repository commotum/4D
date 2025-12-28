## 1. Basic Metadata

Title: Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free.
Evidence: "Gated Attention for Large Language Models: Non-linearity, Sparsity,
and Attention-Sink-Free" (Title page)

Authors: Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, Junyang Lin.
Evidence: "Zihan Qiu*1 , Zekun Wang*1 , Bo Zheng*1 , Zeyu Huang*2 ,
Kaiyue Wen3 , Songlin Yang4 , Rui Men1 , Le Yu1 , Fei Huang1 , Suozhi Huang5 ,
Dayiheng LiuB1 , Jingren Zhou1 , Junyang LinB1" (Title page)

Year: 2025.
Evidence: "arXiv:2505.06708v1 [cs.CL] 10 May 2025" (Title page)

Venue (conference/journal/arXiv): arXiv (preprint).
Evidence: "arXiv:2505.06708v1 [cs.CL] 10 May 2025" (Title page)

## 2. One-Sentence Contribution Summary

The paper claims that adding a head-specific sigmoid gate after SDPA in standard softmax attention improves performance and training stability by introducing non-linearity and sparse, query-dependent modulation that reduces attention sink effects.

## 3. Tasks Evaluated

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling perplexity (PPL) | Other (language modeling perplexity) | Held-out test sets (names not specified) | English, Chinese, Code, Math, Law, Literature | "We also report the perplexity (PPL) of language modeling on diverse held-out test sets, including domains like English, Chinese, Code, Math, Law, and Literature." (Section 3.1 Experimental Setups) |
| Hellaswag | Other (English benchmark; task type not specified in the paper) | Hellaswag | English | "Hellaswag (Zellers et al., 2019) for English" (Section 3.1 Experimental Setups) |
| MMLU | Other (general knowledge benchmark; task type not specified in the paper) | MMLU | general knowledge | "MMLU (Hendrycks et al., 2020) for general knowledge" (Section 3.1 Experimental Setups) |
| GSM8k | Reasoning / relational (math reasoning) | GSM8k | math reasoning | "GSM8k (Cobbe et al., 2021) for math reasoning" (Section 3.1 Experimental Setups) |
| HumanEval | Other (coding benchmark; task type not specified in the paper) | HumanEval | coding | "HumanEval (Chen et al., 2021) for coding" (Section 3.1 Experimental Setups) |
| C-eval | Other (Chinese proficiency benchmark; task type not specified in the paper) | C-eval | Chinese proficiency | "C-eval (Huang et al., 2024) and CMMLU (Li et al., 2023) for Chinese proficiency." (Section 3.1 Experimental Setups) |
| CMMLU | Other (Chinese proficiency benchmark; task type not specified in the paper) | CMMLU | Chinese proficiency | "C-eval (Huang et al., 2024) and CMMLU (Li et al., 2023) for Chinese proficiency." (Section 3.1 Experimental Setups) |
| RULER | Other (long-context benchmark; task type not specified in the paper) | RULER | Not specified in the paper. | "We evaluate models on the RULER benchmark (Hsieh et al., 2024) and summarize results in Tab. 5." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension) |

## 4. Domain and Modality Scope

Single domain? No. Multiple domains are explicitly listed.
Evidence: "We also report the perplexity (PPL) of language modeling on diverse held-out test sets, including domains like English, Chinese, Code, Math, Law, and Literature." (Section 3.1 Experimental Setups)

Multiple domains within the same modality? The paper explicitly lists multiple domains but does not explicitly name the modality.
Evidence: "including domains like English, Chinese, Code, Math, Law, and Literature." (Section 3.1 Experimental Setups)

Multiple modalities? Not specified in the paper.

Domain generalization or cross-domain transfer claimed? Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling perplexity (PPL) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We also report the perplexity (PPL) of language modeling on diverse held-out test sets, including domains like English, Chinese, Code, Math, Law, and Literature." (Section 3.1 Experimental Setups) |
| Hellaswag | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Hellaswag (Zellers et al., 2019) for English" (Section 3.1 Experimental Setups) |
| MMLU | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "MMLU (Hendrycks et al., 2020) for general knowledge" (Section 3.1 Experimental Setups) |
| GSM8k | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "GSM8k (Cobbe et al., 2021) for math reasoning" (Section 3.1 Experimental Setups) |
| HumanEval | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "HumanEval (Chen et al., 2021) for coding" (Section 3.1 Experimental Setups) |
| C-eval | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "C-eval (Huang et al., 2024) and CMMLU (Li et al., 2023) for Chinese proficiency." (Section 3.1 Experimental Setups) |
| CMMLU | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "C-eval (Huang et al., 2024) and CMMLU (Li et al., 2023) for Chinese proficiency." (Section 3.1 Experimental Setups) |
| RULER | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We evaluate models on the RULER benchmark (Hsieh et al., 2024) and summarize results in Tab. 5." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension) |

## 6. Input and Representation Constraints

Context sequence length: "The context sequence length is set to 4096." (Section 3.1 Experimental Setups)

Long-context training/extension: "sequence length of 32k for an additional 80B tokens. This gives us models with a context length of 32k. Subsequently, we use YaRN (Peng et al., 2023) to extend the context length to 128k." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

Fixed or variable input resolution? Not specified in the paper.

Fixed patch size? Not specified in the paper.

Fixed number of tokens? Not specified in the paper.

Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper.

Padding or resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure

Maximum sequence length (training): "The context sequence length is set to 4096." (Section 3.1 Experimental Setups)

Maximum sequence length (extended): "This gives us models with a context length of 32k. Subsequently, we use YaRN (Peng et al., 2023) to extend the context length to 128k." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

Fixed or variable sequence length? Not specified in the paper.

Attention type: "Scaled Product Dot-Product Attention (SDPA): computes attention scores between queries and keys, followed by a softmax normalization." (Section 2.1 Preliminary: Multi-Head Softmax Attention)

Mechanisms to manage computational cost (windowing, pooling, pruning, etc.)? Not specified in the paper. The only explicit compute-cost statement is: "Since the parameters and flops introduced by the gating are small, the wall-time latency introduced by gating is less than 2%." (Section 3.1 Experimental Setups)

## 8. Positional Encoding (Critical Section)

Mechanism used: RoPE, with YaRN for extension.
Evidence: "We increase
the RoPE (Su et al., 2024)
base from 10k to 1M and con-
tinue training on data with a
sequence length of 32k for an additional 80B tokens. This gives us models with a context length of 32k.
Subsequently, we use YaRN (Peng et al., 2023) to extend the context length to 128k." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

Where it is applied (input only, every layer, attention bias, etc.)? Not specified in the paper.

Fixed across all experiments or modified per task? Modified in the context-length extension setting (RoPE base change and YaRN). Evidence is the quote above.

Ablated or compared against alternatives? Not specified in the paper.

## 9. Positional Encoding as a Variable

Core research variable or fixed architectural assumption? Not specified in the paper. The only explicit discussion is in the context-length extension setup.
Evidence: "We increase
the RoPE (Su et al., 2024)
base from 10k to 1M and con-
tinue training on data with a
sequence length of 32k for an additional 80B tokens. This gives us models with a context length of 32k.
Subsequently, we use YaRN (Peng et al., 2023) to extend the context length to 128k." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

Multiple positional encodings compared? Not specified in the paper.

Any claim that PE choice is not critical or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs. Structure)

Model sizes: "We conduct experiments on both MoE models (15B total pa-
rameters with 2.54B activated, 15A2B) and dense models (1.7B total parameters)." (Section 3.1 Experimental Setups)

Dataset sizes: "We train the 15A2B MoE models on 400B tokens." (Table 1 caption)

Dataset sizes (dense models): "for the 1.7B model trained on
400B tokens, we use a maximum LR of 4e-3 and a bsz of 1024. For training on 3.5T tokens, we increase the
maximum LR to 4.5e-3 and the bsz to 2048." (Section 3.2.2 Gated Attention for Dense Models)

Performance gains attributed to architectural gating rather than scaling parameters: "we find that: (i) applying
SDPA output head-specific gating (G1 ) yields the most significant performance improvements (e.g., up to
0.2 PPL reduction and 2 points on MMLU); (ii) the SDPA output gating also improves training stability,
nearly eliminating loss spikes, enabling larger learning rates and enhancing model scalability." (Section 1 Introduction)

Control against parameter scaling: "To provide a fair comparison, we supplement the vanilla MoE baseline (row 1) with parameter
expansion methods, including increasing the number of key-value heads (row 2), increasing the number
of query heads (row 3), and increasing both the total and activated number of experts (row 4). These
methods introduce a comparable or greater number of parameters than the gating mechanisms." (Section 3.2.1 Gated Attention for MoE models)

## 11. Architectural Workarounds

Gating after SDPA (head-specific): "applying
SDPA output head-specific gating (G1 ) yields the most significant performance improvements" (Section 1 Introduction)

Group Query Attention: "We adopt group query attention (GQA) (Ainslie et al., 2023) for the attention part." (Section 3.1 Experimental Setups)

MoE configuration and routing: "The 15A2B MoE models utilize 128 total experts with top-8 softmax gating, fine-grained experts (Dai et al., 2024), global-batch
LBL (Qiu et al., 2025), and z-loss (Zoph et al., 2022)." (Section 3.1 Experimental Setups)

Parameter-parity adjustment: "When using gating, we reduce the FFN's width so that all methods have the same number of parameters." (Table 2 caption)

Stability workarounds tested: "we introduce a clipping operation to constrain the outputs of attention and FFN layers before they enter the residual connection,
limiting their values to the range (-clip, clip)." (Appendix A.5 Other Attempt to Stabilize Training)

Context-length extension mechanism: "Subsequently, we use YaRN (Peng et al., 2023) to extend the context length to 128k." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

## 12. Explicit Limitations and Non-Claims

Limitations stated: "Our work primarily focuses on analyzing the reasons and impacts of attention gating through a series of
ablation studies. However, we acknowledge several limitations. The broader implications of non-linearity
on the dynamics of attention and the overall training process remain under-explored. Although we
observe that eliminating attention sinks improves performance in long-context extension scenarios, we
do not provide a rigorous theoretical explanation for how attention sinks influence the model's ability to
generalize to longer sequences." (Limitations section)

Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

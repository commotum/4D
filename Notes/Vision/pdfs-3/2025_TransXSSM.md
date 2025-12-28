## 1. Basic Metadata
- Title: "TransXSSM: A Hybrid Transformer–State Space Model with Unified Rotary Position Embedding" (p.1, title)
- Authors: "Bingheng Wu1 Jingze Shi1 Yifan Wu1" / "Nan Tang1 Yuyu Luo1" (p.1, author list)
- Year: "18 Jun 2025" (p.1, arXiv line)
- Venue: "arXiv:2506.09507v3 [cs.CL] 18 Jun 2025" and "Preprint. Under review." (p.1)

## 2. One-Sentence Contribution Summary
The paper proposes a "unified rotary position embedding (Unified RoPE) methodology" that establishes "a consistent positional encoding framework for both self-attention and state-space components," and introduces "TransXSSM, a hybrid architecture that coherently integrates the Transformer and SSM layers under this unified positional encoding scheme" (Abstract, p.1).

## 3. Tasks Evaluated
### Task: Language modeling benchmarks
- Task type: Generation.
- Dataset(s): Not specified in the paper.
- Domain: Language / text.
- Evidence:
  > It also delivers higher accuracy: under comparable
  > settings, it surpasses a Transformer baseline by over 4% on language modeling
  > benchmarks. (Abstract, p.1)
  > For language modeling, TransXSSM follows the standard Transformer
  > architecture outline with modifications to include SSM layers. (Section 3)

### Task: Long-context retrieval ("needle in a haystack")
- Task type: Other (retrieval).
- Dataset(s): Synthetic retrieval task (no dataset name given).
- Domain: Language / text.
- Evidence:
  > Long-Context Retrieval (Needle-in-a-Haystack Task). We further evaluated architectures on the
  > “needle in a haystack” synthetic retrieval task. This task tests long-context extraction by embedding
  > a “needle” (random sentence) in a “haystack” (long document) for retrieval. (Section 4.2)

### Task: MMLU
- Task type: Not specified in the paper.
- Dataset(s): MMLU.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

### Task: TriviaQA
- Task type: Not specified in the paper.
- Dataset(s): TriviaQA.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

### Task: ARC
- Task type: Not specified in the paper.
- Dataset(s): ARC.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

### Task: PIQA
- Task type: Not specified in the paper.
- Dataset(s): PIQA.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

### Task: HellaSwag
- Task type: Not specified in the paper.
- Dataset(s): HellaSwag.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

### Task: OBQA
- Task type: Not specified in the paper.
- Dataset(s): OBQA.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

### Task: Winogrande
- Task type: Not specified in the paper.
- Dataset(s): Winogrande.
- Domain: Language / text.
- Evidence:
  > Downstream Tasks Settings. Downstream task evaluation utilized the EleutherAI LM Evaluation
  > Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
  > HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)

## 4. Domain and Modality Scope
- Domain scope: Single domain (language/text).
  > Effective positional encoding is crucial for sequence modeling in language tasks, as it underpins
  > a model’s ability to understand order, perform reasoning, and handle long contexts. (Section 1)
- Multiple domains within same modality: Not specified in the paper.
- Multiple modalities: Not specified in the paper.
- Domain generalization/cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
Evidence used for all rows:
> All models were trained from scratch on the Smollm-Corpus [21] dataset
> using the NeoX tokenizer [22]. (Appendix B.1)
> Downstream task evaluation utilized the EleutherAI LM Evaluation
> Harness [24]. The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28],
> HellaSwag [29], OBQA [30], and Winogrande [31]. (Appendix B.1)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Language modeling benchmarks | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| Needle-in-a-haystack retrieval | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| MMLU | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| TriviaQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| ARC | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| PIQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| HellaSwag | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| OBQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |
| Winogrande | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Quotes above (Appendix B.1). |

## 6. Input and Representation Constraints
- Sequence length settings (explicit in experiments):
  > Models with dmodel = 256 and sequence length 8192 were trained for 8000 steps. (Section 4.1)
- Maximum context length reported:
  > Empirically, we found this design choice crucial for training TransXSSM on contexts up to 16K
  > tokens without divergence. (Section 3)
- Fixed dimensionality / model size settings:
  > We experimented with two model scales, 320M and 1.3B parameters,
  > detailed in Table 4. For the 320M scale, models have dmodel = 768, 24 layers, 12 attention heads, a
  > learning rate of 3e-4, and a batch size of 1M tokens. For the 1.3B scale, dmodel = 2048, 24 layers,
  > 32 attention heads, a learning rate of 2e-4, and a batch size of 2M tokens. State-Space components
  > within Mamba2, Jamba, and TransXSSM uniformly use dstate = 128 and chunk_len = 256.
  > (Appendix B.1)
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Sequence length is fixed per experiment where specified (8192; up to 16K max), but variability across settings is reported. (Evidence above)
- Padding/resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Up to 16K tokens.
  > Empirically, we found this design choice crucial for training TransXSSM on contexts up to 16K
  > tokens without divergence. (Section 3)
- Fixed vs variable sequence length: Explicit fixed lengths are used per experiment (e.g., 8192), but multiple settings are reported.
  > Models with dmodel = 256 and sequence length 8192 were trained for 8000 steps. (Section 4.1)
- Attention type: Global causal self-attention.
  > Self-attention computes pairwise relevance scores between all tokens in
  > a sequence, enabling long-range dependency modeling [4]. (Section 2.1)
  > In causal language modeling, a binary lower-triangular mask L is applied to ensure each position
  > only attends to previous positions (preventing information leakage). (Section 2.1)
- Mechanisms to manage computational cost:
  > the SSM layers efficiently handle the bulk of sequence length (contributing
  > near-linear time complexity and high throughput), while the periodic attention layers inject global
  > context mixing and strong relational reasoning. (Section 3)
  > We adopt a 7:1 ratio of SS to SA blocks per module – that is, each module consists of 7
  > SSM-based sub-layers followed by 1 Transformer attention sub-layer (this ratio is motivated by prior
  > studies on hybrid models [8] and our own experiments). (Section 3)

## 8. Positional Encoding (Critical Section)
- Mechanism: Unified Rotary Position Embedding (RoPE).
  > Rotary Position Embedding (RoPE) is a technique that
  > encodes absolute positions as complex rotations applied to query and key vectors. (Section 2.1)
  > We propose a unified rotary position encoding that applies the same
  > rotational embedding to both self-attention (Transformer) and state-space (SSM) components.
  > (Section 2.2)
- Where it is applied:
  > In practice, we define four position-encoding functions fQ , fK , fC , fB for queries Q, keys K, and the
  > analogous state update vectors (which we denote C and B for the SSM’s internal update). (Section 2.2)
  > Every SS and SA sub-layer in TransXSSM uses
  > the same Unified RoPE as described in Section 2.2. Before computing attention scores or state-space
  > updates, the inputs to each layer are endowed with the RoPE phase appropriate for their position.
  > (Section 3)
- Fixed vs modified per task / ablations:
  > We aim to answer the following research
  > questions: (RQ1) Does Unified RoPE effectively unify position encoding across Transformer and
  > SSM modules, and how does it compare with other positional encodings? (Section 4)
  > Ablation studies further compared RoPE against alternatives (Conv1d + D and at ). (Section 4.1)

## 9. Positional Encoding as a Variable
- Core research variable: Yes.
  > We aim to answer the following research
  > questions: (RQ1) Does Unified RoPE effectively unify position encoding across Transformer and
  > SSM modules, and how does it compare with other positional encodings? (Section 4)
- Multiple positional encodings compared: Yes (RoPE vs Conv1d + D and at).
  > Ablation studies further compared RoPE against alternatives (Conv1d + D and at ). (Section 4.1)
- Claim PE is not critical/secondary: Not claimed.

## 10. Evidence of Constraint Masking
- Model sizes:
  > We experimented with two model scales, 320M and 1.3B parameters,
  > detailed in Table 4. (Appendix B.1)
- Scaling gains reported:
  > TransXSSM furthermore scales more effectively: TransXSSM-1.3B
  > gains 7.22% in average accuracy over its 320M version (versus about 6% gains
  > for equivalent Transformers or SSMs). (Abstract, p.1)
- Dataset size(s): Not specified in the paper (only dataset name is given).
  > All models were trained from scratch on the Smollm-Corpus [21] dataset
  > using the NeoX tokenizer [22]. (Appendix B.1)
- Attributed source of gains: Primarily architectural (unified positional encoding + hybrid design).
  > The unified position embedding provides a solid foundation for effectively fusing different
  > architectural paradigms, leading to a better balance of computational efficiency and model capability
  > at larger scales. (Section 4.3)

## 11. Architectural Workarounds
- Unified positional encoding across modules:
  > We propose a unified rotary position encoding that applies the same
  > rotational embedding to both self-attention (Transformer) and state-space (SSM) components.
  > (Section 2.2)
- Hybrid stacking ratio to balance efficiency and reasoning:
  > We adopt a 7:1 ratio of SS to SA blocks per module – that is, each module consists of 7
  > SSM-based sub-layers followed by 1 Transformer attention sub-layer (this ratio is motivated by prior
  > studies on hybrid models [8] and our own experiments). (Section 3)
- Near-linear sequence handling via SSM-heavy blocks:
  > the SSM layers efficiently handle the bulk of sequence length (contributing
  > near-linear time complexity and high throughput), while the periodic attention layers inject global
  > context mixing and strong relational reasoning. (Section 3)
- Stability mechanisms for long sequences:
  > We use RMSNorm normalization and add a residual skip connection around each
  > SS/SA block and around its subsequent FFN, which is important for stabilizing training with long
  > sequences (mitigating issues like state collapse observed in long-sequence RNNs [9]). (Section 3)
- Causal masking for autoregressive attention:
  > In causal language modeling, a binary lower-triangular mask L is applied to ensure each position
  > only attends to previous positions (preventing information leakage). (Section 2.1)

## 12. Explicit Limitations and Non-Claims
Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Language/text only; tasks are described as "language tasks" and language modeling benchmarks.
- Task structure: Multiple downstream benchmarks (MMLU, TriviaQA, ARC, PIQA, HellaSwag, OBQA, Winogrande) plus a synthetic long-context retrieval task.
- Representation rigidity: Fixed sequence lengths in experiments (8192; up to 16K max) and fixed model dimensions/SSM chunk_len.
- Model sharing vs specialization: Evaluation uses the same trained models across tasks; fine-tuning or task-specific heads are not specified.
- Positional encoding role: Unified RoPE is central and explicitly compared against alternatives.

## 14. Final Classification
Multi-task, single-domain. The paper evaluates multiple tasks within the language domain, e.g., "The benchmark suite included MMLU [25], TriviaQA [26], ARC [27], PIQA [28], HellaSwag [29], OBQA [30], and Winogrande [31]" (Appendix B.1), and frames the setting as "sequence modeling in language tasks" (Section 1). No multi-modal or cross-domain evaluation is claimed.

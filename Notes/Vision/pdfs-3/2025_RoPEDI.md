## 1. Basic Metadata

- Title: The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval
- Authors: Ting-Rui Chiang; Dani Yogatama
- Year: 2025
- Venue: arXiv (arXiv:2502.11276v1 [cs.CL])

Evidence (header, page 1): "The Rotary Position Embedding May Cause Dimension Inefficiency in Attention Heads for Long-Distance Retrieval"; "Ting-Rui Chiang"; "Dani Yogatama"; "arXiv:2502.11276v1 [cs.CL] 16 Feb 2025"

## 2. One-Sentence Contribution Summary

The paper argues that RoPE can make some attention-head dimensions low-utility for long-distance retrieval and supports this via a controlled experiment and analyses of three LLMs on long-context question answering.

Evidence (Abstract, page 1): "We hypothesize that the wide range of rotation angles may prevent LLMs from utilizing those dimensions. To validate this hypothesis, we present a controlled experiment showing that applying RoPE causes low utility of certain dimensions. Our analyses on three LLMs also indicate that these dimensions do not help LLMs do long-context question answering."

## 3. Tasks Evaluated

Task 1:
- Task name: Synthetic key-value retrieval with an attention head (controlled experiment)
- Task type: Other (specify: key-value retrieval / synthetic attention retrieval)
- Dataset(s) used: Synthetic vector tuples (no named dataset)
- Domain: Synthetic vector tuples

Task evidence (Section 4 Controlled Experiment, page 2): "We design a simple experiment where the model needs to learn n vector tuples {(qi , ki , vi )}ni=1 such that the attention head can retrieve vi with qi from any randomly sampled subset of key-value pairs {(k, v)|k ∈ K, v ∈ V } ⊂ {(ki , vi )}ni=1."

Task 2:
- Task name: Long-context question answering
- Task type: Generation; Other (specify: open-domain question answering / retrieval)
- Dataset(s) used: Processed dataset from Liu et al. (2024), derived from NaturalQuestions-Open
- Domain: English natural language documents/questions

Task evidence:
- Section 5.1 Experimental Setup (page 3): "we choose a task that involves long dependence modeling, the long-context question-answering task."
- Section 5.1 Experimental Setup (page 3): "we provide the model with 20 documents for each question, among which only one contains the answer."
- Appendix C Dataset (page 7): "We use the processed dataset from Liu et al. (2024)."
- Appendix C Dataset (page 7): "It is de-
rived from NaturalQuestions-Open (Kwiatkowski
et al., 2019; Lee et al., 2019)."
- Appendix C Dataset (page 7): "The language is English."

## 4. Domain and Modality Scope

- Single domain: Not specified in the paper. Evaluation includes a synthetic controlled experiment and an English long-context QA dataset.
  Evidence: "We design a simple experiment where the model needs to learn n vector tuples {(qi , ki , vi )}ni=1..." (Section 4, page 2) and "The language is English." (Appendix C, page 7)
- Multiple domains within the same modality: Not specified in the paper.
- Multiple modalities: Not specified in the paper.
- Domain generalization or cross-domain transfer claims: Not claimed.

## 5. Model Sharing Across Tasks

The paper uses separately trained attention models for the controlled experiment and evaluates three existing LLMs for long-context QA. No joint multi-task training or shared weights across tasks are explicitly described.

Evidence: "We train attention models with 128 hidden dimensions." (Appendix A, page 7) and "We then inspect three 7B/8B large language models (LLM), Llama-3.1-8B-Instruct (Dubey et al., 2024), QWen-2.5-7B-Instruct (Team, 2024), and OLMo-2-7B-Instruct (OLMo et al., 2024)." (Section 5, page 2)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Synthetic key-value retrieval (controlled experiment) | Not specified in the paper (separately trained attention models for this experiment) | Not specified in the paper | Not specified in the paper | "We train attention models with 128 hidden dimensions." (Appendix A, page 7) |
| Long-context question answering | Not specified in the paper (evaluation on existing LLMs) | Not specified in the paper (prompting only is described) | Not specified in the paper | "We first prompt the LLM to answer the questions in the dataset." (Section 5.2, page 3) |

## 6. Input and Representation Constraints

Explicit constraints stated:
- Controlled experiment model dimensionality: "We train attention models with 128 hidden dimensions." (Appendix A, page 7)
- Controlled experiment K/V count: "We sample 128 out of 1000 key-value pairs for the K, V in Eq. 3." (Appendix A, page 7)
- Controlled experiment maximum position: "a maximum position of 2048" (Appendix A, page 7)
- LLM attention head dimensionality: "These models have 128 dimensions in their attention heads." (Section 5, page 2)
- Long-context QA input structure: "we provide the model with 20 documents for each question, among which only one contains the answer." (Section 5.1, page 3)

Not specified in the paper:
- Fixed or variable input resolution
- Fixed patch size
- Fixed number of tokens beyond the K/V sampling in the controlled experiment
- Padding or resizing requirements

## 7. Context Window and Attention Structure

- Maximum sequence length / position: "a maximum position of 2048" (Appendix A, page 7) for the controlled experiment.
- Fixed or variable length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost (windowing, pooling, token pruning, etc.): Not specified in the paper.

## 8. Positional Encoding (Critical Section)

- Mechanism used: Rotary Position Embedding (RoPE), relative positional encoding via rotation of query/key representations.
  Evidence (Section 2 Background, page 1): "Su et al. (2024) proposed the Rotary Position Embedding (RoPE), which can be applied to the key and query vectors for attention operations. It encodes relative position by rotating the intermediate representations according to their positions in the input sequence."
- Where applied: Key and query vectors.
  Evidence (Section 2 Background, page 1): "RoPE... can be applied to the key and query vectors for attention operations."
- Fixed vs modified across experiments: RoPE is ablated in the controlled experiment (with vs without).
  Evidence (Section 4, page 2): "We train models in two setups, one with RoPE applied on K and the other without (details in §A)."

## 9. Positional Encoding as a Variable

- Positional encoding as a core research variable: Yes.
  Evidence (Abstract, page 1): "We hypothesize that the wide range of rotation angles may prevent LLMs from utilizing those dimensions. To validate this hypothesis, we present a controlled experiment showing that applying RoPE causes low utility of certain dimensions."
- Multiple positional encodings compared: Only RoPE vs no RoPE (ablation).
  Evidence (Section 4, page 2): "We train models in two setups, one with RoPE applied on K and the other without (details in §A)."
- Claim that PE choice is not critical or secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking

Model sizes:
- "We then inspect three 7B/8B large language models (LLM), Llama-3.1-8B-Instruct (Dubey et al., 2024), QWen-2.5-7B-Instruct (Team, 2024), and OLMo-2-7B-Instruct (OLMo et al., 2024)." (Section 5, page 2)

Dataset sizes:
- "we provide the model with 20 documents for each question, among which only one contains the answer." (Section 5.1, page 3)
- "There are 2655 examples in the test set." (Appendix C, page 7)

Attribution of performance gains to scaling model size/data/architecture/training tricks:
- Not specified in the paper. The paper attributes the observed effects to RoPE and dimension utilization, not to scaling.
  Evidence (Abstract, page 1): "We hypothesize that the wide range of rotation angles may prevent LLMs from utilizing those dimensions."

## 11. Architectural Workarounds

- Dimension pruning as a potential efficiency workaround: "LLMs may be made more computationally efficient by pruning these dimensions." (Introduction, page 1)
- Dimension masking procedure used for analysis: "we train a sparse mask that masks out as many dimensions as possible while preserving the attention head’s output." (Section 5.2, page 3)

Not specified in the paper:
- Windowed attention
- Hierarchical stages
- Token pooling / merging
- Task-specific heads
- Fixed grid assumptions

## 12. Explicit Limitations and Non-Claims

- Limited model coverage: "due to limited computational resources, we experiment with only three 7B/8B LLMs." (Section 7 Limitations, page 4)
- No exploration of improvements: "We do not explore how our findings could improve LLMs, such as enhancing computational efficiency." (Section 7 Limitations, page 4)
- Mitigation left for future work: "We also leave the mitigation of the dimensional deficiency for future work, as it may require significant computational resource for additional fine-tuning." (Section 7 Limitations, page 4)

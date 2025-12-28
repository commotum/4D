1. Basic Metadata
- Title: Selective Rotary Position Embedding. Evidence: "S ELECTIVE ROTARY P OSITION E MBEDDING" (p. 1).
- Authors: Sajad Movahedi; Timur Carstensen; Arshia Afzal; Frank Hutter; Antonio Orvieto; Volkan Cevher. (Listed on p. 1.)
- Year: 2025. Evidence: "arXiv:2511.17388v1 [cs.CL] 21 Nov 2025" (p. 1).
- Venue: arXiv preprint / under review. Evidence: "Preprint. Under Review." (p. 1).

2. One-Sentence Contribution Summary
- The paper introduces Selective RoPE, an input-dependent rotary position embedding for linear and softmax attention, and evaluates it on synthetic recall tasks and language modeling.

3. Tasks Evaluated
- Multi-Query Associative Recall (MQAR)
  - Task type: Other (associative recall).
  - Dataset(s): Multi-Query Associative Recall (MQAR).
  - Domain: synthetic sequence tasks.
  - Evidence: "MQAR. We evaluate GLA + Selective RoPE on Multi-Query Associative Recall" (Section 4.2, p. 7); "In the following section we test our proposed model on synthetic and real-world language modeling tasks." (Section 4.2, p. 7).
- MAD: Compress Recall
  - Task type: Other (recall).
  - Dataset(s): MAD benchmark suite.
  - Domain: synthetic sequence tasks.
  - Evidence: "MAD and Copying. We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2, p. 7); "Model                   Compress Fuzzy In-Context Memorize Noisy Selective Average" (Table 1, p. 6).
- MAD: Fuzzy Recall
  - Task type: Other (recall).
  - Dataset(s): MAD benchmark suite.
  - Domain: synthetic sequence tasks.
  - Evidence: "MAD and Copying. We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2, p. 7); "Model                   Compress Fuzzy In-Context Memorize Noisy Selective Average" (Table 1, p. 6).
- MAD: In-Context Recall
  - Task type: Other (recall).
  - Dataset(s): MAD benchmark suite.
  - Domain: synthetic sequence tasks.
  - Evidence: "MAD and Copying. We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2, p. 7); "Model                   Compress Fuzzy In-Context Memorize Noisy Selective Average" (Table 1, p. 6).
- MAD: Memorize
  - Task type: Other (recall).
  - Dataset(s): MAD benchmark suite.
  - Domain: synthetic sequence tasks.
  - Evidence: "MAD and Copying. We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2, p. 7); "Model                   Compress Fuzzy In-Context Memorize Noisy Selective Average" (Table 1, p. 6).
- MAD: Noisy
  - Task type: Other (recall).
  - Dataset(s): MAD benchmark suite.
  - Domain: synthetic sequence tasks.
  - Evidence: "MAD and Copying. We also evaluate our method on the MAD benchmark suite (Poli et al., 2024)" (Section 4.2, p. 7); "Model                   Compress Fuzzy In-Context Memorize Noisy Selective Average" (Table 1, p. 6).
- MAD: Selective Copy
  - Task type: Other (copying/recall).
  - Dataset(s): MAD benchmark suite.
  - Domain: synthetic sequence tasks.
  - Evidence: "This task differs from Selective Copy in MAD in that the entire input sequence has to be copied token-by-token" (Section 4.2, p. 7); "Model                   Compress Fuzzy In-Context Memorize Noisy Selective Average" (Table 1, p. 6).
- String copying
  - Task type: Other (copying).
  - Dataset(s): Copying (train task: copy; eval task: copy).
  - Domain: synthetic sequence tasks.
  - Evidence: "We also evaluate string copying following Jelassi et al. (2024)." (Section 4.2, p. 7); "Train task                        copy" and "Eval task                         copy" (Table 6, p. 24).
- State tracking (permutation composition, S2)
  - Task type: Tracking.
  - Dataset(s): S2 permutation composition.
  - Domain: synthetic sequence tasks.
  - Evidence: "State Tracking. A common way to evaluate the expressivity of a model is state tracking on permutation composition (Liu et al., 2023)." (Section 4.2, p. 7); "Figure 8: State tracking peformance of GLA, Transformer, and DeltaNet with different positional embeddings on S2 and A3 ." (Figure 8 caption, p. 7).
- State tracking (permutation composition, A3)
  - Task type: Tracking.
  - Dataset(s): A3 permutation composition.
  - Domain: synthetic sequence tasks.
  - Evidence: "State Tracking. A common way to evaluate the expressivity of a model is state tracking on permutation composition (Liu et al., 2023)." (Section 4.2, p. 7); "Figure 8: State tracking peformance of GLA, Transformer, and DeltaNet with different positional embeddings on S2 and A3 ." (Figure 8 caption, p. 7).
- LMB. (as listed in Table 2)
  - Task type: Generation (language modeling).
  - Dataset(s): LMB.
  - Domain: language modeling.
  - Evidence: "Model                 LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c Avg." (Table 2, p. 8); "For our language modeling experiments we train 370M parameter versions of GLA" (Section 4.3, p. 8).
- PIQA
  - Task type: Other (multiple-choice task).
  - Dataset(s): PIQA.
  - Domain: language modeling.
  - Evidence: "Model                 LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c Avg." (Table 2, p. 8); "Table 2: Evaluation results on tasks from lm-eval-harness (Gao et al., 2024)" (Table 2 caption, p. 8).
- Hella. (as listed in Table 2)
  - Task type: Other (multiple-choice task).
  - Dataset(s): Hella.
  - Domain: language modeling.
  - Evidence: "Model                 LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c Avg." (Table 2, p. 8); "Table 2: Evaluation results on tasks from lm-eval-harness (Gao et al., 2024)" (Table 2 caption, p. 8).
- Wino. (as listed in Table 2)
  - Task type: Other (multiple-choice task).
  - Dataset(s): Wino.
  - Domain: language modeling.
  - Evidence: "Model                 LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c Avg." (Table 2, p. 8); "Table 2: Evaluation results on tasks from lm-eval-harness (Gao et al., 2024)" (Table 2 caption, p. 8).
- ARC-e
  - Task type: Other (multiple-choice task).
  - Dataset(s): ARC-e.
  - Domain: language modeling.
  - Evidence: "Model                 LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c Avg." (Table 2, p. 8); "Table 2: Evaluation results on tasks from lm-eval-harness (Gao et al., 2024)" (Table 2 caption, p. 8).
- ARC-c
  - Task type: Other (multiple-choice task).
  - Dataset(s): ARC-c.
  - Domain: language modeling.
  - Evidence: "Model                 LMB. LMB. PIQA Hella. Wino. ARC-e ARC-c Avg." (Table 2, p. 8); "Table 2: Evaluation results on tasks from lm-eval-harness (Gao et al., 2024)" (Table 2 caption, p. 8).

4. Domain and Modality Scope
- Single domain or multiple domains? Multiple task domains within the same modality (synthetic tasks and language modeling). Evidence: "In the following section we test our proposed model on synthetic and real-world language modeling tasks." (Section 4.2, p. 7).
- Multiple modalities? Not specified in the paper.
- Domain generalization or cross-domain transfer? Not claimed.

5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| MQAR | No (trained per task) | Not specified | Not specified | "We have carefully followed the training recipe of Arora et al. (2024a) for all models" (Appendix B.2.3, p. 22). |
| MAD: Compress Recall | No (trained per task) | Not specified | Not specified | "= 396 trained models per considered setting (i.e., GLA with Selective RoPE, RoPE or NoPE)." (Appendix B.2.1, p. 22). |
| MAD: Fuzzy Recall | No (trained per task) | Not specified | Not specified | "= 396 trained models per considered setting (i.e., GLA with Selective RoPE, RoPE or NoPE)." (Appendix B.2.1, p. 22). |
| MAD: In-Context Recall | No (trained per task) | Not specified | Not specified | "= 396 trained models per considered setting (i.e., GLA with Selective RoPE, RoPE or NoPE)." (Appendix B.2.1, p. 22). |
| MAD: Memorize | No (trained per task) | Not specified | Not specified | "= 396 trained models per considered setting (i.e., GLA with Selective RoPE, RoPE or NoPE)." (Appendix B.2.1, p. 22). |
| MAD: Noisy | No (trained per task) | Not specified | Not specified | "= 396 trained models per considered setting (i.e., GLA with Selective RoPE, RoPE or NoPE)." (Appendix B.2.1, p. 22). |
| MAD: Selective Copy | No (trained per task) | Not specified | Not specified | "= 396 trained models per considered setting (i.e., GLA with Selective RoPE, RoPE or NoPE)." (Appendix B.2.1, p. 22). |
| String copying | No (trained per task) | Not specified | Not specified | "Train task                        copy" and "Eval task                         copy" (Table 6, p. 24). |
| State tracking (S2) | No (trained per task) | Not specified | Not specified | "For state tracking we adopt the exact experimental setup as described in DeltaProduct" (Appendix B.2.2, p. 22). |
| State tracking (A3) | No (trained per task) | Not specified | Not specified | "For state tracking we adopt the exact experimental setup as described in DeltaProduct" (Appendix B.2.2, p. 22). |
| LMB. | Yes (same model evaluated across tasks) | No (zero-shot) | Not specified | "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3, p. 8). |
| PIQA | Yes (same model evaluated across tasks) | No (zero-shot) | Not specified | "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3, p. 8). |
| Hella. | Yes (same model evaluated across tasks) | No (zero-shot) | Not specified | "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3, p. 8). |
| Wino. | Yes (same model evaluated across tasks) | No (zero-shot) | Not specified | "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3, p. 8). |
| ARC-e | Yes (same model evaluated across tasks) | No (zero-shot) | Not specified | "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3, p. 8). |
| ARC-c | Yes (same model evaluated across tasks) | No (zero-shot) | Not specified | "We follow the default zero-shot evaluation setup in lm-eval-harness, using its standard prompting" (Section 4.3, p. 8). |

6. Input and Representation Constraints
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens / sequence length? Explicit sequence lengths are given in several tasks:
  - "Seq. 512, KV pairs 64" (Figure 7, p. 7).
  - "Train sequence length    128 tokens" and "Eval sequence length     512 tokens" (Table 5, p. 22).
  - "Sequence length                   420" (Table 6, p. 24).
- Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper.
- Padding or resizing requirements? Not specified in the paper.
- Vocabulary constraints (language modeling): "vocabulary size of 32 000" (Section 4.3, p. 8); Copying task: "Vocab size                        26" (Table 6, p. 24).

7. Context Window and Attention Structure
- Maximum sequence length: "context length of 4096" (Section 4.3, p. 8). Additional explicit lengths include "Seq. 512" (Figure 7, p. 7), "Eval sequence length     512 tokens" (Table 5, p. 22), and "Max length (eval)                 512" (Table 6, p. 24).
- Fixed or variable sequence length: Not specified globally. Task-specific bounds are given for copying: "Min length (train)                2" and "Max length (train)                64"; "Min length (eval)                 2" and "Max length (eval)                 512" (Table 6, p. 24).
- Attention type: softmax attention and linear attention are both used. Evidence: "Transformers with softmax attention" (Introduction, p. 1); "Linear attention (Katharopoulos et al., 2020) replaces the exponential kernel in softmax attention" (Section 2, p. 2).
- Global/windowed/hierarchical/sparse? Not specified in the paper.
- Mechanisms to manage computational cost: "sub-quadratic sequence models (modern recurrent architectures) that run in linear time and require only constant memory per step at inference" (Introduction, p. 1); "Linear attention (Katharopoulos et al., 2020) replaces the exponential kernel in softmax attention" (Section 2, p. 2); "enhanced with a forget gate, At" (Section 2, p. 2).

8. Positional Encoding (Critical Section)
- Mechanism: Selective RoPE (input-dependent rotary) and RoPE (fixed rotations), with NoPE as a baseline. Evidence: "we introduce Selective RoPE, an input-dependent rotary embedding mechanism" (Abstract, p. 1); "Rotary Position Embeddings (RoPE) are used to add relative positional information through rotations of the query-key pairs" (Section 2, p. 2); "NoPE" and "RoPE" and "Selective RoPE" appear in Table 1 (p. 6).
- Where applied: to queries and keys. Evidence: "by applying a learned, input-dependent rotary position embedding to the queries and keys" (Introduction, p. 1).
- Fixed vs modified/ablated: Positional encodings are compared and ablated. Evidence: "We ablate adding a rotation (i.e., phase) gate and a learnable bias term" (Section 4.3, p. 8); Table 1 shows NoPE, RoPE, and Selective RoPE variants (p. 6).

9. Positional Encoding as a Variable
- Core research variable or fixed assumption? Core research variable. Evidence: "we introduce Selective RoPE, an input-dependent rotary embedding mechanism" (Abstract, p. 1).
- Multiple positional encodings compared? Yes. Evidence: Table 1 includes "NoPE", "RoPE", and "Selective RoPE" (p. 6).
- Does the paper claim PE choice is not critical? Not claimed.

10. Evidence of Constraint Masking (Scale vs Structure)
- Model sizes: "370M parameter versions" (Section 4.3, p. 8).
- Dataset sizes: "All models are trained on 35B tokens" (Section 4.3, p. 8).
- Performance gains attribution: "input-dependent rotations improve performance in language modeling and on difficult sequence tasks like copying, state tracking, and retrieval" (Abstract, p. 1). The paper does not explicitly attribute gains to scaling model size or data size beyond reporting these sizes.

11. Architectural Workarounds
- Linear attention as a sub-quadratic alternative: "Linear attention (Katharopoulos et al., 2020) replaces the exponential kernel in softmax attention with a kernel with a positive feature map" (Section 2, p. 2).
- Gating/forget mechanisms: "enhanced with a forget gate, At" (Section 2, p. 2); "Selective gating ... adaptively decays history" (Introduction, p. 1).
- Linear-time sequence modeling: "sub-quadratic sequence models (modern recurrent architectures) that run in linear time and require only constant memory per step at inference" (Introduction, p. 1).

12. Explicit Limitations and Non-Claims
- Length extrapolation not studied: "incorporating RoPE is notoriously detrimental to the length-extrapolation capabilities of sequence models (Li et al., 2024). In this paper, we do not investigate this aspect since we consider it to be out of the scope of our research." (Conclusion, p. 10).
- Future work on bias/phase gate: "further investigation of the effect of the extra components used in Selective RoPE, namely the bias term and the phase gate, can be a fruitful direction for future research." (Conclusion, p. 10).
- Future work on gate dimensionality: "we consider the impact of choosing a diagonal as opposed to a scalar forget gate to be an interesting question" (Conclusion, p. 10).

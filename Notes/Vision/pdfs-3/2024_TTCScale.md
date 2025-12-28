## 1. Basic Metadata
- Title: "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" (p. 1)
- Authors: "Charlie Snell          , Jaehoon Lee , Kelvin Xu          and Aviral Kumar" (p. 1)
- Year: 2024. Evidence: "arXiv:2408.03314v1 [cs.LG] 6 Aug 2024" (p. 1)
- Venue: arXiv. Evidence: "arXiv:2408.03314v1 [cs.LG] 6 Aug 2024" (p. 1)

## 2. One-Sentence Contribution Summary
The paper proposes a prompt-difficulty-conditioned allocation of test-time compute to improve LLM performance and compares those gains against scaling model parameters in FLOPs-matched settings.

## 3. Tasks Evaluated
| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| MATH benchmark (math problem solving) | Reasoning / relational | MATH [13]; 12k train / 500 test | High-school competition level math problems | "To this end, we focus on the MATH [13] benchmark, which consists of high-school competition level math problems with a range of difficulty levels." (p. 6) "For all experiments, we use the dataset split consisting of 12k train and 500 test questions, used in Lightman et al. [22]." (p. 6) |

## 4. Domain and Modality Scope
- Evaluation domain scope: Single domain (MATH math problems). Evidence: "To this end, we focus on the MATH [13] benchmark, which consists of high-school competition level math problems with a range of difficulty levels." (p. 6)
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Not specified in the paper.
- Domain generalization / cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| MATH benchmark | Not applicable (single task); base model family used with separate fine-tuned models for revision and PRM/ORM | Yes (revision and PRM finetuning) | Not specified in the paper | "We conduct our analysis using the PaLM 2-S* [3] (Codey) base model." (p. 6) "we specifically finetune models to iteratively revise their answers in complex reasoning-based settings." (p. 4) "We finetune our PRM as a binary classifier, where the model predicts a value between 0 and 1 at each step in the solution." (p. 22) |

## 6. Input and Representation Constraints
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper.
- Padding or resizing requirements? Not specified in the paper.
- Explicit input formatting constraints: "In order to enable the base model to output answers in a step-by-step format to which a PRM can be applied, we use a 4-shot prompt consisting of randomly selected correct answer examples from the PRM800k data released by Lightman et al. [22]." (p. 25)
- Context length constraints during revision training: "We include up to four incorrect answers in context, where the specific number of solutions in context is sampled randomly from a uniform distribution over categories 0 to 4." (p. 11)

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage compute/context: "While our revision model is only trained with up to four previous answers in-context, we can sample longer chains by truncating the context to the most recent four revised responses." (p. 11)

## 8. Positional Encoding (Critical Section)
- Mechanism: Not specified in the paper.
- Where applied: Not specified in the paper.
- Fixed/modified/ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Core research variable? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- PE choice "not critical" or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking
- Model sizes: "We conduct our analysis using the PaLM 2-S* [3] (Codey) base model." (p. 6) "We conduct a FLOPs-matched comparison between a smaller model with additional test-time compute and pretraining a 14x larger model." (p. 3)
- Dataset size: "For all experiments, we use the dataset split consisting of 12k train and 500 test questions, used in Lightman et al. [22]." (p. 6)
- Gains attributed to scaling test-time compute: "By appropriately allocating test-time compute in this way, we are able to greatly improve test-time compute scaling, surpassing the performance of a best-of-N baseline while only using about 4x less computation with both revisions and search (Sections 5 and 6)." (p. 3)
- Scaling data vs parameters: "We focus on the setting in which model parameters are scaled up and training data amount is fixed, matching the approach taken with the open-source LLaMA series of models [41]." (p. 14)

## 11. Architectural Workarounds
- Search-based test-time compute: "We study three search approaches that sample outputs from a few-shot prompted base LLM (see Appendix G)." (p. 7)
- Revision-based test-time compute: "Given a finetuned revision model, we can then sample a sequence of revisions from the model at test-time." (p. 11)
- Prompt formatting for verifiers: "In order to enable the base model to output answers in a step-by-step format to which a PRM can be applied, we use a 4-shot prompt consisting of randomly selected correct answer examples from the PRM800k data released by Lightman et al. [22]." (p. 25)
- Context truncation for revisions: "While our revision model is only trained with up to four previous answers in-context, we can sample longer chains by truncating the context to the most recent four revised responses." (p. 11)

## 12. Explicit Limitations and Non-Claims
- "While we combined verifiers with revisions in Section 6, we did not experiment with PRM tree-search techniques in combination with revisions. Neither did we study other techniques such as critique and revise [23]." (p. 16)
- "Additionally, we found that across the board these schemes provided small gains on hard problems; future work should work to develop new ways of using test-time compute which can circumvent this limitation." (p. 16)
- "estimating our notion of difficulty requires applying a non-trivial amount of test-time compute itself." (p. 16)

## 13. Constraint Profile (Synthesis)
- Domain scope: Single-domain evaluation on MATH math problems only.
- Task structure: Single math-problem reasoning task with step-by-step answer format.
- Representation rigidity: Explicit prompt formatting (4-shot, step-by-step) and limited revision context (up to four prior answers).
- Model sharing vs specialization: Same base model family but separate fine-tuned models for revision and PRM/ORM; no joint multi-task training described.
- Role of positional encoding: Not discussed/unspecified.

## 14. Final Classification
- Classification: Single-task, single-domain.
- Justification: The evaluation "focus[es] on the MATH [13] benchmark, which consists of high-school competition level math problems with a range of difficulty levels" and uses a single dataset split (p. 6). No multi-domain or multi-task evaluation is described beyond this benchmark (p. 6).

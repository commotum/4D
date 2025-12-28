## 1. Basic Metadata

- Title: Combining Induction and Transduction for Abstract Reasoning.
- Authors: Wen-Ding Li; Keya Hu; Carter Larsen; Yuqing Wu; Simon Alford; Caleb Woo; Spencer M. Dunn; Hao Tang; Michelangelo Naim; Dat Nguyen; Wei-Long Zheng; Zenna Tavares; Yewen Pu; Kevin Ellis.
- Year: 2024.
  - Evidence: "arXiv:2411.02272v4 [cs.LG] 2 Dec 2024" (Title page)
- Venue: arXiv preprint.
  - Evidence: "arXiv:2411.02272v4 [cs.LG] 2 Dec 2024" (Title page)

## 2. One-Sentence Contribution Summary

The paper studies whether induction (program synthesis) or transduction (direct prediction) is better for few-shot ARC grid problems by training both on synthetic ARC-style programs and showing they are complementary, with an ensemble improving ARC performance.

- Evidence: "We study this question
on ARC by training neural models for induction (inferring latent functions) and
transduction (directly predicting the test output for a given test input)." (Abstract)
- Evidence: "Ensembling them approaches human-level performance on ARC." (Abstract)

## 3. Tasks Evaluated

### Task 1: ARC public validation split (few-shot grid-to-grid reasoning)
- Task type: Other (specify): few-shot input-output grid transformation / reasoning.
- Dataset(s) used: ARC public validation split.
- Domain: 2D colored grids (synthetic grid puzzles).
- Evidence:
  - "We consider few-shot supervised learning problems where the learner is trained to map members
of an input space X to output space Y." (Section 2, Neural Models for Induction and Transduction)
  - "For K-shot learning, we receive K training input-outputs
(xtrain , ytrain ) ∈ X K × Y K , together with a single test input xtest ∈ X , and predict ytest ∈ Y." (Section 2)
  - "We report performance on the 400-problem public validation split of ARC, which is
harder than the training split." (Section 4, Empirical Study of Induction and Transduction)
  - "Every input from X and output from Y is a 2D grid
ranging from 1–30 pixels per side, with each pixel containing one of ten colors." (Section 2, Instantiating the framework for ARC)

### Task 2: ARC private test set (Kaggle)
- Task type: Other (specify): few-shot input-output grid transformation / reasoning.
- Dataset(s) used: ARC private test set hosted by Kaggle.
- Domain: 2D colored grids (same ARC grid format).
- Evidence:
  - "Our flagship model is too expensive to run on the private test set hosted
by Kaggle." (Section 5, Scaling Our Method)
  - "Every input from X and output from Y is a 2D grid
ranging from 1–30 pixels per side, with each pixel containing one of ten colors." (Section 2)

### Task 3: ConceptARC (ARC-style concept groups)
- Task type: Reasoning / relational; Other (specify): few-shot grid-to-grid transformation.
- Dataset(s) used: ConceptARC.
- Domain: ARC-style 2D colored grids.
- Evidence:
  - "We test on ConceptARC (Moskvichev et al., 2023),
an alternative ARC-style test-set which classifies its tasks into “concept groups” each exemplifying
a single isolated high-level concept such as “sameness” or “above vs below.”" (Section 6)
  - "Every input from X and output from Y is a 2D grid
ranging from 1–30 pixels per side, with each pixel containing one of ten colors." (Section 2)

## 4. Domain and Modality Scope

- Single domain / single modality: ARC-style 2D colored grids.
  - Evidence: "Every input from X and output from Y is a 2D grid
ranging from 1–30 pixels per side, with each pixel containing one of ten colors." (Section 2)
  - Evidence: "We test on ConceptARC (Moskvichev et al., 2023),
an alternative ARC-style test-set..." (Section 6)
- Multiple domains within the same modality: Not specified in the paper.
- Multiple modalities: Not specified in the paper.
- Domain generalization / cross-domain transfer: Not claimed.
  - Evidence: "Our work is only evaluated on ARC." (Section 8, Limitations)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ARC public validation split | Not specified in the paper. | Yes. | Not specified in the paper. | "We then meta-learn by further fine-tuning Llama3.1-8B-instruct for induction or
transduction using a synthetically-generated corpus of problems" (Section 2); "We report performance on the 400-problem public validation split of ARC" (Section 4) |
| ARC private test set (Kaggle) | Not specified in the paper. | Yes (scaled-down evaluation settings). | Not specified in the paper. | "Our flagship model is too expensive to run on the private test set hosted
by Kaggle. We scale down by omitting test-time training, only sampling 336 programs, and reducing
the transduction beam size to 3." (Section 5); "We then meta-learn by further fine-tuning Llama3.1-8B-instruct..." (Section 2) |
| ConceptARC | Not specified in the paper. | Yes (models trained on ARC-Potpourri). | Not specified in the paper. | "We use models trained
on ARC-Potpourri" (Section 6); "We then meta-learn by further fine-tuning Llama3.1-8B-instruct..." (Section 2) |

## 6. Input and Representation Constraints

- Fixed or variable input resolution: Variable within 1–30 pixels per side.
  - Evidence: "Every input from X and output from Y is a 2D grid
ranging from 1–30 pixels per side, with each pixel containing one of ten colors." (Section 2)
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
  - Evidence for tokenization scheme: "We encode 2D colored grids as strings using 1 token per pixel, and use newlines to delimit rows
(Appendix B.1)." (Section 2)
- Fixed dimensionality (e.g., strictly 2D): Yes, 2D grids.
  - Evidence: "Every input from X and output from Y is a 2D grid
ranging from 1–30 pixels per side, with each pixel containing one of ten colors." (Section 2)
- Any padding or resizing requirements: Not specified in the paper.
- Additional explicit representation constraints:
  - "We must include in our prompts for our fine-tuned models the input/output 2D colored grids of
each problem. To do this we represent the problem textually by naming the colors one-by-one. We
renamed certain colors which were more than one token (e.g., maroon→brown saves 1 token/pixel),
and presented the grid as a whitespace-delimited 2D array with newlines delimiting rows." (Appendix B.1)
  - "Appropriate grid sizes: We remove input-output grids with height or width larger than 30,
         aligning with grid sizes in ARC" (Appendix A, Execution and Filtering of Generated Problems)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost (windowing, pooling, pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Not specified in the paper.
- Where it is applied: Not specified in the paper.
- Fixed vs. modified/ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable

- Treated as core research variable or fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- PE choice claimed “not critical” or secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs. Structure)

- Model size(s):
  - "we initialize our
models with Llama3.1-8B-instruct (Dubey et al., 2024)" (Section 2, Instantiating the framework for ARC)
- Dataset size(s):
  - "expands them to make 400k new problems paired with Python solutions." (Section 1, Contributions)
  - "ARC-Heavy: 200k problems from 160 seeds." (Section 5)
  - "ARC-Potpourri: 400k problems from heterogeneous sources." (Section 5)
  - "We report performance on the 400-problem public validation split of ARC" (Section 4)
- Scaling claims:
  - "We find performance saturates quickly when increasing
   manually-labelled data, but scales with compute, both at training and testing time." (Section 1)
  - "Performance scales with dataset size, but quickly saturates with increasing number of seeds." (Section 4)
- Training tricks / test-time procedures:
  - "We improve transduction with test-time training (abbreviated TTT; Sun
et al. (2020)) and a reranking scheme that augments each problem and predicts the most likely output
under multiple augmentations (Appendix E- F). We expand our sampling budget to 20k programs." (Section 5)
- Attribution summary:
  - Scaling model size: Not specified in the paper.
  - Scaling data / compute: Explicitly claimed (quotes above).
  - Architectural hierarchy: Not specified in the paper.
  - Training tricks: Explicitly used (TTT + reranking; quote above).

## 11. Architectural Workarounds

- Induction+transduction ensembling:
  - "Therefore
we ensemble by attempting induction first, then transduction if none of the candidate hypotheses
explained the examples:" (Section 2, Combining induction and transduction)
- Data augmentation and reranking for transduction:
  - "We improve the performance of the transductive model through data augmentation and output
reranking." (Appendix E)
- Test-time training for transduction:
  - "Test time training is an approach for updating model parameters at test time, which we apply
to our transduction model." (Appendix F)
- Increased test-time sampling budget:
  - "We expand our sampling budget to 20k programs." (Section 5)

## 12. Explicit Limitations and Non-Claims

- Limitations on learning from new problems:
  - "Limitations. Our system does not grow more competent at few-shot learning by solving new prob-
blems: Instead, it bootstraps from manually encoded knowledge in the seeds, which is transformed
into a few-shot learner via an LLM training/inference pipeline." (Section 8, Limitations)
  - "A more compelling approach would
be to have the system discover for itself the knowledge that we compiled for it within the seeds, for
instance by practicing on training tasks, without supervising on ground truth solutions." (Section 8)
- Non-claims about broader evaluation:
  - "Our work is only evaluated on ARC." (Section 8)

## 13. Constraint Profile (Synthesis)

- Domain scope: Single modality of ARC-style 2D colored grids; evaluation limited to ARC/ConceptARC.
- Task structure: Few-shot input-output grid mapping with diverse ARC concepts; ConceptARC isolates single concepts.
- Representation rigidity: 2D grids with 1–30 pixels per side, 10 colors, encoded as 1 token per pixel with newline-delimited rows.
- Model sharing vs specialization: Separate induction and transduction models (same LLM backbone, fine-tuned); ensemble combines them; weight-sharing across tasks not specified.
- Role of positional encoding: Not discussed in the paper (treated as unspecified).

## 14. Final Classification

Multi-task, single-domain.

Justification: ARC is described as "a composite of many reasoning datasets" with many tasks, but all are in a single domain of ARC-style 2D colored grids (Sections 1 and 2). The paper evaluates ARC splits and ConceptARC, which is explicitly "ARC-style" (Section 6), so the task set is multi-task within the same modality and domain rather than multi-domain.

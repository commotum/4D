## 1. Basic Metadata
- Title: "The ConceptARC Benchmark: Evaluating Understanding and Generalization in the ARC Domain" (Title page)
- Authors: "Arseny Moskvichev, Victor Vikram Odouard, and Melanie Mitchell" (Title page)
- Year: 2023 ("arXiv:2305.07141v1 [cs.LG] 11 May 2023"; Title page)
- Venue: arXiv ("arXiv:2305.07141v1 [cs.LG] 11 May 2023"; Title page)

## 2. One-Sentence Contribution Summary
The paper introduces ConceptARC, a concept-based evaluation benchmark in the ARC domain to systematically assess abstraction and generalization abilities, and reports human and machine solver performance on it. (Abstract)

## 3. Tasks Evaluated
Task: ConceptARC (ARC-domain grid transformation/analogy tasks)
- Task type: Reasoning / relational; Generation
- Dataset(s): ConceptARC benchmark (16 concept groups, 10 tasks per concept, 3 test inputs per task)
- Domain: Synthetic colored grids (ARC domain)
- Evidence:
  - "In this paper we describe an in-depth evaluation benchmark for the Abstraction and Reasoning Corpus (ARC), a collection of few-shot abstraction and analogy problems developed by Chollet [2019]." (Abstract)
  - "In particular, we describe ConceptARC, a new, publicly available benchmark in the ARC domain that systematically assesses abstraction and generalization abilities on a number of basic spatial and semantic concepts." (Abstract)
  - "ARC consists of a set of analogy problems, exemplified by those given in Figure 1. In particular, each problem consists of a set of demonstrations—initial and transformed grids—and one or more test input grids." (Section 2: The Abstraction and Reasoning Corpus)
  - "The job of the solver is to generate a new grid that results from applying the abstract rule to the test input." (Figure 1 caption, Section 2)
  - "For each concept, we created 10 new ARC tasks that are different instantiations of the concept." (Section 3: The ConceptARC Benchmark)
  - "Each of our tasks has three different test inputs." (Section 3: The ConceptARC Benchmark)

## 4. Domain and Modality Scope
- Single domain? Yes. Evidence: "ConceptARC, a new, publicly available benchmark in the ARC domain" (Abstract).
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Not specified in the paper. (The GPT-4 evaluation uses the "language-only version of GPT-4" but does not claim multimodal evaluation.) Evidence: "the publicly available language-only version of GPT-4" (Section 7: Details of Testing GPT-4).
- Domain generalization or cross-domain transfer claimed? Not claimed. (The paper discusses generalization within the ARC domain.)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ConceptARC tasks (evaluated with ARC-Kaggle programs and GPT-4) | Not specified in the paper. | No for GPT-4 (evaluated zero-shot without fine-tuning); otherwise not specified. | Not specified in the paper. | "the publicly available language-only version of GPT-4 ... in a zero-shot manner (i.e., without any fine-tuning on these tasks)." (Section 7) "The first and second place programs in the ARC-Kaggle challenge both work by performing a heuristic search over a fixed, manually defined set of grid operations" (Section 6: Details of Testing Winning Programs From the ARC-Kaggle Challenge) |

## 6. Input and Representation Constraints
- Grid-based 2D input/output: "each problem consists of a set of demonstrations—initial and transformed grids—and one or more test input grids." (Section 2)
- Output is a grid transformation: "The job of the solver is to generate a new grid that results from applying the abstract rule to the test input." (Figure 1 caption, Section 2)
- Variable grid dimensions (ConceptARC): "ConceptARC benchmark, which allows any grid dimensions" (Section 9: Related Work)
- Explicit encoding used for GPT-4 evaluation: "Within each row of a grid, the colors of each pixel were numerically coded as in the original ARC data files at [10] (these were the inputs to the ARC-Kaggle competitors) and space-separated." (Section 7)
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality beyond 2D grids? Not specified in the paper.
- Padding/resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms for computational cost (windowing/pooling/token pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Not specified in the paper.
- Where applied: Not specified in the paper.
- Fixed/modified/ablated across experiments: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Treated as core research variable or fixed assumption? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- Any claim that PE choice is “not critical” or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking
- Model size(s): Not specified in the paper.
- Dataset size(s): "Because the tasks in ConceptARC were created manually, the corpus is relatively small: 16 concept groups, with 10 tasks per concept-group and three test inputs per task, for a total of 480 test inputs." (Section 8.3: Limitations of Our Studies)
- Performance gains attributed to scaling model size/data/architecture/training tricks? Not specified in the paper.

## 11. Architectural Workarounds
- Heuristic search over fixed grid operations (ARC-Kaggle programs): "The first and second place programs in the ARC-Kaggle challenge both work by performing a heuristic search over a fixed, manually defined set of grid operations to generate a pipeline of these operations" (Section 6).
- Data augmentation via grid transformations: "Both programs were able to increase their success by augmenting the given task demonstrations, for example, by flipping the demonstration input and output grids along the diagonal, by remapping colors, and other heuristic transformations." (Section 6)

## 12. Explicit Limitations and Non-Claims
- Small, manually constructed corpus: "Because the tasks in ConceptARC were created manually, the corpus is relatively small: 16 concept groups, with 10 tasks per concept-group and three test inputs per task, for a total of 480 test inputs." (Section 8.3)
- Limited human sample size: "Our results, showing high human accuracy on these tasks, are based on these relatively small sets of people, whose numbers were limited by the funds we had available for these studies." (Section 8.3)
- Task ambiguity: "there are a small number of tasks in the ConceptARC corpus that are ambiguous—that is, for which test inputs have more than one reasonable solution." (Section 8.3)
- Shortcut solutions: "There are also a small number of tasks that allow for “shortcut solutions”" (Section 8.3)
- Explicit statements about what the model does not attempt to do (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified in the paper.

## 1. Basic Metadata
- Title: How I came in first on ARC-AGI-Pub using Sonnet 3.5 with Evolutionary Test-time Compute. (Post header)
- Authors: Jeremy Berman. (Post header)
- Year: 2024. (Post header)
- Venue: Jeremy’s Substack (Substack blog post). (Post header)

## 2. One-Sentence Contribution Summary
The post introduces “Evolutionary Test-time Compute,” an evolutionary test-time compute method that has Sonnet 3.5 generate and iteratively refine Python transform functions for ARC-AGI challenges and reports a record 53.6% on ARC-AGI-Pub. (Sections: Intro, What is ARC)

## 3. Tasks Evaluated
Task: ARC-AGI grid transformation (abstract pattern recognition puzzles)
- Task type: Other (abstract pattern recognition / grid transformation).
- Dataset(s) used: ARC-AGI / ARC-AGI-Pub challenges; ARC training challenges (60 training challenges).
- Domain: synthetic colored grid puzzles (input/output grids).
- Evidence:
  - “ARC-AGI is an intelligence test designed to measure abstract pattern recognition, similar to an IQ test.” (Section: What is ARC)
  - “The test presents novel patterns through a few examples and then challenges the test-taker to continue the sequence, measuring their ability to identify and generalize underlying rules they’ve never encountered before.” (Section: What is ARC)
  - “Here, you are given two examples of input/output grids and you must fill in the test output grid with the correct colors.” (Section: What is ARC)
  - “The goal is to evolve a Python function that can correctly transform input grids into output grids.” (Section: Architecture)
  - “ARC-AGI-Pub is a leaderboard for solving ARC challenges using the internet.” (Section: What is ARC)
  - “You are given 12 hours of compute on a Kaggle notebook and $10,000 for API costs, to complete 500 challenges (400 of which do not exist on the internet).” (Section: What is ARC)
  - “I compared two architectures across 60 training challenges, each using 200 LLM calls with Sonnet 3.5 and identical system prompts and examples:” (Section: Architecture)

## 4. Domain and Modality Scope
- Single domain? Yes — ARC grid puzzles. Evidence: “Here, you are given two examples of input/output grids and you must fill in the test output grid with the correct colors.” (Section: What is ARC)
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Not specified in the paper.
- Domain generalization or cross-domain transfer claimed? Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ARC-AGI grid transformation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | “After lots of experimenting, I got a record of 53.6% on the public leaderboard using Sonnet 3.5.” (Section: Intro) “I think there’s a reasonable chance ARC could be solved if you fine-tuned Sonnet 3.5 on a 10,000 diverse, correct CoT solutions for the train and eval sets — assuming all of the core knowledge necessary to solve ARC is in those sets.” (Section: Finetune LLMs, Appendix) |
- Training regimen (separate per task vs. joint) is not specified in the paper.

## 6. Input and Representation Constraints
- Grid-based inputs/outputs and color filling: “Here, you are given two examples of input/output grids and you must fill in the test output grid with the correct colors.” (Section: What is ARC)
- Multiple explicit grid encodings: “For grid representation, I provide multiple formats similar to Ryan: Grid dimensions; Base64 image encoding; ASCII representation; Python nested list format (list[list[int]]).” (Section: Prompting)
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper.
- Padding/resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed vs. variable length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost: “LLMs tend to pay less attention to longer contexts than shorter ones, and more context means higher computational costs.” (Section: Architecture)

## 8. Positional Encoding (Critical Section)
Not specified in the paper.

## 9. Positional Encoding as a Variable
Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs. Structure)
- Model size(s): Not specified in the paper.
- Dataset size(s): “You are given 12 hours of compute on a Kaggle notebook and $10,000 for API costs, to complete 500 challenges (400 of which do not exist on the internet).” (Section: What is ARC) “I compared two architectures across 60 training challenges...” (Section: Architecture)
- Test-time compute scaling: “This process repeats multiple times, ultimately generating up to 500 functions using 31 dynamic prompts per challenge.” (Section: Intro)
- Compute budget constraints: “You are given 12 hours of compute on a Kaggle notebook and $10,000 for API costs...” (Section: What is ARC)
- Performance/architecture evidence: “Shallow achieved 70% accuracy (42/60 challenges solved)” and “Deep achieved 75% accuracy (45/60 challenges solved).” (Section: Architecture)
- Attribution to scaling test-time compute: “LLMs can compensate for their generalization limitations through scaled test-time compute guided by evolutionary principles.” (Section: Intro)

## 11. Architectural Workarounds
- Evolutionary search over functions: “The goal is to evolve a Python function that can correctly transform input grids into output grids.” (Section: Architecture)
- Iterative selection/revision: “The evolutionary process follows these steps:” (Section: Architecture) and “Selection & Reproduction” / “Iteration” in the same list. (Section: Architecture)
- Function execution for verification: “I have the LLM generate Python functions, instead of just outputting solution grids, because functions can be executed and verified for correctness... but grids cannot.” (Section: Intro)
- Two-track design (pooling + single-parent): “my submitted algorithm runs two parallel tracks: Traditional single-parent evolution” and “Pooled multi-parent evolution.” (Section: Architecture)

## 12. Explicit Limitations and Non-Claims
- Context-length limitation/cost: “LLMs tend to pay less attention to longer contexts than shorter ones, and more context means higher computational costs.” (Section: Architecture)
- Experimentation budget limitation: “my ability to run large-scale experiments was limited by having only a few thousand dollars in Anthropic credits...” (Section: Architecture)
- Prompt optimality limitation: “I’m sure my prompt is not optimal but I’ve found it to be good enough.” (Section: Prompt Diversity, Appendix)
- Non-claim about AGI: “Solving ARC will be a great step towards AGI — but its solution will not necessarily be AGI.” (Section: ARC and the Path to AGI)
- Limitation of test-time compute for AGI: “I don’t think test-time compute alone can get us to AGI. I think a new architecture may be needed.” (Section: ARC and the Path to AGI)

## 13. Constraint Profile (Synthesis)
- Domain scope: Single benchmark of abstract grid puzzles (ARC-AGI) with input/output grids and color-filling tasks.
- Task structure: Grid transformation from examples (“transform input grids into output grids”) within a single benchmark.
- Representation rigidity: Explicit grid encodings (dimensions, Base64 image, ASCII, nested lists); fixed resolution/patch/token constraints are not specified in the paper.
- Model sharing vs specialization: Single LLM (Sonnet 3.5) is referenced for evaluations; fine-tuning is discussed as a possibility in the Appendix.
- Role of positional encoding: Not discussed.

## 14. Final Classification
Single-task, single-domain.
The evaluation centers on ARC-AGI grid transformation challenges (“input/output grids” and “transform input grids into output grids”), and datasets described are confined to ARC/ARC-AGI-Pub. No additional domains or modalities are described, and no cross-domain transfer is claimed.

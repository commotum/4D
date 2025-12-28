## 1. Basic Metadata
- Title: Olympiad-level formal mathematical reasoning with reinforcement learning
  - Evidence (Article header): "Olympiad-level formal mathematical reasoning with reinforcement learning"
- Authors: Thomas Hubert; Rishi Mehta; Laurent Sartran; Miklós Z. Horváth; Goran Žužić; Eric Wieser; Aja Huang; Julian Schrittwieser; Yannick Schroecker; Hussain Masoom; Ottavia Bertolli; Tom Zahavy; Amol Mandhane; Jessica Yung; Iuliya Beloshapka; Borja Ibarz; Vivek Veeriah; Lei Yu; Oliver Nash; Paul Lezeau; Salvatore Mercuri; Calle Sönne; Bhavik Mehta; Alex Davies; Daniel Zheng; Fabian Pedregosa; Yin Li; Ingrid von Glehn; Mark Rowland; Samuel Albanie; Ameya Velingker; Simon Schmitt; Edward Lockhart; Edward Hughes; Henryk Michalewski; Nicolas Sonnerat; Demis Hassabis; Pushmeet Kohli; David Silver
- Year: 2025
  - Evidence (Article header): "Published: 12 November 2025"
- Venue (journal): Nature
  - Evidence (Article header): "Nature"

## 2. One-Sentence Contribution Summary
The paper introduces AlphaProof, an AlphaZero-inspired reinforcement-learning agent that learns to find formal proofs and improves results on mathematics competition problems.

## 3. Tasks Evaluated
- Task: Formal proof solving on historical mathematics competition problems
  - Task type: Reasoning / relational; Other (formal theorem proving)
  - Dataset(s): Historical mathematics competition problems
  - Domain: Mathematics competition problems / formal proofs
  - Evidence (Abstract):
    - "We present AlphaProof, an AlphaZero-inspired<sup>2</sup> agent that learns to find formal proofs through RL by training on millions of auto-formalized problems."
    - "AlphaProof substantially improves state-of-the-art results on historical mathematics competition problems."
- Task: Solving 2024 IMO non-geometry problems
  - Task type: Reasoning / relational; Other (formal theorem proving)
  - Dataset(s): 2024 IMO competition non-geometry problems
  - Domain: Mathematics competition problems
  - Evidence (Abstract): "At the 2024 IMO competition, our AI system, with AlphaProof as its core reasoning engine, solved three out of the five non-geometry problems, including the competition’s most difficult problem."

## 4. Domain and Modality Scope
- Single domain? Yes — mathematics competition problems.
  - Evidence (Abstract): "A long-standing goal of artificial intelligence is to build systems capable of complex reasoning in vast domains, a task epitomized by mathematics with its boundless concepts and demand for rigorous proof."
  - Evidence (Abstract): "AlphaProof substantially improves state-of-the-art results on historical mathematics competition problems."
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Not specified in the paper.
- Domain generalization / cross-domain transfer claimed? Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Formal proof solving on historical mathematics competition problems | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Solving 2024 IMO non-geometry problems | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

## 6. Input and Representation Constraints
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper.
- Padding or resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Computational cost mechanisms (windowing/pooling/pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Not specified in the paper.
- Where applied: Not specified in the paper.
- Fixed vs modified vs ablated across experiments: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Treated as core research variable or fixed assumption? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- PE choice claimed “not critical” or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size(s): Not specified in the paper.
- Dataset size(s):
  - Evidence (Abstract): "We present AlphaProof, an AlphaZero-inspired<sup>2</sup> agent that learns to find formal proofs through RL by training on millions of auto-formalized problems."
- Test-time data scale:
  - Evidence (Abstract): "For the most difficult problems, it uses Test-Time RL, a method of generating and learning from millions of related problem variants at inference time to enable deep, problem-specific adaptation."
- Compute scale:
  - Evidence (Abstract): "Combined with AlphaGeometry 2<sup>3</sup>, this performance, achieved with multi-day computation, resulted in reaching a score equivalent to that of a silver medallist, marking the first time an AI system achieved any medal-level performance."
- Attribution of gains to scaling model size / scaling data / architecture / training tricks: Not specified in the paper.

## 11. Architectural Workarounds
- Test-Time RL for hard problems (problem-specific adaptation)
  - Evidence (Abstract): "For the most difficult problems, it uses Test-Time RL, a method of generating and learning from millions of related problem variants at inference time to enable deep, problem-specific adaptation."
- AlphaZero-inspired agent
  - Evidence (Abstract): "We present AlphaProof, an AlphaZero-inspired<sup>2</sup> agent that learns to find formal proofs through RL by training on millions of auto-formalized problems."

## 12. Explicit Limitations and Non-Claims
Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Single domain (mathematics competition problems); no evidence of cross-domain evaluation.
- Task structure: Formal proof finding / mathematical reasoning on competition problems.
- Representation rigidity: Not specified in the paper.
- Model sharing vs specialization: Not specified in the paper.
- Role of positional encoding: Not specified in the paper.

## 14. Final Classification
**Single-task, single-domain.** The paper reports evaluation on historical mathematics competition problems and the 2024 IMO non-geometry problems, both within the mathematics competition domain. There is no evidence of multi-domain or multi-modal evaluation, and no separate tasks beyond formal mathematical reasoning are described.

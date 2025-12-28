### 1. Basic Metadata
- Title: Solving olympiad geometry without human demonstrations
- Authors: Trieu H. Trinh; Yuhuai Wu; Quoc V. Le; He He; Thang Luong
- Year: 2024 (published online 17 January 2024)
- Venue (conference/journal/arXiv): Nature (Vol 625, 18 January 2024)

### 2. One-Sentence Contribution Summary
AlphaGeometry is presented as a neuro-symbolic theorem prover for Euclidean plane geometry that uses synthetic data and a language-model-guided symbolic engine to solve olympiad-level geometry problems without human demonstrations.

### 3. Tasks Evaluated
Task 1: Olympiad-level geometry theorem proving (IMO-AG-30)
- Task type: Reasoning / relational (theorem proving)
- Dataset(s): IMO-AG-30 (30 classical geometry problems adapted from IMO since 2000)
- Domain: Classical Euclidean plane geometry
- Evidence: "adapted geometry problems from the IMO competitions since 2000 to a narrower, specialized environment for classical geometry used in interactive graphical proof assistants13,17,19, as discussed in Methods." (An olympiad-level benchmark for geometry, p. 479)
- Evidence: "resulting in a test set of 30 classical geometry problems." (An olympiad-level benchmark for geometry, p. 479)
- Evidence: "The final test set is named IMO-AG-30, highlighting its source, method of translation and its current size." (An olympiad-level benchmark for geometry, p. 479)

Task 2: Geometry theorem proving on a larger curated set
- Task type: Reasoning / relational (theorem proving)
- Dataset(s): 231 geometry problems curated in ref. 17
- Domain: Classical Euclidean plane geometry
- Evidence: "We evaluated AlphaGeometry and other baselines on a larger test set of 231 geometry problems, curated in ref. 17." (Evaluation on a larger test set, p. 482)
- Evidence: "This set covers a wider range of sources outside IMO competitions: textbook examples and exercises, regional olympiads and famous geometry theorems" (Evaluation on a larger test set, p. 482)

Task 3: Human expert evaluation of generated proofs for IMO 2000 and 2015 geometry problems
- Task type: Reasoning / relational (theorem proving outputs evaluated by a human expert)
- Dataset(s): Geometry problems from IMO 2000 and 2015 (subset of the IMO geometry problems)
- Domain: Classical Euclidean plane geometry
- Evidence: "To obtain an expert evaluation in 2000 and 2015, during which AlphaGeometry solves all geometry problems and potentially passes the medal threshold, we submit these solutions to the USA IMO team coach" (Human expert evaluation of AlphaGeometry outputs, p. 479)

### 4. Domain and Modality Scope
- Single domain or multiple domains? Single domain (Euclidean plane geometry only). Evidence: "We focus on Euclidean plane geometry and exclude topics such as geometric inequalities and combinatorial geometry." (Introduction, p. 476)
- Multiple domains within the same modality? Not specified beyond classical geometry; evaluation is restricted to geometry problems only. Evidence: "Geometric inequality and combinatorial geometry, for example, cannot be translated, as their formulation is markedly different to classical geometry." (An olympiad-level benchmark for geometry, p. 479)
- Multiple modalities? Not specified in the paper. The representation is a specialized geometry language: "adopted a more specialized language used in GEX10, JGEX17, MMP/Geometer13 and GeoLogic19, a line of work that aims to provide a logical and graphical environment for synthetic geometry theorems with human-like non-degeneracy and topological assumptions." (Methods: Geometry representation, p. 482)
- Domain generalization or cross-domain transfer? Not claimed; only future-work framing. Evidence: "We consider applying this framework to a wider scope as future work and look forward to further innovations that tackle these challenges." (AlphaGeometry framework and applicability to other domains, p. 482)

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| IMO-AG-30 theorem proving | Not specified in the paper | Yes (global fine-tuning is described) | Not specified in the paper | "We first pretrained the language model on all 100 million synthetically generated proofs, including ones of pure symbolic deduction. We then fine-tuned the language model on the subset of proofs that requires auxiliary constructions" (Language model pretraining and fine-tuning, p. 479); "The final test set is named IMO-AG-30" (An olympiad-level benchmark for geometry, p. 479) |
| 231-problem test set | Not specified in the paper | Yes (global fine-tuning is described) | Not specified in the paper | "We first pretrained the language model on all 100 million synthetically generated proofs, including ones of pure symbolic deduction. We then fine-tuned the language model on the subset of proofs that requires auxiliary constructions" (Language model pretraining and fine-tuning, p. 479); "We evaluated AlphaGeometry and other baselines on a larger test set of 231 geometry problems" (Evaluation on a larger test set, p. 482) |
| Human expert evaluation (IMO 2000/2015) | Not specified in the paper | Yes (global fine-tuning is described) | Not specified in the paper | "We first pretrained the language model on all 100 million synthetically generated proofs, including ones of pure symbolic deduction. We then fine-tuned the language model on the subset of proofs that requires auxiliary constructions" (Language model pretraining and fine-tuning, p. 479); "To obtain an expert evaluation in 2000 and 2015" (Human expert evaluation of AlphaGeometry outputs, p. 479) |

### 6. Input and Representation Constraints
- Geometry-only, 2D domain constraint: "We focus on Euclidean plane geometry and exclude topics such as geometric inequalities and combinatorial geometry." (Introduction, p. 476)
- Specialized symbolic geometry language: "adopted a more specialized language used in GEX10, JGEX17, MMP/Geometer13 and GeoLogic19, a line of work that aims to provide a logical and graphical environment for synthetic geometry theorems with human-like non-degeneracy and topological assumptions." (Methods: Geometry representation, p. 482)
- Narrow formulation/coverage limitation: "Owing to its narrow formulation, 75% of all IMO geometry problems can be adapted to this representation." (Methods: Geometry representation, p. 482)
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper (only a maximum context length is given; see Section 7).
- Fixed dimensionality (strictly 2D)? Implied by the Euclidean plane geometry focus. Evidence: "We focus on Euclidean plane geometry" (Introduction, p. 476)
- Padding or resizing requirements? Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length: "We limit the maximum context length to 1,024 tokens" (Methods: Language model architecture and training, p. 482)
- Fixed or variable length? Not specified; only a maximum is stated. Supporting detail: "Sequence packing38,39 is also used because more than 90% of our sequences are under 200 in length." (Methods: Language model architecture and training, p. 482)
- Attention type (global/windowed/hierarchical/sparse): Not specified. Only a generic transformer is described: "The transformer has 12 layers, embedding dimension of 1,024, eight heads of attention" (Methods: Language model architecture and training, p. 482)
- Cost-management mechanisms: beam search and constrained search budget. Evidence: "We use beam search to explore the top k constructions generated by the language model" (Combining language modelling and symbolic engines, p. 478); "we use a beam size of k = 512, the maximum number of iterations is 16 and the branch-ing factor for each node, that is, the decoding batch size, is 32." (Methods: Parallelized proof search, p. 482)

### 8. Positional Encoding (Critical Section)
- Mechanism: Relative positional encoding (T5-style). Evidence: "We limit the maximum context length to 1,024 tokens and use T5-style relative position embedding37." (Methods: Language model architecture and training, p. 482)
- Where applied (input only / every layer / attention bias): Not specified in the paper.
- Fixed across experiments vs modified/ablated: Not specified in the paper.

### 9. Positional Encoding as a Variable
- The paper does not present positional encoding as a research variable; it appears as a fixed architectural choice. Evidence: "We limit the maximum context length to 1,024 tokens and use T5-style relative position embedding37." (Methods: Language model architecture and training, p. 482)
- Multiple positional encodings compared? Not specified in the paper.
- Any claim that PE is not critical/secondary? Not specified in the paper.

### 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size: "Overall, the transformer has 151 million parameters, excluding embedding layers at its input and output heads." (Methods: Language model architecture and training, p. 482)
- Dataset size (synthetic proofs): "We first pretrained the language model on all 100 million synthetically generated proofs, including ones of pure symbolic deduction." (Language model pretraining and fine-tuning, p. 479)
- Dataset generation scale: "After running this process on 100,000 CPU workers for 72 h, we obtained roughly 500 million synthetic proof examples." (Methods: Parallelized data generation and deduplication, p. 482)
- Auxiliary-construction subset size: "A total of 9 million examples involves at least one auxiliary construction." (Methods: Parallelized data generation and deduplication, p. 482)
- Performance gains attributed to architecture components: "incorporating algebraic deduction added seven solved problems to a total of 14 (DD + AR)" (Table 1 discussion, p. 479)
- Training effect (pretraining ablation): "We find that a language model without pretraining only solves 21 problems." (p. 479)
- Data scaling ablation (training data fraction): "We trained AlphaGeometry on smaller fractions of the original training data (20%, 40%, 60% and 80%) and found that, even at 20% of training data, AlphaGeometry still solves 21 problems" (Methods: The effect of data and search, p. 482)
- Search-budget scaling: "using less than 2% of the search budget (beam size of 8 versus 512) during test time, AlphaGeometry can still solve 21 problems." (Table 1 discussion, p. 479)
- Model-size scaling claims? Not specified in the paper.

### 11. Architectural Workarounds
- Neuro-symbolic coupling to handle search: "AlphaGeometry is a neuro-symbolic system that uses a neural language model, trained from scratch on our large-scale synthetic data, to guide a symbolic deduction engine through infinite branching points in challenging problems." (Introduction, p. 476)
- Integrated symbolic engines (DD + AR): "DD and AR are applied alternately to expand their joint deduction closure." (Methods: Integrating DD and AR, p. 482)
- Traceback to minimize proofs/premises: "Each deduction step needs to be coupled with a traceback algorithm, which returns the minimal set of immediate ancestor statements that is necessary to deduce the conclusion statement of the step." (Methods: Traceback to find minimal proofs, p. 482)
- Proof pruning by exhaustive trial: "we perform exhaustive trial and error, discarding each subset of the auxiliary points and rerunning DD + AR on the smaller subset of premises to verify goal reachability." (Methods: Proof pruning, p. 482)
- Beam search and parallelization for search cost: "We use beam search to explore the top k constructions generated by the language model" (Combining language modelling and symbolic engines, p. 478); "This set-up is highly parallelizable across beams, allowing substantial speed-up when there are parallel computational resources." (Methods: Parallelized proof search, p. 482)

### 12. Explicit Limitations and Non-Claims
- Not addressing full formalization of geometry in general-purpose systems: "We do not directly address this challenge as it requires deep expertise and substantial research outside the scope of theorem-proving methodologies." (Methods: Geometry representation, p. 482)
- No complete solution to geometry representation: "We do not push further for a complete solution to geometry representation as it is a separate and extremely challenging research topic that demands substantial investment from the mathematical formalization community." (Methods: Geometry representation, p. 482)
- Explicit domain exclusions: "We focus on Euclidean plane geometry and exclude topics such as geometric inequalities and combinatorial geometry." (Introduction, p. 476)
- Translation limits for non-classical geometry: "Geometric inequality and combinatorial geometry, for example, cannot be translated, as their formulation is markedly different to classical geometry." (An olympiad-level benchmark for geometry, p. 479)
- Future work (applicability beyond geometry): "We consider applying this framework to a wider scope as future work and look forward to further innovations that tackle these challenges." (AlphaGeometry framework and applicability to other domains, p. 482)

### 13. Constraint Profile (Synthesis)
- Domain scope: Single domain (Euclidean plane geometry), with explicit exclusions of inequalities and combinatorial geometry.
- Task structure: The only evaluated task is geometry theorem proving across translated IMO problems and a larger curated geometry set.
- Representation rigidity: Uses a specialized geometry language with a narrow formulation and partial coverage of IMO geometry problems.
- Model sharing vs specialization: A single pretraining + fine-tuning pipeline is described; no task-specific heads or per-task training are specified.
- Positional encoding: Fixed T5-style relative position embedding; no ablations or alternatives discussed.

### 14. Final Classification
Single-task, single-domain. The paper explicitly limits scope to Euclidean plane geometry ("We focus on Euclidean plane geometry and exclude topics such as geometric inequalities and combinatorial geometry.") and evaluates only geometry theorem proving on geometry-specific test sets ("resulting in a test set of 30 classical geometry problems" and "a larger test set of 231 geometry problems"). (Introduction, p. 476; An olympiad-level benchmark for geometry, p. 479; Evaluation on a larger test set, p. 482)

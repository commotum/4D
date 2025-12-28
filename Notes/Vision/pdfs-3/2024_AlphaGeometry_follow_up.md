1. Number of distinct tasks evaluated: 3.
   - Olympiad-level geometry theorem proving on the IMO-AG-30 benchmark (30 problems). Evidence: “resulting in a test set of 30 classical geometry problems. The final test set is named IMO-AG-30” (An olympiad-level benchmark for geometry, p. 479).
   - Geometry theorem proving on a larger curated set (231 problems). Evidence: “We evaluated AlphaGeometry and other baselines on a larger test set of 231 geometry problems” (Evaluation on a larger test set, p. 482).
   - Human expert evaluation of generated proofs for IMO 2000 and 2015 geometry problems. Evidence: “To obtain an expert evaluation in 2000 and 2015... we submit these solutions to the USA IMO team coach” (Human expert evaluation of AlphaGeometry outputs, p. 479).

2. Number of trained model instances required to cover all tasks: 1.
   - Rationale: The paper describes a single AlphaGeometry language model that is pretrained on 100 million synthetic proofs and then fine-tuned on the auxiliary-construction subset, and this same model is used for all evaluations; no task-specific heads or per-task fine-tunes are described. Evidence: “We first pretrained the language model on all 100 million synthetically generated proofs... We then fine-tuned the language model on the subset of proofs that requires auxiliary constructions” (Language model pretraining and fine-tuning, p. 479).

3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$

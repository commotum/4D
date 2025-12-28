1. Number of distinct tasks evaluated: 16.
   - Language modeling (next-token perplexity) on the Arxiv dataset. Evidence: "Datasets. Our analysis involves training language models on the Arxiv and Books3 datasets" (Datasets, p.5).
   - Language modeling (next-token perplexity) on the Books3 dataset. Evidence: "Datasets. Our analysis involves training language models on the Arxiv and Books3 datasets" (Datasets, p.5).
   - CHE benchmark tasks (14 total): Even Pairs; Modular Arithmetic (Simple); Parity Check; Cycle Navigation; Stack Manipulation; Reverse String; Modular Arithmetic; Solve Equation; Duplicate String; Missing Duplicate; Odds First; Binary Addition; Compute Sqrt; Bucket Sort. Evidence: these tasks are listed in Table 2 (Table 2, p.9) and in Appendix D, Table 4 (Table 4, p.18).

2. Number of trained model instances required to cover all tasks: 16.
   - Rationale: The paper trains language models on Arxiv and Books3 as separate datasets (Datasets, p.5). It also evaluates a suite of distinct CHE tasks listed individually in Table 2 and Appendix D (Table 2, p.9; Table 4, p.18), with no single multi-task model described to cover all tasks. Therefore, covering all tasks requires one trained model per task (2 LM datasets + 14 CHE tasks = 16 models).

3. Task-Model Ratio:

$$
\boxed{
\frac{16\ \text{tasks}}{16\ \text{models}} = 1
}
$$

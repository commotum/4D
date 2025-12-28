Number of distinct tasks evaluated: 9 - language modeling benchmarks; long-context retrieval ("needle in a haystack"); MMLU; TriviaQA; ARC; PIQA; HellaSwag; OBQA; Winogrande. (Abstract: "language modeling benchmarks"; Section 4.2: "Long-Context Retrieval (Needle-in-a-Haystack Task)"; Appendix B.1: "benchmark suite included MMLU, TriviaQA, ARC, PIQA, HellaSwag, OBQA, and Winogrande.")

Number of trained model instances required to cover all tasks: 1 - the paper trains models once on Smollm-Corpus and evaluates that same trained model across the downstream task suite via the LM Evaluation Harness, with no task-specific heads or fine-tuning described. (Appendix B.1: "All models were trained from scratch on the Smollm-Corpus...", and "Downstream task evaluation utilized the EleutherAI LM Evaluation Harness... The benchmark suite included...")

$$
\boxed{
\frac{9\ \text{tasks}}{1\ \text{model}} = 9
}
$$

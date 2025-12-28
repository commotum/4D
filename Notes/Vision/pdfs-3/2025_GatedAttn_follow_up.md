Number of distinct tasks evaluated: 8.
- Language modeling perplexity (PPL) on held-out test sets. Evidence: "We also report the perplexity (PPL) of language modeling on diverse held-out test sets, including domains like English, Chinese, Code, Math, Law, and Literature." (Section 3.1 Experimental Setups)
- Hellaswag. Evidence: "We test the few-shots results on popular benchmarks, including Hellaswag (Zellers et al., 2019) for English..." (Section 3.1 Experimental Setups)
- MMLU. Evidence: "We test the few-shots results on popular benchmarks, including... MMLU (Hendrycks et al., 2020) for general knowledge..." (Section 3.1 Experimental Setups)
- GSM8k. Evidence: "We test the few-shots results on popular benchmarks, including... GSM8k (Cobbe et al., 2021) for math reasoning..." (Section 3.1 Experimental Setups)
- HumanEval. Evidence: "We test the few-shots results on popular benchmarks, including... HumanEval (Chen et al., 2021) for coding..." (Section 3.1 Experimental Setups)
- C-eval. Evidence: "We test the few-shots results on popular benchmarks, including... C-eval (Huang et al., 2024)..." (Section 3.1 Experimental Setups)
- CMMLU. Evidence: "We test the few-shots results on popular benchmarks, including... CMMLU (Li et al., 2023) for Chinese proficiency." (Section 3.1 Experimental Setups)
- RULER (long-context benchmark). Evidence: "We evaluate models on the RULER benchmark (Hsieh et al., 2024) and summarize results in Tab. 5." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

Number of trained model instances required to cover all tasks: 2.
- Model instance for standard benchmarks and PPL via few-shot evaluation (no task-specific heads or fine-tuning described). Evidence: "We test the few-shots results on popular benchmarks, including..." and "We also report the perplexity (PPL) of language modeling..." (Section 3.1 Experimental Setups)
- Separate long-context-extended model instance for RULER, requiring additional training and RoPE/YaRN extension. Evidence: "We increase the RoPE base from 10k to 1M and continue training on data with a sequence length of 32k for an additional 80B tokens... Subsequently, we use YaRN (Peng et al., 2023) to extend the context length to 128k. We evaluate models on the RULER benchmark..." (Section 4.4 SDPA Output Gating Facilitates Context Length Extension)

Task-Model Ratio:
$$
\boxed{
\frac{8\ \text{tasks}}{2\ \text{models}} = 4
}
$$

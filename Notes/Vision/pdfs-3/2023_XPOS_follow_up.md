1. Number of distinct tasks evaluated: 4.
   - Task 1: Language modeling (perplexity) on PG22. Evidence: "We measure perplexity on long document datasets" and "We use books from Project Gutenberg... PG22." (Section 4.2, p.14594)
   - Task 2: Language modeling (perplexity) on QMSum. Evidence: "Besides, we pick QMSum (Zhong et al., 2021) from SCROLLS..." (Section 4.2, p.14594)
   - Task 3: Language modeling evaluation on Arxiv. Evidence: "we run language modeling evaluation on Arxiv..." (Appendix A, p.14600)
   - Task 4: Language modeling evaluation on NarrativeQA. Evidence: "we run language modeling evaluation on... NarrativeQA" (Appendix A, p.14600)

2. Number of trained model instances required to cover all tasks: 1.
   - Rationale: The paper pre-trains a Transformer from scratch and evaluates perplexity on multiple datasets without any task-specific heads or fine-tuning, so one trained LM instance covers all four dataset evaluations. Evidence: "To fairly evaluate different Transformer variants, we pre-train the Transformer from scratch." (Section 4.1, p.14594) plus the multi-dataset evaluation statements in Section 4.2 and Appendix A (p.14594, p.14600).

3. Task-Model Ratio:

$$
\boxed{
\frac{4\ \text{tasks}}{1\ \text{model}} = 4
}
$$

1. Number of distinct tasks evaluated: 9.
   - QA on 2WikiMQA. Evidence: "QA" and "2WikiMQA" are listed under Table 2 (Table 2, p. 9975).
   - QA on HotpotQA. Evidence: "HotpotQA" is listed under Table 2 (Table 2, p. 9975).
   - QA on MultiFieldQA-en. Evidence: "MultiFieldQA-en" is listed under Table 2 (Table 2, p. 9975).
   - Summarization on MultiNews. Evidence: "Summarization" and "MultiNews" are listed under Table 2 (Table 2, p. 9975).
   - Summarization on GovReport. Evidence: "GovReport" is listed under Table 2 (Table 2, p. 9975).
   - Code completion on LCC. Evidence: "Code Completion" and "LCC" are listed under Table 2 (Table 2, p. 9975).
   - Language modeling on WikiText-103. Evidence: "Dataset" and "WikiText-103" are listed in Table 3 (Table 3, p. 9976).
   - Language modeling on OpenWebText2. Evidence: "OpenWebText2" is listed in Table 3 (Table 3, p. 9976).
   - Language modeling on ArXiv. Evidence: "ArXiv" is listed in Table 3 (Table 3, p. 9976).

2. Number of trained model instances required to cover all tasks: 1.
   - Rationale: The paper evaluates single LLMs across multiple generation tasks ("choose three trending LLMs ... to evaluate their performance on various generation tasks") and does not describe task-specific heads or per-task fine-tuning; the language modeling results are compiled from existing models into Table 3 ("statistically collect results from published literatures and form Table 3"). Evidence: Appendix A.1 and A.2 (p. 9974-9975).

3. Task-Model Ratio:

$$
\boxed{
\frac{9\ \text{tasks}}{1\ \text{model}} = 9
}
$$

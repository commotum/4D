Number of distinct tasks evaluated: 2
- Synthetic key-value retrieval in a controlled experiment (attention head retrieves v_i from sampled key-value pairs). (2025_RoPEDI.pdf, Section 4 Controlled Experiment)
- Long-context question answering (20-document QA setup derived from NaturalQuestions-Open). (2025_RoPEDI.pdf, Section 5.1; 2025_RoPEDI.pdf, Appendix C)

Number of trained model instances required to cover all tasks: 2
- One trained attention model for the synthetic retrieval controlled experiment (trained separately with/without RoPE as an ablation). (2025_RoPEDI.pdf, Appendix A)
- One LLM instance for long-context QA (the paper inspects existing LLMs via prompting; no single model is trained to do both synthetic retrieval and long-context QA). (2025_RoPEDI.pdf, Section 5; 2025_RoPEDI.pdf, Section 5.2)

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$

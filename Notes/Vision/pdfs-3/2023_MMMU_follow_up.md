1. Number of distinct tasks evaluated: 2
   - Multiple-choice multimodal question answering (Appendix H.1 “Types of Questions: • Multiple-Choice Questions: Including standard multiple-choice questions and true/false questions. These are characterized by a question followed by several answer choices, with only one correct option.”; Section 4: “For all models, we use the default prompt provided by each model for multi-choice or open QA, if available.”)
   - Open-ended multimodal question answering (Appendix H.1 “• Open-Ended Questions: Encompassing formats like factoid, fill-in-the-blank, calculation-based, and short descriptive responses.”; Section 4: “For all models, we use the default prompt provided by each model for multi-choice or open QA, if available.”)

2. Number of trained model instances required to cover all tasks: 1
   - The evaluation is zero-shot without task-specific fine-tuning, and the same model is prompted for multi-choice or open QA: “Our evaluation is conducted under a zero-shot setting to assess the capability of models to generate accurate answers without fine-tuning or few-shot demonstrations on our benchmark.” and “For all models, we use the default prompt provided by each model for multi-choice or open QA, if available.” (Section 4).

3. Task-Model Ratio

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$

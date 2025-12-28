Number of distinct tasks evaluated: 2
- Task 1: Science question answering (multiple-choice; select the correct answer from options given the question and multimodal contexts). (2022_ScienceQA.pdf, Section 3 Dataset)
- Task 2: Lecture and explanation generation (chain-of-thought generation that outputs the answer followed by lecture and explanation; evaluated as generated lectures/explanations). (2022_ScienceQA.pdf, Section 4.2; Section 5.1)

Number of trained model instances required to cover all tasks: 1
- A single chain-of-thought model is fine-tuned to generate a long sequence consisting of the answer followed by the lecture and explanation, so one model instance can both answer questions and generate lectures/explanations without a task-specific head. (2022_ScienceQA.pdf, Section 4.2)

$$
\boxed{
\frac{2\ \text{tasks}}{1\ \text{model}} = 2
}
$$

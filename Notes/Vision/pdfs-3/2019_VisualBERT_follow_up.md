Number of distinct tasks evaluated: 4 (VQA 2.0, VCR, NLVR2, Flickr30K). Evidence: "We evaluate VisualBERT on four different types of vision-and-language applications: (1) Visual Question Answering (VQA 2.0) ... (2) Visual Commonsense Reasoning (VCR) ... (3) Natural Language for Visual Reasoning (NLVR2) ... and (4) Region-to-Phrase Grounding (Flickr30K)." (Section 4 Experiment, p.4)

Number of trained model instances required to cover all tasks: 4. Each downstream task uses task-specific pre-training and then fine-tuning with task-specific input/output/objective, so one separately trained model instance is required per task. Evidence: "Task-Specific Pre-Training... using the data of the task" and "Fine-Tuning... a task-specific input, output, and objective are introduced" (Section 3.3, p.3); also "to apply BERT to a particular task, a task-specific input, output layer, and objective are introduced, and the model is fine-tuned on the task data" (Section 3.1, p.3).

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$

Number of distinct tasks evaluated: 6.
Evidence: The paper explicitly says it evaluates on six V+L tasks and lists them: Visual Question Answering (VQA), Visual Commonsense Reasoning (VCR), NLVR2, Visual Entailment, Image-Text Retrieval, and Referring Expression Comprehension (abstract and introduction list the six tasks; e.g., "across six V+L tasks (over nine datasets), including Visual Question Answering, Image-Text Retrieval, Referring Expression Comprehension, Visual Commonsense Reasoning, Visual Entailment, and NLVR2." (2020_UNITER.pdf, p.1-3)).

Number of trained model instances required to cover all tasks: 6 separate fine-tuned models (one per task).
Evidence: The paper states it transfers the pre-trained UNITER to each target task and fine-tunes it: "We evaluate UNITER on six V+L tasks by transferring the pre-trained model to each target task and finetuning through end-to-end training." (2020_UNITER.pdf, p.8). It also specifies task-specific heads trained during fine-tuning: "For VQA, VCR, NLVR2, Visual Entailment and Image-Text Retrieval, we extract the joint embedding ... via a multi-layer perceptron (MLP) from the representation of the [CLS] token. For RE Comprehension, we use the MLP to compute the region-wise alignment scores. These MLP layers are learned during the finetuning stage." (2020_UNITER.pdf, p.9). Because each task requires its own fine-tuning and head, a practitioner would need a separate trained model instance per task.

$$
\boxed{
\frac{6\ \text{tasks}}{6\ \text{models}} = 1
}
$$

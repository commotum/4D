1. Number of distinct tasks evaluated: 3 tasks (EgoMCQ multiple-choice questions; EK-100 MIR multi-instance retrieval; Charades-Ego action recognition). Evidence: "After pretraining, we evaluate models on the Ego4D Multiple-Choice Questions (EgoMCQ) benchmark. Before fine-tuning, we directly evaluate the pretrained model on EK-100's multi-instance retrieval (MIR) challenge and the Charades-Ego action recognition challenge..." (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2025_EVA02AT.txt).
2. Number of trained model instances required to cover all tasks: 3 models (one pretrained model instance for EgoMCQ; separate fine-tuned model instances for EK-100 MIR and Charades-Ego, trained "respectively"). Evidence: "After pretraining, we evaluate models on the Ego4D Multiple-Choice Questions (EgoMCQ) benchmark... After that, we fine-tune the pretrained model on the training set of these two benchmarks, respectively" (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2025_EVA02AT.txt).
3. Task-Model Ratio:
$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

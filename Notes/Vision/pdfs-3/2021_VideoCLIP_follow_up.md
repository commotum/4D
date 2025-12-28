1. Number of distinct tasks evaluated: 4 (text-video retrieval, VideoQA, action localization, action segmentation). Evidence: "We directly use our pre-trained model on a diverse set of four tasks in five datasets, including text-video retrieval ... VideoQA ... action localization ... and segmentation" and the same paragraph enumerates these four tasks. "After pre-training, we apply our model for zero-shot transfer without any fine-tuning on target dataset labels." (2021_VideoCLIP.txt)

2. Number of trained model instances required to cover all tasks: 1. Evidence: "After pre-training, we apply our model for zero-shot transfer without any fine-tuning on target dataset labels. We directly use our pre-trained model on a diverse set of four tasks..." indicates a single pre-trained model instance is reused across all tasks without task-specific fine-tuning or separate heads (2021_VideoCLIP.txt).

3. Task-Model Ratio:

$$
\boxed{
\frac{4\ \text{tasks}}{1\ \text{model}} = 4
}
$$

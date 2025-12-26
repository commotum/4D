The term “multi-task” is probably overloaded here. In many papers, “multi-task” refers to evaluating a single architecture across multiple tasks or datasets, often via repeated fine-tuning or training separate task-specific models. In this sense, the architecture is shared, but task capability is fragmented across multiple trained model instances.

This differs fundamentally from a stronger notion of multi-task learning, where a single trained model instance simultaneously supports multiple distinct tasks using shared weights, a unified representation, and a common inference interface.

Does the paper demonstrate a single trained model instance that can perform multiple distinct tasks without task-specific fine-tuning, relying only on shared weights and input conditioning (e.g., prompts, task tokens, or unified objectives)? Or are separate models, fine-tuned weights, or task-specific heads required for each task?

Please report the following, citing evidence:

1. **Number of distinct tasks evaluated**
2. **Number of distinct trained model instances produced**
3. **Task–Model Ratio = (1) / (2)**

For example:

* Pretrain once → fine-tune 8 times → 8 models for 8 tasks
  → **Ratio = 1**
* Jointly train one model on 8 tasks
  → **Ratio = 8**
* Single model trained on a task-agnostic corpus that performs N tasks via conditioning
  → **Ratio = N (unbounded)**
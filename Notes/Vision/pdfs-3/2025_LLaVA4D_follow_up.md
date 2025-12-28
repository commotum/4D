Number of distinct tasks evaluated: 3 (dense captioning, visual QA, visual grounding). Evidence: "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks..." (2025_LLaVA4D.txt, Sec. 4.1).

Number of trained model instances required to cover all tasks: 1. Evidence: the same LLaVA-4D model is trained across tasks in a single three-stage pipeline where Stage 1 uses DC+QA, Stage 2 uses VG, and Stage 3 performs "multi-task instruction fine-tuning" (2025_LLaVA4D.txt, Sec. 4.2).

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$

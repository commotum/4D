1. Number of distinct tasks evaluated: 5 (Image-Text Retrieval, Visual Entailment, Visual Question Answering (VQA), Natural Language for Visual Reasoning (NLVR2), and Visual Grounding). (source: `2021_ALBEF.txt:362`, `2021_ALBEF.txt:364`, `2021_ALBEF.txt:376`, `2021_ALBEF.txt:380`, `2021_ALBEF.txt:413`, `2021_ALBEF.txt:426`)
2. Number of trained model instances required to cover all tasks: 5. The paper adapts the pre-trained model to five downstream tasks and describes a fine-tuning strategy plus task-specific heads/decoders or architectural extensions per task (retrieval fine-tuning; VE MLP classifier; VQA decoder; NLVR2 encoder extension + MLP classifier; visual grounding fine-tuning on RefCOCO+). (source: `2021_ALBEF.txt:362`, `2021_ALBEF.txt:365`, `2021_ALBEF.txt:376`, `2021_ALBEF.txt:383`, `2021_ALBEF.txt:413`, `2021_ALBEF.txt:420`, `2021_ALBEF.txt:428`)
3. Task-Model Ratio:

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$

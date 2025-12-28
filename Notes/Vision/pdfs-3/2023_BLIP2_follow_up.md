1. Number of distinct tasks evaluated: 5
   - Instructed zero-shot image-to-text generation (Section 4.1: "BLIP-2 effectively enables a LLM to understand images while preserving its capability in following text prompts, which allows us to control image-to-text generation with instructions.")
   - Zero-shot visual question answering (Section 4.1: "Zero-shot VQA. We perform quantitative evaluation on the zero-shot visual question answering task.")
   - Image captioning (Section 4.2: "We finetune BLIP-2 models for the image captioning task, which asks the model to generate a text description for the image's visual content.")
   - Visual question answering with supervised fine-tuning (Section 4.3: "Given annotated VQA data, we finetune the parameters of the Q-Former and the image encoder while keeping the LLM frozen.")
   - Image-text retrieval (Section 4.4: "Since image-text retrieval does not involve language generation, we directly finetune the first-stage-pretrained model w/o LLM.")

2. Number of trained model instances required to cover all tasks: 4
   - One zero-shot model instance covers instructed image-to-text generation and zero-shot VQA via prompting (Section 4.1: "We simply append the text prompt after the visual prompt as input to the LLM." and "For OPT models, we use the prompt 'Question: {} Answer:'. For FlanT5 models, we use the prompt 'Question: {} Short answer:'.")
   - Image captioning requires a task-specific fine-tuned model (Section 4.2: "We finetune BLIP-2 models for the image captioning task...")
   - VQA fine-tuning requires a task-specific fine-tuned model (Section 4.3: "Given annotated VQA data, we finetune the parameters of the Q-Former and the image encoder while keeping the LLM frozen.")
   - Image-text retrieval requires a task-specific fine-tuned model without the LLM (Section 4.4: "Since image-text retrieval does not involve language generation, we directly finetune the first-stage-pretrained model w/o LLM.")

3. Task-Model Ratio

$$
\boxed{
\frac{5\ \text{tasks}}{4\ \text{models}} = 1.25
}
$$

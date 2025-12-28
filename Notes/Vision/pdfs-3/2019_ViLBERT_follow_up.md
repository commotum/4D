Number of distinct tasks evaluated: 5 (VQA, VCR, RefCOCO+ referring expressions, caption-based image retrieval, and zero-shot caption-based image retrieval). Evidence: "We transfer our pretrained ViLBERT model to a set of four established vision-and-language tasks ... and one diagnostic task." plus the task sections titled "Visual Question Answering (VQA)", "Visual Commonsense Reasoning (VCR)", "Grounding Referring Expressions", "Caption-Based Image Retrieval", and "'Zero-shot' Caption-Based Image Retrieval" (2019_ViLBERT.txt, Sec. 3.2).

Number of trained model instances required to cover all tasks: 5. Evidence: the paper states they "follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end" and "In all cases, the modification is trivial - typically amounting to learning a classification layer," which implies separate fine-tuned models for each of the four transfer tasks (VQA, VCR, RefCOCO+, caption-based retrieval) (2019_ViLBERT.txt, Sec. 3.2). The diagnostic zero-shot retrieval uses the pretrained model without fine-tuning: "we directly apply the pretrained ... without finetuning" and "We directly use the ViLBERT model trained on Conceptual Captions dataset" (2019_ViLBERT.txt, Sec. 3.2). Thus, 4 fine-tuned models + 1 pretrained model = 5 model instances.

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$

1. Number of distinct tasks evaluated: 4 (VQAv2, NLVR2, image-to-text retrieval, text-to-image retrieval). (source: `2021_ViLT.txt:756`, `2021_ViLT.txt:757`, `2021_ViLT.txt:1166`, `2021_ViLT.txt:1167`)
2. Number of trained model instances required to cover all tasks: 3. VQAv2 is fine-tuned as a classification task; NLVR2 uses its own task-specific head; retrieval is fine-tuned once and evaluated for both image-to-text and text-to-image retrieval, so one retrieval model covers both retrieval directions. (source: `2021_ViLT.txt:798`, `2021_ViLT.txt:800`, `2021_ViLT.txt:806`, `2021_ViLT.txt:808`, `2021_ViLT.txt:1155`, `2021_ViLT.txt:1161`, `2021_ViLT.txt:763`, `2021_ViLT.txt:1166`, `2021_ViLT.txt:1167`)
3. Task-Model Ratio:

$$
\boxed{
\frac{4\ \text{tasks}}{3\ \text{models}} = 1.33
}
$$

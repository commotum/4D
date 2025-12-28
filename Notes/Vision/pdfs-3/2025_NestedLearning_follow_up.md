1. Number of distinct tasks evaluated: 40 (CLINC, Banking, DBpedia, LongHealth, QASPER, MK-NIAH, CTNL; NIAH single-needle pass-key/number/uuid, NIAH multi-query, NIAH multi-value, BABILong; Wikitext, LMB, PIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, SIQA, BoolQ; SWDE, NQ, DROP, FDA, SQuAD, TQA; MAD tasks: Compress., ICR, Fuzzy ICR, Copying, Selective Memory; formal language recognition tasks: Parity, (aa)*, (abab)*, a^n b^n, a^n b^n c^n, Shuffle-2; ImageNet-21K classification). [2025_NestedLearning.pdf:34] [2025_NestedLearning.pdf:35] [2025_NestedLearning.pdf:36] [2025_NestedLearning.pdf:37] [2025_NestedLearning.pdf:38] [2025_NestedLearning.pdf:39]
2. Number of trained model instances required to cover all tasks: 5 (Llama3-based Hope with continual pre-training for the continual learning/in-context tasks; Hope trained from scratch on ~50B tokens for NIAH; a BABILong fine-tuned Hope model; Hope trained from scratch on 30B/100B tokens for language modeling/commonsense and the recall/MAD/formal-language tasks; ViT trained on ImageNet-21K for vision). [2025_NestedLearning.pdf:34] [2025_NestedLearning.pdf:36] [2025_NestedLearning.pdf:37] [2025_NestedLearning.pdf:38] [2025_NestedLearning.pdf:39]
3. Task-Model Ratio:
$$
\boxed{
\frac{40\ \text{tasks}}{5\ \text{models}} = 8
}
$$
[2025_NestedLearning.pdf:34] [2025_NestedLearning.pdf:35] [2025_NestedLearning.pdf:36] [2025_NestedLearning.pdf:37] [2025_NestedLearning.pdf:38] [2025_NestedLearning.pdf:39]

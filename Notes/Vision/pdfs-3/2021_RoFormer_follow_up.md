Task capability analysis (RoFormer)

Distinct tasks evaluated (11):
1) Long text classification benchmark datasets (abstract: "we evaluate ... on various long text classification benchmark datasets"). (2021_RoFormer.txt:48-49)
2) Machine translation (WMT14 En-De; "sequence-to-sequence language translation tasks"). (2021_RoFormer.txt:904-910)
3) Pre-training language modeling with MLM loss (BookCorpus + Wikipedia). (2021_RoFormer.txt:931-942)
4) GLUE tasks: MRPC, SST-2, QNLI, STS-B, QQP, MNLI (6 tasks). (2021_RoFormer.txt:1022-1029)
5) Language modeling with Performer + RoPE on Enwik8 ("pre-training task of language modeling"). (2021_RoFormer.txt:1086-1101)
6) Semantic text matching on CAIL2019-SCM ("semantic text matching"; task description). (2021_RoFormer.txt:1219-1231)

Total tasks = 1 + 1 + 1 + 6 + 1 + 1 = 11.

Trained model instances required (11):
- GLUE tasks are fine-tuned separately ("fine-tune each of the aforementioned downstream tasks"), so 6 models for the 6 GLUE tasks. (2021_RoFormer.txt:1036-1037)
- Machine translation, MLM pre-training, Performer LM, and CAIL2019-SCM are each trained/evaluated in their own experiments, implying distinct trained instances for those tasks. (2021_RoFormer.txt:904-910, 931-942, 1086-1101, 1238-1239)
- Long text classification benchmarks are evaluated but datasets are unspecified in the text; counted as one task-type/model instance here. (2021_RoFormer.txt:48-49)

Task–Model Ratio:
$$
\boxed{
\frac{11\ \text{tasks}}{11\ \text{models}} = 1
}
$$

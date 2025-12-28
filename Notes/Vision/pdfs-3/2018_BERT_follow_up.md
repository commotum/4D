Number of distinct tasks evaluated: 12
- Task 1: MNLI (Multi-Genre Natural Language Inference). (2018_BERT.pdf.txt, Appendix B.1)
- Task 2: QQP (Quora Question Pairs). (2018_BERT.pdf.txt, Appendix B.1)
- Task 3: QNLI (Question Natural Language Inference). (2018_BERT.pdf.txt, Appendix B.1)
- Task 4: SST-2 (Stanford Sentiment Treebank). (2018_BERT.pdf.txt, Appendix B.1)
- Task 5: CoLA (Corpus of Linguistic Acceptability). (2018_BERT.pdf.txt, Appendix B.1)
- Task 6: STS-B (Semantic Textual Similarity Benchmark). (2018_BERT.pdf.txt, Appendix B.1)
- Task 7: MRPC (Microsoft Research Paraphrase Corpus). (2018_BERT.pdf.txt, Appendix B.1)
- Task 8: RTE (Recognizing Textual Entailment). (2018_BERT.pdf.txt, Appendix B.1)
- Task 9: SQuAD v1.1 (extractive QA). (2018_BERT.pdf.txt, Section 4.2)
- Task 10: SQuAD v2.0 (extractive QA with unanswerable questions). (2018_BERT.pdf.txt, Section 4.3)
- Task 11: SWAG (commonsense sentence completion). (2018_BERT.pdf.txt, Section 4.4)
- Task 12: CoNLL-2003 Named Entity Recognition (NER). (2018_BERT.pdf.txt, Section 5.3)

Number of trained model instances required to cover all tasks: 12
- The paper states that "Each downstream task has separate fine-tuned models" even though they share the same pre-trained initialization, implying one trained model per task. (2018_BERT.pdf.txt, Section 3)
- Each task uses task-specific inputs/outputs and output layers, so different tasks require separate heads and fine-tuning. (2018_BERT.pdf.txt, Section 3.2)

$$
\boxed{
\frac{12\ \text{tasks}}{12\ \text{models}} = 1
}
$$

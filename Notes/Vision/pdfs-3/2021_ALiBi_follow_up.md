Number of distinct tasks evaluated: 3
- Task 1: Language modeling (next-token prediction) on WikiText-103. (p.2)
- Task 2: Language modeling on the Toronto BooksCorpus. (Appendix A.3, p.20)
- Task 3: Language modeling on the CC100+RoBERTa corpus (RoBERTa training corpora + CC-100 English). (p.7)

Number of trained model instances required to cover all tasks: 3
- A model is trained/evaluated on WikiText-103 to develop and test the method. (p.2; p.6)
- A separate experiment is run on a different domain (Toronto BooksCorpus). (p.6; Appendix A.3, p.20)
- A separate large-scale training run uses the CC100+RoBERTa corpus. (p.7)

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

Number of distinct tasks evaluated: 3
- Task 1: English-to-German machine translation on WMT 2014 English-German. (p.7)
- Task 2: English-to-French machine translation on WMT 2014 English-French. (p.7)
- Task 3: English constituency parsing on the WSJ portion of the Penn Treebank (plus a semi-supervised variant on additional corpora). (p.9)

Number of trained model instances required to cover all tasks: 3
- Separate training runs are described for English-German translation and English-French translation, implying distinct models per task/dataset. (p.7)
- A separate 4-layer Transformer is trained for English constituency parsing, indicating another task-specific model instance. (p.9)

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

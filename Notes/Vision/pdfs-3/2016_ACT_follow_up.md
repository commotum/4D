1. Number of distinct tasks evaluated: 5.
Evidence: The paper reports "four synthetic problems: determining the parity of binary vectors, applying binary logic operations, adding integers, and sorting real numbers" and additionally presents "character-level language modelling results on the Hutter prize Wikipedia dataset" (2016_ACT.txt, Abstract). It also summarizes the experiments as "four synthetic tasks and one real-world language processing task" (2016_ACT.txt, Section 3).

2. Number of trained model instances required to cover all tasks: 5.
Evidence: Each task is trained with a task-specific architecture and output head, implying separate model instances per task: parity uses a "simple RNN" with a "single sigmoidal output unit" (2016_ACT.txt, Section 3.1); logic uses a "single-layer LSTM" with a "single sigmoidal unit" (2016_ACT.txt, Section 3.2); addition uses a "single-layer LSTM" with "size 11 softmax" outputs (2016_ACT.txt, Section 3.3); sort uses a "single-layer LSTM" with a "size 15 softmax" output layer (2016_ACT.txt, Section 3.4); Wikipedia uses a "single layer of 1500 cells" with a "size 256 softmax classification layer" (2016_ACT.txt, Section 3.5).

$$
\boxed{
\frac{5\ \text{tasks}}{5\ \text{models}} = 1
}
$$

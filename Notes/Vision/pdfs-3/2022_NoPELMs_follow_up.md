Number of distinct tasks evaluated: 5
- Task 1: Causal language modeling (next-token prediction) on WikiText-103. Evidence: the paper frames the task as predicting the next token and defines the "Canonical Setting (WikiText-103)." (Section 3 Experiment Setup)
- Task 2: Causal language modeling (next-token prediction) on The Pile. Evidence: the paper defines the "Large-Scale Setting (The Pile)" and evaluates perplexity there. (Section 3 Experiment Setup)
- Task 3: Masked language modeling on The Pile (RoBERTa large). Evidence: "training a masked language model based on RoBERTa large ... on the Pile" and Table 4 reports MLM perplexities. (Section 5 Analysis; Table 4)
- Task 4: Absolute position prediction probe (token position classification). Evidence: "2-layer feed-forward ReLU network to predict the absolute position (0 to 1023) of each token." (Section 5 Analysis)
- Task 5: Word order sensitivity via prefix shuffling. Evidence: "shuffle the order of the tokens in the prefix and compute the loss only for that specific token" and the experiment is run on WikiText-103. (Appendix B Word Order Analysis)

Number of trained model instances required to cover all tasks: 4
- One causal LM trained from scratch for WikiText-103 (canonical setting). Evidence: models are "trained from scratch" in the canonical WikiText-103 setting. (Section 3 Experiment Setup)
- One causal LM trained from scratch for The Pile (large-scale setting). Evidence: the Large-Scale Setting uses a separate Pile-trained LM. (Section 3 Experiment Setup)
- One masked LM trained on The Pile (RoBERTa large). Evidence: "training a masked language model based on RoBERTa large ... on the Pile." (Section 5 Analysis; Table 4)
- One task-specific probe head for position prediction on top of the frozen LM representations. Evidence: a "2-layer feed-forward ReLU network" is trained to predict positions, while LM weights are not changed. (Section 5 Analysis)
- The word-order analysis reuses the WikiText-103 NoPos LM without training a new model, so it does not add an extra trained instance. (Appendix B Word Order Analysis)

$$
\boxed{
\frac{5\ \text{tasks}}{4\ \text{models}} = 1.25
}
$$

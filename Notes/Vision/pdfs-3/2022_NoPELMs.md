# 2022_NoPELMs Survey Answers

1. Basic Metadata
- Title: Transformer Language Models without Positional Encodings Still Learn Positional Information.
  Evidence: "Transformer Language Models without Positional Encodings Still Learn Positional Information" (Title block).
- Authors: Adi Haviv; Ori Ram; Ofir Press; Peter Izsak; Omer Levy.
- Year: 2022.
  Evidence: "arXiv:2203.16634v2 [cs.CL] 5 Dec 2022" (front matter).
- Venue: arXiv (cs.CL).
  Evidence: "arXiv:2203.16634v2 [cs.CL] 5 Dec 2022" (front matter).

2. One-Sentence Contribution Summary
The paper shows that causal transformer language models can be trained without explicit positional encodings while remaining competitive, and that positional information can be inferred implicitly via the causal attention mask.
Evidence:
- "NoPos models are competitive with position-aware models" (Introduction).
- "tied to the causal attention mask, which implicitly injects positional information into the self-attention layer in order to preserve the autoregressive nature of language models." (Introduction).

3. Tasks Evaluated
- Task: Causal language modeling (next-token prediction)
  - Task type: Generation (next-token LM)
  - Dataset(s): WikiText-103; The Pile
  - Domain: natural language text (Wikipedia; multi-source English text)
  - Evidence: "Intuitively, encoding positional information explicitly is crucial for enabling transformer language models to predict the next token in a sequence." (Section 3 Experiment Setup)
  - Evidence: "The Canonical Setting (WikiText-103). The WikiText-103 corpus (Merity et al., 2017) consists of over 100 million words extracted from a set of high-quality Wikipedia articles." (Section 3)
  - Evidence: "The Large-Scale Setting (The Pile). The Pile (Gao et al., 2020) is an 800GB English text dataset composed of Common Crawl and 22 other diverse sources." (Section 3)
  - Evidence: "we compared the validation set perplexity of models trained from scratch with no explicit positional information (denoted as NoPos) to those trained with the various positional encoding methods discussed in Section 2." (Section 3 Experiment Setup)

- Task: Masked language modeling (MLM)
  - Task type: Reconstruction / Other (MLM)
  - Dataset(s): The Pile (excerpt)
  - Domain: natural language text
  - Evidence: "masked language models (MLM) (Devlin et al., 2019), which use order-invariant attention (since no causal mask is applied)." (Introduction)
  - Evidence: "We tested this corollary by training a masked language model based on RoBERTa large (Liu et al., 2019) on the Pile (see App. C for hyperparameters)." (Section 5 Analysis)
  - Evidence: "The model architecture is based on RoBERTa large (Liu et al., 2019), and processes 128 tokens per sequence." (Table 4 caption)

- Task: Position prediction probe (absolute position classification)
  - Task type: Classification
  - Dataset(s): The Pile (representations from 1.3B models trained on 1024-token sequences)
  - Domain: natural language text
  - Evidence: "2-layer feed-forward ReLU network to predict the absolute position (0 to 1023) of each token (i.e., as a multiclass classification problem)." (Section 5 Analysis)
  - Evidence: "We used the 1.3B parameter models trained over 1024-token sequences of the Pile (Section 3)." (Footnote 4)

- Task: Word order sensitivity (prefix shuffling test)
  - Task type: Other (order-sensitivity analysis)
  - Dataset(s): WikiText-103
  - Domain: natural language text
  - Evidence: "shuffle the order of the tokens in the prefix and compute the loss only for that specific token." (Appendix B Word Order Analysis)
  - Evidence: "The experiment was conducted on the NoPos model with an input sequence length of 512 using the WikiText-103 dataset." (Appendix B)

4. Domain and Modality Scope
- Domain/modality: Single modality (text) with multiple text domains/sources.
  - Evidence: "The WikiText-103 corpus (Merity et al., 2017) consists of over 100 million words extracted from a set of high-quality Wikipedia articles." (Section 3)
  - Evidence: "The Pile (Gao et al., 2020) is an 800GB English text dataset composed of Common Crawl and 22 other diverse sources." (Section 3)
- Domain generalization or cross-domain transfer: Not claimed.
  - Evidence: "Overall, we show that transformer language modeling without explicit positional encoding is robust to the selection of corpus, model size, and sequence length." (Section 4 Results)

5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Causal LM (WikiText-103) | No (trained from scratch per setting) | No | No | "we compared the validation set perplexity of models trained from scratch with no explicit positional information (denoted as NoPos) to those trained with the various positional encoding methods discussed in Section 2." (Section 3 Experiment Setup) |
| Causal LM (The Pile) | No (trained from scratch per setting) | No | No | "we compared the validation set perplexity of models trained from scratch with no explicit positional information (denoted as NoPos) to those trained with the various positional encoding methods discussed in Section 2." (Section 3 Experiment Setup) |
| Masked LM (The Pile) | No | No | No | "We tested this corollary by training a masked language model based on RoBERTa large (Liu et al., 2019) on the Pile (see App. C for hyperparameters)." (Section 5 Analysis) |
| Position prediction probe | Yes (LM weights reused, frozen) | No (LM frozen) | Yes (probe MLP) | "2-layer feed-forward ReLU network to predict the absolute position (0 to 1023) of each token (i.e., as a multiclass classification problem)." and "Notably, we do not change the weights of the evaluated LMs" (Section 5 Analysis) |
| Word order sensitivity | Yes (NoPos LM reused) | No | No | "The experiment was conducted on the NoPos model with an input sequence length of 512 using the WikiText-103 dataset." (Appendix B) |

6. Input and Representation Constraints
- Word-level tokenization (WikiText-103): "tokenized at the word level" (Section 3).
- GPT-2 tokenizer (Pile): "We used GPT-2's tokenizer (Radford et al., 2019) to convert the text into token sequences over a vocabulary of 50K tokens." (Section 3).
- Fixed sequence length in canonical setting: "input sequence length, which was shortened to 512 tokens (instead of 3072)" (Section 3).
- Default sequence length in large-scale setting: "The default input sequence length is 1024 tokens." (Section 3).
- Sequence length variants: "Specifically, we experiment with sequences of lengths {256, 512, 1024, 2048}." (Section 3).
- MLM sequence length: "processes 128 tokens per sequence." (Table 4 caption).
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
- Padding/resizing requirements: Not specified in the paper.

7. Context Window and Attention Structure
- Maximum sequence length: 2048 tokens for causal LM experiments; 128 tokens for MLM.
  - Evidence: "Specifically, we experiment with sequences of lengths {256, 512, 1024, 2048}." (Section 3)
  - Evidence: "processes 128 tokens per sequence." (Table 4 caption)
- Fixed vs variable length: Variable across experiments, fixed within each setting.
  - Evidence: "The default input sequence length is 1024 tokens." and "Specifically, we experiment with sequences of lengths {256, 512, 1024, 2048}." (Section 3)
- Attention type:
  - Causal/global (autoregressive): "tied to the causal attention mask, which implicitly injects positional information into the self-attention layer in order to preserve the autoregressive nature of language models." (Introduction)
  - Bidirectional/global for MLM: "masked language models (MLM) (Devlin et al., 2019), which use order-invariant attention (since no causal mask is applied)." (Introduction) and "bidirectional transformer encoders (which are used in masked language modeling, e.g. Devlin et al. 2019) do not contain causal attention masks or any other limitation on the attention mechanism;" (Section 5)
- Computational cost mechanisms (windowing/pooling/pruning): Not specified in the paper.

8. Positional Encoding (Critical Section)
- NoPos (none): "we compared the validation set perplexity of models trained from scratch with no explicit positional information (denoted as NoPos) to those trained with the various positional encoding methods discussed in Section 2." (Section 3 Experiment Setup).
- Learned absolute embeddings: "Learned. Embeddings trained to represent absolute positions (Sukhbaatar et al., 2015; Gehring et al., 2017)." (Section 2)
  - Application: "Absolute positions are commonly encoded as vectors (one for each position), which are then added to the input tokens' embeddings and fed to the first layer of the transformer." (Section 2)
- Sinusoidal: "Sinusoidal. Constant vectors computed by a non-parametric function of the input token's absolute position. Sine and cosine functions of different frequencies are used, such that each dimension of the positional encoding corresponds to a sinusoid." (Section 2)
  - Application: same absolute-position mechanism added to input embeddings (see quote above).
- ALiBi: "ALiBi. Attention with LInear BIases (Press et al., 2022) injects information about the relative distances between tokens by adding negative biases to attention scores, which grow linearly with the distance between each pair of tokens." (Section 2)
  - Application: attention bias (supported by "Relative positions are typically encoded as biases (added to attention scores) within the self-attention layers." Section 2).
- PE fixed vs varied: Multiple positional encodings are compared across experiments.
  - Evidence: "We compare the performance of language models trained with no explicit positional information (NoPos language models) to those trained with three different position-aware mechanisms, namely: sinusoidal embeddings (Vaswani et al., 2017), learned embeddings (Gehring et al., 2017), and ALiBi (Press et al., 2022)." (Introduction)

9. Positional Encoding as a Variable
- Core research variable: Yes.
  - Evidence: "We compare the performance of language models trained with no explicit positional information (NoPos language models) to those trained with three different position-aware mechanisms, namely: sinusoidal embeddings (Vaswani et al., 2017), learned embeddings (Gehring et al., 2017), and ALiBi (Press et al., 2022)." (Introduction)
- Multiple positional encodings compared: Yes (learned, sinusoidal, ALiBi).
  - Evidence: same quote as above.
- Claim that PE choice is not critical/secondary: Not explicitly stated; the paper reports competitiveness of NoPos.
  - Evidence: "NoPos models are competitive with position-aware models" (Introduction).

10. Evidence of Constraint Masking
- Model sizes: "small (125M parameters), medium (350M parameters), large (760M parameters) and the XL (1.3B parameters) variants" (Section 3).
- Dataset sizes: "The WikiText-103 corpus (Merity et al., 2017) consists of over 100 million words" and "The Pile (Gao et al., 2020) is an 800GB English text dataset" (Section 3).
- Scaling effect: "Smaller models benefit from fixed, non-parametric positional encodings (Sinusoidal and ALiBi), but these performance gaps diminish as the models scale up." (Table 2 caption)
- Robustness across scale/sequence: "Overall, we show that transformer language modeling without explicit positional encoding is robust to the selection of corpus, model size, and sequence length." (Section 4 Results)
- Primary attribution of gains (model scaling/data scaling/training tricks/architecture): Only model-size scaling is explicitly linked to narrowing gaps (see Table 2 caption); other attributions are not specified.

11. Architectural Workarounds
- Causal masking (autoregressive constraint): "tied to the causal attention mask, which implicitly injects positional information into the self-attention layer in order to preserve the autoregressive nature of language models." (Introduction)
- Windowed attention, hierarchical stages, token pooling/merging, task-specific heads for core LM: Not specified in the paper.

12. Explicit Limitations and Non-Claims
- Scale range limitation: "Our work explores language models in the 125M to 1.3B parameter range." (Section 9 Limitations)
- Uncertainty at larger scales: "the current biggest models are more than one hundred times bigger (in terms of parameters) than our 1.3B parameter models, and so the results in that setting can be unexpected." (Section 9 Limitations)
- Reproducibility/resource limits: "training models at the 1.3B parameter scale is resource-intensive and might hinder reproducibility." (Section 9 Limitations)
- Performance gap: "NoPos is always slightly worse, suggesting that the inductive bias of positional encoding is indeed important." (Section 9 Limitations)
- Non-claim about MLM extension: "this phenomenon does not extend to transformer encoders trained on the MLM objective." (Conclusion)

13. Constraint Profile (Synthesis)
- Domain scope: English text only; datasets include Wikipedia and multi-source web text (single modality).
- Task structure: Primarily next-token language modeling, plus MLM and probe/analysis tasks within text.
- Representation rigidity: Fixed sequence lengths per experiment (512/1024/2048 for causal LM; 128 for MLM) and fixed tokenization schemes.
- Model sharing vs specialization: Separate training per setting; probes reuse frozen LM weights with a separate head.
- Role of positional encoding: Central experimental variable (NoPos vs learned/sinusoidal/ALiBi).

14. Final Classification
Multi-task, single-domain.
Justification: The paper evaluates multiple tasks within language modeling, including causal LM and MLM ("masked language models (MLM) (Devlin et al., 2019), which use order-invariant attention (since no causal mask is applied)."), and all evaluations are on English text datasets (WikiText-103 and The Pile). There is no multi-modal evaluation, so the scope remains single-domain text.

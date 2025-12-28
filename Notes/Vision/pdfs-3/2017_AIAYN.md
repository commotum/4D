## 1. Basic Metadata
Title: Attention Is All You Need.
Authors: Ashish Vaswani; Noam Shazeer; Niki Parmar; Jakob Uszkoreit; Llion Jones; Aidan N. Gomez; Lukasz Kaiser; Illia Polosukhin.
Year: 2017.
Venue: 31st Conference on Neural Information Processing Systems (NIPS 2017); arXiv:1706.03762.
Evidence: "Attention Is All You Need" (p.1). "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA." (p.1). "arXiv:1706.03762v7 [cs.CL] 2 Aug 2023" (p.1).

## 2. One-Sentence Contribution Summary
The paper proposes the Transformer, a sequence transduction architecture based solely on attention that removes recurrence and convolutions to improve parallelization and translation quality.
Evidence: "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely." (p.1, Abstract).

## 3. Tasks Evaluated
Task: Machine translation (English-German).
Task type: Generation (sequence transduction).
Dataset(s): WMT 2014 English-German.
Domain: Natural language text.
Evidence: "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs." (p.7, Section 5.1). "On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4." (p.8, Section 6.1).

Task: Machine translation (English-French).
Task type: Generation (sequence transduction).
Dataset(s): WMT 2014 English-French.
Domain: Natural language text.
Evidence: "For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]." (p.7, Section 5.1). "On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model." (p.8, Section 6.1).

Task: English constituency parsing.
Task type: Other (constituency parsing / structured generation).
Dataset(s): Wall Street Journal (WSJ) portion of the Penn Treebank; high-confidence and BerkleyParser corpora (semi-supervised).
Domain: Natural language text.
Evidence: "To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing." (p.9, Section 6.3). "We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences." (p.9, Section 6.3). "We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]." (p.9, Section 6.3).

## 4. Domain and Modality Scope
Evaluation scope: Multiple tasks within a single modality (text). Evidence: "Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train." (p.1, Abstract). "To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing." (p.9, Section 6.3).
Single domain vs multi-domain: The paper evaluates on multiple tasks within the same modality (text); no multimodal evaluation is reported. Evidence as above.
Domain generalization / cross-domain transfer: Not claimed. The paper claims task generalization to another text task: "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data." (p.1, Abstract).

## 5. Model Sharing Across Tasks
The paper does not explicitly state whether weights are shared across tasks, or whether any pretraining/fine-tuning or joint multi-task training is used; it describes training per dataset/task.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| WMT 2014 English-German translation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs." (p.7, Section 5.1) |
| WMT 2014 English-French translation | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]." (p.7, Section 5.1) |
| English constituency parsing | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We trained a 4-layer transformer with dmodel = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences." (p.9, Section 6.3) |

## 6. Input and Representation Constraints
Variable-length sequences: "mapping one variable-length sequence of symbol representations (x1 , ..., xn ) to another sequence of equal length (z1 , ..., zn )" (p.6, Section 4).
Tokenization/vocabulary constraints: "Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens." (p.7, Section 5.1). "For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]." (p.7, Section 5.1). "We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting." (p.9, Section 6.3).
Embedding dimensionality: "we use learned embeddings to convert the input tokens and output tokens to vectors of dimension dmodel ." (p.5, Section 3.4). "all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512." (p.3, Section 3.1). "We trained a 4-layer transformer with dmodel = 1024" (p.9, Section 6.3).
Batching constraints: "Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens." (p.7, Section 5.1).
Maximum output length constraints (inference): "We set the maximum output length during inference to input length + 50, but terminate early when possible [38]." (p.8, Section 6.1). "During inference, we increased the maximum output length to input length + 300." (p.10, Section 6.3).
Fixed or variable input resolution: Not specified in the paper.
Fixed patch size: Not specified in the paper.
Fixed number of tokens: Not specified in the paper.
Fixed dimensionality (e.g., strictly 2D): Not specified in the paper beyond the embedding dimension dmodel noted above.
Padding or resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
Maximum sequence length: Not specified as an absolute length; only relative output limits are given. Evidence: "We set the maximum output length during inference to input length + 50" (p.8, Section 6.1) and "During inference, we increased the maximum output length to input length + 300." (p.10, Section 6.3).
Fixed or variable sequence length: Variable-length sequences are explicitly discussed. Evidence: "mapping one variable-length sequence of symbol representations (x1 , ..., xn ) to another sequence of equal length (z1 , ..., zn )" (p.6, Section 4).
Attention type: Global self-attention. Evidence: "Each position in the encoder can attend to all positions in the previous layer of the encoder." (p.5, Section 3.2.3). "self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position." (p.5, Section 3.2.3).
Mechanisms to manage computational cost: No such mechanism is introduced in the described model; a restricted variant is mentioned as future work. Evidence: "self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work." (p.7, Section 4).

## 8. Positional Encoding (Critical Section)
Mechanism: Sinusoidal (sine and cosine) positional encodings. Evidence: "In this work, we use sine and cosine functions of different frequencies:" (p.6, Section 3.5).
Where applied: Added to input embeddings at the bottoms of encoder and decoder stacks. Evidence: "we add \"positional encodings\" to the input embeddings at the bottoms of the encoder and decoder stacks." (p.6, Section 3.5).
Application depth: Input only (added to embeddings); no statement that it is applied at every layer. Evidence as above.
Fixed across experiments or modified: The main model uses sinusoidal encodings; a learned embedding variant is tested. Evidence: "We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E))." (p.6, Section 3.5) and "In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model." (p.9, Section 6.2).
Modified per task: Not specified in the paper.

## 9. Positional Encoding as a Variable
Core research variable vs fixed assumption: The paper treats positional encoding as a component with an ablation/variant, not as the central research variable. Evidence: "We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E))." (p.6, Section 3.5).
Multiple positional encodings compared: Yes, sinusoidal vs learned positional embeddings. Evidence: "In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model." (p.9, Section 6.2).
Claim that PE choice is not critical or secondary: The paper reports nearly identical results, implying limited sensitivity. Evidence: "found that the two versions produced nearly identical results" (p.6, Section 3.5).

## 10. Evidence of Constraint Masking (Scale vs Structure)
Model size(s): Base and big configurations are reported in Table 3 (p.9), including dmodel and parameter counts (base: 65 x 10^6 params; big: 213 x 10^6 params). Evidence: Table 3 (p.9) "base" and "big" rows.
Dataset size(s): "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs." (p.7, Section 5.1). "For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences" (p.7, Section 5.1). "about 40K training sentences" (p.9, Section 6.3). "approximately 17M sentences" (p.9, Section 6.3).
Attribution of gains: Scaling model size is explicitly linked to better performance. Evidence: "we observe in rows (C) and (D) that, as expected, bigger models are better" (p.9, Section 6.2). Performance gains are reported for the big model on translation tasks: "On the WMT 2014 English-to-German translation task, the big transformer model ... outperforms the best previously reported models" (p.8, Section 6.1).
Scaling data or training tricks as primary attribution: Not explicitly claimed. The paper reports regularization/training details but does not attribute gains primarily to them. Evidence of training tricks: "We apply dropout [33] to the output of each sub-layer" (p.8, Section 5.4).
Architectural hierarchy as attribution: Not explicitly claimed.

## 11. Architectural Workarounds
Multi-head attention to counteract reduced resolution from attention averaging: "reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2." (p.2, Section 2).
Masked decoder self-attention to preserve autoregression: "We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions." (p.3, Section 3.1).
Residual connections and layer normalization for deep stacks: "We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]." (p.3, Section 3.1).
Positional encodings to inject order information without recurrence/convolution: "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence." (p.6, Section 3.5).
Shared embedding/softmax weights to reduce parameters: "we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation" (p.5, Section 3.4).

## 12. Explicit Limitations and Non-Claims
Future work on restricted attention: "self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work." (p.7, Section 4).
Future work on other modalities and local attention: "We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video." (p.10, Section 7).
Future work on reducing sequential generation: "Making generation less sequential is another research goals of ours." (p.10, Section 7).
Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

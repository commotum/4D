### 1. Basic Metadata
- Title: ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING
- Authors: Jianlin Su; Yu Lu; Shengfeng Pan; Ahmed Murtadha; Bo Wen; Yunfeng Liu
- Year: 2023 (arXiv v5, 8 Nov 2023)
- Venue: arXiv (arXiv:2104.09864v5 [cs.CL])
Evidence: "arXiv:2104.09864v5 [cs.CL] 8 Nov 2023" (front matter)

### 2. One-Sentence Contribution Summary
RoFormer proposes Rotary Position Embedding (RoPE), a rotation-matrix positional encoding integrated into self-attention to incorporate relative position information, and evaluates it across NLP tasks.

### 3. Tasks Evaluated
- Task: Long text classification (benchmark datasets, unspecified) | Task type: Classification | Dataset(s): Not specified in the paper | Domain: Not specified in the paper | Evidence: "Finally, we evaluate the enhanced transformer with rotary position
embedding, also called RoFormer, on various long text classification benchmark datasets." (Abstract)
- Task: Sequence-to-sequence language translation | Task type: Other (machine translation) | Dataset(s): WMT 2014 English-German | Domain: natural language text | Evidence: "We first demonstrate the performance of RoFormer on sequence-to-sequence language translation tasks." (Section 4.1) and "We choose the standard WMT 2014 English-German datasetBojar et al. [2014], which consists of approximately 4.5
million sentence pairs." (Section 4.1.1)
- Task: Pre-training language modeling (MLM) | Task type: Other (masked language modeling) | Dataset(s): BookCorpus; Wikipedia Corpus | Domain: natural language text | Evidence: "The second experiment is to validate the performance of our proposal in terms of learning contextual representations." (Section 4.2) and "We use the BookCorpus Zhu et al. [2015] and the Wikipedia Corpus Foundation [2021] from Huggingface Datasets
library (Apache License 2.0) for pre-training. The corpus is further split into train and validation sets at 8:2 ratio. We
use the masked language-modeling (MLM) loss values of the training process as an evaluation metric." (Section 4.2.1)
- Task: GLUE downstream tasks (MRPC, SST-2, QNLI, STS-B, QQP, MNLI) | Task type: Other (GLUE tasks; task type not specified in paper) | Dataset(s): MRPC, SST-2, QNLI, STS-B, QQP, MNLI | Domain: natural language text | Evidence: "Fine-tuning on GLUE tasks" (Section 4.3) and "We look at several datasets from GLUE, i.e. MRPC Dolan and Brockett [2005], SST-2 Socher et al. [2013], QNLI
Rajpurkar et al. [2016], STS-B Al-Natsheh [2017], QQP Chen et al. [2018b] and MNLI Williams et al. [2018]." (Section 4.3.1)
- Task: Pre-training language modeling with Performer + RoPE | Task type: Other (language modeling) | Dataset(s): Enwik8 | Domain: natural language text (English Wikipedia) | Evidence: "We demonstrate its performance with the pre-training task of language
modeling." (Section 4.4) and "We carry out tests on the Enwik8 dataset Mahoney [2006], which is from English Wikipedia that includes markup,
special characters and text in other languages in addition to English text." (Section 4.4.1)
- Task: Semantic text matching (CAIL2019-SCM) | Task type: Other (semantic text matching) | Dataset(s): CAIL2019-SCM | Domain: Chinese legal case texts | Evidence: "We choose Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM)Xiao et al. [2019] dataset to illustrate
the ability of RoFormer in dealing with long texts, i.e., semantic text matching." (Section 4.5.3) and "The task is to predict whether the pair (A, B) is closer than (A, C) under a predefined
similarity measure." (Section 4.5.3)

### 4. Domain and Modality Scope
- Modality: Text / NLP. Evidence: "various natural language processing (NLP) tasks, including
context representation learning Devlin et al. [2019], machine translation Vaswani et al. [2017], and language modeling
Radford et al. [2019], to name a few." (Introduction)
- Domain scope: Multiple datasets and languages within the text modality (English/German and Chinese). Evidence: "We choose the standard WMT 2014 English-German datasetBojar et al. [2014], which consists of approximately 4.5
million sentence pairs." (Section 4.1.1) and "We carry out tests on the Enwik8 dataset Mahoney [2006], which is from English Wikipedia that includes markup,
special characters and text in other languages in addition to English text." (Section 4.4.1) and "In addition to experiments on English data, we show additional results on Chinese data." (Section 4.5)
- Domain generalization / cross-domain transfer: Not claimed. The only related statement is "We claim that
this is the attribute to the excellent generalizability of the proposed RoPE." (Section 4.5.2)

### 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Long text classification (unspecified) | Not specified in the paper | Not specified in the paper | Not specified in the paper | Not specified in the paper. |
| Machine translation | Not specified in the paper | Not specified in the paper | Not specified in the paper | Not specified in the paper. |
| Pre-training language modeling (BERT/RoFormer) | Not specified in the paper | Not specified in the paper | Not specified in the paper | "The second experiment is to validate the performance of our proposal in terms of learning contextual representations." (Section 4.2) |
| GLUE tasks | Pre-trained weights used | Yes (per task) | Not specified in the paper | "Consistent with the previous experiments, we fine-tune the weights of our pre-trained RoFormer across various GLUE" (Section 4.3) and "We use Huggingface Transformers library (Apache License 2.0)Wolf et al. [2020] to fine-tune each of the aforementioned" (Section 4.3.2) |
| Performer + RoPE language modeling | Not specified in the paper | Not specified in the paper | Not specified in the paper | Not specified in the paper. |
| CAIL2019-SCM | Pre-trained weights used | Not specified in the paper | Not specified in the paper | "We apply the pre-trained RoFormer model to CAIL2019-SCM with different input lengths." (Section 4.5.4) |

### 6. Input and Representation Constraints
- Embedding dimensionality: RoPE assumes an even embedding dimension. Evidence: "In order to generalize our results in 2D to any xi ∈ Rd where d is even, we divide the d-dimension space into d/2
sub-spaces and combine them in the merit of the linearity of the inner product, turning f{q,k} into:" (Section 3.2.2)
- Tokenization details are mentioned only via a table, not as explicit constraints in the prose. Evidence: "we tabulate their
tokenization level and position embedding information in Table (3)." (Section 4.5.1)
- Fixed/variable input resolution, patch size, fixed number of tokens, padding/resizing requirements: Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length in experiments: "We train both BERT and RoFormer with batch size 64 and maximum sequence length of 512 for 100k steps." (Section 4.2.2) and "We use Huggingface Transformers library (Apache License 2.0)Wolf et al. [2020] to fine-tune each of the aforementioned
downstream tasks for 3 epochs, with a maximum sequence length of 512, batch size of 32 and learning rates 2,3,4,5e-5." (Section 4.3.2) and "learning rate 1e-4, batch size
128 and a fixed maximum sequence length of 1024, etc." (Section 4.4.1) and "RoFormer on long texts, we conduct experiments on long documents whose length exceeds 512 characters." (Section 4.5) and "However, when increasing the maximum input text length to 1024, RoFormer outperforms WoBERT by an absolute
improvement of 1.5%." (Section 4.5.4)
- Fixed vs variable sequence length: RoPE is described as having "the flexibility of sequence length" (Abstract), but experiments use fixed maximum lengths (quotes above).
- Attention type: Self-attention and linear attention are both used. Evidence: "PLMs utilize the self-attention mechanism
to semantically capture the contextual representation of a given corpus." (Introduction) and "Performer Choromanski et al. [2020] introduces an alternative attention mechanism, linear attention, which is designed
to avoid quadratic computation cost that scales with input sequence length." (Section 4.4)
- Computational cost management: Linear attention is explicitly used to avoid quadratic cost. Evidence: "linear attention, which is designed
to avoid quadratic computation cost that scales with input sequence length." (Section 4.4) and "the proposed
RoPE can be easily implemented in the PerFormer model to realize the relative position encoding while keeping its
linearly scaled complexity in self-attention." (Section 4.4)

### 8. Positional Encoding (Critical Section)
- Mechanism: RoPE with rotation matrices and explicit relative position in self-attention. Evidence: "Specifically, the
proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates
the explicit relative position dependency in self-attention formulation." (Abstract)
- Where applied: In the self-attention block. Evidence: "For RoFormer, we replace the sinusoidal position encoding in the self-attention block of the baseline model with our
proposed RoPE and realizes self-attention according to Equation (16)." (Section 4.2.2)
- Fixed vs modified/ablated: RoPE replaces sinusoidal PE in BERT, and is compared with and without RoPE in Performer. Evidence: "we replace the original sinusoidal position encoding of BERT with our RoPE during the pre-training step." (Section 4.2) and "we report the loss
curves of the pre-training process with and without RoPE under the same settings" (Section 4.4.1)

### 9. Positional Encoding as a Variable
- Core research variable: Yes. Evidence: "Then, we propose a novel method named Rotary
Position Embedding(RoPE) to effectively leverage the positional information." (Abstract)
- Multiple positional encodings compared: Yes (sinusoidal vs RoPE; with/without RoPE). Evidence: "we replace the original sinusoidal position encoding of BERT with our RoPE during the pre-training step." (Section 4.2) and "with and without RoPE" (Section 4.4.1)
- Claim that PE choice is not critical: Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model size(s): "BERT Devlin et al. [2019] is adopted as our baseline model. Note that we use bert-base-uncased in our experiments." (Section 4.2.1) and "We incorporate RoPE into the 12 layer
char-based PerFormer with 768 dimensions and 12 heads2 ." (Section 4.4.1)
- Dataset size(s): "approximately 4.5
million sentence pairs." (Section 4.1.1) and "We pre-train RoFormer on approximately 34GB of data collected from Chinese Wikipedia, news and forums." (Section 4.5.2)
- Training tricks / optimization: "Label smoothing with 0.1 is also adopted." (Section 4.1.2)
- Attribution of gains: Improvements are attributed to RoPE rather than scaling, with no explicit claim that scaling data/model size is the primary driver. Evidence: "Table 1: The proposed RoFormer gives better BLEU scores compared to its baseline alternative Vaswani et al. [2017]
on the WMT 2014 English-to-German translation taskBojar et al. [2014]." (Section 4.1) and "As shown on the right plot of Figure (3), substituting RoPE into Performer leads to rapid convergence and lower loss" (Section 4.4.2)

### 11. Architectural Workarounds
- Linear attention to avoid quadratic cost: "Performer Choromanski et al. [2020] introduces an alternative attention mechanism, linear attention, which is designed
to avoid quadratic computation cost that scales with input sequence length." (Section 4.4)
- Keeping linear complexity with RoPE in Performer: "the proposed
RoPE can be easily implemented in the PerFormer model to realize the relative position encoding while keeping its
linearly scaled complexity in self-attention." (Section 4.4)

### 12. Explicit Limitations and Non-Claims
- Stated limitations:
"Although we provide theoretical groundings as well as promising experimental justifications, our method is limited by
following facts:
• Despite the fact that we mathematically format the relative position relations as rotations under 2D sub-spaces,
there lacks of thorough explanations on why it converges faster than baseline models that incorporates other
position encoding strategies.
• Although we have proved that our model has favourable property of long-term decay for intern-token products,
Section (3.3), which is similar to the existing position encoding mechanisms, our model shows superior
performance on long texts than peer models, we have not come up with a faithful explanation.
Our proposed RoFormer is built upon the Transformer-based infrastructure, which requires hardware resources for
pre-training purpose." (Limitations of the work)
- Explicit non-claims (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified in the paper.

### 13. Constraint Profile (Synthesis)

Constraint Profile:
- Domain scope: NLP/text only, across multiple datasets and languages (English/German and Chinese).
- Task structure: Multiple NLP tasks (translation, language modeling, GLUE tasks, semantic text matching), no multi-modal evaluation.
- Representation rigidity: Experiments use fixed maximum sequence lengths (512/1024), and RoPE assumes even embedding dimension.
- Model sharing vs specialization: Pre-trained weights are fine-tuned per GLUE task; other task-sharing details are not specified.
- Role of positional encoding: Central research variable with explicit replacements and with/without comparisons.

### 14. Final Classification
Classification: Multi-task, single-domain.

Justification: The paper evaluates multiple NLP tasks, including "sequence-to-sequence language translation tasks" (Section 4.1), "the pre-training task of language
modeling" (Section 4.4), "Fine-tuning on GLUE tasks" (Section 4.3), and "semantic text matching" (Section 4.5.3). All evaluations are within the text modality, spanning English/German and Chinese datasets, with no evidence of multiple modalities or cross-domain transfer claims.

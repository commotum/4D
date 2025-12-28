### 1. Basic Metadata
- Title: "Nested Learning: The Illusion of Deep Learning Architecture" (page 1).
- Authors: "Ali Behrouz , Meisam Razaviyayn, Peilin Zhong, and Vahab Mirrokni" (page 1).
- Year: 2025. Evidence: "A version of this work is published at Neural Information Processing Systems (NeurIPS) 2025." (page 1).
- Venue: Neural Information Processing Systems (NeurIPS). Evidence: "A version of this work is published at Neural Information Processing Systems (NeurIPS) 2025." (page 1).

### 2. One-Sentence Contribution Summary
The paper proposes Nested Learning (NL), a multi-level optimization paradigm that reframes architectures and optimizers as nested associative memories and introduces Hope (self-modifying Titans + Continuum Memory System) to improve continual learning and long-context reasoning.

### 3. Tasks Evaluated
- Task: Class-incremental intent classification (CLINC)
  - Task type: Classification
  - Dataset(s): CLINC (Larson et al. 2019)
  - Domain: Natural language dialog queries (multi-domain)
  - Evidence: "Class Incremental Learning. First, we focus on class-incremental learning tasks on three datasets: • CLINC (Larson et al. 2019): CLINC is a multi-domain intent classification benchmark designed for task-oriented dialog systems, with a special focus on detecting out-of-scope (OOS) queries." (page 34)

- Task: Class-incremental intent classification (Banking)
  - Task type: Classification
  - Dataset(s): Banking (Casanueva et al. 2020)
  - Domain: Natural language customer service queries (single domain: banking)
  - Evidence: "Banking (Casanueva et al. 2020): Banking dataset is a single-domain intent classification benchmark focused on fine-grained customer service queries in the banking domain." (page 34)

- Task: Class-incremental topic classification (DBpedia)
  - Task type: Classification
  - Dataset(s): DBpedia (Auer et al. 2007)
  - Domain: Natural language Wikipedia abstracts
  - Evidence: "DBpedia (Auer et al. 2007): DBpedia is a text classification benchmark from Wikipedia where article abstracts are expected to be categorized into ontology topic classes." (page 34)

- Task: Long-context clinical question answering (LongHealth)
  - Task type: Reasoning / relational (multiple-choice QA)
  - Dataset(s): LongHealth (Adams et al. 2025)
  - Domain: Medical/clinical text
  - Evidence: "LongHealth (Adams et al. 2025): This is a benchmark for long-context clinical question answering with multiple-choice QA tasks based on extensive fictional patient records." (page 34)

- Task: Information-seeking question answering (QASPER)
  - Task type: Reasoning / relational (QA)
  - Dataset(s): QASPER (Dasigi et al. 2021)
  - Domain: NLP research papers (full-text)
  - Evidence: "QASPER (Dasigi et al. 2021): This benchmark is an information-seeking QA dataset centered on full-length NLP research papers." (page 34)

- Task: Multi-key needle-in-a-haystack (MK-NIAH from RULER)
  - Task type: Other (long-context retrieval)
  - Dataset(s): RULER / MK-NIAH (Hsieh et al. 2024)
  - Domain: Long-context natural language text
  - Evidence: "MK-NIAH (Hsieh et al. 2024): We use the task of multiple keys in needle-in-haystack from RULER (Hsieh et al. 2024). This setup requires models to not only locate but also extract multiple pieces of information distributed throughout a long text." (page 35)

- Task: Needle-in-a-haystack (NIAH) variants: single needle (pass-key, number, uuid), multi-key, multi-query, multi-value
  - Task type: Other (long-context retrieval)
  - Dataset(s): RULER / NIAH (Hsieh et al. 2024)
  - Domain: Long-context natural language text
  - Evidence: "Needle-in-a-Haystack (NIAH) Tasks. In the first part, we focus on the needle-in-a-haystack with different setups of: (1) single needle but different types (i.e., pass-key, number, and uuid), (2) multi-key, (3) multi-query, and (4) multi-value, all follows Hsieh et al. (2024)." (page 36)

- Task: BABILong long-context understanding
  - Task type: Reasoning / relational (long-context reasoning-in-a-haystack)
  - Dataset(s): BABILong (Kuratov et al. 2024)
  - Domain: Long-context natural language text
  - Evidence: "Long context understanding tasks, including needle-in-a-haystack (Hsieh et al. 2024) and BABILong (Kuratov et al. 2024) benchmarks." (page 5)

- Task: Continual Translation of a Novel Language (CTNL)
  - Task type: Generation (translation)
  - Dataset(s): MTOB (Tanzer et al. 2024) and Manchu (Pei et al. 2025)
  - Domain: Natural language text (translation)
  - Evidence: "we design a new continual learning task that the LLM learns two new languages in context and is expected to translate phrases to English." (page 36)

- Task: Language modeling benchmarks
  - Task type: Generation
  - Dataset(s): Wikitext; LMB (LAMBADA)
  - Domain: Natural language text
  - Evidence: "Datasets: We evaluate Hope and baselines on Wikitext (Merity et al. 2017), LMB (Paperno et al. 2016)..." (page 37)

- Task: Common-sense reasoning benchmarks
  - Task type: Reasoning / relational (multiple-choice or classification)
  - Dataset(s): PIQA, HellaSwag, WinoGrande, ARC-easy (ARC-e), ARC-challenge (ARC-c), SIQA, BoolQ
  - Domain: Natural language text
  - Evidence: "Datasets: We evaluate Hope and baselines on Wikitext (Merity et al. 2017), LMB (Paperno et al. 2016), PIQA (Bisk et al. 2020), HellaSwag (Zellers et al. 2019), WinoGrande (Sakaguchi et al. 2021), ARC-easy (ARC-e) and ARC-challenge (ARC-c) (Clark et al. 2018), SIQA (Sap et al. 2019), and BoolQ (Clark et al. 2019) benchmarks." (page 37)

- Task: In-context recall (reading comprehension / QA)
  - Task type: Reasoning / relational (QA)
  - Dataset(s): SWDE, NQ, DROP, FDA, SQuAD, TQA
  - Domain: Natural language text
  - Evidence: "we follow Arora et al. (2024) and perform experiments on SWDE (Lockard et al. 2019), NQ (Kwiatkowski et al. 2019), DROP (Dua et al. 2019), FDA (Arora et al. 2023), SQUAD (Rajpurkar et al. 2016), and TQA (Kembhavi et al. 2017)" (page 38)

- Task: MAD synthetic benchmark (recall/memorization/compression/copying)
  - Task type: Other (synthetic recall/memorization/compression/copying)
  - Dataset(s): MAD (Poli et al. 2024)
  - Domain: Synthetic sequences
  - Evidence: "MAD benchmark (Poli et al. 2024), which is a synthetic benchmark, evaluating the performance of models in recall, memorization, compression, and copying tasks." (page 38)

- Task: Formal language recognition
  - Task type: Classification (formal language recognition)
  - Dataset(s): Formal language recognition benchmark (Irie et al. 2023)
  - Domain: Synthetic formal languages
  - Evidence: "In this section, we focus on formal language recognition tasks and follow the construction of the benchmark by Irie et al. (2023)." (page 38)

- Task: Image classification (ViT on ImageNet-21K)
  - Task type: Classification
  - Dataset(s): ImageNet-21K
  - Domain: Natural images
  - Evidence: "ImageNet. In this experiment, we focus on ViT (Dosovitskiy et al. 2021) architecture for vision tasks and pre-train it on ImageNet-21K, which consists of 11M images corresponding to 10,450 classes." (page 39)

### 4. Domain and Modality Scope
- Multiple domains within the same modality (text): "CLINC is a multi-domain intent classification benchmark... spanning 10 broad domains" and "Banking dataset is a single-domain intent classification benchmark focused on fine-grained customer service queries in the banking domain." (page 34)
- Multiple modalities: text tasks plus vision tasks, e.g., "pre-train it on ImageNet-21K, which consists of 11M images" (page 39).
- Domain generalization / cross-domain transfer: Not claimed in the paper.

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| CLINC class-incremental intent classification | Not specified in the paper. | Yes (continual pre-training described, but task-specific fine-tuning not specified). | Not specified in the paper. | "As the backbone of our Hope models we use Llama3-8B and Llama-3B... followed by continual pre-training with 15B tokens." (page 34) |
| Banking class-incremental intent classification | Not specified in the paper. | Yes (continual pre-training described, but task-specific fine-tuning not specified). | Not specified in the paper. | "As the backbone of our Hope models we use Llama3-8B and Llama-3B... followed by continual pre-training with 15B tokens." (page 34) |
| DBpedia class-incremental topic classification | Not specified in the paper. | Yes (continual pre-training described, but task-specific fine-tuning not specified). | Not specified in the paper. | "As the backbone of our Hope models we use Llama3-8B and Llama-3B... followed by continual pre-training with 15B tokens." (page 34) |
| LongHealth long-context clinical QA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "we perform in-context question/answering... LongHealth (Adams et al. 2025)" (page 34) |
| QASPER information-seeking QA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "we perform in-context question/answering... QASPER (Dasigi et al. 2021)" (page 34) |
| MK-NIAH multi-key needle-in-a-haystack | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We use the task of multiple keys in needle-in-haystack from RULER (Hsieh et al. 2024)." (page 35) |
| NIAH variants (single needle / multi-key / multi-query / multi-value) | Not specified in the paper. | No (trained from scratch). | Not specified in the paper. | "we use about 50B tokens... to train all the models from scratch." (page 36) |
| BABILong long-context understanding | Not specified in the paper. | Yes for small models (fine-tuned); other models not specified. | Not specified in the paper. | "We follow the original setup of the benchmark and fine-tune the small models with the same process as Kuratov et al. (2024)." (page 36) |
| CTNL translation (continual translation of novel language) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "we design a new continual learning task that the LLM learns two new languages in context and is expected to translate phrases to English." (page 36) |
| Language modeling (Wikitext, LMB) | Not specified in the paper. | No (trained from scratch). | Not specified in the paper. | "We train models with about 760M and 1.3B parameters, trained with 30B and 100B tokens, respectively." (page 37) |
| Common-sense reasoning (PIQA, HellaSwag, WinoGrande, ARC-e/ARC-c, SIQA, BoolQ) | Not specified in the paper. | No (trained from scratch). | Not specified in the paper. | "We train models with about 760M and 1.3B parameters, trained with 30B and 100B tokens, respectively." (page 37) |
| In-context recall (SWDE, NQ, DROP, FDA, SQuAD, TQA) | Not specified in the paper. | Not specified in the paper (uses same setup as Section 9.3). | Not specified in the paper. | "We use the same set of baselines and experimental setup as the above previous section." (page 38) |
| MAD synthetic benchmark | Not specified in the paper. | Not specified in the paper (uses same setup as Section 9.3). | Not specified in the paper. | "We use the same set of baselines and experimental setup as the above previous section." (page 38) |
| Formal language recognition | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "we focus on formal language recognition tasks and follow the construction of the benchmark by Irie et al. (2023)." (page 38) |
| ImageNet-21K ViT classification | Not specified in the paper. | No (pre-training on ImageNet-21K). | Not specified in the paper. | "pre-train it on ImageNet-21K" (page 39) |

### 6. Input and Representation Constraints
- Fixed patch size for ViT: "We use a patch size of 16" (page 39).
- Fixed vocabulary size for language models: "with a vocabulary size of 32K" (page 36).
- Local convolution window size: "use local convolutions with window size of 4." (page 33).
- Chunked processing for sequence inputs: "given an input sequence {𝒙𝑡 }𝑡𝐿=1 and chunk size 1 ≤ 𝐶 ≤ 𝐿, we split the sequence into ⌈ 𝐶𝐿 ⌉ chunks" (page 32).
- Token dimensionality stated symbolically: "let 𝒙𝑡 ∈ R𝑑 for 𝑡 = 1, . . . , 𝐿 be the input" (page 33).
- Padding or resizing requirements: Not specified in the paper.
- Fixed vs variable input resolution (vision): Not specified beyond the fixed patch size.

### 7. Context Window and Attention Structure
- Maximum sequence length reported: "Hope maintains its good performance even for 10M context length" (page 37).
- Fixed vs variable sequence length: Not specified in the paper; multiple context lengths are used across benchmarks.
- Attention type(s): "replace the self-modifying Titans with softmax global attention" (page 33). Sliding-window attention is noted as an option: "limit the optimization process to the past 𝑐 tokens... which is equivalent to the sliding window attention (SWA)." (page 22).
- Mechanisms to manage computational cost:
  - CMS update scheduling: "updates are restricted to blocks approaching their scheduled update time (based on their frequency)." (page 28)
  - Chunk-wise parallelization: "we split the sequence into ⌈ 𝐶𝐿 ⌉ chunks... This allows for generating all the elements for the entire chunk in parallel" (page 32).

### 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Not specified in the paper.
- Evidence of only positional mention: "the projection of each token is a function of the token itself and its position" (page 30).
- Where applied / fixed vs modified / ablations: Not specified in the paper.

### 9. Positional Encoding as a Variable
- Treated as a core research variable? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- Claims that PE choice is not critical? Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model sizes:
  - "As the backbone of our Hope models we use Llama3-8B and Llama-3B" (page 34).
  - "We train models with about 760M and 1.3B parameters" (page 37).
  - "24M and 86M models" for ViT (page 39).
  - "we use two scales of 140M and 1.3B" (page 40).
- Dataset sizes:
  - "continual pre-training with 15B tokens" (page 34).
  - "use about 50B tokens" (page 36).
  - "trained with 30B and 100B tokens" (page 37).
  - "ImageNet-21K, which consists of 11M images corresponding to 10,450 classes" (page 39).
- Scaling vs architecture attribution:
  - Scaling acknowledged: "scaling the model size to enhance its expressivity" (page 2).
  - Architectural hierarchy emphasized for gains: "Comparing Hope with ICL, the main difference comes from Hope’s multiple levels of in-context learning... indicating the effectiveness of CMS’s design" (page 34).
  - Long-context gains attributed to CMS: "Hope maintains its good performance even for 10M context length, mainly due to its CMS design." (page 37).

### 11. Architectural Workarounds
- Continuum Memory System (CMS) for multi-timescale memory: "Continuum Memory System (CMS) is formalized as a chain of MLP blocks... each of which associated with a chunk size" (page 27).
- Chunk-wise training / parallelization: "we split the sequence into ⌈ 𝐶𝐿 ⌉ chunks... This allows for generating all the elements for the entire chunk in parallel" (page 32).
- Sliding window attention: "limit the optimization process to the past 𝑐 tokens... equivalent to the sliding window attention (SWA)." (page 22).
- Meta-learned initialization for fast adaptation: "this initial state can be meta-learned to adapt fast to a context." (page 12).
- Self-modifying learning module: "we present self-modifying deep associative memory where the models generate their own values" (page 31).
- Local convolutions for mixing: "use local convolutions with window size of 4." (page 33).
- Gating as a persistent memory proxy: "the gating of linear attention acts as a persistent memory and the initialization of the memory module." (page 23).
- Hope architecture (combining Titans + CMS): "we present Hope architecture: A neural learning module that incorporates self-modifying Titans followed by Continuum Memory System." (page 33).

### 12. Explicit Limitations and Non-Claims
- Catastrophic forgetting not solved: "catastrophic forgetting is not \"solved\" in general." (page 40).
- M3 optimizer overhead: "The M3 optimizer per se, however, might suffer from computational overhead and so face challenges when scaling to larger networks" (page 29).
- Excluded comparison due to cost differences: "we exclude their comparison with Hope mainly due to the fact that Hope has higher memory usage and there are fundamental differences in their computational costs" (page 35).
- Performance drop without fine-tuning: "the performance of all small models, including Hope, can drop significantly when used without fine-tuning." (page 37)

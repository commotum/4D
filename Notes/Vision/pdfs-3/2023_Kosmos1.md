## 1. Basic Metadata
Title: Language Is Not All You Need: Aligning Perception with Language Models. Evidence: "Language Is Not All You Need: Aligning Perception with Language Models" (page 1).
Authors: Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei.
Year: 2023. Evidence: "arXiv:2302.14045v2 [cs.CL] 1 Mar 2023" (page 1).
Venue: arXiv. Evidence: "arXiv:2302.14045v2 [cs.CL] 1 Mar 2023" (page 1).

## 2. One-Sentence Contribution Summary
The paper introduces KOSMOS-1, a multimodal large language model that aligns perception with language models to enable zero-shot and few-shot multimodal tasks without finetuning. Evidence: "we introduce KOSMOS -1, a Multimodal Large Language Model (MLLM) that can perceive general modalities, learn in context (i.e., few-shot), and follow instructions (i.e., zero-shot)." (Abstract, page 2) and "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract, page 2).

## 3. Tasks Evaluated
### 3.1 Language Tasks (text inputs)
| Task | Task type | Dataset(s) | Domain | Evidence (quote + section/page) |
| --- | --- | --- | --- | --- |
| StoryCloze | Reasoning / relational (cloze/completion) | StoryCloze [MRL+ 17] | Text (language) | "cloze and completion tasks (i.e, StoryCloze, HellaSwag)" (Section 4.8.1) and "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| HellaSwag | Reasoning / relational (cloze/completion) | HellaSwag [ZHB+ 19] | Text (language) | "cloze and completion tasks (i.e, StoryCloze, HellaSwag)" (Section 4.8.1) and "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| Winograd | Reasoning / relational (Winograd-style) | Winograd [LDM12b] | Text (language) | "Winograd-style tasks (i.e, Winograd, Winogrande)" (Section 4.8.1) and "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| Winogrande | Reasoning / relational (Winograd-style) | Winogrande [SBBC20] | Text (language) | "Winograd-style tasks (i.e, Winograd, Winogrande)" (Section 4.8.1) and "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| PIQA | Reasoning / relational (commonsense reasoning) | PIQA [BZB+ 20] | Text (language) | "commonsense reasoning (i.e, PIQA)" (Section 4.8.1) and "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| BoolQ | Other (question answering) | BoolQ [CLC+ 19] | Text (language) | "BoolQ [CLC+ 19]" and "Question answering" (Table 1, page 3) plus "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| CB | Reasoning / relational (textual entailment) | CB [dMST19] | Text (language) | "CB [dMST19]" and "Textual entailment" (Table 1, page 3) plus "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| COPA | Reasoning / relational (causal reasoning) | COPA [RBG11] | Text (language) | "COPA [RBG11]" and "Causal reasoning" (Table 1, page 3) plus "Text inputs are directly fed into the models as in vanilla language models." (Section 4.8) |
| Rendered SST-2 | Classification (OCR-free sentiment classification) | Rendered SST-2 [RKH+ 21] | Rendered text images | "OCR-free language understanding is a task that focuses on understanding text and images without relying on Optical Character Recognition (OCR)." and "sentences from the Stanford Sentiment Treebank [SPW+ 13] dataset are rendered as images. The model is asked to predict the sentiment of the text within the images." (Section 4.3) plus "Rendered SST-2 [RKH+ 21]" and "OCR-free sentiment classification" (Table 1, page 3) |
| HatefulMemes | Classification (OCR-free meme classification) | HatefulMemes [KFM+ 20] | Image memes with text | "We evaluate OCR-free language understanding on the Rendered SST-2 [RKH+ 21] test set and HatefulMemes [KFM+ 20] validation set." (Section 4.3.1) plus "HatefulMemes [KFM+ 20]" and "OCR-free meme classification" (Table 1, page 3) |

### 3.2 Cross-Modal Transfer Tasks (language-only evaluation of visual commonsense)
| Task | Task type | Dataset(s) | Domain | Evidence (quote + section/page) |
| --- | --- | --- | --- | --- |
| RelativeSize | Reasoning / relational (object size) | R ELATIVE S IZE [BHCF16] | Text (language-only) | "We compare KOSMOS -1 and the LLM baseline on three object commonsense reasoning datasets, R ELATIVE S IZE [BHCF16], M EMORY C OLOR [NHJ21] and C OL ORT ERMS [BBBT12] datasets." (Section 4.9.2) and "Is {Item1} larger than {Item2}? {Answer}" (Table 15, page 16) and "We use only text as our input and do not include any images." (Section 4.9.2) |
| MemoryColor | Classification (object color) | M EMORY C OLOR [NHJ21] | Text (language-only) | "We compare KOSMOS -1 and the LLM baseline on three object commonsense reasoning datasets, R ELATIVE S IZE [BHCF16], M EMORY C OLOR [NHJ21] and C OL ORT ERMS [BBBT12] datasets." (Section 4.9.2) and "The color of {Object} is? {Answer}" (Table 15, page 16) and "We use only text as our input and do not include any images." (Section 4.9.2) |
| ColorTerms | Classification (object color) | C OL ORT ERMS [BBBT12] | Text (language-only) | "We compare KOSMOS -1 and the LLM baseline on three object commonsense reasoning datasets, R ELATIVE S IZE [BHCF16], M EMORY C OLOR [NHJ21] and C OL ORT ERMS [BBBT12] datasets." (Section 4.9.2) and "The color of {Object} is? {Answer}" (Table 15, page 16) and "We use only text as our input and do not include any images." (Section 4.9.2) |

### 3.3 Nonverbal Reasoning Task
| Task | Task type | Dataset(s) | Domain | Evidence (quote + section/page) |
| --- | --- | --- | --- | --- |
| Raven IQ Test (Raven's Progressive Matrices) | Reasoning / relational (nonverbal reasoning) | Raven IQ test / Raven's Progressive Matrices | Image matrices | "Raven’s Progressive Matrices [CJS90, JR03] is one of the most common tests to evaluate nonverbal reasoning." (Section 4.2) and "we construct a dataset of the Raven IQ test. It consists of 50 examples collected from different websites" (Section 4.2.1) and "Given eight images presented in a 3 × 3 matrix, the task is to identify the following element from six similar candidates." (Section 4.2) |

### 3.4 Perception-Language Tasks
| Task | Task type | Dataset(s) | Domain | Evidence (quote + section/page) |
| --- | --- | --- | --- | --- |
| Image captioning (COCO) | Generation (image captioning) | MS COCO Caption [LMB+ 14] | Images | "Image captioning involves generating a natural language description of an image" (Section 4.1) and "We evaluate the caption generation on MS COCO Caption [LMB+ 14], and Flickr30k [YLHH14]." (Section 4.1.1) |
| Image captioning (Flickr30k) | Generation (image captioning) | Flickr30k [YLHH14] | Images | "Image captioning involves generating a natural language description of an image" (Section 4.1) and "We evaluate the caption generation on MS COCO Caption [LMB+ 14], and Flickr30k [YLHH14]." (Section 4.1.1) |
| Visual question answering (VQAv2) | Other (visual question answering) | VQAv2 [GKSS+ 17] | Images | "visual question answering aims to answer a natural language question with respect to an image." (Section 4.1) and "For visual question-answering tasks, we evaluate zero-shot and few-shot results on test-dev set of VQAv2 [GKSS+ 17] and test-dev set of VizWiz [GLS+ 18], respectively." (Section 4.1.1) |
| Visual question answering (VizWiz) | Other (visual question answering) | VizWiz [GLS+ 18] | Images | "visual question answering aims to answer a natural language question with respect to an image." (Section 4.1) and "For visual question-answering tasks, we evaluate zero-shot and few-shot results on test-dev set of VQAv2 [GKSS+ 17] and test-dev set of VizWiz [GLS+ 18], respectively." (Section 4.1.1) |
| Web page question answering | Other (question answering) | WebSRC [CZC+ 21] | Web pages | "Web page question answering aims at finding answers to questions from web pages." (Section 4.4) and "We compare the performance on the Web-based Structural Reading Comprehension (WebSRC) dataset [CZC+ 21]." (Section 4.4.1) |

### 3.5 Vision Tasks
| Task | Task type | Dataset(s) | Domain | Evidence (quote + section/page) |
| --- | --- | --- | --- | --- |
| Zero-shot image classification | Classification | ImageNet [DDS+ 09] | Images | "We report the zero-shot image classification performance on ImageNet [DDS+ 09]. Image classification comprehends an entire image as a whole and aims to assign a label to the image." (Section 4.6) |
| Zero-shot image classification with descriptions | Classification (with textual descriptions) | CUB [WBW+ 11] (bird classification with descriptions) | Images + category descriptions | "Following CUB [WBW+ 11], we construct a bird classification dataset that contains images and natural-language descriptions of categories." and "Our goal is to classify images given the categories’ descriptions." (Section 4.7) |

## 4. Domain and Modality Scope
Is evaluation performed on a single domain? No; multiple domains are evaluated (text-only language tasks, images, web pages, rendered text images). Evidence: "As shown in Table 1, the KOSMOS -1 model natively supports language, perception-language, and vision tasks." (Section 1, page 4) and "Web page question answering aims at finding answers to questions from web pages." (Section 4.4) and "sentences from the Stanford Sentiment Treebank [SPW+ 13] dataset are rendered as images." (Section 4.3).
Multiple domains within the same modality? Yes; within vision, tasks span images, web pages, and rendered text images (see quotes above).
Multiple modalities? Yes; the evaluation spans language and vision. Evidence: "As shown in Table 1, the KOSMOS -1 model natively supports language, perception-language, and vision tasks." (Section 1, page 4).
Domain generalization or cross-domain transfer claimed? Cross-modal transfer is claimed; domain generalization is not claimed. Evidence: "We also show that MLLMs can benefit from cross-modal transfer, i.e., transfer knowledge from language to multimodal, and from multimodal to language." (Abstract, page 2). Domain generalization or cross-domain transfer across domains is not specified in the paper.

## 5. Model Sharing Across Tasks
Evidence for shared weights and no per-task finetuning: "We evaluate various settings, including zero-shot, few-shot, and multimodal chain-of-thought prompting, on a wide range of tasks without any gradient updates or finetuning." (Abstract, page 2) and "Once the models are trained, we can directly evaluate the models in zero-shot and few-shot settings on both language tasks and multimodal tasks." (Section 2).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| StoryCloze | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| HellaSwag | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| Winograd | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| Winogrande | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| PIQA | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| BoolQ | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| CB | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| COPA | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| Rendered SST-2 | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| HatefulMemes | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| RelativeSize | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| MemoryColor | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| ColorTerms | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| Raven IQ Test | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| COCO captioning | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| Flickr30k captioning | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| VQAv2 | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| VizWiz | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| WebSRC | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| ImageNet | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |
| CUB (descriptions) | Yes | No | Not specified in the paper. | "without any gradient updates or finetuning" (Abstract, page 2) |

Note: The paper also uses a language-only instruction-tuned variant for ablations, but this is not per-task finetuning; it is described as "language-only instruction tuning" applied broadly (Section 3.3).

## 6. Input and Representation Constraints
- Input sequence format: "we flatten input as a sequence decorated with special tokens." (Section 2.1)
- Image delimiters: "The special tokens <image> and </image> indicate the beginning and end of encoded image embeddings." (Section 2.1)
- Multimodal embedding: "An embedding module is used to encode both text tokens and other input modalities into vectors." (Section 2.1)
- Vision encoder: "we employ a vision encoder as the embedding module for input images." (Section 2.1)
- Image token reduction: "Resampler [ADL+ 22] is used as an attentive pooling mechanism to reduce the number of image embeddings." (Section 2.1)
- Fixed image resolution (training): "The images are preprocessed into 224×224 resolution during training." (Section 3.2)
- Fixed image resolution (evaluation): "The image resolution is 224×224." (Section 4.1.1)
- Vision backbone dimensionality: "the image representation is obtained from a pretrained CLIP ViT-L/14 model with 1,024 feature dimensions." (Section 3.2)
- Max sequence length: "Max length 2,048" (Table 17, page 22) and "Max length of text corpora 2,048" (Table 18, page 22).
- Text tokenization: "We use SentencePiece [KR18] to tokenize the text." (Section 3.2)
- Fixed patch size: Not specified in the paper (the model name "CLIP ViT-L/14" is given, but patch size is not explicitly described).
- Fixed number of tokens: Not specified in the paper (only a maximum length is stated).
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
- Padding/resizing requirements: Explicit resizing to 224x224 is stated (quotes above); padding requirements are not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: "Max length 2,048" (Table 17, page 22).
- Fixed vs variable length: Not explicitly stated; the positional encoding is described as length-extrapolatable: "The method can better generalize to different lengths, i.e., training on short while testing on longer sequences." (Section 2.2)
- Attention type: The backbone is a causal Transformer: "the backbone of KOSMOS -1 is a Transformer-based causal language model." and "The left-to-right causal model processes the sequence in an auto-regressive manner" with "The causal masking is used to mask out future information." (Section 2.2)
- Windowed/hierarchical/sparse attention: Not specified in the paper.
- Cost management: "Resampler [ADL+ 22] is used as an attentive pooling mechanism to reduce the number of image embeddings." (Section 2.1)

## 8. Positional Encoding (Critical Section)
- Mechanism: "We employ X P OS [SDP+ 22] relative position encoding for better long-context modeling." (Section 2.2) and "Relative position embedding xPos [SDP+ 22]" (Table 17, page 22).
- Where applied (input vs every layer vs attention bias): Not specified in the paper.
- Fixed across experiments vs modified per task vs ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable
- The paper treats positional encoding as a fixed architectural choice; no comparisons or ablations are described. Evidence: "We employ X P OS [SDP+ 22] relative position encoding for better long-context modeling." (Section 2.2). Multiple positional encodings are not compared and no claim that PE choice is "not critical" is made.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size: "The MLLM component has 24 layers with 2,048 hidden dimensions, 8,192 FFN intermediate size, and 32 attention heads, resulting in about 1.3B parameters." and "The total number of parameters of KOSMOS -1 is about 1.6B." (Section 3.2)
- Training data scale: "We use a batch size of 1.2 million tokens ... and train KOSMOS -1 for 300k steps, corresponding to about 360 billion tokens." (Section 3.2) and "we train KOSMOS -1 from scratch on web-scale multimodal corpora" (Abstract, page 2).
- Dataset sizes (examples): "LAION-2B contains about 2B English image-caption pairs, LAION-400M consists of 400M English image-caption pairs, and COYO-700M has 700M English image-caption pairs." (Appendix B.1.2, page 24) and "we end up with about 71 million documents for training." (Appendix B.1.3, page 24).
- Performance attribution vs scaling: The paper highlights strong performance despite smaller size: "our model is able to accomplish this feat with a smaller size of 1.6B, compared to Flamingo models." (Section 4.1.2). It does not explicitly claim that gains are primarily due to scaling model size or data; instead, it reports scale and compares results.
- Training tricks / prompting: "we perform language-only instruction tuning" (Section 3.3) and "Chain-of-thought prompting ... can significantly improve the performance in complex tasks." (Section 4.5).

## 11. Architectural Workarounds
- Token reduction: "Resampler [ADL+ 22] is used as an attentive pooling mechanism to reduce the number of image embeddings." (Section 2.1)
- Stability for scaling: "We use M AGNETO [WMH+ 22], a Transformer variant... It introduces an extra LayerNorm to each sublayer ... which allows us to effectively scale up the models without pain." (Section 2.2)
- Long-context positional encoding: "We employ X P OS [SDP+ 22] relative position encoding for better long-context modeling." (Section 2.2)
- Frozen vision backbone: "the image representation is obtained from a pretrained CLIP ViT-L/14 model" and "We freeze the parameters of the CLIP model except for the last layer during training." (Section 3.2)
- Causal masking: "The causal masking is used to mask out future information." (Section 2.2)

## 12. Explicit Limitations and Non-Claims
- Limitation vs human level: "Although there is still a large performance gap between the current model and the average level of adults" (Section 4.2.2).
- Future work: "In the future, we would like to scale up KOSMOS -1 in terms of model size ... and integrate the speech [WCW+ 23] capability into KOSMOS -1." (Conclusion, Section 5).
- Non-claim about external tools: "KOSMOS -1 does not access any external tools or resources." (Section 4.3.2).
- Other explicit non-claims (open-world learning, unrestrained multi-task learning, meta-learning): Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Multiple domains and modalities, spanning language tasks and vision/perception tasks (e.g., language tasks and "Web page question answering" on web pages).
- Task structure: Many benchmarked tasks across language, perception-language, and vision; evaluation is largely zero- and few-shot with prompting.
- Representation rigidity: Flattened token sequence with <image> delimiters; images resized to 224x224; max length 2,048.
- Model sharing vs specialization: Single pretrained model evaluated across tasks without task-specific finetuning or gradient updates.
- Positional encoding: Fixed XPOS relative position encoding; no ablation or comparison reported.

## 14. Final Classification
Classification: Multi-task, multi-domain (constrained).
Justification: The paper evaluates "language, perception-language, and vision tasks" (Section 1, page 4) spanning text-only tasks, image tasks, and web-page QA, indicating multiple domains and modalities. However, evaluations are on fixed benchmark datasets with zero-/few-shot prompting and no task-specific finetuning, which makes the setup broad but still constrained rather than unrestrained.

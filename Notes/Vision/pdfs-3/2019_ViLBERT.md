## 1. Basic Metadata
Title: ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks
Authors: Jiasen Lu; Dhruv Batra; Devi Parikh; Stefan Lee
Year: 2019
Venue: arXiv (arXiv:1908.02265v1 [cs.CV])

## 2. One-Sentence Contribution Summary
The paper introduces ViLBERT, a two-stream co-attentional transformer pretrained on Conceptual Captions to learn task-agnostic visual grounding for transfer to multiple vision-and-language tasks.

## 3. Tasks Evaluated
Overall task list (as stated):
> "We pretrain our model through two proxy tasks on the large, automatically collected Conceptual Captions dataset and then transfer it to multiple established vision-and-language tasks – visual question answering, visual commonsense reasoning, referring expressions, and caption-based image retrieval – by making only minor additions to the base architecture." (Abstract)

Task: Visual Question Answering (VQA)
Task type: Classification; Reasoning / relational
Dataset(s): VQA 2.0 (COCO)
Domain: COCO images
Evidence:
> "Visual Question Answering (VQA). The VQA task requires answering natural language questions about images." (Sec. 3.2, Visual Question Answering (VQA))
> "We train and evaluate on the VQA 2.0 dataset [3] consisting of 1.1 million questions about COCO images [5] each with 10 answers." (Sec. 3.2, Visual Question Answering (VQA))
> "As in [30], we treat VQA as a multi-label classification task – assigning a soft target score to each answer based on its relevancy to the 10 human answer responses." (Sec. 3.2, Visual Question Answering (VQA))

Task: Visual Commonsense Reasoning (VCR) Q->A
Task type: Classification; Reasoning / relational
Dataset(s): Visual Commonsense Reasoning (VCR)
Domain: movie scenes
Evidence:
> "Visual Commonsense Reasoning (VCR). Given an image, the VCR task presents two problems – visual question answering (Q→A) and answer justification (QA→R) – both being posed as multiplechoice problems." (Sec. 3.2, Visual Commonsense Reasoning (VCR))
> "The Visual Commonsense Reasoning (VCR) dataset consists of 290k multiple choice QA problems derived from 110k movie scenes." (Sec. 3.2, Visual Commonsense Reasoning (VCR))

Task: Visual Commonsense Reasoning (VCR) QA->R
Task type: Classification; Reasoning / relational
Dataset(s): Visual Commonsense Reasoning (VCR)
Domain: movie scenes
Evidence:
> "Visual Commonsense Reasoning (VCR). Given an image, the VCR task presents two problems – visual question answering (Q→A) and answer justification (QA→R) – both being posed as multiplechoice problems." (Sec. 3.2, Visual Commonsense Reasoning (VCR))
> "The Visual Commonsense Reasoning (VCR) dataset consists of 290k multiple choice QA problems derived from 110k movie scenes." (Sec. 3.2, Visual Commonsense Reasoning (VCR))

Task: Visual Commonsense Reasoning (VCR) Q->AR
Task type: Classification; Reasoning / relational
Dataset(s): Visual Commonsense Reasoning (VCR)
Domain: movie scenes
Evidence:
> "The holistic setting (Q→AR) requires both the chosen answer and then the chosen rationale to be correct." (Sec. 3.2, Visual Commonsense Reasoning (VCR))
> "The Visual Commonsense Reasoning (VCR) dataset consists of 290k multiple choice QA problems derived from 110k movie scenes." (Sec. 3.2, Visual Commonsense Reasoning (VCR))

Task: Grounding Referring Expressions (RefCOCO+)
Task type: Detection (localization)
Dataset(s): RefCOCO+
Domain: Not specified in the paper.
Evidence:
> "Grounding Referring Expressions. The referring expression task is to localize an image region given a natural language reference. We train and evaluate on the RefCOCO+ dataset [32]." (Sec. 3.2, Grounding Referring Expressions)

Task: Caption-Based Image Retrieval
Task type: Other (image-text retrieval)
Dataset(s): Flickr30k
Domain: images from Flickr
Evidence:
> "Caption-based image retrieval is the task of identifying an image from a pool given a caption describing its content." (Sec. 3.2, Caption-Based Image Retrieval)
> "We train and evaluate on the Flickr30k dataset [26] consisting of 31,000 images from Flickr with five captions each." (Sec. 3.2, Caption-Based Image Retrieval)

Task: 'Zero-shot' Caption-Based Image Retrieval
Task type: Other (image-text retrieval)
Dataset(s): Flickr30k (zero-shot)
Domain: images from Flickr
Evidence:
> "‘Zero-shot’ Caption-Based Image Retrieval. The previous tasks are all transfer tasks that include dataset specific fine-tuning. In this ‘zero-shot’ task, we directly apply the pretrained the multi-modal alignment prediction mechanism to caption-based image retrieval in Flickr30k [26] without finetuning (thus the description as ‘zero-shot’)." (Sec. 3.2, ‘Zero-shot’ Caption-Based Image Retrieval)

## 4. Domain and Modality Scope
Single domain evaluation? No.
Evidence (multiple datasets/domains):
> "We train and evaluate on the VQA 2.0 dataset [3] consisting of 1.1 million questions about COCO images [5] each with 10 answers." (Sec. 3.2, Visual Question Answering (VQA))
> "The Visual Commonsense Reasoning (VCR) dataset consists of 290k multiple choice QA problems derived from 110k movie scenes." (Sec. 3.2, Visual Commonsense Reasoning (VCR))
> "We train and evaluate on the Flickr30k dataset [26] consisting of 31,000 images from Flickr with five captions each." (Sec. 3.2, Caption-Based Image Retrieval)

Multiple domains within the same modality? Yes (multiple image datasets/sources).
Evidence: same as above.

Multiple modalities? Yes (vision + language).
Evidence:
> "We present ViLBERT (short for Vision-and-Language BERT), a model for learning task-agnostic joint representations of image content and natural language." (Abstract)
> "Our model which we call ViLBERT is shown in Fig. 1 and consists of two parallel BERT-style models operating over image regions and text segments." (Sec. 2.2, ViLBERT: Extending BERT to Jointly Represent Images and Text)

Domain generalization / cross-domain transfer claimed? Not claimed.
Evidence (transfer stated, but no explicit generalization claim):
> "We pretrain our model through two proxy tasks on the large, automatically collected Conceptual Captions dataset and then transfer it to multiple established vision-and-language tasks – visual question answering, visual commonsense reasoning, referring expressions, and caption-based image retrieval – by making only minor additions to the base architecture." (Abstract)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| VQA | Yes (pretrained base reused per task) | Yes | Yes (two-layer MLP) | "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end." (Sec. 3.2) ; "To fine-tune ViLBERT on VQA, we learn a two layer MLP on top of the element-wise product of the image and text representations hIMG and hCLS , mapping this representation to 3,129 possible answers." (Sec. 3.2, Visual Question Answering (VQA)) |
| VCR Q->A | Yes (pretrained base reused per task) | Yes | Yes (linear layer) | "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end." (Sec. 3.2) ; "We learn a linear layer on top of the post-elementwise product representation to predict a score for each pair." (Sec. 3.2, Visual Commonsense Reasoning (VCR)) |
| VCR QA->R | Yes (pretrained base reused per task) | Yes | Yes (linear layer) | "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end." (Sec. 3.2) ; "We learn a linear layer on top of the post-elementwise product representation to predict a score for each pair." (Sec. 3.2, Visual Commonsense Reasoning (VCR)) |
| VCR Q->AR | Yes (pretrained base reused per task) | Yes | Yes (linear layer) | "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end." (Sec. 3.2) ; "We learn a linear layer on top of the post-elementwise product representation to predict a score for each pair." (Sec. 3.2, Visual Commonsense Reasoning (VCR)) |
| RefCOCO+ | Yes (pretrained base reused per task) | Yes | Yes (linear layer) | "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end." (Sec. 3.2) ; "For fine-tuning, we pass the final representation hvi for each image region i into a learned linear layer to predict a matching score." (Sec. 3.2, Grounding Referring Expressions) |
| Caption-Based Image Retrieval | Yes (pretrained base reused per task) | Yes | Yes (alignment score + softmax) | "We follow a fine-tuning strategy where we modify the pretrained base model to perform the new task and then train the entire model end-to-end." (Sec. 3.2) ; "We compute the alignment score (as in alignment prediction pretraining) for each and apply a softmax." (Sec. 3.2, Caption-Based Image Retrieval) |
| 'Zero-shot' Caption-Based Image Retrieval | Yes (pretrained model used) | No | No new head stated (uses alignment prediction objective) | "In this ‘zero-shot’ task, we directly apply the pretrained the multi-modal alignment prediction mechanism to caption-based image retrieval in Flickr30k [26] without finetuning (thus the description as ‘zero-shot’)." (Sec. 3.2) ; "We use the alignment prediction objective as a scoring function" (Sec. 3.2, ‘Zero-shot’ Caption-Based Image Retrieval) |

## 6. Input and Representation Constraints
Fixed or variable input resolution? Not specified in the paper.
Fixed patch size? Not specified in the paper.
Fixed number of tokens?
- Visual tokens: variable; "We select regions where class detection probability exceeds a confidence threshold and keep between 10 to 36 high-scoring boxes." (Sec. 3.1, Implementation Details)
- Text tokens: token sequence with special tokens; no maximum specified. "BERT operates over sequences of discrete tokens comprised of vocabulary words and a small set of special tokens: SEP, CLS, and MASK." (Sec. 2.1, Text Representation)
Fixed dimensionality / representation specifics:
- Text input representation uses embeddings + positional + segment encodings. "For a given token, the input representation is a sum of a token-specific learned embedding [28] and encodings for position (i.e. token’s index in the sequence) and segment (i.e. index of the token’s sentence if multiple exist)." (Sec. 2.1, Text Representation)
- Visual stream hidden size: "Transformer and co-attentional transformer blocks in the visual stream have hidden state size of 1024 and 8 attention heads." (Sec. 3.1, Implementation Details)
- Language stream size: "BERTBASE model [12] which has 12 layers of transformer blocks with each block having a hidden state size of 762 and 12 attention heads." (Sec. 3.1, Implementation Details)
Spatial / region representation constraints:
- "Image Representations. We generate image region features by extracting bounding boxes and their visual features from a pre-trained object detection network (see Sec. 3.1)." (Sec. 2.2, Image Representations)
- "Unlike words in text, image regions lack a natural ordering. we encode spatial location instead, constructing a 5-d vector from region position (normalized top-left and bottom-right coordinates) and the fraction of image area covered. This is then projected to match the dimension of the visual feature and they are summed." (Sec. 2.2, Image Representations)
- "We mark the beginning of an image region sequence with a special IMG token representing the entire image (i.e. mean-pooled visual features with a spatial encoding corresponding to the entire image)." (Sec. 2.2, Image Representations)
Padding or resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
Maximum sequence length: Not specified in the paper.
Sequence length fixed or variable?
- Visual stream: variable number of regions (10-36). "We select regions where class detection probability exceeds a confidence threshold and keep between 10 to 36 high-scoring boxes." (Sec. 3.1, Implementation Details)
- Text stream: variable sequence of tokens; no maximum specified. "BERT operates over sequences of discrete tokens comprised of vocabulary words and a small set of special tokens: SEP, CLS, and MASK." (Sec. 2.1, Text Representation)
Attention type:
- Standard multi-head attention + co-attention. "We introduce a co-attentional transformer layer... the module computes query, key, and value matrices as in a standard transformer block. However, the keys and values from each modality are passed as input to the other modality’s multi-headed attention block." (Sec. 2.2, Co-Attentional Transformer Layers)
Mechanisms to manage computational cost:
- Sparse cross-modal interaction: "This structure allows for variable depths for each modality and enables sparse interaction through co-attention." (Fig. 1 caption)
- Caching for efficiency: "For efficiency, we cache the linguistic stream representation before the first Co-TRM layer – effectively freezing the linguistic representation before fusion." (Sec. 3.2, Caption-Based Image Retrieval)

## 8. Positional Encoding (Critical Section)
Positional encoding mechanism used:
- Text: position + segment encodings added to token embeddings (absolute position by index). "For a given token, the input representation is a sum of a token-specific learned embedding [28] and encodings for position (i.e. token’s index in the sequence) and segment (i.e. index of the token’s sentence if multiple exist)." (Sec. 2.1, Text Representation)
- Vision: spatial location encoding from region coordinates, projected and summed. "we encode spatial location instead, constructing a 5-d vector from region position (normalized top-left and bottom-right coordinates) and the fraction of image area covered. This is then projected to match the dimension of the visual feature and they are summed." (Sec. 2.2, Image Representations)
Where it is applied:
- Text: input representation (sum of embeddings). "the input representation is a sum of a token-specific learned embedding [28] and encodings for position... and segment..." (Sec. 2.1, Text Representation)
- Vision: spatial encoding summed with visual features; IMG token gets spatial encoding for entire image. "This is then projected to match the dimension of the visual feature and they are summed." and "We mark the beginning of an image region sequence with a special IMG token representing the entire image (i.e. mean-pooled visual features with a spatial encoding corresponding to the entire image)." (Sec. 2.2, Image Representations)
Fixed across experiments? Not specified in the paper.
Modified per task? Not specified in the paper.
Ablated or compared against alternatives? Not specified in the paper.

## 9. Positional Encoding as a Variable
Treated as a core research variable? Not specified in the paper.
Treated as a fixed architectural assumption? Not specified in the paper (only described as part of the input representation).
Multiple positional encodings compared? Not specified in the paper.
Claims PE choice is \"not critical\" or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking
Model size(s):
- "BERTBASE model [12] which has 12 layers of transformer blocks with each block having a hidden state size of 762 and 12 attention heads." (Sec. 3.1, Implementation Details)
- "Transformer and co-attentional transformer blocks in the visual stream have hidden state size of 1024 and 8 attention heads." (Sec. 3.1, Implementation Details)
Dataset size(s):
- "Conceptual Captions is a collection of 3.3 million image-caption pairs automatically scraped from alt-text enabled web images." (Sec. 3.1, Training ViLBERT)
- "Since some links had become broken by the time we downloaded the data, our model is trained with around 3.1 million image-caption pairs." (Sec. 3.1, Training ViLBERT)
Performance gains attributed to scaling data:
- "We can see that the accuracy grows monotonically as the amount of data increases, which suggests that ViLBERT may benefit from even more pretraining data." (Sec. 4, Results and Analysis)
Performance gains attributed to scaling model size:
- "We choose to use the BASE model due to concerns over training time but find it likely the more powerful BERTLARGE model could further boost performance." (Sec. 3.1, Implementation Details)
Performance gains attributed to architecture depth/structure:
- "We find that VQA and Image Retrieval tasks benefit from greater depth - performance increases monotonically until a layer depth of 6." (Sec. 4, Results and Analysis)
- "Our architecture improves performance over a single-stream model." (Sec. 4, Results and Analysis)
Training tricks as primary driver? Not specified in the paper.

## 11. Architectural Workarounds
- Two-stream architecture with co-attention to control cross-modal interaction depth. "we develop a two-stream architecture modelling each modality separately and then fusing them through a small set of attention-based interactions." (Sec. 2.2, ViLBERT: Extending BERT to Jointly Represent Images and Text)
- Sparse cross-modal interaction / variable depth. "This structure allows for variable depths for each modality and enables sparse interaction through co-attention." (Fig. 1 caption)
- Co-attentional transformer layer exchanging keys/values between modalities. "the keys and values from each modality are passed as input to the other modality’s multi-headed attention block." (Sec. 2.2, Co-Attentional Transformer Layers)
- Region-based visual inputs from object detector. "We generate image region features by extracting bounding boxes and their visual features from a pre-trained object detection network (see Sec. 3.1)." (Sec. 2.2, Image Representations)
- Task-specific heads for transfer. "In all cases, the modification is trivial – typically amounting to learning a classification layer." (Sec. 3.2, Vision-and-Language Transfer Tasks)
- Efficiency workaround for retrieval. "For efficiency, we cache the linguistic stream representation before the first Co-TRM layer – effectively freezing the linguistic representation before fusion." (Sec. 3.2, Caption-Based Image Retrieval)

## 12. Explicit Limitations and Non-Claims
Stated limitations / missing task families:
> "While we address many vision-and-language tasks in Sec. 3.2, we do miss some families of tasks including visually grounded dialog [4, 45], embodied tasks like question answering [7] and instruction following [8], and text generation tasks like image and video captioning [5]." (Sec. 5, Related Work)
Open questions:
> "There are open questions on how to incorporate long sequences of images and text found in dialog, embodied tasks, and video processing. Further, it is unclear how to effectively decode output text from our bidirectional model as existing greedy decoders like beam-search do not apply." (Sec. 5, Related Work)
Future work statements:
> "We consider extensions of our model to other vision-and-language tasks (including those requiring generation) as well as multi-task learning as exciting future work." (Sec. 6, Conclusion)
Explicit non-claims about open-world learning / unrestrained multi-task learning / meta-learning: Not specified in the paper.

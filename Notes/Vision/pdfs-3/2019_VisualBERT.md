# VisualBERT (2019) Survey Answers

## 1. Basic Metadata
- Title: VisualBERT: A Simple and Performant Baseline for Vision and Language
- Authors: Liunian Harold Li; Mark Yatskar; Da Yin; Cho-Jui Hsieh; Kai-Wei Chang
- Year: 2019
- Venue: arXiv (arXiv:1908.03557v1 [cs.CV], 9 Aug 2019)

## 2. One-Sentence Contribution Summary
VisualBERT introduces a simple BERT-based vision-language model that pre-trains on image-caption data to learn joint representations for multiple vision-and-language tasks.

## 3. Tasks Evaluated

### Task: Visual Question Answering (VQA 2.0)
- Task type: Classification
- Dataset(s): VQA 2.0 (COCO)
- Domain: Natural images (COCO)
- Evidence (quotes):
  - "Given an image and a question, the task is to correctly answer the question." (Section 4.1 VQA, p.4)
  - "We use the VQA 2.0 (Goyal et al., 2017), consisting of over 1 million questions about images from COCO." (Section 4.1 VQA, p.4)
  - "consider it a classification problem, where the model only needs to choose one answer from a limited answer pool." (Appendix A VQA, p.13)

### Task: Visual Commonsense Reasoning (VCR)
- Task type: Classification (multi-choice)
- Dataset(s): VCR
- Domain: Movie scenes
- Evidence (quotes):
  - "VCR consists of 290k questions derived from 110k movie scenes, where the questions focus on visual commonsense." (Section 4.2 VCR, p.5)
  - "The task is decomposed into two multi-choice sub-tasks wherein we train individual models:" (Section 4.2 VCR, p.5)
  - "question answering" (Section 4.2 VCR, p.5)
  - "answer justification" (Section 4.2 VCR, p.5)
  - "The model is trained to classify which of the four input sequences is correct." (Section 4.2 VCR, p.5)

### Task: Natural Language for Visual Reasoning (NLVR2)
- Task type: Classification (binary true/false)
- Dataset(s): NLVR2
- Domain: Web images
- Evidence (quotes):
  - "NLVR2 is a dataset for joint reasoning about natural language and images, with a focus on semantic diversity, compositionality, and visual reasoning challenges." (Section 4.3 NLVR2, p.6)
  - "The task is to determine whether a natural language caption is true about a pair of images." (Section 4.3 NLVR2, p.6)
  - "The dataset consists of over 100k examples of English sentences paired with web images." (Section 4.3 NLVR2, p.6)

### Task: Flickr30K Entities (Region-to-Phrase Grounding)
- Task type: Other (phrase grounding / localization)
- Dataset(s): Flickr30K Entities
- Domain: Natural images
- Evidence (quotes):
  - "Flickr30K Entities dataset tests the ability of systems to ground phrases in captions to bounding regions in the image." (Section 4.4 Flickr30K Entities, p.6)
  - "The task is, given spans from a sentence, selecting the bounding regions they correspond to." (Section 4.4 Flickr30K Entities, p.6)
  - "The dataset consists of 30k images and nearly 250k annotations." (Section 4.4 Flickr30K Entities, p.6)

## 4. Domain and Modality Scope
- Modality scope: Vision + language (multimodal) across all evaluations.
  - Evidence: "We evaluate VisualBERT on four different types of vision-and-language applications:" (Section 4 Experiment, p.4)
- Domain scope: Multiple domains within the same modality (images + text).
  - Evidence for different domains:
    - COCO natural images: "images from COCO" (Section 4.1 VQA, p.4)
    - Movie scenes: "movie scenes" (Section 4.2 VCR, p.5)
    - Web images: "web images" (Section 4.3 NLVR2, p.6)
    - Flickr30K images: "30k images" (Section 4.4 Flickr30K Entities, p.6)
- Domain generalization / cross-domain transfer claim:
  - "Despite substantial domain difference between COCO and VCR, with VCR covering scenes from movies, pre-training on COCO still helps significantly." (Section 4.2 VCR, p.5)

## 5. Model Sharing Across Tasks
- Evidence for training protocol and task-specific heads:
  - "Our training procedure contains three phases:" (Section 3.3 Training VisualBERT, p.3)
  - "Task-Agnostic Pre-Training Here we train VisualBERT on COCO using two visually-grounded language model objectives." (Section 3.3 Training VisualBERT, p.3)
  - "Task-Specific Pre-Training Before fine-tuning VisualBERT to a downstream task, we find it beneficial to train the model using the data of the task with the masked language modeling with the image objective." (Section 3.3 Training VisualBERT, p.3)
  - "Fine-Tuning This step mirrors BERT fine-tuning, where a task-specific input, output, and objective are introduced, and the Transformer is trained to maximize performance on the task." (Section 3.3 Training VisualBERT, p.3)
  - "to apply BERT to a particular task, a task-specific input, output layer, and objective are introduced, and the model is fine-tuned on the task data from pre-trained parameters." (Section 3.1 Background, p.3)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| VQA 2.0 | Pre-trained weights shared as initialization; task trained separately | Yes | Yes | "Task-Specific Pre-Training..." and "Fine-Tuning... task-specific input, output" (Section 3.3, p.3) |
| VCR | Pre-trained weights shared as initialization; task trained separately | Yes | Yes | "Task-Specific Pre-Training..." and "Fine-Tuning... task-specific input, output" (Section 3.3, p.3) |
| NLVR2 | Pre-trained weights shared as initialization; task trained separately | Yes | Yes | "Task-Specific Pre-Training..." and "Fine-Tuning... task-specific input, output" (Section 3.3, p.3) |
| Flickr30K Entities | Pre-trained weights shared as initialization; task trained separately | Yes | Yes | "Task-Specific Pre-Training..." and "Fine-Tuning... task-specific input, output" (Section 3.3, p.3) |

## 6. Input and Representation Constraints
- Image input is region proposals treated as tokens:
  - "image features extracted from object proposals are treated as unordered input tokens and fed into VisualBERT along with text." (Introduction, p.1)
- Visual tokens correspond to detected bounding regions:
  - "corresponds to a bounding region in the image, derived from an object detector." (Section 3.2 VisualBERT, p.3)
- Text length cap:
  - "text sequences whose lengths are longer than 128 are capped." (Section 4 Experiment, p.4)
- Fixed number of region proposals for NLVR2:
  - "use 144 proposals per image." (Section 4.3 NLVR2, p.6)
- Reduced number of visual features in ablations:
  - "all these models are trained with only 36 features per image (including the full model)." (Section 5.1 Ablation Study, p.7)
- Avoid grid-level features to limit sequence length:
  - "We do not use grid-level features from ResNet152 because it results in longer sequences and longer training time." (Appendix A VQA, p.13)
- Visual/text embedding dimensionality alignment:
  - "If text and visual input embeddings are of different dimension, we project the visual embeddings into a space of the same dimension as the text embeddings." (Section 4 Experiment, footnote 1, p.4)
- VCR alignment uses position embeddings for matched words/regions:
  - "The dataset also provides alignments between words and bounding regions that are referenced to in the text, which we utilize by using the same position embeddings for matched words and regions." (Section 4.2 VCR, p.5)
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens (global across all tasks): Not specified in the paper.
- Fixed dimensionality beyond projection to BERT hidden size: Not specified in the paper.
- Padding or resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length (text):
  - "text sequences whose lengths are longer than 128 are capped." (Section 4 Experiment, p.4)
- Sequence length fixed or variable:
  - Text is variable up to the 128 cap; visual tokens vary by dataset with fixed proposal counts in some settings (e.g., "use 144 proposals per image." Section 4.3 NLVR2, p.6). Beyond these, Not specified in the paper.
- Attention type:
  - "VisualBERT consists of a stack of Transformer layers that implicitly align elements of an input text and regions in an associated input image with self-attention." (Abstract, p.1)
  - This indicates global self-attention over combined text and image tokens; no windowed/hierarchical/sparse attention is specified.
- Mechanisms to manage computational cost:
  - "text sequences whose lengths are longer than 128 are capped." (Section 4 Experiment, p.4)
  - "We do not use grid-level features from ResNet152 because it results in longer sequences and longer training time." (Appendix A VQA, p.13)
  - "use 144 proposals per image." (Section 4.3 NLVR2, p.6)
  - "all these models are trained with only 36 features per image (including the full model)." (Section 5.1 Ablation Study, p.7)

## 8. Positional Encoding (Critical Section)
- Mechanism used:
  - "a position embedding ep , indicating the position of the token in the sentence." (Section 3.1 Background, p.3)
- Where applied:
  - "computed as the sum of 1) a token embedding et , specific to the subword, 2) a segment embedding es , indicating which part of text the token comes from (e.g., the hypothesis from an entailment pair) and 3) a position embedding ep , indicating the position of the token in the sentence." (Section 3.1 Background, p.3)
- Visual-side positional handling when alignments are provided (task-specific use in VCR):
  - "The dataset also provides alignments between words and bounding regions that are referenced to in the text, which we utilize by using the same position embeddings for matched words and regions." (Section 4.2 VCR, p.5)
- Fixed across experiments vs modified per task:
  - Position embeddings are described as part of the base input embedding; VCR explicitly uses position embeddings for matched words and regions (quote above). Beyond this, Not specified in the paper.
- Ablations or comparisons of positional encoding:
  - Not specified in the paper.

## 9. Positional Encoding as a Variable
- Treated as a core research variable? Not specified in the paper.
- Fixed architectural assumption? Position embeddings are defined as part of the input embedding; no alternative encodings are discussed. Evidence: "a position embedding ep , indicating the position of the token in the sentence." (Section 3.1 Background, p.3)
- Multiple positional encodings compared? Not specified in the paper.
- PE choice claimed not critical or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale)
- Model size(s):
  - "The Transformer encoder in all models has the same configuration as BERTBASE : 12 layers, a hidden size of 768, and 12 self-attention heads." (Section 4 Experiment, p.4)
- Dataset size(s):
  - COCO pre-training: "Karpathy train split (Karpathy & Fei-Fei, 2015) of COCO for task-agnostic pre-training, which has around 100k images with 5 captions each." (Section 4 Experiment, p.4)
  - VQA 2.0: "consisting of over 1 million questions about images from COCO." (Section 4.1 VQA, p.4)
  - VCR: "VCR consists of 290k questions derived from 110k movie scenes" (Section 4.2 VCR, p.5)
  - NLVR2: "over 100k examples of English sentences paired with web images." (Section 4.3 NLVR2, p.6)
  - Flickr30K Entities: "The dataset consists of 30k images and nearly 250k annotations." (Section 4.4 Flickr30K Entities, p.6)
- Performance gains attributed to scaling data or architecture:
  - "Both variants underperform, showing that pre-training on paired vision and language data is important." (Section 5.1 Ablation Study, p.7)
  - "Overall, the results confirm that the most important design choices are task-agnostic pre-training (C1) and early fusion of vision and language (C2). In pre-training, both the inclusion of additional COCO data and using both images and captions are paramount." (Section 5.1 Ablation Study, p.7)
- Performance gains attributed primarily to scaling model size or training tricks: Not specified in the paper.

## 11. Architectural Workarounds
- Use of object proposals instead of grid features to limit sequence length:
  - "image features extracted from object proposals are treated as unordered input tokens and fed into VisualBERT along with text." (Introduction, p.1)
  - "We do not use grid-level features from ResNet152 because it results in longer sequences and longer training time." (Appendix A VQA, p.13)
- Sequence length cap:
  - "text sequences whose lengths are longer than 128 are capped." (Section 4 Experiment, p.4)
- Limiting number of proposals per image:
  - "use 144 proposals per image." (Section 4.3 NLVR2, p.6)
  - "all these models are trained with only 36 features per image (including the full model)." (Section 5.1 Ablation Study, p.7)
- Early vs late fusion architectural variant:
  - "VisualBERT w/o Early Fusion: VisualBERT but where image representations are not combined with the text in the initial Transformer layer but instead at the very end with a new Transformer layer." (Section 4 Experiment, p.4)
- Segment embeddings to handle multiple images (NLVR2):
  - "We modify the segment embedding mechanism in VisualBERT and assign features from different images with different segment embeddings." (Section 4.3 NLVR2, p.6)
- Additional self-attention block for phrase grounding:
  - "For task specific fine-tuning, we introduce an additional self-attention block and use the average attention weights from each head to predict the alignment between boxes and phrases." (Section 4.4 Flickr30K Entities, p.6)
- Projection to match visual/text embedding dimensionality:
  - "If text and visual input embeddings are of different dimension, we project the visual embeddings into a space of the same dimension as the text embeddings." (Section 4 Experiment, footnote 1, p.4)

## 12. Explicit Limitations and Non-Claims
- Stated future work / limitations:
  - "For future work, we are curious about whether we could extend VisualBERT to image-only tasks, such as scene graph parsing and situation recognition. Pre-training VisualBERT on larger caption datasets such as Visual Genome and Conceptual Caption is also a valid direction." (Section 6 Conclusion and Future Work, p.9)
- Explicit statements about what the model does not attempt to do (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified in the paper.

# Flamingo (2022_Flamingo.pdf) Survey Answers

## 1. Basic Metadata
- Title: "Flamingo: a Visual Language Model for Few-Shot Learning" (page 1).
- Authors: Jean-Baptiste Alayrac; Jeff Donahue; Pauline Luc; Antoine Miech; Iain Barr; Yana Hasson; Karel Lenc; Arthur Mensch; Katie Millican; Malcolm Reynolds; Roman Ring; Eliza Rutherford; Serkan Cabi; Tengda Han; Zhitao Gong; Sina Samangooei; Marianne Monteiro; Jacob Menick; Sebastian Borgeaud; Andrew Brock; Aida Nematzadeh; Sahand Sharifzadeh; Mikolaj Binkowski; Ricardo Barreira; Oriol Vinyals; Andrew Zisserman; Karen Simonyan (page 1).
- Year: 2022 ("arXiv:2204.14198v2" and "15 Nov 2022" on page 1).
- Venue: "36th Conference on Neural Information Processing Systems (NeurIPS 2022)." (page 1); arXiv:2204.14198v2 (page 1).

## 2. One-Sentence Contribution Summary
Flamingo introduces a visual language model that can be prompted with a few examples to perform a wide range of image/video tasks without task-specific fine-tuning.

## 3. Tasks Evaluated
The paper states it evaluates on "a wide array of 16 popular multimodal image/video and language benchmarks" (Section 3, page 7) and Table 6 enumerates the benchmarks with task descriptions (page 31). The abstract also characterizes the task spectrum as including "visual question-answering," "captioning," and "multiple-choice visual question-answering" (page 1).

| Task | Task type | Dataset(s) used | Domain | Evidence (quote + page) |
| --- | --- | --- | --- | --- |
| ImageNet-1k object classification | Classification | ImageNet-1k [94] | Image | "Object classification" (Table 6, page 31). |
| COCO captioning | Generation | MS-COCO [15] | Image | "Scene description" (Table 6, page 31). |
| VQAv2 visual question answering | Reasoning / relational; Other (QA) | VQAv2 [3] | Image | "Scene understanding QA" (Table 6, page 31). |
| OKVQA visual question answering | Reasoning / relational; Other (QA) | OKVQA [69] | Image | "External knowledge QA" (Table 6, page 31). |
| Flickr30k captioning | Generation | Flickr30k [139] | Image | "Scene description" (Table 6, page 31). |
| VizWiz visual question answering | Reasoning / relational; Other (QA) | VizWiz [35] | Image | "Scene understanding QA" (Table 6, page 31). |
| TextVQA visual question answering (text reading) | Reasoning / relational; Other (QA) | TextVQA [100] | Image | "Text reading QA" (Table 6, page 31). |
| Visual Dialog | Other (Dialogue) | VisDial [20] | Image | "Visual Dialogue" (Table 6, page 31). |
| Hateful Memes | Classification | HatefulMemes [54] | Image | "Meme classification" (Table 6, page 31). |
| Kinetics700 action classification | Classification | Kinetics700 2020 [102] | Video | "Action classification" (Table 6, page 31). |
| VATEX video captioning | Generation | VATEX [122] | Video | "Event description" (Table 6, page 31). |
| MSVDQA video question answering | Reasoning / relational; Other (QA) | MSVDQA [130] | Video | "Event understanding QA" (Table 6, page 31). |
| YouCook2 video captioning | Generation | YouCook2 [149] | Video | "Event description" (Table 6, page 31). |
| MSRVTTQA video question answering | Reasoning / relational; Other (QA) | MSRVTTQA [130] | Video | "Event understanding QA" (Table 6, page 31). |
| iVQA video question answering | Reasoning / relational; Other (QA) | iVQA [135] | Video | "Event understanding QA" (Table 6, page 31). |
| RareAct composite action retrieval | Other (retrieval) | RareAct [73] | Video | "Composite action retrieval" (Table 6, page 31). |
| NextQA temporal/causal QA | Reasoning / relational; Other (QA) | NextQA [129] | Video | "Temporal/Causal QA" (Table 6, page 31). |
| STAR multiple-choice QA | Reasoning / relational; Classification | STAR [128] | Video | "Multiple-choice QA" (Table 6, page 31). |

## 4. Domain and Modality Scope
- Single domain? No. The evaluation spans "16 multimodal image/video and language benchmarks" (Appendix B.1.4, page 30).
- Multiple domains within the same modality? Yes. The benchmarks span both images and videos ("image/video" in "16 multimodal image/video and language benchmarks"; page 30).
- Multiple modalities? Yes. The model handles "sequences of arbitrarily interleaved visual and textual data" and can "seamlessly ingest images or videos as inputs" (Abstract, page 1).
- Domain generalization or cross-domain transfer claims? Not claimed. There is no explicit statement of domain generalization or cross-domain transfer; therefore: Not specified in the paper.

## 5. Model Sharing Across Tasks
Evidence for shared weights across tasks: "On six tasks, Flamingo even outperforms the fine-tuned SotA despite using a single set of model weights and only 32 task-specific examples." (Section 3.1, page 8). Evidence for per-task fine-tuning (subset of tasks): "We fine-tune Flamingo on all nine tasks where Flamingo does not achieve SotA with few-shot learning." (Table 2 caption, page 8; tasks listed in Table 2 include VQAv2, COCO, VATEX, VizWiz, MSRVTTQA, VisDial, YouCook2, TextVQA, HatefulMemes).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k object classification | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| COCO captioning | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes COCO). |
| VQAv2 visual question answering | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes VQAv2). |
| OKVQA visual question answering | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| Flickr30k captioning | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| VizWiz visual question answering | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes VizWiz). |
| TextVQA visual question answering | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes TextVQA). |
| Visual Dialog | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes VisDial). |
| Hateful Memes | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes HatefulMemes). |
| Kinetics700 action classification | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| VATEX video captioning | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes VATEX). |
| MSVDQA video question answering | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| YouCook2 video captioning | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes YouCook2). |
| MSRVTTQA video question answering | Yes (few-shot single model) | Yes (fine-tuned results reported) | Not specified in the paper | "using a single set of model weights" (page 8); "We fine-tune Flamingo on all nine tasks..." (page 8; Table 2 includes MSRVTTQA). |
| iVQA video question answering | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| RareAct composite action retrieval | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| NextQA temporal/causal QA | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |
| STAR multiple-choice QA | Yes (few-shot single model) | Not specified in the paper | Not specified in the paper | "using a single set of model weights" (page 8). |

## 6. Input and Representation Constraints
- Image resizing and padding: "The visual inputs are resized to 320 × 320 while preserving their aspect ratios, padding the image with the mean value if required." (Appendix B.1.2, page 29).
- Contrastive pretraining resolution: "The training image resolution is 288 × 288" (Appendix B.1.3, page 30).
- Fine-tuning resolution: "we ... increase the resolution of the input images from 320 × 320 to 480 × 480." (Appendix B.2.2, page 33).
- Vision encoder representation: "We use the output of the final stage, a 2D spatial grid of features that is flattened to a 1D sequence." (Section 2.1, page 5).
- Video representation: "For video inputs, frames are sampled at 1 FPS and encoded independently to obtain a 3D spatio-temporal grid of features to which learned temporal embeddings are added. Features are then flattened to 1D" (Section 2.1, page 5).
- Fixed number of visual tokens: "produces a fixed number of visual outputs (64)" (Section 2.1, page 5).
- Training text length and number of images (M3W): "we sample a random subsequence of L = 256 tokens and take up to the first N = 5 images included in the sampled sequence. Further images are discarded in order to save compute." (Section 2.4, page 6).
- More images at inference: "trained with sequences limited to only 5 images on M3W, they are still able to benefit from up to 32 images or videos during inference." (Section 3.1, page 8).
- Fixed video frames during training, larger at inference: "For video training, we temporally sample a clip of 8 frames sampled at one frame per second (fps) from each training video. Although our model was trained with a fixed number of 8 frames, at inference time, we input 30 frames at 3 FPS." (Appendix B.1.2, page 29).
- Patch size: Not specified in the paper.
- Fixed number of tokens in the language stream beyond the M3W sample length: Not specified in the paper (aside from the M3W L=256 sample and the LM maximum sequence length in Section 7).

## 7. Context Window and Attention Structure
- Maximum sequence length: "maximum sequence length (2048) our LMs have been trained on" (Appendix D.1, page 38).
- Fixed or variable length: Training on M3W uses a fixed sampled length ("random subsequence of L = 256 tokens"; page 6), but the model is designed to handle variable numbers of images/videos at inference ("seamlessly generalise to any number of visual inputs"; Section 2.3, page 6) and can use up to 32 images/videos at inference (page 8).
- Attention type and masking: "masking the full text-to-image cross-attention matrix" and "At a given text token, the model attends to the visual tokens of the image that appeared just before it in the interleaved sequence, rather than to all previous images" (Section 2.3, page 6). This is a masked/sparse cross-attention to the most recent image/video.
- Compute/complexity mechanisms: "produces a fixed number of visual outputs (64), reducing the computational complexity of the vision-text cross-attention" (Section 2.1, page 5); "take up to the first N = 5 images ... Further images are discarded in order to save compute" (Section 2.4, page 6); and cross-attention insertion frequency is reduced for larger models ("add a GATED XATTN - DENSE every fourth layer for Flamingo-9B and every seventh for Flamingo-80B"; Section 3.3, page 9).

## 8. Positional Encoding (Critical Section)
- Mechanism: "adding a learnt temporal position encoding to each feature within a given video frame (an image being considered as a single-frame video). Note that we only use temporal encodings and no explicit spatial grid position encodings; we did not observe improvements from the latter." (Appendix A.1.1, page 23).
- Where it is applied: The temporal position encoding is added to visual features before flattening in the Perceiver Resampler pipeline (Appendix A.1.1, page 23).
- Fixed vs modified: For video inference they modify temporal embeddings via interpolation: "linearly interpolating the learnt temporal position embedding of the Perceiver Resampler at inference time." (Appendix B.1.2, page 29).
- Positional encoding in the language model: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Core variable or fixed assumption? Positional encoding is treated as a fixed design choice for the visual stream; the only explicit comparison is that they do not use spatial encodings because "we did not observe improvements from the latter" (Appendix A.1.1, page 23).
- Multiple positional encodings compared? Only the presence/absence of explicit spatial grid position encodings is mentioned: "we only use temporal encodings and no explicit spatial grid position encodings; we did not observe improvements from the latter." (Appendix A.1.1, page 23). No other PE variants are compared.
- Claim that PE choice is not critical? Not claimed. The only statement is the lack of improvement from spatial encodings (page 23).

## 10. Evidence of Constraint Masking (Scale vs. Structure)
- Model sizes: "Flamingo-3B, Flamingo-9B and Flamingo-80B" (Section 2.2, page 5) with parameter counts summarized in Table 5 (page 28).
- Dataset sizes: M3W: "There are 43.3M instances (documents) in total, with a total of 185M images and 182 GB of text." (Table 14, page 47); LTIP: "The dataset contains 312M image-text pairs." (Table 15, page 50); VTP: "The dataset contains 27M video-text pairs." (Table 16, page 52); ALIGN: "ALIGN [50] dataset, composed of 1.8 billion images paired with alt-text." (Section 2.4, page 6).
- Scaling model size and shots: "the larger the model, the better the few-shot performance" and "The performance also improves with the number of shots." (Section 3.1, page 8).
- Data mixture importance: "removing the interleaved image-text dataset M3W leads to a decrease of more than 17% in performance while removing the conventional paired image-text pairs also decreases performance" (Section 3.3, page 9).
- Training/optimization tricks: "We accumulate gradients over all datasets, which we found outperforms a “round-robin” approach" (Section 2.4, page 6).

## 11. Architectural Workarounds
- Perceiver Resampler reduces cross-attention cost: "produces a fixed number of visual outputs (64), reducing the computational complexity of the vision-text cross-attention." (Section 2.1, page 5).
- Single-image masked cross-attention: "At a given text token, the model attends to the visual tokens of the image that appeared just before it ... rather than to all previous images." (Section 2.3, page 6).
- Gated cross-attention layers for stability: "insert gated cross-attention dense blocks ... we use a tanh-gating mechanism" (Section 2.2, page 5).
- Cross-attention frequency for compute trade-off: "add a GATED XATTN - DENSE every fourth layer for Flamingo-9B and every seventh for Flamingo-80B." (Section 3.3, page 9).
- Limiting number of images for compute: "take up to the first N = 5 images ... Further images are discarded in order to save compute." (Section 2.4, page 6).
- Freezing pretrained LM to prevent forgetting: "freezing the language model is a better alternative" to avoid catastrophic forgetting (Section 3.3, page 9).

## 12. Explicit Limitations and Non-Claims
- Classification limitation: "the classification performance of Flamingo lags behind that of state-of-the-art contrastive models" (Section 5, page 10).
- Sequence-length limitation: "LMs generalise poorly to sequences longer than the training ones." (Section 5, page 10).
- In-context learning sensitivity: "in-context learning is known to be highly sensitive to various aspects of the demonstrations" (Section 5, page 10).
- Intersectional bias analysis not done: "We did not investigate intersectional biases." (Model Card, page 46).
- Non-claim about human decision-making: "The model is not intended to inform decisions about matters central to human life or flourishing." (Model Card, page 46).
- Out-of-scope uses: "Uses of the model for visually conditioned language generation in harmful or deceitful settings." (Model Card, page 45).

## 13. Constraint Profile (Synthesis)
- Domain scope: Multi-task across image and video benchmarks ("16 multimodal image/video and language benchmarks"; page 30), i.e., multi-domain but constrained to fixed benchmark datasets.
- Task structure: Mix of QA, captioning, classification, dialogue, and retrieval tasks (Table 6, page 31), all evaluated via few-shot prompting with fixed templates (Appendix B.1.5, page 32).
- Representation rigidity: Inputs are resized to fixed resolutions (320 × 320, with specified padding; page 29), fixed visual token count (64; page 5), and training sequences capped (L=256, N=5; page 6).
- Model sharing vs specialization: A single model is used across tasks with shared weights (page 8), with optional per-task fine-tuning reported for a subset (page 8).
- Positional encoding role: Learned temporal embeddings for visual features only, no explicit spatial encodings (page 23); PE is not a primary experimental variable.

## 14. Final Classification
**Multi-task, multi-domain (constrained).** The paper evaluates on "16 multimodal image/video and language benchmarks" (page 30), spanning both image and video datasets and multiple task types (Table 6, page 31), which is multi-task and multi-domain. The evaluation is still constrained to fixed benchmark suites and few-shot prompting with a single shared model (page 8), rather than open-ended, unbounded multi-domain learning.

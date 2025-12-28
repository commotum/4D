### 1. Basic Metadata
- Title: What matters when building vision-language models?
- Authors: Hugo Laurençon; Léo Tronchon; Matthieu Cord; Victor Sanh.
- Year: 2024.
- Venue: arXiv (arXiv:2405.02246v1 [cs.CV]).
- Evidence: "What matters when building vision-language models?" and author list "Hugo Laurençon∗,1,2                   Léo Tronchon∗,1       Matthieu Cord2                 Victor Sanh1" and "arXiv:2405.02246v1 [cs.CV] 3 May 2024" (p. 1).

### 2. One-Sentence Contribution Summary
The paper seeks to identify which design choices matter for VLMs via controlled experiments and uses those findings to build Idefics2, an efficient 8B VLM.
- Evidence: "To address this issue, we conduct extensive experiments around pre-trained models, architecture choice, data, and training methods. Our consolidation of findings includes the development of Idefics2, an efficient foundational VLM of 8 billion parameters." (Abstract).

### 3. Tasks Evaluated
Task list with task type, dataset(s), domain, and evidence.

1) VQAv2
- Task type: Reasoning / relational; Other (visual question answering).
- Dataset: VQAv2.
- Domain: Not specified in the paper.
- Evidence: "VQAv2 (Goyal et al., 2017) for general visual question answering" (Section 3).

2) TextVQA
- Task type: Reasoning / relational; Other (OCR / text reading).
- Dataset: TextVQA.
- Domain: Natural images.
- Evidence: "TextVQA (Singh et al., 2019) for OCR abilities" (Section 3). Also, "(Singh et al., 2019) for text reading on natural images" (Section 4.2).

3) OKVQA
- Task type: Reasoning / relational; Other (VQA with external knowledge).
- Dataset: OKVQA.
- Domain: Not specified in the paper.
- Evidence: "OKVQA (Marino et al., 2019) for external knowledge" (Section 3).

4) COCO (Captioning)
- Task type: Generation (captioning).
- Dataset: COCO.
- Domain: Not specified in the paper.
- Evidence: "COCO (Lin et al., 2014) for captioning." (Section 3).

5) MMMU
- Task type: Reasoning / relational; Other (multidiscipline college-level problems).
- Dataset: MMMU.
- Domain: Not specified in the paper.
- Evidence: "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidis-
discipline college-level problems" (Section 4.2).

6) MathVista
- Task type: Reasoning / relational; Other (mathematical reasoning).
- Dataset: MathVista.
- Domain: Not specified in the paper.
- Evidence: "MathVista (Lu et al., 2024) for mathematical reasoning" (Section 4.2).

7) MMBench
- Task type: Reasoning / relational; Other (perception and reasoning tasks).
- Dataset: MMBench.
- Domain: Not specified in the paper.
- Evidence: "MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2).

8) DocVQA
- Task type: Reasoning / relational; Other (document QA / text extraction).
- Dataset: DocVQA.
- Domain: Images with text (explicitly "text in an image").
- Evidence: "TextVQA and DocVQA, which require a sufficiently high resolution to extract the text in an image" (Section 3.4).

### 4. Domain and Modality Scope
- Modalities: Multiple modalities (image + text inputs, text outputs).
- Evidence: "Vision-language models (VLMs) that take images and texts as inputs and output texts" (Section 1).
- Single domain vs multiple domains within same modality: Not specified in the paper.
- Domain generalization / cross-domain transfer: Not claimed.

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| VQAv2 | Yes (same base model evaluated across tasks). | No per-task fine-tune specified for base evaluation. | Not specified in the paper. | "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1) |
| TextVQA | Yes (base and instruction-tuned models evaluated across tasks). | Base: no per-task fine-tune specified; Instruction: trained on multi-task mixture. | Not specified in the paper. | "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1). Also: "To do so, we create and release The Cauldron8 , a massive collection of 50 vision-language datasets," (Section 4.2) and "We instruction-tune the base model using DoRA (Liu et al., 2024) (a variant of LoRA)." (Section 4.2). |
| OKVQA | Yes (same base model evaluated across tasks). | No per-task fine-tune specified for base evaluation. | Not specified in the paper. | "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1). |
| COCO (Captioning) | Yes (same base model evaluated across tasks). | No per-task fine-tune specified for base evaluation. | Not specified in the paper. | "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1). |
| MMMU | Yes (same instruction-tuned model evaluated across tasks). | Instruction-tuned once on multi-task mixture; no per-task fine-tune stated. | Not specified in the paper. | "(Benchmark, Split, Metric): (MMMU, val/test, MMMU score), (MathVista, testmini, MMMU score)," (Section 4.2/Table 9) and "Idefics2 with 64 or 320 tokens per image is the same model (same weights), only the inference differs." (Section 4.2/Table 9). |
| MathVista | Yes (same instruction-tuned model evaluated across tasks). | Instruction-tuned once on multi-task mixture; no per-task fine-tune stated. | Not specified in the paper. | "(Benchmark, Split, Metric): (MMMU, val/test, MMMU score), (MathVista, testmini, MMMU score)," (Section 4.2/Table 9) and "Idefics2 with 64 or 320 tokens per image is the same model (same weights), only the inference differs." (Section 4.2/Table 9). |
| MMBench | Yes (same instruction-tuned model evaluated across tasks). | Instruction-tuned once on multi-task mixture; no per-task fine-tune stated. | Not specified in the paper. | "MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2) and "Idefics2 with 64 or 320 tokens per image is the same model (same weights), only the inference differs." (Section 4.2/Table 9). |
| DocVQA | Not specified for general evaluation; task-specific fine-tuning is described in OCR ablation. | Yes in OCR ablation (task-specific fine-tune). | Not specified in the paper. | "Instead, we fine-tune the checkpoints on DocVQA for 500 steps with a learning rate of 1e − 5" (Appendix A.1.4). |

### 6. Input and Representation Constraints
- Variable image resolution with explicit maxima per stage.
  - Evidence: "In the first stage, we limit the max image resolution to 384 pixels" and "we increase the resolution to a maximum of 980 pixels." (Section 4.1).
- Aspect ratio preserving (no resizing for that strategy).
  - Evidence: "we pass the image patches to the vision encoder without resizing the image or modifying its aspect ratio." (Section 3.3).
- Visual token count (standard 64; optional higher counts via image splitting).
  - Evidence: "visual tokens (64 in our standard configuration)." (Section 2).
  - Evidence: "Each single image becomes a list of 5 images: 4 crops and the original image. This way, at inference, the model is able to deal with standalone images (64 visual tokens per image), as well as artificially augmented images (320 visual tokens in total per image)." (Section 3.4).
- Resizing requirement for sub-images in image splitting.
  - Evidence: "the sub-images are resized to the original image’s size." (Section 3.4).
- Fixed patch size / fixed number of tokens / fixed dimensionality: Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length values used in pre-training are dataset-dependent.
  - Evidence: "maximum sequence length of 2’048" and "maximum sequence length of 1’536" and "PDF documents represent the remaining 20% of the examples with a maximum sequence length of 1’024." (Section 4.1).
- Fixed vs variable sequence length: Not specified in the paper (only dataset-dependent maxima are given).
- Attention structure described as cross-attention vs fully autoregressive concatenation.
  - Evidence: "In the cross-attention architecture (Alayrac et al., 2022; Laurençon et al., 2023; Awadalla et al., 2023), the images encoded through the vision backbone are injected at different layers within the language model by interleaving cross-attention blocks in which the text cross-attends to the image hidden states." (Section 2).
  - Evidence: "In contrast, in the fully autoregressive architecture (Koh et al., 2023; Driess et al., 2023; Liu et al., 2023), the output of the vision encoder is directly concatenated to the sequence of text embeddings, and the entire sequence is passed as input to the language model." (Section 2).
- Mechanisms to manage compute cost:
  - Learned pooling with perceiver resampler.
    - Evidence: "We reduce the sequence length of each image’s hidden states by using a perceiver resampler (Jaegle et al., 2021; Alayrac et al., 2022; Bai et al., 2023) as a form of trainable Transformer-based pooling." and "reduces the number of visual tokens necessary for each image from 729 to 64" (Section 3.3).
  - Image splitting to trade compute for performance.
    - Evidence: "Splitting images into sub-images during training allow trading compute efficiency for more performance during inference." (Finding 6).

### 8. Positional Encoding (Critical Section)
- Mechanism used: pre-trained positional embeddings from SigLIP are interpolated to support higher resolution; the paper notes they can be interpreted as absolute or relative depending on setup.
  - Evidence: "we interpolate the pre-trained positional embeddings to allow for a higher resolution" (Section 3.3).
  - Evidence: "Since SigLIP is trained with a fixed resolution, the positional embeddings can be interpreted both as absolute or relative positions. With the aspect ratio and resolution preserving, these positions become relative positional embeddings." (Section 3.3, footnote 2).
- Where applied: Not specified in the paper (beyond being in the vision encoder).
- Fixed vs modified vs ablated: Modified for aspect-ratio/resolution preserving via interpolation; no ablations or alternatives stated.

### 9. Positional Encoding as a Variable
- Core research variable? Not specified in the paper.
- Fixed architectural assumption? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- Claim PE is “not critical” or secondary? Not specified in the paper.
- Evidence: The only explicit PE-related statement is about interpolation and interpretation: "we interpolate the pre-trained positional embeddings to allow for a higher resolution" and the footnote on absolute vs relative positions (Section 3.3).

### 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size(s): "Idefics2, a foundational VLM with 8 billion parameters." (Section 1).
- Dataset size(s): "dataset of interleaved image-text documents with 350 million images and 115 billion text tokens." (Section 4.1). Also, "It corresponds to approximately 1.5 billion images and 225 billion text tokens." (Section 4.1).
- Performance gains attributed to stronger backbones: "we observe that the greatest improve- ment in the performance on vision-language benchmarks comes from changing the language model to a better one." (Section 3.1).
- Performance gains attributed to pooling/token reduction: "the learned pooling is effective in two ways: it increases the performance by 8.5 points on average and reduces the number of visual tokens necessary for each image from 729 to 64" (Section 3.3).
- Performance gains attributed to image splitting: "Splitting images into sub-images during training allow trading compute efficiency for more performance during inference." (Finding 6).

### 11. Architectural Workarounds
- Cross-attention vs fully autoregressive fusion (architectural choice for efficiency and performance).
  - Evidence: "In the cross-attention architecture (Alayrac et al., 2022; Laurençon et al., 2023; Awadalla et al., 2023), the images encoded through the vision backbone are injected at different layers within the language model by interleaving cross-attention blocks in which the text cross-attends to the image hidden states." and "In contrast, in the fully autoregressive architecture (Koh et al., 2023; Driess et al., 2023; Liu et al., 2023), the output of the vision encoder is directly concatenated to the sequence of text embeddings, and the entire sequence is passed as input to the language model." (Section 2).
- Learned pooling via perceiver resampler to reduce tokens.
  - Evidence: "We reduce the sequence length of each image’s hidden states by using a perceiver resampler (Jaegle et al., 2021; Alayrac et al., 2022; Bai et al., 2023) as a form of trainable Transformer-based pooling." and "reduces the number of visual tokens necessary for each image from 729 to 64" (Section 3.3).
- Aspect ratio and resolution preserving input processing.
  - Evidence: "we pass the image patches to the vision encoder without resizing the image or modifying its aspect ratio." (Section 3.3).
- Image splitting into sub-images to trade compute for performance.
  - Evidence: "Each single image becomes a list of 5 images: 4 crops and the original image." (Section 3.4).
- Parameter-efficient adaptation for stability.
  - Evidence: "we leverage Low-Rank Adaptation (Hu et al., 2022) to adapt the pre-trained parameters while using standard full fine-tuning for the newly initialized ones." (Section 3.2).
- Instruction-tuning with DoRA.
  - Evidence: "We instruction-tune the base model using DoRA (Liu et al., 2024) (a variant of LoRA)." (Section 4.2).

### 12. Explicit Limitations and Non-Claims
- Limitations / risks from red-teaming (bias and harmful outputs):
  - Evidence: "nuanced contextual understanding, often perpetuating harmful stereotypes. Noteworthy instances include:" (Appendix A.4).
- Security risks identified:
  - Evidence: "Additionally, we identify behaviors that increase security risks that already exist:" followed by examples (Appendix A.4).
- OCR limitation note:
  - Evidence: "these security concerns are currently limited by the model’s occasional inability to accurately read text within images." (Appendix A.4).
- Other explicit limitations: "we acknowledge that the open VLM community is missing a large well-trained vision encoder." (Section 3.1).
- Explicit non-claims (e.g., open-world learning, unrestrained multi-task learning): Not specified in the paper.

### 13. Constraint Profile (Synthesis)
Constraint Profile:
- Domain scope: Multimodal inputs/outputs (images + text), but explicit domain generalization claims are not made.
- Task structure: Multiple benchmark tasks (VQA, OCR/text reading, captioning, reasoning) evaluated with shared models.
- Representation rigidity: Variable resolution with explicit maxima; standard 64 visual tokens with optional 320 via image splitting.
- Model sharing vs specialization: Base and instruction-tuned models are evaluated across tasks; DocVQA has a task-specific fine-tuning in OCR ablation.
- Positional encoding: Interpolated positional embeddings for resolution changes; not treated as a primary experimental variable.

### 14. Final Classification
Multi-task, multi-domain (constrained).
- Justification: The paper evaluates a single VLM across multiple tasks and benchmarks (e.g., VQAv2, TextVQA, OKVQA, COCO, MMMU, MathVista, MMBench, DocVQA) rather than a single task, and the tasks span different application domains within vision-language (VQA, captioning, document/text understanding). Evidence includes: "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1) and "(Benchmark, Split, Metric): (MMMU, val/test, MMMU score), (MathVista, testmini, MMMU score)," and "(TextVQA, val, VQA acc.), (MMBench, test, accuracy)." (Section 4.2/Table 9).
- The evaluation remains constrained to specified benchmarks and does not claim unrestrained multi-domain generalization.

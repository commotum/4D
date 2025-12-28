## 1. Basic Metadata
- Title: Align before Fuse: Vision and Language Representation Learning with Momentum Distillation.
  Evidence: "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation" (Title block).
- Authors: Junnan Li, Ramprasaath R. Selvaraju, Akhilesh D. Gotmare, Shafiq Joty, Caiming Xiong, Steven C.H. Hoi.
  Evidence: "Junnan Li, Ramprasaath R. Selvaraju, Akhilesh D. Gotmare / Shafiq Joty, Caiming Xiong, Steven C.H. Hoi" (Title block).
- Year: 2021.
  Evidence: "arXiv:2107.07651v2 [cs.CV] 7 Oct 2021" (Title block).
- Venue (conference/journal/arXiv): NeurIPS 2021; arXiv preprint.
  Evidence: "35th Conference on Neural Information Processing Systems (NeurIPS 2021), Sydney, Australia." (Title page) and "arXiv:2107.07651v2 [cs.CV] 7 Oct 2021" (Title block).

## 2. One-Sentence Contribution Summary
ALBEF introduces a vision-language pre-training framework that aligns image and text representations with an image-text contrastive loss before cross-modal fusion and uses momentum distillation to learn from noisy web data, improving downstream V+L task performance.

## 3. Tasks Evaluated
- Image-Text Retrieval (TR/IR)
  - Task type: Other (retrieval/ranking).
  - Dataset(s): Flickr30K, COCO.
  - Domain: Not specified in the paper (described as image-text pairs/images only).
  - Evidence: "Image-Text Retrieval contains two subtasks: image-to-text retrieval (TR) and text-to-image retrieval (IR). We evaluate ALBEF on the Flickr30K [49] and COCO benchmarks" (Section 5, Downstream V+L Tasks) and "Image-Text Retrieval. We consider two datasets for this task: COCO and Flickr30K." (Appendix A, Downstream Task Details).
- Visual Entailment (SNLI-VE)
  - Task type: Classification; Reasoning / relational.
  - Dataset(s): SNLI-VE (constructed using SNLI and Flickr30K).
  - Domain: Not specified in the paper (described as image + text only).
  - Evidence: "Visual Entailment (SNLI-VE5 [51]) is a fine-grained visual reasoning task to predict whether the relationship between an image and a text is entailment, neutral, or contradictory. We follow UNITER [2] and consider VE as a three-way classification problem" (Section 5, Downstream V+L Tasks) and "We evaluate on the SNLI-VE dataset [51], which is constructed using the Stanford Natural Language Inference (SNLI) [60] and Flickr30K datasets." (Appendix A, Downstream Task Details).
- Visual Question Answering (VQA)
  - Task type: Generation.
  - Dataset(s): VQA2.0 (constructed using images from COCO); additional QA pairs from Visual Genome for training.
  - Domain: Not specified in the paper (described as image + question only).
  - Evidence: "Visual Question Answering (VQA [52]) requires the model to predict an answer given an image and a question" and "we consider VQA as an answer generation problem" (Section 5, Downstream V+L Tasks) and "We conduct experiment on the VQA2.0 dataset [52], which is constructed using images from COCO... and include additional question-answer pairs from Visual Genome." (Appendix A, Downstream Task Details).
- Natural Language for Visual Reasoning (NLVR2)
  - Task type: Classification; Reasoning / relational.
  - Dataset(s): NLVR2.
  - Domain: Not specified in the paper (described as a pair of images + text).
  - Evidence: "Natural Language for Visual Reasoning (NLVR2 [19]) requires the model to predict whether a text describes a pair of images." (Section 5, Downstream V+L Tasks).
- Visual Grounding (weakly supervised)
  - Task type: Other (visual grounding/localization).
  - Dataset(s): RefCOCO+.
  - Domain: Not specified in the paper (described as image + textual description only).
  - Evidence: "Visual Grounding aims to localize the region in an image that corresponds to a specific textual description. We study the weakly-supervised setting, where no bounding box annotations are available. We perform experiments on the RefCOCO+ [56] dataset" (Section 5, Downstream V+L Tasks) and "We conduct experiments on the RefCOCO+ dataset [56]" (Appendix A, Downstream Task Details).

## 4. Domain and Modality Scope
- Multiple modalities: Yes (vision + language).
  Evidence: "Vision-and-Language Pre-training (VLP) aims to learn multimodal representations from large-scale image-text pairs" (Section 1, Introduction).
- Single domain vs multiple domains within the same modality: Not explicitly specified; the evaluation spans multiple datasets but the paper does not label these as distinct domains.
  Evidence: "We evaluate ALBEF on the Flickr30K [49] and COCO benchmarks" and the additional task datasets listed in Section 5 and Appendix A (Downstream V+L Tasks; Appendix A).
- Domain generalization or cross-domain transfer claims: Not claimed.

## 5. Model Sharing Across Tasks
The paper adapts a pre-trained model to each downstream task with task-specific fine-tuning; it does not describe joint multi-task training with shared weights across tasks.
Evidence: "We adapt the pre-trained model to five downstream V+L tasks. We introduce each task and our fine-tuning strategy below." (Section 5, Downstream V+L Tasks).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image-Text Retrieval | No (task-specific fine-tuning from shared pre-trained init) | Yes | Yes (ITM classifier head used) | "We evaluate ALBEF on the Flickr30K [49] and COCO benchmarks, and fine-tune the pre-trained model using the training samples from each dataset. During fine-tuning, we jointly optimize the ITC loss (Equation 2) and the ITM loss (Equation 4)." (Section 5) and "append a fully-connected (FC) layer followed by softmax to predict a two-class probability pitm" (Section 3.2, Pre-training Objectives). |
| Visual Entailment (SNLI-VE) | No (task-specific fine-tuning from shared pre-trained init) | Yes | Yes (MLP classifier) | "We follow UNITER [2] and consider VE as a three-way classification problem, and predict the class probabilities using a multi-layer perceptron (MLP) on the multimodal encoder’s representation of the [CLS] token." (Section 5) and "We fine-tune the pre-trained model for 5 epochs" (Appendix A). |
| VQA | No (task-specific fine-tuning from shared pre-trained init) | Yes | Yes (answer decoder) | "we consider VQA as an answer generation problem... we use a 6-layer transformer decoder to generate the answer" (Section 5) and "The answer decoder is initialized using the pre-trained weights from the multimodal encoder, and finetuned with a conditional language-modeling loss." (Section 5) and "We fine-tune the model for 8 epochs" (Appendix A). |
| NLVR2 | No (task-specific fine-tuning from shared pre-trained init) | Yes | Yes (MLP classifier; multimodal encoder replicated) | "We extend our multimodal encoder to enable reasoning over two images... each layer of the multimodal encoder is replicated... We append a MLP classifier on the multimodal encoder’s [CLS] representation for prediction." (Section 5) and "We fine-tune the model for 10 epochs" (Appendix A). |
| Visual Grounding | No (task-specific fine-tuning from shared pre-trained init) | Yes | No explicit new head; inference uses Grad-CAM ranking | "We perform experiments on the RefCOCO+ [56] dataset, and fine-tune the model using only image-text supervision following the same strategy as image-text retrieval." (Section 5) and "During inference, we extend Grad-CAM [9] to acquire heatmaps, and use them to rank the detected proposals" (Section 5). |

## 6. Input and Representation Constraints
- Input resolution during pre-training: 256 × 256 random crops.
  Evidence: "During pre-training, we take random image crops of resolution 256 × 256 as input" (Section 3.5, Implementation Details).
- Input resolution during fine-tuning/downstream: 384 × 384.
  Evidence: "During fine-tuning, we increase the image resolution to 384 × 384" (Section 3.5, Implementation Details) and "All downstream tasks receive input images of resolution 384 × 384." (Appendix A).
- Inference resizing/cropping: resize without cropping.
  Evidence: "During inference, we resize the images without any cropping." (Appendix A).
- Patch-based image representation is used, but fixed patch size is not explicitly stated.
  Evidence: "We use a 12-layer visual transformer ViT-B/16 [38] as the image encoder" and "interpolate the positional encoding of image patches" (Section 3.1; Section 3.5).
- Sequence/token representations for both modalities are used; fixed number of tokens is not specified.
  Evidence: "An input image I is encoded into a sequence of embeddings: {v cls , v 1 , ..., v N }" and "The text encoder transforms an input text T into a sequence of embeddings {wcls , w1 , ..., wN }" (Section 3.1, Model Architecture).
- Fixed dimensionality (e.g., strictly 2D) and padding requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed vs variable sequence length: Not specified in the paper.
- Attention type: Transformer self-attention and cross-attention; no windowed/hierarchical/sparse attention is described.
  Evidence: "The image features are fused with the text features through cross attention at each layer of the multimodal encoder." (Section 3.1) and "each block contains a self-attention layer, a cross-attention layer, and a feed-forward layer" (Section 5, NLVR2 description).
- Mechanisms for computational cost:
  - Detector-free and lower-resolution images.
    Evidence: "Unlike most existing methods, our method does not require bounding box annotations nor high-resolution images." (Abstract) and "ALBEF is detector-free and requires lower resolution images, it also enjoys much faster inference speed compared to most existing methods" (Section 6.3).
  - Retrieval inference uses top-k filtering to reduce scoring cost.
    Evidence: "Because k can be set to be very small, our inference speed is much faster than methods that require computing the ITM score for all image-text pairs" (Section 5, Image-Text Retrieval).

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism type (absolute/relative/RoPE/etc.): Not specified in the paper.
- Where applied: Positional encoding is applied to image patches (details beyond this are not specified).
  Evidence: "interpolate the positional encoding of image patches" (Section 3.5, Implementation Details).
- Fixed vs modified: Modified when changing resolution during fine-tuning via interpolation.
  Evidence: "During fine-tuning, we increase the image resolution to 384 × 384 and interpolate the positional encoding of image patches" (Section 3.5).
- Ablations/alternatives: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Treated as a core research variable: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- Claim that positional encoding choice is “not critical” or secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size(s):
  Evidence: "Our model consists of a BERTbase with 123.7M parameters and a ViT-B/16 with 85.8M parameters." (Section 3.5, Implementation Details).
- Dataset size(s):
  Evidence: "The total number of unique images is 4.0M, and the number of image-text pairs is 5.1M... include the much noisier Conceptual 12M dataset [43], increasing the total number of images to 14.1M" (Section 3.4, Pre-training Datasets).
- Scaling data as a driver of gains:
  Evidence: "Given the considerable amount of improvement of ALBEF when the number of training images increases from 4M to 14M, we hypothesize that it has potential to further grow by training on larger-scale web image-text pairs." (Section 6.2).
- Architectural/training tricks credited for gains (ITC, hard negatives, MoD):
  Evidence: "adding ITC substantially improves the pre-trained model’s performance across all tasks" and "adding momentum distillation improves learning for both ITC... MLM... and on all downstream tasks" (Section 6.1).
- Scaling model size: Not specified in the paper.

## 11. Architectural Workarounds
- Detector-free image encoder and cross-modal fusion after alignment:
  Evidence: "We first encode the image and text independently with a detector-free image encoder and a text encoder. Then we use a multimodal encoder to fuse the image features with the text features through cross-modal attention." (Section 1, Introduction).
- Contrastive alignment before fusion (ITC):
  Evidence: "We introduce an intermediate image-text contrastive (ITC) loss on representations from the unimodal encoders" (Section 1, Introduction).
- Momentum distillation to handle noisy web data:
  Evidence: "To improve learning from noisy web data, we propose momentum distillation, a self-training method which learns from pseudo-targets produced by a momentum model." (Abstract).
- Hard-negative mining for ITM with low overhead:
  Evidence: "We propose a strategy to sample hard negatives for the ITM task with zero computational overhead." (Section 3.2, Pre-training Objectives).
- Task-specific architectural heads:
  - VQA decoder: "we use a 6-layer transformer decoder to generate the answer" (Section 5).
  - VE classifier: "predict the class probabilities using a multi-layer perceptron (MLP)" (Section 5).
  - NLVR2 classifier and encoder replication: "each layer of the multimodal encoder is replicated... We append a MLP classifier" (Section 5).
- Retrieval inference acceleration (top-k filtering):
  Evidence: "Then we take the top-k candidates and calculate their ITM score sitm for ranking. Because k can be set to be very small, our inference speed is much faster" (Section 5).
- Visual grounding via Grad-CAM ranking:
  Evidence: "During inference, we extend Grad-CAM [9] to acquire heatmaps, and use them to rank the detected proposals" (Section 5).

## 12. Explicit Limitations and Non-Claims
- Social impact / deployment caution:
  Evidence: "additional analysis on the data and the model is necessary before deploying it in practice, because web data may contain unintended private information, unsuitable images, or harmful texts, and only optimizing accuracy may have unwanted social implications." (Section 7, Conclusion and Social Impacts).
- SNLI-VE evaluation caveat:
  Evidence: "results on SNLI-VE should be interpreted with caution because its test data has been reported to be noisy" (Section 5, footnote).
- Data contamination / future work on decontamination:
  Evidence: "Strictly speaking, our model is not allowed to see the val/test images of RefCOCO+, but it has been exposed to those images during pre-training... leave it as future work to decontaminate the data." (Appendix A, Visual Grounding).
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

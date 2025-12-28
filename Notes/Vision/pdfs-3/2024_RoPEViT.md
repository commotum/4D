## 1. Basic Metadata
- Title: Rotary Position Embedding for Vision Transformer. Evidence: "Rotary Position Embedding for Vision Transformer" (p.1, title).
- Authors: Byeongho Heo; Song Park; Dongyoon Han; Sangdoo Yun. Evidence: "Byeongho Heo        Song Park       Dongyoon Han         Sangdoo Yun" (p.1, author line).
- Year: 2024. Evidence: "arXiv:2403.13298v2 [cs.CV] 16 Jul 2024" (p.1).
- Venue: arXiv (preprint). Evidence: "arXiv:2403.13298v2 [cs.CV] 16 Jul 2024" (p.1).

## 2. One-Sentence Contribution Summary
This paper introduces and evaluates 2D RoPE (including RoPE-Mixed with mixed-axis learnable frequencies) for ViT and Swin to improve positional embedding and multi-resolution vision performance.
- Evidence: "This study provides a comprehensive analysis of RoPE when applied to ViTs, utilizing practical implementations of RoPE for 2D vision data." (p.1, Abstract)
- Evidence: "This paper aims to improve position embedding for vision transformers by applying an extended Rotary Position Embedding (RoPE) [29]." (p.2, Introduction)
- Evidence: "we propose to use mixed axis frequencies for 2D RoPE, named RoPE-Mixed." (p.2, Introduction)

## 3. Tasks Evaluated
- Multi-resolution classification
  - Task type: Classification
  - Dataset(s): ImageNet-1k
  - Domain: natural images
  - Evidence: "4.1             Multi-resolution classification" (p.10, Section 4.1)
  - Evidence: "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (p.10, Section 4.1)
- Object detection
  - Task type: Detection
  - Dataset(s): MS-COCO
  - Domain: natural images
  - Evidence: "4.2   Object detection" (p.12, Section 4.2)
  - Evidence: "We verify 2D RoPE in object detection on MS-COCO [16]." (p.12, Section 4.2)
- Semantic segmentation
  - Task type: Segmentation
  - Dataset(s): ADE20k
  - Domain: natural images
  - Evidence: "4.3     Semantic segmentation" (p.13, Section 4.3)
  - Evidence: "We train 2D RoPE ViT and Swin for semantic segmentation on ADE20k [40, 41]." (p.13, Section 4.3)

## 4. Domain and Modality Scope
- Single domain vs multiple domains (same modality): Multiple datasets within the same modality (vision) are evaluated: ImageNet-1k, MS-COCO, ADE20k. Evidence: "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (p.10); "We verify 2D RoPE in object detection on MS-COCO [16]." (p.12); "We train 2D RoPE ViT and Swin for semantic segmentation on ADE20k [40, 41]." (p.13)
- Multiple modalities: Not specified in the paper.
- Domain generalization or cross-domain transfer: Not claimed in the paper.

## 5. Model Sharing Across Tasks
The paper describes separate task training sections. Detection and segmentation explicitly use ImageNet-1k pre-trained backbones.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Multi-resolution classification (ImageNet-1k) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "4.1             Multi-resolution classification" and "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (p.10, Section 4.1) |
| Object detection (MS-COCO) | Backbone pre-trained on ImageNet-1k. | Not specified in the paper (pre-trained backbone stated). | Yes (DINO detector). | "RoPE is applied to the backbone ViT, which is pre-trained on ImageNet-1k with DeiT-III 400epochs recipe." (p.12, Table 1 caption); "DINO [39] detector is trained using ViT and Swin as backbone network." (p.12, Section 4.2) |
| Semantic segmentation (ADE20k) | Backbone pre-trained on ImageNet-1k. | Not specified in the paper (pre-trained backbone stated). | Yes (UperNet / Mask2Former). | "ImageNet-1k pre-trained weights from §4.1 are used for pre-trained weights." and "For ViT, we use UperNet [37] with ViT training recipe [21]. For Swin, Mask2Former [2] for segmentation is used with the Swin." (p.13, Section 4.3) |

## 6. Input and Representation Constraints
- Patch size: "APE is generally added to the feature right after the patchification layer computes tokens from 16 × 16 or 32 × 32 patch images." (p.4, Section 3.1)
- 2D token grid / fixed dimensionality (2D): "pn = (pxn , pyn ) where pxn ∈ {0, 1, ..., W }, pyn ∈ {0, 1, ..., H} for token width W and height H." (p.6, Section 3.2)
- Training resolution (classification): "Note that we use the ImageNet-1k standard image resolution 224 × 224 for training." (p.10, Section 4.1)
- Variable input resolution at evaluation: "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (p.10, Section 4.1)
- Segmentation input resolution: "the ViT-UperNet setting uses 512 × 512 images for inputs." (p.13, Section 4.3)
- Fixed number of tokens: Not specified in the paper.
- Padding or resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Sequence length fixed or variable: Not specified explicitly; varying image sizes are evaluated. Evidence: "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (p.10, Section 4.1)
- Attention type:
  - Windowed attention: "For multi-resolution inference, we change the window size of the window attention." (p.11, Fig. 5 caption)
  - Hierarchical architecture: "Swin Transformer 2D RoPE variants are applied to Swin Transformers, a milestone work in hierarchical ViT with relative position embedding RPB." (p.11, Section 4.1)
  - Window-block + global attention (ViTDet): "DINO-ViTDet uses ViT backbone with window-block attention, but still, a few layers remain as global attention." (p.12, Section 4.2)
- Mechanisms to manage computational cost:
  - Pooling in hierarchical ViT: "Hierarchical ViT such as Swin Transformer [17] increase the spatial length of tokens at early layers using pooling." (p.2, Related Works)
  - RoPE compute overhead minimized: "The rotation matrix in Eq. 12 and 14 is pre-computed before inference." (p.9, Section 3.3)

## 8. Positional Encoding (Critical Section)
- Mechanisms used and where applied:
  - APE (absolute): "APE is generally added to the feature right after the patchification layer computes tokens from 16 × 16 or 32 × 32 patch images." (p.4, Section 3.1)
  - RPB (relative bias): "While APE is added to network features, RPB is directly applied to the attention matrix of every self-attention layer since it is the only position that can handle relative relations in transformer architecture." (p.4, Section 3.1)
  - RoPE (relative, rotary): "Rotary Position Embedding (RoPE) [29] was introduced to apply to key and query in self-attention layers as channel-wise multiplications, which is distinct from conventional position embeddings - APE is added to the stem layer; RPB is added to an attention matrix." (p.3, Section 3)
  - 2D RoPE variants: "This section introduces feasible implementations of 2D RoPE for input images: axial and learnable frequency." (p.6, Section 3.2)
- Where applied:
  - RoPE is applied to query and key in self-attention layers. Evidence: "Rotary Position Embedding (RoPE) [29] was introduced to apply to key and query in self-attention layers as channel-wise multiplications, which is distinct from conventional position embeddings - APE is added to the stem layer; RPB is added to an attention matrix." (p.3, Section 3)
- Fixed vs modified / ablated across experiments:
  - Multiple positional encodings are compared: "We compare the conventional position embeddings (APE, RPB) with two variants of 2D RoPE RoPE-Axial (Eq. 12) and RoPE-Mixed (Eq. 14)." (p.9, Section 4)
  - Combinations are tested: "We measure the performance of RoPE-Mixed when it is used with APE." (p.10, Section 4.1) and "We also measure performance when RoPE-Mixed is used together with RPB." (p.12, Section 4.2)

## 9. Positional Encoding as a Variable
- Core research variable: Yes; the study centers on positional embedding comparisons. Evidence: "We compare the conventional position embeddings (APE, RPB) with two variants of 2D RoPE RoPE-Axial (Eq. 12) and RoPE-Mixed (Eq. 14)." (p.9, Section 4)
- Multiple positional encodings compared: APE, RPB, RoPE-Axial, RoPE-Mixed, RoPE+APE, RoPE+RPB. Evidence: "We compare the conventional position embeddings (APE, RPB) with two variants of 2D RoPE RoPE-Axial (Eq. 12) and RoPE-Mixed (Eq. 14)." (p.9); "We measure the performance of RoPE-Mixed when it is used with APE." (p.10); "We also measure performance when RoPE-Mixed is used together with RPB." (p.12)
- Claim that PE choice is not critical or secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model sizes evaluated: "We apply 2D RoPE to ViT-S, ViT-B, and ViT-L." (p.10, Section 4.1) and "We train Swin-T, Swin-S, and Swin-B on ImageNet-1k with 300 epochs of Swin Transformer training recipe [17]." (p.11, Section 4.1)
- Dataset sizes: Not specified in the paper.
- Attribution of gains (not scaling data/model size):
  - "The results show that 2D RoPE is a beneficial option for position embedding in transformers with impressive performance improvements on high-resolution images, i.e., extrapolation of images." (p.2, Introduction)
  - "We believe that RoPE is highly effective due to the extrapolation in global attention." (p.12, Section 4.2)
  - "The improvement might originate from the extrapolation performance of RoPE since the ViT-UperNet setting uses 512 × 512 images for inputs." (p.13, Section 4.3)
- Scaling model size or data as primary driver: Not claimed in the paper.
- Training recipes (context only): "All ViTs are trained on ImageNet-1k [4] with DeiT-III [32]’s 400 epochs training recipe." (p.10, Fig. 4 caption)

## 11. Architectural Workarounds
- Windowed attention (Swin): "For multi-resolution inference, we change the window size of the window attention." (p.11, Fig. 5 caption)
- Hierarchical stages with pooling (Swin): "Hierarchical ViT such as Swin Transformer [17] increase the spatial length of tokens at early layers using pooling." (p.2, Related Works)
- Window-block attention + residual global attention (ViTDet): "DINO-ViTDet uses ViT backbone with window-block attention, but still, a few layers remain as global attention." (p.12, Section 4.2)
- Task-specific heads:
  - Detection head: "DINO [39] detector is trained using ViT and Swin as backbone network." (p.12, Section 4.2)
  - Segmentation heads: "For ViT, we use UperNet [37] with ViT training recipe [21]. For Swin, Mask2Former [2] for segmentation is used with the Swin." (p.13, Section 4.3)
- Fixed grid/patchification: "patchification layer computes tokens from 16 × 16 or 32 × 32 patch images." (p.4, Section 3.1)
- Low-overhead RoPE computation: "The rotation matrix in Eq. 12 and 14 is pre-computed before inference." (p.9, Section 3.3)

## 12. Explicit Limitations and Non-Claims
Not specified in the paper.

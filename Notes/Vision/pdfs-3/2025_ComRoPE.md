## 1. Basic Metadata
Title: ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices
Evidence (Title): "ComRoPE: Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices" (Title page)
Authors: Hao Yu; Tangyu Jiang; Shuning Jia; Shannan Yan; Shunning Liu; Haolong Qian; Guanghao Li; Shuting Dong; Huaisong Zhang; Chun Yuan
Evidence (Authors): "Hao Yu1 Tangyu Jiang1† Shuning Jia1,2 Shannan Yan1 Shunning Liu1 Haolong Qian1 Guanghao Li1 Shuting Dong1 Huaisong Zhang1 Chun Yuan1†" (Title page)
Year: 2025
Venue: arXiv
Evidence (Year/Venue): "arXiv:2506.03737v1 [cs.CV] 4 Jun 2025" (Title page)

## 2. One-Sentence Contribution Summary
ComRoPE is a trainable RoPE formulation using commuting angle matrices to make positional encoding scalable and robust to position offsets in Transformers.
Evidence: "In this work, we propose ComRoPE, which generalizes RoPE by defining it in terms of trainable commuting angle matrices." (Abstract)
Evidence: "we demonstrate that pairwise commutativity of these matrices is essential for RoPE to achieve scalability and positional robustness." (Abstract)

## 3. Tasks Evaluated
Task 1: 2D image classification
- Task type: Classification
- Dataset(s): ImageNet-1K
- Domain: natural images
- Evidence: "We first assess their scalability in 2D image classification across different resolutions." (Section 4, Experiments)
- Evidence: "All models are trained at a standard resolution of 224 × 224 and evaluated across multiple resolutions to test their robustness and scalability on the ImageNet-1K dataset [5]." (Section 4.1.1, Setup)
- Evidence (additional setting on ImageNet): "we fine-tune the Vision Transformer pre-trained in CLIP [27] on ImageNet by simply replacing the standard attention mechanism with each RoPE method." (Appendix B.2)

Task 2: Object detection
- Task type: Detection
- Dataset(s): MS COCO
- Domain: natural images
- Evidence: "Additionally, we conduct object detection experiments to demonstrate the generalizability of our approach." (Section 4, Experiments)
- Evidence: "We evaluate ComRoPE-LD, LieRE, and APE on the MS COCO dataset [17]." (Section 4.2, Object detection)

Task 3: 3D classification
- Task type: Classification
- Dataset(s): UCF-101
- Domain: video
- Evidence: "To further examine the ability to handle higher-dimensional data, we perform 3D classification experiments, which are detailed in Appendix B." (Section 4, Experiments)
- Evidence: "we conduct a 3D classification task on UCF-101 [31]." (Appendix B.1)

## 4. Domain and Modality Scope
Single domain? No; evaluation spans natural images and video.
Evidence: "ImageNet-1K dataset [5]." (Section 4.1.1, Setup)
Evidence: "MS COCO dataset [17]." (Section 4.2, Object detection)
Evidence: "we conduct a 3D classification task on UCF-101 [31]." (Appendix B.1)
Multiple domains within the same modality? Yes; all datasets are visual (images and video). Evidence: same as above.
Multiple modalities? Not specified in the paper.
Domain generalization or cross-domain transfer? Not specified in the paper.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 2D image classification (ImageNet-1K) | Not specified in the paper. | Main experiments: trained from scratch; separate fine-tuning experiment on ImageNet is reported. | Not specified in the paper. | "The models are trained from scratch using randomly initialized parameters, ensuring no influence from pre-trained weights or external priors." (Section 4.1.1, Setup); "we fine-tune the Vision Transformer pre-trained in CLIP [27] on ImageNet..." (Appendix B.2) |
| Object detection (MS COCO) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We adopt ViT-S as our backbone and apply ComRoPE to the attention layers. To ensure consistency with the pre-trained model, we initialize the angle matrix to zero." (Section 4.2, Object detection) |
| 3D classification (UCF-101) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "we conduct a 3D classification task on UCF-101 [31]. The details of the model and configuration can be found in Appendix C." (Appendix B.1) |

## 6. Input and Representation Constraints
- Training resolution fixed at 224 × 224, evaluation across multiple resolutions. Evidence: "All models are trained at a standard resolution of 224 × 224 and evaluated across multiple resolutions..." (Section 4.1.1, Setup)
- Fixed patch size in 2D classification experiments: Patch Size 16. Evidence: "Patch Size 16" (Table 5, Appendix C.1)
- Fixed image size in 2D classification experiments: Image Size 224. Evidence: "Image Size 224" (Table 5, Appendix C.1)
- 3D classification uses fixed image size, frame count, and patch size in the reported configuration. Evidence: "Image Size 224", "Frame Count 8", "Patch Size 16" (Table 6, Appendix C.2)
- Coordinates are normalized to [0,1] per axis in images. Evidence: "For an image with shape H × W, we scale both the height H and the width W to 1. Therefore, for a pixel located at (h, w) in the raw image, its coordinate is treated as (h/H, w/W)." (Section 3.4.1, Relative scaling and center offset)
- Patch representation uses the patch center as the aggregation location. Evidence: "we adopt the center point of a patch as the aggregation location." (Section 3.4.1, Relative scaling and center offset)
- Position perturbation during training is explicitly applied to patch coordinates. Evidence: "we add perturbations to the coordinates of the patches." (Section 3.4.2, Position perturbation)
- Data augmentation includes resizing and random cropping. Evidence: "we apply only basic data augmentation techniques, such as resizing and random cropping..." (Section 4.1.1, Setup)
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality (strictly 2D)? Not specified in the paper. Evidence: "we perform 3D classification experiments..." (Section 4, Experiments)

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type: Self-attention with RoPE applied in attention layers. Evidence: "self-attention layers are replaced with RoPE self-attention parameterized by angle matrices." (Section 4.1.1, Setup)
- Computational cost management: block size capped to limit complexity. Evidence: "Therefore, we limit the block size to a maximum of 8 to balance performance with additional costs." (Section 4.3.2, Impact of block size)

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: RoPE parameterized by trainable commuting angle matrices (ComRoPE), i.e., a relative positional encoding applied in attention. Evidence: "we propose ComRoPE, which generalizes RoPE by defining it in terms of trainable commuting angle matrices." (Abstract)
- Where it is applied: attention layers. Evidence: "self-attention layers are replaced with RoPE self-attention parameterized by angle matrices." (Section 4.1.1, Setup)
- Fixed vs modified/compared: multiple positional encodings are compared and block size is varied. Evidence: "We evaluate our proposed methods (ComRoPE-LD and ComRoPE-AP) against APE, vanilla RoPE (as introduced by RoFormer), and LieRE." (Section 4.1.1, Setup); "varying the block size from 2 to 8." (Section 4.3.2, Impact of block size)

## 9. Positional Encoding as a Variable
- Core research variable? Yes; the paper introduces ComRoPE and compares against other positional encodings. Evidence: "In this work, we propose ComRoPE, which generalizes RoPE by defining it in terms of trainable commuting angle matrices." (Abstract); "We evaluate our proposed methods (ComRoPE-LD and ComRoPE-AP) against APE, vanilla RoPE (as introduced by RoFormer), and LieRE." (Section 4.1.1, Setup)
- Multiple positional encodings compared? Yes. Evidence: same as above.
- Claims that PE is not critical or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking
- Model sizes/architectures reported: "we utilize a standard Vision Transformer (ViT-B/16) architecture" (Section 4.1.1, Setup); "We adopt ViT-S as our backbone..." (Section 4.2, Object detection); "Layers 12" and "Hidden Dimension 768" (Table 5, Appendix C.1); "Layers 8" and "Hidden Dimension 384" (Table 6, Appendix C.2).
- Dataset sizes: Not specified in the paper.
- Performance gains attributed to commutativity/angle-matrix design rather than scale: "ComRoPE-LD and ComRoPE-AP exhibit a more gradual decrease in performance, thanks to their commutative properties that enhance positional robustness." (Section 4.1.2); "These findings illustrate the effectiveness of trainable commutative angle matrices..." (Section 4.1.2)
- Scaling model size or data as primary driver? Not specified in the paper.

## 11. Architectural Workarounds
- Block-diagonal construction to ensure commutativity. Evidence: "Note that if two matrices are both block diagonal with the same block sizes, where the corresponding blocks are commutative, then these two matrices are commutative." (Section 3.3)
- Block size capped to control complexity. Evidence: "Therefore, we limit the block size to a maximum of 8 to balance performance with additional costs." (Section 4.3.2)
- Position perturbation for robustness across scales. Evidence: "we add perturbations to the coordinates of the patches." (Section 3.4.2)
- Relative coordinate scaling and center-offset assumption. Evidence: "we scale both the height H and the width W to 1." and "we adopt the center point of a patch as the aggregation location." (Section 3.4.1)
- Windowed attention, hierarchical stages, token pooling/merging, task-specific heads? Not specified in the paper.

## 12. Explicit Limitations and Non-Claims
- Limitations: "Our implementation depends on torch.matrix exp, which is slow and memory-intensive on large models." (Section H, Limitations); "the other is strict commutativity restrictions. We currently require relatively strong conditions for the angle matrices to commute, which may restrict the expressiveness of the resulting embeddings." (Section H, Limitations)
- Future work: "Identifying weaker—yet still sufficient—conditions could broaden the method’s capacity and applicability." (Section H, Limitations)
- Explicit non-claims (e.g., open-world learning, unrestrained multi-task learning): Not specified in the paper.

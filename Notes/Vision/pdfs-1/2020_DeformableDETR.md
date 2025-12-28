### 1. Basic Metadata
Title: Deformable DETR: Deformable Transformers for End-to-End Object Detection.
Authors: Xizhou Zhu; Weijie Su; Lewei Lu; Bin Li; Xiaogang Wang; Jifeng Dai.
Year: 2021.
Venue: ICLR (conference).

Evidence:
> "D EFORMABLE DETR: D EFORMABLE T RANSFORMERS
> FOR E ND - TO -E ND O BJECT D ETECTION" (Cover page)
> "Xizhou Zhu1∗ , Weijie Su2∗ ‡ , Lewei Lu1 , Bin Li2 , Xiaogang Wang1,3 , Jifeng Dai1†" (Cover page)
> "Published as a conference paper at ICLR 2021" (Cover page)

### 2. One-Sentence Contribution Summary
The paper proposes Deformable DETR to address DETR's slow convergence and limited feature spatial resolution by using deformable attention that attends to a small set of sampling points around a reference.

Evidence:
> "it suffers from slow convergence and limited feature spatial resolution, due to the limitation of Transformer attention modules in processing image feature maps." (Abstract)
> "we proposed Deformable DETR, whose attention modules only attend to a small set of key sampling points around a reference." (Abstract)

### 3. Tasks Evaluated
Task: Object detection.
Task type: Detection.
Dataset(s): COCO 2017 (train/val/test-dev).
Domain: Image domain (COCO 2017); no other domain described.

Evidence:
> "Deformable DETR is an end-to-end object detector, which is efficient and fast-converging." (Section 6 Conclusion)
> "Dataset. We conduct experiments on COCO 2017 dataset (Lin et al., 2014). Our models are trained on the train set, and evaluated on the val set and test-dev set." (Section 5 Experiment)
> "In the image domain, where the key elements are usually of image pixels, Nk can be very large and the convergence is tedious." (Section 3 Revisiting Transformers and DETR)

### 4. Domain and Modality Scope
Evaluation domain: Single domain (COCO 2017 object detection); no other datasets reported.
Multiple domains within the same modality: Not specified in the paper.
Multiple modalities: Not specified; the paper operates in the image domain.
Domain generalization or cross-domain transfer: Not claimed.

Evidence:
> "Dataset. We conduct experiments on COCO 2017 dataset (Lin et al., 2014). Our models are trained on the train set, and evaluated on the val set and test-dev set." (Section 5 Experiment)
> "In the image domain, where the key elements are usually of image pixels, Nk can be very large and the convergence is tedious." (Section 3 Revisiting Transformers and DETR)

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Object detection (COCO 2017) | Single-task model; no cross-task sharing described | Backbone pre-trained on ImageNet; trained on COCO | Yes; regression + classification branches in the detection head | "ImageNet (Deng et al., 2009) pre-trained ResNet-50 (He et al., 2016) is utilized as the backbone for ablations." (Section 5 Experiment); "A 3-layer feed-forward neural network (FFN) and a linear projection are added on top of the object query features (produced by the decoder) as the detection head." (Section 3 DETR); "The FFN acts as the regression branch to predict the bounding box coordinates" (Section 3 DETR); "The linear projection acts as the classification branch to produce the classification results." (Section 3 DETR) |

### 6. Input and Representation Constraints
Fixed or variable input resolution: Not specified in the paper.
Fixed patch size: Not specified in the paper.
Fixed number of tokens: Object query count is fixed in experiments (increased from 100 to 300).
Fixed dimensionality: Uses 2-d reference points and pixel-based feature maps.
Other explicit representation constraints: Multi-scale feature maps with the same resolutions; all feature maps use C = 256 channels; multi-scale feature maps extracted without FPN.

Evidence:
> "Both the input and output of the encoder are of multi-scale feature maps with the same resolutions." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "All the multi-scale feature maps are of C = 256 channels." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "Both the key and query elements are of pixels from the multi-scale feature maps." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "a 2-d reference point" (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "the number of object queries is increased from 100 to 300." (Section 5 Experiment)
> "Multi-scale feature maps are extracted without FPN (Lin et al., 2017a)." (Section 5 Experiment)

### 7. Context Window and Attention Structure
Maximum sequence length: Not specified in the paper; object query count is fixed in experiments and each query samples K points.
Fixed or variable sequence length: Object query count fixed; feature-map size not specified.
Attention type: Sparse, deformable attention over a small set of sampling points; multi-scale aggregation.
Mechanisms to manage computational cost: small fixed number of keys per query; K = 4 by default.

Evidence:
> "the deformable attention module only attends to a small set of key sampling points around a reference point, regardless of the spatial size of the feature maps, as shown in Fig. 2." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "By assigning only a small fixed number of keys for each query, the issues of convergence and feature spatial resolution can be mitigated." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "M = 8 and K = 4 are set for deformable attentions by default." (Section 5 Experiment)
> "the number of object queries is increased from 100 to 300." (Section 5 Experiment)
> "The module can be naturally extended to aggregating multi-scale features, without the help of FPN (Lin et al., 2017a)." (Section 1 Introduction)

### 8. Positional Encoding (Critical Section)
Positional encoding mechanism: Positional embeddings are used, but the paper does not specify absolute vs relative or other types.
Where applied: Positional embeddings are used on input feature maps and on object queries; a scale-level embedding is added in the encoder; in the two-stage variant, object query positional embeddings are set from region proposal coordinates.
Fixed across experiments or modified per task: Not specified in the paper.

Evidence:
> "The inputs are of ResNet feature maps (with encoded positional embeddings)." (Section 3 Revisiting Transformers and DETR)
> "N object queries represented by learnable positional embeddings (e.g., N = 100)." (Section 3 Revisiting Transformers and DETR)
> "To identify which feature level each query pixel lies in, we add a scale-level embedding, denoted as el , to the feature representation, in addition to the positional embedding." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "the positional embeddings of object queries are set as positional embeddings of region proposal coordinates." (Appendix A.4 More Implementation Details)

### 9. Positional Encoding as a Variable
Core research variable: Not specified in the paper.
Multiple positional encodings compared: Not specified in the paper.
Claims that positional encoding is not critical or secondary: Not specified in the paper.

### 10. Evidence of Constraint Masking
Model size(s): Parameter counts and FLOPs are reported; multiple backbones are evaluated.
Dataset size(s): Not specified in the paper (dataset name only).
Primary attribution of gains: Improvements are attributed to deformable attention modules and architectural variants like iterative refinement and two-stage design.

Evidence:
> "Deformable DETR                         50   43.8   62.6   47.7   26.4   47.1   58.0   40M   173G       325        19" (Table 1)
> "Deformable DETR                        ResNet-50           46.9 66.4   50.8   27.7 49.7   59.9" (Table 3)
> "Deformable DETR                        ResNet-101          48.7 68.1   52.9   29.1 51.5   62.0" (Table 3)
> "With the aid of iterative bounding box refinement and two-stage paradigm, our method can further improve the detection accuracy." (Section 5.1 Comparison with DETR)
> "At the core of Deformable DETR are the (multi-scale) deformable attention modules, which is an efficient attention mechanism in processing image feature maps." (Section 6 Conclusion)

### 11. Architectural Workarounds
- Deformable attention samples a small set of points around a reference point to avoid full attention over all spatial locations.
- Multi-scale aggregation is built into the attention module instead of using FPNs.
- Decoder cross-attention is replaced with multi-scale deformable attention.
- Iterative bounding box refinement across decoder layers.
- Two-stage variant uses an encoder-only proposal generator; no NMS before the second stage.
- Multi-scale feature maps are extracted without FPN.

Evidence:
> "the deformable attention module only attends to a small set of key sampling points around a reference point, regardless of the spatial size of the feature maps, as shown in Fig. 2." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "The module can be naturally extended to aggregating multi-scale features, without the help of FPN (Lin et al., 2017a)." (Section 1 Introduction)
> "we only replace each cross-attention module to be the multi-scale deformable attention module, while leaving the self-attention modules unchanged." (Section 4.1 Deformable Transformers for End-to-End Object Detection)
> "Iterative Bounding Box Refinement. Here, each decoder layer refines the bounding boxes based on the predictions from the previous layer." (Appendix A.4 More Implementation Details)
> "we remove the decoder and form an encoder-only Deformable DETR for region proposal generation." (Section 4.2 Additional Improvements and Variants for Deformable DETR)
> "No NMS is applied before feeding the region proposals to the second stage." (Section 4.2 Additional Improvements and Variants for Deformable DETR)
> "Multi-scale feature maps are extracted without FPN (Lin et al., 2017a)." (Section 5 Experiment)

### 12. Explicit Limitations and Non-Claims
Not specified in the paper.

### 13. Constraint Profile (Synthesis)
Constraint Profile:
- Single dataset (COCO 2017) and a single task (object detection) define a narrow evaluation scope.
- Representation is pixel-based with multi-scale feature maps and fixed channel dimension (C = 256).
- Object query count is fixed in experiments (100 to 300), indicating a fixed-size query set.
- Model is trained for detection with a pretrained backbone; no multi-task sharing is described.
- Positional embeddings are used but not explored as an experimental variable.

### 14. Final Classification
Single-task, single-domain.

Justification: The paper frames Deformable DETR as "an end-to-end object detector" and evaluates only on "COCO 2017 dataset" with train/val/test-dev splits. No additional tasks, datasets, or domains are reported.

Evidence:
> "Deformable DETR is an end-to-end object detector, which is efficient and fast-converging." (Section 6 Conclusion)
> "Dataset. We conduct experiments on COCO 2017 dataset (Lin et al., 2014). Our models are trained on the train set, and evaluated on the val set and test-dev set." (Section 5 Experiment)

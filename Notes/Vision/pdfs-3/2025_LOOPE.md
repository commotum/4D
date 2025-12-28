# LOOPE Survey Responses

## 1. Basic Metadata
- Title: "LOOPE: Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers" (Title block, p.1)
- Authors: "Md Abtahi Majeed Chowdhury Md Rifat Ur Rahman Akil Ahmad Taki" (Title block, p.1)
- Year: 2025 ("arXiv:2504.14386v1 [cs.CV] 19 Apr 2025", p.1)
- Venue: arXiv ("arXiv:2504.14386v1 [cs.CV]", p.1)

## 2. One-Sentence Contribution Summary
The paper proposes LOOPE, a learnable patch-ordering positional encoding that addresses optimal 2D-to-1D patch ordering in ViTs and evaluates it (with the Three-Cell Experiment and PESI metrics) to improve positional information retention and classification accuracy.

## 3. Tasks Evaluated
### Task: Oxford-IIIT evaluation
- Task name: Oxford-IIIT evaluation
- Task type: Classification
- Dataset(s) used: Oxford-IIIT
- Domain: Not specified in the paper.
- Evidence: "We evaluate the effectiveness of different positional encodings on Vision Transformer architectures using the Oxford-IIIT and CIFAR-100 datasets." (Section 4.2, p.5); "Empirical results show that our PE significantly improves classification accuracy across various ViT architectures." (Abstract, p.1)

### Task: CIFAR-100 evaluation
- Task name: CIFAR-100 evaluation
- Task type: Classification
- Dataset(s) used: CIFAR-100
- Domain: Not specified in the paper.
- Evidence: "We evaluate the effectiveness of different positional encodings on Vision Transformer architectures using the Oxford-IIIT and CIFAR-100 datasets." (Section 4.2, p.5); "Empirical results show that our PE significantly improves classification accuracy across various ViT architectures." (Abstract, p.1)

### Task: Three-Cell Experiment (synthetic 6-class classification)
- Task name: Three-Cell Experiment
- Task type: Classification
- Dataset(s) used: "a synthetic dataset of 224 × 224 RGB images" (Section 3.2, p.3)
- Domain: Synthetic RGB images
- Evidence: "we construct a synthetic dataset of 224 × 224 RGB images" (Section 3.2, p.3); "Formally, each synthetic image Is is partitioned into a 14 × 14 grid" (Section 3.2, p.3); "a simple 6-class image classification task is enough." (Section 3.2, p.4)

## 4. Domain and Modality Scope
- Single vs. multiple domains: Multiple domains within the same modality (vision/images) are used, as the evaluation includes Oxford-IIIT, CIFAR-100, and a synthetic RGB dataset. Evidence: "We evaluate the effectiveness of different positional encodings on Vision Transformer architectures using the Oxford-IIIT and CIFAR-100 datasets." (Section 4.2, p.5); "we construct a synthetic dataset of 224 × 224 RGB images" (Section 3.2, p.3)
- Multiple modalities: Not specified in the paper.
- Domain generalization or cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Oxford-IIIT evaluation | Not specified in the paper. | Not specified in the paper (pretrained weights are used). | Not specified in the paper. | "Batch sizes were 96 for Oxford-IIIT" (Section 4.1, p.5); "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1, p.5) |
| CIFAR-100 evaluation | Not specified in the paper. | Not specified in the paper (pretrained weights are used). | Not specified in the paper. | "Batch sizes were ... 64 for CIFAR-100" (Section 4.1, p.5); "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1, p.5) |
| Three-Cell Experiment | Not specified in the paper. | Not specified in the paper (pretrained weights are used). | Not specified in the paper. | "Batch sizes were ... 64 for CIFAR-100 and our novel Three cell dataset." (Section 4.1, p.5); "All models were used with ImageNet-1K pretrained weights for a baseline comparison with the other experiments." (Section 4.1, p.5) |

## 6. Input and Representation Constraints
- 2D-to-1D mapping assumption: "a fundamental challenge arises when mapping a 2D grid to a 1D sequence." (Abstract, p.1)
- Synthetic dataset resolution: "a synthetic dataset of 224 × 224 RGB images" (Section 3.2, p.3)
- Patch size constraint (synthetic dataset): "no two neighboring 16 × 16 patches share common color information." (Section 3.2, p.3)
- Fixed grid for synthetic data: "partitioned into a 14 × 14 grid" (Section 3.2, p.3)
- Coordinate bounds for grid: "0 ≤ xi , yi ≤ 13" (Section 3.2, p.3)
- CrossViT resolution and patch sizes: "we used 240×240 images with mixed patch sizes (12×12, 16×16)." (Section 4.1, p.5)
- Resolution comparison: "Oxford-IIIT (384×384)" (Section 4.1, p.5)
- Padding/resizing requirements: Not specified in the paper.
- Fixed number of tokens / fixed dimensionality beyond explicit grid sizes: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper (only explicit grid size is "14 × 14" for the synthetic dataset). (Section 3.2, p.3)
- Fixed or variable sequence length: Not specified in the paper.
- Attention type: The paper references self-attention but does not specify global/windowed/hierarchical/sparse attention. Evidence: "permutation-invariant nature of self-attention." (Abstract, p.1)
- Mechanisms to manage computational cost (windowing, pooling, token pruning, etc.): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
- Mechanism used (sinusoidal APE): "sinusoidal P E = sin(XW T )|cos(XW T )" (Abstract, p.1)
- LOOPE mechanism (learnable patch order + sinusoidal PE): "We are proposing a Learnable patch ordering method which generates stable yet dynamic order, X, combining with XG and XC , X = XG + XC" (Section 3, p.3); "E(X) = E(XG + XC ) = sin(XWT )|cos(XWT )" (Section 3.1, p.3)
- Positional encoding variants compared: "each trained with five positional encoding methods: Zero PE, Learnable PE, Sinusoidal PE, Hilbert PE, and our proposed Learnable Hilbert PE." (Section 4.2, p.5)
- Absolute vs. relative framing: "absolute positional embeddings (APE)" and "relative positional embeddings (RPE)" are discussed. (Abstract, p.1)
- Where PE is applied (input only, every layer, attention bias): Not specified in the paper.
- Whether PE is fixed or modified per task/experiment: PE is varied across experiments (multiple PEs are compared). Evidence: "each trained with five positional encoding methods" (Section 4.2, p.5)

## 9. Positional Encoding as a Variable
- Core research variable or fixed assumption: Core research variable. Evidence: "the impact of patch ordering in positional embeddings" and "we propose LOOPE, a learnable patch-ordering method" (Abstract, p.1)
- Multiple positional encodings compared: Yes. Evidence: "each trained with five positional encoding methods: Zero PE, Learnable PE, Sinusoidal PE, Hilbert PE, and our proposed Learnable Hilbert PE." (Section 4.2, p.5)
- Claim that PE choice is not critical or secondary: Not claimed.

## 10. Evidence of Constraint Masking (Scale vs. Structure)
- Model sizes / model families evaluated: "The tested models include ViT-Base, DeiT-Base, DeiT-Small, CaiT, and Cross-ViT" (Section 4.2, p.5); Table 4 also lists "ResNet-50" and "Inception-V3" (Table 4, p.7).
- Dataset sizes: Not specified in the paper.
- Attribution of gains: The paper attributes gains to patch order/PE design rather than scale. Evidence: "optimizing this order enhances performance in downstream vision tasks" (Conclusion, p.8); "our PE significantly improves classification accuracy across various ViT architectures." (Abstract, p.1)
- Claims about scaling model size or data size as primary drivers: Not claimed.

## 11. Architectural Workarounds
- Learnable patch ordering with static + contextual components: "We are proposing a Learnable patch ordering method which generates stable yet dynamic order, X, combining with XG and XC , X = XG + XC" (Section 3, p.3). Purpose: optimize spatial order for positional embeddings.
- Space-filling curve static order for arbitrary grids: "Static Patch Order: The Hilbert curve maps a 2n × 2n grid to a 1D sequence while preserving spatial locality but cannot handle arbitrary rectangular grids. To generate patch order for arbitrary image shape, we used generalized Hilbert order, also known as Gilbert Order" (Section 3.1, p.3). Purpose: preserve locality and handle arbitrary 2D shapes.
- Dynamic ordering via context bias: "Context bias,XC introduces two key properties: (1) non-integer position and (2) dynamic ordering." (Section 3, p.3). Purpose: enable local order manipulation.
- Fixed grid assumptions in evaluation: "partitioned into a 14 × 14 grid" (Section 3.2, p.3). Purpose: constrain synthetic benchmark structure.

## 12. Explicit Limitations and Non-Claims
- Stated limitation: "our proposed LOOPE framework does not claim to deliver state-of-the-art results across all dimensions" (Conclusion, p.8)
- Future work / non-claims: "it establishes a solid foundation for future research to further investigate and refine positional embeddings in vision models." (Conclusion, p.8)
- Explicit statements about not attempting open-world learning / unrestrained multi-task learning / meta-learning: Not specified in the paper.

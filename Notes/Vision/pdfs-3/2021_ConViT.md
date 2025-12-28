## 1. Basic Metadata
- Title: “ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases” (Front matter/title block)
- Authors: “Stéphane d’Ascoli 1 2 Hugo Touvron 2 Matthew L. Leavitt 2 Ari S. Morcos 2 Giulio Biroli 1 2 Levent Sagun 2” (Front matter/author list)
- Year: “arXiv:2103.10697v2 [cs.CV] 10 Jun 2021” (Front matter)
- Venue: “Proceedings of the 38 th International Conference on Machine Learning, PMLR 139, 2021.” (Front matter)

## 2. One-Sentence Contribution Summary
The paper introduces gated positional self-attention (GPSA) to softly inject convolutional inductive bias into ViTs and shows that replacing early SA layers with GPSA yields ConViT models that improve sample efficiency and ImageNet performance relative to DeiT.

## 3. Tasks Evaluated
Task 1: ImageNet-1k image classification
- Task type: Classification
- Dataset(s): ImageNet-1k
- Domain: Images (single modality)
- Evidence:
  - “Similar to language Transformers like BERT (Devlin et al., 2018), the ViT uses an extra “class token”, appended to the sequence of patches to predict the class of the input.” (Section 3. Approach)
  - “Table 1. Performance of the models considered, trained from scratch on ImageNet. … Top-1 accuracy is measured on ImageNet-1k test set without distillation (see SM. B for distillation).” (Table 1, Performance of the ConViT)

Task 2: Subsampled ImageNet-1k image classification (data-efficiency experiments)
- Task type: Classification
- Dataset(s): Subsampled ImageNet-1k
- Domain: Images (single modality)
- Evidence:
  - “Both models are trained on a subsampled version of ImageNet-1k, where we only keep a variable fraction (leftmost column) of the images of each class for training.” (Table 2, Sample efficiency of the ConViT)
  - “Similar to language Transformers like BERT (Devlin et al., 2018), the ViT uses an extra “class token”, appended to the sequence of patches to predict the class of the input.” (Section 3. Approach)

Task 3: CIFAR100 image classification
- Task type: Classification
- Dataset(s): CIFAR100
- Domain: Images (single modality)
- Evidence:
  - “In Fig. 11, we display the time evolution of the top-1 accuracy of our ConViT+ models on CIFAR100, ImageNet and subsampled ImageNet, along with a comparison with the corresponding DeiT+ models.” (Appendix C. Further performance results)
  - “For CIFAR100, we kept all hyperparameters unchanged, but rescaled the images to 224 × 224 and increased the number of epochs (adapting the learning rate schedule correspondingly) to mimic the ImageNet scenario.” (Appendix C. Further performance results)
  - “Similar to language Transformers like BERT (Devlin et al., 2018), the ViT uses an extra “class token”, appended to the sequence of patches to predict the class of the input.” (Section 3. Approach)

Task 4: ImageNet (first 100 classes) image classification (ablation setting)
- Task type: Classification
- Dataset(s): ImageNet (first 100 classes)
- Domain: Images (single modality)
- Evidence:
  - “We examined the effects of these hyperparameters on ConViT-S, trained on the first 100 classes of ImageNet.” (Section 4. Investigating the role of locality)
  - “Similar to language Transformers like BERT (Devlin et al., 2018), the ViT uses an extra “class token”, appended to the sequence of patches to predict the class of the input.” (Section 3. Approach)

## 4. Domain and Modality Scope
- Evaluation scope: Multiple datasets within the same modality (images).
  - Evidence for modality (images): “Architectural details The ViT slices input images of size 224 into 16 × 16 non-overlapping patches of 14 × 14 pixels…” (Section 3. Approach, Architectural details)
  - Evidence for multiple datasets: “In Fig. 11, we display the time evolution of the top-1 accuracy of our ConViT+ models on CIFAR100, ImageNet and subsampled ImageNet…” (Appendix C. Further performance results)
- Domain generalization / cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k classification | Not specified in the paper. | No; trained from scratch. | Not specified in the paper. | “Table 1. Performance of the models considered, trained from scratch on ImageNet.” (Table 1, Performance of the ConViT) / “Similar to language Transformers like BERT … to predict the class of the input.” (Section 3. Approach) |
| Subsampled ImageNet-1k classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | “Both models are trained on a subsampled version of ImageNet-1k…” (Table 2, Sample efficiency of the ConViT) |
| CIFAR100 classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | “For CIFAR100, we kept all hyperparameters unchanged, but rescaled the images to 224 × 224 and increased the number of epochs…” (Appendix C. Further performance results) |
| ImageNet (first 100 classes) classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | “ConViT-S, trained on the first 100 classes of ImageNet.” (Section 4. Investigating the role of locality) |

## 6. Input and Representation Constraints
- Fixed input resolution (224) and patchification: “Architectural details The ViT slices input images of size 224 into 16 × 16 non-overlapping patches of 14 × 14 pixels…” (Section 3. Approach, Architectural details)
- Patch size referenced explicitly: “the queries and keys are linear projections of the embeddings of 16 × 16 pixel patches X ∈ RL×Demb.” (Section 2. Background)
- Fixed embedding dimensionality across blocks: “It then propagates the patches through 12 blocks which keep their dimensionality constant.” (Section 3. Approach, Architectural details)
- 2D positional encoding assumption: “the relative positional encodings rij … only depend on the distance between pixels i and j, denoted by a two-dimensional vector δij.” (Section 2. Background)
- Class token added for classification: “the ViT uses an extra “class token”, appended to the sequence of patches to predict the class of the input.” (Section 3. Approach)
- Explicit resizing requirement for CIFAR100: “For CIFAR100, we kept all hyperparameters unchanged, but rescaled the images to 224 × 224…” (Appendix C. Further performance results)
- Input-resolution change implies positional-embedding handling: “the absolute positional embeddings could easily be removed, dispensing with the need to interpolate the embeddings when changing the input resolution…” (Section 3. Approach)
- Fixed number of tokens: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
  - Evidence that sequence length is denoted abstractly: “They take as input a sequence of L embeddings … and output a sequence of L embeddings…” (Section 2. Background)
- Sequence length fixed or variable: Not specified in the paper.
- Attention type:
  - Multi-head self-attention over patch sequences: “Multi-head SA layers use several self-attention heads in parallel… They take as input a sequence of L embeddings…” (Section 2. Background)
  - GPSA mixes content and positional attention: “GPSA layers sum the content and positional terms after the softmax, with their relative importances governed by a learnable gating parameter λh…” (Section 3. Approach)
- Mechanisms to manage computational cost / parameter count (positional attention):
  - “the number of relative positional encodings rδ is quadratic in the number of patches.” (Section 3. Approach, Adaptive attention span)
  - “To avoid this, we leave the relative positional encodings rδ fixed, and train only the embeddings vpos…” (Section 3. Approach, Adaptive attention span)
  - “we take Dpos = 3 to get rid of the useless zero components.” (Section 3. Approach)
- Windowing / pooling / token pruning: Not specified in the paper (they explicitly avoid restricting attention to a subset of patches). Evidence: “This led some authors to restrict the attention to a subset of patches around the query patch… To avoid this, we leave the relative positional encodings rδ fixed, and train only the embeddings vpos…” (Section 3. Approach)

## 8. Positional Encoding (Critical Section)
- Mechanism(s) used:
  - Absolute positional embeddings at input: “the positional information is instead injected to each patch before the first layer, by adding a learnable positional embedding of dimension Demb.” (Section 3. Approach)
  - Relative positional encodings in attention: “using encodings rij of the relative position of patches i and j (Ramachandran et al., 2019)” and “the relative positional encodings rij … only depend on the distance between pixels i and j, denoted by a two-dimensional vector δij.” (Section 2. Background)
  - GPSA positional term with fixed relative encodings and learned vpos: “To avoid this, we leave the relative positional encodings rδ fixed, and train only the embeddings vpos … The initial values of rδ and vpos are given by Eq. 5, where we take Dpos = 3…” (Section 3. Approach)
- Where applied:
  - Input only (absolute): “positional information is instead injected to each patch before the first layer, by adding a learnable positional embedding…” (Section 3. Approach)
  - Attention mechanism (relative positional term): “GPSA layers sum the content and positional terms after the softmax…” and “using encodings rij of the relative position of patches i and j.” (Section 3. Approach; Section 2. Background)
- Fixed across experiments / modified / ablated:
  - Fixed relative encodings: “we leave the relative positional encodings rδ fixed…” (Section 3. Approach)
  - Absolute embeddings kept for fairness: “For fairness, and since they are computationally cheap, we keep the absolute positional embeddings of the ViT active in the ConViT.” (Section 3. Approach)
  - Ablated at test time: “In Tab. 5, we explore the importance of the absolute positional embeddings injected to the input…” (Appendix F. Further ablations)

## 9. Positional Encoding as a Variable
- Core research variable vs fixed assumption: The paper explicitly studies positional information by masking absolute positional embeddings and manipulating positional/content attention.
  - Evidence: “In Tab. 5, we explore the importance of the absolute positional embeddings injected to the input…” (Appendix F. Further ablations)
  - Evidence: “we manually set the gating parameter σ(λ) to 1 (no content attention) or 0 (no positional attention).” (Appendix F. Further ablations)
- Multiple positional encodings compared: The paper compares keeping vs masking absolute positional embeddings and examines positional vs content attention in GPSA (Appendix F).
- Claims about PE criticality: “This also shows that the absolute positional information contained in the embeddings is not very useful.” (Appendix F. Further ablations)

## 10. Evidence of Constraint Masking (Scale vs. Structure)
- Model sizes (parameters): Table 1 enumerates model sizes (e.g., “DeiT … 6M … ConViT … 6M … DeiT … 22M … ConViT … 27M … DeiT … 86M … ConViT … 86M …”). (Table 1, Performance of the ConViT)
- Dataset sizes / scaling data:
  - “by subsampling each class of the ImageNet-1k dataset by a fraction f = {0.05, 0.1, 0.3, 0.5, 1}…” (Sample efficiency of the ConViT)
- Performance gains attributed to architectural bias / locality:
  - “Table 2. The convolutional inductive bias strongly improves sample efficiency.” (Table 2 caption)
  - “Figure 11. The convolutional inductive bias is particularly useful for large models applied to small datasets.” (Figure 11 caption, Appendix C)
  - “The relative improvement of the ConViT over the DeiT increases with model size.” (Figure 11 caption, Appendix C)
  - “The relative improvement of the ConViT over the DeiT increases as the dataset becomes smaller.” (Figure 11 caption, Appendix C)
- Training tricks (distillation): “As shown in SM. B, hard distillation improves performance…” and “distillation requires an additional forward pass through a pre-trained CNN at each step of training…” (Section 3. Approach / Sample efficiency discussion)

## 11. Architectural Workarounds
- GPSA layers replace early SA layers: “The ConViT is simply a ViT where the first 10 blocks replace the SA layers by GPSA layers with a convolutional initialization.” (Section 3. Approach)
- Convolutional initialization (soft inductive bias): “We initialize the GPSA layers to mimic the locality of convolutional layers…” (Abstract)
- Gated combination of content and positional attention: “GPSA layers sum the content and positional terms after the softmax, with their relative importances governed by a learnable gating parameter λh…” (Section 3. Approach)
- Fixed relative positional encodings to reduce parameter count: “the number of relative positional encodings rδ is quadratic in the number of patches… To avoid this, we leave the relative positional encodings rδ fixed, and train only the embeddings vpos…” (Section 3. Approach)
- Class-token handling with GPSA: “We solve this problem by appending the class token to the patches after the last GPSA layer…” (Section 3. Approach)

## 12. Explicit Limitations and Non-Claims
- Future work: “Another direction which will be explored in future work is the following: if SA layers benefit from being initialized as random convolutions, could one reduce even more drastically their sample complexity by initializing them as pre-trained convolutions?” (Section 5. Conclusion and perspectives)
- Explicit non-claims about scope (e.g., open-world learning, unrestrained multi-task learning, meta-learning): Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Image datasets only (ImageNet-1k, CIFAR100, and subsets), i.e., a single modality (images) with multiple datasets (“CIFAR100, ImageNet and subsampled ImageNet”).
- Task structure: Classification only, using a class token “to predict the class of the input.”
- Representation rigidity: Fixed input resolution and patching (“input images of size 224 into 16 × 16 non-overlapping patches of 14 × 14 pixels”), fixed-depth 12-block backbone with constant dimensionality.
- Model sharing vs specialization: No explicit cross-task sharing described; ImageNet models are “trained from scratch on ImageNet.”
- Role of positional encoding: Both absolute positional embeddings at input and relative positional encodings in GPSA are used; absolute PE is ablated and deemed “not very useful.”

## 14. Final Classification
Single-task, single-domain.
Justification: All evaluations are image classification, using a class token “to predict the class of the input,” and reported as top-1 accuracy on ImageNet-1k, subsampled ImageNet-1k, and CIFAR100. These are multiple datasets within the same image modality, with no stated cross-domain transfer or multi-modal evaluation.

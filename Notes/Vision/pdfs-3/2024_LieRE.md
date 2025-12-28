### 1. Basic Metadata
- Title: LieRE: Lie Rotational Positional Encodings. Evidence: "LieRE: Lie Rotational Positional Encodings" (p.1).
- Authors: Sophie Ostmeier; Brian Axelrod; Maya Varma; Michael Moseley; Akshay Chaudhari; Curtis Langlotz. Evidence: "Sophie Ostmeier 1 * Brian Axelrod * Maya Varma 1 Michael Moseley 2 Akshay Chaudhari 2 † Curtis Langlotz 2 †" (p.1).
- Year: 2025. Evidence: "Proceedings of the 42 nd International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025." (p.1).
- Venue: Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267. Evidence: same as above (p.1).

### 2. One-Sentence Contribution Summary
LieRE proposes a Lie-group-based relative positional encoding that learns skew-symmetric bases mapped to rotation matrices to modify attention for spatial data, aiming to improve 2D/3D vision performance and resolution generalization (e.g., Section 4 method description, p.4).

### 3. Tasks Evaluated
- Task: 2D image classification; Task type: Classification; Dataset(s): CIFAR-100, ImageNet-1k; Domain: natural images; Evidence: "We begin with CIFAR-100 and ImageNet-1k benchmarks to evaluate LieRE in 2D vision tasks." (p.6, Section 5.1) and "CIFAR-100 and ImageNet-1k image classification task" (p.6, Section 5.1).
- Task: Synthetic spatial reasoning (arrow direction); Task type: Classification, Reasoning / relational; Dataset(s): synthetic images; Domain: synthetic grids; Evidence: "we designed a synthetic image classification task (Shah et al., 2024)." (p.6, Section 5.2), "The task presents a 108×108 pixel image containing a 9×9 grid (81 cells)." (p.6, Section 5.2), and "The objective is to identify the direction of this specific arrow." (p.6, Section 5.2) plus "This task requires understanding spatial relationships." (p.6, Figure 5 caption).
- Task: 3D video classification; Task type: Classification; Dataset(s): UCF101; Domain: natural video; Evidence: "To assess LieRE’s performance on 3D data, we use the UCF101 video classification benchmark (Soomro et al., 2012)." (p.7, Section 5.3).
- Task: Multi-resolution image classification (generalization to unseen resolutions); Task type: Classification; Dataset(s): ImageNet-1k validation set; Domain: natural images; Evidence: "In this section we compare the ability of methods to generalize to image resolutions not seen during training." (p.7, Section 5.6) and "We evaluate the accuracy on the ImageNet validation set with varying inference resolutions. Specifically, we scale the input images to resolutions of 196 × 196, 256 × 256, 320 × 320, 384 × 384, and 448 × 448 pixels per dimension," (p.8, Section 5.6).

### 4. Domain and Modality Scope
- Scope: Multiple domains within the same modality (vision). Evidence: "Experiments on 2D image classification (CIFAR-100, ImageNet-1k) and 3D video classification (UCF101)" (p.8, Conclusion) and "synthetic image classification task" (p.6, Section 5.2).
- Multiple modalities? Not specified in the paper; only vision tasks are described.
- Domain generalization or cross-domain transfer: Not claimed; the paper instead discusses resolution generalization ("generalize to image resolutions not seen during training," p.7, Section 5.6).

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| 2D image classification (CIFAR-100, ImageNet-1k) | No (trained from scratch) | No | Not specified in the paper | "All models use ViT-based architectures trained from scratch with standard data augmentations (RandAugment)." (p.6, Section 5.1) and "We avoid using pre-trained weights in order to help reproducibility and comparability of the results between methods." (p.5, Section 5.1).
| Synthetic spatial reasoning | Not specified in the paper | Not specified in the paper | Not specified in the paper | "We train the models on 800,000 examples and observe that they generally converge after the first 400,000 examples." (p.6, Section 5.2).
| 3D video classification (UCF101) | No (trained from scratch) | No | Not specified in the paper | "All models use a ViT-style backbone with 3D patch tokenization, trained from scratch with no hyperparameter tuning" (p.7, Section 5.3).
| Multi-resolution classification (ImageNet) | Yes within task (pre-train + fine-tune) | Yes | Not specified in the paper | "The first recipe matches the rest of the paper and consists of training the models on images of size 224 × 224 for 200 epochs. The second adds an additional fine-tuning step at size 256 × 256 for 30 epochs." (p.7, Section 5.6; p.8, Section 5.6).

### 6. Input and Representation Constraints
- 2D patch sizing and resizing: "We use a patch size of 4 × 4 on the original 32 × 32 image for CIFAR-100 and a patch size of 16 × 16 on the randomly cropped and resized 224 × 224 image." (p.13, Appendix B.2).
- 3D patch sizing and resizing: "a patch size of 2 × 16 × 16 on the randomly cropped and resized 8 × 224 × 224 video/image." (p.13, Appendix B.3).
- Synthetic grid constraint: "The task presents a 108×108 pixel image containing a 9×9 grid (81 cells)." (p.6, Section 5.2).
- Multi-resolution training and inference: "training the models on images of size 224 × 224 for 200 epochs. The second adds an additional fine-tuning step at size 256 × 256 for 30 epochs." (p.7, Section 5.6; p.8, Section 5.6) and "We evaluate the accuracy on the ImageNet validation set with varying inference resolutions. Specifically, we scale the input images to resolutions of 196 × 196, 256 × 256, 320 × 320, 384 × 384, and 448 × 448 pixels per dimension," (p.8, Section 5.6).
- Dimensionality assumption: "positions are n-dimensional vectors" (p.4, Section 4).
- Fixed number of tokens: Not specified in the paper.
- Padding requirements: Not specified in the paper (only "randomly cropped and resized" is stated).

### 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed vs variable sequence length: Not specified in the paper.
- Attention type: The paper describes standard softmax attention, e.g., "scores = QK√ dk , W = softmax(scores) and final outputs z = WV." (p.4, Section 3.2).
- Computational cost mechanisms (windowing, pooling, pruning): Not specified as architectural changes; compute is reported as dominated by attention, e.g., "runtime is dominated by the quadratic attention component" (p.15, Section D.1).

### 8. Positional Encoding (Critical Section)
- Mechanism: "When encoding positions p ∈ Rn , LieRE learns a skew-symmetric basis of matrices {Ai } for i ∈ [n]. It encodes a position by writing it in this basis,   pi Ai . We then map the resulting skew-symmetric matrix to a high-dimensional rotation via the matrix exponential." (p.4, Section 4).
- Where applied: "LieRE uses the rotation matrix computed above to modify the keys and queries of the standard attention mechanism. LieRE’s final step is to modify token i’s query and keys as Q′i = R(pi )Qi and Ki′ = R(pi )Ki." (p.4, Section 4).
- Layer/head usage: "By default, the skew bases are learned separately for every layer and attention head except in the experimental section focused on sharing parameters across heads and layers." (p.5).
- Compared/ablated across alternatives: "We compare LieRE to absolute positional encodings, RoPE-Mixed (Heo et al., 2024), and VisionLLaMA (Chu et al., 2024)." (p.6, Section 5.1) and "We control capacity via imposing a block-diagonal structure on the basis matrices. Smaller blocks (e.g., 2 × 2) repli- cate RoPE-Mixed, while larger blocks increase expressivity, with LieRE64 using a fully dense basis." (p.7, Section 5.4).

### 9. Positional Encoding as a Variable
- Core research variable: "To assess the impact of LieRE and other positional encodings on ViT performance, we evaluate several encoding schemes across diverse tasks" (p.2, Introduction).
- Multiple positional encodings compared: "We compare LieRE to absolute positional encodings, RoPE-Mixed (Heo et al., 2024), and VisionLLaMA (Chu et al., 2024)." (p.6, Section 5.1).
- PE claimed not critical/secondary: Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model sizes evaluated: "ViT-Tiny (22M), ViT-Base (85M), ViT-Large (302M)" (p.13, Figure 8).
- Dataset size variation: "We train the models on 800,000 examples" (p.6, Section 5.2) and "training on only 20–90% of the CIFAR-100 dataset" (p.6, Section 5.1).
- Architectural capacity emphasis: "LieRE introduces minimal overhead–—only 580k addi- tional parameters (0.68% for ViT-B)–—yet offers a flex- ible mechanism for increasing representational capacity." (p.7, Section 5.4).
- Compute scaling: "LieRE enables a 3.9× reduction in training compute while maintaining the accuracy achieved by absolute position encodings after 200 epochs." (p.14, Section D).
- Training tricks noted: "All vision experiments used RandAugment (Cubuk et al., 2020)." (p.13, Appendix B.2).

### 11. Architectural Workarounds
- Capacity control via basis structure: "We control capacity via imposing a block-diagonal structure on the basis matrices. Smaller blocks (e.g., 2 × 2) repli- cate RoPE-Mixed, while larger blocks increase expressivity, with LieRE64 using a fully dense basis." (p.7, Section 5.4).
- Parameter sharing knob: "By default, the skew bases are learned separately for every layer and attention head except in the experimental section focused on sharing parameters across heads and layers." (p.5).
- 3D tokenization: "All models use a ViT-style backbone with 3D patch tokenization" (p.7, Section 5.3).
- Fixed grid assumption in synthetic task: "The task presents a 108×108 pixel image containing a 9×9 grid (81 cells)." (p.6, Section 5.2).

### 12. Explicit Limitations and Non-Claims
- Limitations stated: "For 1D input, LieRE reduces to RoPE with learnable phases (proof in appendix A)." (p.8, Section 7) and "this may limit its applicability to other architectures—–such as convolutional neural networks—–that do not rely on the attention mechanism." (p.8, Section 7) and "The current formula- tion encodes vector positions in Rd . While sufficient for many applications, it may not directly apply to tasks that require pose encoding in SE(3) (e.g., robotics)." (p.8, Section 7) and "Lastly, in its current implementation, LieRE relies on the accuracy and numerical stability of the matrix exponential in PyTorch." (p.8, Section 7).
- Explicit non-claims about open-world or unrestrained multi-task learning: Not specified in the paper.

### 13. Constraint Profile (Synthesis)
- Domain scope: Vision-only, spanning 2D images, 3D videos, and synthetic grids ("2D image classification (CIFAR-100, ImageNet-1k) and 3D video classification (UCF101)" p.8; "synthetic image classification task" p.6).
- Task structure: All evaluations are classification tasks, including a synthetic spatial reasoning classification setup (p.6 and p.7).
- Representation rigidity: Fixed patch sizes and explicit image/video resolutions are specified (p.13), with a fixed synthetic grid size (p.6).
- Model sharing vs specialization: Models are trained from scratch per task with no pre-trained weights, except for a pretrain+fine-tune recipe in the multi-resolution experiment (p.5, p.7, p.8).
- Role of positional encoding: Core experimental variable with direct comparisons across PE schemes and basis capacities (p.2, p.6, p.7).

### 14. Final Classification
Multi-task, multi-domain (constrained). The paper evaluates multiple tasks (2D image classification, 3D video classification, and a synthetic spatial reasoning task) within the vision modality (p.6, p.7, p.8), and models are trained from scratch per task without joint multi-task training (p.5, p.6, p.7). The experimental setup is constrained to classification with fixed patching/resolution schemes and explicit positional encoding comparisons, rather than unrestrained multi-task learning (p.6, p.7, p.13).

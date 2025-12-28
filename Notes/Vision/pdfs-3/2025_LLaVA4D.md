## 1. Basic Metadata
- Title: LLaVA-4D: Embedding SpatioTemporal Prompt into LMMs for 4D Scene Understanding
- Authors: Hanyu Zhou, Gim Hee Lee
- Year: 2025
- Venue: arXiv (arXiv:2505.12253v1 [cs.CV] 18 May 2025)
- Evidence: "arXiv:2505.12253v1 [cs.CV] 18 May 2025" (front matter, p1)

## 2. One-Sentence Contribution Summary
- The paper proposes LLaVA-4D, a general LMM that embeds a dynamic-aware 4D spatiotemporal prompt to improve 4D scene understanding of static backgrounds and dynamic objects. (Abstract)
- Evidence: "In this paper, we propose LLaVA-4D, a general LMM framework with a novel spatiotemporal prompt for visual representation in 4D scene understanding." (Abstract, p1)

## 3. Tasks Evaluated

Task: Dense captioning (2D/3D/4D)
- Task type: Generation
- Dataset(s) used: Scan2Cap; Chat4D (includes 2D/3D/4D DC)
- Domain: RGB-D scans (Scan2Cap); multi-view videos / dynamic 4D scenes (Chat4D)
- Evidence:
  - "Scan2cap: Context-aware dense captioning in rgb-d scans." (References, p11)
  - "Chat4D dataset includes 2D, 3D, and 4D vision-language training sets for dense captioning, QA, and visual grounding." (Figure 4 caption, Sec. 4, p6)
  - "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Sec. 5.1, p7)
  - "Given a multi-view video input sequence I, our LLaVA-4D achieves 4D scene understanding..." (Sec. 3 Overview, p3)

Task: Visual question answering (2D/3D/4D)
- Task type: Reasoning / relational
- Dataset(s) used: ScanQA; Chat4D (includes 2D/3D/4D QA)
- Domain: 3D scenes (ScanQA); multi-view videos / dynamic 4D scenes (Chat4D)
- Evidence:
  - "Scanqa: 3d question answering for spatial scene understanding." (References, p11)
  - "Chat4D dataset includes 2D, 3D, and 4D vision-language training sets for dense captioning, QA, and visual grounding." (Figure 4 caption, Sec. 4, p6)
  - "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Sec. 5.1, p7)
  - "Given a multi-view video input sequence I, our LLaVA-4D achieves 4D scene understanding..." (Sec. 3 Overview, p3)

Task: Visual grounding / referring (2D/3D/4D)
- Task type: Reasoning / relational
- Dataset(s) used: Multi3DRefer; ScanRefer; Chat4D (includes 2D/3D/4D VG)
- Domain: 3D scenes / RGB-D scans (Multi3DRefer, ScanRefer); multi-view videos / dynamic 4D scenes (Chat4D)
- Evidence:
  - "Multi3drefer: Grounding text description to multiple 3d objects." (References, p11)
  - "Scanrefer: 3d object localization in rgb-d scans using natural language." (References, p12)
  - "Chat4D dataset includes 2D, 3D, and 4D vision-language training sets for dense captioning, QA, and visual grounding." (Figure 4 caption, Sec. 4, p6)
  - "We compare all competing methods on multiple 3D datasets: Scan2Cap [43], ScanQA [41], ScanRef [55] and Multi3DRefer [44] and our Chat4D dataset." (Sec. 5.1, p7)

## 4. Domain and Modality Scope
- Single domain? No; the paper explicitly uses multiple domains within the same modality ("our dataset includes 2D, 3D and 4D vision-language data types"). (Sec. 4.1, p6)
- Multiple domains within the same modality? Yes; the paper states "our dataset includes 2D, 3D and 4D vision-language data types" (Sec. 4.1, p6).
- Multiple modalities? Yes; "Large multimodal models (LMMs) aim to learn the representation alignment between language and other modalities such as vision" (Sec. 1 Introduction, p1).
- Domain generalization / cross-domain transfer claimed? Not specified in the paper.

## 5. Model Sharing Across Tasks
Note: The paper describes a single three-stage training pipeline using DC, QA, and VG tasks, but it does not explicitly state whether separate heads or separate weights are used per task.
- Evidence for shared training pipeline:
  - "Stage 1: Content Alignment. The training sets of the DC and QA tasks in the 2D&3D vision-language data of our Chat4D are used to initially align the content between visual and linguistic representations." (Sec. 4.2, p7)
  - "Stage 2: Spatiotemporal Coordinate Alignment. ... we use the training data of the VG task in the 2D&3D vision-language subset of our Chat4D..." (Sec. 4.2, p7)
  - "Stage 3: 4D Task Instruction Fine-Tuning... through a multi-task instruction fine-tuning strategy." (Sec. 4.2, p7)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Dense captioning (DC) | Not specified in the paper. | "Stage 1: Content Alignment. The training sets of the DC and QA tasks..." and "Stage 3: 4D Task Instruction Fine-Tuning... multi-task instruction fine-tuning strategy." (Sec. 4.2, p7) | Not specified in the paper. | See quotes above. |
| Visual QA (QA) | Not specified in the paper. | "Stage 1: Content Alignment. The training sets of the DC and QA tasks..." and "Stage 3: 4D Task Instruction Fine-Tuning... multi-task instruction fine-tuning strategy." (Sec. 4.2, p7) | Not specified in the paper. | See quotes above. |
| Visual grounding (VG) | Not specified in the paper. | "Stage 2: Spatiotemporal Coordinate Alignment. ... we use the training data of the VG task..." and "Stage 3: 4D Task Instruction Fine-Tuning... multi-task instruction fine-tuning strategy." (Sec. 4.2, p7) | Not specified in the paper. | See quotes above. |

## 6. Input and Representation Constraints
- Multi-view video input: "Given a multi-view video input sequence I, our LLaVA-4D achieves 4D scene understanding..." (Sec. 3 Overview, p3)
- 4D coordinate construction from geometry: "we construct 4D coordinate tensors [x, y, z, t] from multi-view videos using visual geometry" (Sec. 3 Overview, p3)
- Uses SfM/MVS for geometry: "we use SfM [29] for camera pose P = [R | T ] and MVS [30] for depth D." (Sec. 3.1, p4)
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper; the paper states it constructs "4D coordinate tensors [x, y, z, t]" (Sec. 3 Overview, p3).
- Padding or resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost: Not specified in the paper.
- Related evidence: "Cross-attention fusion module is a transformer-based network architecture [34]." (Sec. 5.1, p7)

## 8. Positional Encoding (Critical Section)
- Mechanism used: spatiotemporal encoding of coordinates with learnable Fourier features for spatial position and a temporal encoding with motion information.
- Evidence:
  - "Spatiotemporal Encoding. We perform spatiotemporal encoding to convert the 4D coordinates into learnable feature patterns." (Sec. 3.1, p4)
  - "we adopt the same spatial position encoding strategy for objects and background via learnable Fourier feature [33]" (Sec. 3.1, p4)
  - "we add motion information into the temporal encoding:" (Sec. 3.1, p4)
- Where applied: on 4D coordinates and on language tokens (coordinate-aligned linguistic tokens).
- Evidence:
  - "we construct 4D coordinate tensors [x, y, z, t] from multi-view videos using visual geometry" (Sec. 3 Overview, p3)
  - "We concatenate 4D-aware visual tokens with coordinate-aligned linguistic tokens for the LLM to reason." (Sec. 3.3, p6)
- Absolute/relative/RoPE/etc. category: Not specified in the paper.
- Fixed across experiments vs modified per task: Not specified in the paper (ablation compares with/without encoding; see Sec. 5.3).

## 9. Positional Encoding as a Variable
- The paper ablates coordinate encoding and textual coordinate encoding.
- Evidence:
  - "Role of 4D Coordinate Encoding. In Table 3, we analyze the impact of 3D position encoding and 1D time encoding on the performance of 4D understanding." (Sec. 5.3, p8)
  - "Impact of Textual Coordinate Encoding. Table 5 ablates the impact of textual coordinate encoding on scene understanding by the LMM." (Sec. 5.3, p8)
- Claim that PE choice is not critical or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size(s): "Our LLaVA-4D model utilizes the pre-trained weights of LLaVA-1.5-7B [17] and the vision encoder of CLIP-ViT-L-336px [3, 59]." (Sec. 5.1, p7)
- Dataset size(s): "These datasets cover dense captioning (DC), visual QA and visual grounding (VG) tasks with a total of 654.5K samples." (Sec. 4.1, p6) and "produce a dataset of 224.6K samples." (Sec. 4.1, p6)
- Compute scale: "The whole model is trained on 8 RTX 4090 GPUs over 86 hours" (Sec. 5.1, p7)
- Performance gains attributed to architecture: "Coordinate embedding is the key to improving the overall performance of 4D understanding tasks by a large margin." (Sec. 5.3, p8)

## 11. Architectural Workarounds
- Dynamic-aware 4D coordinate encoding as a prompt: "we construct 4D coordinate tensors [x, y, z, t] from multi-view videos... The encoded position and time are concatenated as a spatiotemporal prompt to guide visual fusion" (Sec. 3 Overview, p3)
- Spatiotemporal-disentangled vision embedding: "disentangle these visual features into spatiotemporal components" (Sec. 3 Overview, p3)
- Cross-attention fusion: "embed encoded 4D coordinate features into these spatiotemporal features via cross-attention fusion" (Sec. 3 Overview, p3)
- Coordinate-aligned language embedding: "We concatenate 4D-aware visual tokens with coordinate-aligned linguistic tokens for the LLM to reason." (Sec. 3.3, p6)

## 12. Explicit Limitations and Non-Claims
- Limitations / future work:
  - "Limitation. While our model performs well on most 3D and 4D dynamic scenes, it struggles with fast-moving objects due to motion blur from frame-based cameras. This reduces discriminability of the spatiotemporal feature and thus weakens 4D understanding. In future work, we plan to incorporate event cameras [60] with high temporal resolution to improve dynamic representation." (Sec. 5.3, p8)
- Explicit non-claims (e.g., not attempting open-world or unrestrained multi-task learning): Not specified in the paper.

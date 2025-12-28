## 1. Basic Metadata
- Title: CIRCLE-ROPE: CONE-LIKE DECOUPLED ROTARY POSITIONAL EMBEDDING FOR LARGE VISION-LANGUAGE MODELS
- Authors: Chengcheng Wang; Jianyuan Guo; Hongguang Li; Yuchuan Tian; Ying Nie; Chang Xu; Kai Han
- Year: 2025
- Venue (conference/journal/arXiv): arXiv preprint (under review). Evidence: "arXiv:2505.16416v2 [cs.CV] 4 Oct 2025" (p.1) and "Preprint. Under review." (p.1)

## 2. One-Sentence Contribution Summary
The paper proposes Circle-RoPE to remove cross-modal positional biases in vision-language models by remapping image token indices so they are equidistant to text tokens while preserving intra-image spatial structure. Evidence: "we propose Circle-RoPE, a novel encoding scheme designed to eliminate spurious cross-modal biases." (Abstract, p.1) and "Our key idea is to project image token indices onto a ring that is orthogonal to the linear axis of text token indices, thereby forming a cone-like structure in the positional encoding space." (Abstract, p.1)

## 3. Tasks Evaluated
| Task name | Task type | Dataset(s) used | Domain | Evidence (quote + location) |
| --- | --- | --- | --- | --- |
| MMMU (val) | Other (task type not specified in the paper) | MMMUval [29] | Not specified in the paper. | "MMMUval [29]" (Table 2, p.8) |
| MMMU-Pro (overall/avg) | Other (task type not specified in the paper) | MMMU-Prooverall [30]; MMMU_Pro-avg | Not specified in the paper. | "MMMU-Prooverall [30]" (Table 2, p.8) and "MMMU_Pro-avg" (Table 5, p.9) |
| MathVista (mini) | Other (task type not specified in the paper) | MathVistamini [15]; MathVista_MINI | Not specified in the paper. | "MathVistamini [15]" (Table 2, p.8) and "MathVista_MINI" (Table 4, p.8) |
| MMStar | Other (task type not specified in the paper) | MMStar [3] | Not specified in the paper. | "MMStar [3]" (Table 2, p.8) |
| AI2D | Other (task type not specified in the paper) | AI2D [9]; AI2D_TEST | Not specified in the paper. | "AI2D [9]" (Table 2, p.8) and "AI2D_TEST" (Table 4, p.8) |
| RealWorldQA | Other (task type not specified in the paper) | RealWorldQA [25] | Not specified in the paper. | "RealWorldQA [25]" (Table 2, p.8) |
| InfoVQA | Other (task type not specified in the paper) | InfoVQA [17] | Not specified in the paper. | "InfoVQA [17]" (Table 2, p.8) |
| ChartQA | Other (task type not specified in the paper) | ChartQA_TEST | Not specified in the paper. | "ChartQA_TEST" (Table 4, p.8) |
| MathVision | Other (task type not specified in the paper) | MathVision | Not specified in the paper. | "MathVision" (Table 6, p.13) |
| MMMU (test, visualization) | Other (task type not specified in the paper) | MMMUtest benchmark [29] | Not specified in the paper. | "evaluations performed on the MMMUtest benchmark [29]." (Sec. 5.6, p.9) |

## 4. Domain and Modality Scope
- Single domain? Not specified in the paper.
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Yes. Evidence: "vision-language models (VLMs)" (Abstract, p.1) and "text and image tokens" (Abstract, p.1).
- Domain generalization or cross-domain transfer? Not claimed in the paper.

## 5. Model Sharing Across Tasks
The paper does not explicitly state whether separate models are trained per task or whether a single model is reused; it only describes a unified training setup and a single positional-encoding modification.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| MMMU (val) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| MMMU-Pro (overall/avg) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| MathVista (mini) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| MMStar | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| AI2D | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| RealWorldQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| InfoVQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| ChartQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| MathVision | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |
| MMMU (test, visualization) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7) and "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) |

## 6. Input and Representation Constraints
- Image resolution: "Image Resolution           512×512" (Table 7, p.13). Fixed vs variable is not specified.
- Max sequence length: "Max Sequence Length           4096" (Table 7, p.13). Fixed vs variable is not specified.
- Image token representation uses 2D grid coordinates: "In M-RoPE [20], image token indices are represented separately by width and height coordinates, text tokens use 1D positional index equivalent to standard RoPE." (Sec. 4.1, p.5)
- Fixed patch size, fixed number of tokens, padding/resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: "Max Sequence Length           4096" (Table 7, p.13).
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Computational cost controls (windowing/pooling/token pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
- Mechanism: RoPE with a new Circle-RoPE variant. Evidence: "Rotary Position Embedding (RoPE) is a widely adopted technique for encoding relative positional information in large language models (LLMs)." (Abstract, p.1) and "we propose Circle-RoPE, a novel encoding scheme designed to eliminate spurious cross-modal biases." (Abstract, p.1)
- Components: "Circle-RoPE consists of two components: Circular Image Token Index Projection (CIP, Sec. 4.1) and Alternating Geometry Encoding (AGE, Sec. 4.2)" (Sec. 4, p.4).
- Where applied: AGE changes positional encoding across layers: "Alternating Geometry Encoding (AGE), which cyclically switches between the M-RoPE [20] index and the Circle-RoPE index across different Transformer layers" (Sec. 4.2, p.6).
- Fixed vs modified: The paper varies Circle-RoPE settings and AGE strategies in ablations: "Performance comparison across different CIP configurations." (Table 3, p.8) and "Performance comparison across different AGE configurations." (Table 4, p.8). Task-specific PE changes are not specified.

## 9. Positional Encoding as a Variable
- Core research variable? Yes. Evidence: "we propose Circle-RoPE, a novel encoding scheme designed to eliminate spurious cross-modal biases." (Abstract, p.1)
- Multiple positional encodings compared? Yes. Evidence: "We compute PTD for three typical multimodal encoding methods, i.e., hard embeeding (Figure 1(a)), unordered embedding (Figure 1(b)), and spatial embedding (Figure 1(c))." (Sec. 3, p.3)
- Claim that PE choice is not critical or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size(s): "Base Model               Qwen2.5-VL-3B" (Table 7, p.13).
- Dataset size(s): "we randomly sample one-tenth of the MAmmoTH-VL Instruct dataset (12M) [8] and exclude all video data, resulting in a subset named MAmmoTH-VL-Sub (1M)." (Sec. 5.1, p.7)
- Attribution of gains: "The only modification introduced is in the implementation of the positional encoding method; all other configurations are retained from the baseline model." (Sec. 5.1, p.7) and "our method achieves significant performance improvements compared to the baseline." (Sec. 5.1, p.7). No claims that scaling model or data size is the primary driver are stated.

## 11. Architectural Workarounds
- Circle-RoPE is built with explicit geometric transformations; the paper states that "The CIP process consists of three key steps:" (Sec. 4.1, p.5), then details coordinate centralization, mixed-angle circular mapping, and target-plane rotation in Sec. 4.1.
- Layer-wise alternation to manage geometry bias: "Alternating Geometry Encoding (AGE), which cyclically switches between the M-RoPE [20] index and the Circle-RoPE index across different Transformer layers" (Sec. 4.2, p.6).
- Multi-image ordering: "When the input contains multiple images, we explicitly encode their sequential order by translating each image’s circular-encoding center along a fixed global axis." (Sec. 4.3, p.6)

## 12. Explicit Limitations and Non-Claims
- Limitation (adaptation cost): "Circle-RoPE exhibits a measurable adaptation cost: at early training (3k steps) its performance is slightly below the SFT baseline (Table 6)." (Appendix A.1, p.13)
- Limitation (backbone mismatch): "adopting a more dissimilar backbone would likely incur a larger adaptation cost that is computationally prohibitive." (Appendix A.1, p.13)
- Other explicit limitations, non-claims, or future work statements: Not specified in the paper.

## 13. Constraint Profile (Synthesis)
- Domain scope: Vision-language models with text and image tokens are the focus; domain types are not specified beyond that. Evidence: "vision-language models (VLMs)" and "text and image tokens" (Abstract, p.1).
- Task structure: Multiple benchmark evaluations across a "diverse range of datasets" are used. Evidence: "This section evaluates the performance of Circle-RoPE on a diverse range of datasets" (Sec. 5.2, p.7).
- Representation rigidity: Image resolution and max sequence length are explicitly set (512×512; 4096), and image tokens are represented by width/height coordinates. Evidence: Table 7 (p.13) and Sec. 4.1 (p.5).
- Model sharing vs specialization: The paper describes a unified training setup but does not specify per-task fine-tuning or separate heads. Evidence: "All experiments are conducted under a unified training setup." (Sec. 5.1, p.7)
- Positional encoding: Central research variable with explicit ablations and layer-wise alternation. Evidence: Sec. 3 (p.3), Sec. 4.2 (p.6), Tables 3-4 (p.8).

## 14. Final Classification
Multi-task, single-domain. The paper evaluates on multiple datasets (e.g., "MMMUval [29]", "MMStar [3]", "AI2D [9]", "InfoVQA [17]") and states it tests a "diverse range of datasets" (Sec. 5.2, p.7), indicating multiple tasks. It does not claim domain generalization or cross-domain transfer, and the modality focus is explicitly vision-language ("vision-language models (VLMs)" and "text and image tokens" in the Abstract, p.1), so the evidence supports a multi-task setup within a single modality scope rather than unrestrained multi-domain learning.

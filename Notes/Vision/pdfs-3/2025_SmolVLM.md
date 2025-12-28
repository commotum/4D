## 1. Basic Metadata
- Title: SmolVLM: Redefining small and efficient multimodal models.
- Authors: Andres Marafioti; Orr Zohar; Miquel Farre; Merve Noyan; Elie Bakouch; Pedro Cuenca; Cyril Zakka; Loubna Ben Allal; Anton Lozhkov; Nouamane Tazi; Vaibhav Srivastav; Joshua Lochner; Hugo Larcher; Mathieu Morlon; Lewis Tunstall; Leandro von Werra; Thomas Wolf.
- Year: 2025.
- Venue: arXiv (arXiv:2504.05299v1 [cs.AI], 7 Apr 2025).
Evidence: "SmolVLM: Redefining small and efficient multimodal models" (Title page). "arXiv:2504.05299v1 [cs.AI] 7 Apr 2025" (Title page).

## 2. One-Sentence Contribution Summary
SmolVLM introduces compact multimodal models and systematically explores architectural/tokenization/data choices to achieve strong image and video performance with low inference memory.
Evidence: "We introduce SmolVLM, a series of compact multimodal models specifically engineered for resource-efficient inference." (Abstract). "Through this, we identify key design choices that yield substantial performance gains on image and video tasks with minimal memory footprints." (Abstract).

## 3. Tasks Evaluated
Evidence that multiple benchmarks are evaluated: "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3).
Group labels used in the paper: "Single-Image", "Multi-task", "Video" (Table 1, Section 4.3).

Single-Image benchmarks:
- OCRBench — Task type: Other (specify: Character Recognition); Dataset(s): OCRBench (Liu et al., 2024e); Domain: Single-Image / Character Recognition. Evidence: "OCRBench (Liu et al., 2024e)" and "Character Recognition" (Table 1, Section 4.3).
- AI2D — Task type: Other (specify: Science Diagrams); Dataset(s): AI2D (Kembhavi et al., 2016); Domain: Single-Image / Science Diagrams. Evidence: "AI2D (Kembhavi et al., 2016)" and "Science Diagrams" (Table 1, Section 4.3).
- ChartQA — Task type: Other (specify: Chart Understanding); Dataset(s): ChartQA (Masry et al., 2022); Domain: Single-Image / Chart Understanding. Evidence: "ChartQA (Masry et al., 2022)" and "Chart Understanding" (Table 1, Section 4.3).
- TextVQA — Task type: Other (specify: Text Understanding); Dataset(s): TextVQA (Singh et al., 2019); Domain: Single-Image / Text Understanding. Evidence: "TextVQA (Singh et al., 2019)" and "Text Understanding" (Table 1, Section 4.3).
- DocVQA — Task type: Other (specify: Document Understanding); Dataset(s): DocVQA (Mathew et al., 2021); Domain: Single-Image / Document Understanding. Evidence: "DocVQA (Mathew et al., 2021)" and "Document Understanding" (Table 1, Section 4.3).
- ScienceQA — Task type: Other (specify: High-school Science); Dataset(s): ScienceQA (Lu et al., 2022); Domain: Single-Image / High-school Science. Evidence: "ScienceQA (Lu et al., 2022)" and "High-school Science" (Table 1, Section 4.3).

Multi-task benchmarks:
- MMMU — Task type: Other (specify: College-level Multidiscipline); Dataset(s): MMMU (Yue et al., 2024a); Domain: Multi-task / College-level Multidiscipline. Evidence: "MMMU (Yue et al., 2024a)" and "College-level Multidiscipline" (Table 1, Section 4.3).
- MathVista — Task type: Other (specify: General Math Understanding); Dataset(s): MathVista (Lu et al., 2024b); Domain: Multi-task / General Math Understanding. Evidence: "MathVista (Lu et al., 2024b)" and "General Math Understanding" (Table 1, Section 4.3).
- MMStar — Task type: Reasoning / relational (descriptor: Multidisciplinary Reasoning); Dataset(s): MMStar (Chen et al., 2024a); Domain: Multi-task / Multidisciplinary Reasoning. Evidence: "MMStar (Chen et al., 2024a)" and "Multidisciplinary Reasoning" (Table 1, Section 4.3).

Video benchmarks:
- Video-MME — Task type: Other (specify: General Video Understanding); Dataset(s): Video-MME (Fu et al., 2024); Domain: Video / General Video Understanding. Evidence: "Video-MME (Fu et al., 2024)" and "General Video Understanding" (Table 1, Section 4.3).
- MLVU — Task type: Other (specify: MovieQA + MSRVTT-Cap); Dataset(s): MLVU (Zhou et al., 2024); Domain: Video / MovieQA + MSRVTT-Cap. Evidence: "MLVU (Zhou et al., 2024)" and "MovieQA + MSRVTT-Cap" (Table 1, Section 4.3).
- MVBench — Task type: Reasoning / relational (descriptor: Multiview Reasoning); Dataset(s): MVBench (Li et al., 2024b); Domain: Video / Multiview Reasoning. Evidence: "MVBench (Li et al., 2024b)" and "Multiview Reasoning" (Table 1, Section 4.3).
- WorldSense — Task type: Other (specify: Temporal + Physics); Dataset(s): WorldSense (Hong et al., 2025); Domain: Video / Temporal + Physics. Evidence: "WorldSense (Hong et al., 2025)" and "Temporal + Physics" (Table 1, Section 4.3).
- TempCompass — Task type: Other (specify: Temporal Understanding); Dataset(s): TempCompass (Liu et al., 2024d); Domain: Video / Temporal Understanding. Evidence: "TempCompass (Liu et al., 2024d)" and "Temporal Understanding" (Table 1, Section 4.3).

## 4. Domain and Modality Scope
- Single domain? No; multiple visual domains are explicitly listed. Evidence: "The visual components comprise document understanding, captioning, and visual question answering (including 2% dedicated to multi-image reasoning), chart understanding, table understanding, and visual reasoning tasks." (Section 4.1).
- Multiple domains within the same modality? Yes; the paper enumerates multiple visual domains in the vision training mix (same evidence as above).
- Multiple modalities? Yes; images and videos are both used. Evidence: "Images are split into subimages, frames are sampled from videos, and then encoded into visual features." (Figure 2). "SmolVLM models extend beyond static images, demonstrating robust video comprehension capabilities." (Abstract).
- Does the paper claim domain generalization or cross-domain transfer? Not explicitly; the closest statement is "We demonstrate that SmolVLM models generalize effectively to video tasks, achieving competitive scores on challenging benchmarks like Video-MME, highlighting their suitability for diverse multimodal scenarios and real-time, on-device applications." (Introduction contributions).

## 5. Model Sharing Across Tasks
The paper reports benchmark results but does not specify per-task fine-tuning or heads. Evidence: "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| OCRBench | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| AI2D | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| ChartQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| TextVQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| DocVQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| ScienceQA | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| MMMU | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| MathVista | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| MMStar | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| Video-MME | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| MLVU | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| MVBench | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| WorldSense | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |
| TempCompass | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3). |

## 6. Input and Representation Constraints
- Input resolution (resizing requirement): "For SmolVLM, this resizes the longest edge of images to 1920 in the 256M and 500M models and 1536 in the 2.2B." (Section 4.2).
- Patch size: Not specified in the paper (encoder is named "SigLIP-B/16" but no explicit patch-size statement).
- Visual token count example: "a single 512 × 512 image encoded with SigLIP-B/16 requires 1024 tokens." (Section 2.2).
- Context-length limit (token budget): "we adopt a 16k-token context for SmolVLM and an 8k-token limit for smaller variants." (Section 2.2).
- Image/video preprocessing: "Images are split into subimages, frames are sampled from videos, and then encoded into visual features." (Figure 2). "high-resolution images are divided into multiple sub-images along with a downsized version of the original." (Section 2.3).
- Video resizing: "video frames were instead rescaled to the resolution of the image encoder." (Section 2.3).
- Token compression: "These features are first rearranged via a pixel-shuffle operation, then mapped into the LLM input space as visual tokens using an MLP projection." (Figure 2). "Pixel shuffle rearranges spatial features into additional channels, reducing spatial resolution but increasing representational density." (Section 2.2).
- Fixed dimensionality/padding requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: "we adopt a 16k-token context for SmolVLM and an 8k-token limit for smaller variants." (Section 2.2).
- Fixed vs. variable sequence length: Not specified in the paper beyond the stated limits.
- Attention type: "we adopt a self-attention architecture in which visual tokens from the vision encoder are concatenated with textual tokens and jointly processed by a language model." (Section 2.2).
- Compute-cost mechanisms: "Pixel shuffle rearranges spatial features into additional channels, reducing spatial resolution but increasing representational density." (Section 2.2). "high-resolution images are divided into multiple sub-images along with a downsized version of the original." (Section 2.3). "frame averaging was excluded from SmolVLM’s final design, and video frames were instead rescaled to the resolution of the image encoder." (Section 2.3).

## 8. Positional Encoding (Critical Section)
- Mechanism used: RoPE is explicitly referenced via base scaling: "we extended the context capacity by increasing the RoPE base from 10k to 273k." (Section 2.2).
- Where applied (input only / every layer / attention bias): Not specified in the paper.
- Fixed across experiments or modified per task: Not specified; only a RoPE base change is described (quote above).
- Additional positional tokenization detail (for sub-image positions): "we introduced positional tokens, significantly improving training convergence and reducing stalls." (Section 3.1). "Learned positional tokens outperform raw text tokens for compact VLMs." (Section 3.1).

## 9. Positional Encoding as a Variable
- Treated as a core research variable vs. fixed assumption: The paper lists positional encoding among the explored architectural choices: "including encoder-LM parameter balance, tokenization methods, positional encoding, and training data composition." (Introduction contributions).
- Multiple positional encodings compared? Not specified in the paper; no explicit comparison of different PE mechanisms is stated beyond the RoPE base change and positional tokenization notes.
- Claim that PE choice is “not critical” or secondary? Not specified in the paper.

## 10. Evidence of Constraint Masking
- Model sizes: "We construct three variants of SmolVLM... SmolVLM-256M... SmolVLM-500M... SmolVLM-2.2B." (Section 4).
- Scaling model size: "Increasing SmolVLM’s parameter count consistently yields substantial performance improvements across all evaluated benchmarks." (Section 4.3).
- Dataset sizes: Not explicitly stated; dataset names with sizes are listed for video data (e.g., "LLaVA-video-178k" and "Vista-400k"). Evidence: "For video, we sample visual description and captioning from LLaVA-video-178k... temporal understanding from Vista-400k" (Section 4.1).
- Training tricks / architecture choices linked to gains: "Compact VLMs significantly benefit from extended context lengths." (Section 2.2). "Small VLMs benefit from more aggressive visual token compression." (Section 2.2). "System prompts and media intro/outro tokens significantly improve compact VLM performance, particularly for video tasks." (Section 3.2).
- Architectural hierarchy: Not specified in the paper.

## 11. Architectural Workarounds
- Pixel-shuffle token compression: "Pixel shuffle rearranges spatial features into additional channels, reducing spatial resolution but increasing representational density." (Section 2.2).
- Image splitting for high resolution: "high-resolution images are divided into multiple sub-images along with a downsized version of the original." (Section 2.3).
- Avoiding frame averaging and rescaling frames: "frame averaging was excluded from SmolVLM’s final design, and video frames were instead rescaled to the resolution of the image encoder." (Section 2.3).
- Extended context via RoPE base scaling: "we extended the context capacity by increasing the RoPE base from 10k to 273k." (Section 2.2).
- Positional tokens for sub-image positions: "we introduced positional tokens, significantly improving training convergence and reducing stalls." (Section 3.1).
- Prompt-based training strategies: "System prompts and media intro/outro tokens significantly improve compact VLM performance, particularly for video tasks." (Section 3.2).

## 12. Explicit Limitations and Non-Claims
Not specified in the paper.

## 13. Constraint Profile (Synthesis)
Constraint Profile:
- Domain scope: Multiple visual domains and video tasks are used, but all within curated benchmarks and task lists ("Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3) and the vision-task list in Section 4.1).
- Task structure: Benchmark-driven evaluation across single-image, multi-task, and video groupings ("Single-Image", "Multi-task", "Video" labels in Table 1, Section 4.3).
- Representation rigidity: Explicit resizing and token-compression constraints ("resizes the longest edge of images to 1920... and 1536" (Section 4.2); pixel shuffle compression in Section 2.2).
- Model sharing vs. specialization: Per-task sharing/fine-tuning is not specified; only benchmark results are reported (Section 4.3).
- Role of positional encoding: RoPE base scaling and positional tokenization are part of the architectural exploration (Section 2.2; Section 3.1; Introduction contributions).

## 14. Final Classification
Multi-task, multi-domain (constrained).
Justification: The paper evaluates across many named benchmarks spanning single-image, multi-task, and video categories ("Table 1 summarizes results across nine demanding vision-language benchmarks and five video benchmarks." (Section 4.3)). It also enumerates multiple visual domains and video task types in its training/evaluation setup (Section 4.1), but all are defined, benchmarked tasks rather than open-ended or unrestrained multi-domain learning.

## 1. Basic Metadata

- Title: Head-Wise Adaptive Rotary Positional Encoding for Fine-Grained Image Generation
- Authors (verbatim from PDF header): "Jiaye Li1∗, Baoyou Chen1∗, Hui Li1 , Zilong Dong2 , Jingdong Wang3 , Siyu Zhu1,4†"
- Year: 2025 (arXiv date); conference publication listed as ICLR 2026
- Venue: ICLR 2026 (conference paper) and arXiv (arXiv:2510.10489v1)

Evidence (header): "Published as a conference paper at ICLR 2026" and "arXiv:2510.10489v1 [cs.CV] 12 Oct 2025" and "Jiaye Li1∗, Baoyou Chen1∗, Hui Li1 , Zilong Dong2 , Jingdong Wang3 , Siyu Zhu1,4†"

## 2. One-Sentence Contribution Summary

The paper proposes HARoPE, a head-wise adaptive extension of RoPE that inserts an SVD-parameterized linear transformation before the rotary mapping to improve fine-grained image generation while preserving RoPE’s relative-position property.

Evidence (Abstract): "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping." and "This lightweight modification enables dynamic frequency reallocation, semantic alignment of rotary planes, and head-specific positional receptive fields while rigorously preserving RoPE’s relative-position property."

## 3. Tasks Evaluated

Task 1:
- Task name: Image understanding (ImageNet)
- Task type: Classification
- Dataset(s): ImageNet
- Domain: Natural images

Task evidence:
- Section 4 Experiments: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation."
- Section 4.1 Experimental Setups (Dataset): "Image understanding experiments use ImageNet at 224 × 224 with standard resize and center-crop."
- Section 4.1 Experimental Setups (Implementation): "For image understanding, we train ViT-B from scratch with AdamW, learning rate 5 × 10−4 and a 5-epoch warmup from 1 × 10−6 , batch size 256, and 300 training epochs."
- Section 4.1 Experimental Setups (Metrics): "For image understanding, we report Top-1 accuracy."

Task 2:
- Task name: Class-conditional ImageNet generation
- Task type: Generation
- Dataset(s): ImageNet
- Domain: Natural images

Task evidence:
- Section 4 Experiments: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation."
- Section 4.1 Experimental Setups (Implementation): "For class-conditional image generation, we use DiT-B/2 with a constant learning rate 1 × 10−4 , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation."
- Section 4.1 Experimental Setups (Dataset): "For ImageNet generation, we encode images using Stable Diffusion’s VAE into z ∈ RH/8×W/8×4 with H ∈ {128, 256, 512}."
- Section 4.2 Comparison to Existing Works: "Class-Conditioned ImageNet Generation. On ImageNet with DiT-B/2 (Table 2), HARoPE attains the lowest FID-50K (8.90) and the highest IS (127.01), while matching the strongest Precision (0.74) and achieving the best Recall (0.55)."

Task 3:
- Task name: Text-to-image generation (FLUX, MMDiT)
- Task type: Generation
- Dataset(s): BLIP30-60k instruction-tuning set (60k prompt–image pairs); MS-COCO (train split)
- Domain: Natural images with text prompts

Task evidence:
- Section 4 Experiments: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation."
- Abstract: "Extensive experiments on class-conditional ImageNet and text-to-image generation (Flux and MMDiT) demonstrate that HARoPE consistently improves performance over strong RoPE baselines and other extensions."
- Section 4.1 Experimental Setups (Implementation): "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate 2 × 10−5 , weight decay 0.01, and batch size 64."
- Section 4.1 Experimental Setups (Dataset): "Text-to-image experiments with the FLUX model use the BLIP30-60k instruction-tuning set of 60k prompt–image pairs. For MMDiT-based text-to-image generation, we utilize the train split of the MS-COCO dataset Lin et al. (2014)."

## 4. Domain and Modality Scope

- Single domain: Yes (image domain).
  Evidence (Appendix A.1): "Our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation."
- Multiple domains within the same modality: Not specified in the paper.
- Multiple modalities: Yes (text-to-image implies text + image).
  Evidence (Section 4.1 Experimental Setups, Dataset): "Text-to-image experiments with the FLUX model use the BLIP30-60k instruction-tuning set of 60k prompt–image pairs."
- Domain generalization or cross-domain transfer claims: Not claimed. (No explicit claim of cross-domain transfer; extrapolation is only across resolutions within ImageNet.)
  Evidence (Section 4.3 Extrapolation): "Models are trained on the standard ImageNet-1k resolution of 224 × 224 and tested at progressively larger resolutions."

## 5. Model Sharing Across Tasks

Different backbones are used per task (ViT-B, DiT-B/2, FLUX.1-dev, MMDiT), with no joint multi-task training described.

Evidence (Section 4.1 Experimental Setups): "For image understanding, we train ViT-B from scratch with AdamW, learning rate 5 × 10−4 and a 5-epoch warmup from 1 × 10−6 , batch size 256, and 300 training epochs. For class-conditional image generation, we use DiT-B/2 with a constant learning rate 1 × 10−4 , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation. For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate 2 × 10−5 , weight decay 0.01, and batch size 64."

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image understanding (ViT-B) | No (separate backbone) | No (trained from scratch) | Not specified | "For image understanding, we train ViT-B from scratch with AdamW, learning rate 5 × 10−4 and a 5-epoch warmup from 1 × 10−6 , batch size 256, and 300 training epochs." |
| Class-conditional ImageNet generation (DiT-B/2) | No (separate backbone) | Not specified | Not specified | "For class-conditional image generation, we use DiT-B/2 with a constant learning rate 1 × 10−4 , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." |
| Text-to-image generation (FLUX.1-dev) | No (separate backbone) | Yes | Not specified | "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate 2 × 10−5 , weight decay 0.01, and batch size 64." |
| Text-to-image generation (MMDiT) | No (separate backbone) | Not specified | Not specified | "For MMDiT-based text-to-image generation, we utilize the train split of the MS-COCO dataset Lin et al. (2014)." |

## 6. Input and Representation Constraints

Explicit constraints stated in the paper:
- Fixed training resolution and resizing/cropping for ImageNet understanding: "Image understanding experiments use ImageNet at 224 × 224 with standard resize and center-crop." (Section 4.1 Experimental Setups, Dataset)
- ImageNet generation uses VAE-encoded latents with specific resolutions: "For ImageNet generation, we encode images using Stable Diffusion’s VAE into z ∈ RH/8×W/8×4 with H ∈ {128, 256, 512}." (Section 4.1 Experimental Setups, Dataset)
- Text-to-image high-resolution setting: "as shown in Table 3, when integrated into the large-scale FLUX model for text-to-image generation at a high resolution of 1024 × 1024, HARoPE again yields improved performance on both the GenEval and DPG-Bench metrics compared to the original RoPE." (Section 4.3)
- Resolution extrapolation across training/evaluation: "Models are trained on the standard ImageNet-1k resolution of 224 × 224 and tested at progressively larger resolutions." (Section 4.3 Extrapolation)
- Explicit 2D positional formulation for images: "For 2D positions (x, y), a standard extension partitions the feature dimensions across axes and applies independent rotations:" (Section 3.1)

Not specified in the paper:
- Fixed patch size
- Fixed number of tokens
- Padding requirements (beyond resize/center-crop)

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost (windowing, pooling, pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)

Mechanism and type:
- HARoPE is a rotary positional encoding with a head-wise adaptive linear transform before the rotary map.
  Evidence (Abstract): "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping."
- RoPE is relative-position preserving: "RoPE encodes absolute positions via complex-plane rotations while preserving a strict relative-offset property in attention Su et al. (2024)." (Section 2 Related Works)

Where applied:
- Applied to queries and keys immediately before the rotary mapping, per head.
  Evidence (Section 3.3): "HARoPE inserts, for each attention head h with per-head dimension d, a learnable linear transform Ah ∈ Rd×d immediately before the rotary map." and "q′h = Rm Ah qh ,      k′h = Rn Ah kh ." and "The same Ah is applied to queries and keys, and position dependence remains confined to the rotary maps, HARoPE preserves strict relative-offset dependence:"

Fixed vs modified across experiments:
- Positional encoding is varied across experiments (APE, 2D-RoPE variants, STRING/Rethinking RoPE, HARoPE).
  Evidence (Section 4.1 Baselines): "We compare against strong positional encoding baselines. For image understanding, we include absolute positional embeddings (APE), 2D-RoPE in axial and mixed forms (Heo et al., 2024), STRING (Schenck et al., 2025)/Rethinking RoPE (Liu et al., 2025), and HARoPE. For class-conditional generation on ImageNet, we evaluate APE, Vanilla RoPE, 2D-RoPE (Axial), VideoRoPE (Wei et al., 2025), STRING/Rethinking RoPE, and HARoPE."

Not specified in the paper:
- Whether positional encoding is applied only at input vs every layer (beyond the per-head attention description)
- Positional encoding as attention bias

## 9. Positional Encoding as a Variable

- Positional encoding is a core research variable: Yes.
  Evidence (Abstract): "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping."
- Multiple positional encodings compared: Yes.
  Evidence (Section 4.1 Baselines): "We compare against strong positional encoding baselines. For image understanding, we include absolute positional embeddings (APE), 2D-RoPE in axial and mixed forms (Heo et al., 2024), STRING (Schenck et al., 2025)/Rethinking RoPE (Liu et al., 2025), and HARoPE. For class-conditional generation on ImageNet, we evaluate APE, Vanilla RoPE, 2D-RoPE (Axial), VideoRoPE (Wei et al., 2025), STRING/Rethinking RoPE, and HARoPE."
- Multiple HARoPE parameterizations compared: Yes.
  Evidence (Section 4.3 Ablation Study): "We conduct an ablation study to evaluate the impact of the matrix parameterization in HARoPE’s adaptation module."
- Claim that PE choice is “not critical” or secondary: Not stated in the paper.

## 10. Evidence of Constraint Masking

Model sizes / model families mentioned:
- "For image understanding, we train ViT-B from scratch with AdamW, learning rate 5 × 10−4 and a 5-epoch warmup from 1 × 10−6 , batch size 256, and 300 training epochs." (Section 4.1 Implementation)
- "For class-conditional image generation, we use DiT-B/2 with a constant learning rate 1 × 10−4 , no weight decay, batch size 256, and EMA with decay 0.9999 for evaluation." (Section 4.1 Implementation)
- "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model for 4,000 iterations using LoRA (rank 32), AdamW with learning rate 2 × 10−5 , weight decay 0.01, and batch size 64." (Section 4.1 Implementation)
- "For MMDiT-based text-to-image generation, we utilize the train split of the MS-COCO dataset Lin et al. (2014)." (Section 4.1 Dataset)

Dataset sizes / scale:
- BLIP30-60k size is explicit: "Text-to-image experiments with the FLUX model use the BLIP30-60k instruction-tuning set of 60k prompt–image pairs." (Section 4.1 Dataset)
- ImageNet and MS-COCO are named, but sizes are not specified in the paper beyond dataset names.

Attribution of gains:
- Improvements are attributed to the architectural positional-encoding change (HARoPE), not to scaling model size or data.
  Evidence (Abstract): "Extensive experiments on class-conditional ImageNet and text-to-image generation (Flux and MMDiT) demonstrate that HARoPE consistently improves performance over strong RoPE baselines and other extensions." and "This lightweight modification enables dynamic frequency reallocation, semantic alignment of rotary planes, and head-specific positional receptive fields while rigorously preserving RoPE’s relative-position property."

No explicit claim that scaling model size or scaling data is the primary source of gains is stated in the paper.

## 11. Architectural Workarounds

Architectural techniques introduced:
- Head-wise adaptive linear transformation before rotary mapping (SVD-parameterized).
  Evidence (Abstract): "We propose HARoPE, a head-wise adaptive extension that inserts a learnable linear transformation parameterized via singular value decomposition (SVD) before the rotary mapping."
- Head-specific specialization and cross-axis mixing.
  Evidence (Introduction): "By projecting queries and keys through this SVD-based change of basis, HARoPE aligns rotary planes with semantically meaningful directions and facilitates explicit cross-axis mixing. Moreover, endowing each attention head with an independent SVD equips the model with specialized positional receptive fields, promoting complementary multi-scale behaviors. Crucially, using the same adaptation for queries and keys preserves RoPE’s offset equivariance, encouraging that attention depends on positions only through relative differences."
- Lightweight drop-in modification.
  Evidence (Abstract): "This lightweight modification enables dynamic frequency reallocation, semantic alignment of rotary planes, and head-specific positional receptive fields while rigorously preserving RoPE’s relative-position property." and "The method serves as an effective drop-in replacement, offering a principled and adaptable solution for enhancing positional awareness in transformer-based image generative models."

No windowed attention, hierarchical stages, or token pooling/merging are described for this method.

## 12. Explicit Limitations and Non-Claims

Stated limitations / future work:
- Domain limitation: "Our evaluation is primarily confined to the image domain due to our computational constraints; the generalizability of the approach to other multi-dimensional data modalities, such as video, audio, or 3D content, remains an open question for empirical validation." (Appendix A.1)
- Static (non-input-conditional) adaptation: "Another consideration is the static nature of the learned transformation matrices, which are fixed after training. Although the head-wise specialization is beneficial, the adaptation process is not input-conditional." (Appendix A.1)
- Suggested future work: "Exploring dynamic transformations that can adapt based on input content or evolve during inference could further enhance the flexibility and performance of the positional encoding mechanism." (Appendix A.1)

Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

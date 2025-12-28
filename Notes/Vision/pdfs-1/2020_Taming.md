# Survey Answers: Taming Transformers for High-Resolution Image Synthesis

## 1. Basic Metadata
- Title: Taming Transformers for High-Resolution Image Synthesis
  - Evidence (Title page): "Taming Transformers for High-Resolution Image Synthesis"
- Authors: Patrick Esser; Robin Rombach; Björn Ommer
  - Evidence (Title page): "Patrick Esser*    Robin Rombach*      Björn Ommer"
- Year: 2021
  - Evidence (Title page): "arXiv:2012.09841v3 [cs.CV] 23 Jun 2021"
- Venue: arXiv
  - Evidence (Title page): "arXiv:2012.09841v3 [cs.CV] 23 Jun 2021"

## 2. One-Sentence Contribution Summary
The paper claims to address the infeasibility of applying transformers directly to high-resolution images by representing images with a learned discrete codebook and modeling their compositions with a transformer to enable high-resolution image synthesis.

## 3. Tasks Evaluated

- Task: Unconditional image modeling (generation)
  - Task type: Generation
  - Dataset(s): ImageNet (IN), Restricted ImageNet (RIN), LSUN Churches and Towers (LSUN-CT)
  - Domain: RIN = animal classes; LSUN-CT = churches and towers; ImageNet domain not specified in the paper.
  - Evidence (Section 4.1): "Results Tab. 1 reports results for unconditional image modeling on ImageNet (IN) [14], Restricted ImageNet (RIN) [65], consisting of a subset of animal classes from ImageNet, LSUN Churches and Towers (LSUN-CT) [79]"

- Task: Conditional image modeling from depth maps (depth-to-image)
  - Task type: Generation
  - Dataset(s): RIN (D-RIN)
  - Domain: animal classes (RIN)
  - Evidence (Section 4.1): "and for conditional image modeling of RIN conditioned on depth maps obtained with the approach of [60] (D-RIN)"
  - Evidence (Figure 6 caption, Section 4.2): "Top: Depth-to-image on RIN"

- Task: Conditional image modeling from semantic layouts
  - Task type: Generation
  - Dataset(s): S-FLCKR
  - Domain: landscapes
  - Evidence (Section 4.1): "and of landscape images collected from Flickr conditioned on semantic layouts (S-FLCKR) obtained with the approach of [7]."

- Task: Semantic image synthesis (semantic segmentation mask -> image)
  - Task type: Generation
  - Dataset(s): ADE20K, S-FLCKR, COCO-Stuff
  - Domain: S-FLCKR = web-scraped landscapes; ADE20K and COCO-Stuff domains not specified in the paper.
  - Evidence (Section 4.2): "(i): Semantic image synthesis, where we condition on semantic segmentation masks of ADE20K [83], a web-scraped landscapes dataset (S-FLCKR) and COCO-Stuff [6]."

- Task: Structure-to-image (depth or edge information -> image)
  - Task type: Generation
  - Dataset(s): RIN, IN (ImageNet)
  - Domain: RIN = animal classes; IN domain not specified in the paper.
  - Evidence (Section 4.2): "(ii): Structure-to-image, where we use either depth or edge information to synthesize images from both RIN and IN (see Sec. 4.1)."
  - Evidence (Figure 6 caption, Section 4.2): "bottom: Edge-guided synthesis on IN."

- Task: Pose-guided person generation
  - Task type: Generation
  - Dataset(s): DeepFashion
  - Domain: person images (explicitly described as person generation)
  - Evidence (Figure 4 caption, Section 4.2): "4th row: Pose-guided person generation on DeepFashion."
  - Evidence (Section 4.2): "(iii): Pose-guided synthesis: Instead of using the semantically rich information of either segmentation or depth maps, Fig. 4 shows that the same approach as for the previous experiments can be used to build a shape-conditional generative model on the DeepFashion [45] dataset."

- Task: Stochastic superresolution
  - Task type: Reconstruction, Generation
  - Dataset(s): ImageNet (IN)
  - Domain: Not specified in the paper.
  - Evidence (Section 4.2): "(iv): Stochastic superresolution, where low-resolution images serve as the conditioning information and are thereby upsampled. We train our model for an upsampling factor of 8 on ImageNet and show results in Fig. 6."
  - Evidence (Figure 6 caption, Section 4.2): "2nd row: Stochastic superresolution on IN"

- Task: Class-conditional image synthesis
  - Task type: Generation
  - Dataset(s): RIN, IN (ImageNet)
  - Domain: RIN = animal classes; IN domain not specified in the paper.
  - Evidence (Section 4.2): "(v): Class-conditional image synthesis: Here, the conditioning information c is a single index describing the class label of interest. Results for the RIN and IN dataset are demonstrated in Fig. 4 and Fig. 8, respectively."

- Task: Class-conditional image synthesis (ImageNet, explicit training)
  - Task type: Generation
  - Dataset(s): ImageNet
  - Domain: Not specified in the paper.
  - Evidence (Section 4.4): "we train a class-conditional ImageNet transformer on 256 × 256 images"

- Task: Unconditional face synthesis
  - Task type: Generation
  - Dataset(s): FacesHQ (CelebA-HQ + FFHQ)
  - Domain: faces
  - Evidence (Section 4.3): "Results Fig. 7 shows results for unconditional synthesis of faces on FacesHQ, the combination of CelebA-HQ [31] and FFHQ [33]."

- Task: High-resolution unconditional image generation
  - Task type: Generation
  - Dataset(s): LSUN-CT
  - Domain: churches and towers
  - Evidence (Section 4.2, High-Resolution Synthesis): "We evaluate this approach on unconditional image generation on LSUN-CT and FacesHQ (see Sec. 4.3)"

- Task: High-resolution conditional synthesis
  - Task type: Generation
  - Dataset(s): D-RIN, COCO-Stuff, S-FLCKR
  - Domain: RIN = animal classes; S-FLCKR = landscapes; COCO-Stuff domain not specified in the paper.
  - Evidence (Section 4.2, High-Resolution Synthesis): "and conditional synthesis on D-RIN, COCO-Stuff and S-FLCKR"

- Task: Image completion (half completions)
  - Task type: Generation
  - Dataset(s): S-FLCKR
  - Domain: landscapes
  - Evidence (Supplementary, Figure 27 caption): "we use our f = 16 S-FLCKR model to obtain high-fidelity image completions of the inputs depicted on the left (half completions)."

- Task: Pixel-space modeling comparison on CIFAR10
  - Task type: Generation
  - Dataset(s): CIFAR10
  - Domain: Not specified in the paper.
  - Evidence (Section 4.4): "we follow [8] and learn a dictionary of 512 RGB values on CIFAR10 to operate directly on pixel space and train the same transformer architecture on top of our VQGAN with a latent code of size 16 × 16 = 256."

## 4. Domain and Modality Scope
- Scope: Multiple domains within the same modality (images).
  - Evidence (Section 4.1): "Results Tab. 1 reports results for unconditional image modeling on ImageNet (IN) [14], Restricted ImageNet (RIN) [65], consisting of a subset of animal classes from ImageNet, LSUN Churches and Towers (LSUN-CT) [79]"
  - Evidence (Section 4.2): "(i): Semantic image synthesis, where we condition on semantic segmentation masks of ADE20K [83], a web-scraped landscapes dataset (S-FLCKR) and COCO-Stuff [6]."
- Modality: 2D RGB images.
  - Evidence (Section 3.1): "any image x ∈ R^{H×W×3} can be represented by a spatial collection of codebook entries zq ∈ R^{h×w×nz}"
- Domain generalization / cross-domain transfer: Not claimed.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Unconditional image modeling (IN/RIN/LSUN-CT) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Conditional image modeling from depth maps (D-RIN) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Conditional image modeling from semantic layouts (S-FLCKR) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Semantic image synthesis (ADE20K/S-FLCKR/COCO-Stuff) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Structure-to-image (depth/edge on RIN/IN) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Pose-guided synthesis (DeepFashion) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Stochastic superresolution (ImageNet) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Class-conditional synthesis (RIN/IN/ImageNet) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Unconditional face synthesis (FacesHQ) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Image completion (S-FLCKR) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |
| Pixel-space modeling comparison (CIFAR10) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "All of these examples make use of the same methodology. Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths." (Section 4.2) |

## 6. Input and Representation Constraints
- Images are treated as 2D RGB arrays and encoded into discrete spatial codes.
  - Evidence (Section 3.1): "any image x ∈ R^{H×W×3} can be represented by a spatial collection of codebook entries zq ∈ R^{h×w×nz}"
- The image representation is a sequence of codebook indices with length h · w.
  - Evidence (Section 3.1): "An equivalent representation is a sequence of h · w indices which specify the respective entries in the learned codebook."
- Downsampling factor sets spatial code resolution.
  - Evidence (Section 3.2): "we can adapt the number of downsampling blocks m of our VQGAN to reduce images of size H × W to h = H/2m × w = W/2m"
- Fixed sequence length for typical training (16 × 16 tokens).
  - Evidence (Section 4.1): "we usually set |Z|= 1024 and train all subsequent transformer models to predict sequences of length 16 · 16, as this is the maximum feasible length to train a GPT2-medium architecture (307 M parameters) [58] on a GPU with 12GB VRAM."
- Cropping to fixed size for transformer inputs.
  - Evidence (Section 4.3): "During training, we always crop images to obtain inputs of size 16 × 16 for the transformer, i.e. when modeling images with a factor f in the first stage, we use crops of size 16f × 16f."
- Patch-wise training and sliding-window sampling for high-resolution images.
  - Evidence (Section 3.2): "we therefore have to work patch-wise and crop images to restrict the length of s to a maximally feasible size during training. To sample images, we then use the transformer in a sliding-window manner as illustrated in Fig. 3."
- Conditioning information with spatial extent is encoded as a second codebook and prepended to the token sequence.
  - Evidence (Section 3.2): "If the conditioning information c has spatial extent, we first learn another VQGAN to obtain again an index-based representation r ∈ {0, . . . , |Zc|−1}hc × wc with the newly obtained codebook Zc. Due to the autoregressive structure of the transformer, we can then simply prepend r to s and restrict the computation of the negative log-likelihood to entries p(si | s<i , r)."
- Fixed patch size / padding requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length (typical training): 16 × 16 tokens.
  - Evidence (Section 4.1): "we usually set |Z|= 1024 and train all subsequent transformer models to predict sequences of length 16 · 16"
- Sequence length constrained by transformer attention; patch-wise cropping used.
  - Evidence (Section 3.2): "The attention mechanism of the transformer puts limits on the sequence length h · w of its inputs s."
  - Evidence (Section 3.2): "we therefore have to work patch-wise and crop images to restrict the length of s to a maximally feasible size during training."
- High-resolution sampling uses a sliding-window (windowed) approach.
  - Evidence (Section 3.2): "To sample images, we then use the transformer in a sliding-window manner as illustrated in Fig. 3."
  - Evidence (Figure 3 caption): "Figure 3. Sliding attention window."
- Attention heads: 16 heads used in the transformer.
  - Evidence (Supplementary, Table 8 caption): "For every experiment, we set the number of attention heads in the transformer to nh = 16."
- Attention type beyond sliding-window: Not specified in the paper.

## 8. Positional Encoding (Critical Section)
Not specified in the paper.
- Evidence (Supplementary, Sec. B): "Transformer Architecture Our transformer model is identical to the GPT2 architecture [58]" (no explicit positional encoding described).

## 9. Positional Encoding as a Variable
Not specified in the paper. No ablations or comparisons of positional encoding are described.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Model size scaling is discussed (85M to 310M parameters).
  - Evidence (Section 4.1): "we vary the model capacities between 85M and 310M parameters"
- Maximum feasible sequence length tied to model capacity / hardware limits.
  - Evidence (Section 4.1): "we usually set |Z|= 1024 and train all subsequent transformer models to predict sequences of length 16 · 16, as this is the maximum feasible length to train a GPT2-medium architecture (307 M parameters) [58] on a GPU with 12GB VRAM."
- Parameter scaling comparison to prior work.
  - Evidence (Section 4.4): "Note that our model uses ≃ 10× less parameters than VQVAE-2, which has an estimated parameter count of 13.5B"
- Performance gains attributed to context-rich vocabularies / architectural compression rather than simply scaling data or parameters.
  - Evidence (Section 4.3): "For small receptive fields, or equivalently small f, the model cannot capture coherent structures. For an intermediate value of f = 8, the overall structure of images can be approximated, but inconsistencies of facial features such as a half-bearded face and of viewpoints in different parts of the image arise. Only our full setting of f = 16 can synthesize high-fidelity samples."
  - Evidence (Section 4.4): "We observe improvements of 18.63% for FIDs and 14.08× faster sampling of images."
- Dataset size noted as a bottleneck (data scale constraint).
  - Evidence (Supplementary, Sec. E): "the bottleneck for our approach on face synthesis is given by the dataset size since it has the capacity to almost perfectly fit the training data."

## 11. Architectural Workarounds
- Two-stage architecture with VQGAN codebook + autoregressive transformer and patch discriminator.
  - Evidence (Figure 2 caption, Section 3): "Our approach uses a convolutional VQGAN to learn a codebook of context-rich visual parts, whose composition is subsequently modeled with an autoregressive transformer architecture. A discrete codebook provides the interface between these architectures and a patch-based discriminator enables strong compression while retaining high perceptual quality."
- Downsampling to reduce sequence length.
  - Evidence (Section 3.2): "we can adapt the number of downsampling blocks m of our VQGAN to reduce images of size H × W to h = H/2m × w = W/2m"
- Patch-wise cropping and sliding-window sampling to manage sequence length.
  - Evidence (Section 3.2): "we therefore have to work patch-wise and crop images to restrict the length of s to a maximally feasible size during training. To sample images, we then use the transformer in a sliding-window manner as illustrated in Fig. 3."
- Separate VQGAN for spatial conditioning, prepended to sequence.
  - Evidence (Section 3.2): "If the conditioning information c has spatial extent, we first learn another VQGAN to obtain again an index-based representation r ∈ {0, . . . , |Zc|−1}hc × wc ... we can then simply prepend r to s"
- Attention layer at lowest resolution in VQGAN.
  - Evidence (Section 3.1): "To aggregate context from everywhere, we apply a single attention layer on the lowest resolution."
- Patch-based discriminator in VQGAN.
  - Evidence (Supplementary, Table 7 caption): "For the discriminator, we use a patch-based model as in [28]."
- Reusing VQGAN across tasks (reduces sequence length, avoids task-specific modules).
  - Evidence (Section 4.2): "Instead of requiring task specific architectures or modules, the flexibility of the transformer allows us to learn appropriate interactions for each task, while the VQGAN — which can be reused across different tasks — leads to short sequence lengths."

## 12. Explicit Limitations and Non-Claims
- Reconstruction quality degrades beyond a critical downsampling factor; patch-wise training is required.
  - Evidence (Section 3.2): "we observe degradation of the reconstruction quality beyond a critical value of m, which depends on the considered dataset. To generate images in the megapixel regime, we therefore have to work patch-wise and crop images to restrict the length of s to a maximally feasible size during training."
- High-resolution generation assumes spatially invariant statistics or spatial conditioning information.
  - Evidence (Section 3.2): "Our VQGAN ensures that the available context is still sufficient to faithfully model images, as long as either the statistics of the dataset are approximately spatially invariant or spatial conditioning information is available."
- Best reconstruction FID comes at impractical sequence length.
  - Evidence (Section 4.4): "Using the same hierarchical codebook setting as in VQVAE-2 with our model provides the best reconstruction FID, albeit at the cost of a very long and thus impractical sequence."
- Dataset size is a bottleneck for face synthesis and can lead to overfitting.
  - Evidence (Supplementary, Sec. E): "the bottleneck for our approach on face synthesis is given by the dataset size since it has the capacity to almost perfectly fit the training data."
- Early-stopping optimality is unclear.
  - Evidence (Supplementary, Sec. E): "it is not clear if early-stopping based on it is optimal if one is mainly interested in the quality of samples."
- Ordering of image tokens is not clearly defined a priori.
  - Evidence (Supplementary, Sec. F): "For images and their discrete representations, in contrast, it is not clear which linear ordering to use."

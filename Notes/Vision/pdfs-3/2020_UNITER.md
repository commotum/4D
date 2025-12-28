## 1. Basic Metadata

- Title: "UNITER: UNiversal Image-TExt Representation Learning" (p.1)
- Authors: "Yen-Chun Chen? , Linjie Li? , Licheng Yu? , Ahmed El Kholy" and "Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu" (p.1)
- Year: 2020 (evidence: "arXiv:1909.11740v3 [cs.CV] 17 Jul 2020" (p.1))
- Venue: arXiv (evidence: "arXiv:1909.11740v3 [cs.CV] 17 Jul 2020" (p.1))

## 2. One-Sentence Contribution Summary

Primary contribution (one sentence): The paper introduces UNITER as a universal image-text representation for V+L tasks ("We introduce UNITER, a
powerful UNiversal Image-TExt Representation for V+L tasks." (p.3)).

## 3. Tasks Evaluated

Overall evaluation set: "across six V+L tasks (over nine datasets), including Visual Question" / "Answering, Image-Text Retrieval, Referring Expression Comprehension," / "Visual Commonsense Reasoning, Visual Entailment, and NLVR2 .1" (p.1)

- VQA
  - Task type: Classification
  - Dataset(s): VQA
  - Domain: COCO (Image Src.)
  - Evidence: "In VQA, VCR and NLVR2 tasks, given an input image (or a pair of images)
and a natural language question (or description), the model predicts an answer" (Section 4.1, p.9)
  - Dataset evidence: "1 VQA                  VQA         COCO            204K 1.1M VQA-score" (Table 6, p.19)
  - Task type evidence: "we formulate VQA, VCR,
NLVR2 , Visual Entailment and RE Comprehension as classification problems" (Section 4.1, p.9)

- VCR
  - Task type: Classification; Reasoning / relational
  - Dataset(s): VCR
  - Domain: Movie Clips (Image Src.)
  - Evidence: "In VQA, VCR and NLVR2 tasks, given an input image (or a pair of images)
and a natural language question (or description), the model predicts an answer" (Section 4.1, p.9)
  - Dataset evidence: "2 VCR                  VCR         Movie Clips     110K 290K Accuracy" (Table 6, p.19)
  - Task type evidence: "we formulate VQA, VCR,
NLVR2 , Visual Entailment and RE Comprehension as classification problems" (Section 4.1, p.9)

- NLVR2
  - Task type: Classification; Reasoning / relational
  - Dataset(s): NLVR2
  - Domain: Web Crawled (Image Src.)
  - Evidence: "The goal is to determine whether a
natural language statement is true about the given image pair." (Section A.2, p.20)
  - Dataset evidence: "3 NLVR2                NLVR2       Web Crawled 214K 107K Accuracy" (Table 6, p.19)
  - Task type evidence: "we formulate VQA, VCR,
NLVR2 , Visual Entailment and RE Comprehension as classification problems" (Section 4.1, p.9)

- Visual Entailment (SNLI-VE)
  - Task type: Classification
  - Dataset(s): SNLI-VE
  - Domain: Flickr30K (Image Src.)
  - Evidence: "For Visual Entailment, we evaluate on the SNLI-VE dataset. The
goal is to predict whether a given image semantically entails an input sentence." (Section 4.1, p.9)
  - Dataset evidence: "4 Visual Entailment    SNLI-VE     Flickr30K        31K    507K Accuracy" (Table 6, p.19)
  - Task type evidence: "we formulate VQA, VCR,
NLVR2 , Visual Entailment and RE Comprehension as classification problems" (Section 4.1, p.9)

- Image-Text Retrieval
  - Task type: Other (ranking / retrieval)
  - Dataset(s): COCO; Flickr30K
  - Domain: COCO; Flickr30K (Image Src.)
  - Evidence: "For Image-Text Retrieval,
we consider two datasets (COCO and Flickr30K) and evaluate the model in two
settings: Image Retrieval (IR) and Text Retrieval (TR)." (Section 4.1, p.9)
  - Task type evidence: "Image-Text Retrieval, we formulate it as a ranking problem." (Section 4.1, p.9)
  - Dataset evidence: "COCO        COCO             92K    460K" and "Flickr30K Flickr30K          32K    160K" (Table 6, p.19)

- Referring Expression (RE) Comprehension
  - Task type: Other (grounding / region selection); Classification
  - Dataset(s): RefCOCO; RefCOCO+; RefCOCOg
  - Domain: COCO (Image Src. column for RE Comprehension)
  - Evidence: "Referring Expression
(RE) Comprehension requires the model to select the target from a set of im-
age region proposals given the query description." (Section 4.1, p.9)
  - Dataset evidence: "RefCOCO                      20K    142K" / "6 RE Comprehension RefCOCO+ COCO                    20K    142K Accuracy" / "RefCOCOg                     26K     95K" (Table 6, p.19)
  - Task type evidence: "we formulate VQA, VCR,
NLVR2 , Visual Entailment and RE Comprehension as classification problems" (Section 4.1, p.9)

## 4. Domain and Modality Scope

- Modalities: Multiple modalities (image + text). Evidence: "Given a pair of
image and sentence, UNITER takes the visual regions of the image and textual
tokens of the sentence as inputs." (Section 3.1, p.4)
- Domain scope: Multiple domains within the same modality, as evaluation datasets come from multiple image sources (COCO, Movie Clips, Web Crawled, Flickr30K). Evidence: "1 VQA                  VQA         COCO" / "2 VCR                  VCR         Movie Clips" / "3 NLVR2                NLVR2       Web Crawled" / "4 Visual Entailment    SNLI-VE     Flickr30K" (Table 6, p.19)
- Domain generalization / cross-domain transfer: Not claimed in the paper.

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| VQA | Yes (same pre-trained UNITER) | Yes | Yes (MLP on [CLS]) | "We evaluate UNITER on six V+L tasks11 by transferring the pre-trained model
to each target task and finetuning through end-to-end training." (Section 4, p.8); "For VQA, VCR, NLVR2 , Visual Entailment and Image-Text Retrieval, we ex-
tract the joint embedding of the input image-text pairs via a multi-layer percep-
tron (MLP) from the representation of the [CLS] token." (Section 4.1, p.9) |
| VCR | Yes (same pre-trained UNITER) | Yes | Yes (MLP on [CLS]) | "We evaluate UNITER on six V+L tasks11 by transferring the pre-trained model
to each target task and finetuning through end-to-end training." (Section 4, p.8); "For VQA, VCR, NLVR2 , Visual Entailment and Image-Text Retrieval, we ex-
tract the joint embedding of the input image-text pairs via a multi-layer percep-
tron (MLP) from the representation of the [CLS] token." (Section 4.1, p.9) |
| NLVR2 | Yes (same pre-trained UNITER) | Yes | Yes (MLP on [CLS]) | "We evaluate UNITER on six V+L tasks11 by transferring the pre-trained model
to each target task and finetuning through end-to-end training." (Section 4, p.8); "For VQA, VCR, NLVR2 , Visual Entailment and Image-Text Retrieval, we ex-
tract the joint embedding of the input image-text pairs via a multi-layer percep-
tron (MLP) from the representation of the [CLS] token." (Section 4.1, p.9) |
| Visual Entailment | Yes (same pre-trained UNITER) | Yes | Yes (MLP on [CLS]) | "We evaluate UNITER on six V+L tasks11 by transferring the pre-trained model
to each target task and finetuning through end-to-end training." (Section 4, p.8); "For VQA, VCR, NLVR2 , Visual Entailment and Image-Text Retrieval, we ex-
tract the joint embedding of the input image-text pairs via a multi-layer percep-
tron (MLP) from the representation of the [CLS] token." (Section 4.1, p.9) |
| Image-Text Retrieval | Yes (same pre-trained UNITER) | Yes | Yes (MLP on [CLS]) | "We evaluate UNITER on six V+L tasks11 by transferring the pre-trained model
to each target task and finetuning through end-to-end training." (Section 4, p.8); "For VQA, VCR, NLVR2 , Visual Entailment and Image-Text Retrieval, we ex-
tract the joint embedding of the input image-text pairs via a multi-layer percep-
tron (MLP) from the representation of the [CLS] token." (Section 4.1, p.9) |
| RE Comprehension | Yes (same pre-trained UNITER) | Yes | Yes (region-wise MLP) | "We evaluate UNITER on six V+L tasks11 by transferring the pre-trained model
to each target task and finetuning through end-to-end training." (Section 4, p.8); "For RE Comprehension,
we use the MLP to compute the region-wise alignment scores." (Section 4.1, p.9) |

## 6. Input and Representation Constraints

- Input modalities: "Given a pair of
image and sentence, UNITER takes the visual regions of the image and textual
tokens of the sentence as inputs." (Section 3.1, p.4)
- Visual representation: "Specifically, in Image Embedder, we first use Faster R-CNN2 to extract the
visual features (pooled ROI features) for each region." (Section 3.1, p.4)
- Location features: "We also encode the location
features for each region via a 7-dimensional vector." (Section 3.1, p.4)
- Text representation: "For Text Embedder, we follow BERT [9] and tokenize the
input sentence into WordPieces [51]." (Section 3.1, p.4)
- Position information: "token4 is obtained via summing up its word embedding and position embedding,
followed by another LN layer.5" (Section 3.1, p.5)
- Position/location requirement: "self-attention
mechanism in Transformer is order-less, thus it is necessary to explicitly encode
the positions of tokens and the locations of regions as additional inputs." (Section 3.1, p.4)
- Input resolution: Not specified in the paper.
- Patch size: Not specified in the paper.
- Fixed number of tokens/regions: Not specified; sequence length is dynamically adjusted ("we implement dynamic sequence length
to reduce padding and batch examples by number of input units (text tokens +
image regions)." (Section A.2, p.19))
- Padding / resizing requirements: Not specified beyond dynamic sequence length to reduce padding (Section A.2, p.19).

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed vs variable length: Variable sequence length is used ("we implement dynamic sequence length
to reduce padding and batch examples by number of input units (text tokens +
image regions)." (Section A.2, p.19))
- Attention type: Transformer self-attention ("We adopt Transformer [49] as the core of our model, to leverage its elegant
self-attention mechanism designed for learning contextualized representations." (Section 1, p.2))
- Windowed / hierarchical / sparse attention: Not specified in the paper.
- Cost management mechanisms: "we implement dynamic sequence length
to reduce padding and batch examples by number of input units (text tokens +
image regions)." (Section A.2, p.19)

## 8. Positional Encoding (Critical Section)

- Mechanism: Position embeddings for text and location features for regions.
  - Text: "token4 is obtained via summing up its word embedding and position embedding,
followed by another LN layer.5" (Section 3.1, p.5)
  - Vision: "We also encode the location
features for each region via a 7-dimensional vector." (Section 3.1, p.4)
  - Rationale: "self-attention
mechanism in Transformer is order-less, thus it is necessary to explicitly encode
the positions of tokens and the locations of regions as additional inputs." (Section 3.1, p.4)
- Where applied: Input embedding stage (summed into embeddings before LN in the text and image embedders). Evidence: "token4 is obtained via summing up its word embedding and position embedding,
followed by another LN layer.5" (Section 3.1, p.5) and "The final visual embedding for each region is obtained
by summing up the two FC outputs and then passing through a layer normal-
ization (LN) layer." (Section 3.1, p.4)
- Fixed / modified / ablated: Not specified in the paper.
- Absolute vs relative / RoPE / axial / bias-based: Not specified in the paper.

## 9. Positional Encoding as a Variable

- Core research variable vs fixed assumption: Positional encoding is described as part of the input embedding design, with no indication that it is treated as a research variable ("token4 is obtained via summing up its word embedding and position embedding,
followed by another LN layer.5" (Section 3.1, p.5)).
- Multiple positional encodings compared: Not specified in the paper.
- Claim that PE choice is not critical: Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)

- Model size(s): "We report exper-
imental results on two model sizes: UNITER-base with 12 layers and UNITER-
large with 24 layers." (Section 4, p.8) and "UNITER-base: L=12, H=768, A=12, Total Parameters=86M. UNITER-large:" / "L=24, H=1024, A=16, Total Parameters=303M" (Section 4, p.8)
- Dataset size(s): "we obtain 5.6M image-
text pairs for training and 131K image-text pairs for our internal validation," (Section 3.3, p.8)
- Data scaling evidence: "With doubled data size, the model continues to improve
(405.24 in L14)." (Section 4.2, p.11)
- Training tricks as drivers: "We observe significant performance improve-
ments from adding WRA, especially on VQA and RefCOCO+." (Section 4.2, p.10) and "This indicates that the conditional masking strategy enables the model
to learn better joint image-text representations effectively." (Section 4.2, p.10)
- Scaling model size / architectural hierarchy: Not explicitly attributed to performance gains in the paper.

## 11. Architectural Workarounds

- Region-based visual tokens (vs. raw pixels): "Specifically, in Image Embedder, we first use Faster R-CNN2 to extract the
visual features (pooled ROI features) for each region." (Section 3.1, p.4)
- Explicit region location encoding: "We also encode the location
features for each region via a 7-dimensional vector." (Section 3.1, p.4)
- Conditional masking (avoid masking both modalities): "Note that each time we only mask one modality while
keeping the other modality intact" (Section 3.1, p.5)
- Word-Region Alignment via Optimal Transport: "we propose
WRA via the use of Optimal Transport" (Section 3.1, p.5)
- Dynamic sequence length to reduce padding: "we implement dynamic sequence length
to reduce padding and batch examples by number of input units (text tokens +
image regions)." (Section A.2, p.19)
- Task-specific MLP heads: "For VQA, VCR, NLVR2 , Visual Entailment and Image-Text Retrieval, we ex-
tract the joint embedding of the input image-text pairs via a multi-layer percep-
tron (MLP) from the representation of the [CLS] token." and "For RE Comprehension,
we use the MLP to compute the region-wise alignment scores." (Section 4.1, p.9)

## 12. Explicit Limitations and Non-Claims

- Limitation on pre-training input format: "Since UNITER
only handles one image and one text input at pre-training" (Section A.2, p.20)
- Object detector contamination caveat: "Strictly, our object detector is not allowed to train with
these val/test images." (Section A.2, p.21)
- Future work: "Future work includes studying early interaction between raw image pixels
and sentence tokens, as well as developing more effective pre-training tasks." (Section 5, p.14)
- Future work on strict features: "We leave this study and RE comprehension with strictly
correct features to future work." (Section A.2, p.21)

## 13. Constraint Profile (Synthesis)

- Domain scope: Multiple datasets with different image sources (COCO, Movie Clips, Web Crawled, Flickr30K), but all within the image-text modality family.
- Task structure: Six predefined V+L tasks evaluated via classification or ranking heads; no open-ended multi-domain task mix.
- Representation rigidity: Fixed use of region-level Faster R-CNN features plus 7D location vectors and WordPiece tokens with position embeddings.
- Model sharing vs specialization: One pre-trained UNITER backbone is fine-tuned per task with MLP heads (including region-wise heads for RE).
- Positional encoding: Position embeddings and region location features are used at input; no alternative PE variants compared.

## 14. Final Classification

Multi-task, multi-domain (constrained).

Justification: The paper evaluates a single pre-trained model across multiple tasks ("across six V+L tasks (over nine datasets), including Visual Question" / "Answering, Image-Text Retrieval, Referring Expression Comprehension," / "Visual Commonsense Reasoning, Visual Entailment, and NLVR2 .1" (p.1)). These tasks are drawn from multiple image sources/domains (e.g., COCO, Movie Clips, Web Crawled, Flickr30K in Table 6), but remain within the constrained image-text setting rather than unrestrained multi-domain learning.

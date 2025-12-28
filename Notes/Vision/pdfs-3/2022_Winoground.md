## 1. Basic Metadata
- Title: "Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality" (title page)
- Authors: "Tristan Thrush¶ *, Ryan Jiang‡ , Max Bartolo§ , Amanpreet Singh¶ , Adina Williams† , Douwe Kiela¶ , Candace Ross† *" (title page)
- Year: 2022 ("arXiv:2204.03162v2 [cs.CV] 22 Apr 2022") (title page)
- Venue: arXiv ("arXiv:2204.03162v2 [cs.CV] 22 Apr 2022") (title page)

## 2. One-Sentence Contribution Summary
- "We present a novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning, which we call Winoground." (Abstract)

## 3. Tasks Evaluated
- Winoground caption selection (text score)
  - Task type: Other (image-text matching / retrieval); Reasoning / relational
  - Dataset(s): Winoground
  - Domain: Images + captions (Getty Images)
  - Evidence: "The first metric is the text score, which measures whether a model can select the correct caption, given an image." (3.2 Metrics)
  - Evidence: "Winoground, for measuring visio-linguistic compositional reasoning, whereby two images and two captions have to be matched correctly; both captions contain exactly the same set of words, ordered in such a way that each describes primarily one of the images." (Introduction)
  - Evidence: "We have secured a license from Getty Images to distribute images for research purposes. Thus, the expert annotators were given access to the Getty Images API [25]," (3.1 Dataset)

- Winoground image selection (image score)
  - Task type: Other (image-text matching / retrieval); Reasoning / relational
  - Dataset(s): Winoground
  - Domain: Images + captions (Getty Images)
  - Evidence: "The second metric is the image score, which measures whether a model can select the correct image, given a caption." (3.2 Metrics)

- Winoground group score (joint matching)
  - Task type: Other (image-text matching / retrieval); Reasoning / relational
  - Dataset(s): Winoground
  - Domain: Images + captions (Getty Images)
  - Evidence: "So, we also evaluate using the group score, where every combination for a given example {(C0 , I0 ), (C0 , I1 ), (C1 , I0 ), (C1 , I1 )} must be correctly scored by the model in order for the example to be considered correct." (3.2 Metrics)

- Dataset size: "Our dataset has 1600 image-text pairs in total, with 800 correct and 800 incorrect pairings. These comprise 400 examples, with 800 unique captions and images." (3.1 Dataset)

## 4. Domain and Modality Scope
- Single domain? The evaluation dataset is built from Getty Images: "We have secured a license from Getty Images to distribute images for research purposes. Thus, the expert annotators were given access to the Getty Images API [25]," (3.1 Dataset)
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Yes (images + text): "Winoground, for measuring visio-linguistic compositional reasoning, whereby two images and two captions have to be matched correctly; both captions contain exactly the same set of words, ordered in such a way that each describes primarily one of the images." (Introduction)
- Domain generalization or cross-domain transfer? Not claimed.

## 5. Model Sharing Across Tasks
All metrics use the same image-caption scoring function and are applied to pretrained models.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Text score (caption selection) | Yes | No (except UniT for image-text alignment) | No (uses existing image-text matching head or similarity score) | "The first metric is the text score, which measures whether a model can select the correct caption, given an image." (3.2 Metrics); "trained with an image-text matching classification head or that produce a similarity score between the two modalities1 ." (4.1 Vision & Language Transformers); "UniT is the only model we selected that was not pretrained on image-text matching. To get image-text alignment scores, we finetuned UniT on image-text matching loss using MS-COCO [48]" (footnote) |
| Image score (image selection) | Yes | No (except UniT for image-text alignment) | No (uses existing image-text matching head or similarity score) | "The second metric is the image score, which measures whether a model can select the correct image, given a caption." (3.2 Metrics); "trained with an image-text matching classification head or that produce a similarity score between the two modalities1 ." (4.1 Vision & Language Transformers); "UniT is the only model we selected that was not pretrained on image-text matching. To get image-text alignment scores, we finetuned UniT on image-text matching loss using MS-COCO [48]" (footnote) |
| Group score (joint matching) | Yes | No (except UniT for image-text alignment) | No (uses existing image-text matching head or similarity score) | "So, we also evaluate using the group score, where every combination for a given example {(C0 , I0 ), (C0 , I1 ), (C1 , I0 ), (C1 , I1 )} must be correctly scored by the model in order for the example to be considered correct." (3.2 Metrics); "trained with an image-text matching classification head or that produce a similarity score between the two modalities1 ." (4.1 Vision & Language Transformers); "UniT is the only model we selected that was not pretrained on image-text matching. To get image-text alignment scores, we finetuned UniT on image-text matching loss using MS-COCO [48]" (footnote) |

## 6. Input and Representation Constraints
- Input pairing: "Let (C0 , I0 ) and (C1 , I1 ) be two image-caption pairs." (3.1 Dataset)
- Caption word-order constraint: "Winoground, for measuring visio-linguistic compositional reasoning, whereby two images and two captions have to be matched correctly; both captions contain exactly the same set of words, ordered in such a way that each describes primarily one of the images." (Introduction)
- Caption ordering focus: "In this work, we study the image-grounding of twin sentences with identical but differently ordered words." (Introduction)
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D)? Not specified in the paper.
- Padding/resizing requirements? Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type (reported for evaluated models): "All single-stream models use merged attention, where the language and visual input attend to both themselves and the other modality." (4.1 Vision & Language Transformers)
- Attention type (contrastive FLAVA/CLIP): "CLIP and the contrastive configuration of FLAVA lack cross-modal attention." (4.1 Vision & Language Transformers)
- Attention type (ViLBERT): "ViLBERT has language-only transformer layers that are then fused by cross-modal transformer layers." (4.1 Vision & Language Transformers)
- Attention type (LXMERT/FLAVA/UniT): "LXMERT, the ITM configuration of FLAVA, and UniT each use language-only and vision-only layers that are also fused by cross-modal transformer layers, which perform a combo of modality-specific attention and co-attention across modalities." (4.1 Vision & Language Transformers)
- Mechanisms to manage computational cost (windowing/pooling/token pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism mentioned (for ViT-based models): "ViT, images are flattened into patches that are linearly projected and combined with a position encoding." (4.1 Vision & Language Transformers)
- Absolute/relative/RoPE/axial/bias/implicit: Not specified in the paper.
- Where applied (input only vs every layer vs attention bias): Not specified in the paper.
- Fixed/modified/ablated across experiments: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Core research variable vs fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- Claims PE choice is not critical/secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)
- Winoground evaluation dataset size: "Our dataset has 1600 image-text pairs in total, with 800 correct and 800 incorrect pairings. These comprise 400 examples, with 800 unique captions and images." (3.1 Dataset)
- Pretraining dataset sizes are reported in millions: "Model                       Datasets                                                             # Images, Captions (Millions)   Architecture    Attention" (Table 2); "FLAVA IT M [68]             COCO, SBU, LN, CC, VG, WIT, CC 12M, RC, YFCC100M                                     70.00, 70.00    dual-stream     modality-specific, merged" (Table 2)
- Data scale correlation with performance: "We find highly significant correlations between the size of the multimodal pretraining dataset and the scores, if we remove CLIP and FLAVA as outliers." (6.3)
- Data scale differences: "CLIP and FLAVA were trained on an order of magnitude more data than the other models." (6.2)
- Attribution of gains: "A potential explanation could be the large-scale pretraining used by CLIP and FLAVA, the large training dataset used to train the object detector for VinVL, or the ViT approach for image features used by ViLT, FLAVA, and CLIP that encodes every portion of the image." (6.1)
- Model sizes: Not specified in the paper.
- Architectural hierarchy vs training tricks as primary driver: Not explicitly specified in the paper.

## 11. Architectural Workarounds
- Note: The paper introduces a dataset and evaluates existing models; no new architectural workarounds are proposed for Winoground itself.
- Region features for some models: "UNITER, ViLLA) [12,23,47,51,76] use region features extracted from the fc6 layer of a Faster R-CNN [58] trained on Visual Genome [43]." (4.1 Vision & Language Transformers)
- ViT patching + position encoding: "ViT, images are flattened into patches that are linearly projected and combined with a position encoding." (4.1 Vision & Language Transformers)
- UniT backbone: "UniT [35] alternatively uses a transformer network [79] on top of a convolutional network following Carion et al. [9]." (4.1 Vision & Language Transformers)
- Attention/fusion schemes across models: "All single-stream models use merged attention, where the language and visual input attend to both themselves and the other modality." (4.1 Vision & Language Transformers); "CLIP and the contrastive configuration of FLAVA lack cross-modal attention." (4.1 Vision & Language Transformers); "ViLBERT has language-only transformer layers that are then fused by cross-modal transformer layers." (4.1 Vision & Language Transformers); "LXMERT, the ITM configuration of FLAVA, and UniT each use language-only and vision-only layers that are also fused by cross-modal transformer layers, which perform a combo of modality-specific attention and co-attention across modalities." (4.1 Vision & Language Transformers)

## 12. Explicit Limitations and Non-Claims
- "Winoground is English-only and translation to other languages may be nontrivial [50]. Expert curation is time-consuming and our dataset is limited in size." (Broader Impact & Limitations)
- "Multimodal datasets containing images of people require thoughtful consideration of how people are represented (see [5] for a detailed analysis of the stereotypes present in many multimodal datasets)." (Broader Impact & Limitations)
- "We used gender underspecified human denoting terms (e.g., person, child) to avoid issues with inferring gender identity from images [61]." (Broader Impact & Limitations)
- "Our annotators disproportionately come from the USA and the same could be true for our crowdworkers." (Broader Impact & Limitations)
- Explicit non-claims about open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

## 1. Basic Metadata
- Title: Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision
- Authors: Chao Jia; Yinfei Yang; Ye Xia; Yi-Ting Chen; Zarana Parekh; Hieu Pham; Quoc V. Le; Yunhsuan Sung; Zhen Li; Tom Duerig
- Year: 2021
- Venue: Proceedings of the 38 th International Conference on Machine Learning (PMLR 139); arXiv:2102.05918v2

## 2. One-Sentence Contribution Summary
- Introduces ALIGN, a simple dual-encoder contrastive model trained on over a billion noisy image alt-text pairs to scale visual and vision-language representation learning without heavy data curation, enabling strong transfer to vision-only classification and vision-language retrieval.

## 3. Tasks Evaluated

### Task: Image-to-text and text-to-image retrieval
- Task type: Retrieval
- Dataset(s): Flickr30K; MSCOCO
- Domain: Not specified in the paper (image-text datasets)
- Evidence (Section 4.2): “We evaluate ALIGN models on image-to-text and text-to-image retrieval tasks, with and without finetuning. Two benchmark datasets are considered: Flickr30K (Plummer et al., 2015) and MSCOCO (Chen et al., 2015).”

### Task: CxC intra-/inter-modal retrieval (image→text, text→image, text→text, image→image)
- Task type: Retrieval
- Dataset(s): Crisscrossed Captions (CxC)
- Domain: Not specified in the paper (extension of MSCOCO)
- Evidence (Section 4.2): “With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-to-image, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS).”

### Task: CxC semantic similarity (STS, SIS, SITS)
- Task type: Reasoning / relational (semantic similarity)
- Dataset(s): Crisscrossed Captions (CxC)
- Domain: Not specified in the paper (caption-caption, image-image, and image-caption pairs)
- Evidence (Section 4.2): “With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-to-image, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS).”

### Task: Zero-shot visual classification (ImageNet + variants)
- Task type: Classification
- Dataset(s): ImageNet ILSVRC-2012; ImageNet-R; ImageNet-A; ImageNet-V2
- Domain: ImageNet-R includes “non-natural images such as art, cartoons, sketches”; ImageNet-A is “more challenging images for ML models”; otherwise not specified
- Evidence (Section 4.3): “We first apply zero-shot transfer of ALIGN to visual classification tasks on ImageNet ILSVRC-2012 benchmark (Deng et al., 2009) and its variants including ImageNet-R(endition) (Hendrycks et al., 2020) (non-natural images such as art, cartoons, sketches), ImageNet-A(dversarial) (Hendrycks et al., 2021) (more challenging images for ML models), and ImageNet-V2 (Recht et al., 2019).”

### Task: Fine-grained visual classification
- Task type: Classification
- Dataset(s): Oxford Flowers-102; Oxford-IIIT Pets; Stanford Cars; Food101 (also ImageNet)
- Domain: Not specified in the paper
- Evidence (Section 4.3): “We also transfer the image encoder to downstream visual classification tasks. For this purpose, we use the ImageNet as well as a handful of smaller fine-grained classification datasets such as Oxford Flowers-102 (Nilsback & Zisserman, 2008), Oxford-IIIT Pets (Parkhi et al., 2012), Stanford Cars (Krause et al., 2013), and Food101 (Bossard et al., 2014).”

### Task: VTAB visual classification (19 tasks)
- Task type: Classification
- Dataset(s): Visual Task Adaptation Benchmark (VTAB)
- Domain: “natural, specialized and structured image classification tasks”
- Evidence (Section 4.3): “Visual Task Adaptation Benchmark (VTAB) (Zhai et al., 2019) which consists of 19 diverse (covering subgroups of natural, specialized and structured image classification tasks) visual classification tasks with 1000 training samples each.”

### Task: Multilingual image-text retrieval (Multi30k)
- Task type: Retrieval
- Dataset(s): Multi30k
- Domain: “multilingual image text retrieval dataset”
- Evidence (Section 8): “We test the multilingual model on Multi30k, a multilingual image text retrieval dataset extends Flickr30K (Plummer et al., 2015) to German (de) (Elliott et al., 2016), French (fr) (Elliott et al., 2017) and Czech (cs) (Barrault et al., 2018).”
- Evidence (Section 8): “The evaluation metric is mean Recall (mR), which computes the average score of Recall@1, Recall@5 and Recall@10 on image-to-text retrieval and text-to-image retrieval tasks.”

### Task: Word similarity (SimLex-999)
- Task type: Other (word similarity)
- Dataset(s): SimLex-999
- Domain: Word pairs / text
- Evidence (Appendix B): “we also evaluate the word representation from ALIGN model5 on SimLex-999 (Hill et al., 2015), which is a task to compare word similarity for 999 word pairs.”

### Task: ImageNet KNN evaluation (ablation)
- Task type: Classification (nearest-neighbor retrieval)
- Dataset(s): ImageNet
- Domain: Not specified in the paper
- Evidence (Section 6): “In the ablation study, we compare model performance mostly on MSCOCO zero-shot retrieval and ImageNet K-Nearest-neighbor (KNN) tasks.”
- Evidence (Section 6 footnote): “For each image in the validation set of ImageNet, we retrieve its nearest neighbors from the training set w/ pre-trained image encoder.”

## 4. Domain and Modality Scope
- Single domain? No. The paper evaluates across different image distributions and task groups, e.g., “ImageNet-R(endition) (Hendrycks et al., 2020) (non-natural images such as art, cartoons, sketches)” and VTAB’s “natural, specialized and structured image classification tasks.” (Section 4.3)
- Multiple domains within the same modality? Yes. Evidence: “ImageNet-R(endition) (Hendrycks et al., 2020) (non-natural images such as art, cartoons, sketches)” and VTAB “natural, specialized and structured image classification tasks.” (Section 4.3)
- Multiple modalities? Yes. Evidence: “The representations can be used for vision-only or vision-language task transfer.” (Figure 1 caption)
- Domain generalization or cross-domain transfer? Not explicitly claimed as domain generalization; the paper does claim robustness across distributions: “ALIGN shows great robustness on classification tasks with different image distributions.” (Section 5.2)

## 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Flickr30K/MSCOCO image↔text retrieval | Yes (same pretrained ALIGN) | Both: zero-shot and fine-tuned per dataset | No | “We evaluate ALIGN models on image-to-text and text-to-image retrieval tasks, with and without finetuning. Two benchmark datasets are considered: Flickr30K (Plummer et al., 2015) and MSCOCO (Chen et al., 2015).” (Section 4.2) |
| CxC retrieval (image↔text, text↔text, image↔image) | Yes (uses MSCOCO fine-tuned ALIGN) | Fine-tuned on MSCOCO; evaluated on CxC | No | “With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-to-image, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS).” and “we can directly evaluate the MSCOCO fine-tuned ALIGN model on CxC annotations.” (Section 4.2) |
| CxC semantic similarity (STS/SIS/SITS) | Yes (uses MSCOCO fine-tuned ALIGN) | Fine-tuned on MSCOCO; evaluated on CxC | No | “With extended annotations, CxC enables four intra- and inter-modal retrieval tasks including image-to-text, text-to-image, text-to-text, and image-to-image retrieval, and three semantic similarity tasks including semantic textual similarity (STS), semantic image similarity (SIS), and semantic image-text similarity (SITS).” and “we can directly evaluate the MSCOCO fine-tuned ALIGN model on CxC annotations.” (Section 4.2) |
| Zero-shot ImageNet (+ variants) classification | Yes (pretrained ALIGN) | No | No | “We first apply zero-shot transfer of ALIGN to visual classification tasks on ImageNet ILSVRC-2012 benchmark (Deng et al., 2009) and its variants including ImageNet-R(endition) (Hendrycks et al., 2020) (non-natural images such as art, cartoons, sketches), ImageNet-A(dversarial) (Hendrycks et al., 2021) (more challenging images for ML models), and ImageNet-V2 (Recht et al., 2019).” and “If we directly feed the texts of classnames into the text encoder, ALIGN is able to classify images into candidate classes via image-text retrieval.” (Sections 4.3, 5.2) |
| ImageNet classification (head-only / full fine-tune) | Yes (pretrained ALIGN image encoder) | Yes | Yes (classification head) | “For ImageNet, results from two settings are reported: training the top classification layer only (with frozen ALIGN image encoder) and fully fine-tuned.” (Section 4.3) |
| Fine-grained visual classification (Oxford Flowers-102, Oxford-IIIT Pets, Stanford Cars, Food101) | Yes (pretrained ALIGN image encoder) | Yes | Yes (classification head) | “We first train the classification head and then fine-tune all layers, except with batch norm statistics frozen.” (Section 5.3) |
| VTAB (19 tasks) | Yes (pretrained ALIGN image encoder) | Yes (per-task training) | Not specified in the paper | “Each task is trained on 800 images” and “After the sweep, the selected hyperparameters are used to train on the combined training and validation splits of 1000 images for each task.” (Section 5.3) |
| Multi30k multilingual retrieval | Separate multilingual model (ALIGNmling) | No (zero-shot) | No | “A multilingual model ALIGNmling is trained using this data.” and “We test the multilingual model on Multi30k, a multilingual image text retrieval dataset extends Flickr30K (Plummer et al., 2015) to German (de) (Elliott et al., 2016), French (fr) (Elliott et al., 2017) and Czech (cs) (Barrault et al., 2018).” and “We evaluate the zero-shot model performance of ALIGN and compare it with M3 P (Huang et al., 2020a) and UC2 (Zhou et al., 2021).” (Section 8) |
| SimLex-999 word similarity | Yes (ALIGN word representation) | No | No | “we also evaluate the word representation from ALIGN model5 on SimLex-999 (Hill et al., 2015), which is a task to compare word similarity for 999 word pairs.” (Appendix B) |
| ImageNet KNN (ablation) | Yes (pretrained image encoder) | No | No | “we compare model performance mostly on MSCOCO zero-shot retrieval and ImageNet K-Nearest-neighbor (KNN) tasks.” and “we retrieve its nearest neighbors from the training set w/ pre-trained image encoder.” (Section 6) |

## 6. Input and Representation Constraints
- Image-size filtering: “we remove pornographic images and keep only images whose shorter dimension is larger than 200 pixels and aspect ratio is smaller than 3.” (Section 3)
- Text-length filtering: “those that are either too short (<3 unigrams) or too long (>20 unigrams).” (Section 3)
- Fixed image resolution for pretraining: “The image encoder is trained at resolution of 289 × 289 pixels no matter what EfficientNet variant is used.” (Section 4.2)
- Resizing/cropping: “We first resize input images to 346 × 346 resolution and then perform random crop (with additional random horizontal flip) in training and central crop in evaluation.” (Section 4.2)
- Text token limit: “For BERT we use wordpiece sequence of maximum 64 tokens since the input texts are no longer than 20 unigrams.” (Section 4.2)
- Train/eval resolution for classification: “Specifically, train/eval resolution is 289/360 with frozen visual features, and is 475/600 when fine-tuning all variables.” (Section 5.3) and “The train/eval resolution is fixed at 289/360.” (Section 5.3)
- Fixed patch size / fixed number of tokens / fixed dimensionality / padding: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: 64 wordpiece tokens. Evidence: “For BERT we use wordpiece sequence of maximum 64 tokens since the input texts are no longer than 20 unigrams.” (Section 4.2)
- Fixed vs variable length: Not specified beyond a maximum length and the “no longer than 20 unigrams” constraint. (Section 4.2)
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost: Not specified for attention; other scaling-related mechanisms include “EfficientNet with global pooling” and “we concatenate embeddings from all computing cores to form a much larger batch.” (Section 4.1)

## 8. Positional Encoding (Critical Section)
- Not specified in the paper. There is no description of a positional encoding mechanism, where it is applied, or whether it is fixed/ablated.

## 9. Positional Encoding as a Variable
- Not specified in the paper. No comparisons or claims about positional encoding being critical or secondary are stated.

## 10. Evidence of Constraint Masking (Scale Compensating for Structure)
- Dataset scale: “The result is a much larger (1.8B image-text pairs) but noisier dataset.” (Section 3) and “datasets: full ALIGN training data, 10% randomly sampled ALIGN training data, and Conceptual Captions (CC-3M, around 3M images).” (Section 6.2)
- Model scaling: “We train EfficientNet from B1 to L2 for the image encoder and BERT-Mini to BERT-Large for the text encoder.” (Section 6.1)
- Scaling claim vs noise: “We show that the scale of our corpus can make up for its noise and leads to state-of-the-art representations even with such a simple learning scheme.” (Abstract)
- Data scaling evidence: “a large scale training set is essential to allow scaling up of our models and to achieve better performance.” (Section 6.2)
- Model size vs data size: “Conversely, a larger model is required to fully utilize the larger dataset – the smaller B3+BERT-mini almost saturate at 10% of ALIGN data, while with the larger B7+BERT-base, there is a clear improvement with full ALIGN data.” (Section 6.2)

## 11. Architectural Workarounds
- Simple dual-encoder for scalable retrieval: “We pre-train ALIGN using a dual-encoder architecture. The model consists of a pair of image and text encoders with a cosine-similarity combination function at the top.” (Section 4.1)
- Efficient encoder choices: “We use EfficientNet with global pooling (without training the 1x1 conv layer in the classification head) as the image encoder and BERT with [CLS] token embedding as the text embedding encoder (we generate 100k wordpiece vocabulary from our training dataset). A fully-connected layer with linear activation is added on top of BERT encoder to match the dimension from the image tower.” (Section 4.1)
- Avoiding expensive cross-attention: “advanced models emerge with cross-modal attention layers (Liu et al., 2019a; Lu et al., 2019; Chen et al., 2020c; Huang et al., 2020b) and show superior performance in image-text matching tasks. However, they are orders of magnitudes slower and hence impractical for image-text retrieval systems in the real world. In contrast, our model inherits the simplest VSE form, but still outperforms all previous cross-attention models in image-text matching benchmarks.” (Section 2)
- Large in-batch negatives across cores: “we concatenate embeddings from all computing cores to form a much larger batch.” (Section 4.1)
- Task-specific classification head during transfer: “we first freeze the learned visual features and only train the classification head. Afterwards we fine-tune all layers.” (Section 5.3)

## 12. Explicit Limitations and Non-Claims
- Intra-modal vs cross-modal imbalance and future work: “We suspect it is because the training objective of ALIGN focuses on cross-modal (image-text) matching instead of intra-modal matching. Parekh et al. (2021) suggest multitask learning could produce more balanced representations. We leave it to the future work.” (Section 5.1)
- Safety/fairness caveats: “additional analysis of the data and the resulting model is necessary before the use of the model in practice.” (Section 10)
- Data balancing and sensitivity concerns: “data balancing efforts may be required to prevent reinforcing stereotypes from the web data. Additional testing and training around sensitive religious or cultural items should be taken to understand and mitigate the impact from possibly mislabeled data.” (Section 10)
- Demographic skew and misuse: “Further analysis should also be taken to ensure that the demographic distribution of humans and related cultural items like clothing, food, and art do not cause model performance to be skewed. Analysis and balancing would be required if such models will be used in production. Finally, unintended misuse of such models for surveillance or other nefarious purposes should be prohibited.” (Section 10)

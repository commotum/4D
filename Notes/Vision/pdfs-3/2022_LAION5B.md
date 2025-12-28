1. Basic Metadata

Title: LAION-5B: An open large-scale dataset for training next generation image-text models.
Evidence: "LAION-5B: An open large-scale dataset for training next generation image-text models" (Title page).

Authors: Christoph Schuhmann; Romain Beaumont; Richard Vencu; Cade Gordon; Ross Wightman; Mehdi Cherti; Theo Coombes; Aarush Katta; Clayton Mullis; Mitchell Wortsman; Patrick Schramowski; Srivatsa Kundurthy; Katherine Crowson; Ludwig Schmidt; Robert Kaczmarczyk; Jenia Jitsev.
Evidence (Title page):
"Christoph Schuhmann1 §§°° Romain Beaumont1 §§°° Richard Vencu1,3,8 §§°°
Cade Gordon2 §§°° Ross Wightman1 §§ Mehdi Cherti 1,10 §§
Theo Coombes1 Aarush Katta1 Clayton Mullis1 Mitchell Wortsman6
Patrick Schramowski1,4,5 Srivatsa Kundurthy1 Katherine Crowson1,8,9
Ludwig Schmidt6 °° Robert Kaczmarczyk1,7 °° Jenia Jitsev1,10 °°"

Year: 2022.
Venue: arXiv.
Evidence: "arXiv:2210.08402v1 [cs.CV] 16 Oct 2022" (Title page).

2. One-Sentence Contribution Summary

The paper introduces LAION-5B, an open dataset of billions of CLIP-filtered image-text pairs to enable large-scale training and replication/fine-tuning of vision-language models.
Evidence: "we present LAION-5B - a dataset consisting of 5.85 billion CLIP-filtered image-text pairs, of which 2.32B contain English language. We show successful replication and fine-tuning of foundational models like CLIP, GLIDE and Stable Diffusion using the dataset" (Abstract).

3. Tasks Evaluated

3.1 Zero-shot classification tasks (VTAB+)
General evidence for task type: "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1).
General evidence for dataset list: "Table 5: Datasets used for zero-shot classification evaluation (VTAB+)." (Appendix E.3).

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k zero-shot classification | Classification | ImageNet-1k | Not specified in the paper. | "ImageNet-1k" (Table 5, Appendix E.3). |
| ImageNet-v2 zero-shot classification | Classification | ImageNet-v2 | Not specified in the paper. | "ImageNet-v2" (Table 5, Appendix E.3). |
| ImageNet-R zero-shot classification | Classification | ImageNet-R | Not specified in the paper. | "ImageNet-R" (Table 5, Appendix E.3). |
| ImageNet Sketch zero-shot classification | Classification | ImageNet Sketch | Not specified in the paper. | "ImageNet Sketch" (Table 5, Appendix E.3). |
| ObjectNet zero-shot classification | Classification | ObjectNet | Not specified in the paper. | "ObjectNet" (Table 5, Appendix E.3). |
| ImageNet-A zero-shot classification | Classification | ImageNet-A | Not specified in the paper. | "ImageNet-A" (Table 5, Appendix E.3). |
| CIFAR-10 zero-shot classification | Classification | CIFAR-10 | Not specified in the paper. | "CIFAR-10" (Table 5, Appendix E.3). |
| CIFAR-100 zero-shot classification | Classification | CIFAR-100 | Not specified in the paper. | "CIFAR-100" (Table 5, Appendix E.3). |
| MNIST zero-shot classification | Classification | MNIST | Not specified in the paper. | "MNIST" (Table 5, Appendix E.3). |
| Oxford Flowers 102 zero-shot classification | Classification | Oxford Flowers 102 | Not specified in the paper. | "Oxford Flowers 102" (Table 5, Appendix E.3). |
| Stanford Cars zero-shot classification | Classification | Stanford Cars | Not specified in the paper. | "Stanford Cars" (Table 5, Appendix E.3). |
| SVHN zero-shot classification | Classification | SVHN | Not specified in the paper. | "SVHN" (Table 5, Appendix E.3). |
| Facial Emotion Recognition 2013 zero-shot classification | Classification | Facial Emotion Recognition 2013 | Not specified in the paper. | "Facial Emotion Recognition 2013" (Table 5, Appendix E.3). |
| RenderedSST2 zero-shot classification | Classification | RenderedSST2 | Not specified in the paper. | "RenderedSST2" (Table 5, Appendix E.3). |
| Oxford-IIIT Pets zero-shot classification | Classification | Oxford-IIIT Pets | Not specified in the paper. | "Oxford-IIIT Pets" (Table 5, Appendix E.3). |
| Caltech-101 zero-shot classification | Classification | Caltech-101 | Not specified in the paper. | "Caltech-101" (Table 5, Appendix E.3). |
| Pascal VOC 2007 Classification zero-shot classification | Classification | Pascal VOC 2007 Classification | Not specified in the paper. | "Pascal VOC 2007 Classification" (Table 5, Appendix E.3). |
| SUN397 zero-shot classification | Classification | SUN397 | Not specified in the paper. | "SUN397" (Table 5, Appendix E.3). |
| FGVC Aircraft zero-shot classification | Classification | FGVC Aircraft | Not specified in the paper. | "FGVC Aircraft" (Table 5, Appendix E.3). |
| Country211 zero-shot classification | Classification | Country211 | Not specified in the paper. | "Country211" (Table 5, Appendix E.3). |
| Describable Textures zero-shot classification | Classification | Describable Textures | Not specified in the paper. | "Describable Textures" (Table 5, Appendix E.3). |
| GTSRB zero-shot classification | Classification | GTSRB | Not specified in the paper. | "GTSRB" (Table 5, Appendix E.3). |
| STL10 zero-shot classification | Classification | STL10 | Not specified in the paper. | "STL10" (Table 5, Appendix E.3). |
| Diabetic Retinopathy zero-shot classification | Classification | Diabetic Retinopathy | Not specified in the paper. | "Diabetic Retinopathy" (Table 5, Appendix E.3). |
| EuroSAT zero-shot classification | Classification | EuroSAT | Not specified in the paper. | "EuroSAT" (Table 5, Appendix E.3). |
| RESISC45 zero-shot classification | Classification | RESISC45 | Not specified in the paper. | "RESISC45" (Table 5, Appendix E.3). |
| PatchCamelyon zero-shot classification | Classification | PatchCamelyon | Not specified in the paper. | "PatchCamelyon" (Table 5, Appendix E.3). |
| CLEVR Counts zero-shot classification | Classification | CLEVR Counts | Not specified in the paper. | "CLEVR Counts" (Table 5, Appendix E.3). |
| CLEVR Object Distance zero-shot classification | Classification | CLEVR Object Distance | Not specified in the paper. | "CLEVR Object Distance" (Table 5, Appendix E.3). |
| DSPRITES Orientation zero-shot classification | Classification | DSPRITES Orientation | Not specified in the paper. | "DSPRITES Orientation" (Table 5, Appendix E.3). |
| DSPRITES Position zero-shot classification | Classification | DSPRITES Position | Not specified in the paper. | "DSPRITES Position" (Table 5, Appendix E.3). |
| SmallNORB Elevation zero-shot classification | Classification | SmallNORB Elevation | Not specified in the paper. | "SmallNORB Elevation" (Table 5, Appendix E.3). |
| SmallNORB Azimuth zero-shot classification | Classification | SmallNORB Azimuth | Not specified in the paper. | "SmallNORB Azimuth" (Table 5, Appendix E.3). |
| DMLAB zero-shot classification | Classification | DMLAB | Not specified in the paper. | "DMLAB" (Table 5, Appendix E.3). |
| KITTI closest vehicle distance zero-shot classification | Classification | KITTI closest vehicle distance | Not specified in the paper. | "KITTI closest vehicle distance" (Table 5, Appendix E.3). |

3.2 Few-shot linear probe classification tasks
Evidence: "we evaluate few-shot linear probe performance on seven datasets commonly used to benchmark transfer performance." (Few-shot transfer paragraph).
Evidence for datasets: "Figure 10: Evaluating few-shot linear probe performance on ImageNet." and "Figure 11 displays few-shot performance on Food101 [7], Cars [35], CIFAR-10 & 100 [37], DTD [12] and SUN397 [85]." (Few-shot transfer paragraph).

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet few-shot linear probe classification | Classification | ImageNet | Not specified in the paper. | "Figure 10: Evaluating few-shot linear probe performance on ImageNet." (Figure 10 caption). |
| Food101 few-shot linear probe classification | Classification | Food101 | Not specified in the paper. | "Food101" (Few-shot transfer paragraph). |
| Cars few-shot linear probe classification | Classification | Cars | Not specified in the paper. | "Cars" (Few-shot transfer paragraph). |
| CIFAR-10 few-shot linear probe classification | Classification | CIFAR-10 | Not specified in the paper. | "CIFAR-10 & 100" (Few-shot transfer paragraph). |
| CIFAR-100 few-shot linear probe classification | Classification | CIFAR-100 | Not specified in the paper. | "CIFAR-10 & 100" (Few-shot transfer paragraph). |
| DTD few-shot linear probe classification | Classification | DTD | Not specified in the paper. | "DTD" (Few-shot transfer paragraph). |
| SUN397 few-shot linear probe classification | Classification | SUN397 | Not specified in the paper. | "SUN397" (Few-shot transfer paragraph). |

3.3 Retrieval tasks
Evidence: "Table 8: CLIP Zero-Shot retrieval results on the Flickr30K test set. We show retrieval performance at 1, 5, and 10 samples for both image to text and text to image." and "Table 9: CLIP Zero-Shot retrieval results on the MSCOCO test set. We show retrieval performance at 1, 5, and 10 samples for both image to text and text to image." (Appendix E.3).

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| Flickr30K retrieval (image to text, text to image) | Retrieval | Flickr30K | Not specified in the paper. | "CLIP Zero-Shot retrieval results on the Flickr30K test set" (Table 8, Appendix E.3). |
| MSCOCO retrieval (image to text, text to image) | Retrieval | MSCOCO | Not specified in the paper. | "CLIP Zero-Shot retrieval results on the MSCOCO test set" (Table 9, Appendix E.3). |

3.4 Text-to-image generation tasks

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| GLIDE/LAIONIDE text-to-image generation (fine-tuned on LAION-5B) | Generation | LAION-5B | Not specified in the paper. | "we aim to re-introduce the ability to generate imagery of humans into these checkpoints by finetuning them on LAION-5B." and "We finetune the released GLIDE 64 pixel base (filtered) checkpoint from OpenAI on LAION-5B." (Appendix F.1). |
| Stable Diffusion text-to-image generation (trained on LAION-5B subsets) | Generation | LAION-2B-en; laion-high-resolution; laion-improved-aesthetics | Not specified in the paper. | "Stable Diffusion is a generative latent diffusion model trained on various LAION-5B subsets:" and the listed subsets: "• 237,000 steps at 256x256 on LAION-2B-en"; "• 194,000 steps at 512x512 on laion-high-resolution"; "• 515,000 steps at 512x512 on laion-improved-aesthetics" (Appendix F.2). |

4. Domain and Modality Scope

- Single domain? Not specified in the paper.
- Multiple domains within the same modality? Not specified in the paper.
- Multiple modalities? Yes. Evidence: "Learning from multimodal data such as text, images, and audio is a longstanding research challenge" (Introduction) and "image-text pairs" in "a dataset consisting of 5.85 billion CLIP-filtered image-text pairs" (Abstract).
- Domain generalization / cross-domain transfer? Not claimed. The paper reports distribution-shift evaluation: "we follow [94] and evaluate robustness performance on ImageNet distribution shift datasets [3, 23, 25, 61, 82]." (Section 5.2.1).

5. Model Sharing Across Tasks

5.1 Zero-shot classification tasks (VTAB+ list)
Evidence for zero-shot evaluation protocol: "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." and "For each downstream dataset, we use a set of pre-defined prompts for each class" (Section 5.2.1).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet-1k zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| ImageNet-v2 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| ImageNet-R zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| ImageNet Sketch zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| ObjectNet zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| ImageNet-A zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| CIFAR-10 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| CIFAR-100 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| MNIST zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Oxford Flowers 102 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Stanford Cars zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| SVHN zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Facial Emotion Recognition 2013 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| RenderedSST2 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Oxford-IIIT Pets zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Caltech-101 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Pascal VOC 2007 Classification zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| SUN397 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| FGVC Aircraft zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Country211 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Describable Textures zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| GTSRB zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| STL10 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| Diabetic Retinopathy zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| EuroSAT zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| RESISC45 zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| PatchCamelyon zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| CLEVR Counts zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| CLEVR Object Distance zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| DSPRITES Orientation zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| DSPRITES Position zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| SmallNORB Elevation zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| SmallNORB Azimuth zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| DMLAB zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |
| KITTI closest vehicle distance zero-shot classification | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "Following CLIP [58] and subsequent works, we evaluate the models on zero-shot classification." (Section 5.2.1). |

5.2 Few-shot linear probe tasks
Evidence: "we evaluate few-shot linear probe performance on seven datasets commonly used to benchmark transfer performance." (Few-shot transfer paragraph).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| ImageNet few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Figure 10: Evaluating few-shot linear probe performance on ImageNet." (Figure 10 caption). |
| Food101 few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Food101" (Few-shot transfer paragraph). |
| Cars few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Cars" (Few-shot transfer paragraph). |
| CIFAR-10 few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "CIFAR-10 & 100" (Few-shot transfer paragraph). |
| CIFAR-100 few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "CIFAR-10 & 100" (Few-shot transfer paragraph). |
| DTD few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "DTD" (Few-shot transfer paragraph). |
| SUN397 few-shot linear probe classification | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "SUN397" (Few-shot transfer paragraph). |

5.3 Retrieval tasks
Evidence: "CLIP Zero-Shot retrieval results on the Flickr30K test set" (Table 8) and "CLIP Zero-Shot retrieval results on the MSCOCO test set" (Table 9).

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Flickr30K retrieval (image to text, text to image) | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "CLIP Zero-Shot retrieval results on the Flickr30K test set" (Table 8, Appendix E.3). |
| MSCOCO retrieval (image to text, text to image) | Not specified in the paper. | No (zero-shot). | Not specified in the paper. | "CLIP Zero-Shot retrieval results on the MSCOCO test set" (Table 9, Appendix E.3). |

5.4 Text-to-image generation tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| GLIDE/LAIONIDE text-to-image generation | Not specified in the paper. | Yes. | Not specified in the paper. | "We finetune the released GLIDE 64 pixel base (filtered) checkpoint from OpenAI on LAION-5B." (Appendix F.1). |
| Stable Diffusion text-to-image generation | Not specified in the paper. | Not specified in the paper (trained on LAION-5B subsets). | Not specified in the paper. | "Stable Diffusion is a generative latent diffusion model trained on various LAION-5B subsets:" (Appendix F.2). |

6. Input and Representation Constraints

- Fixed input resolution for CLIP reproduction models is explicitly listed in Table 3: "ViT-B/32      768 / 512    512               12 / 12    224x224   10 M    151 M"; "ViT-B/16      768 / 512    512               12 / 12    224x224   29 M    150 M"; "ViT-B/16+     896 / 640    640               12 / 12    240x240   40 M    208 M"; "ViT-L/14      1024 / 768   768               24 / 12    224x224   97 M    428 M" (Table 3, Appendix E.1).
- GLIDE fine-tuning specifies base and upscaling image sizes: "We finetune the released GLIDE 64 pixel base (filtered) checkpoint from OpenAI on LAION-5B." and "For upscaling from 64x64 images to 256x256 images, we use the unmodified weights from OpenAI GLIDE-upsample-filtered." (Appendix F.1).
- Stable Diffusion training resolutions are explicit: "• 237,000 steps at 256x256 on LAION-2B-en" and "• 194,000 steps at 512x512 on laion-high-resolution" (Appendix F.2).
- Fixed patch size, fixed number of tokens, fixed dimensionality (2D), padding/resizing requirements: Not specified in the paper.

7. Context Window and Attention Structure

Maximum sequence length, fixed vs variable length, attention type (global/windowed/hierarchical/sparse), and attention cost mechanisms are not specified in the paper.

8. Positional Encoding (Critical Section)

Not specified in the paper.

9. Positional Encoding as a Variable

Not specified in the paper.

10. Evidence of Constraint Masking

- Dataset scale: "we present LAION-5B - a dataset consisting of 5.85 billion CLIP-filtered image-text pairs, of which 2.32B contain English language." (Abstract).
- Model sizes: "ViT-B/32      768 / 512    512               12 / 12    224x224   10 M    151 M"; "ViT-B/16      768 / 512    512               12 / 12    224x224   29 M    150 M"; "ViT-B/16+     896 / 640    640               12 / 12    240x240   40 M    208 M"; "ViT-L/14      1024 / 768   768               24 / 12    224x224   97 M    428 M" (Table 3, Appendix E.1).
- Scaling data/model/compute and performance: "we can report that increasing either model or data scale for CLIP pre-training results in improvement of zero-shot classification performance on various downstream transfer targets." (Section 5.2.1). "Seeing same number of samples on larger data scale leads consistently to better zero-shot transfer performance, when investing enough into training compute." (Figure 12 caption).
- Architectural hierarchy or training tricks as primary drivers: Not specified in the paper beyond the scaling observations above.

11. Architectural Workarounds

- Computational scaling workaround for loss computation: "By turning memory complexity from O(N 2 ) into O(nN ), we slash memory overhead due to scaling from GBs down to MBs." (Appendix E.2).
- Windowed attention, hierarchical stages, token pooling/merging, or other architectural workarounds: Not specified in the paper.

12. Explicit Limitations and Non-Claims

- "LAION-5B is not a finished data product." (Introduction).
- "The large scale of current image-text datasets makes it infeasible to thoroughly investigate all aspects of a dataset in a single publication." (Section 6 Technical Limitations).
- "we strongly recommend that LAION-5B should only be used for academic research purposes in its current form. We advise against any applications in deployed systems without carefully investigating behavior and possible biases of models trained on LAION-5B." (Introduction).
- "Due to the known biases of the dataset, under no circumstance should any models be put into production using the dataset as is. It is neither safe nor responsible. As it stands, the dataset should be solely used for research purposes in its uncurated state." (Appendix A.5, Q43).
- "Likewise, this dataset should not be used to aid in military or surveillance tasks." (Appendix A.5, Q43).

13. Constraint Profile (Synthesis)

- Domain scope: Image-text data and evaluations on vision-language tasks; multiple modalities are explicit (image and text).
- Task structure: Multiple tasks including classification (zero-shot and few-shot), retrieval, and text-to-image generation.
- Representation rigidity: Fixed input resolutions are specified for CLIP reproduction models (224x224 or 240x240) and explicit image sizes for GLIDE/Stable Diffusion training; other representation constraints (tokens/patches) are not specified.
- Model sharing vs specialization: Zero-shot classification and retrieval are reported as zero-shot evaluations (no fine-tuning stated); generative models are explicitly fine-tuned/trained separately.
- Positional encoding: Not specified in the paper.

14. Final Classification

Classification: Multi-task, single-domain.
Justification: The paper evaluates multiple tasks (e.g., "we evaluate the models on zero-shot classification" and "CLIP Zero-Shot retrieval results on the Flickr30K test set" and GLIDE fine-tuning) while staying within image-text data ("a dataset consisting of 5.85 billion CLIP-filtered image-text pairs"). The evaluations are diverse in task type but remain within the vision-language modality.

1) Number of distinct tasks evaluated: 38 - Flickr30K image-to-text and text-to-image retrieval (2) and MSCOCO image-to-text and text-to-image retrieval (2); CxC image-to-text, text-to-image, text-to-text, and image-to-image retrieval (4) plus CxC STS/SIS/SITS semantic similarity (3); ImageNet classification (variants ImageNet-R/ImageNet-A/ImageNet-V2 treated as variants, not separate tasks); fine-grained classification on Oxford Flowers-102, Oxford-IIIT Pets, Stanford Cars, and Food101 (4); VTAB's 19 visual classification tasks; Multi30k image-to-text and text-to-image retrieval (2); SimLex-999 word similarity (1). (2021_ALIGN.pdf, Section 4.2; 2021_ALIGN.pdf, Section 4.3; 2021_ALIGN.pdf, Section 8; 2021_ALIGN.pdf, Appendix B)

2) Number of trained model instances required to cover all tasks: 28 - one pretrained ALIGN model for zero-shot retrieval/zero-shot ImageNet/SimLex, plus fine-tuned retrieval models for Flickr30K and MSCOCO (MSCOCO fine-tuned model also used for CxC), one ImageNet classification model with a trained head, four fine-grained classification models (Flowers-102, Pets, Cars, Food101) with task-specific heads, 19 VTAB task-specific models, and one multilingual ALIGNmling model for Multi30k retrieval. (2021_ALIGN.pdf, Section 4.2; 2021_ALIGN.pdf, Section 5.1; 2021_ALIGN.pdf, Section 5.2; 2021_ALIGN.pdf, Section 5.3; 2021_ALIGN.pdf, Section 8; 2021_ALIGN.pdf, Appendix B)

3) Task-Model Ratio:

$$
\boxed{
\frac{38\ \text{tasks}}{28\ \text{models}} = 1.36
}
$$

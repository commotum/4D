1. Number of distinct tasks evaluated: 45.
   Counted task groups (with evidence):
   - Zero-shot classification on VTAB+ (35 datasets): ImageNet-1k, ImageNet-v2, ImageNet-R, ImageNet Sketch, ObjectNet, ImageNet-A, CIFAR-10, CIFAR-100, MNIST, Oxford Flowers 102, Stanford Cars, SVHN, Facial Emotion Recognition 2013, RenderedSST2, Oxford-IIIT Pets, Caltech-101, Pascal VOC 2007 Classification, SUN397, FGVC Aircraft, Country211, Describable Textures (DTD), GTSRB, STL10, Diabetic Retinopathy, EuroSAT, RESISC45, PatchCamelyon, CLEVR Counts, CLEVR Object Distance, DSPRITES Orientation, DSPRITES Position, SmallNORB Elevation, SmallNORB Azimuth, DMLAB, KITTI closest vehicle distance. Evidence: VTAB+ is evaluated over 35 tasks and Table 5 lists these datasets (Section 5.2.1, p.8; Appendix E.3, Table 5, p.37-38).
   - Few-shot linear probe classification on 7 datasets: ImageNet, Food101, Cars, CIFAR-10, CIFAR-100, DTD, SUN397. Evidence: "few-shot linear probe performance on seven datasets" with Figure 10 (ImageNet) and Figure 11 (Food101, Cars, CIFAR-10 & 100, DTD, SUN397) (Appendix E.3, p.38-39).
   - Image-text retrieval (zero-shot) on Flickr30K and MSCOCO (image-to-text and text-to-image). Evidence: Table 8 (Flickr30K) and Table 9 (MSCOCO) (Appendix E.3, p.43-44).
   - Text-to-image generation (single task capability): GLIDE fine-tuning on LAION-5B and Stable Diffusion trained on LAION-5B subsets. Evidence: Section 5.3 and Appendix F.1/F.2 (p.9, p.41-45).
   Note: The few-shot datasets overlap with VTAB+ but are counted separately here because the paper explicitly evaluates few-shot linear probe transfer, which requires training a task-specific classifier head.

2. Number of trained model instances required to cover all tasks: 9.
   Rationale:
   - 1 CLIP-style model instance covers all zero-shot classification and retrieval tasks (zero-shot classification via prompts; CLIP zero-shot retrieval on Flickr30K/MSCOCO). Evidence: Section 5.2.1 (zero-shot prompt-based classification, p.8) and Tables 8-9 (zero-shot retrieval, p.43-44).
   - 7 task-specific linear-probe models (one per few-shot dataset), since few-shot linear probe evaluation trains a dataset-specific classifier head. Evidence: Appendix E.3 (few-shot linear probe on seven datasets, p.38-39).
   - 1 text-to-image generation model instance (counted once because GLIDE and Stable Diffusion are architectural variants of the same text-to-image task capability). Evidence: Section 5.3 and Appendix F.1/F.2 (p.9, p.41-45).

3. Task-Model Ratio:

$$
\boxed{
\frac{45\ \text{tasks}}{9\ \text{models}} = 5
}
$$

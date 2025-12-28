1. Number of distinct tasks evaluated: 17.
   Counted task types (with evidence):
   - Zero-shot image classification across standard datasets (e.g., ImageNet/CIFAR/etc.) via class-name prompts. Evidence: “To perform zero-shot classification… use the names of all the classes…” (Section 3.1.2, p.6).
   - Fine-grained object classification. Evidence: “tasks such as… many types of fine-grained object classification.” (Abstract, p.1).
   - OCR / text recognition (incl. OCR-based semantic tasks). Evidence: “tasks such as OCR” (Abstract, p.1) and “MNIST, SVHN, and IIIT5K… Hateful Memes and SST-2” for OCR evaluation (Appendix E.2, p.45).
   - Action recognition in videos. Evidence: “action recognition in videos” (Abstract, p.1) and UCF101/Kinetics-700/RareAct evaluation (Appendix E.3, p.46–47).
   - Geo-localization / scene recognition. Evidence: “geo-localization” (Abstract, p.1) and Country211 / IM2GPS evaluation (Appendix E.4, p.47).
   - Satellite image classification. Evidence: “satellite image classification (EuroSAT and RESISC45)” (Section 3.1.5, p.9).
   - Medical imaging (lymph node tumor detection). Evidence: “lymph node tumor detection (PatchCamelyon)” (Section 3.1.5, p.9).
   - Counting objects in synthetic scenes. Evidence: “counting objects in synthetic scenes (CLEVRCounts)” (Section 3.1.5, p.9).
   - Traffic sign recognition. Evidence: “German traffic sign recognition (GTSRB)” (Section 3.1.5, p.9).
   - Distance-to-nearest-car estimation. Evidence: “recognizing distance to the nearest car (KITTI Distance)” (Section 3.1.5, p.9).
   - Facial emotion recognition. Evidence: “facial emotion recognition” (Section 3.2, p.11; Appendix A.1, p.37).
   - Image–text retrieval (text and image retrieval). Evidence: “CLIP pre-trains for the task of image-text retrieval” and evaluates zero-shot retrieval (Appendix E.1, p.45).
   - Robustness to natural distribution shift (ImageNet variants). Evidence: “performance on a set of 7 distribution shifts: ImageNetV2… ImageNet Sketch… Youtube-BB… ImageNet-Vid… ObjectNet… ImageNet Adversarial… ImageNet Rendition.” (Section 3.3, p.13).
   - Human comparison on Oxford IIT Pets (zero-/one-/two-shot humans). Evidence: “We had five different humans look at… Oxford IIT Pets…” (Section 4, p.16).
   - Demographic classification / bias probes (FairFace race/gender/age). Evidence: “We evaluated two versions of CLIP on the FairFace dataset…” (Section 7.1, p.21).
   - Surveillance: CCTV scene classification + fine-grained detection. Evidence: “performance on classification of images from CCTV cameras…” (Section 7.2, p.24).
   - Zero-shot celebrity / identity recognition (CelebA). Evidence: “zero-shot celebrity identification… CelebA” (Section 7.2, p.24–25).

2. Number of trained model instances required to cover all tasks: 1.
   Rationale: A single pretrained CLIP model performs the above tasks via zero-shot prompts or similarity-based inference, with the zero-shot classifier cached and reused across datasets (“For zero-shot evaluation, we cache the zero-shot classifier… and reuse it for all subsequent predictions.” Section 3.1.2, p.6). The paper’s task coverage is enabled by this single shared model (“After pre-training, natural language is used to reference learned visual concepts… enabling zero-shot transfer to downstream tasks.” Abstract, p.1).

3. Task–Model Ratio:

$$
\boxed{
\frac{17\ \text{tasks}}{1\ \text{model}} = 17
}
$$

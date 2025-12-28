Distinct tasks evaluated (8 total):
- Image classification/recognition (ImageNet and robustness datasets) (p.8-9).
- Video action recognition (Kinetics-400/600/700, Moments-in-Time) (p.8).
- Image-text retrieval (MSCOCO, Flickr30K) (p.9).
- Video-text retrieval (MSR-VTT) (p.9).
- Visual question answering (VQA v2) (p.10, p.18).
- Visual entailment (SNLI-VE) (p.10, p.18).
- Visual reasoning (NLVR2) (p.10, p.18).
- Image captioning (MSCOCO, NoCaps) (p.10, p.19).

Trained model instances required to cover all tasks:
- 1 pretrained CoCa checkpoint used zero-shot for image classification and image/video-text retrieval (single pretrained checkpoint; zero-shot uses frozen parameters) (p.8-9).
- 1 task-specific model for video action recognition (learned pooler + softmax; frozen-feature/finetuning) (p.6, p.8, p.17).
- 1 each for VQA, SNLI-VE, NLVR2 (pooler + linear classifier; fine-tuned) (p.10, p.18).
- 1 for image captioning (fine-tuned with captioning loss on MSCOCO; evaluated on MSCOCO/NoCaps) (p.10, p.19).

Total trained model instances: 6.

$$
\boxed{
\frac{8\ \text{tasks}}{6\ \text{models}} = 1.33\overline{3}
}
$$

1. Number of distinct tasks evaluated: 3.
   - Multi-resolution classification on ImageNet-1k. Evidence: "We report the accuracy on the ImageNet-1k validation set as varying image sizes." (p.10, Section 4.1)
   - Object detection on MS-COCO. Evidence: "We verify 2D RoPE in object detection on MS-COCO [16]." (p.12, Section 4.2)
   - Semantic segmentation on ADE20k. Evidence: "We train 2D RoPE ViT and Swin for semantic segmentation on ADE20k [40, 41]." (p.13, Section 4.3)

2. Number of trained model instances required to cover all tasks: 3.
   - Rationale: Each task uses a task-specific head/decoder or training setup: classification on ImageNet-1k (Section 4.1), detection uses the DINO detector head, and segmentation uses UperNet or Mask2Former heads. These task-specific heads/decoders require separate trained model instances per task. Evidence: "DINO [39] detector is trained using ViT and Swin as backbone network." (p.12, Section 4.2); "For ViT, we use UperNet [37]... For Swin, Mask2Former [2] for segmentation is used with the Swin." (p.13, Section 4.3)

3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

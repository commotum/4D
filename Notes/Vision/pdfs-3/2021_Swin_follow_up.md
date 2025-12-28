1) Number of distinct tasks evaluated: 4 — image classification (ImageNet-1K), object detection (COCO), instance segmentation (COCO), and semantic segmentation (ADE20K). (2021_Swin.pdf, Section 4, p5; 2021_Swin.pdf, Section 4.2, p6; 2021_Swin.pdf, Section 4.3, p8)

2) Number of trained model instances required to cover all tasks: 4 — each task uses a task-specific head/decoder: classification uses a dedicated linear classifier head; object detection uses COCO detection frameworks; instance segmentation is evaluated with Mask R-CNN/HTC++-style models that report mask AP (mask head); and semantic segmentation uses a separate decoder (UperNet). Therefore each task requires its own trained model instance. (2021_Swin.pdf, Appendix A2.1, p9; 2021_Swin.pdf, Section 4.2, p6; 2021_Swin.pdf, Table 2, p7; 2021_Swin.pdf, Section 4.3, p8)

3) Task-Model Ratio:

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$

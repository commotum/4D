1. Number of distinct tasks evaluated: 3 (2D image classification, object detection, 3D classification; evaluated on ImageNet-1K, MS COCO, and UCF-101 respectively). Evidence: "We first assess their scalability in 2D image classification... conduct object detection experiments... perform 3D classification experiments" and the datasets listed for each task (ImageNet-1K, MS COCO, UCF-101). (2025_ComRoPE.pdf: "We first assess their scalability in 2D image classification... conduct object detection experiments... perform 3D classification experiments"; "on the ImageNet-1K dataset"; "MS COCO dataset"; "3D classification task on UCF-101")
2. Number of trained model instances required to cover all tasks: 3. Evidence: the 2D classification setup uses a ViT-B/16 architecture on ImageNet-1K; the object detection task uses a ViT-S backbone on MS COCO; and the 3D classification task uses a modified (smaller) model configuration because standard ViT-Base is not applicable for that task, implying a separate trained instance per task. (2025_ComRoPE.pdf: "Vision Transformer (ViT-B/16)" and "ImageNet-1K dataset"; "We adopt ViT-S as our backbone" and "MS COCO dataset"; "we conduct a 3D classification task on UCF-101" and "standard ViT-Base is not applicable. We modified the model parameters...")
3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

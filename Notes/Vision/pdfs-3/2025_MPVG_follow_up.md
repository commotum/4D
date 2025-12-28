1. Number of distinct tasks evaluated: 3 (image classification, object detection, semantic segmentation).
- Image classification: "We evaluate the performance of our methods on ImageNet-1K ... and CIFAR-100" (Image Classification, p. 18158).
- Object detection: "On object detection, we evaluate our methods on COCO 2017" (Object Detection, p. 18158).
- Semantic segmentation: "On semantic segmentation, we evaluate our methods on ADE20K" (Semantic Segmentation, p. 18159).

2. Number of trained model instances required to cover all tasks: 3.
- Image classification uses a classification head (MLP) for prediction: "the output of this token is then used to make class predictions via Multi-Layer Perceptron (MLP)" (Introduction, p. 18154).
- Object detection uses a task-specific head: "we select the ViT-Adapter-Ti (Chen et al. 2022) model based on Mask R-CNN (He et al. 2017)" (Object Detection, p. 18158).
- Semantic segmentation uses a task-specific head: "We select the ViT-Adapter- Ti (Chen et al. 2022) model based on UperNet (Xiao et al. 2018)" (Semantic Segmentation, p. 18159).
Because detection and segmentation require different task-specific heads (Mask R-CNN vs UperNet) from classification, covering all three tasks requires separate trained model instances.

3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

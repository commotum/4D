1. Number of distinct tasks evaluated: 4.
   - CIFAR-100 image classification (Section 5.1, p.6: "We begin with CIFAR-100 and ImageNet-1k benchmarks to evaluate LieRE in 2D vision tasks.")
   - ImageNet-1k image classification (Section 5.1, p.6: "We begin with CIFAR-100 and ImageNet-1k benchmarks to evaluate LieRE in 2D vision tasks.")
   - Synthetic spatial reasoning image classification (Section 5.2, p.6: "we designed a synthetic image classification task" and "The objective is to identify the direction of this specific arrow.")
   - UCF101 3D video classification (Section 5.3, p.7: "we use the UCF101 video classification benchmark")
   - Note: The multi-resolution evaluation uses the ImageNet validation set at varying inference resolutions, so it is an evaluation variant of ImageNet classification rather than a separate task (Section 5.6, p.7-8: "We evaluate the accuracy on the ImageNet validation set with varying inference resolutions.")

2. Number of trained model instances required to cover all tasks: 4 separate models (one per task dataset), since the 2D models are trained from scratch (Section 5.1, p.6: "All models use ViT-based architectures trained from scratch"), the synthetic task uses its own training set (Section 5.2, p.6: "We train the models on 800,000 examples"), and the 3D UCF101 models are trained from scratch (Section 5.3, p.7: "trained from scratch").

3. Task-Model Ratio:
$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$

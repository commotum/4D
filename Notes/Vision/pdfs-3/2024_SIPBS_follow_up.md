1. Number of distinct tasks evaluated: 8.
   - Musk1 (binary molecular drug activity prediction). Evidence: the paper lists five binary benchmark datasets and states that "the first two datasets (Musk1 and Musk2) cover the application of MIL for molecular drug activity predictions" (Section 4.1.1 Benchmark MIL datasets).
   - Musk2 (binary molecular drug activity prediction). Evidence: same benchmark list and the statement above that identifies Musk1 and Musk2 as molecular drug activity datasets (Section 4.1.1 Benchmark MIL datasets).
   - Elephant (binary image classification). Evidence: the benchmark list includes Elephant, and it states "Elephant, Tiger, and Fox, are related to image classification" (Section 4.1.1 Benchmark MIL datasets).
   - Tiger (binary image classification). Evidence: same statement listing Elephant, Tiger, and Fox as image classification datasets (Section 4.1.1 Benchmark MIL datasets).
   - Fox (binary image classification). Evidence: same statement listing Elephant, Tiger, and Fox as image classification datasets (Section 4.1.1 Benchmark MIL datasets).
   - MIL-MNIST (multi-class handwritten digit classification). Evidence: "an additional dataset for multi-class classification is created from well-known MNIST digits (MIL-MNIST) for digit classification" (Section 4.1.2 MIL-based MNIST dataset).
   - MIL-based CIFAR-10 (multi-class object recognition). Evidence: "MIL datasets for multi-class classification using images from the CIFAR-10 dataset for object recognition" (Section 4.1.3 MIL-based CIFAR-10 dataset).
   - Colon cancer histopathology (binary classification). Evidence: "experiments on colon cancer histopathology images" and "This dataset consists of 100 H&E images belonging to binary classes" (Section 4.1.4 Colon cancer dataset).

2. Number of trained model instances required to cover all tasks: 8.
   - Rationale: The paper evaluates these eight datasets separately and does not describe a single jointly trained multi-task model. The embedding network is dataset-type dependent (fully connected for benchmark datasets vs convolutional/LeNet5 for MIL-MNIST and MIL-based CIFAR-10), implying separate trained instances per dataset/task to cover all evaluations (Section 4.1.1-4.1.4; Section 5.9.1 Embedding network).

3. Task-Model Ratio:

$$
\boxed{
\frac{8\ \text{tasks}}{8\ \text{models}} = 1
}
$$

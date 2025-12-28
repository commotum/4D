1. Basic Metadata
- Title: Simultaneous instance pooling and bag representation selection approach for multiple-instance learning (MIL) using vision transformer
- Authors: Muhammad Waqas; Muhammad Atif Tahir; Muhammad Danish Author; Sumaya Al-Maadeed; Ahmed Bouridane; Jia Wu
- Year: 2024
- Venue: Neural Computing and Applications

2. One-Sentence Contribution Summary
The paper proposes ViT-IWRS, a vision-transformer-based MIL framework that models instance relationships, computes instance weights, and selects among multiple bag representation vectors to improve bag-level classification under uncertain instance relationship assumptions.

3. Tasks Evaluated
Task 1: Molecular drug activity prediction (binary classification)
- Task type: Classification
- Dataset(s): Musk1, Musk2
- Domain: molecular data
- Evidence: "The first two datasets (Musk1 and Musk2) cover the application of MIL for molecular drug activity predictions [ 23 ]." (Section 4.1.1 Benchmark MIL datasets)

Task 2: Animal image classification (binary classification)
- Task type: Classification
- Dataset(s): Elephant, Tiger, Fox
- Domain: natural images (image segments)
- Evidence: "The later three datasets: Elephant, Tiger, and Fox, are related to image classification [ 26 ]; features of image segments constitute the bags in these datasets. The positive bags hold one or more instances related to the animal of interest while the negative bags contain other animals." (Section 4.1.1 Benchmark MIL datasets)

Task 3: Handwritten digit classification (multi-class)
- Task type: Classification
- Dataset(s): MIL-MNIST
- Domain: natural images (digits)
- Evidence: "In addition to the existing benchmark MIL dataset, an additional dataset for multi-class classification is created from well-known MNIST digits (MIL-MINST) for digit classification [ 48 ]. The dataset consists of gray-scale digit images of size \(28 \times 28\) , and the images are randomly selected to form a bag where each digit represents an instance." (Section 4.1.2 MIL-based MNIST dataset)

Task 4: Object recognition (multi-class)
- Task type: Classification
- Dataset(s): MIL-based CIFAR-10
- Domain: natural images
- Evidence: "We construct more challenging MIL datasets for multi-class classification using images from the CIFAR-10 dataset for object recognition MIL application [ 49 ]. The CIFAR-10 dataset contains 60000 images divided into ten classes, each image is of size \(32 \times 32\) , and classes are completely mutually exclusive." (Section 4.1.3 MIL-based CIFAR-10 dataset)

Task 5: Colon cancer detection in histopathology (binary classification)
- Task type: Classification
- Dataset(s): Colon cancer histopathology images
- Domain: medical images (H&E WSI)
- Evidence: "For this study, we conducted experiments on colon cancer histopathology images [ 24 ] to test the efficiency of ViT-IWRS." (Section 4.1.4 Colon cancer dataset)
- Evidence: "This dataset consists of 100 H&E images belonging to binary classes." (Section 4.1.4 Colon cancer dataset)
- Evidence: "Every WSI represents a bag with several 27 \(\times\) 27 patches. The bag is labeled as positive if it has one or more nuclei from the epithelial class." (Section 4.1.4 Colon cancer dataset)

4. Domain and Modality Scope
- Domain scope: Multiple domains. Evidence: "The performance of ViT-IWRS is evaluated using different datasets for binary and multi-class classification problems. These datasets have been used to assess the performance of MIL algorithms in the literature and cover a range of MIL application domains, such as molecular activity prediction, image classification, object detection, and medical image classification." (Section 4.1 Details of datasets and evaluation measure)
- Modalities: Multiple modalities (molecular data and images). Evidence: "The first two datasets (Musk1 and Musk2) cover the application of MIL for molecular drug activity predictions [ 23 ]." (Section 4.1.1 Benchmark MIL datasets) and "Detecting cancerous regions in hematoxylin and eosin (H &E) stained whole-slide images (WSI) are vital in clinical settings [ 50 ]." (Section 4.1.4 Colon cancer dataset)
- Domain generalization or cross-domain transfer: The paper claims generalization across different MIL application domains, but does not claim cross-domain transfer. Evidence: "To demonstrate the generalization ability of the proposed approach, the experiments are performed on multiple types of data from different MIL application domains." (Introduction) Cross-domain transfer: Not claimed.

5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Molecular drug activity prediction (Musk1, Musk2) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "The embedding network for benchmark datasets mainly consists of fully connected layers. In contrast, the MIL-MNIST and MIL-based CIFAR-10 datasets network comprises convolution layers with other related operations based on the LeNet5 architecture [ 61 ]." (Section 5.9.1 Embedding network) |
| Animal image classification (Elephant, Tiger, Fox) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "The embedding network for benchmark datasets mainly consists of fully connected layers. In contrast, the MIL-MNIST and MIL-based CIFAR-10 datasets network comprises convolution layers with other related operations based on the LeNet5 architecture [ 61 ]." (Section 5.9.1 Embedding network) |
| Handwritten digit classification (MIL-MNIST) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "The embedding network for benchmark datasets mainly consists of fully connected layers. In contrast, the MIL-MNIST and MIL-based CIFAR-10 datasets network comprises convolution layers with other related operations based on the LeNet5 architecture [ 61 ]." (Section 5.9.1 Embedding network) |
| Object recognition (MIL-based CIFAR-10) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "The embedding network for benchmark datasets mainly consists of fully connected layers. In contrast, the MIL-MNIST and MIL-based CIFAR-10 datasets network comprises convolution layers with other related operations based on the LeNet5 architecture [ 61 ]." (Section 5.9.1 Embedding network) |
| Colon cancer detection (histopathology) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "The embedding network for benchmark datasets mainly consists of fully connected layers. In contrast, the MIL-MNIST and MIL-based CIFAR-10 datasets network comprises convolution layers with other related operations based on the LeNet5 architecture [ 61 ]." (Section 5.9.1 Embedding network) |

6. Input and Representation Constraints
- Bag/instance representation: "In binary MIL classification problem, for a given bag \({\varvec{B}}_{i}=\left\{ {\varvec{x}}_{i, 1}, {\varvec{x}}_{i, 2}, {\varvec{x}}_{i, 3}, \ldots , {\varvec{x}}_{i, mi}\right\}\) of mi total instances with d dimensions, where \({\varvec{x}}_{i, j}\) represents jth instance of ith bag." (Section 3.1 Problem formulation)
- Variable number of instances: "This process transforms the bag with a variable number of instances to a manageable vector representation and transforms the MIL problem into a classical supervised learning problem." (Section 3.5 Computation of bag representation vectors)
- Embedding network depends on data type: "the embedding network can consist of multi-layer perceptron (MLP) or convolution layers, depending upon the nature of the data." (Section 3.3 Vision transformer for bag encoding in MIL)
- Dataset-specific input sizes: "The dataset consists of gray-scale digit images of size \(28 \times 28\)" (Section 4.1.2 MIL-based MNIST dataset); "The CIFAR-10 dataset contains 60000 images divided into ten classes, each image is of size \(32 \times 32\)" (Section 4.1.3 MIL-based CIFAR-10 dataset); "Every WSI represents a bag with several 27 \(\times\) 27 patches." (Section 4.1.4 Colon cancer dataset)
- Fixed patch size, fixed number of tokens, or fixed resolution across tasks: Not specified in the paper.

7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable length: Variable length (bags have variable numbers of instances). Evidence: "This process transforms the bag with a variable number of instances to a manageable vector representation and transforms the MIL problem into a classical supervised learning problem." (Section 3.5 Computation of bag representation vectors)
- Attention type: Global self-attention. Evidence: "The encoding process involves a multi-head-self-attention process that captures the global dependencies between the instances in the bag." (Section 3 Proposed methodology)
- Computational cost management (windowing, pooling, token pruning): Not specified in the paper. Related cost note: "ViT uses a self-attention mechanism with quadratic complexity, making it more computationally expensive than traditional attention algorithms." (Section 5.8 Time efficiency comparison)

8. Positional Encoding (Critical Section)
- Standard ViT description: "Vision Transformers (ViT) takes 1D patch embeddings as input. Therefore, the image is transformed into a sequence of two-dimensional flattened patches, and a trainable linear projection converts the generated patches to one-dimensional vectors. The projected image patches are called patch embeddings. A learnable embedding called class token is also prepended to patch embeddings. Moreover, the positional embeddings which are added to preserve the positional information of patches in the image." (Section 3.2 Vision transformer)
- Positional encoding used in MIL: None. Evidence: "Here, the positional embeddings are not used as bag representation follows a permutation invariant structure." (Section 3.3 Vision transformer for bag encoding in MIL)
- Where applied and whether fixed: Not specified in the paper beyond the statement that positional embeddings are not used for MIL bag encoding.

9. Positional Encoding as a Variable
- Treated as a research variable or fixed assumption: Fixed assumption (not used in MIL). Evidence: "Here, the positional embeddings are not used as bag representation follows a permutation invariant structure." (Section 3.3 Vision transformer for bag encoding in MIL)
- Multiple positional encodings compared: Not specified in the paper.
- Claim that PE choice is not critical: Not specified in the paper.

10. Evidence of Constraint Masking
- Model size/scale evidence: "The ViT depth and the number of attention heads are the essential parameters in the proposed approach." (Section 5.9.4 Analysis for ViT depth and attention heads)
- Dataset size evidence: "the model is trained for 50, 100, 150, 200, 300, and 400 generated training bags, respectively, while the performance is evaluated on 1000 test bags." (Section 4.1.2 MIL-based MNIST dataset); "The training sets are built with 500 and 5000 bags, while the test set is created with 1000 bags." (Section 4.1.3 MIL-based CIFAR-10 dataset); "This dataset consists of 100 H&E images belonging to binary classes." (Section 4.1.4 Colon cancer dataset)
- Performance attribution: "The experimental results in Table 10 show that the removal of RSN from the proposed ViT-IWRS results in performance degradation. Therefore, the use of the RSN block is essential to achieve improved results." (Section 5.10.1 Effect of RSN block) and "The experimental results in Table 10 show that the removal of Transformer Encoding and RSN from the proposed ViT-IWRS results in performance degradation. Therefore, the use of this block is essential to attain improved results." (Section 5.10.2 Effect of transformer encoding)
- Claims that gains come from scaling model or data: Not specified in the paper.

11. Architectural Workarounds
- Global dependency modeling via MHSA: "The encoding process involves a multi-head-self-attention process that captures the global dependencies between the instances in the bag." (Section 3 Proposed methodology)
- Attention pooling for instance weighting and bag representation: "In this step, the weight for each instance in the bag is computed using the attention approach [ 13 , 15 ]. ... the instances in the bag are pooled using a weighted average operation to obtain representation vectors for the bag." (Section 3.4 Instance weight computation)
- Weight sharing between instance weighting and classification: "In this study, the weights of the transformer classification head are shared to learn instance weight and bag representation vector classification simultaneously. This process helps to enhance the connection between the loss and instance weighting process." (Section 3.4 Instance weight computation)
- Representation selection subnetwork: "RSN performs hard selection using Gumbel SoftMax in an end-to-end approach [ 46 ]." (Section 3.6 Representation selection subnetwork (RSN))
- Variable-length to fixed representation: "This process transforms the bag with a variable number of instances to a manageable vector representation and transforms the MIL problem into a classical supervised learning problem." (Section 3.5 Computation of bag representation vectors)

12. Explicit Limitations and Non-Claims
- Limitations and future work: "Although the proposed approach produces promising results on several datasets related to images, this approach is less computationally expensive as compared to existing pooling techniques. Furthermore, the performance of ViT-IWRS is effective when labels are entirely dependent on the structural properties of the instances, such as molecular datasets. The proposed loss function can be further extended to handle multi-instance multi-target regression problems, such as Drug Discovery and Environmental Monitoring. In the future, we intend to explore the application of the proposed approach to multiple-instance and multiple-label learning (MIML) tasks and incorporate the structural details of the bag into the self-attention process." (Section 6 Conclusion)
- Explicit non-claims about open-world, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

13. Constraint Profile (Synthesis)
- Domain scope: Multiple domains and modalities (molecular activity prediction, natural images, medical images) evaluated across separate datasets.
- Task structure: Bag-level MIL classification only (binary and multi-class).
- Representation rigidity: Variable-length bags with d-dimensional instances; dataset-specific image sizes; no fixed token length stated.
- Model sharing vs specialization: Not specified; embedding network varies by dataset type (fully connected vs convolutional).
- Role of positional encoding: Positional embeddings explicitly not used for MIL bag encoding.

14. Final Classification
Multi-task, multi-domain (constrained). The evaluation spans multiple MIL application domains (molecular activity prediction, image classification/object recognition, and medical image classification) and both binary and multi-class bag classification tasks. However, all experiments remain within supervised MIL classification and there is no claim of cross-domain transfer or unrestrained multi-task learning, so the setup is multi-domain but constrained.

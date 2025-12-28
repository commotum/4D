1. Basic Metadata
- Title: Adaptive patch selection to improve Vision Transformers through Reinforcement Learning
- Authors: Francesco Cauteruccio; Michele Marchetti; Davide Traini; Domenico Ursino; Luca Virgili
- Year: 2025
- Venue (conference/journal/arXiv): Applied Intelligence (journal)
  Evidence:
  > "Adaptive patch selection to improve Vision Transformers through Reinforcement Learning" (Article header)
  > "Cauteruccio, Francesco" (Metadata)
  > "Marchetti, Michele" (Metadata)
  > "Traini, Davide" (Metadata)
  > "Ursino, Domenico" (Metadata)
  > "Virgili, Luca" (Metadata)
  > "2025-04-01" (Metadata)
  > "Applied Intelligence" (Metadata)

2. One-Sentence Contribution Summary
- Summary: The paper proposes AgentViT, a reinforcement-learning framework that selects important image patches to reduce ViT computational cost while maintaining competitive performance in image classification.
  Evidence:
  > "we propose a new framework, called AgentViT, which uses Reinforcement Learning to train an agent that selects the most important patches to improve the learning of a ViT. The goal of AgentViT is to reduce the number of patches processed by a ViT, and thus its computational load, while still maintaining competitive performance." (Abstract)

3. Tasks Evaluated
| Task name | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| Image classification | Classification | CIFAR10; FashionMNIST; Imagenette\(^+\) (subset of ImageNet) | Not specified in the paper (datasets described as RGB images, grayscale clothing items, and a subset of ImageNet). | > "We tested AgentViT on CIFAR10, FashionMNIST, and Imagenette\(^+\) (which is a subset of ImageNet) in the image classification task" (Abstract)
> "CIFAR10 [24] consists of 60,000 RGB images of size \(32 \times 32\) pixels, categorized into 10 classes." (Experiments - 4.1 Experimental setup)
> "FashionMNIST [25] (FMNIST) contains 70,000 grayscale images of size \(28 \times 28\) pixels, representing 10 classes of clothing items, such as t-shirts, trousers, dresses, and sneakers." (Experiments - 4.1 Experimental setup)
> "It consists of 13,394 images of \(320 \times 320\) pixels categorized into 10 classes." (Experiments - 4.1 Experimental setup)
> "It thus contains 100,000 images categorized in the 10 classes of the original Imagenette." (Experiments - 4.1 Experimental setup) |

4. Domain and Modality Scope
- Single domain / multiple domains / modalities: Multiple domains within the same modality (images).
  Evidence:
  > "As datasets used to evaluate our approach across different domains and levels of complexity, we used CIFAR10, FashionMNIST, and an extended version of Imagenette." (Experiments - 4.1 Experimental setup)
- Domain generalization or cross-domain transfer: Not claimed; transfer learning is mentioned for the agent (pre-trained weights).
  Evidence:
  > "the use of an external agent enables transfer learning by using pre-trained weights to speed up training." (Introduction)

5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image classification (CIFAR10) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Image classification (FashionMNIST) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |
| Image classification (Imagenette\(^+\)) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. |

6. Input and Representation Constraints
- Fixed input resolution (per dataset):
  Evidence:
  > "CIFAR10 [24] consists of 60,000 RGB images of size \(32 \times 32\) pixels" (Experiments - 4.1 Experimental setup)
  > "FashionMNIST [25] (FMNIST) contains 70,000 grayscale images of size \(28 \times 28\) pixels" (Experiments - 4.1 Experimental setup)
  > "It consists of 13,394 images of \(320 \times 320\) pixels" (Experiments - 4.1 Experimental setup)
- Fixed patch size: patch_size is defined as a hyperparameter.
  Evidence:
  > "patch_size: the pixel size of each non-overlapping patch" (Experiments - 4.1 Experimental setup)
- Fixed embedding dimensionality: dim is defined as a hyperparameter.
  Evidence:
  > "dim: the embedding size of each patch" (Experiments - 4.1 Experimental setup)
- Fixed number of tokens for the initial representation (N patches):
  Evidence:
  > "the agent receives the state representation s, consisting of the attention values of the N patches" (Description of AgentVIT - Schematic workflow of AgentViT (3.1))
  > "given an input image decomposed into N patches, the output of the first attention layer generates an attention matrix of size \(N \times d\)" (Description of AgentVIT - State (3.2))
  > "The action space \(\mathcal{A}\) of AgentViT consists of N discrete actions, each corresponding to the selection of a specific patch." (Description of AgentVIT - Action (3.3))
- Variable number of selected tokens (patch pruning during training):
  Evidence:
  > "The agent is incentivized to select a number of patches close to the value specified by the user" (Description of AgentVIT - Reward (3.4))
- Padding/resizing requirements: Not specified in the paper.

7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed vs. variable sequence length: Not specified for the input sequence; N patches are referenced per image and the agent selects a subset.
  Evidence:
  > "the agent receives the state representation s, consisting of the attention values of the N patches" (Description of AgentVIT - Schematic workflow of AgentViT (3.1))
  > "The action space \(\mathcal{A}\) of AgentViT consists of N discrete actions, each corresponding to the selection of a specific patch." (Description of AgentVIT - Action (3.3))
  > "The agent is incentivized to select a number of patches close to the value specified by the user" (Description of AgentVIT - Reward (3.4))
- Attention type: Global attention across all patches.
  Evidence:
  > "The self-attention mechanism in ViTs requires the computation of attention scores between all pairs of patches in each layer" (Introduction)
- Mechanisms to manage computational cost:
  Evidence:
  > "The goal of AgentViT is to reduce the number of patches processed by a ViT, and thus its computational load" (Abstract)
  > "The agent computes Q-values for each patch and only patches with Q-values greater than the mean are selected." (Fig. 1 caption, Description of AgentVIT - Schematic workflow of AgentViT (3.1))

8. Positional Encoding (Critical Section)
- Positional encoding mechanism used: Not specified in the paper.
- Where applied (input only / every layer / attention bias): Not specified in the paper.
- Fixed across experiments / modified per task / ablated: Not specified in the paper.

9. Positional Encoding as a Variable
- Treated as core research variable or fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- Claims PE choice is "not critical" or secondary: Not specified in the paper.

10. Evidence of Constraint Masking
- Model size(s): Not specified in the paper.
- Dataset size(s):
  Evidence:
  > "CIFAR10 [24] consists of 60,000 RGB images" (Experiments - 4.1 Experimental setup)
  > "FashionMNIST [25] (FMNIST) contains 70,000 grayscale images" (Experiments - 4.1 Experimental setup)
  > "It thus contains 100,000 images categorized in the 10 classes of the original Imagenette." (Experiments - 4.1 Experimental setup)
- Performance/efficiency claims in the paper:
  Evidence:
  > "The goal of AgentViT is to reduce the number of patches processed by a ViT, and thus its computational load, while still maintaining competitive performance." (Abstract)
  > "AgentViT (resp., AgentSimpleViT) is able to train a Vision Transformer in less time than that required by ViT (resp., SimpleViT) when trained without removing any patches. This reduction in training time is achieved with comparable accuracy." (Discussion)
- Scaling model size / scaling data / architectural hierarchy / training tricks: Not specified in the paper as primary drivers of performance.

11. Architectural Workarounds
- RL-based patch pruning using attention values (external agent):
  Evidence:
  > "we propose a new framework, called AgentViT, which uses Reinforcement Learning to train an agent that selects the most important patches" (Abstract)
  > "(i) The image is processed by the first ViT layer, where attention values are extracted and an average attention value per patch is computed to form the state representation. (ii) The agent computes Q-values for each patch and only patches with Q-values greater than the mean are selected." (Fig. 1 caption, Description of AgentVIT - Schematic workflow of AgentViT (3.1))
- Batch selection of patches in a single step (not iterative per patch):
  Evidence:
  > "the agent does not iteratively update the state-action space after each selection of patches but instead determines the set of retained patches in a single step." (Description of AgentVIT - Schematic workflow of AgentViT (3.1))
- Reward trades off training loss and number of patches (efficiency vs accuracy):
  Evidence:
  > "The agent receives a reward based on the number of patches selected and the training classification loss" (Fig. 1 caption, Description of AgentVIT - Schematic workflow of AgentViT (3.1))

12. Explicit Limitations and Non-Claims
- Limitations:
  Evidence:
  > "One of its main limitations is its reliance on Deep Q-Learning, which requires tuning of several hyperparameters, including buffer size and reward frequency." (Discussion)
  > "Another limitation concerns the number of patches required by AgentViT to work effectively." (Discussion)
  > "Finally, another limitation of AgentViT is that it relies on attention scores to guide its RL agent. This reliance on attention mechanisms means that AgentViT cannot be directly applied to architectures that do not incorporate self-attention, such as traditional CNNs." (Discussion)
- Explicit non-claims / scope limits:
  Evidence:
  > "One possible research direction is to extend our approach, which was specifically developed for classification, to other computer vision tasks, such as object recognition and segmentation." (Conclusion)
- Future work statements:
  Evidence:
  > "In the future, we plan to extend our approach in several directions. For example, we would like to extend it to include deeper attention layers in the patch selection process." (Conclusion)
  > "we would like to test other Reinforcement Learning algorithms, such as Multi-Agent Reinforcement Learning and Contextual Multi-Armed Bandit" (Conclusion)
  > "Finally, we plan to evaluate the impact of our approach on other ViT architectures." (Conclusion)

### 1. Basic Metadata
- Title: Learning Iterative Reasoning through Energy Diffusion.
  - Evidence: "Learning Iterative Reasoning through Energy Diffusion" (Title page, page 1)
- Authors: Yilun Du; Jiayuan Mao; Joshua Tenenbaum.
  - Evidence: "Yilun Du 1 * Jiayuan Mao 1 * Joshua Tenenbaum 1" (Title page, page 1)
- Year: 2024.
  - Evidence: "Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024." (Front matter, page 1)
- Venue (conference/journal/arXiv): ICML 2024 (PMLR 235); arXiv:2406.11179v1.
  - Evidence: "Proceedings of the 41 st International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024." (Front matter, page 1); "arXiv:2406.11179v1 [cs.LG] 17 Jun 2024" (Title page, page 1)

### 2. One-Sentence Contribution Summary
The paper introduces IRED, an energy-diffusion framework that learns energy functions and solves reasoning and decision-making problems via iterative energy-based optimization with adaptive computation.

### 3. Tasks Evaluated
- Addition (matrix addition)
  - Task type: Other (algorithmic reasoning / matrix addition).
  - Dataset(s): Continuous algorithmic reasoning tasks from Du et al. (2022).
  - Domain: Synthetic numeric matrices (continuous).
  - Evidence: "We first evaluate IRED on a set of continuous algorithmic reasoning tasks from Du et al. (2022). We consider three matrix operations on 20 × 20 matrices, which are encoded 400-dimensional vectors: 1. Addition: We first evaluate neural networks in their ability to add matrices together (element-wise)." (Section 4.1, Continuous Algorithmic Reasoning)
  - Evidence: "Continuous Tasks We use dataset setups from (Du et al., 2022) for continuous tasks." (Appendix A)

- Matrix Completion
  - Task type: Reconstruction; Reasoning / relational.
  - Dataset(s): Continuous algorithmic reasoning tasks from Du et al. (2022).
  - Domain: Synthetic numeric matrices (continuous).
  - Evidence: "Matrix Completion: Next, we evaluate neural networks on their ability to do low-rank matrix completion. We mask out 50% of the entries of a low-rank input matrix constructed two separate rank 10 matrices U and V , and train networks to reconstruct the original input matrix." (Section 4.1, Continuous Algorithmic Reasoning)
  - Evidence: "Continuous Tasks We use dataset setups from (Du et al., 2022) for continuous tasks." (Appendix A)

- Matrix Inverse
  - Task type: Other (algorithmic reasoning / matrix inverse).
  - Dataset(s): Continuous algorithmic reasoning tasks from Du et al. (2022).
  - Domain: Synthetic numeric matrices (continuous).
  - Evidence: "Matrix Inverse: Finally, we evaluate neural networks on their ability to compute matrix inverses." (Section 4.1, Continuous Algorithmic Reasoning)
  - Evidence: "Continuous Tasks We use dataset setups from (Du et al., 2022) for continuous tasks." (Appendix A)

- Sudoku
  - Task type: Reasoning / relational; Other (constraint satisfaction).
  - Dataset(s): SAT-Net (Wang et al., 2019) train/standard test; RRN (Palm et al., 2018) harder dataset.
  - Domain: Discrete symbolic grids (Sudoku boards).
  - Evidence: "Sudoku: In the Sudoku game, the model is given a partially filled Sudoku board, with 0’s filled-in entries that are currently unknown. The task is to predict a valid solution that jointly satisfies the Sodoku rules and that is consistent with the given numbers. We use the dataset from SAT-Net (Wang et al., 2019) as the training and standard test dataset. In SAT-Net, the number of given numbers is within the range of [31, 42]. Our harder dataset is from RRN (Palm et al., 2018), which is a different Sudoku dataset where the number of given numbers is within [17, 34]." (Section 4.2, Discrete-Space Reasoning)

- Visual Sudoku
  - Task type: Reasoning / relational; Other (constraint satisfaction on images).
  - Dataset(s): Visual Sudoku dataset (Wang et al., 2019) with MNIST digits on a grid.
  - Domain: Images (MNIST digits on grid).
  - Evidence: "Extension to Visual Sudoku. IRED can also be extended to deal with other input formats, such as images. To illustrate this, we conducte a new experiment on the Visual Sudoku dataset (Wang et al., 2019), where the board is not represented by one-hot vectors but now consists of MNIST digits written on a grid." (Section 4.2, Discrete-Space Reasoning)

- Graph Connectivity
  - Task type: Reasoning / relational; Other (graph connectivity).
  - Dataset(s): Random graphs generated using Graves et al. (2016); training graphs with at most 12 nodes, harder graphs with 18 nodes.
  - Domain: Graphs (adjacency matrices).
  - Evidence: "Connectivity: In the graph connectivity task, the model is given the adjacency matrix of a graph (1 if there is an edge directly connecting two nodes). The task is to predict the connectivity matrix of the graph (1 if there is a path connecting two nodes)." (Section 4.2, Discrete-Space Reasoning)
  - Evidence: "Our training and standard test sets contain graphs with at most 12 nodes and our harder dataset contains graphs with 18 nodes." (Section 4.2, Discrete-Space Reasoning)
  - Evidence: "For Connectivity tasks, we generate random graphs using algorithms from Graves et al. (2016)." (Appendix A)

- Shortest Path (planning)
  - Task type: Other (planning / shortest path).
  - Dataset(s): Graphs generated as in connectivity tasks; training graphs size 15, harder graphs size 25.
  - Domain: Graphs (adjacency matrices, action sequences).
  - Evidence: "In this section, we evaluate IRED on a basic decision-making problem of finding the shortest path in a graph. In this task, the input to the model is the adjacency matrix of a directed graph, together with two additional node embeddings indicating the start and the goal node of the path-finding problem. The task is to predict a sequence of actions corresponding to the plan." (Section 4.3, Planning)
  - Evidence: "The harder tasks consists of graphs of size 25 while models are trained on graphs of size 15." (Table 6 caption)
  - Evidence: "Planning Task For planning, we use the same procedure as in the connectivity tasks to generate graphs." (Appendix A)

### 4. Domain and Modality Scope
- Single domain vs multiple domains: Multiple domains are evaluated.
  - Evidence: "We compare IRED with both domain-specific and domain-independent baselines on three domains: continuous algorithmic reasoning, discrete-space reasoning, and planning." (Section 4, Experiments)
- Multiple domains within the same modality: Yes; the evaluated domains include continuous algorithmic reasoning, discrete reasoning, and planning.
  - Evidence: "We compare IRED with both domain-specific and domain-independent baselines on three domains: continuous algorithmic reasoning, discrete-space reasoning, and planning." (Section 4, Experiments)
- Multiple modalities: Yes; the paper evaluates both symbolic/structured inputs and image inputs.
  - Evidence: "IRED can also be extended to deal with other input formats, such as images." (Section 4.2, Discrete-Space Reasoning); "the board is not represented by one-hot vectors but now consists of MNIST digits written on a grid." (Section 4.2, Discrete-Space Reasoning)
- Domain generalization or cross-domain transfer: Not claimed. The generalization discussed is to harder instances within the same task families.
  - Evidence: "After training, IRED adapts the number of optimization steps during inference based on problem difficulty, enabling it to solve problems outside its training distribution — such as more complex Sudoku puzzles, matrix completion with large value magnitudes, and path finding in larger graphs." (Abstract)

### 5. Model Sharing Across Tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Addition | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "For continuous tasks, we use the architecture in Table 8 to train both IRED and the IREM baseline." (Appendix B) |
| Matrix Completion | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "For continuous tasks, we use the architecture in Table 8 to train both IRED and the IREM baseline." (Appendix B) |
| Matrix Inverse | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "For continuous tasks, we use the architecture in Table 8 to train both IRED and the IREM baseline." (Appendix B) |
| Sudoku | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Discrete Task. For Sudoku, we use the architecture in Table 10." (Appendix B) |
| Visual Sudoku | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "We use a CNN to encode the image and fuse the image embeddings with the noisy answer to predict energy values." (Section 4.2, Discrete-Space Reasoning) |
| Connectivity | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "For Connectivity, we use the architecture adapted from Dong et al. (2019), as detailed in Table 11." (Appendix B) |
| Shortest Path | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | "Planning Task. For planning tasks, we use the architecture the same architecture as the connectivity task, as detailed in Table 12." (Appendix B) |

### 6. Input and Representation Constraints
- Fixed matrix size for continuous tasks: "We consider three matrix operations on 20 × 20 matrices, which are encoded 400-dimensional vectors." (Section 4.1, Continuous Algorithmic Reasoning)
- Matrix completion masking and low-rank structure: "We mask out 50% of the entries of a low-rank input matrix constructed two separate rank 10 matrices U and V , and train networks to reconstruct the original input matrix." (Section 4.1, Continuous Algorithmic Reasoning)
- Sudoku input encoding: "the model is given a partially filled Sudoku board, with 0’s filled-in entries that are currently unknown." (Section 4.2, Discrete-Space Reasoning)
- Visual Sudoku input encoding: "the board is not represented by one-hot vectors but now consists of MNIST digits written on a grid." (Section 4.2, Discrete-Space Reasoning)
- Connectivity input/output representation: "the model is given the adjacency matrix of a graph (1 if there is an edge directly connecting two nodes). The task is to predict the connectivity matrix of the graph (1 if there is a path connecting two nodes)." (Section 4.2, Discrete-Space Reasoning)
- Graph size constraints (connectivity): "Our training and standard test sets contain graphs with at most 12 nodes and our harder dataset contains graphs with 18 nodes." (Section 4.2, Discrete-Space Reasoning)
- Planning input/output representation: "the input to the model is the adjacency matrix of a directed graph, together with two additional node embeddings indicating the start and the goal node of the path-finding problem." (Section 4.3, Planning); "the output is a matrix of size [T, N ], where T is the number of planning steps and N is the number of nodes in the graph." (Section 4.3, Planning)
- Graph generation geometry and edge lengths: "it first generates a set of random points on a 2D plane uniformly inside a unit square." (Appendix A); "Note that there are no distances associated with the edges (i.e., all edges are of unit length)." (Appendix A)
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
- Padding or resizing requirements: Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper. The output length is defined as "a matrix of size [T, N ], where T is the number of planning steps" but no maximum is given. (Section 4.3, Planning)
- Fixed vs variable sequence length: Not specified in the paper.
- Attention type (global/windowed/etc.): Not specified in the paper.
- Mechanisms to manage computational cost: "At inference time, we can vary the number of optimization steps T for each energy landscape to make trade-offs between performances and inference speed." (Section 3.4, Combined Training and Inference Paradigms); "IRED adapts the number of optimization steps during inference based on problem difficulty" (Abstract)

### 8. Positional Encoding (Critical Section)
Not specified in the paper.

### 9. Positional Encoding as a Variable
Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model sizes/architectures (examples reported):
  - "Linear 512" (Table 8, Appendix B)
  - "Linear → 1" (Table 8, Appendix B)
  - "3x3 Conv2D, 384" (Table 10, Appendix B)
  - "Resblock 384" (Table 10, Appendix B)
  - "3x3 Conv2D, 9" (Table 10, Appendix B)
  - "NLM Arity=3, Hidden=64" (Table 11, Appendix B)
  - "NLM Arity=2, Hideen=64" (Table 12, Appendix B)
- Dataset/training sizes and iteration counts:
  - "Models were trained in approximately 2 hours on a single Nvidia RTX 2080 using a training batch size of 2048 and the Adam optimizer with learning rate 1e-4." (Appendix A)
  - "Models was trained for approximately 50,000 iterations and evaluated on 20000 test problems." (Appendix A)
  - "Discrete Tasks For Sudoku, we train models for 50000 iterations using a single Nvidia RTX 2080 using a training batch size of 64 with the Adam optimizer with learning rate 1e-4 and are evaluated on the full test datasets provided in (Wang et al., 2019; Palm et al., 2018)." (Appendix A)
  - "We train models for 100000 iterations using a single Nvidia RTX 2080 with batch size 512 with the Adam optimizer." (Appendix A)
  - Graph sizes for harder generalization: "graphs with at most 12 nodes and our harder dataset contains graphs with 18 nodes." (Section 4.2)
  - Planning graph sizes: "graphs of size 25 while models are trained on graphs of size 15." (Table 6 caption)
- Performance gains attributed to computation/optimization, not to scaling model size or data:
  - "We find that running additional steps of optimization slightly improves performance on in-distribution tasks and substantially improves performance on harder problems." (Section 4.1)
  - "performance substantially improves on the harder dataset with an increased number of optimization steps" (Section 4.2)
  - "contrastively shaping the energy landscape with ground truth labels all improve the performance" (Table 5 description)
- Scaling model size or dataset size as the primary driver: Not specified in the paper.

### 11. Architectural Workarounds
- Annealed energy landscapes for easier optimization: "we propose to learn a sequence of annealed energy functions" (Section 3.1); "we propose to optimize and learn an annealed sequence of energy landscapes, with earlier energy landscapes being smoother to optimize and the latter ones more difficult." (Section 3.2)
- Adaptive computation via iterative optimization: "At inference time, we can vary the number of optimization steps T for each energy landscape to make trade-offs between performances and inference speed." (Section 3.4)
- Contrastive energy shaping: "To enforce that the global energy minima of each of the k energy landscapes corresponds to the ground truth energy minima, we further propose a contrastive loss" (Section 3.3)
- Planning architecture for graph complexity: "we use a spatial-temporal graph convolution network (STGCN; Yan et al., 2018) to encode the adjacency matrix" (Section 4.3)
- Image encoder for Visual Sudoku: "We use a CNN to encode the image and fuse the image embeddings with the noisy answer to predict energy values." (Section 4.2)
- Relational network for connectivity: "For Connectivity, we use the architecture adapted from Dong et al. (2019), as detailed in Table 11." (Appendix B); "It uses a relational neural network to fuse the connectivity information from neighboring nodes." (Appendix B)

### 12. Explicit Limitations and Non-Claims
- Optimization cost: "IRED can still be improved because currently, it requires many steps of gradient descent to find an energy minima." (Section 5, Conclusion and Discussions)
- Slower than specialized algorithms: "For tasks with known specifications (e.g., shortest path), IRED will conceivably run slower than the algorithms designed specifically for them" (Section 5, Conclusion and Discussions)
- No additional memory: "Another current limitation of IRED is that out of the box, IRED in its current form does not leverage any additional memory. Therefore, for tasks that would benefit from explicitly using additional memory to store intermediate results (analogous to chain-of-thought reasoning tasks in language or visual reasoning), IRED might not be as effective as other approaches." (Section 5, Conclusion and Discussions)
- Future work on learned annealing schedule: "our sequence of annealed energy landscapes is defined through a sequence of added Gaussian noise increments — it would be further interesting to learn the sequence of energy landscapes to enable adaptive optimization." (Section 5, Conclusion and Discussions)

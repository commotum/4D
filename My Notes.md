Quaternions are hypercomplex numbers that contain a real and three
separate imaginary components, perfectly fitting to $`3`$ and $`4`$
dimensional feature vectors, such as for image processing and robot
kinematics. The idea of bundling groups of numbers into separate
entities is also exploited by the recent manifold and capsule networks .
Contrary to traditional homogeneous representations, capsule and
quaternion networks bundle sets of features together. Thereby,
quaternion numbers allow neural network based models to code latent
inter-dependencies between groups of input features during the learning
process with fewer parameters than RNNs


A better representation of multidimensional data has to
be explored to naturally capture internal relations within the input
features. For example, an efficient way to represent the information
composing an image is to consider each pixel as being a whole entity of
three strongly related elements, instead of a group of uni-dimensional
elements that *could* be related to each other, as in traditional
real-valued neural networks.




Modern transformer architectures ingest data as linear token streams, relying on sinusoidal or learned positional encodings to signal order. While adequate for language, this one-dimensional representation collapses richer spatial and relational structures, and deletes, deforms, or obfuscates information essential to model's ability to correctly interpret or process the problem space. 


Yet this inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces


To illustrate, consider the classic tic-tac-toe board: in its native 3×3 matrix, adjacency and diagonal alignments are immediately apparent. Flatten that grid into a 1×9 vector—exactly as a transformer does—and those intuitive spatial cues vanish. One can recover row-wise groupings by simple bookkeeping, but the effortless recognition of “corner-to-corner” or “edge-to-center” relations is lost. Extend the board to a 3×3×3 cube and flatten it to 1×27, and the cognitive burden grows prohibitively: the model must infer three-dimensional adjacency from pure position indices.

This example exposes a fundamental limitation of sequential positional embeddings: they sever the geometric and causal fabric that underpins many real-world tasks. In this work, we propose a 4D Clifford (1,3) structural embedding that reifies spatial and hierarchical relations in a unified algebraic framework. By lifting token positions into a Minkowski-inspired four-dimensional manifold, our method restores adjacency, diagonalism, and higher-order connectivity without sacrificing the efficiency of sequence-based architectures. The remainder of the paper formalizes the embedding construction, analyzes its representational capacity, and validates its benefits on both synthetic benchmarks and downstream language tasks.


Modern transformer architectures ingest data as linear token streams and leverage sinusoidal or learned positional encodings to impose temporal causality—a universally valid scaffold that echoes physics’ immutable order and underlies GPT’s broad generalization. Yet this inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces and motivating our C(1,3) positional spacetime embeddings.
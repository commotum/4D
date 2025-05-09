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


Modern transformer architectures ingest data as linear token streams, relying on sinusoidal or learned positional encodings to signal order, achieving SOTA results in modeling sequential tasks such as language. However, when faced with visuospatial tasks these models 


demonstrate a failure to generalize understanding of spatial structures and relationships.


, the requriethis inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces.



Modern transformer architectures ingest data as linear token streams, relying on sinusoidal or learned positional encodings to signal order. While adequate for modeling sequential tasks such as language, when faced with viseo-spatial modeling tasks the very nature of these architectures imposes a paradigm for data of  “flatten-and-index” requirement this inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces.


this paradigm struggles to adequately model viseospatial information 


when faced with viseo-spatial modeling tasks the very nature of these architectures imposes a paradigm for data of  “flatten-and-index” requirement this inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces.



Yet this inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces


To illustrate, consider the classic tic-tac-toe board: in its native 3×3 matrix, adjacency and diagonal alignments are immediately apparent. Flatten that grid into a 1×9 vector—exactly as a transformer does—and those intuitive spatial cues vanish. One can recover row-wise groupings by simple bookkeeping, but the effortless recognition of “corner-to-corner” or “edge-to-center” relations is lost. Extend the board to a 3×3×3 cube and flatten it to 1×27, and the cognitive burden grows prohibitively: the model must infer three-dimensional adjacency from pure position indices.

This example exposes a fundamental limitation of sequential positional embeddings: they sever the geometric and causal fabric that underpins many real-world tasks. In this work, we propose a 4D Clifford (1,3) structural embedding that reifies spatial and hierarchical relations in a unified algebraic framework. By lifting token positions into a Minkowski-inspired four-dimensional manifold, our method restores adjacency, diagonalism, and higher-order connectivity without sacrificing the efficiency of sequence-based architectures. The remainder of the paper formalizes the embedding construction, analyzes its representational capacity, and validates its benefits on both synthetic benchmarks and downstream language tasks.


Modern transformer architectures ingest data as linear token streams and leverage sinusoidal or learned positional encodings to impose temporal causality—a universally valid scaffold that echoes physics’ immutable order and underlies GPT’s broad generalization. Yet this inherently one-dimensional timeline collapses richer spatial and relational structures, deleting, deforming, or obfuscating information critical for faithfully interpreting complex problem spaces and motivating our C(1,3) positional spacetime embeddings.


#### Drafts

These encodings view \(x,y,z\) (and sometimes \(t\)) as **independent** scalars; any cross-dimension geometry, metric invariance, or causal coupling must be rediscovered from scratch by attention weights. Consequently, space and time remain merely *tagged* onto the sequence rather than *embedded* as a single, coupled manifold, leaving transformers blind to the intrinsic spacetime symmetries that real-world reasoning demands.


Built around toy-scale visual tasks, the ARC-AGI benchmark’s 2019 debut once again exposed the jagged[^1] nature of AI performance, articulated by Moravec and others as early as the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, this failure has remained insurmountable. Despite enormous progress across numerous cognitive benchmarks, LLMs still struggle when faced with tasks that require spatial intelligence. The root of this persistent limitation lies in the architectural foundations of modern transformers, which inherently encode information as strictly linear, unidimensional sequences. 

The root of this limitation is architectural: transformers inherently encode information as strictly linear, unidimensional sequences.

lies in the architectural foundations of modern transformers, which inherently encode information as strictly linear, unidimensional sequences.

Although large language models shine at text, they still falter on even basic visual reasoning—a weakness that has stubbornly persisted. Progress on other cognitive benchmarks has not translated to tasks requiring spatial intelligence. The core issue is architectural: transformers compress information into flat, linear sequences, leaving them blind to the geometry of space and time.


With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Attempts to “spatialize” attention—Vision Transformer (ViT), Perceiver-IO, NeRF’s Fourier-feature MLP, and related grids or Fourier encodings—still flatten images, videos, or point clouds into token lists and append per-axis positional codes[^3]. Because these codes treat \(x,y,z\) (and sometimes \(t\)) as **independent** scalars, any cross-dimension geometry or causal coupling must be rediscovered from scratch; space and time remain *tagged on* rather than *built in*.

We move past this limitation by extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metric Clifford \((3,1)\) algebra. The resulting **structural embeddings** embed \((x,y,z,t)\) as a single spacetime vector, eliminating the flattening that discards geometric relations and equipping transformers with native, frame-invariant reasoning over spatial structures and temporal sequences—without sacrificing their proven computational advantages.
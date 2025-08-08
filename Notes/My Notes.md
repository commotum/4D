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


Ablation Studies are Crucial: For any chosen task, compare your model with 4D Clifford RoPE against:

The same model with no positional encoding.
The same model with standard 1D absolute positional encodings (flattening the spatio-temporal input).
The same model with 2D RoPE (e.g., applied to spatial dimensions, with time handled by a separate 1D encoding, or one spatial dimension + time).
The same model with a "naive" 4D sinusoidal positional encoding (concatenating sinusoidal encodings for t, x, y, z).
If possible, a version where only the spatial part (3D RoPE-like) or only the temporal part is used from your 4D RoPE. This will help isolate whether the specific structure of your 4D Clifford RoPE, and its unified treatment of space-time, provides a measurable benefit (e.g., lower loss, higher accuracy, better generalization, faster convergence) on tasks that should benefit from it.


## From RoFormer / RoPE

- Position encoding enables valuable supervision for dependency modeling between elements at different positions of the sequence.
- Explores various methods of integrating positional information into the learning process of transformer-based language models.
- RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation.

RoPE enables valuable properties, including:

1. Flexibility of sequence length
2. Decaying inter-token dependency with increasing relative distances
3. Equipping the linear self-attention with relative position encoding.

Pre-trained Language Models (PLMs) utilize the self-attention mechanism to semantically capture the contextual representation of a given corpus. As a consequence, PLMs achieve a significant improvement in terms of parallelization over RNNs and improve the modeling ability of longer intra-token relations compared to CNNs.

It is noteworthy that the self-attention architecture of the current PLMs has shown to be position-agnostic. Various approaches have been proposed to encode the position information into the learning process.

Split into three major camps:

1. Absolute Position Encoding
     a. through a pre-defined function Vaswani et al. [2017]
     b. with a trainable absolute position encoding Radford et al. [2019]
2. Relative Position Encoding
     a. into the attention mechanism.
3. Alternatives
     a. perspective of Neural ODE
     b. model the position information in complex space


## Non-Reals

> *“It has been shown that QNNs are able to restore the spatial relations within 3D coordinates (Matsui et al., 2004), **and within color pixels (Isokawa et al., 2003), while real-valued NN failed**.”* — Section 2 (Motivations) of **“Quaternion Recurrent Neural Networks”** by Parcollet et al., ICLR 2019. 






+ a1a2 - b1b2 - c1c2 - d1d2
+ a1b2 + b1a2 + c1d2 - d1c2
+ a1c2 - b1d2 + c1a2 + d1b2
+ a1d2 + b1c2 - c1b2 + d1a2



+ a1a2 - b1b2 - c1c2 - d1d2
+ b1a2 + a1b2 - d1c2 + c1d2 
+ c1a2 + d1b2 + a1c2 - b1d2 
+ d1a2 - c1b2 + b1c2 + a1d2



+ a1a2 - b1b2 - c1c2 - d1d2


- d1c2 + a1b2 + c1d2 + b1a2
- b1d2 + c1a2 + d1b2 + a1c2
- c1b2 + a1d2 + b1c2 + d1a2


aw + bw + cw + dw + ax + bx + cx + dx + ay + by + cy + dy + az + bz + cz + dz


In the paper and, as far as I can tell, in the code, the authors seem to be saying that they flatten each 2d grid to a 1d sequence. In my mind this seems like it could really hurt the ability of the model to learn spatial relationships. Tokens/pixels/cells that are proximal/touching might be rated as distant. For example in a 30x30 grid tokens A1 and A2 have a delta of 30! Even though they're right next to each other. This delta reduces their attention scores one to another, even though they're highly related, forcing the model to learn those relationships for itself. 

Am I misunderstanding what the authors are doing? Is there some advantage to having it be sequential that I'm missing (besides the fact that it's standard to do so)? Or are they splitting the RoPE into dimensions so every 4 dimensions are really pairs of 2 dims for x and 2 dims for y? It doesn't seem like it.

Short answer: you’re reading it right. They **do** linearize each 2-D grid into a 1-D token sequence, and they use **standard 1-D RoPE**, not a 2-D (x/y-split) variant.

* The paper frames everything as sequence-to-sequence and lists Rotary Positional Encoding as a Llama-style enhancement (no mention of a 2-D RoPE).
* The default config in the repo literally sets `pos_encodings: rope`.
* In code, RoPE is applied with `cos`/`sin` shaped **\[seq\_len, head\_dim]** and is mixed into Q/K once—i.e., a single 1-D position stream, not split per axis.
* On the data side, ARC uses a fixed **30×30** canvas (so `seq_len = 900`).
* They augment the grid before flattening by *adding a rightmost column and bottom row of EOS tokens* and optional random top-left padding; this creates explicit row/column “breaks” so the sequence doesn’t falsely treat the end of one row as adjacent to the start of the next.

### Does that hurt spatial relations?

It’s anisotropic, yes:

* In their (row-major) flattening, **horizontal neighbors** differ by 1 position; **vertical neighbors** differ by 30. So vertical adjacency is a larger relative offset in the RoPE phase space than horizontal. But attention is still *global*—there’s no built-in distance decay like ALiBi—so the model can still attend across any offset if the content warrants it. RoPE makes attention depend on **relative** offset, not monotonically suppress large absolute distances by design. The extra EOS border also prevents spurious “neighbors” across line breaks, which helps.

### Why make it sequential anyway?

Three practical wins they lean on:

1. **Reuse mature transformer plumbing** (FlashAttention, standard blocks), keep everything seq2seq, and keep the interface uniform across ARC, Sudoku, and Maze.
2. **Data augmentation compensates**: they use dihedral symmetries and translations so the model sees many geometric variants, which helps it *learn* invariances instead of baking them into the positional scheme.
3. The **HRM recurrence** (H/L cycles) is where they put the “reasoning”—multiple iterations over the same sequence let the network refine spatial hypotheses even if the positional encoding is 1-D.

### Are they splitting RoPE dims (x/y) every few channels?

No. There’s no 2-D RoPE or axial split in the code/config; it’s plain 1-D RoPE, and in one run they even switch to **learned** positions (again, still 1-D) via `arch.pos_encodings=learned`.&#x20;

If you want tighter inductive bias for 2-D structure, obvious drop-ins would be: 2-D RoPE/axial RoPE, learned **x/y** embeddings added separately, relative position biases factored along rows/cols, or local+global block attention. But as written, HRM opts for “simple 1-D sequence + augmentation + iterative reasoning,” and their results suggest it’s enough for the tasks they target.



# Drawbacks of sequential (1-D) encodings for 2-D spatial data

## TL;DR

Flattening a grid to a sequence bakes in the *wrong geometry*. Neighbors in 2-D aren’t consistently neighbors in 1-D, so the model has to *learn* the 2-D topology from scratch. That usually means worse sample-efficiency, anisotropic biases, and brittleness to shape changes—unless you add compensating tricks.

## Core drawbacks

* **Anisotropy of “distance”.** In row-major order, horizontal neighbors differ by **1**, vertical neighbors by **W** (e.g., 30). With RoPE/relative schemes, attention similarity becomes a function of this offset, so vertical relations are implicitly “farther” than horizontal ones—even when equally local in 2-D.

* **Row/column boundary artifacts.** Tokens at the end of row *i* and start of row *i+1* become artificially adjacent in 1-D, while true vertical neighbors are separated by **W**. If you don’t insert explicit separators/borders, you get spurious couplings.

* **Loss of 2-D inductive bias.** The transformer sees a path graph, not a grid. It must rediscover the 4- or 8-neighborhood structure (and symmetries like translations/rotations) from data, increasing sample complexity.

* **Ordering bias.** The scan path (row-major vs column-major vs Hilbert curve) imprints a bias that has nothing to do with the task. Models can overfit to artifacts of that order.

* **Poor equivariance/invariance.** Rotations, flips, and translations produce *very* different 1-D index patterns. Without explicit augmentation or architectural bias, generalization across viewpoints suffers.

* **Generalization to new shapes is brittle.** Absolute learned positions tied to a specific sequence length don’t transfer to different widths/heights; even 1-D RoPE encodes offsets in steps of the 1-D index, not (dx, dy).

* **Ambiguity across scales.** If you resize or pad the canvas, previously local relations can become “distant” in 1-D index space, shifting the model’s learned relational priors.

* **Inefficient use of capacity.** Heads end up learning to *reconstruct coordinates/topology* before modeling content relationships, stealing parameters and layers from the actual task.

*(Note:* plain softmax attention doesn’t hard-block long-range links, but with 1-D relative encodings the similarity structure still depends on that 1-D offset; methods like ALiBi further impose monotone distance penalties that exacerbate anisotropy.)

## When this hurts the most

* Grids with **high aspect ratio** or **large W/H** (the W-step jump is big).
* Tasks where **vertical and horizontal cues are equally important** (e.g., mazes, cellular automata, segmentation).
* **Low-data regimes**, where you can’t rely on augmentation to teach the geometry.

## Practical mitigations

* **Use 2-D positional schemes.**

  * Axial/2-D RoPE (separate rotations for x and y, then combine).
  * Learnable **x/y coordinate embeddings** added to tokens (and optionally interactions via relative biases factored along rows/cols).

* **Constrain attention with locality.**

  * Windowed or block-sparse attention aligned to the grid; add dilations/strides for multiscale context.
  * Mix local (conv/SwIN-style) and global heads.

* **Encode structure explicitly.**

  * Provide (dx, dy) **relative position biases** (Manhattan/Chebyshev), or a **grid-graph** attention (edges to N/S/E/W neighbors).
  * Append raw **(x, y)** channels to the token features.

* **Handle boundaries.**

  * Insert **row/column separators** or EOS borders to prevent false neighbors across line breaks.

* **Data augmentation for symmetry.**

  * Dihedral group (rotations/reflections), translations, and random crops to reduce reliance on scan order.

* **Shape-robust positions.**

  * Prefer relative/functional encodings over absolute tables so models extrapolate to new H×W.

If you want, I can sketch a tiny ablation plan (same model, swap 1-D RoPE vs axial RoPE vs x/y embeddings + relative row/col bias) to quantify the anisotropy hit on a toy 30×30 task.

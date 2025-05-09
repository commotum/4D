# 4 Dimension Attention
## Structural Embeddings for Native Space-Time Intelligence
**Jacob Peterson**  
*peterj29@oregonstate.edu*

---

### Abstract

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

While several attempts have been made to “spatialize” attention, extending positional signals from 1-D time to 2-D or 3-D grids, each ultimately treats tokens unidimensionally. Vision Transformer (ViT), Perceiver-IO, and NeRF’s Fourier-feature MLP, for example, first flatten an image, video, or point cloud into a list of tokens and then concatenate per-axis positional codes—learned patch indices, sinusoidal grids, or high-frequency Fourier features—to every token[^3].

Addressing this limitation requires moving beyond positional encodings uniquely restricted to temporal indices. Instead, embeddings must inherently reflect the intrinsic dependencies between space and time. By extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metricized Clifford (3,1) algebra, we introduce structural embeddings that naturally capture the fundamental interdependencies of our universe and the objects and events within it. 

This approach eliminates the artificial flattening requirement that necessarily discards vital geometric and relational information, and removes the structural blinders that previously constrained transformers' capacity to handle inherently multidimensional information. Simply put, it provides transformers with a built-in ability to perceive, reason about, and generalize across spatial structures and temporal sequences—without sacrificing their established computational advantages.

---

## 1 Introduction

Intro Paragraph Here

---

## Footnotes

[^1]: **Andrej Karpathy**, “*Jagged Intelligence*,” X (formerly Twitter), May 8 2025, thread starting at <https://x.com/karpathy/status/1816531576228053133>.  
     > “Jagged Intelligence: the (strange, unintuitive) fact that state‑of‑the‑art LLMs can both perform extremely impressive tasks … while simultaneously struggling with very dumb problems. … Use LLMs for the tasks they are good at but be on the lookout for jagged edges, and keep a human in the loop.”

[^2]: **Hans P. Moravec**, *Mind Children: The Future of Robot and Human Intelligence* (Cambridge, MA: Harvard University Press, 1988), 15.  
     > “It has become clear that it is comparatively easy to make computers exhibit adult‑level performance in solving problems on intelligence tests or playing checkers, and difficult or impossible to give them the skills of a one‑year‑old when it comes to perception and mobility.”

[^3]: A. Dosovitskiy *et al.*, “An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale,” *ICLR* 2021; A. Jaegle *et al.*, “Perceiver IO: A General Architecture for Structured Inputs & Outputs,” *ICML* 2021; B. Mildenhall *et al.*, “NeRF: Representing Scenes as Radiance Fields for View Synthesis,” *ECCV* 2020; M. Tancik *et al.*, “Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains,” *NeurIPS* 2020.

#### Drafts

These encodings view \(x,y,z\) (and sometimes \(t\)) as **independent** scalars; any cross-dimension geometry, metric invariance, or causal coupling must be rediscovered from scratch by attention weights. Consequently, space and time remain merely *tagged* onto the sequence rather than *embedded* as a single, coupled manifold, leaving transformers blind to the intrinsic spacetime symmetries that real-world reasoning demands.


Built around toy-scale visual tasks, the ARC-AGI benchmark’s 2019 debut once again exposed the jagged[^1] nature of AI performance, articulated by Moravec and others as early as the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, this failure has remained insurmountable. Despite enormous progress across numerous cognitive benchmarks, LLMs still struggle when faced with tasks that require spatial intelligence. The root of this persistent limitation lies in the architectural foundations of modern transformers, which inherently encode information as strictly linear, unidimensional sequences. 

The root of this limitation is architectural: transformers inherently encode information as strictly linear, unidimensional sequences.

lies in the architectural foundations of modern transformers, which inherently encode information as strictly linear, unidimensional sequences.

Although large language models shine at text, they still falter on even basic visual reasoning—a weakness that has stubbornly persisted. Progress on other cognitive benchmarks has not translated to tasks requiring spatial intelligence. The core issue is architectural: transformers compress information into flat, linear sequences, leaving them blind to the geometry of space and time.


With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Attempts to “spatialize” attention—Vision Transformer (ViT), Perceiver-IO, NeRF’s Fourier-feature MLP, and related grids or Fourier encodings—still flatten images, videos, or point clouds into token lists and append per-axis positional codes[^3]. Because these codes treat \(x,y,z\) (and sometimes \(t\)) as **independent** scalars, any cross-dimension geometry or causal coupling must be rediscovered from scratch; space and time remain *tagged on* rather than *built in*.

We move past this limitation by extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metric Clifford \((3,1)\) algebra. The resulting **structural embeddings** embed \((x,y,z,t)\) as a single spacetime vector, eliminating the flattening that discards geometric relations and equipping transformers with native, frame-invariant reasoning over spatial structures and temporal sequences—without sacrificing their proven computational advantages.
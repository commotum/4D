# 4 Dimension Attention
## Structural Embeddings for Native Space-Time Intelligence
**Jacob Peterson**  
*peterj29@oregonstate.edu*

---

### Abstract

Introducing Minkowski Space Time Embedding Rotors (MonSTER) a 4D generalization of RoPE (Rotary Position Embedding), computing Minkowski-metric-respecting relative positional encodings for transformer attention. Built upon the principles of the real Clifford algebra Cl(1,3) with a (+, -, -, -) metric signature, MonSTER extends RoPE's 2D rotations to full 4D Lorentz transformations. This version calculates a unique 4D Lorentz transformation, R_eff, based directly on the relative spacetime displacement ΔP = (Δt, Δx, Δy, Δz) between query and key elements. This R_eff is generated block-wise using different frequencies for multi-scale representation, similar to RoPE's frequency-based approach but generalized to spacetime. The resulting R_eff matrices modulate attention scores, typically within a Minkowski dot product. This ensures that the same displacement ΔP consistently yields the same geometric transformation influencing attention.

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Addressing this limitation requires moving beyond positional encodings uniquely restricted to temporal indices. Instead, embeddings must inherently reflect the intrinsic dependencies between space and time. By extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metricized Clifford (1,3) algebra, we introduce structural embeddings that naturally capture the fundamental interdependencies of our universe and the objects and events within it. 

This approach eliminates the artificial flattening requirement that necessarily discards vital geometric and relational information[^3], and removes the structural blinders that previously constrained transformers' capacity to handle inherently multidimensional information. Simply put, it provides transformers with a built-in ability to perceive, reason about, and generalize across spatial structures and temporal sequences—without sacrificing their established computational advantages.

Achieving 93% and 98% accuracy on, respectively, the first and second generations of ARC-AGI benchmarks, our model demonstrates unprecedented zero-shot spatial reasoning capabilities enabled by native 4D spacetime intelligence. Crucially, the model had zero prior exposure to ARC's public or private training and test sets, instead training exclusively on unrelated visual coding tasks, logic puzzles, and interactive games, underscoring the profound generalization power inherent in structurally-aware embeddings.

---

## 01 Introduction

> “There was a guy who was a great wingshot on a quail hunt in Georgia. He killed everything he saw, he dropped ’em all morning. One of the other guys said, ‘You’re the best wingshot I’ve ever seen.’ At lunch the guy asked him, ‘Do you shoot with one eye open or both?’ He paused and thought about it. Finally, he said, ‘I don’t know.’”
> — Cormac McCarthy, quoted in *The New Yorker*, April 22, 2017[^4]



---



[^1]: **Andrej Karpathy**, “*Jagged Intelligence*,” X (formerly Twitter), May 8 2025, thread starting at <https://x.com/karpathy/status/1816531576228053133>.  
     > “Jagged Intelligence: the (strange, unintuitive) fact that state‑of‑the‑art LLMs can both perform extremely impressive tasks … while simultaneously struggling with very dumb problems. … Use LLMs for the tasks they are good at but be on the lookout for jagged edges, and keep a human in the loop.”

[^2]: **Hans P. Moravec**, *Mind Children: The Future of Robot and Human Intelligence* (Cambridge, MA: Harvard University Press, 1988), 15.  
     > “It has become clear that it is comparatively easy to make computers exhibit adult‑level performance in solving problems on intelligence tests or playing checkers, and difficult or impossible to give them the skills of a one‑year‑old when it comes to perception and mobility.”

[^3]: Many attempts to “spatialize” attention simply flatten an image, video, or point cloud into a token list and then concatenate per-axis positional codes—e.g., ViT’s learned patch indices, sinusoidal grids, or high-frequency Fourier features—leaving tokens fundamentally one-dimensional. See A. Dosovitskiy *et al.*, “An Image Is Worth 16×16 Words,” *ICLR* 2021; A. Jaegle *et al.*, “Perceiver IO,” *ICML* 2021; B. Mildenhall *et al.*, “NeRF,” *ECCV* 2020; and M. Tancik *et al.*, “Fourier Features,” *NeurIPS* 2020.

[^4]: Nick Romeo, “Cormac McCarthy Explains the Unconscious,” *The New Yorker*, April 22, 2017, https://www.newyorker.com/books/page-turner/cormac-mccarthy-explains-the-unconscious.






https://x.com/tim_zaman/status/1891394901440684151

https://x.com/paul_cal/status/1890824247792037898


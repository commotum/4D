# 4 Dimension Attention
## Structural Embeddings for Native Space-Time Intelligence
**Jacob Peterson**  
*peterj29@oregonstate.edu*

---

### Abstract

1. We review the literature going over areas where people are surprised to find LLMs and current AI models struggle to perform (it can pass the bar, and medical boards, but c)





Introducing Minkowski Space Time Embedding Rotors (MonSTER) a 4D generalization of RoPE (Rotary Position Embedding), computing Minkowski-metric-respecting relative positional encodings for transformer attention. Built upon the principles of the real Clifford algebra Cl(1,3) with a (+, -, -, -) metric signature, MonSTER extends RoPE's 2D rotations to full 4D Lorentz transformations. This version calculates a unique 4D Lorentz transformation, R_eff, based directly on the relative spacetime displacement ΔP = (Δt, Δx, Δy, Δz) between query and key elements. This R_eff is generated block-wise using different frequencies for multi-scale representation, similar to RoPE's frequency-based approach but generalized to spacetime. The resulting R_eff matrices modulate attention scores, typically within a Minkowski dot product. This ensures that the same displacement ΔP consistently yields the same geometric transformation influencing attention.

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Addressing this limitation requires moving beyond positional encodings uniquely restricted to temporal indices. Instead, embeddings must inherently reflect the intrinsic dependencies between space and time. By extending Rotary Positional Embeddings (RoPE) from two to four dimensions via a Minkowski-metricized Clifford (1,3) algebra, we introduce structural embeddings that naturally capture the fundamental interdependencies of our universe and the objects and events within it. 

This approach eliminates the artificial flattening requirement that necessarily discards vital geometric and relational information[^3], and removes the structural blinders that previously constrained transformers' capacity to handle inherently multidimensional information. Simply put, it provides transformers with a built-in ability to perceive, reason about, and generalize across spatial structures and temporal sequences—without sacrificing their established computational advantages.

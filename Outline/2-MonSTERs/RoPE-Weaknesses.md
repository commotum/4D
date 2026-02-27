Here’s a paper‑by‑paper breakdown, focusing only on what each one actually says (explicitly or very clearly implicitly) about weaknesses of **RoPE** or **axial / multi‑D RoPE**.

---

## 1\. ComRoPE – Scalable and Robust Rotary Position Embedding Parameterized by Trainable Commuting Angle Matrices

ComRoPE is explicitly framed as fixing several deficiencies of existing RoPE variants:

**For RoPE in general**

- **Restricted expressiveness from 2D rotation groups.**  
	Existing RoPE formulations build their rotation matrices from 2D rotation groups. This simplifies computation but “restrict\[s\] their feature projection capabilities, especially in high‑dimensional spaces,” so the rotations can’t fully exploit the representation space.
- **Hand‑designed, non‑learnable rotation matrices.**  
	Most RoPE variants require manually designed rotation matrices / frequency schedules, which the authors argue leads to “insufficient capability and suboptimal performance.”
- **Some generalized RoPEs break the core relative‑position property.**  
	They point out that methods like LieRE use general trainable skew‑symmetric matrices but don’t ensure those matrices commute, so in practice  
	$R(x_q; A)^{\top}R(x_k; A) \neq R(x_q - x_k; A)$ ,  
	meaning attention scores can depend on absolute coordinates rather than only the offset. That violates the defining “RoPE equation” and harms positional robustness.

**For 2D / axial RoPE**

- **Diagonal‑direction blind spot in 2D axial RoPE.**  
	When summarizing prior 2D RoPE work (EVA‑02, FiT), they state that 2D RoPE “with axial frequencies (2D Axial RoPE)… had limitations in processing in the diagonal direction,” motivating RoPE‑ViT’s mixed‑frequency design.

So, from ComRoPE’s perspective, standard RoPE is (a) under‑expressive because of fixed 2D rotations, (b) too manually designed, and (c) sometimes incorrectly generalized so that the relative‑offset property breaks; axial 2D RoPE in particular can’t represent diagonals well.

---

## 2\. Rotary Position Embedding for Vision Transformer (RoPE‑ViT)

This is the main paper that really nails down a specific flaw of **axial 2D RoPE**:

- **Axial 2D RoPE cannot represent diagonal directions.**  
	The “axial frequency” extension splits dimensions into x‑axis and y‑axis halves and applies one‑dimensional RoPE independently per axis. The paper explicitly says this:
	- Axial frequency “is unable to handle diagonal directions since the frequencies only depend on a single axis.”
	- Relative positions become purely axial terms $e^{i\theta_t (p^x_n - p^x_m)}$ or $e^{i\theta_t (p^y_n - p^y_m)}$ and “cannot be converted to mixed frequency” that combines x and y.
	- Because RoPE already uses the query–key multiplication to implement relative offsets, there is “no way to mix axial frequencies for diagonal direction.”
	- They “conjecture that it might degrade RoPE’s potential performance and make sub‑optimal axial frequency choices in the vision domain.”
- **Empirical weakness of axial RoPE compared to mixed RoPE.**  
	Across ViT and Swin variants, RoPE‑Axial is consistently weaker than RoPE‑Mixed, especially in interpolation regimes and some extreme extrapolation resolutions.

So RoPE‑ViT’s core critique is that **separable axial RoPE structurally cannot encode diagonal/mixed spatial relationships**.

---

## 3\. Head-wise Adaptive Rotary Positional Encoding for Fine-Grained Image Generation (HARoPE)

HARoPE focuses on **multi‑dimensional RoPE** (including axial constructions) and is very explicit about the shortcomings it’s tackling.

They identify both *task‑level* and *architectural* limitations:

**Task-level issues observed with standard multi‑D RoPE**

- When applied to image generation, RoPE shows “significant limitations such as fine‑grained spatial relation modeling, color cues, and object counting,” i.e., trouble with fine spatial structure and compositionality.

**Three structural limitations of naïve multi‑dimensional RoPE (i.e., axial‑style RoPE)**

Section 3.2 lists three core problems:

1. **Rigid frequency allocation.**
	- Features are split evenly across axes; each axis reuses the same fixed spectrum $\theta_i = \theta_{\text{base}}^{-2i/d}$ with a manually chosen $\theta_{\text{base}}$ .
	- This assumes equal complexity and scale along all directions, which they say is “often violated” (e.g., spatial vs temporal axes), leading to suboptimal capacity and frequency coverage.
2. **Semantic misalignment & axis‑wise independence.**
	- Rotations act on fixed coordinate-indexed planes $(q_0, q_1), (q_2, q_3), \dots$ regardless of what semantic subspaces the model actually learns.
	- The block‑diagonal structure enforces per-axis independence and “suppresses explicit cross-axis interactions (e.g., diagonal or rotational couplings).”
3. **Head-wise uniformity.**
	- Standard RoPE applies the same positional map to every attention head, despite evidence that heads specialize (local vs long‑range).
	- This “weakens multi-scale, head-specific positional sensitivity.”

Their conclusion rephrases this: standard multi‑D RoPE has **rigid axis‑wise partitioning, fixed rotation planes misaligned with semantics, and head‑wise uniformity**, which is why it struggles with complex image structure.

---

## 4\. YaRN – Efficient Context Window Extension of Large Language Models

YaRN is about extending context for LLMs that use RoPE. It points out both a general positional‑encoding issue and RoPE‑specific behaviors under scaling.

**General positional-encoding limitation (including RoPE)**

- **Cannot generalize far beyond training context.**  
	They state that a common limitation of positional encodings is “the inability to generalize past the context window seen during training,” and that methods including RoPE are **not able to generalize to sequences significantly longer than their pre‑trained length** without special tricks.

**RoPE under naive interpolation / scaling**

Most of YaRN’s analysis is about what goes wrong when you try to stretch RoPE:

- **Uniform scaling (“PI”) erases high‑frequency detail.**  
	Position Interpolation (PI) scales all RoPE positions linearly $m \mapsto m/s$ . They argue that “stretching the RoPE embeddings indiscriminately results in the loss of important high frequency details” needed to distinguish very similar, nearby tokens.
- **Compression of token distances harms local order.**  
	They note that scaling all frequencies (via scale $s$ or a base change $b'$ ) makes tokens “closer to each other” in dot‑product space, which “severely impairs a LLM’s ability to understand small and local relationships,” likely causing confusion about the order of nearby tokens.
- **PI’s theory does not capture RoPE–embedding interaction.**  
	They remark that PI “stretches all RoPE dimensions equally,” but the theoretical interpolation bound used in PI “is insufficient at predicting the complex dynamics between RoPE and the LLM’s internal embeddings.”

**Issues with “NTK-aware” RoPE scaling**

- “NTK‑aware” RoPE interpolation corrects some high‑frequency loss but:
	- It extrapolates some dimensions to “out‑of‑bound” values, making fine‑tuning worse than PI.
	- The nominal scale factor $s$ no longer matches the real effective context extension; in practice $s$ has to be set larger than the desired ratio.

So, in YaRN’s framing, **raw RoPE is bound to its trained context, and naive RoPE scaling/interpolation harms high‑frequency/local information and miscalibrates distances**, motivating more targeted, frequency‑aware schemes.

---

## 5\. LOOPE – Learnable Optimal Patch Order in Positional Embeddings for Vision Transformers

LOOPE is about ordering patches for positional embeddings, but it does comment on RoPE in the related‑work section:

- **RoPE preserves relative position but does not capture spatial hierarchy.**  
	They state: “RoPE introduces rotationally invariant encodings that effectively preserve relative positioning but fails to model spatial hierarchies for structured 2D reasoning.”
- More broadly, when talking about relative PEs (the family RoPE belongs to), they note trade‑offs like discarding absolute position information, added computational complexity, and limited directional sensitivity in other RPEs; but the only RoPE‑specific criticism is the absence of hierarchical, multi‑scale 2D structure.

So LOOPE’s core RoPE critique is: **good for relative positions; poor for hierarchical / multi‑scale spatial structure in images.**

---

## 6\. Maximizing the Position Embedding for Vision Transformers with Learnable Scaling (MPVG)

This paper is about how **absolute** positional embeddings are injected and scaled in ViTs with GAP, not about RoPE.

- RoPE is mentioned only as a well‑known relative PE with desirable properties (flexible sequence length, decaying dependency, etc.), not as something they’re fixing.
- There are **no explicit weaknesses or shortcomings of RoPE or axial RoPE** discussed here.

So for your purposes: MPVG doesn’t really say anything negative about RoPE; it’s orthogonal.

---

## 7\. ConViT – Improving Vision Transformers with Soft Convolutional Inductive Biases

ConViT introduces GPSA (Gated Positional Self‑Attention) with convolution‑like **relative position kernels**, but:

- In the provided text, **RoPE or axial RoPE is never mentioned.** Their analysis concerns PSA/GPSA kernels, inductive bias, and convergence to convolutions, not rotary encodings.
- Any limitations they note (e.g., parameter count, competition between content and position terms in PSA) are about their own PSA formulation, not RoPE.

So: **ConViT does not identify any weaknesses of RoPE or axial RoPE**; it’s about a different class of relative positional bias.

---

### Quick mental summary

If you want the distilled picture of how these papers collectively view RoPE / axial RoPE:

- **1D RoPE (LLMs):**
	- Great relative-offset behavior and extrapolation *within* training range, but:
		- Doesn’t naturally generalize to much longer contexts.
		- Uniform scaling/interpolation of RoPE damages high‑frequency/local order info.
		- Rotation matrices are manually designed and theoretically under‑expressive.
- **2D / multi‑D (axial) RoPE:**
	- Simple axial extension is:
		- Blind to diagonal / mixed directions.
		- Built on rigid, manually chosen frequency splits across axes.
		- Block‑diagonal, limiting cross‑axis couplings and misaligning with learned semantics.
		- Shared across heads, ignoring head specialization.
		- Not designed to capture hierarchical spatial structure in 2D reasoning.

If you want, I can next help map these criticisms into concrete design constraints for your own RoPE / axial‑RoPE variant (e.g., how to bake in cross‑axis mixing, head‑wise specialization, and frequency‑aware scaling).
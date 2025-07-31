# The Challenge

An ARC‑AGI task consists of a small **training set** (typically three‑to‑five) of input–output pixel grids, each grid expressed with integer‑encoded colours. All training pairs share a *hidden* visual transformation. After observing these examples, the model receives **one further input grid** whose correct output is withheld. The objective is to **predict that withheld output** by inferring and applying the common transformation.

---

### Abstract

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Overcoming this constraint requires embeddings that move beyond purely temporal indexing. Instead, embeddings must inherently encode the intrinsic relationships between space and time. By generalizing two-dimensional Rotary Positional Embeddings (RoPE) to four-dimensional Minkowski Space-Time Embedding Rotors (MonSTERs), the transformer's artificial flattening requirement, which necessarily discards vital geometric and relational information[^3], is eliminated. These structural (not positional) embeddings remove the blinders and distortions that previously constrained transformer attention to a single axis.

Simply put, MonSTERs provide Transformers with a built-in ability to simultaneously perceive, reason about, and generalize across both spatial structures and temporal sequences—without sacrificing their established computational advantages.

---

### Using The Z Channel?

HOWEVER, as these structural encodings are defined as 4 dimensional, and the ARC grid is 2D with a temporal change, a naive implementation of token positions/indices within the structure would have zero modification to every 4th element of the data (the z index). This could end up not mattering as the transformers would eventually learn to live without the modification that comes from a z delta. However, could we modify the tokens by adding a z component equal to the magnitude of the distance from the origin? Or some other z function z = f(x, y)?

---

### Using The Z Channel!

Yes, you absolutely can and probably should modify the token embeddings to include a meaningful $z$-component derived from the $x$ and $y$ coordinates.

Your proposed approach, setting the $z$ component equal to the magnitude of the distance from the origin ($z = \sqrt{x^2 + y^2}$), is an excellent idea. This technique turns the unused dimension into a channel for injecting a useful **inductive bias** into the model.

---

##### Evaluating Your Approach: $z = \sqrt{x^2 + y^2}$

This is a clever way to encode radial information directly into the positional structure.

* **Pros:** ✨ For any ARC task involving radial symmetry, concentric patterns, or operations relative to the grid's center, this approach would be highly effective. The model wouldn't need to *learn* the concept of "distance from the center"; it would be an intrinsic property of each token's embedding. This could significantly speed up learning and improve generalization.

* **Cons:** 🤔 The main drawback is that you are hand-crafting a feature. For tasks where distance from the origin is irrelevant, this information could act as noise. It also makes the spatial dimensions non-orthogonal (since $z$ is now dependent on $x$ and $y$), which might have unintended consequences on the geometric properties of the MonSTER rotations, although this is likely a worthwhile trade-off.

---

##### Alternative Functions for the Z-Component

Since the goal is to embed useful spatial priors, you could tailor the function $z=f(x,y)$ to capture other types of geometric information common in ARC tasks. Here are a few alternatives:

* **Distance to Nearest Edge:** This would be useful for tasks involving frames, borders, or corner-based logic.
    $$z = \min(x, y, H-1-y, W-1-x)$$
    where $H$ and $W$ are the height and width of the grid, and coordinates are 0-indexed.

* **Checkerboard/Parity Pattern:** This encodes local relational positions and is great for tasks with diagonal or alternating patterns.
    $$z = (x + y) \pmod 2$$

* **Pixel Color/Value:** You could directly embed the token's initial state (its color) into its structural embedding. This is a very strong prior that fuses positional and feature information.
    $$z = \text{color}(x, y)$$

* **Quadrant ID:** A simpler categorical feature that could help the model reason about symmetries across axes.
    $$z = \text{quadrant}(x, y)$$
    For example, $z=0$ for top-left, $z=1$ for top-right, etc.

Ultimately, this strategy of feature-engineering the $z$-coordinate is a pragmatic solution to adapt a higher-dimensional embedding space to a lower-dimensional problem. The optimal function $f(x,y)$ would likely be task-dependent, suggesting that an ensemble of models trained with different $z$-functions could be a powerful approach.
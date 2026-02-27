Good question — that line *is* pretty dense.

Here’s what that LOOPE sentence is getting at, unpacked.

---

## 1\. “RoPE preserves relative positioning”

RoPE is a **relative** positional encoding:

- The attention score between tokens at positions *i* and *j* depends only on *(i − j)* (their offset), not on their absolute coordinates.
- In 2D/image form, that means it nicely encodes things like:
	- “this patch is 1 step to the right of that one”
	- “this patch is 3 steps up and 2 steps left from that one”

So LOOPE is acknowledging that:

> RoPE does a good job at encoding **pairwise relative positions** (local geometry).

That’s what they mean by “effectively preserve relative positioning.”

---

## 2\. “Rotationally invariant encodings”

This is a bit of shorthand. In the RoPE-ViT / 2D RoPE context:

- The positional information is represented as **complex rotations** (or 2D rotation matrices) applied to feature pairs.
- Because these are rotations in a shared space, some *global* transformations (like uniform shifts and certain basis rotations) don’t change the *relative* structure.

So “rotationally invariant” here is basically:

> the *form* of the encoding is built out of rotations, and what matters for attention is the **relative phase difference**, not the absolute angle itself.

That ties back to the “relative, not absolute” idea.

---

## 3\. “Fails to model spatial hierarchies for structured 2D reasoning”

This is the key criticism.

By “spatial hierarchies,” they mean **multi-scale / structured information** like:

- grouping patches into **larger regions** (super-patches, objects, parts)
- understanding relationships like:
	- this patch is in the **top-left quadrant** of the image
	- this patch is part of the **same object** as some distant patch
	- coarse layout: “sky over ground over road”

RoPE, as normally used:

- gives every patch a position via a fixed-frequency rotational map.
- treats all positions at a single, flat scale (one fixed grid / frequency spectrum).
- is excellent at *pairwise offsets* but **doesn’t explicitly encode**:
	- levels of scale (patch → region → whole image)
	- tree-like, nested structure
	- “this is locally close but globally far” vs “globally close but locally far,” etc.

LOOPE’s point is:

> RoPE gives you **good local/mid-range relative geometry**, but no built-in notion of **coarse vs fine spatial structure** — i.e., no hierarchical organization of space.

That’s important for “structured 2D reasoning” tasks where you care about layout at multiple levels (like scene understanding, complex object relationships, etc.), not just pairwise distances.

---

## 4\. How this contrasts with what they care about

LOOPE is all about:

- the **1D ordering of patches** that you feed into sinusoidal / Fourier-style PEs, and
- how that ordering affects the model’s ability to preserve:
	- monotonicity
	- translation properties
	- multi-scale structure when you flatten a 2D grid.

So from their perspective:

- Classical sinusoidal APE + a good 1D patch order can **encode some hierarchical patterns** (via frequency structure + ordering).
- RoPE, being purely relative and flat in its usual form, **doesn’t directly give you those hierarchical constraints** — it just gives you offset-aware rotations.

That’s why they summarize RoPE as:

> “great for relative positions; not designed to encode hierarchical, multi-scale 2D structure.”

---

If you want, next step could be to connect this with HARoPE and ComRoPE, which *do* try to add more structure/flexibility on top of RoPE (e.g., head-wise adaptation, higher-dimensional rotation groups).
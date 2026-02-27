Do you agree or disagree? By adopting the standard transformer architecture, researchers implicitly commit to a fixed-index, fixed-context positional scheme. For single-task vision models—such as classification, detection, or image generation—this restriction is largely inconsequential, as the task structure, dimensionality, and invariances are known in advance and can be amortized over scale. However, for unrestrained multi-task spatiotemporal learning, this architectural commitment becomes a dominant constraint. Expressing variable-dimensional structure, dynamic grouping, and cross-instance comparison within a fixed positional index requires increasingly long contexts, heavy padding, and complex attention approximations, driving up computational cost and model size. In this regime, positional encoding is not the primary limitation but a proxy for a deeper architectural friction: the effort required to simulate flexible spatiotemporal structure within a fixed-index sequence model. The absence of a canonical positional encoding in vision is therefore best understood not as an unresolved encoding choice, but as evidence that the standard transformer interface imposes an unnecessary burden on unrestrained spatial reasoning.

**Short answer:** I mostly agree with the thrust of the argument, with one important qualification about *where* the constraint actually bites and *why* transformers have remained competitive despite it.

---

### Where I agree

Your core claim—that standard transformers *implicitly commit to a fixed-index, fixed-context interface* —is accurate and often underappreciated.

1. **Fixed-index sequence as an architectural commitment**  
	Even with learned, relative, rotary, or implicit positional encodings, the transformer’s interface is still:
	- a flat sequence
	- with a fixed maximum length
	- addressed by index-based attention
	Positional encoding choices only modulate *how* indices are interpreted; they do not remove the fact that the model must *simulate* structure rather than *represent* it natively.
2. **Vision succeeds because structure can be amortized**  
	In single-task vision settings (classification, detection, generation):
	- spatial dimensionality is fixed
	- invariances are known (translation, locality, scale)
	- the task distribution is narrow
	Under these conditions, the cost of flattening space into tokens is largely paid once, at scale. Patch embeddings + attention work because the model never has to *re-negotiate* the meaning of space.
3. **Unrestrained spatiotemporal learning breaks the illusion**  
	As soon as you introduce:
	- variable numbers of entities
	- dynamic grouping and regrouping
	- cross-instance comparison
	- evolving spatial frames or reference systems
	the transformer is forced into:
	- longer contexts
	- padding and masking
	- quadratic or approximate attention
	- ad hoc routing mechanisms
	All of this is *simulation overhead* —not learning difficulty per se, but architectural friction.
4. **Positional encoding as a proxy problem**  
	I strongly agree with your reframing:
	> positional encoding is not the primary limitation but a proxy
	The persistent lack of a “canonical” positional encoding in vision is a symptom of the deeper mismatch between:
	- flexible spatial reasoning
	- and a fixed-index sequence abstraction
	If the interface were right, positional encoding would be boring.

---

### Where I would qualify the argument

The one place I’d soften the claim is the implication that this makes the standard transformer *fundamentally unsuitable* rather than *strategically overused*.

1. **Transformers are interface-stable, not structure-optimal**  
	The reason transformers dominate is not that they are ideal, but that they provide:
	- a uniform interface
	- massive tooling and hardware support
	- composability across modalities
	Researchers tolerate architectural friction because the ecosystem payoff is enormous.
2. **The model can internalize structure—but inefficiently**  
	Large transformers *do* learn latent spatial graphs, object files, and temporal abstractions—but only by:
	- burning parameters
	- expanding context
	- relying on scale to smooth over mismatch
	This is why performance continues to improve, but cost curves look ugly.
3. **The real constraint is not position, but addressability**  
	I’d sharpen your thesis slightly:
	> The bottleneck is not position, but **how computation is addressed**.
	Transformers can only attend by index.  
	What unrestrained spatiotemporal reasoning wants is:
	- object-addressable memory
	- dynamic relational binding
	- variable-arity interaction
	Position just happens to be where the mismatch becomes visible first.

---

### A concise reframing

If I were to compress your argument into a single line I agree with:

> The absence of a canonical positional encoding in vision is not an open problem to be solved, but a signal that the sequence-based transformer interface is doing unnecessary work to emulate spatial structure it does not natively support.

That’s a strong and, in my view, correct diagnosis.

---

### Bottom line

- ✅ You’re right that positional encoding debates often miss the deeper issue
- ✅ You’re right that fixed-index sequence models impose hidden costs on flexible spatial reasoning
- ⚠️ The transformer’s persistence is less about correctness and more about ecosystem gravity

If you’re heading toward architectures that treat *structure as first-class* rather than *emergent from attention*, your argument is already standing on solid ground.

If it *is* an open problem, I’d restate it like this:

> **Find a positional (or more generally, coordinate) interface that lets a model generalize across vision tasks and spatiotemporal regimes with minimal architectural hacks, minimal context growth, and stable inductive biases.**

That’s genuinely open— *not* because we lack clever encodings, but because we don’t yet have a universally satisfying **contract** between geometry and computation.

Here are the strongest “it’s open” arguments, plus what a real solution would have to look like.

---

## Why it can still be an open problem

### 1) “Canonical” might mean task- and scale-invariant

For language, discrete order gives you a natural default. For vision, a good default would need to behave well across:

- resolutions and aspect ratios
- crops/augmentations
- multi-view or multi-camera geometry
- video frame rates / variable sampling
- dense prediction and sparse object reasoning
- 2D ↔ 3D transitions (images to world)

A positional scheme that is *truly canonical* would make those transformations “cheap” for the model.

### 2) The issue might be representation, not architecture

Even within a fixed-index transformer, the *right* positional representation could:

- reduce the need for long contexts by enabling better factorization
- make grouping/relations easier to express (e.g., via relative geometry)
- provide better extrapolation to unseen sizes and shapes
- stabilize attention patterns across scales

So yes: it’s plausible that we haven’t yet found the encoding that makes the sequence interface “good enough” for most spatiotemporal reasoning.

### 3) Vision’s “no canonical encoding” might reflect missing priors

Vision has multiple plausible geometric priors, and none fully dominates:

- translation equivariance (CNN-like)
- scale-space structure
- perspective/projective geometry
- object-centric coordinates
- scene-centric / world coordinates
- temporal coherence

If the right answer is a *blend*, the open problem is: **how do we combine them without brittle heuristics?**

---

## What would count as “solving” it?

A solved positional/coordinate interface would have most of these properties:

1. **Resolution extrapolation**: train at one size, run at many, no degradation cliff
2. **Transform consistency**: predictable behavior under crop/resize/rotate/time-resample
3. **Compositional geometry**: easy to express “part-of”, “same object”, “corresponds to”
4. **Cross-instance comparability**: align entities across images/videos without padding hell
5. **Efficiency**: doesn’t require quadratic attention over ever-growing token grids
6. **Plug-and-play**: works across classification/detection/segmentation/video/3D-ish tasks

If someone produced an encoding + minimal interface tweaks that reliably gave you these, people *would* call it canonical.

---

## Two ways the “open problem” could resolve

### A) A better positional encoding within the transformer

This would mean an encoding that effectively gives you:

- robust relative geometry
- multi-scale coherence
- maybe some notion of “frame” or “chart” selection

…without requiring the model to learn all of that from scratch.

### B) A better interface that makes “position” less central

If the breakthrough is actually an interface change (object-addressable tokens, dynamic sparsity, hierarchical routing, etc.), then “positional encoding” becomes canonical almost automatically—because the *units of computation* are no longer tied to a rigid grid.

In that world, the open problem wasn’t “encoding,” but encoding was the visible tip of it.

---

## Reconciling your original claim with “it’s open”

Both can be true depending on definitions:

- If “canonical positional encoding” means **a single best way to inject coordinates into a flat token sequence**, then it may indeed be a proxy for deeper friction.
- If it means **a stable geometric contract that enables broad generalization and efficiency**, then yes—it’s plausibly an open problem, and a valuable one.

So the sharpest version is:

> The open problem is not “which sinusoid,” but “what geometric *contract* should the model’s computational primitives obey?”

---

If you want, I can propose a concrete taxonomy of what “canonical” could mean (equivariance-based, correspondence-based, world-geometry-based, object-centric, etc.) and what empirical tests would actually distinguish “solved” from “merely competitive.”
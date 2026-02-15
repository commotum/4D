Absolutely. Here is your **language modeling timeline**, rewritten to mirror the exact structural format of the RL section: same heading weights, same “Arbitrary constant” framing, same cadence, same citation style.

---

## The History of Language Modeling as the Removal of Arbitrary Constants

### 1) Yoshua Bengio (2003): removed the “words are unrelated symbols” constant

**Arbitrary constant:** words are discrete IDs with no geometry — “cat” and “dog” are no closer than “cat” and “entropy.”

**What became learnable:** the representational geometry of words.

The Neural Probabilistic Language Model introduced jointly learned distributed word embeddings. Word identity remained discrete at the token level, but its representation became a continuous vector optimized with the model. Meaning was no longer imposed by symbolic isolation; it was learned from co-occurrence structure.

This removed the assumption that lexical similarity must be hard-coded or externally defined.
Representation became trainable.

---

### 2) Tomas Mikolov (2010, RNNLM): removed the “fixed context window” constant

**Arbitrary constant:** language depends only on the previous *k* tokens (n-gram assumption).

**What became learnable:** how much history matters, via a recurrent state.

Recurrent neural language models replaced fixed windows with a learned state update rule. In principle, the model could preserve arbitrarily long dependencies by deciding what information to retain.

The hard window boundary disappeared.
Memory over time became learned rather than specified.

A new constraint emerged — forced compression into a single hidden state — but the explicit horizon limit was gone.

---

### 3) Ashish Vaswani (2017, Transformer): removed the “recurrent bottleneck” constant

**Arbitrary constant:** all past information must be compressed into one evolving vector state.

**What became learnable:** content-addressable access to prior tokens via attention.

The Transformer replaced recurrence with self-attention, allowing direct access to any prior token in the context window. Information no longer had to survive sequential compression; it could be retrieved dynamically based on learned queries.

The constraint removed was architectural: memory organization was no longer forced into a single state bottleneck.

Memory access itself became learned.

---

### 4) Alec Radford (2019, GPT-2): removed the “task-specific supervision” constant

**Arbitrary constant:** each capability requires its own dataset, objective, and training pipeline.

**What became learnable:** general linguistic competence from a single next-token objective.

GPT-2 demonstrated that large-scale next-token prediction on diverse web text yields models that exhibit task behavior through prompting rather than retraining. Translation, summarization, question answering — these were no longer separate training problems.

The constant removed was predefined task boundaries.

Capability became an emergent property of scale and distributional learning rather than explicit supervision.

---

### 5) Richard Zhang (2025–2026, Recursive Language Models): removed the “everything must fit in one attention window” constant

**Arbitrary constant:** all usable information must be present simultaneously inside a fixed attention window.

**What became learnable:** what context to retrieve, when, and how to recursively incorporate it.

Recursive Language Models treat the prompt as an inspectable environment. Instead of consuming all context in a single attention pass, the model can decompose, retrieve, and selectively reintroduce information into working memory.

The fixed boundary of uniform context exposure is removed.
Context selection becomes model-directed rather than window-determined.

Memory becomes dynamic and recursive.

---

## The pattern in language modeling

Across these shifts, progress repeatedly comes from stopping the human from hard-coding:

* **what similarity means** (symbolic IDs → learned embeddings)
* **how far context extends** (fixed window → recurrent state)
* **how memory must be structured** (single state → attention-based access)
* **what tasks exist** (task-specific objectives → general next-token pretraining)
* **what context is usable** (fixed window → recursive retrieval)

The objective — predict the next token — remains constant.

What changes, at each inflection point, is who decides what matters.

Each major advance removes a modeling constraint that once seemed necessary.
Each time, a fixed architectural constant becomes a learned statistical object.

That repeated removal of arbitrary constants may be the real engine of progress in language modeling.

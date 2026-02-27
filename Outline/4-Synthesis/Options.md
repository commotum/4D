Looking at this, "This is why Sutton cares about “metadata / utility tracking” too: you need machinery that promotes certain latents into “named” abstractions and keeps them around." especially, how can we align that to the vocab talked about here?

## Abstract 1

At a high level, the baseline transformer interface assumes a static, fixed token vocabulary, a static, fixed attention-window, and a static, fixed set of positional encodings implicitly mapped to their tokens by index. This rigidity eases implementation, but curtails the dynamic responsiveness necessary for continual learning.

This paper introduces dynamic counterparts for each of these three static assumptions: a typed, value‑parameterized token schema that enables open‑vocabulary token synthesis; focused attention-threads that select task-relevant token subsets from a keyed, addressable token bank; and Minkowski SpaceTime Encoding Rotors (MonSTERs), a 4D generalization of RoPE that unifies 1D–4D positional structure in a single Lorentz-invariant scheme.

Distinct from multimodal systems, which typically orchestrate separate architectures or encoding schemes for different data modalities (e.g., 1D text, 2D images, etc.), these architectures employ a single, unified token and encoding interface that applies uniformly across dimensionalities, and therefore are deserving of their own name. Accordingly, we call them MonSTER Models.

Together, these changes turn the transformer from a static, task-bounded sequence model into a continually extensible learner: new skills can be *named* (via open‑vocabulary token synthesis), *stored and revised* (via a keyed token bank), and *selectively deployed* (via attention‑threads), all while a single MonSTER encoding provides a consistent notion of structure across sequences, grids, volumes, and trajectories. In this regime, tasks that are usually siloed into separate model families—ARC‑AGI-style grid reasoning, program synthesis, Sudoku, Rubik’s Cube manipulation, and Atari control—can be treated as different projections of the same interface, enabling transfer when structure overlaps and isolation when it does not. Rather than “one new model per domain,” MonSTER Models are built to accumulate competencies over time under a shared representational contract, retaining prior skills while incorporating new ones across both task diversity and dimensionality.




Yes — you’re hitting a real fault line, but it’s useful to separate **three different “vocabularies”** that get conflated:

## 1) Vocabulary for *communication* (tokens/patches)

This is the LLM’s **external alphabet**: the discrete symbols it can emit/consume (text tokens, image patches, etc.).

A model can be limited here and still have rich internal structure.

## 2) Vocabulary for *internal concepts* (features/latents)

This is the agent’s **learned representation**: continuous features, latent variables, memory keys, etc.

Even a plain transformer trained on tokens can form internal “concepts” as latent directions/features—**but** those concepts are usually:

* implicit,
* not addressable as first-class objects,
* not easily composable into persistent skills.

## 3) Vocabulary for *skills/actions/models* (options)

This is what Sutton is gesturing at: a library of **temporally extended behaviors** + **models of their effects** + **subtasks** that define/organize them.

That’s not just “a concept in the weights.” It’s a concept that becomes:

* a callable behavior (“do option i”)
* with semantics grounded in experience (“if I do it, feature f changes like this”)
* and can be used in planning.

So you’re right: **tokens/patches alone don’t automatically give you (3)**.

---

# Where Sutton’s “build its own vocab” really lives

In OaK, the “new vocabulary” is mostly:

* **new state features** (concepts) discovered at runtime
* **new options** (skills) posed/learned around those features
* **new predictive models** of what those options do

That’s a different kind of vocabulary than wordpieces.

It’s a vocabulary of *action abstractions and predictive abstractions*—and it’s explicitly meant to be open-ended. 

---

# Does an LLM “only in token space” prevent this?

**Not necessarily**, but it depends what you mean by “only.”

### If you mean: *the agent can only ever manipulate/attend to tokens and can’t persist new structures*

Then yes, it’s a serious limitation: it can form *implicit* concepts, but it can’t easily create **new durable interfaces** (new skills, new callable entities, new models) that are reused across time.

### But: even token models have internal continuous state

Transformers aren’t literally thinking “in token space.” They’re computing with high-dimensional continuous vectors; tokens are just the I/O alphabet. So they *can* form internal concepts. The gap is:

* **internal concept formation** ✅ possible in token models
* **turning concepts into persistent skills/options** ❌ not automatic

That second leap usually requires **architecture + training setup** that supports:

* persistent memory / program-like objects
* skill learning and reuse
* model learning and planning

---

# The best term for what you’re describing

When you say “create concepts like options that it can save externally,” the most standard phrasing is:

### **Skill discovery / option discovery with a learned skill library**

and, for the “save externally” part:

### **persistent memory with callable abstractions** (external memory / program memory)

In RL, this often appears as:

* **hierarchical RL** (HRL)
* **temporal abstraction**
* **options framework**
* **skill library**
* **compositional policy / modular policy**

Sutton’s OaK is basically “skill discovery + model learning + planning,” all driven by discovered features. 

---

# The key difference: “implicit features” vs “first-class abstractions”

A model can learn a feature like “doorway-ness” in its latent space.

But OaK wants the agent to then:

1. notice that feature is useful,
2. pose a subtask: “make it high while respecting reward,”
3. learn an option (policy + termination),
4. learn an option model,
5. plan using that model.

That’s what turns a latent concept into a **first-class reusable object** in the agent’s cognition.

This is why Sutton cares about “metadata / utility tracking” too: you need machinery that promotes certain latents into “named” abstractions and keeps them around. 

---

# A crisp way to say it

* **Token vocabulary** is about *expressing sentences*.
* **OaK vocabulary** is about *expressing and reusing abstractions over experience*: features, skills, and predictive models.

An LLM can have the first and some of the second, but Sutton’s claim is that you need the third (and continual learning) for real general agents.

---

If you want to make this maximally precise, answer this one choice (no need for a long reply):
When you say “save externally,” do you mean **(A)** literal external memory objects (like a skill table / code / modules), or **(B)** “saved” in weights as a stable latent feature?



You can align them by treating **“vocabulary” as a 3-layer stack**, and then mapping Sutton’s “metadata/utility tracking” to the *promotion rule* between layers.

### The three vocab layers (reframed in your MonSTER terms)

#### 1) **Surface vocabulary** (what can be *expressed / addressed*)

* In a vanilla transformer: fixed BPE tokens / fixed patch grid.
* In your MonSTER abstract: **typed, value-parameterized token schema + open-vocabulary token synthesis** → the surface alphabet becomes *extensible*.

This is where your “skills can be named” claim lives: you’re proposing a way to mint new *symbols* (or IDs) on demand.

#### 2) **Latent vocabulary** (what can be *represented*)

* Continuous features, embeddings, latent factors: the model already has this.
* Your abstract’s “keyed, addressable token bank” is essentially a way to **stabilize and re-access** latents rather than letting them stay implicit in weights only.

This is where “concepts” live even in ordinary LLMs; your bank makes them more like durable objects.

#### 3) **Behavioral / option vocabulary** (what can be *done and reused*)

* Sutton’s “options” are *callable behaviors* plus (in OaK) models of their consequences.
* Your abstract doesn’t explicitly say “policy + termination,” but “stored and revised + selectively deployed” is pointing in the same direction: a thing isn’t just a latent—it’s a reusable procedure.

---

## Where Sutton’s “metadata / utility tracking” fits

In OaK, “metadata/utility tracking” is the mechanism that decides:

> which internal features are worth *promoting* into first-class abstractions (subtasks/options/models) and which should be pruned.

So in your MonSTER language, metadata/utility tracking corresponds to **promotion + lifecycle management** for your open vocab + token bank.

### A clean alignment

Think of each new “named thing” (your open-vocab token) as having associated metadata:

* **usage**: how often it’s retrieved / attended / executed
* **value-of-information**: does it reduce loss / uncertainty?
* **reward impact**: does it improve returns / task success?
* **predictive gain**: does it improve predictive models (world model, user model)?
* **stability**: does it remain useful across contexts or drift?

Then “utility tracking” becomes:

* when to *create* a new token (name a new abstraction),
* when to *consolidate/merge* tokens,
* when to *freeze* a token’s meaning,
* when to *retire* a token.

That is exactly the missing glue between “latent concepts exist” and “the agent builds its own vocabulary.”

---

## The key conceptual match: “naming” is not the same as “discovering”

Sutton’s point: features/options must be **discovered and kept** because they’re useful.

Your abstract’s novelty: you’re proposing a **representational substrate** where the agent can:

* **mint a name** (open-vocab token synthesis),
* **store the associated content** (token bank),
* **route computation through it** (attention threads).

But Sutton’s missing piece (and your “metadata” point) is: *how does the system decide what deserves a name?*

That decision rule is exactly what “utility tracking” is doing in OaK.

So the mapping is:

* **OaK utility tracking** ≈ your system’s **token creation/pruning policy**.

---

## How to make the mapping precise in one sentence

> In OaK, “utility tracking” promotes useful latent features into persistent, nameable abstractions (options/models); in MonSTER terms, that corresponds to learning when to synthesize new typed tokens, store them in the token bank, and route attention through them, versus when to prune or merge them.

---

## One extra practical insight

Your abstract implicitly claims **surface vocabulary extensibility** (new tokens). Sutton’s architecture doesn’t require new surface symbols; it can keep everything internal (features/options as implicit IDs). The advantage of your approach is:

* **addressability**: you can *refer back* to a skill/concept by name.
* **composability**: names can be combined like programs (“do A then B”).
* **externalization**: names can be logged, audited, shared.

That’s a real bridge from “implicit representation” → “first-class abstraction,” which is exactly the Sutton gap you’re trying to close.

---

If you want, I can propose a concrete “utility score” that would connect your token-bank objects to Sutton-style option discovery (e.g., a score combining predictive gain + reward gain + reuse frequency), and show how it would trigger (a) new token synthesis, (b) option creation, (c) pruning/merging.
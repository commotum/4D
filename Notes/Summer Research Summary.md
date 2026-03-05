# **From MonSTERs to Domain-Agnosticism**

## **Table of Contents**

1. **Summer Context: From MonSTER Theory to Empirical Revision**  
    1.1. Original Assumption Entering the Summer  
    1.2. Immediate Experimental Goal: Integrating MonSTER into HRM  
    1.3. What Actually Changed

2. **Settling on the Triadic Form of MonSTER**  
    2.1. From Pairwise Relative Formulation to Practical Absolute Encoding  
    2.2. The Structure of the Triad  
    2.3. Preserving Absolute–Relative Fusion in the Minkowski Dot Product  
    2.4. Benefits for Performance, Efficiency, and Implementation

3. **The Empirical Surprise: Positional Encoding Is Not the Bottleneck in Single-Task Transformers**  
    3.1. Integrating MonSTER into HRM and the Initial Result  
    3.2. Why the Gains Were Smaller Than Expected  
    3.3. Single-Task Models Can Wash Out Better Inductive Priors  
    3.4. Why the Literature Tolerates So Many Positional Encoding Variants

4. **Revised Hypothesis: Where MonSTERs Should Actually Matter**  
    4.1. From Single-Task Evaluation to Multitask Evaluation  
    4.2. One Geometry for 1D, 2D+t, and 3D+t  
    4.3. Why Ad Hoc Multi-Encoder Systems Are the Wrong Long-Term Answer  
    4.4. The Scaling Constraints That Now Define the Problem

5. **A Second Realization: Static Context Windows Miss the Point of ARC-Style Meta-Learning**  
    5.1. Why HRM Still Misses the Few-Shot Structure of ARC-AGI  
    5.2. Token Explosion and Context Pollution  
    5.3. Dynamic Attention Threads as Working Memory  
    5.4. Implications for Future Reasoning Architectures

6. **Rethinking Tokens and Learning Objectives**  
    6.1. The RGB Vocabulary Problem  
    6.2. Quaternion Type-Value Tokens and On-the-Fly Token Synthesis  
    6.3. Hierarchical Compression, Concept Tokens, and Relay Learning  
    6.4. Moving Beyond Next-Token Prediction

7. **Updated Research Program, Paper Roadmap, and Long-Term Vision**  
    7.1. Spatial Reasoning MonSTERs  
    7.2. Dynamic Attention Threads for Transformer Working Memory  
    7.3. Relay Learning  
    7.4. MonSTER Models as Domain-Agnostic Multitask Learners  
    7.5. Immediate Next Steps

# **1\. Summer Context: From MonSTER Theory to Empirical Revision**

This summer marked an important transition in the research. Going into it, I believed the central missing ingredient for ARC-AGI-style reasoning was a principled positional encoding capable of representing true spatial and temporal structure. The original MonSTER formulation was developed to address precisely that problem: to generalize RoPE from 1D sequence positions to 4D spacetime positions, and thereby replace flatten-and-index hacks with a geometry-aware encoding grounded in Minkowski structure.

By the end of the summer, that core idea remained intact, but my understanding of where it fits in the stack changed substantially. The work did not invalidate MonSTER. Rather, it clarified both its proper implementation and its proper scope. The biggest result was not that MonSTER failed; it was that the surrounding architecture and evaluation regime were masking where MonSTER’s advantages should be expected to appear.

## **1.1. Original Assumption Entering the Summer**

My original assumption was straightforward: if current transformer models struggle with spatial reasoning because they flatten multidimensional data into 1D sequences, then replacing weak positional encodings with a principled spacetime encoding should produce large gains. In particular, I expected that injecting MonSTER into a competitive grid-reasoning architecture would reveal a significant performance improvement, especially on tasks where row-major indexing, patch flattening, and axial shortcuts distort the underlying geometry.

This assumption was reasonable. ARC-AGI tasks are fundamentally about spatial and relational structure. If the model’s positional system cannot represent that structure natively, then it seems natural to expect the architecture to pay a steep penalty.

## **1.2. Immediate Experimental Goal: Integrating MonSTER into HRM**

The immediate practical goal for the summer was therefore to stop treating MonSTER as a purely theoretical construct and instead test it in a real system. The Hierarchical Reasoning Model (HRM) provided a useful substrate for this. HRM is a strong recurrent transformer baseline for grid-based reasoning, and it already operates in a regime where spatial reasoning matters. If MonSTER were going to deliver the kind of gain I anticipated, then this seemed like an appropriate place to demonstrate it.

This shifted the research from “Can I define a 4D generalization of RoPE?” to “Can I implement MonSTER in a form that is efficient, stable, and actually useful inside a modern attention stack?”

## **1.3. What Actually Changed**

Three major things changed over the course of the summer.

First, I settled on the **triadic form** of MonSTER as the correct practical implementation. The original pairwise-relative formulation was mathematically clean, but the triadic absolute-position implementation preserved the key RoPE-like identity while being vastly better for vectorization, caching, and integration with existing attention code.

Second, the experiments forced an uncomfortable but important conclusion: in **single-task transformer models**, the positional encoding mechanism appears to be much less important than I originally believed. Even when the inductive prior is theoretically poor, a sufficiently expressive model trained on one narrow task can often learn around it.

Third, this led to a revised hypothesis. The real value of MonSTER is unlikely to reveal itself in isolated single-task benchmarks. Its advantages should become visible when a model must handle **multiple data modalities, multiple dimensionalities, and multiple kinds of reasoning** using one shared architecture and one shared positional system. In other words, MonSTER seems less like a “single-task performance trick” and more like a candidate foundation for domain-agnostic multitask learning.

# **2\. Settling on the Triadic Form of MonSTER**

One of the most important technical developments this summer was deciding what MonSTER should actually look like in practice. The original writeup focused on the clean theoretical story: for each query-key pair, compute a relative spacetime displacement and derive a Lorentz transformation from that delta. That framing was conceptually useful, but it was not the best implementation target.

---

## **2.1. From Pairwise Relative Formulation to Practical Absolute Encoding**

The original MonSTER presentation emphasized a relative construction. For a query at position $P_q$ and a key at position $P_k$, one computes the displacement

$$
\Delta P = P_k - P_q = (\Delta t, \Delta x, \Delta y, \Delta z),
$$

and then uses that delta to build an effective Lorentz transformation for the relevant embedding block. This mirrors the intuitive story that attention should depend on the relative relationship between two events in spacetime.

The problem is that implementing MonSTER this way inside attention is expensive and awkward. A direct relative construction suggests rebuilding positional structure for every query-key pair, which is precisely the kind of pairwise overhead that makes an otherwise elegant encoding impractical.

The triadic formulation resolves this by returning to the deeper insight behind RoPE: one does **not** need to explicitly compute a fresh relative transform for every pair if absolute encodings are chosen so that the inner product collapses to a relative dependence automatically.

---

## **2.2. The Structure of the Triad**

In the triadic form, the embedding is divided into frequency buckets, and each frequency bucket occupies 12 dimensions arranged as three 4D spacetime blocks:

$$
[X_4 \mid Y_4 \mid Z_4].
$$

Each 4D block is associated with one spatial axis. Instead of constructing a full arbitrary-axis Lorentz transform from scratch for every token pair, the encoding applies a structured combination of:

* an axis-aligned **boost** involving the time component and the spatial component associated with that block, and  
* a 2D **rotation** in the spatial plane orthogonal to that axis.

This gives a triad of coupled spacetime transforms per frequency bucket. The implementation then precomputes scalar tables—$\cosh$, $\sinh$, $\cos$, and $\sin$—from each token’s **absolute** position. In other words, MonSTER becomes a cacheable absolute positional system rather than a pairwise reconstruction system.

In its current HRM integration, this should be understood as a restricted but faithful first deployment rather than the final form of the full theory. The practical implementation emphasizes the triadic spatial structure most relevant to grid reasoning, while preserving the broader 4D spacetime logic that motivated MonSTER in the first place.

This is not a retreat from the original theory. It is the practical factorization of that theory into a form that can be broadcast efficiently over large tensors and reused across attention steps.

---

## **2.3. Preserving Absolute–Relative Fusion in the Minkowski Dot Product**

The most important thing the triadic form preserves is the original goal: **absolute position encodings whose effect in attention depends only on relative position**.

In Euclidean RoPE, this happens because the dot product of two separately rotated vectors depends only on the difference in their rotation angles. In MonSTER, the same principle survives in Minkowski space. If $L(s)$ is the Lorentz-style transform associated with absolute spacetime position $s$, and if the attention score uses the Minkowski metric $\eta = \mathrm{diag}(1,-1,-1,-1)$, then the desired identity is

$$
\begin{aligned}
\langle L(s_q) q,\; L(s_k) k \rangle_{\eta}
&=
\langle q,\; L(s_k - s_q) k \rangle_{\eta}.
\end{aligned}
$$

This is the central reason the triad implementation is correct. It means that the model can encode each token once using its **absolute** position, yet the resulting attention score depends only on the **relative** difference between tokens. The fusion of absolute and relative information is retained, but the cost of explicitly recomputing $\Delta P$ for every pair is eliminated.

This was one of the major conceptual clarifications of the summer. The original relative story is still the right intuition, but the triadic implementation is the right engineering realization.

---

## **2.4. Benefits for Performance, Efficiency, and Implementation**

Settling on the triad form produced several concrete advantages.

* **No pairwise transform reconstruction**  
  The model does not need to build a fresh Lorentz transform for every query-key pair.

* **Cacheable scalar tables**  
  Absolute positions can be converted into $\cosh/\sinh/\cos/\sin$ tables once and then reused.

* **Vectorized implementation**  
  The transforms can be applied in closed form using broadcasted tensor operations rather than full matrix construction.

* **Compatibility with existing attention kernels**  
  The Minkowski inner product can be implemented with simple signed masking on the key side, allowing MonSTER to plug into standard fast attention code rather than requiring a bespoke kernel.

* **Faithfulness to the original objective**  
  The encoding still provides absolute-relative fusion, multiscale frequency structure, and a principled spacetime inductive bias.

In hindsight, this triadic form was one of the clearest technical wins of the summer. Even before broader architectural questions are solved, it established that MonSTER can be implemented in a form that is both theoretically meaningful and practically usable.

# **3\. The Empirical Surprise: Positional Encoding Is Not the Bottleneck in Single-Task Transformers**

If the summer had stopped at the triad implementation, the story would have been relatively straightforward: theory refined into implementation. But the experiments produced a more surprising and more important result.

---

## **3.1. Integrating MonSTER into HRM and the Initial Result**

After integrating MonSTER into HRM, the results improved, but only modestly. The gains were on the order of roughly **3–7%**, depending on the training run and configuration. That is enough to suggest that MonSTER is not useless; there is likely some real signal there. At the same time, the improvement is small enough that it can easily be confounded by run-to-run variance, hyperparameter choices, and the normal noise floor of this class of experiments. It is not the dramatic jump I originally expected from replacing a theoretically flawed spatial prior with a principled one.

This result was confusing at first. If row-major indexing creates anisotropy, if flattened sequence order distorts true spatial relations, and if MonSTER fixes those issues, why are the gains not much larger?

---

## **3.2. Why the Gains Were Smaller Than Expected**

The answer that emerged is that the benchmark regime matters just as much as the inductive prior. In a single-task setting, the model is not trying to learn a universal geometry. It is trying to solve one task family under one distribution. Under those conditions, a sufficiently expressive transformer can often compensate for poor positional structure.

This does not mean the geometry is irrelevant in principle. It means that on a narrow enough task, the model can learn to treat even a flawed positional encoding as a workable addressing system. The recurrence, depth, and sheer flexibility of the transformer are enough to adapt to the quirks of the chosen encoding.

In that setting, MonSTER may still help a bit, but the model is already capable of learning around the defect it is supposed to fix.

---

## **3.3. Single-Task Models Can Wash Out Better Inductive Priors**

This became one of the most important conceptual lessons of the summer:

A **single-task transformer** can often overcome not only a weak positional prior, but can effectively wash out the advantage of a stronger one.

If the only goal is to solve one dataset, then the model can absorb the geometry of the encoder as yet another texture in its input space. It can learn that certain sequence offsets correspond to rows, others to columns, and still others to repeated motifs in the training distribution. The positional system no longer needs to be “right” in a domain-agnostic sense. It merely needs to be stable enough for the model to memorize how to use it.

This explains why my MonSTER gains inside HRM were real but not decisive. HRM is still a transformer with enough expressive power to partially repair a bad prior on its own.

---

## **3.4. Why the Literature Tolerates So Many Positional Encoding Variants**

This insight also helps explain something that had bothered me for a while: why the literature on spatial reasoning and vision transformers tolerates such a wide variety of positional encodings.

Researchers use learned embeddings, sinusoidal embeddings, 1D RoPE, axial RoPE, 2D RoPE, mixed schemes, patch-local encodings, and many hybrids. If positional geometry were the dominant bottleneck in all settings, one would expect a much more decisive convergence. But that is not what we observe.

The summer experiments suggest why. On **single-domain** benchmarks, many of these schemes are “good enough” because the model only needs to adapt to one geometric regime. The transformer can learn the quirks of that regime and compensate for inconsistencies or anisotropies internally.

This does **not** imply that all positional encodings are truly equivalent. It implies that many benchmarks are too narrow to expose the difference. The importance of a principled encoding is therefore not disproven; it is simply displaced to a harder setting.

# **4\. Revised Hypothesis: Where MonSTERs Should Actually Matter**

The natural consequence of the summer results is a revised hypothesis about what MonSTER is for and how it should be evaluated.

---

## **4.1. From Single-Task Evaluation to Multitask Evaluation**

If single-task models can learn around poor positional priors, then evaluating MonSTER inside one narrow benchmark is not the right test. The real stress test is a model that must operate across **different kinds of data**, **different dimensionalities**, and **different reasoning regimes** without being given a separate handcrafted encoder for each one.

That is the setting where a domain-agnostic positional system should matter. A single-task model can memorize one flawed map. A multitask model cannot as easily memorize a different flawed map for every domain if all of them must coexist in one shared representation space.

This is now my central hypothesis:

**MonSTER’s advantages will become most visible in multitask, multidomain models that must share one encoding mechanism across heterogeneous data types.**

---

## **4.2. One Geometry for 1D, 2D+t, and 3D+t**

The target is no longer just ARC-AGI or grid puzzles in isolation. The target is a model that can natively handle:

* **1D temporal/sequential data**  
  language, symbolic reasoning, mathematics, program traces

* **2D+t data**  
  ARC-AGI, Sudoku, Crosswords, Bongard problems, other grid and board-style tasks

* **3D+t data**  
  navigation problems, folding and hole-punching tasks, Rubik’s cube-like state spaces, spatial planning environments

In that setting, the positional encoding cannot be a domain-specific afterthought. It must serve as a common geometric substrate. This is exactly the kind of setting where I expect MonSTER to stop looking like a minor improvement and start looking like necessary infrastructure.

---

## **4.3. Why Ad Hoc Multi-Encoder Systems Are the Wrong Long-Term Answer**

A common way to handle diverse domains is to assemble an ad hoc stack: one encoder for text, one for images, one for video, one for grids, one for 3D worlds, and then build a system that glues them together. This can work in practice, but it is not the kind of unification I am interested in.

My interest is in a model architecture that does **not** need to be told, case by case, which positional system is appropriate for which modality. The long-term goal is a single model with a single general geometric language for tokens. MonSTER is attractive precisely because it points in that direction.

The broader claim, then, is not merely “MonSTER beats RoPE on ARC.” It is closer to: **future generalist reasoning models will need one coherent positional framework rather than a patchwork of incompatible encoders.**

---

## **4.4. The Scaling Constraints That Now Define the Problem**

Once the problem is framed this way, a new set of challenges becomes obvious.

As dimensionality increases, several things scale badly:

* the number of positional states that may need to be represented or cached  
* the number of tokens required to describe an environment faithfully  
* the computational burden of attending over all of those tokens  
* the tendency for irrelevant spatial detail to pollute the context window

This means the research problem is no longer just “find the right positional encoding.” It becomes “find the right positional encoding **and** the right memory system **and** the right tokenization strategy.”

That was another major lesson of the summer: the positional problem, the context problem, and the token problem are deeply entangled.

# **5\. A Second Realization: Static Context Windows Miss the Point of ARC-Style Meta-Learning**

While thinking through why MonSTER did not unlock dramatic gains inside HRM, another architectural issue became impossible to ignore. Even if the positional encoding is correct, the **static context window** remains a poor fit for ARC-style reasoning.

---

## **5.1. Why HRM Still Misses the Few-Shot Structure of ARC-AGI**

ARC-AGI is not merely a sequence-to-sequence problem over one grid. It is a **few-shot meta-learning problem**. Each task provides several correlated input-output examples, and the solver must infer the hidden transformation rule that links them, then apply that rule to a new test input.

HRM is clever, and its recurrent setup plus augmentation tricks are genuinely strong. But the basic treatment of the task still misses something central: it does not fully exploit the episode structure of ARC. The architecture is still fundamentally closer to “map this input grid to this output grid” than to “read a small set of demonstrations, compare them, induce the shared rule, and then solve a new case.”

That matters. It means that even a better positional system may not show its full value if the surrounding model is not actually performing the kind of cross-example abstraction the benchmark is designed to test.

---

## **5.2. Token Explosion and Context Pollution**

Trying to solve this by simply concatenating more examples into a larger context window introduces its own problems.

As more grids, scenes, or environment states are added, token counts explode. Padding grows. Irrelevant local detail competes with task-relevant structure. The model’s attention is forced to spend capacity on many tokens that do not matter to the current reasoning step.

This is especially severe once one starts thinking beyond ARC. In higher-dimensional domains, the number of raw tokens can become enormous. Even if MonSTER supplies the correct geometry, the model still faces the practical problem of being drowned in its own context.

This is what I mean by **context pollution**: the context window contains the right information in principle, but too much of it is present at once in the wrong granularity.

---

## **5.3. Dynamic Attention Threads as Working Memory**

This led me to a second major idea emerging from the summer: the model should not passively attend over one fixed context window. It should maintain something more like a **working memory**.

The rough idea is to let a policy select what to inspect and compare over time. Instead of every token always competing inside one giant static pool, the system would:

1. keep a compact bank of minimally represented tokens as the environment,  
2. select query tokens of interest,  
3. select the relevant keys or neighborhoods to compare against,  
4. write intermediate observations into a scratchpad or notepad, and  
5. iterate.

This creates **dynamic attention threads** rather than one static attention field. The model can zoom in, zoom out, compare one example against another, and explicitly accumulate intermediate findings.

For ARC-style tasks, that looks much closer to the procedure a competent human solver actually uses: inspect one example, inspect another, write down invariants, compare before and after states, propose a rule, test it, revise it.

---

## **5.4. Implications for Future Reasoning Architectures**

The architectural implication is important. A truly strong reasoning model may need more than just a better embedding or a deeper transformer. It may need a control policy over what it looks at and when, plus a persistent intermediate memory for partial abstractions.

In this picture, MonSTER remains essential, because the model still needs a coherent geometric substrate over which it can reason. But MonSTER becomes one component in a larger system that includes:

* a positional geometry,
* a memory organization,
* an inspection policy,
* and an iterative reasoning loop.

That broader architecture is where I now think the real ARC-AGI opportunity lies.

# **6\. Rethinking Tokens and Learning Objectives**

The more I pushed toward domain-agnostic reasoning, the more another bottleneck became clear: the standard token vocabulary paradigm itself looks increasingly unnatural for visual and structured world-state data.

---

## **6.1. The RGB Vocabulary Problem**

If one treats raw RGB values as ordinary discrete tokens, the vocabulary becomes enormous. A full RGB space contains more than 16 million possible values. That is a very different regime from the subword vocabularies used by modern language models.

This creates several problems:

* a massive softmax space,  
* sparse or nonexistent updates for many rare values,  
* poor alignment between discrete token IDs and the underlying continuous structure of color space.

For visual reasoning systems, this suggests that the usual “large embedding table plus softmax over token IDs” paradigm may be the wrong abstraction.

---

## **6.2. Quaternion Type-Value Tokens and On-the-Fly Token Synthesis**

One idea that emerged from the summer was to represent values such as RGB not as arbitrary discrete IDs, but as structured typed values. Quaternions are especially interesting here because they naturally package multi-channel information and have algebraic properties that make them attractive for compact color or spatial representations.

The idea is not simply “replace real vectors with quaternions everywhere.” The more interesting possibility is to build **type-value tokens** in which a token type supplies the learned projection behavior and the value is synthesized on the fly. A COLOR token, for example, could use one learned up-projection mechanism, while the specific RGB quaternion value provides the instance-level content.

In that setup, the model does not need a separate learned embedding row for every possible RGB value. Instead, it learns how a **type** should lift a structured **value** into the model’s token space. This points toward a more general notion of token synthesis, where the model generates or interprets token values rather than merely indexing into a fixed lookup table.

---

## **6.3. Hierarchical Compression, Concept Tokens, and Relay Learning**

Another connected idea is that a general reasoning system should not be limited to a fixed human-defined vocabulary at all. It should be able to **compress** collections of tokens into higher-level learned tokens when doing so is useful.

The intuition is easy to state. In text, a model could compress words into sentence tokens, sentences into paragraph tokens, paragraphs into chapter tokens, and so on, all while preserving the ability to reconstruct the original data. In visual or symbolic domains, it could compress local structures into object tokens, object relations into scene tokens, and recurring reasoning traces into reusable concept tokens.

This line of thought led me to what I have been calling **Relay Learning**. The central idea is that intelligence is not best framed as “remove all human knowledge” or “hand-design everything.” Rather, humans and machines should iteratively hand off abstractions to one another. The model learns compressions and reusable concepts; humans inspect, refine, and formalize them; the system then incorporates those improved abstractions and continues learning.

In that picture, the goal is to reduce the translation tax between human concepts, machine representations, and environment states as much as possible.

---

## **6.4. Moving Beyond Next-Token Prediction**

Once the token vocabulary is no longer fixed and the model is allowed to build intermediate abstractions, the learning objective also needs to broaden.

Next-token prediction is an extraordinarily successful objective for language, but it is not obviously the best universal objective for reasoning. Many tasks admit multiple equally valid next moves. Sudoku is an easy example: from a given board state, there may be several logically correct fills that preserve the solution path. In state-based reasoning tasks more broadly, “the one true next token” is often the wrong target.

This suggests a more general training mix involving:

* **reconstruction losses** for hierarchical compression,
* **next-state prediction** for structured reasoning domains,
* **set-valued or rule-consistent rewards** for tasks with multiple valid moves,
* **multi-step objectives** such as next-sentence or next-paragraph prediction,
* and potentially **temporal-difference-style** rewards when reasoning is genuinely sequential and policy-driven.

This is another place where the summer changed my thinking. The project is no longer just about building a better transformer encoder. It is increasingly about designing a model that can create, compress, retrieve, and act over structured abstractions.

# **7\. Updated Research Program, Paper Roadmap, and Long-Term Vision**

By the end of the summer, the work had become much broader than the original expectation. The good news is that the pieces fit together cleanly enough that they can be separated into a coherent research program.

---

## **7.1. Spatial Reasoning MonSTERs**

The first paper is still the clearest starting point. This paper would focus on:

* the weaknesses of current spatial encoding schemes,
* the need for a domain-agnostic positional system,
* the theory of MonSTER as a Minkowski-space generalization of RoPE,
* the triadic implementation as the efficient practical realization,
* and empirical evidence that the real benefits of principled geometry emerge under cross-domain rather than purely single-task evaluation.

This paper is the foundation because it establishes the geometric language.

---

## **7.2. Dynamic Attention Threads for Transformer Working Memory**

The second paper would address the realization that a static context window is a poor abstraction for meta-learning and large structured environments.

Its focus would be:

* dynamic selection of queries and keys,
* scratchpad or notepad memory,
* iterative observation and hypothesis testing,
* and working-memory behavior over large token banks.

This paper would shift the contribution from “better positions” to “better reasoning procedures over positioned data.”

---

## **7.3. Relay Learning**

The third paper would focus on the token and objective side of the problem:

* quaternion or other structured type-value token formulations,
* token synthesis in place of fixed vocab lookup for large value spaces,
* hierarchical compression into learned concept tokens,
* continual refinement of reusable abstractions,
* and training objectives that extend beyond next-token prediction.

This is the paper where the vocabulary problem, the abstraction problem, and the human-machine handoff problem come together.

---

## **7.4. MonSTER Models as Domain-Agnostic Multitask Learners**

The final integrative paper would combine the previous pieces into the broader thesis:

A successful general reasoning architecture should not rely on the three brittle assumptions of the standard transformer baseline—namely,

* a fixed and domain-specific positional scheme,
* a static attention field over one monolithic context window,
* and a fixed human-defined vocabulary.

Instead, it should provide:

* a unified geometric substrate,
* dynamic working memory,
* and structured token synthesis with abstraction learning.

This is the sense in which I now think of “MonSTER Models” as domain-agnostic multitask learners. The phrase no longer refers only to a positional encoding. It refers to an architectural family.

---

## **7.5. Immediate Next Steps**

The most important next steps are now clearer than they were before the summer began:

1. **Finalize the triadic MonSTER implementation and its ablations**  
   Establish clearly what it buys on its own, and where its gains saturate in single-task settings.

2. **Build the right benchmark suite**  
   Move beyond isolated ARC-style evaluation and toward a multitask suite spanning language-like, grid-like, and higher-dimensional spatial tasks.

3. **Prototype dynamic working memory**  
   Test whether dynamic attention threads actually improve meta-learning behavior relative to static concatenation.

4. **Develop typed value token experiments**  
   Especially for color, grid values, and structured environment states.

5. **Separate the papers cleanly**  
   So that each contribution can be evaluated on its own rather than disappearing into one overly broad system paper.

The main conclusion of the summer is therefore not that my original idea was wrong. It is that the original idea was only one piece of the real problem.

MonSTER remains, in my view, the right direction for positional geometry. But the summer made it clear that geometry alone is not enough. To build a model that genuinely reasons across domains, one must solve geometry, memory, tokenization, and abstraction together. That is a larger agenda than the one I started with, but it is also a more coherent one.

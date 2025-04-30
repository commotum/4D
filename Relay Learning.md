We introduce **Taxoformer**, a novel factorized embedding framework that carves each 512-dim token embedding into a compact **type** subspace (16 D) and a residual **instance** subspace (496 D), with the type table **tied across all layers** to enforce a shared semantic skeleton. This hierarchical inductive bias ensures that “coarse” semantics (e.g. *animal-ness*) remain invariant as representations deepen, while the instance subspace encodes idiosyncratic nuance. By channelling all type-related gradients into a single, low-rank manifold, Taxoformer accelerates convergence (up to 20 % fewer training steps on standard language-modeling tasks), curbs overfitting (5–10 % relative gain in low-resource generalization), and yields introspectable internals via straightforward probing of the 16-D taxonomy.  

Unlike prior work on mixture-of-experts or hyperbolic embeddings, our method requires no external hierarchy: the taxonomy emerges organically from data under a single, shared embedding table. We validate on WikiText-103 and GLUE benchmarks, and demonstrate that type–instance disentanglement not only bolsters performance but also furnishes a transparent axis for semantic analysis. Taxoformer thus offers a parsimonious yet potent inductive bias—transforming “attention is all you need” into “attention plus taxonomy is even better.”


**Abstract**  
We propose **Relay Learning**, a principled framework that bridges the “bitter lesson” of scale—where pure statistical methods eclipse handcrafted heuristics—and a “better lesson” in which expert knowledge is revived by endowing it with a learnable scaling factor.  In Relay Learning, each 512-dim token embedding is **factorized** into a low-rank **taxonomy** component (e.g. human-expert types) and a high-capacity **instance** component, with the taxonomy table **tied globally** across layers.  This simple relay of information—from human insight to statistical estimator and back—yields three key advantages: (1) **accelerated convergence**, since all taxonomy-related gradients reinforce a single embedding manifold rather than fragment across layers; (2) **continual learning** via compartmentalized subspaces that isolate semantics from idiosyncratic noise, mitigating catastrophic forgetting; and (3) **personalization** by permitting a downstream user-type head to hand off bespoke instance tokens, unlocking models that learn both generalizable concepts and individual preferences.  We validate Relay Learning on standard language modeling and low-resource transfer benchmarks, demonstrating up to 25 % fewer training steps, a 7 % relative gain in few-shot robustness, and clear disentanglement of taxonomy versus instance semantics under probing.  By formalizing AI development as a **relay race**—where humans set the coarse strategy and statistical engines execute the fast hand-offs—Relay Learning delivers a parsimonious yet potent inductive bias, charting a path toward more efficient, resilient, and interpretable neural systems.

Prophetic perfect tense in practice.

“The feature vectors associated with each word are learned, but they could be initialized using prior knowledge of semantic features."

@article{bengio2003neural,
  title        = {A Neural Probabilistic Language Model},
  author       = {Bengio, Yoshua and Ducharme, R{\'e}jean and Vincent, Pascal and Jauvin, Christian},
  journal      = {Journal of Machine Learning Research},
  volume       = {3},
  pages        = {1137--1155},
  year         = {2003},
  url          = {http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf}
}


"Last time, I asked: 'What does mathematics mean to you?', and some people answered: 'The manipulation of numbers, the manipulation of structures.' And if I had asked what music means to you, would you have answered: 'The manipulation of notes?' "
-Serge Lang
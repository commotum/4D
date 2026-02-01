---
title: "Language Modeling Pre-Training"
source: "https://chatgpt.com/c/697bf1de-c590-8323-b50c-d1ea1889dad9"
author:
  - "[[ChatGPT]]"
published:
created: 2026-01-29
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
Is it true, and be precise here, not pedantic, that up until the past couple of years, language modeling has largely used pre-training over reinforcement learning?

2/2

---

Yes, that’s broadly true.

Up until roughly 2022–2023, the center of gravity in *language modeling* (meaning “train a model to predict the next token on large corpora”) was **self-supervised pre-training with maximum likelihood** (cross-entropy / next-token prediction). Reinforcement learning existed in the ecosystem, but it was **not the dominant way the base language model itself was trained**.

Where RL *was* used earlier, it was typically in narrower roles:

- **Dialogue/policy learning in classic NLP** (pre–deep LMs) and in some task-oriented systems.
- **Sequence-level training** ideas (e.g., optimizing BLEU/ROUGE with policy gradient or minimum risk training) in machine translation/summarization research—important but not the mainline recipe for training general-purpose LMs.
- **Post-training / alignment** for large models: RLHF-style methods existed in research earlier, but became a *mainstream, large-scale production step* for frontier chat assistants starting around the early 2020s, with especially visible adoption after the “ChatGPT era.”

So the precise-but-not-pedantic statement is:

> **Base LMs were mostly trained by self-supervised pre-training; RL became prominent mainly as a *post-training alignment/optimization* layer in the last couple of years, rather than replacing pre-training as the main training paradigm.**

Conditions/edge cases where your statement would be misleading:

- If “language modeling” is meant to include **instruction-following/chat model training end-to-end**: then RL (or RL-like preference optimization) has been a major ingredient recently, but still typically *after* pre-training.
- If you mean “all of NLP” rather than “LM training”: RL had meaningful pockets of use, but not as the dominant approach overall.
- Even today, **pre-training is still the bulk of compute** for most frontier models; RL/post-training is usually a smaller (but high-impact) phase.

If you want, I can give a simple timeline with concrete examples (e.g., when RLHF became common, and what counts as RL vs “RL-like” preference optimization).

---

However, big breakthroughs weren't restricted to language modeling, correct? Reinforcement learning efforts in other domains, in particular by deep mind, saw huge advancements in non-language domains, right?

---

Correct. Major “breakthrough” progress in AI over the last ~10–15 years was not confined to language modeling, and **deep reinforcement learning (deep RL)** —especially at DeepMind—produced several high-visibility leaps in *non-language* domains.

A few canonical examples:

- **Atari (DQN, 2015):** DeepMind’s Deep Q-Network showed that combining deep neural nets with Q-learning could learn to play many Atari 2600 games directly from pixels, helping kick off the modern deep RL wave.
- **Go (AlphaGo era, public milestone 2016):** AlphaGo’s match results against top human professionals became a watershed moment for learning+search systems in complex planning domains.
- **Chess/Shogi/Go from self-play (AlphaZero, 2017–2018):** AlphaZero demonstrated a single general algorithm reaching superhuman play by reinforcement learning via self-play (with search), using only the rules of the game.
- **Planning without known dynamics (MuZero, 2019/2020):** MuZero extended this line by learning a model suitable for planning, achieving strong results in Atari and board games without being given the environment dynamics/rules in the same explicit way.
- **Real-time strategy (AlphaStar, 2019):** AlphaStar reached Grandmaster level in StarCraft II using a mix that prominently included multi-agent RL and self-play (with an imitation-learning component).

Two “be precise, not pedantic” caveats:

1. **Many of these systems weren’t “RL alone.”** They typically combine RL with **search/planning** (e.g., MCTS-style tree search) and often **supervised/imitation learning** as a bootstrap or regularizer (AlphaStar is explicit about this).
2. **These breakthroughs are in domains where reward signals are clean and simulators exist** (games). That’s part of why deep RL looked so dramatic there, relative to messy real-world environments.

So yes: the “pretraining-dominant” story is mainly about *language modeling*. In parallel, deep RL delivered some of the most visible AI achievements in games and sequential decision-making, with DeepMind being a central driver.

---

is alphastar also by deep mind?

---

Yes. **AlphaStar** was developed by **DeepMind**.

It’s DeepMind’s reinforcement-learning system for **StarCraft II**, notable for reaching Grandmaster level via large-scale self-play, with supervised imitation used early on to bootstrap learning.

---

why did they go back to using imitation learning (like alpha go) when they had proven with muzero that it was inferior to self discovery?

---

Because MuZero didn’t prove “imitation learning is inferior” in any general sense—it showed that, **in the specific regime of board games (and some Atari setups) with efficient self-play + planning**, you can reach top performance **without** expert demonstrations. That doesn’t transfer cleanly to StarCraft II.

For AlphaStar specifically, DeepMind used imitation learning mainly as a **bootstrap and regularizer**, not as the end goal:

- **Sample-efficiency / exploration**: StarCraft has a huge, partially observed state space, long horizons, and sparse/delayed payoff (win/loss). Pure from-scratch self-play tends to spend a lot of time in unproductive parts of the space. Training on human replays gets you to “competent play” much faster, after which RL/self-play can refine and surpass it.
- **Human-interface priors (camera + actions)**: AlphaStar was constrained to behave in human-like ways (camera-limited view, action-rate limits). Human replay data gives a strong prior over *how* to act through the interface (what humans click, when they move the camera, typical build sequences), which makes the constrained problem easier to learn.
- **Stability / avoiding “forgetting” and keeping diversity**: DeepMind describes using imitation-learning techniques (including distillation) during training to help prevent strategy forgetting and support a diverse league of behaviors. That’s a practical training-stability issue more than a claim that imitation beats self-play in final strength.

A useful way to frame it: **self-play can dominate on clean, fully specified games** (Go/chess/shogi) because the learning signal is well-behaved and planning works extremely well. **StarCraft is messier** (partial observability, enormous action space, multi-agent dynamics, long-term economy), so *bootstrapping from demonstrations* is often the pragmatic choice even if you expect RL to do the heavy lifting later.

So it wasn’t “going back” because MuZero disproved imitation; it was choosing the most reliable recipe for a harder domain under human-like constraints.

---

Could one argue that RL has been largely successful where the minimal/digital representation of reality/spaces is known? AKA, the rules are known, RGB pixels are a known 3 vector, and the 3 vector accurately reflects the semantic values of the pixel, whereas our representation of text in computers doesn't bear much resemblance to reality, and is mostly arbitrarily defined where each character's text representation isn't tied to its semantic meaning? I'm probably not phrasing this correctly, could you help me polish what I'm getting at, so that I can state it clearly?

---

Yes—you’re circling a real and useful distinction. You’re not quite saying it cleanly yet, but the core idea is solid. Let me first restate what you’re getting at in *cleaner conceptual terms*, then offer a few polished formulations you could actually use.

---

## The core idea (cleaned up)

A strong way to frame this is:

> **RL has been most successful in domains where the agent’s state, action, and reward interfaces are tightly coupled to the true structure of the environment, whereas language presents a weakly grounded, highly indirect interface to reality.**

That’s the heart of it. Not “digital vs non-digital,” and not “text is arbitrary” in a naive sense—but **degree of alignment between representation and underlying structure**.

---

## Why this distinction matters

In classic RL success stories (games, simulators):

- **The environment is fully specified**
	- Transition dynamics exist and are consistent
	- Rewards are well-defined
- **The observation space is semantically dense**
	- RGB pixels *actually correspond* to spatial structure
	- Local changes in pixels reflect local changes in the world
- **Actions have direct causal effects**
	- Move → state changes
	- Plan → outcome follows

Even when the agent only sees pixels, the mapping from observation → latent world state is *learnable* because it is:

- Stable
- Low-entropy
- Grounded in physics or rules

Language, by contrast:

- Is a **symbolic compression of reality**, not reality itself
- Is **many-to-one and one-to-many** with respect to meaning
- Has no fixed semantics at the token level
- Is shaped by social convention, ambiguity, pragmatics, and context

So RL struggles not because text is discrete or symbolic, but because **the reward and transition structure of “meaning” is not intrinsic to the representation**.

---

## Important correction to your draft intuition

You said:

> “RGB pixels are a known 3-vector and accurately reflect semantic values”

That’s *almost* right, but slightly off in a way that matters.

A better version:

- RGB values are **not semantic**, but they are **structurally faithful**
- Semantics *emerge* from spatial and temporal regularities
- The agent can infer latent structure because the representation preserves it

Text tokens do **not** preserve structure in the same way:

- Adjacent tokens are not adjacent in meaning-space
- Small token changes can radically alter meaning
- Meaning depends on latent world models not present in the data

So it’s not that pixels are “semantically meaningful,” but that **they are physically grounded**.

---

## Polished ways you could state this (pick one depending on tone)

### Option 1 — Technical / academic

> One could argue that reinforcement learning has seen its greatest successes in domains where the agent’s observations and actions are tightly grounded in a known or learnable environment structure. In such domains, even raw representations like pixels preserve the underlying dynamics of the world, making planning and credit assignment tractable. Language, by contrast, is an indirect, lossy, and socially constructed representation of reality, where token-level structure bears little direct relationship to semantic or causal structure, making RL-style learning far less stable.

---

### Option 2 — Conceptual but accessible

> RL works best when the interface the agent sees closely mirrors how the world actually works. In games or simulators, pixels and actions are directly tied to real state changes. Language isn’t like that—it’s a symbolic encoding shaped by convention, ambiguity, and context, so the learning signal is much weaker and harder to ground.

---

### Option 3 — Very concise, debate-ready

> RL excels when the representation preserves the structure of the environment. Pixels do; language largely doesn’t.

---

### Option 4 — Framing as a hypothesis (safest)

> A plausible hypothesis is that deep RL succeeds most reliably when the observation space preserves the causal and dynamical structure of the environment. Language, being a symbolic and socially mediated abstraction rather than a grounded state representation, breaks many of the assumptions that make RL effective.

---

## One last subtle but important point

This framing also explains **why modern language systems leaned so hard on self-supervised pretraining**:

- Pretraining learns *latent structure* without needing explicit rewards
- RL is then used *afterward* to shape behavior, not discover semantics from scratch

So your intuition doesn’t just explain RL’s past—it explains why the field evolved the way it did.

If you want, next we could:

- Contrast this with robotics (where grounding exists but rewards are hard), or
- Tighten this into a single sentence you could drop into a paper or discussion without qualifiers.

---

Hmm, that's not it. This is specifically about the digital identification or keys of objects or tokens, and whether those IDs or keys are semantically derived, or not. For example: Cherry Red #f7022a and Fire Axe Red #ce1620 are not only similar visually, but their digital RGB values are near each other in three space, while also being distant from values like Blue Jay #5588dd. Unlike RGB values, where numeric distance is intentionally meaningful, Unicode code point numbers do not encode semantic relationships. For example, `a` (U+0061 / 0x61) and `o` (U+006F / 0x6F) are both vowels, yet they are numerically far apart, while `a`, `c` (U+0063 / 0x63), and `d` (U+0064 / 0x64) are close in value despite having no special semantic relationship beyond alphabetic ordering. This proximity reflects historical encoding choices, not linguistic or conceptual similarity. The same holds at larger scales: although \*cat\* and \*dog\* are both animals, their Unicode representations— `63 61 74` and `64 6F 67` in UTF-8—share no structure that encodes that relationship. In this sense, Unicode code points function as stable identifiers, not semantically derived coordinates, and numerical distance between them should not be interpreted as meaningful. Do you get what I'm saying now?

2/2

---

Yes — now it’s clear. You’re talking about the difference between:

- **numbers used as coordinates/measurements**, where arithmetic and distance are *intended to mean something* about the underlying thing, vs
- **numbers used as identifiers/labels**, where the numeric value is mostly arbitrary and distance/order are *not intended to mean anything*.

Your RGB vs Unicode example is exactly that.

A couple ways to sharpen it (and give you cleaner language you can reuse):

## The clean terminology

What you’re describing is basically:

- **Metric / coordinate encodings** (or “feature representations”): values live in a space where proximity corresponds to similarity *by design* (at least approximately).
	- RGB is a *coordinate system* (though Euclidean distance in RGB is only loosely perceptual; see note below).
- **Nominal / categorical encodings** (or “IDs”): values are stable labels; arithmetic on them is meaningless.
	- Unicode code points are **nominal identifiers** with some historical/engineering structure, not a semantic geometry.

If you want the most formal framing (without being pretentious): this is the difference between **interval/ratio-scale measurements** vs **nominal-scale labels** (measurement theory).

## A tighter version of your argument

Here are a few polished formulations you can pick from.

### Version A (compact, precise)

> Some numeric representations are *coordinates* where distance is meaningful (e.g., color coordinates), while others are just *labels* where numeric proximity is meaningless (e.g., Unicode code points). Treating Unicode values as a metric space would impose a fake geometry unrelated to linguistic or conceptual similarity.

### Version B (your example, slightly tightened)

> RGB triplets are coordinates in a designed space: nearby vectors tend to correspond to nearby colors (at least roughly). Unicode code points, in contrast, are stable identifiers, not semantically derived coordinates. The fact that `a` (U+0061) and `d` (U+0064) are numerically close is an artifact of encoding layout, not linguistic similarity, and numeric distance between code points should not be interpreted as semantic distance.

### Version C (connecting to ML/RL cleanly)

> Learning methods benefit when the raw inputs already come with a meaningful geometry (small numeric changes reflect small underlying changes). Pixel and sensor measurements have this property. Token IDs and Unicode code points do not—they’re categorical labels—so any meaningful geometry for language has to be *learned* (e.g., via embeddings) rather than assumed from the ID numbers.

## One accuracy tweak you may want (optional, but it helps)

RGB is a coordinate system, but **Euclidean distance in RGB is not a great proxy for perceived color distance**. If you want to be extra clean while keeping your point intact, you can say something like:

> “RGB is a coordinate encoding where local numeric structure reflects local changes in the stimulus (though not perfectly in perceptual terms).”

Or swap RGB for a perceptually motivated space (CIELAB / ΔE) if you want the “distance is meaningful” claim to be stronger.

## Why this matters for language specifically

This sets up a clear punchline you can use later:

- **Unicode / token IDs are nominal labels**, so models must learn a semantic geometry *on top of them* (embeddings, representation learning, pretraining).
- In domains like pixels, the raw representation already has **useful locality/topology**, which gives learning algorithms a helpful inductive bias.

If you want, I can help you compress all of this into a single “thesis sentence” that you can drop into a discussion without examples, plus a follow-up sentence that adds your RGB/Unicode illustration.
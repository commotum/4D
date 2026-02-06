---
title: "RL Models for Multiple Games"
source: "https://chatgpt.com/c/698278fb-f24c-8323-aa0d-ee5dc8f4172f"
author:
  - "[[ChatGPT]]"
published:
created: 2026-02-03
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
Are there Reinforcement Learning models out today that can learn to play multiple games with a single model? I know with architectures like alphago they could be trained or finetuned to do different things, but usually it was different models per task, not a single model capable of doing many things the way LLMs do today. --- CONTEXT: Oren Neumann and Gros (2024): In "AlphaZero Neural Scaling and Zipf's Law: a Tale of Board Games and Power Laws", Neumann & Gros attempt to apply the quanta model to explain the neural scaling laws seen in game-playing RL. In particular, they find that the frequencies at which different game states are visited typically follow a power law, and they attempt to connect this power law to the scaling laws their RL setup exhibits. Intriguingly, they find that the games where RL exhibits inverse scaling behavior are also the games where the state distribution exhibits unusual structure where end-game states have high frequency (in contrast to games like chess where particular end-game board states are likely never repeated). This paper has me wondering whether there might be some theory, for LLMs, that unifies pre-training and RL ("reasoning") scaling laws, and perhaps some quanta-like story in terms of frequencies will have a role to play there.

---

---

Yes — there *are* RL(-ish) systems today that use **one set of weights** to play **many games**, but they show up in a few “families,” and they tend to work best when the games share a common interface (same observation format + same action set), or when you condition the policy on something like a task/game identifier (analogous to prompting).

Here are concrete examples that are closest to what you’re describing:

## Multi-game “classic” deep RL (online RL) with one policy network

### IMPALA (Atari-57 / DMLab-30)

A key early example is **IMPALA**, which explicitly set out to “solve a large collection of tasks using a single reinforcement learning agent with a single set of parameters,” and demonstrated multi-task training on **DMLab-30** and **Atari-57**.

### PopArt-IMPALA (one agent across 57 Atari games)

A follow-up line of work combined IMPALA with **PopArt** normalization to handle the “different reward scale per game” problem. DeepMind described this as producing **a single agent** that can play “a whole set of 57 diverse Atari video games” with **above-human median performance** across the set.

**What this gets you:** one trained policy network that can act across many games, without swapping weights each time.  
**What it doesn’t fully get you:** the “drop-in generality” feeling of LLMs across wildly different interfaces. Atari is relatively friendly here because observation and action formats can be standardized.

## “LLM-like” generalists: sequence models trained on many games (offline RL)

### Multi-Game Decision Transformer (single Transformer, up to 46 Atari games)

This is one of the cleanest demonstrations of the “LLM vibe” in RL: a **single transformer-based model with a single set of weights**, trained **purely offline**, that can play **up to 46 Atari games** “simultaneously,” near human level. They also emphasize that trends seen in language/vision carry over, including **performance scaling with model size** and **rapid adaptation to new games via fine-tuning**.

Notably, the same paper also reports a form of **inverse scaling** for some TD-style methods (they say performance can *drop* with larger models for certain approaches), which is extremely relevant to the “inverse scaling” thread you raised.

**Why this matters for your context:** it pushes RL toward “pretrain on a giant dataset of trajectories, then act by conditional generation,” which is structurally much closer to language modeling than classic online RL.

## Generalist agents that span many games (and beyond)

### Gato (generalist policy model across 604 tasks, including Atari)

DeepMind’s **Gato** is an existence proof that you can train **one model** across a very wide task mixture (games + robotics + more). The paper explicitly states their results across tasks come from “a single pretrained model with a single set of weights,” and reports Atari performance across a set of games within that single model.

This isn’t “pure RL” in the narrow sense (it’s largely trained from offline data / imitation-style objectives), but it *is* a single policy model that spans many environments in the LLM-ish sense.

### SIMA / SIMA 2 (multiworld, instruction-following across 3D games)

**SIMA** trains agents to follow free-form language instructions across a diverse range of 3D environments including commercial video games, using a generic keyboard/mouse interface.  
And **SIMA 2** (announced Nov 13, 2025) is presented as the next step: a Gemini-powered agent meant to operate across many 3D virtual worlds.

Again: this is not “AlphaZero-style self-play RL,” but it *is* squarely in the “single model, many games/worlds” direction.

## Multi-game board game style (closer in spirit to AlphaZero)

A lot of AlphaZero-style pipelines historically still trained **separate networks per game** because the input/output structure is tightly tied to board size + move encoding.

But there are now papers explicitly targeting **shared weights across multiple board games**. For instance, **AlphaViT** (transformer/Vision Transformer inside an AlphaZero-like framework) claims agents that “can play multiple board games of varying sizes using a single neural network with shared weights,” and reports simultaneous training across games like Connect 4, Gomoku, and Othello using one network.

This is quite close to the “one model, many games” thing you’re looking for, but note it’s still within a “board-as-image/tokens” family where a transformer can help unify variable board sizes/action spaces.

## Why multi-game RL still doesn’t feel as universal as LLMs

A few practical reasons the “one model for everything” paradigm came later in RL:

- **Action-space mismatch** is brutal. LLMs always output tokens. Games have wildly different action sets and semantics. Multi-game success often relies on standardizing the interface (e.g., Atari-like shared joystick actions; or “keyboard/mouse everywhere” in SIMA).
- **Reward scale + sparsity** differences cause “task imbalance.” PopArt is basically a direct attack on this.
- **Non-stationary data**: in online RL you’re constantly changing the data distribution as the policy improves, which complicates the neat “fixed dataset → scaling law” story you get in language modeling.
- **Interference/negative transfer**: when tasks compete inside one network, some tasks get worse unless you do something explicit to balance gradients/capacity.

## Connecting to your Zipf / quanta / unified scaling-law question

Your instinct here is very aligned with where the field has been moving:

- Oren Neumann and Claudius Gros explicitly frame AlphaZero scaling through a language-model-inspired lens: they claim game-state visitation follows Zipf, and relate “inverse scaling” to unusual Zipf curves where end-game states are unusually frequent.
- The *sequence modeling* perspective (“RL as tokens”) makes the bridge to LLMs much tighter. The Trajectory Transformer line is explicit about repurposing large-scale sequence modeling tools for RL.
- Multi-Game Decision Transformers then goes one step further: it explicitly says scaling trends from language/vision show up in this multi-game RL regime, and it also observes inverse-scaling-like pathologies for some RL objectives/models as you scale.

A reasonable “unified” story (without overcommitting) is:

1. **Pretraining** (language or offline trajectories) learns high-frequency structure first (the head of a Zipf-like distribution).
2. **RL / planning / fine-tuning** shifts the effective data distribution toward “harder” or more decision-critical regions — which might be tail events (rare states, rare but crucial decisions), or might be weirdly *high-frequency but low-signal* regions (like repetitive end-game clusters), depending on the environment.
3. Scaling behavior (and inverse scaling) could then reflect *how model capacity reallocates* across that frequency spectrum under the training dynamics — which is exactly the kind of mechanism Neumann & Gros are pointing at for AlphaZero.

If you want, I can also map the multi-game agents above onto your “state-frequency / quanta” framing more explicitly (e.g., what the “tokens” are, what the “dataset” is, and where Zipf-like mixtures plausibly come from in each setup).
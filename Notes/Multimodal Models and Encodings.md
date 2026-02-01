---
title: "Multimodal Models and Encodings"
source: "https://chatgpt.com/c/697f2c6f-b3f4-8320-8bcb-5c429267cb24"
author:
  - "[[ChatGPT]]"
published:
created: 2026-02-01
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
Do most multimodal models today, are they more like systems where they use separate models and positional encodings per dimension (like one encoding / embedding model for image generation and classification, with another for text) or are there any truly multi-dimensional models that have a singular positonal encoding mechanism trained on all types of data from the ground up?

---

Most *deployed* multimodal systems today still look more like **“specialized parts + a shared core”** than a single, uniform model that treats every modality with one universal positional scheme.

## 1) The dominant pattern: separate encoders + a text LLM “spine”

For vision+language models (and many audio+language models), the most common setup is:

- a **vision encoder** (often a ViT/CLIP-style model with its own **2D / patch positional structure**),
- optionally a **resampler** / pooling module,
- a **projection layer** that maps visual features into the LLM’s embedding space,
- then a **language model** that operates on a **1D token sequence**.

This is the “two-component” (vision encoder + LLM) family you see across lots of open and closed VLMs, and surveys/overviews describe it as the standard recipe.

**Positional encoding implication:**  
You effectively have **multiple positional systems**:

- Vision side: encodes spatial structure (patch grid / relative positions, etc.).
- LLM side: encodes 1D sequence positions for the *inserted* “image tokens” (which are really projected features).

So it’s *not* “one positional encoding trained on all data from the ground up”; it’s a **fusion of representations** where spatial structure is mostly handled before the LLM ever sees it.

Classic example of this modular “connector” style is DeepMind ’s Flamingo, which explicitly uses a vision encoder + resampler + cross-attention into a language model.

## 2) “Native multimodal” single-network claims exist — but “single positional encoding for raw everything” is still uncommon

Some frontier models are described publicly as **a single model trained end-to-end across modalities**. For example, OpenAI describes GPT-4o as “a single new model end-to-end across text, vision, and audio.”

However, even when the *core network* is unified, in practice there are usually still modality-specific steps somewhere (e.g., turning images/audio into a token-like representation, or using modality-specific input adapters). Public writeups typically don’t give enough detail to conclude there’s literally **one universal positional encoding operating directly on raw pixels + raw waveform + raw text** with no modality-specific structure.

So: “native multimodal” often means **one transformer backbone / one set of weights**, not necessarily **one shared positional geometry for all raw modalities**.

## 3) The closest thing to what you’re describing: early-fusion “all tokens in one stream”

There *are* models trained “from inception” on **mixed-modal token sequences**, where images are tokenized into discrete tokens (e.g., via VQ-style tokenizers) and interleaved with text so the transformer processes **one stream**.

Examples:

- Chameleon: “early-fusion token-based mixed-modal” models that can understand/generate images and text in arbitrary sequence.
- Unified-IO 2: tokenizes multiple modalities into a shared space and uses a single encoder–decoder transformer.

**Positional encoding implication:**  
These are much closer to a **single positional mechanism** in the *main transformer* because everything is serialized as tokens. But there’s still an important caveat:

- Images usually aren’t fed as raw pixels; they’re fed as **tokens produced by an image tokenizer** (which itself imposes structure and often has its own inductive biases).
- Even if the transformer uses a single 1D position index over the combined stream, the *image tokenization* typically preserves 2D neighborhood information implicitly (by raster order, patch order, learned codebooks, etc.). In other words, the “shared positional encoding” is partly achieved by **making vision look like language**, not by inventing a single geometric positional system that natively spans text+2D images+audio.

## 4) A different “universal-input” direction: Perceiver-style models

Perceiver IO was explicitly proposed as a general architecture handling arbitrary inputs/outputs, using flexible positional features (e.g., Fourier features) to represent structured positions.  
This is conceptually aligned with “one model family that doesn’t assume the modality,” though real instantiations still choose specific input representations and position features per data type.

## Bottom line

- **Most multimodal models in use**: *separate encoder(s) per modality* + a shared LLM backbone; effectively **multiple positional systems**, even if the fusion is tight.
- **Some models move toward “true early fusion”**: a single transformer over a mixed token stream (images tokenized into discrete tokens), which is the closest to “one positional encoding mechanism,” but it’s achieved by **tokenizing modalities into a shared token space**, not by a single native positional geometry over raw modalities.
- **Frontier “native multimodal” models**: may be one end-to-end network, but public info usually isn’t sufficient to say they use one unified positional encoding over raw text+pixels+audio.

If you tell me which modalities you care about (vision+text only? audio too? video?) I can map the common positional-encoding choices and where “modality-specific” structure usually sneaks back in.
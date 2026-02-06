---
title: "Thread by @karpathy"
source: "https://x.com/karpathy/status/2017703360393318587"
author:
  - "[[@karpathy]]"
published: 2026-01-31
created: 2026-02-03
description: "licklider escher faraday"
tags:
  - "clippings"
---
**Andrej Karpathy** @karpathy [2026-01-31](https://x.com/karpathy/status/2017703360393318587)

nanochat can now train GPT-2 grade LLM for <<$100 (~$73, 3 hours on a single 8XH100 node).

GPT-2 is just my favorite LLM because it's the first time the LLM stack comes together in a recognizably modern form. So it has become a bit of a weird & lasting obsession of mine to train a model to GPT-2 capability but for much cheaper, with the benefit of ~7 years of progress. In particular, I suspected it should be possible today to train one for <<$100.

Originally in 2019, GPT-2 was trained by OpenAI on 32 TPU v3 chips for 168 hours (7 days), with $8/hour/TPUv3 back then, for a total cost of approx. $43K. It achieves 0.256525 CORE score, which is an ensemble metric introduced in the DCLM paper over 22 evaluations like ARC/MMLU/etc.

As of the last few improvements merged into nanochat (many of them originating in modded-nanogpt repo), I can now reach a higher CORE score in 3.04 hours (~$73) on a single 8XH100 node. This is a 600X cost reduction over 7 years, i.e. the cost to train GPT-2 is falling approximately 2.5X every year. I think this is likely an underestimate because I am still finding more improvements relatively regularly and I have a backlog of more ideas to try.

A longer post with a lot of the detail of the optimizations involved and pointers on how to reproduce are here:

https://github.com/karpathy/nanochat/discussions/481…

Inspired by modded-nanogpt, I also created a leaderboard for "time to GPT-2", where this first "Jan29" model is entry #1 at 3.04 hours. It will be fun to iterate on this further and I welcome help! My hope is that nanochat can grow to become a very nice/clean and tuned experimental LLM harness for prototyping ideas, for having fun, and ofc for learning.

The biggest improvements of things that worked out of the box and simply produced gains right away were 1) Flash Attention 3 kernels (faster, and allows window\_size kwarg to get alternating attention patterns), Muon optimizer (I tried for ~1 day to delete it and only use AdamW and I couldn't), residual pathways and skip connections gated by learnable scalars, and value embeddings. There were many other smaller things that stack up.

Image: semi-related eye candy of deriving the scaling laws for the current nanochat model miniseries, pretty and satisfying!

![Image](https://pbs.twimg.com/media/HABO1KxbEAE9_6h?format=jpg&name=large)

---

**Youssef El Manssouri** @yoemsri [2026-01-31](https://x.com/yoemsri/status/2017705288673935647)

The Cost of Intelligence is falling faster than Moore's Law.

A 600x reduction in 7 years implies that we are limited by algorithmic inefficiency. At this rate, training a GPT-4 class model becomes a high school science project by 2032.

---

**Aaryan Kakad** @aaryan\_kakad [2026-01-31](https://x.com/aaryan_kakad/status/2017731048881459454)

I tried training a nanogpt on the book “the beginning of infinity” inspired by your video.

Loved the step by step implementation.

I just used tiktoken instead of character level tokenizer.

Would absolutely love to watch some more similar videos. Keep teaching great stuff to

> 2026-01-26
> 
> 🚨 I trained a GPT-style language model on a single philosophy book.
> 
> Inspired by @karpathy's tutorial, I built a nanoGPT from scratch and trained it on The Beginning of Infinity by @DavidDeutschOxf.
> 
> @karpathy used the character-level tokenizer from the tutorial, I used tiktoken
> 
> ![Image](https://pbs.twimg.com/media/G_lX-PnbgAAJF7q?format=png&name=large)

---

**Delta Force Game** @DeltaForce\_Game

New Season MORPHOSIS is Live.

"Vlinder": New Support Operator

🦋Reborn from the Cocoon
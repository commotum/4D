---
title: "$100 Model Parameters"
source: "https://chatgpt.com/c/698261dd-a688-8324-91f6-34df517fb2fb"
author:
  - "[[ChatGPT]]"
published:
created: 2026-02-03
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
Can you tell me the number of parameters of the model that was only 100 bucks to train? Like is it GPT2-xl? or 115m?

2/2

---

---

The “<<$100 to train” model in Andrej Karpathy ’s nanochat speedrun is **~1.38B parameters** — specifically **1,384,124,976**.

So yeah: it’s **much closer to GPT‑2 XL (~1.5B params)** than to **~115M/117M** (that’s the small GPT‑2 size). It’s not literally GPT‑2 XL though — it’s a **different architecture** tuned to *match/beat GPT‑2’s capability metric*, not to replicate GPT‑2’s exact config.

Here’s the parameter breakdown they posted for the Jan 29 “d24” model:

- **wte (token embeddings):** 50,331,648
- **value\_embeds:** 603,979,776
- **lm\_head:** 50,331,648
- **transformer\_matrices:** 679,481,856
- **scalars:** 48
- **total:****1,384,124,976**

(Those **value embeddings** are a huge chunk of why it lands in the ~1.4B range.)

And if you’re tying it back to the tweet: he describes hitting GPT‑2-grade performance in **~3.04 hours for ~$73 on 8×H100**, i.e., under $100.

---

So what performance is being matched here? Like if it's a different size model, what is the task, and which model's capability score at that task is the "benchmark"

---

---

In that nanochat “<$100 to GPT‑2” thing, the **benchmark isn’t “match GPT‑2’s loss/perplexity”** and it isn’t a single task. It’s a **capability score** called **CORE** from the DataComp-LM chmark"\] paper (DCLM).

## What “performance” is being matched?

**CORE = “Core centered accuracy”** over a **subset of 22 eval tasks**. For each task, they take the model’s accuracy and **linearly rescale it so that:**

- **0 = random guessing**
- **1 = perfect accuracy**

…then they aggregate across the 22 tasks.

That normalization is important: it makes tasks with different choice counts (binary vs 4-way vs 5-way) comparable, because “random baseline” is different.

## What exactly is the “task”?

It’s the **suite** (CORE), not one dataset. In the DCLM paper’s Appendix G, the **Core tasks** are listed like this (with the shot settings):

- **AGI Eval LSAT‑AR** (3‑shot)
- **ARC Easy** (10‑shot)
- **ARC Challenge** (10‑shot)
- **Big-Bench (6 tasks, all 10‑shot):** QA Wikidata, Dyck languages, Operators, Repeat Copy Logic, CS Algorithms, Language Identification
- **BoolQ** (10‑shot)
- **CommonsenseQA** (10‑shot)
- **COPA** (0‑shot)
- **CoQA** (0‑shot)
- **HellaSwag** (evaluated at **0‑shot and 10‑shot**)
- **Jeopardy** (10‑shot)
- **LAMBADA** (0‑shot)
- **OpenBookQA** (0‑shot)
- **PIQA** (10‑shot)
- **SQuAD** (10‑shot)
- **Winograd Schema Challenge** (0‑shot)
- **Winogrande** (0‑shot)

(That “22 tasks” count comes from how the suite is defined; e.g., some items are multiple separate datasets under one bullet, and **HellaSwag is explicitly evaluated in two shot settings**.)

## Which model’s score is the benchmark?

The target/threshold is **GPT‑2’s CORE score**, stated as:

- **GPT‑2 CORE = 0.256525**

nanochat’s “time to GPT‑2” leaderboard is literally: *how fast can you train a model that reaches or exceeds that CORE score.*

So the “GPT‑2 grade”te (normalized centered accuracy over those 22 tasks), this new model’s score ≥ GPT‑2’s score.”

Not “it’s the same architecture,” and not “it matches GPT‑2 on every benchmark humans care about.”

If you want, I can also ret a number like **0.2565** intuitively (it’s “~25.7% of the way from random to perfect, averaged across the suite”), and why people like CORE for small-scale experiments (low-variance signal).
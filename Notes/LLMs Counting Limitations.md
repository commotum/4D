---
title: "LLMs Counting Limitations"
source: "https://chatgpt.com/c/698276ab-2a10-8320-ab8e-afd893210c74"
author:
  - "[[ChatGPT]]"
published:
created: 2026-02-03
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
Most LLMs seem to have the ability to count up to pretty high numbers, and I'm wondering are there any papers that have seriously investigated this to find out where that starts degrading?

2/2

---

---

Yep — there *are* papers that take “counting” seriously and actually sweep the difficulty so you can see where things start to break. One catch: “counting” gets studied in a few different ways, and the degradation point depends *a lot* on which one you mean:

- **Sequential counting / enumeration**: generate or verify a sequence with exactly *N* items (closest to “count to 500/1000”).
- **Cardinality counting**: “how many X are in this list/string?”
- **Length constraints**: “write exactly N words/tokens/sentences…”
- **Character-level counting**: “how many r’s in strawberry?”

Below are the papers that most directly probe *where performance degrades* (increasing N / increasing length / out-of-distribution).

---

## Papers that explicitly sweep N (closest to “count up to big numbers”)

### Sequential Enumeration in Large Language Models (2025)

This is the closest thing I’ve seen to “let’s systematically test how far they can count” in a controlled way.

- **What they test:** two classic cognitive-science tasks:
	- **Naming**: given a sequence, report how many items are in it.
	- **Production**: given a target number, generate a sequence with that many items.
- **Range:** target numerosities **10 → 100** (in steps of 10).
- **What “degradation” looks like:**
	- Errors and variance **increase as the target numerosity increases** (so you get a clear curve even within 10–100).
	- Even top models do much better when explicitly instructed to count, *but* counting a provided list (“naming”) stays notably harder than generating one (“production”).
	- They also report that internal “mental counting” signals get noisier later in the sequence (variance noticeably rising after ~40 steps in one analysis).
	- There are **base-10 / decade artifacts** (dips around multiples of ten), and they argue this partly reflects training-data frequency biases for decade number words.

**Why this matters for your question:** It shows degradation is already measurable well before “huge” N, and it’s not a sharp cliff — it’s an error-growth pattern as N rises.

---

## Papers that show “counting breaks” at surprisingly small lengths (letters/words)

If your intuition comes from “LLMs can count to 1000,” it’s worth noting that **exact** counting in the presence of tokenization and formatting is *much* less robust than it looks.

### Counting Ability of Large Language Models and Impact of Tokenization (2024)

- **What they focus on:** counting under different tokenization regimes and how that changes outcomes.
- **Key takeaway:** tokenization can materially shift where counting accuracy starts to degrade, and counting gets worse as sequences/strings get longer and token boundaries get uglier.

### LLM The Genius Paradox… (NAACL 2025)

- **What they test:** word-based counting tasks like character occurrence, substring occurrence, word length, distinct characters.
- **What they find:**
	- “Prompt engineering” alone doesn’t reliably fix counting.
	- Explicit *reasoning strategies* help a lot more, and code-generation (when allowed) can solve perfectly because it offloads the counting to an explicit algorithm.

### Large Language Models Lack Understanding of Character-Level Composition (2024)

- Focuses on failures of character-level handling (a root cause of a lot of “counting letters” pain).

### Why Do LLMs Struggle to Count… (2024)

- Looks at how counting errors relate to factors like frequency/complexity (useful if you care about “rare numbers/strings”).

**Why these matter:** They show “counting” can fall apart at *tiny* scales depending on representation — which helps explain why a model might “count to 1000” in one format but fail miserably in another.

---

## “Train short, test long”: where counting degrades out-of-distribution

A different but very relevant notion of degradation is: **you train on small N and test on larger N** (classic length generalization / algorithmic generalization).

### Language Models Need Inductive Biases to Count Inductively (2024)

- **Core result:** traditional RNNs can generalize counting much more “inductively,” while Transformers typically need positional-embedding tricks and still struggle out-of-domain.

This speaks directly to *where* degradation happens in an algorithmic sense: beyond the training regime, the wheels come off unless the architecture/training gives you the right bias.

---

## Mechanistic / theoretical work that explains “why there must be a limit”

These don’t always give you a single “it fails at N=\_\_\_\_,” but they *do* formalize why unbounded exact counting is hard in vanilla Transformers.

### When Can Transformers Count to n? (theory)

- Gives conditions under which counting occurrences is possible; highlights scaling requirements (e.g., representation capacity vs sequence length).

### Contextual Counting: A Mechanistic Study… (2024)

- Introduces a toy counting task and studies how architecture/positional encoding impacts both accuracy and OOD generalization.

---

## A paper that frames the “degradation” intuition you’re describing

### Embers of autoregression… (2024)

They explicitly call out “seemingly simple” tasks like counting words as failure modes and argue performance is shaped by next-token prediction pressures (probability/frequency effects).

This lines up with a common real-world observation: when outputs become **low-probability text** (rare numerals, weird formatting, long rigid sequences), accuracy drops.

---

## So… do we have a clean, universal “LLMs degrade after N = X” answer?

Not really — and the literature kind of supports *why*:

- The most “count-like” controlled sweeps I’ve seen for modern frontier LLMs often top out around **N=100** (because evaluation becomes expensive/long and error checking gets annoying).
- Even within that range, you already see **error growth with N**, format sensitivity, and base-10 artifacts.
- On character/word counting variants, you see breakdowns at much smaller scales due to tokenization and representation issues.

---

## If you want your “degradation point” for “count to N”

If your target is literally: “print `1, 2, 3, …, N` ” (or similar), papers don’t yet standardize this up to 10k/100k. But it’s very easy to measure rigorously:

1. **Force a rigid format** (one number per line).
2. **Use deterministic decoding** (temperature 0) to reduce randomness.
3. **Parse the output** and check:
	- first error index (where it deviates)
	- types of errors (skip, repeat, digit swap, formatting break)
4. **Sweep N** (e.g., 50, 100, 200, 500, 1000, …) and plot a survival curve: “% runs correct up to step k”.

That kind of evaluation would slot directly into the conceptual framework these papers use (enumeration error vs target length).

---

If you tell me what you mean by “count” (e.g., *generate 1…N*, *count items in a list*, *count letters*, *exactly N tokens/words*), I can point you to the 2–3 most relevant papers from the list above and summarize the exact experimental setup + what their degradation curves look like.
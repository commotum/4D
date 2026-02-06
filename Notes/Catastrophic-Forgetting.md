The “original” catastrophic forgetting / *catastrophic interference* reference people usually mean is **McCloskey & Cohen (1989)**, *“Catastrophic interference in connectionist networks: The sequential learning problem.”* ([AW][1]) (Often mentioned alongside **Ratcliff (1990)** for a closely related demo in recognition-memory models. ([BPB][2]))

### What McCloskey & Cohen actually did (and why it was memorable)

Your memory is **close in spirit** (train on Task 1, then Task 2, Task 1 falls apart), but it wasn’t “modular arithmetic on one number.”

**Experiment / simulation 1: arithmetic facts (ones → twos)**

* They built a simple **3-layer backprop network** to learn basic arithmetic facts (single-digit stuff like “6 + 1”). ([AW][1])
* They note that if you train **concurrently** (mix all problems), the net can do great; but they wanted to mimic more human-like curricula, so they tested **sequential training**. ([AW][1])
* The famous punchline: after learning the **ones addition facts** perfectly, then doing **just one learning trial** on the **twos facts**, performance on the ones facts dropped hard — **from 100% to 57%**, and after two trials to **30%** (even using a “best match” scoring that’s fairly forgiving). ([AW][1])
* Even more vivid: the errors looked like **overwrite**, e.g. it starts answering “6 + 1” as if it were a twos problem. ([AW][1])

So yeah: *“train on the next thing and it forgets the previous thing”* is exactly the phenomenon — but the concrete toy task was **ones-vs-twos addition facts**, not modular arithmetic.

**Experiment / simulation 2: classic A–B then A–C paired associates**

* They also simulated **retroactive interference**: learn an **A–B list** (cue A maps to response B), then learn **A–C** (same cues A, new responses C).
* Here the forgetting was *even more brutal*: after **three learning trials** on A–C (which only got them to ~20% correct on A–C), performance on the original A–B list fell **from 100% to 0%**. ([AW][1])

That “A–B then A–C destroys A–B” pattern is basically the cleanest lab model of “new learning overwrites old mappings,” and their point was: vanilla backprop + distributed representations can make it **catastrophic**, not mild.

### Where the “modular arithmetic” memory might come from

A lot of later continual-learning writeups use **tiny synthetic tasks** (parity, mod, simple function families) because they’re:

* easy to generate endlessly,
* easy to measure forgetting precisely,
* easy to make “Task 2 conflicts with Task 1” on purpose.

So you might be remembering a *later* pedagogical example, not the 1989 demo.

### “This is a far cry from today’s LLMs,” right?

Yes—and also, annoyingly, **not really**.

**Why it *is* different**

* Modern LLMs are usually trained on a **big interleaved mixture** of data/objectives, which is closer to the “concurrent training works fine” setting McCloskey & Cohen already observed. ([AW][1])
* Also, in day-to-day use, LLM weights are typically **frozen**, so they don’t “forget” across your chats in the literal weight-overwriting sense. (What people call “amnesia” in agents is often **state / working memory** problems, not catastrophic forgetting.)

**Why it’s *not* different (when you actually do continual updates)**
As soon as you do **sequential fine-tuning / continual instruction tuning** on an LLM, the same basic pathology shows up: updating shared weights for the new thing can degrade old capabilities.

* A big empirical study on **continual instruction tuning** found catastrophic forgetting is **generally observed** in LLMs in the **1B–7B** range, and (in that range) *forgetting got worse as scale increased*, in their experiments. ([arXiv][3])
* A more recent EMNLP Findings paper ties LLM catastrophic forgetting during fine-tuning to properties of the **loss landscape** (flatness/sharpness) and uses sharpness-aware optimization as a mitigation angle. ([ACL Anthology][4])

So the *toy task* is ancient and tiny, but the **mechanism** (“sequential gradient updates on shared parameters cause interference”) absolutely still matters for LLMs the moment you ask them to **keep learning over time**.

### The clean mental model

* **Catastrophic forgetting** = a problem you get when you keep **updating weights** over time without preserving old constraints (via replay, regularization, modularization, etc.).
* **Agent amnesia / drift** = a problem you get even with **frozen weights**, because inference is stateless and you’re reconstructing intent from text each step.

And yep — that maps straight onto the “continual learning vs stateful working memory” split we were talking about.



[1]: https://www.andywills.info/hbab/mccloskeycohen.pdf "doi:10.1016/S0079-7421(08)60536-8"
[2]: https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/6/60429/files/2018/07/psychrev90a-1jt2c34.pdf "rev97020285.tif"
[3]: https://arxiv.org/abs/2308.08747 "[2308.08747] An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning"
[4]: https://aclanthology.org/2024.findings-emnlp.249/ "Revisiting Catastrophic Forgetting in Large Language Model Tuning - ACL Anthology"

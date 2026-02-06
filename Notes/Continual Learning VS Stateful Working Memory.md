---
title: "RL Scaling to AGI"
source: "https://chatgpt.com/c/69826128-d110-8322-bb27-e660b87c6a2e"
author:
  - "[[ChatGPT]]"
published:
created: 2026-02-03
description: "ChatGPT is your AI chatbot for everyday use. Chat with the most advanced AI to explore ideas, solve problems, and learn faster."
tags:
  - "clippings"
---
What's going on here? --- title: "Thread by @JasonBotterill" source: "https://x.com/SirMrMeowmeow/status/2017801328928661527" author: - "\[\[@JasonBotterill\]\]" published: 2026-01-31 created: 2026-02-03 description: "licklider escher faraday" tags: - "clippings" --- \*\*JB\*\* @JasonBotterill \[2026-02-01\](https://x.com/JasonBotterill/status/2017797905961341095) After watching Jerry’s post-OpenAI interviews it seems he genuinely believed you could just scale RL to AGI a year ago, but something changed and now he believes static weights will ever be enough and no one is investing serious amounts of compute in trying continual learning --- \*\*Sir Mr Meow Meow\*\* @SirMrMeowmeow \[2026-02-01\](https://x.com/SirMrMeowmeow/status/2017801328928661527) there are several problems with "just scale RL to AGI" imo the biggest is you can RL for verifiable domains like Math & Coding but even with heavy RL, agents are extremely amnesiac. They do not hold intent over time, // drifty af They are frozen models that re-read the chat history repeatedly to then re-infer whatever they are doing... which has very strong Telephone game effects. For some tasks that are highly decomposable or symbollic that's not such a problem, they may have enough context in view at any given time to figure out the task. However semantic meaning for different timescales is invariant.:/:x He's hoping "well maybe if they learn they would remember what they are doing and be better agents". However i would claim even if continuous learning were solved & perfect you would only have episode like memories & optimization of tacit information & skills (so esp like muscle memory or learning the feel for a coding project or tactics ect )--which would def be helpful but doesn't really fix the inference to inference amnesia adequately. The models are missing a kind of working memory or state control. Every inference is basically a blank slate rereading its journal. --- \*\*JB\*\* @JasonBotterill \[2026-02-01\](https://x.com/JasonBotterill/status/2017803315736699116) Like the current approach definitely has ton of room but once released, if it has some hiccup in how it approaches a problem that it needs to step over, it’s gonna keep experiencing that small hiccup every time you use it. Which is annoying even if it ultimately solves the --- \*\*Sir Mr Meow Meow\*\* @SirMrMeowmeow \[2026-02-01\](https://x.com/SirMrMeowmeow/status/2017801809549791255) i think once we have some good Cont Learning stuff + Stateful inference stuff (like a side car for getting at least some latent context to survive or a latent ladder) we will be in business. Also prob world model stuffs might help with complex visual things. --- \*\*Chargeblast\*\* @chargeblast Buying alerts from processors? You're giving them a front-row seat to your dispute data.

2/2

---

---

You’ve basically got **two different things mashed into one clip**:

## 1) The weird --- blocks are just metadata

That top chunk is **YAML front matter** —a common format in Markdown note apps / clippers. It’s not part of the thread; it’s bookkeeping:

- `title`, `source`, `author` = where the clip came from
- `published` = when the thread (or the first post in it) was posted
- `created` = when *you* saved the clip
- `tags: clippings` = “this is an imported clip” bucket
- `description: "licklider escher faraday"` = just a freeform description/keywords field (could be something your clipper generated, or something you typed earlier)

So: **formatting from a clipping tool**, not “mystery code.”

## 2) The actual thread is a debate about why “scale RL → AGI” might not work

In the thread, Jason Botterill is reacting to interviews by Jerry Tworek (who recently left OpenAI) on Unsupervised Learning with Jacob Effron. Those interviews/recaps frame his updated view roughly like: scaling pretraining + RL keeps giving improvements, but *static* (non-updating) models hit walls and can get “stuck,” so **continual learning** (systems that can update based on failure) is necessary for AGI-level behavior.

### What each person is saying in plain English

- **JB’s point:**  
	“A year ago Jerry sounded like: just keep scaling RL and we’ll get to AGI. Now he’s saying: static weights won’t be enough; we need continual learning — but labs aren’t really pouring massive compute into that.”  
	(Also: the tweet you pasted reads a bit like it’s missing a “not/never” — because the interview summaries say he thinks *static models can’t* be enough.)
- **Sir Mr Meow Meow ’s reply:**  
	“Even if RL makes models great at stuff with a clear ‘right answer’ (math/coding), agents still feel **amnesiac**. They don’t hold intent over time because each model call is basically stateless — you keep re-feeding chat history, and it drifts like a telephone game.”
	He’s drawing a line between two problems:
	- **Learning over weeks/months** (continual learning / updating the model)
	- **Remembering over minutes/hours** (working memory / state across steps)
	And he’s saying: continual learning might help long-term skill/tacit knowledge, but it **doesn’t automatically solve** the “every inference is a blank slate” problem.
- **JB follow-up about the “hiccup”:**  
	That’s the practical product version of “static weights”: if the model has a consistent failure mode, it’ll keep having it until someone retrains/patches it. It won’t self-correct just because you used it more.
- **Sir Mr Meow Meow’s proposed fix:**  
	“We need *both* continual learning *and* **stateful inference** (some mechanism so useful internal context survives across steps), plus maybe better ‘world model’ style components for complex perception/planning.”

## The key idea they’re circling

They’re arguing about **what’s missing for long-horizon ‘agent’ behavior**:

- RL scaling can make a model *better at a graded task*.
- Continual learning would let it *update after deployment*.
- But if the system **can’t maintain state/intent moment-to-moment**, you still get that “smart but forgetful intern” vibe: brilliant per-message, unreliable over long projects.

## Why that “Chargeblast” line is there

That last line:

> “Buying alerts from processors? You're giving them a front-row seat to your dispute data.”

…is almost certainly **not part of this AI thread**. It’s from Chargeblast and reads like a standalone viral marketing tweet about chargeback/dispute-alert products and data exposure. Your clipping tool probably sucked in an unrelated post that was adjacent in the feed (or a promoted item).

If you tell me what you clipped this with (Readwise/Omnivore/Obsidian plugin/etc.), I can usually tell you *exactly* why that extra tweet got appended — but functionally: **it’s a stray.**

---

Sorry, yeah, just explain to me the big ideas and the context around the key players and their thinking and peoples thinking about that thinking as outlined in the thread

---

---

Here’s the “big picture” of what’s happening in that thread: it’s a small window into a bigger argument in AI right now about **what kind of “memory + learning” you need to get from clever chatbots to robust, long-horizon agents**.

## The cast and why they matter

- **Jerry Tworek** — recently left **OpenAI**, and in interviews has been emphasizing (a) RL + scaling works but has limits, and (b) **continual learning** is a missing ingredient for AGI (models get “stuck” and don’t recover).
- **Jacob Effron** — host of **Unsupervised Learning with Jacob Effron** (a **Redpoint Ventures** show) who interviewed Jerry; his post and the episode description are the “source material” Jason is reacting to.
- **Jason Botterill** — in your pasted thread, he’s a *commentator* reading Jerry’s interviews as a meaningful shift: from “scale RL to AGI” vibes → toward “static weights won’t be enough; continual learning needed.”
- **Sir Mr Meow Meow** — pushes back: even if continual learning is solved, that doesn’t fix “agent amnesia” unless you also solve **stateful inference / working memory**.

That last pair is basically “AI Twitter doing peer review in public”: one person summarizing an influential researcher’s updated view, and another person saying “you’re mixing up two different problems.”

## The core idea: they’re arguing about two kinds of memory

A useful way to decode the thread is to separate **learning** from **state**:

Most confusion in these debates comes from people saying “memory” but meaning different things.

## What Jerry (as summarized in those interviews) is arguing

From the podcast description + the host’s writeup, the key points attributed to Jerry look like:

1. **Scaling pretraining + RL gives predictable gains… but hits constraints.**  
	The episode summary frames this as: scaling delivers improvements, but is constrained by data availability and generalization beyond the training objective.  
	This is consistent with older reporting on his “reasoning” work: he describes scaling as multi-dimensional and focused on training models to deliberate/explore more during problem solving (the “o1” era).
2. **“Static weights” systems get stuck — and stay stuck.**  
	The podcast description explicitly says he now believes **continual learning** is necessary for AGI because current models hit walls and become “hopeless” when stuck.  
	The LinkedIn writeup repeats the same theme: “static models can never achieve AGI,” and the “intelligence finds a way” line is used to contrast humans vs current models.
3. **Why he left OpenAI is framed as: go chase the next paradigm.**  
	The Verge’s blurb says he left to explore research “hard to do at OpenAI.”  
	The LinkedIn writeup also frames it as wanting freedom to explore what’s missing in how the world trains models.

So: Jerry’s “thinking” (as relayed here) is basically **“today’s recipe is powerful but brittle; we need systems that can update when they fail.”**

## Jason’s interpretation of Jerry’s thinking

In your excerpt, Jason does two things:

- **He reads Jerry’s stance as a shift over ~a year.**  
	Whether or not Jerry literally said “just scale RL to AGI,” Jason’s point is about perceived momentum: the “RL can carry us” optimism → replaced by “nope, without continual learning we won’t get there.”
- **He translates it into a product complaint:**  
	“If the model has a hiccup, it will keep having that hiccup every time.”  
	That’s a very practical restatement of the “static weights” critique: if nothing about the system truly updates from the failure, you’re doomed to rediscover the same pothole over and over.

He also claims “no one is investing serious compute” into continual learning. That’s a *field-level critique* (incentives + research difficulty), not really a claim about Jerry specifically.

## Sir Mr Meow Meow’s counter-argument

He’s not really saying Jerry is *wrong* — he’s saying **continual learning isn’t the main bottleneck for the behavior people want from agents.**

His core claims in the thread:

1. **RL shines where feedback is verifiable (math/coding),** but it doesn’t automatically create robust agents.
2. **Agents are “amnesiac”** because each model call is effectively stateless: it re-reads the “journal” (chat history / notes) and re-infers what it’s doing.
3. This re-inference creates **drift / telephone-game effects**: tiny misinterpretations compound across steps.
4. Even “perfect” continual learning mostly gives:
	- better skills (“muscle memory,” tacit competence)
	- maybe episodic memories  
		…but **doesn’t fix working-memory / intent-holding** unless inference itself becomes stateful.

So his suggested missing ingredient is **“working memory or state control”** —some mechanism that keeps latent context alive across steps, rather than reconstructing it from text every time.

This is the key disagreement:

- Jerry (as summarized): “Models don’t revise beliefs from failure; continual learning is necessary.”
- SirMrMeowMeow: “Even if you revise beliefs over time, you still need a *stateful* mechanism to hold intent and reduce drift right now.”

These can both be true. They’re just aimed at different failure modes.

## “People thinking about that thinking”

You actually get a clean example of the meta-discussion in the LinkedIn comments on the interview writeup:

- One commenter says the “static models getting stuck” point is underrated, and compares it to orgs where assumptions stop updating and teams mistake iteration for progress.
- Another commenter pushes back a bit: humans also hit walls sometimes (philosophy questions, etc.).

That’s exactly the vibe of “thinking about the thinking”:

- Some folks treat “stuckness” as *the* signature gap between current models and “real intelligence.”
- Others argue “getting stuck” isn’t unique to models, so it can’t be the whole definition of AGI.

## Why this debate keeps coming up right now

A little context that makes the thread make sense:

- **Reasoning models + agent scaffolding exposed the long-horizon cracks.**  
	When you use a model for single-shot Q&A, statelessness is mostly fine.  
	When you use it as an agent over 100+ steps (coding projects, research, operating a browser), you suddenly care a lot about:
	- holding goals steady
	- not drifting
	- recovering from local failures  
		Jerry’s public work history is deeply tied to reasoning + code/agent deployments (o1/o3, Codex, “ChatGPT Agent”), so it makes sense his “AGI bottleneck” intuitions would be informed by those exact pain points.
- **Incentives matter.**  
	It’s easier to justify compute on training runs that improve standard benchmarks and ship to users than it is to justify risky continual-learning systems that could:
	- degrade unpredictably
	- be attacked/poisoned
	- cause reproducibility + safety nightmares  
		(This is the subtext behind Jason’s “no one is investing serious compute” jab.)

## A clean way to reconcile the thread: three stacked layers

If you want one “map” that fits everyone’s point:

1. **Skill acquisition (RL / post-training):** gets you better at tasks you can reward.
2. **Stateful execution (working memory / state control):** makes agents stable across long plans.
3. **Continual learning (updating from failure over time):** lets the whole system improve instead of repeating bugs forever.

Jerry is emphasizing #3 as “AGI-critical.”  
SirMrMeowMeow is emphasizing #2 as “agent-critical.”  
Jason is reacting to the strategic implication: “if #3 is required, why aren’t we seeing huge investment there?”

If you want, I can also translate those into **concrete engineering approaches people mean** (e.g., what “stateful inference sidecar” could look like in practice, versus what “continual learning” would look like without turning into a security nightmare).
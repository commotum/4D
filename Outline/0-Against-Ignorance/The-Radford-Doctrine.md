# The Radford Doctrine

It's simple really. Build for the task that eats the rest and the rest will take care of itself.

Now, I promise I'll explain more shortly, but we have to start by ....


You see, I've always been frustrated by the way people try to explain the divergence of language modeling from reinforcement learning. It's been phrased as the war against supervised methods. An unwinnable war, truly. Because creating labeled data was expensive. 

I refuse to accept that. 

Misconception about knowledge vs anti-knowledge

present an alternative history of 






Now, I'm not saying that Sutton is wrong, exactly. Just, he might've been a bit early. 


We refer to this principle as the Radford Doctrine: if you build for the task that eats the rest, then the rest will take care of itself. 


We refer to this principle as the Radford Doctrine: if you build for the task that eats the rest, then the rest will take care of themselves.

Broad objectives eliminate narrow architectures.

Build for the task that eats the rest, and the rest takes care of itself.

multitask learning favors objectives that make tasks look alike—pick a task that eats the rest.

Generality comes from collapsing tasks, not enumerating them—pick a task that eats the rest.



Got it. You want the setup to *land* on your slogan and make it feel inevitable: **don’t bolt crutches onto the architecture for each task; instead, choose an objective that subsumes tasks** — *pick a task that eats the rest.*

Here are **10 options** you can drop in near the start of the paper (varying from academic to bloggy), all explicitly including the line.

---

## 1) Clean, paper-safe

Sutton is directionally correct that hard-coded design-time choices can hobble generality, but the culprit isn’t “human knowledge”—it’s freezing arbitrary interface decisions (fixed vocabularies, fixed attention patterns, fixed positional contracts) and then mistaking them for neutral defaults. Our claim is simple: if you want an architecture to outlive a single benchmark, stop adding task-specific crutches and **pick a task that eats the rest**.

---

## 2) Interface-tax framing

The modern transformer inherits a translation tax: each new domain must be contorted to fit fixed tokens, fixed windows, and fixed interaction rules, so progress accumulates as patches rather than principles. That tax is invisible in single-task work, where the interface can be amortized, and brutal in multitask settings, where mismatched assumptions collide. The escape hatch is not more patchwork—it is **to pick a task that eats the rest**.

---

## 3) “Not knowledge, commitment” framing

The mistake isn’t using priors; it’s committing to the wrong ones. A one-hot vocabulary denies structure, a fixed attention pattern denies adaptive interaction, and both are strong assumptions disguised as “defaults.” Instead of engineering the model around each task, we should engineer a single objective that makes tasks look alike—i.e., **pick a task that eats the rest**.

---

## 4) Longevity / staying-power framing

Architectures optimized for a paper win accumulate crutches: task-specific tokenizations, hand-tuned attention masks, domain-specific positional hacks. Architectures built to last remove arbitrary commitments and force reuse through a shared training objective. If you want staying power, don’t specialize the interface—**pick a task that eats the rest**.

---

## 5) “Crutches vs contracts” framing

Most “improvements” to general models quietly hard-code a contract: what counts as a token, which interactions are allowed, what geometry position must obey. Those contracts help in a narrow regime, then become liabilities when tasks must share a model. The alternative is to stop hand-authoring contracts per domain and **pick a task that eats the rest**.

---

## 6) Compact, quotable, still clear

Single-task engineering rewards architectural crutches; multitask learning punishes them. Fixed vocabularies, fixed windows, and fixed attention patterns aren’t neutral—they’re brittle commitments. If the goal is a generalist, the strategy is simple: don’t build a new interface per task—**pick a task that eats the rest**.

---

## 7) “Don’t adapt the model; adapt the task” framing

The deepest lesson isn’t “avoid knowledge,” it’s “avoid frozen guesses.” When you hard-code how the model represents and routes information, you force every new domain to pay an interface conversion cost. The Radford-style alternative is to unify behavior under one broad objective: don’t adapt the model to tasks—**pick a task that eats the rest**.

---

## 8) “Arbitrary defaults are anti-knowledge” framing

One-hot IDs and fixed attention masks are often treated as modest baselines, but they are anti-knowledge priors: they erase structure and pre-commit interaction patterns the world does not guarantee. You can patch around that for one dataset; you can’t scale it across domains. For generality, the right move is not another patch—it is **to pick a task that eats the rest**.

---

## 9) “General methods vs general objectives” framing

“General methods” are often interpreted as “no domain knowledge,” but that misses the practical failure mode: arbitrary design-time structure that locks in how the model must see and compute. The goal is not to remove structure; it is to make structure learnable under a unified objective. In short: stop customizing the architecture per task and **pick a task that eats the rest**.

---

## 10) Slightly spicy, blog-to-paper bridge

If you want to publish, give the architecture crutches; if you want it to last, take the crutches away. The hard part isn’t sprinkling clever inductive bias onto one benchmark—it’s building an interface that survives task diversity without constant redesign. The doctrine we follow is blunt: don’t build a model for each task—**pick a task that eats the rest**.

---

If you tell me which abstract you’re leaning toward (your Abstract 1 feels closest), I can rewrite the **first paragraph + last paragraph of the intro** so it cleanly connects:

**(Triumvirate constraints → translation tax → single-task illusion → Radford Doctrine → MonSTER interface proposal)**

…and it’ll read like a paper, not a manifesto.




You're still not quite seeing what I'm saying. Sutton's argument is too broad and is actually inaccurate if one analyzes it at a deeper level. However, he is directionally correct, that when we fix the architecture ahead of time we hobble the models.

He's still framing it as, "expert systems," vs, "general methods," and therefore argues that human knowledge is a weakness. This is wrong. The correct view isn't "we shouldn't give it our knowledge," or give the architecture a head start by sharing what we know, the correct view is, "we need to remove the fixed parts of the architecture that are arbitrary, that exist because we didn't know what we were doing, or were too tired to figure it out." One hot encoding for symbols and text doesn't work because the input is not structural or based on the real world or knowledge. In fact one-hot-symbolic-encoding is exactly the opposite of baking in knowledge. It's baking in ignorance. It's forcing structure that is meaningless at best, and destructive or interferential as worst.

Do you see what I'm saying? So forcing a model to use a pre-determined attention method, like  fixing the attention window to be full self-attention for all tokens, or as fixing it to be sliding window, is just as bad as forcing the model to represent things as one-hot-encoding IDs. 

Do you think you could articulate this idea succinctly?


Sutton is directionally correct that generality can be hobbled by hard-coded design time architectural decisions. 


However, he  can hobble generality, but the problem isn’t “giving the system knowledge.” The problem is freezing arbitrary design choices—constraints that exist because we didn’t know better, not because the world demands them. A one-hot ID encoding isn’t “neutral”; it’s a strong prior that denies structure (it bakes in ignorance by making all symbols equally unrelated). Likewise, forcing a fixed attention pattern (full, sliding, etc.) is a computational one-hot: it hard-codes which interactions are allowed instead of letting the model learn what should interact and when. The goal isn’t to remove priors; it’s to remove arbitrary priors and let the system learn both its representations and its compute routing (what to compare/attend/retrieve) from experience.

If you want it even tighter (one sentence):

Don’t confuse “using knowledge” with “hard-coding arbitrary structure”: one-hot symbols and fixed attention are not humble defaults, they’re anti-knowledge constraints that force the model to waste capacity relearning what should have been learnable or adaptable in the first place.



The goal isn’t to remove priors; it’s to remove arbitrary priors and let the system learn both its representations and its compute routing (what to compare/attend/retrieve) from experience.

One-hot symbols and fixed attention are not humble defaults, they’re anti-knowledge constraints that force the model to waste capacity 

No causal intervention: the model’s outputs don’t causally shape the data stream it learns from (during pretraining).
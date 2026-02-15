Yes — RL has a *very* similar “removal of arbitrary constants” storyline. It just plays out along slightly different axes than language modeling, because RL is constrained by **interaction**, **credit assignment**, **exploration**, and **time** rather than a clean supervised corpus + next-token objective.

Here’s a clean, parallel version you can drop into your “fixed vs learned” framing.

## The History of Reinforcement Learning as the Removal of Arbitrary Constants

### 1) Sutton (1988): removed the “must wait for the full return” constant

**Arbitrary constant:** you can only learn from *complete* outcomes (Monte Carlo: wait until episode ends).
**What became learnable:** *online prediction/control* via bootstrapping.

Temporal-Difference (TD) learning updates predictions using other predictions, enabling incremental learning before an episode terminates. That removes a hard episodic constraint and makes learning naturally “streamable.” ([Springer][1])

---

### 2) Watkins (1989 → 1992): removed the “must learn on-policy” constant

**Arbitrary constant:** to learn values, you must follow the same policy you’re evaluating/improving.
**What became learnable:** *off-policy control* — learn optimal behavior while behaving differently (e.g., exploratory).

Watkins’ Q-learning is framed as learning control from delayed rewards; later writeups (e.g., Watkins 1992) explicitly reference the 1989 thesis and convergence proof sketch. ([cs.rhul.ac.uk][2])

---

### 3) Williams (1992): removed the “need differentiable dynamics / differentiable argmax” constant

**Arbitrary constant:** you can only optimize policies when the environment is differentiable (or when you can backprop through dynamics / planning decisions).
**What became learnable:** direct optimization of *stochastic* policies via likelihood-ratio (“score function”) gradients.

REINFORCE is the canonical point here: it gives a general gradient-following approach for stochastic units, without requiring differentiable environment dynamics. ([Springer][3])

*(This is one of the cleanest “we removed a hard requirement” moments in RL: the world stays non-differentiable, but the learning signal becomes differentiable in expectation.)*

---

### 4) Sutton (1990–1991): removed the “planning and learning are separate modules” constant

**Arbitrary constant:** either you (a) do model-free learning from real experience, *or* (b) do planning with a perfect simulator/model — but they’re separate systems.
**What became learnable:** an *integrated* loop where a learned model can generate hypothetical experience for planning updates.

Dyna explicitly frames itself as integrating learning, planning, and reacting; you alternate updates from real experience and simulated experience from the learned model. ([ACM Digital Library][4])

---

### 5) Sutton, Precup, Singh (1999): removed the “1 timestep = 1 action forever” constant

**Arbitrary constant:** the only unit of behavior is the environment’s primitive action at every step.
**What became learnable:** *temporal abstractions* (options / macro-actions) with initiation sets + policies + termination conditions.

The options framework is the classic formalization of learning and planning with temporally extended actions (semi-MDP view). This is literally “learn the time scale” rather than fixing it by hand. ([ScienceDirect][5])

---

### 6) Deep RL (DQN, 2015): removed the “features must be hand-designed” constant

**Arbitrary constant:** RL only works well when humans provide good state features (low-dimensional, fully observed, or carefully engineered).
**What became learnable:** the representation itself — policies/values directly from high-dimensional inputs (pixels).

DQN is famous precisely because it learns successful policies “directly from high-dimensional sensory inputs” with end-to-end deep RL. ([Nature][6])

---

### 7) Self-play + search (AlphaGo Zero / AlphaZero, 2017): removed the “human gameplay data is required” constant

**Arbitrary constant:** strong performance in complex games requires expert demonstrations and human-curated priors beyond rules.
**What became learnable:** the training curriculum itself (self-play as the data generator) + the policy/value as the evaluator for search.

AlphaGo Zero is explicitly framed as learning “without human data” beyond the rules. AlphaZero generalizes the approach across games. ([Nature][7])

---

### 8) MuZero (2019–2020): removed the “must know the environment’s dynamics (or rules) to plan” constant

**Arbitrary constant:** planning systems require an explicit, correct simulator/rules engine.
**What became learnable:** a planning-sufficient latent model that supports lookahead without needing true dynamics.

MuZero is explicitly described as achieving strong performance “without any knowledge of [the] underlying dynamics,” by combining search with a learned model. ([Nature][8])

---

## Two “later chapters” that match your thesis perfectly

These are especially nice if you want to argue that RL is also converging on “remove fixed constraints; let learning decide.”

### A) Goal-conditioned / multi-task value functions (UVFA, 2015)

**Arbitrary constant:** one value function per task/goal.
**What became learnable:** generalization over goals inside a single approximator (condition on goal). ([Proceedings of Machine Learning Research][9])

### B) Offline RL (BCQ 2018/2019; CQL 2020)

**Arbitrary constant:** you *must* keep interacting online to learn.
**What became learnable:** policies from a fixed dataset (with explicit mechanisms to handle distribution shift / overestimation). ([arXiv][10])

---

## The pattern in RL, stated in the same “who decides what matters?” language

Across these shifts, RL progress repeatedly comes from **stopping the human from hard-coding**:

* *when* learning can happen (end of episode → online TD) ([Springer][1])
* *which* policy must generate data (on-policy → off-policy) ([cs.rhul.ac.uk][2])
* *whether* you need differentiable worlds (score-function gradients) ([Springer][3])
* *how* planning relates to learning (Dyna, then later MuZero) ([ACM Digital Library][4])
* *what* the time scale of behavior is (options) ([ScienceDirect][5])
* *what* representation “counts” as state (deep RL from pixels) ([Nature][6])
* *where* data comes from (self-play) ([Nature][7])
* *whether* interaction is required at all (offline RL) ([NeurIPS Proceedings][11])

So yes: RL has the same “remove arbitrary constants → unlock scale + generality” rhythm.

---

## One important difference vs language modeling

In your LM timeline, the objective “next token” stays constant while the interface/memory/representation changes.

In RL, the objective is always “maximize expected return,” but what *counts* as the return signal is often sparse/delayed and the data distribution is *controlled by the agent*. That’s why RL’s “constants” are often about:

* credit assignment over time (TD),
* exploration and data generation (self-play),
* how much of the world model must be known vs learned (Dyna/MuZero),
* what time scale actions live on (options).

That difference actually strengthens your framing: **RL makes the “fixed vs learned” debate unavoidable** because “the data stream” is itself endogenously produced.


[1]: https://link.springer.com/article/10.1007/BF00115009?utm_source=chatgpt.com "Learning to predict by the methods of temporal differences"
[2]: https://www.cs.rhul.ac.uk/~chrisw/thesis.html?utm_source=chatgpt.com "PhD Thesis: Learning from Delayed Rewards"
[3]: https://link.springer.com/article/10.1007/BF00992696?utm_source=chatgpt.com "Simple statistical gradient-following algorithms for ..."
[4]: https://dl.acm.org/doi/abs/10.1145/122344.122377?utm_source=chatgpt.com "Dyna, an integrated architecture for learning, planning, and ..."
[5]: https://www.sciencedirect.com/science/article/pii/S0004370299000521?utm_source=chatgpt.com "A framework for temporal abstraction in reinforcement ..."
[6]: https://www.nature.com/articles/nature14236?utm_source=chatgpt.com "Human-level control through deep reinforcement learning"
[7]: https://www.nature.com/articles/nature24270?utm_source=chatgpt.com "Mastering the game of Go without human knowledge"
[8]: https://www.nature.com/articles/s41586-020-03051-4?utm_source=chatgpt.com "Mastering Atari, Go, chess and shogi by planning with a ..."
[9]: https://proceedings.mlr.press/v37/schaul15.pdf?utm_source=chatgpt.com "Universal Value Function Approximators"
[10]: https://arxiv.org/abs/1812.02900?utm_source=chatgpt.com "Off-Policy Deep Reinforcement Learning without Exploration"
[11]: https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"

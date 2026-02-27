I’ll pull authoritative publication lists (Sutton’s CV/homepage, Google Scholar, DBLP/Semantic Scholar) and merge them into one deduped list, then mark the “most significant” works (highly cited/award-winning/field-shaping) with quick notes and links.

“Every significant paper” is a judgment call (Sutton has **hundreds** of publications). So I treated *significant* as: papers that (a) introduced a named idea/algorithm/framework that became standard in RL, (b) are canonical references in RL courses/texts, or (c) are major theoretical/architectural advances (especially around off-policy learning + function approximation). I cross-checked against Sutton’s DBLP bibliography. [DBLP](https://dblp.org/pid/48/6070)

Below is a **curated, high-signal list** (chronological-ish, grouped by era). If you want the **complete, unfiltered bibliography** (literally everything), DBLP is the best single index to start from. [DBLP](https://dblp.org/pid/48/6070)

---

## 1980s: early RL + temporal-difference learning

- **1983** — *Neuronlike adaptive elements that can solve difficult learning control problems* (Barto, Sutton, Anderson), **IEEE Trans. Systems, Man, and Cybernetics**. [DBLP](https://dblp.org/pid/48/6070)
- **1985** — *Training and Tracking in Robotics* (Selfridge, Sutton, Barto), **IJCAI**. [DBLP](https://dblp.org/pid/48/6070)
- **1988** — *Learning to Predict by the Methods of Temporal Differences* (Sutton), **Machine Learning**. *(Foundational TD learning paper.)* [Incomplete Ideas +1](https://www.incompleteideas.net/papers/sutton-88-scan.pdf?utm_source=chatgpt.com)
- **1989** — *Sequential Decision Probelms and Neural Networks* (Barto, Sutton, Watkins), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)

---

## 1990s: Dyna-style integration, generalization, eligibility traces, options, policy gradients

### Dyna / integrated learning + planning

- **1990** — *Integrated Modeling and Control Based on Reinforcement Learning* (Sutton), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **1990** — *Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming* (Sutton), **Machine Learning / ML**. [DBLP](https://dblp.org/pid/48/6070)
- **1991** — *Planning by Incremental Dynamic Programming* (Sutton), **ML 1991**. [DBLP](https://dblp.org/pid/48/6070)
- (Closely related Dyna thread) **1991** — *Dyna, an Integrated Architecture for Learning, Planning, and Reacting* (Sutton), **SIGART Bulletin**. [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/122344.122377?utm_source=chatgpt.com)

### Representation + prediction

- **1995** — *Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding* (Sutton), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **1995** — *TD Models: Modeling the World at a Mixture of Time Scales* (Sutton), **ICML**. [DBLP](https://dblp.org/pid/48/6070)

### Eligibility traces / foundations of modern TD control variants

- **1996** — *Reinforcement Learning with Replacing Eligibility Traces* (Singh, Sutton), **Machine Learning**. [DBLP](https://dblp.org/pid/48/6070)

### Temporal abstraction: the Options framework (hierarchical RL cornerstone)

- **1997** — *Multi-time Models for Temporally Abstract Planning* (Precup, Sutton), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **1998** — *Theoretical Results on Reinforcement Learning with Temporally Abstract Options* (Precup, Sutton, Singh), **ECML**. [DBLP](https://dblp.org/pid/48/6070)
- **1998** — *Intra-Option Learning about Temporally Abstract Actions* (Sutton, Precup, Singh), **ICML**. [DBLP](https://dblp.org/pid/48/6070)
- **1998** — *Improved Switching among Temporally Abstract Actions* (Sutton, Singh, Precup, Ravindran), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **1999** — *Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning* (Sutton, Precup, Singh), **Artificial Intelligence (journal)**. *(The classic “Options” paper.)* [DBLP +1](https://dblp.org/pid/48/6070)

### Policy gradients / actor-critic lineage

- **1999** — *Policy Gradient Methods for Reinforcement Learning with Function Approximation* (Sutton, McAllester, Singh, Mansour), **NeurIPS (NIPS)**. [DBLP +1](https://dblp.org/pid/48/6070)

### Big-picture theory framing

- **1999** — *Open Theoretical Questions in Reinforcement Learning* (Sutton), **EuroCOLT**. [DBLP](https://dblp.org/pid/48/6070)

---

## 2000s: off-policy + traces, TD networks, least-squares TD, natural actor-critic, stable off-policy learning

### Off-policy evaluation / off-policy TD with approximation

- **2000** — *Eligibility Traces for Off-Policy Policy Evaluation* (Precup, Sutton, Singh), **ICML**. [DBLP](https://dblp.org/pid/48/6070)
- **2001** — *Off-Policy Temporal Difference Learning with Function Approximation* (Precup, Sutton, Dasgupta), **ICML**. [DBLP](https://dblp.org/pid/48/6070)

### Keepaway soccer (widely used empirical testbed)

- **2000** — *Reinforcement Learning for 3 vs. 2 Keepaway* (Stone, Sutton, Singh), **RoboCup**. [DBLP](https://dblp.org/pid/48/6070)
- **2001** — *Keepaway Soccer: A Machine Learning Testbed* (Stone, Sutton), **RoboCup**. [DBLP](https://dblp.org/pid/48/6070)

### Temporal-Difference Networks (predictive representations thread)

- **2004** — *Temporal-Difference Networks* (Sutton, Tanner), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **2005** — *Temporal Abstraction in Temporal-difference Networks* (Sutton, Rafols, Koop), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **2005** — *TD(λ) networks: temporal-difference networks with eligibility traces* (Tanner, Sutton), **ICML**. [DBLP](https://dblp.org/pid/48/6070)
- **2005** — *Using Predictive Representations to Improve Generalization in Reinforcement Learning* (Rafols, Ring, Sutton, Tanner), **IJCAI**. [DBLP](https://dblp.org/pid/48/6070)

### Off-policy learning with options

- **2005** — *Off-policy Learning with Options and Recognizers* (Precup, Sutton, Paduraru, Koop, Singh), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)

### Least-squares TD / incremental methods

- **2006** — *Incremental Least-Squares Temporal Difference Learning* (Geramifard, Bowling, Sutton), **AAAI**. [DBLP](https://dblp.org/pid/48/6070)
- **2006** — *iLSTD: Eligibility Traces and Convergence Analysis* (Geramifard, Bowling, Zinkevich, Sutton), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)

### Natural actor-critic (policy gradient efficiency thread)

- **2007** — *Incremental Natural Actor-Critic Algorithms* (Bhatnagar, Sutton, Ghavamzadeh, Lee), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **2009** — *Natural actor-critic algorithms* (Bhatnagar, Sutton, Ghavamzadeh, Lee), **Automatica**. [DBLP](https://dblp.org/pid/48/6070)

### Stable off-policy TD with function approximation (GTD family begins)

- **2008** — *A Convergent O(n) Temporal-difference Algorithm for Off-policy Learning with Linear Function Approximation* (Sutton, Szepesvári, Maei), **NeurIPS (NIPS)**. [NeurIPS Proceedings +1](https://proceedings.neurips.cc/paper/2008/hash/e0c641195b27425bb056ac56f8953d24-Abstract.html?utm_source=chatgpt.com)
- **2009** — *Fast gradient-descent methods for temporal-difference learning with linear function approximation* (Sutton et al.), **ICML**. [DBLP +1](https://dblp.org/pid/48/6070)
- **2009** — *Convergent Temporal-Difference Learning with Arbitrary Smooth Function Approximation* (Maei et al., incl. Sutton), **NeurIPS (NIPS)**. [DBLP +1](https://dblp.org/pid/48/6070)

### Go search with TD (influential application + planning/search combo)

- **2007** — *Reinforcement Learning of Local Shape in the Game of Go* (Silver, Sutton, Müller), **IJCAI**. [DBLP](https://dblp.org/pid/48/6070)

---

## 2010s: gradient-TD extensions, Horde/GVFs, off-policy actor-critic, true-online + emphatic TD, importance sampling, variance/meta-learning

### Gradient-TD for prediction/control with traces

- **2010** — *Toward Off-Policy Learning Control with Function Approximation* (Maei, Szepesvári, Bhatnagar, Sutton), **ICML**. [DBLP +1](https://dblp.org/pid/48/6070)
- **2010** — *GQ(λ): A general gradient algorithm for temporal-difference prediction learning with eligibility traces* (Maei, Sutton), **AGI 2010**. [Incomplete Ideas +1](https://incompleteideas.net/papers/maei-sutton-10.pdf?utm_source=chatgpt.com)

### Lifelong / continual off-policy learning architecture

- **2011** — *Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction* (Sutton et al.), **AAMAS**. [cs.swarthmore.edu](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/horde1.pdf?utm_source=chatgpt.com)
- **2012** — *Scaling life-long off-policy learning* (White, Modayil, Sutton), **ICDL-EPIROB**. [DBLP](https://dblp.org/pid/48/6070)

### Off-policy actor-critic (policy gradient under off-policy data)

- **2012** — *Linear Off-Policy Actor-Critic* (Degris, White, Sutton), **ICML**. [DBLP](https://dblp.org/pid/48/6070)
- **2012** — *Off-Policy Actor-Critic* (Degris, White, Sutton), **CoRR**. [DBLP](https://dblp.org/pid/48/6070)

### TD search in Go (journal version)

- **2012** — *Temporal-difference search in computer Go* (Silver, Sutton, Müller), **Machine Learning (journal)**. [DBLP](https://dblp.org/pid/48/6070)

### Importance sampling for off-policy prediction

- **2014** — *Weighted importance sampling for off-policy learning with linear function approximation* (Mahmood, van Hasselt, Sutton), **NeurIPS (NIPS)**. [DBLP](https://dblp.org/pid/48/6070)
- **2015** — *Off-policy learning based on weighted importance sampling with linear computational complexity* (Mahmood, Sutton), **UAI**. [DBLP](https://dblp.org/pid/48/6070)

### True-online / emphatic TD (stability + correctness threads)

- **2016** — *True Online Temporal-Difference Learning* (van Seijen, Mahmood, Pilarski, Machado, Sutton), **JMLR**. [DBLP](https://dblp.org/pid/48/6070)
- **2016** — *An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning* (Sutton, Mahmood, White), **JMLR**. [DBLP](https://dblp.org/pid/48/6070)

### Actor-critic with nonlinear function approximation (later thread)

- **2017** — *Forward Actor-Critic for Nonlinear Function Approximation in Reinforcement Learning* (Veeriah, van Seijen, Sutton), **AAMAS**. [DBLP](https://dblp.org/pid/48/6070)

### Variance / multi-step control variates / meta step-sizes

- **2018** — *Comparing Direct and Indirect Temporal-Difference Methods for Estimating the Variance of the Return* (Sherstan et al., incl. Sutton), **UAI**. [DBLP](https://dblp.org/pid/48/6070)
- **2018** — *Per-decision Multi-step Temporal Difference Learning with Control Variates* (De Asis, Sutton), **UAI**. [DBLP](https://dblp.org/pid/48/6070)
- **2018** — *TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent* (Kearney et al., incl. Sutton), **CoRR**. [DBLP](https://dblp.org/pid/48/6070)

---

## 2020s: average-reward RL, options for planning, and recent convergence results

- **2021** — *Average-Reward Off-Policy Policy Evaluation with Function Approximation* (Zhang, Wan, Sutton, Whiteson), **ICML**. [DBLP](https://dblp.org/pid/48/6070)
- **2021** — *Average-Reward Learning and Planning with Options* (Wan, Naik, Sutton), **NeurIPS**. [DBLP](https://dblp.org/pid/48/6070)
- **2022** — *Toward Discovering Options that Achieve Faster Planning* (Wan, Sutton), **CoRR**. [DBLP](https://dblp.org/pid/48/6070)
- **2024** — *On Convergence of Average-Reward Q-Learning in Weakly Communicating Markov Decision Processes* (Wan, Yu, Sutton), **CoRR**. [DBLP](https://dblp.org/pid/48/6070)
- **2025** — *Swift-Sarsa: Fast and Robust Linear Control* (Javed, Sutton), **CoRR**. [DBLP](https://dblp.org/pid/48/6070)

---

## If you literally meant “everything he’s written”

Two practical ways to get the *complete* list (every workshop paper, tech report, journal, etc.):

1. **DBLP author page** (best structured + deduped for CS venues). [DBLP](https://dblp.org/pid/48/6070)
2. Sutton’s own “incompleteideas” publication pages (authoritative, but my attempt to fetch it returned an error at the moment). [Incomplete Ideas](https://incompleteideas.net/publications.html?utm_source=chatgpt.com)

---

If you tell me what you mean by **“significant”** (e.g., “top 25 by citations,” “anything that introduced a named algorithm,” “only journal + NeurIPS/ICML papers,” “only single-author,” etc.), I can tighten this into a cleaner short-list that matches your definition.

I’ll pull Sutton’s first-author papers and filter down to the handful that are widely treated as field-shaping “canonical” references, then give you a clean annotated list with links and citations.

Here’s a **tight “canonical, trajectory-shifting” shortlist** of Sutton papers **where he is first author** —the ones that introduced or decisively cemented ideas that became standard parts of RL.

I’m **not** including highly influential Sutton papers where he’s *not* first author (e.g., many core options/off-policy/trace papers led by Precup, Singh, Maei, van Seijen, etc.), even though some of those absolutely changed the field.

## The canonical first‑author Sutton papers

### 1) Temporal‑Difference learning becomes the backbone of RL prediction

**Sutton, R. S. (1988). *Learning to Predict by the Methods of Temporal Differences*. Machine Learning.**[Springer +1](https://link.springer.com/article/10.1007/BF00115009?utm_source=chatgpt.com)  
Why it changed the field: this is the paper that made **TD learning** a central paradigm—bootstrapping from other predictions rather than waiting for final outcomes. It’s hard to overstate how much of modern RL (value learning, TD control variants, actor‑critic value baselines, etc.) builds on this.

---

### 2) Model‑based RL as an integrated “learn + plan” loop (Dyna roots)

**Sutton, R. S. (1990). *Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming*. ICML 1990 (Machine Learning Proceedings).**[ScienceDirect +1](https://www.sciencedirect.com/science/chapter/edited-volume/abs/pii/B9781558601413500304?utm_source=chatgpt.com)  
Why it changed the field: this is one of the key early statements of what we now call **model‑based RL** as an *architecture*: learn a model from experience, then use it for planning / simulated experience.

---

### 3) The crisp “Dyna” statement everyone cites

**Sutton, R. S. (1991). *Dyna, an Integrated Architecture for Learning, Planning, and Reacting*. SIGART Bulletin.**[ACM Digital Library +1](https://dl.acm.org/doi/abs/10.1145/122344.122377?utm_source=chatgpt.com)  
Why it changed the field: a short, canonical articulation of **Dyna** that helped set the agenda for integrating real experience with simulated/planning updates.

(If you only keep one Dyna citation in your own work, it’s usually this one.)

---

### 4) Practical generalization that made RL work at scale (representation matters)

**Sutton, R. S. (1996). *Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding*. NeurIPS 1995 proceedings (published as NIPS 8).**[NeurIPS Papers +2 UT Austin Computer Science +2](https://papers.nips.cc/paper/1109-generalization-in-reinforcement-learning-successful-examples-using-sparse-coarse-coding?utm_source=chatgpt.com)  
Why it changed the field: this is a canonical reference for **tile coding / coarse coding** style representations—hugely influential in making RL with function approximation reliable and practical before deep RL (and still widely used for linear/online RL).

---

### 5) Temporal abstraction becomes a first‑class theory object (Options framework)

**Sutton, R. S., Precup, D., & Singh, S. (1999). *Between MDPs and Semi‑MDPs: A Framework for Temporal Abstraction in Reinforcement Learning*. Artificial Intelligence.**[ScienceDirect +1](https://www.sciencedirect.com/science/article/pii/S0004370299000521?utm_source=chatgpt.com)  
Why it changed the field: this is the **Options** framework paper—the reference that made temporally extended actions a clean, reusable building block for hierarchical RL and planning.

---

### 6) Policy gradients become a mainstream alternative to value‑based control

**Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (1999). *Policy Gradient Methods for Reinforcement Learning with Function Approximation*. NeurIPS 1999.**[NeurIPS Papers +1](https://papers.neurips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf?utm_source=chatgpt.com)  
Why it changed the field: it helped establish **policy gradient** methods (and the actor‑critic lineage) as principled, general tools—especially important once function approximation is unavoidable.

---

### 7) First stable off‑policy TD + linear function approximation (breakthrough on “the deadly triad”)

**Sutton, R. S., Szepesvári, C., & Maei, H. R. (2008). *A Convergent O(n) Temporal‑Difference Algorithm for Off‑policy Learning with Linear Function Approximation*. NeurIPS 2008.**[NeurIPS Proceedings +1](https://proceedings.neurips.cc/paper/2008/hash/e0c641195b27425bb056ac56f8953d24-Abstract.html?utm_source=chatgpt.com)  
Why it changed the field: a major theoretical/algorithmic milestone— **stable off‑policy TD learning** with linear function approximation and linear complexity.

---

### 8) The GTD family becomes usable/practical (widely cited “workhorse” paper)

**Sutton, R. S., Maei, H. R., Precup, D., Bhatnagar, S., Silver, D., Szepesvári, C., & Wiewiora, E. (2009). *Fast Gradient‑Descent Methods for Temporal‑Difference Learning with Linear Function Approximation*. ICML 2009.**[icml.cc +1](https://icml.cc/Conferences/2009/papers/546.pdf?utm_source=chatgpt.com)  
Why it changed the field: pushed the **gradient‑TD** approach into a more practical form and cemented the “stable TD with approximation + off‑policy data” research direction.

(If you’re tracing the modern off‑policy evaluation/control theory lineage, this is one of the anchor citations.)

---

### 9) Lifelong/off‑policy learning at scale: Horde and the GVF worldview

**Sutton, R. S., Modayil, J., Delp, M., Degris, T., Pilarski, P. M., White, A., & Precup, D. (2011). *Horde: a Scalable Real‑time Architecture for Learning Knowledge from Unsupervised Sensorimotor Interaction*. AAMAS 2011.**[DBLP +1](https://dblp.org/rec/conf/atal/SuttonMDDPWP11.html?utm_source=chatgpt.com)  
Why it changed the field: helped reframe RL agents as learning **many predictions/behaviors in parallel**, online, off‑policy, in real time—an architectural and conceptual shift tied to **general value functions** and continual learning.

---

### 10) Emphatic TD: a clean stabilization idea for off‑policy TD

**Sutton, R. S., Mahmood, A. R., & White, M. (2016). *An Emphatic Approach to the Problem of Off‑policy Temporal‑Difference Learning*. JMLR.**[Journal of Machine Learning Research +1](https://jmlr.org/papers/volume17/14-488/14-488.pdf?utm_source=chatgpt.com)  
Why it changed the field: introduced **emphasis weighting** as a principled mechanism to restore stability properties under off‑policy training—now a key piece of the theory toolkit around the deadly triad.

---

## “One notch more inclusive” (still first‑author, influential, but less universally cited as the turning points)

If you want to expand slightly while still staying canonical-adjacent:

- **Sutton, R. S. & Tanner, B. (2004). *Temporal‑Difference Networks*. NeurIPS 2004.**[NeurIPS Papers +1](https://papers.nips.cc/paper/2545-temporal-difference-networks?utm_source=chatgpt.com)
- **Sutton, R. S., Precup, D., & Singh, S. (1998). *Intra‑Option Learning about Temporally Abstract Actions*. ICML 1998.**[Incomplete Ideas +1](https://www.incompleteideas.net/papers/SPS-98-ICML.pdf?utm_source=chatgpt.com)
- **Sutton, R. S. (1995). *TD Models: Modeling the World at a Mixture of Time Scales*. ICML 1995.**[ScienceDirect +1](https://www.sciencedirect.com/science/article/abs/pii/B9781558603776500724?utm_source=chatgpt.com)

---

If you want, I can also produce this as:

- a **BibTeX block** for the core list, or
- a **ranked “top 5”** (the absolute minimum set), depending on whether you’re building a reading list or a citation backbone.
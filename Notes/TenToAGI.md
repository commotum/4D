Lex Fridman interviews John Carmack. At about 1:50 Carmack says:

"I am not a madman for saying that it is likely that the code for artificial general intelligence is going to be tens of thousands of line of code not millions of lines of code. This is code that conceivably one individual could write, unlike writing a new web browser or operating system and, based on the progress that AI as machine learning had made in the recent decade, it's likely that the important things that we don't know are relatively simple. There's probably a handful of things and my bet is I think there's less than six key insights that need to be made. Each one of them can probably be written on the back of an envelope. We don't know what they are, but when they're put together in concert with GPUs at scale and the data that we all have access to, that we can make something that behaves like a human being or like a living creature and that can then be educated in whatever ways that we need to get to the point where we can have universal remote works where anything that somebody does mediated by a computer and doesn't require physical interaction, that an AGI will be able to do."

He also believes that antecedents of all the critical ideas are already in the literature, but have been lost.

Pull together the top 10 candidates for "critical ideas" needed. They should pull from what we know about the brain, citing high quality research from various domains outside of computer science.

---

Here’s a list of 10 “candidate critical ideas” that look plausibly like the sort of compact, stackable insights Carmack is talking about, with their roots in neuroscience / cognitive science rather than CS proper.

Each one is something that (a) we see pretty clearly in the brain, (b) has at least a semi‑formal theory around it, and (c) could plausibly be coded as a relatively small module, assuming you already have some generic NN / GPU infrastructure.

---

## 1. Hierarchical predictive world models (predictive coding)

**Core idea:**
The cortex doesn’t just recognize patterns; it constantly *predicts* its sensory inputs and only propagates errors up the hierarchy. Higher areas send predictions down; lower areas send back residuals. Perception = inference in a generative model.

**Brain evidence**

* Rao & Ballard’s classic model of visual cortex: feedback connections carry predictions; feedforward connections carry prediction errors. When trained on natural images, simple-cell-like receptive fields and extra-classical surround effects emerge naturally. ([PubMed][1])
* Broader predictive coding reviews argue that this “prediction + error” scheme can explain much of cortical organization, across sensory domains. ([arXiv][2])

**Why this might be a “critical line of code”**
A compact, generic predictive-coding loop (predict → compare → send only errors) embedded at every layer could give you:

* Efficient representation
* Unsupervised learning from raw streams
* Natural handling of missing or noisy data

Modern deep models approximate this with autoregressive objectives, but most still treat prediction as a *training loss*, not the *online operating mode* of the whole system.

---

## 2. Active inference and allostatic control (action as prediction)

**Core idea:**
The brain doesn’t just passively minimize prediction error; it *acts* to make its predictions come true, especially predictions about its own internal state (“body budget”). This is allostasis: regulating by *anticipating* needs, not just reacting.

**Brain evidence**

* Friston’s “free energy principle” frames action, perception, and learning as jointly minimizing a single quantity (variational free energy), making control just another aspect of inference in a generative model. ([Nature][3])
* Allostasis work (Sterling) argues that efficient regulation requires anticipating needs and preparing to satisfy them before they arise, rather than returning to fixed setpoints after deviations. ([PubMed][4])
* Interoceptive predictive coding (Barrett & Simmons) models limbic regions predicting the internal bodily state; interoceptive experience reflects predictions constrained by visceral feedback, not just raw signals. ([PMC][5])
* Homeostatic reinforcement learning models show how reward signals can be derived from correcting homeostatic deviations, linking physiological stability and reward-seeking in a formal way. ([PMC][6])

**Why this might be a “critical line of code”**
An AGI that simply “maximizes reward” is underspecified and unstable. A compact active‑inference/homeostatic module—predict internal variables → choose actions that minimize expected deviation from desired setpoints—could:

* Turn vague “goals” into well-structured control problems
* Tie exploration and learning to internal uncertainty / risk rather than arbitrary reward shaping
* Provide a natural notion of “self‑preservation” and stability

---

## 3. Complementary learning systems and replay (fast + slow learning)

**Core idea:**
You need *two* learning systems: a fast, high‑plasticity system for specific experiences, and a slow, integrative system for abstract structure. The brain implements this with hippocampus (episodic, fast) and neocortex (semantic, slow) plus offline replay.

**Brain evidence**

* McClelland, McNaughton & O’Reilly’s “complementary learning systems” (CLS) explain why hippocampus supports rapid one‑shot learning while neocortex learns slowly from interleaved experience to avoid catastrophic interference. ([PubMed][7])
* Later CLS reviews flesh out how this division shows up across domains like vocabulary learning and generalization. ([ResearchGate][8])
* Hippocampal indexing theory: hippocampus stores an “index” pointing to patterns of neocortical activity; reactivating the index reinstates the cortical pattern, giving episodic recall. ([PubMed][9])
* Hippocampal replay (during rest or sleep) replays place-cell and event sequences; strong evidence this supports memory consolidation and planning. ([PMC][10])

**Why this might be a “critical line of code”**
A simple architectural rule—*fast episodic buffer + slow structural learner + replay*—addresses:

* Catastrophic forgetting
* Few-shot learning with long-term abstraction
* Integrating new knowledge without blowing up the old

Many deep systems have pieces of this (experience replay, memories), but usually as training hacks rather than as a principled dual‑system algorithm shaped by constraints of interference and consolidation.

---

## 4. Working memory, cognitive control, and a global workspace

**Core idea:**
The brain has a capacity-limited “workspace” that can hold a few items, selectively gate information in and out, and broadcast its contents widely, enabling flexible, serial reasoning and task control.

**Brain evidence**

* Baddeley’s multicomponent working-memory model (phonological loop, visuospatial sketchpad, central executive) emphasizes a small, actively maintained buffer with specialized subsystems. ([PubMed][11])
* Miller & Cohen’s integrative theory of prefrontal cortex: cognitive control arises from actively maintained PFC patterns that represent goals and bias processing elsewhere. ([PubMed][12])
* O’Reilly & Frank’s computational model of PFC–basal ganglia circuits shows how learned gating signals can control what enters/exits working memory, enabling complex sequential behavior without a “homunculus.” ([PubMed][13])
* Global Neuronal Workspace (Dehaene, Mashour): when a representation “ignites” a distributed fronto‑parietal network, it becomes globally accessible, supporting conscious report and flexible use. ([PubMed][14])

**Why this might be a “critical line of code”**
Transformers have attention, but they don’t really implement:

* A *small, discrete* workspace with hard gating
* Global broadcast tied to ignition/threshold events
* Learned policies over *what* to think about next

A few dozen lines implementing a gated working buffer + broadcast mechanism, plugged into an otherwise generic network, might be exactly the “System 2” bolt‑on Carmack is gesturing at.

---

## 5. Hierarchical action control and motor primitives (skills as chunks)

**Core idea:**
Behavior is organized hierarchically: low-level motor “synergies” combine into reusable chunks, which combine into extended action sequences and habits. The brain seems to implement this via spinal/motor primitives, basal ganglia, and hierarchical control loops.

**Brain evidence**

* Botvinick’s review of hierarchically organized behavior links psychological evidence of action hierarchies to neural models involving prefrontal cortex and basal ganglia. ([PMC][15])
* Balleine and colleagues highlight hierarchical goal‑directed action and action chunking, emphasizing the role of cortical–basal ganglia loops in building chained action sequences and habits. ([ScienceDirect][16])
* Work on motor primitives and synergies shows that spinal circuits implement reusable modules (synergies) that can be combined to generate complex movements. ([PMC][17])

**Why this might be a “critical line of code”**
A simple rule like “compress frequently co-occurring action subsequences into macro-actions, then plan over the macros” is very compact code, but:

* Dramatically speeds up planning
* Gives natural notions of “skills” and “habits”
* Matches real motor control structure

Hierarchical RL approximates this, but usually not with the strong structural biases (e.g., separate circuitry, distinct learning regimes) seen in the nervous system.

---

## 6. Neural syntax and compositional assemblies (variable-like binding)

**Core idea:**
The brain seems to compose “thoughts” by temporarily binding together cell assemblies into structured sequences, with oscillatory codes organizing multiple items in parallel (like a limited working memory list).

**Brain evidence**

* Hebb’s original “cell assembly” and “phase sequence” ideas (1949) proposed that transiently co‑active groups of neurons underlie perception, memory, and thought. ([PMC][18])
* Buzsáki’s “neural syntax” work argues that sequences of cell assemblies, coordinated by oscillations, provide a syntax‑like structure for neural information processing. ([PMC][19])
* Lisman & Jensen’s theta–gamma coding model: nested gamma cycles within a theta cycle can represent multiple ordered items in working memory. ([PMC][20])
* Hippocampal place‑cell sequences and replay encode spatial trajectories and prospective paths to goals, hinting at a general “sequence-of-assemblies” representation used in planning. ([PubMed][21])

**Why this might be a “critical line of code”**
A lightweight binding scheme—e.g., oscillatory “slots” or other temporal tagging for multiple tokens—could give a neural net:

* Variable‑like role/filler binding
* Short sequences in working memory without confusion
* Compositional mental programs built from a small library of assemblies

Most modern NNs lack an explicit, biologically plausible binding mechanism; they emulate it via sheer scale and training.

---

## 7. Causal, model-based learning with rich priors (core knowledge)

**Core idea:**
Humans (including very young children) don’t just memorize associations; they infer *causal* structure from very sparse data, drawing on built‑in “core knowledge” systems (objects, agents, number, space) and powerful Bayesian-like inference over structured models.

**Brain / cognitive evidence**

* Tenenbaum, Griffiths & Kemp: theory-based Bayesian models of cognition, where learning is inference over structured hypotheses (e.g., causal graphs, intuitive theories), explaining strong generalization from few examples. ([PubMed][22])
* Gopnik and Schulz show that children behave like little scientists, updating causal “theories” via evidence and interventions, consistent with causal Bayes nets. ([ScienceDirect][23])
* Schulz et al. demonstrate that preschoolers can infer different causal structures (chain vs common cause vs interaction) from patterns of interventions and outcomes. ([PubMed][24])
* Spelke & Kinzler’s core knowledge work identifies domain-specific systems for objects, agents, number, and geometry that are present in infants and likely scaffold later learning. ([harvardlds.org][25])
* Lake et al. argue that humanlike intelligence requires compositional structured models, causal reasoning, and model‑based learning, not just pattern recognition. ([PubMed][26])

**Why this might be a “critical line of code”**
A small “causal hypothesis engine” (generate structured hypotheses, update weights based on data, plan interventions) combined with deep perception could:

* Explain human‑level sample efficiency
* Support generalization far outside the training manifold
* Allow explicit reasoning about “what if” and “why” questions

Deep learning is starting to flirt with causal models, but usually without the strong, structured priors and active experimentation that developmental data suggest.

---

## 8. Intrinsic motivation, curiosity, and neuromodulation

**Core idea:**
Brains don’t explore randomly or only for extrinsic reward; they’re driven by curiosity and uncertainty reduction, implemented via neuromodulatory systems like dopamine.

**Brain evidence**

* Schultz and others: midbrain dopamine neurons encode reward prediction errors—differences between expected and received reward—driving learning in basal ganglia and frontal cortex. ([PubMed][27])
* Kidd & Hayden’s review of curiosity: curiosity behaves like an intrinsic reward signal, yet seems to share mechanisms with extrinsic reward, modulating learning and memory. ([PubMed][28])
* Gruber et al. show that states of curiosity enhance learning and hippocampal-dependent memory, again implicating dopaminergic modulation. ([ORCA][29])
* Broader “systems neuroscience of curiosity” work maps circuits that respond to information gaps, novelty, and uncertainty. ([ResearchGate][30])

**Why this might be a “critical line of code”**
A compact intrinsic-motivation module (e.g., reward ≈ reduction in prediction error / uncertainty / novelty within safe bounds) can:

* Drive efficient exploration without hand-designed curricula
* Balance exploitation and exploration adaptively
* Help the agent build rich world models even when extrinsic reward is sparse

Many RL algorithms bolt on curiosity bonuses, but they’re usually ad-hoc; brain data suggest more principled, biologically grounded formulations.

---

## 9. Embodiment, sensorimotor contingencies, and affordances

**Core idea:**
Intelligence is fundamentally sensorimotor. Perception is about what you *can do* with things; seeing is a way of acting; the body’s morphology and interoceptive state heavily shape cognition.

**Brain / cognitive evidence**

* O’Regan & Noë’s sensorimotor theory: “seeing is a way of acting”; vision is mastering lawful sensorimotor contingencies, not building a static internal picture. ([PubMed][31])
* Gibson’s ecological approach and affordances: organisms perceive action possibilities (“affordances”) directly in the environment, relative to their own capabilities. ([QIU Library][32])
* Embodied cognition literature shows that bodily states, perception–action systems, and environment coupling systematically shape “abstract” cognition. ([Stanford Encyclopedia of Philosophy][33])
* Interoceptive predictive coding and constructed emotion (Barrett) place internal bodily state and allostasis at the center of the brain’s generative model, not as peripheral inputs. ([PMC][5])

**Why this might be a “critical line of code”**
Even if an AGI is “virtual,” explicit modeling of sensorimotor contingencies and affordances (including internal ones, like compute/energy budgets) could:

* Ground concepts in possible actions instead of arbitrary vectors
* Provide natural abstractions for tools, interfaces, and environments
* Integrate external actions and internal regulation into one control loop

Right now, a lot of AI is “disembodied text”; a small affordance‑based action interface + world model could be a huge qualitative jump.

---

## 10. Social cognition, theory of mind, and learning from others

**Core idea:**
Human intelligence is deeply social. We constantly infer other minds, track beliefs and desires, and learn culturally. The brain dedicates specialized networks to “theory of mind” (ToM) and social reasoning.

**Brain / cognitive evidence**

* Reviews of ToM neuroimaging show a consistent network: temporo‑parietal junction (TPJ), medial prefrontal cortex (mPFC), and superior temporal sulcus support reasoning about others’ beliefs, desires, and perspectives. ([PMC][34])
* Saxe & Kanwisher identify right TPJ as selectively engaged when thinking about other people’s mental states, rather than physical facts. ([ScienceDirect][35])
* Hyde and others reinforce that TPJ is central for theory of mind and social cognition. ([Journal of Neuroscience][36])
* Work integrating interoceptive inference with social cognition suggests we may understand others by inferring their internal states via similar predictive mechanisms used for our own bodies. ([ScienceDirect][37])

**Why this might be a “critical line of code”**
A lightweight ToM / social-learning module—“build models of other agents’ beliefs, goals, and learning rules, and update yours accordingly”—could let AGI:

* Imitate, cooperate, and align with humans more naturally
* Use demonstrations, explanations, and norms as data
* Avoid naive, antisocial “pure utility maximizer” behaviors

Most present systems treat humans as part of the environment, not as *agents with minds*; a small change in that representational stance may have outsized consequences.

---

## How this ties back to Carmack’s “few key insights” view

You can cluster these 10 into a smaller set of “mega‑insights,” e.g.:

1. **Predictive generative models everywhere** (Items 1, 2, 9)
2. **Dual‑system learning and memory with replay** (Item 3, part of 6)
3. **Global workspace + working memory + compositional syntax** (Items 4, 6)
4. **Hierarchical control & skills, driven by curiosity under allostatic constraints** (Items 5, 8, part of 2 and 9)
5. **Structured causal and social models of the world and other agents** (Items 7, 10)

Each cluster is conceptually rich but codeable as a relatively small set of mechanisms once you’ve committed to a particular representation (e.g., neurons as continuous units, spiking, etc.). And importantly, the antecedents for *all* of them are already in the literature—neuroscience, psychology, and cognitive science have been circling these themes for decades.

If you’d like, next step could be: pick one or two of these and sketch how they might look as concrete algorithmic “blocks” that could plug into current deep learning infrastructure (without going full sci‑fi).

[1]: https://pubmed.ncbi.nlm.nih.gov/10195184/?utm_source=chatgpt.com "Predictive coding in the visual cortex: a functional ..."
[2]: https://arxiv.org/pdf/2112.10048?utm_source=chatgpt.com "Predictive Coding Theories of Cortical Function"
[3]: https://www.nature.com/articles/nrn2787?utm_source=chatgpt.com "The free-energy principle: a unified brain theory?"
[4]: https://pubmed.ncbi.nlm.nih.gov/21684297/?utm_source=chatgpt.com "Allostasis: a model of predictive regulation"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4731102/?utm_source=chatgpt.com "Interoceptive predictions in the brain - PMC - PubMed Central"
[6]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4270100/?utm_source=chatgpt.com "Homeostatic reinforcement learning for integrating reward ..."
[7]: https://pubmed.ncbi.nlm.nih.gov/7624455/?utm_source=chatgpt.com "Why there are complementary learning systems in ... - PubMed"
[8]: https://www.researchgate.net/publication/51853137_Complementary_Learning_Systems?utm_source=chatgpt.com "(PDF) Complementary Learning Systems"
[9]: https://pubmed.ncbi.nlm.nih.gov/3008780/?utm_source=chatgpt.com "The hippocampal memory indexing theory"
[10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC5847173/?utm_source=chatgpt.com "The Role of Hippocampal Replay in Memory and Planning"
[11]: https://pubmed.ncbi.nlm.nih.gov/21961947/?utm_source=chatgpt.com "Working memory: theories, models, and controversies"
[12]: https://pubmed.ncbi.nlm.nih.gov/11283309/?utm_source=chatgpt.com "An integrative theory of prefrontal cortex function"
[13]: https://pubmed.ncbi.nlm.nih.gov/16378516/?utm_source=chatgpt.com "a computational model of learning in the prefrontal cortex and ..."
[14]: https://pubmed.ncbi.nlm.nih.gov/32135090/?utm_source=chatgpt.com "Conscious Processing and the Global Neuronal Workspace ..."
[15]: https://pmc.ncbi.nlm.nih.gov/articles/PMC2783353/?utm_source=chatgpt.com "Hierarchically organized behavior and its neural foundations"
[16]: https://www.sciencedirect.com/science/article/abs/pii/S2352154615000790?utm_source=chatgpt.com "Hierarchical control of goal-directed action in the cortical– ..."
[17]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3660223/?utm_source=chatgpt.com "Motor primitives and synergies in spinal cord and after injury"
[18]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7137474/?utm_source=chatgpt.com "Donald O. Hebb and the Organization of Behavior: 17 years in ..."
[19]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3005627/?utm_source=chatgpt.com "Neural syntax: cell assemblies, synapsembles and readers"
[20]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3648857/?utm_source=chatgpt.com "The Theta-Gamma Neural Code - PMC - PubMed Central"
[21]: https://pubmed.ncbi.nlm.nih.gov/23594744/?utm_source=chatgpt.com "Hippocampal place-cell sequences depict future paths to ..."
[22]: https://pubmed.ncbi.nlm.nih.gov/16797219/?utm_source=chatgpt.com "Theory-based Bayesian models of inductive learning and ..."
[23]: https://www.sciencedirect.com/science/article/abs/pii/S1364661304001640?utm_source=chatgpt.com "Mechanisms of theory formation in young children"
[24]: https://pubmed.ncbi.nlm.nih.gov/17444973/?utm_source=chatgpt.com "Preschool children learn about causal structure from ..."
[25]: https://www.harvardlds.org/wp-content/uploads/2017/01/SpelkeKinzler07-1.pdf?utm_source=chatgpt.com "Core knowledge"
[26]: https://pubmed.ncbi.nlm.nih.gov/27881212/?utm_source=chatgpt.com "Building machines that learn and think like people"
[27]: https://pubmed.ncbi.nlm.nih.gov/9054347/?utm_source=chatgpt.com "A neural substrate of prediction and reward"
[28]: https://pubmed.ncbi.nlm.nih.gov/26539887/?utm_source=chatgpt.com "The Psychology and Neuroscience of Curiosity - PubMed - NIH"
[29]: https://orca.cardiff.ac.uk/id/eprint/123206/1/R1_Curiosity_Chapter_submitted.pdf?utm_source=chatgpt.com "Curiosity and Learning: A Neuroscientific Perspective Authors"
[30]: https://www.researchgate.net/publication/343218485_Systems_neuroscience_of_curiosity?utm_source=chatgpt.com "Systems neuroscience of curiosity | Request PDF"
[31]: https://pubmed.ncbi.nlm.nih.gov/12239892/?utm_source=chatgpt.com "A sensorimotor account of vision and visual consciousness"
[32]: https://library.uniq.edu.iq/storage/books/file/The%20Ecological%20Approach%20to%20Visual%20Perception%20Approach/1667383098The%20Ecological%20Approach%20to%20Visual%20Perception%20Classic%20Edition%20%28James%20J.%20Gibson%29%20%28z-lib.org%29%20%281%29.pdf?utm_source=chatgpt.com "The Ecological Approach to Visual Perception"
[33]: https://plato.stanford.edu/entries/embodied-cognition/?utm_source=chatgpt.com "Embodied Cognition - Stanford Encyclopedia of Philosophy"
[34]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6871093/?utm_source=chatgpt.com "Are there theory of mind regions in the brain? A review of the ..."
[35]: https://www.sciencedirect.com/science/article/abs/pii/S0028393205001223?utm_source=chatgpt.com "Making sense of another mind: The role of the right ..."
[36]: https://www.jneurosci.org/content/38/18/4264?utm_source=chatgpt.com "Functional Organization of the Temporal–Parietal Junction ..."
[37]: https://www.sciencedirect.com/science/article/pii/S0278262615300130?utm_source=chatgpt.com "The role of interoceptive inference in theory of mind"

Ok, so here's the deal. I've been working on this project for two years now, and roughly, in timeline order, (I could be off by a little on some things).

1. I decide I'm gonna work on the Arc-AGI challenge for my research class
2. I start by watching Andrej Karpathy's zero to hero course
3. Original plan is to have a VAE take an image as input, and generate the p5.js code that generated it like so:

---

# The Challenge

An ARC‑AGI task consists of a small **training set** (typically three‑to‑five) of input–output pixel grids, each grid expressed with integer‑encoded colours.
All training pairs share a *hidden* visual transformation.
After observing these examples, the model receives **one further input grid** whose correct output is withheld.
The objective is to **predict that withheld output** by inferring and applying the common transformation.

ARC-AGI tasks are explicitly few-shot/meta-learning problems. Each task provides 3-5 correlated input-output grid pairs (demonstrations) that share a hidden visual transformation rule. The model must infer this rule from the correlations across examples and apply it to a new test input. This requires treating the task as a holistic "episode" where inter-example relationships (e.g., how the transformation evolves or is consistent across pairs) are key.

---

# The Dataset

Our training corpus contains **millions of explicit code–image pairs**, each of which is a p5.js sketch alongside the exact image(s) it produces.

* **Single images** – sketches that draw one static picture.
* **Before/After image pairs** – sketches that apply a visual operation, yielding a *before* and *after* frame.
* **Before/After image sets** – sketches that apply the *same* transformation to multiple inputs, producing aligned before/after sets.

The dataset is organised as a *graduated curriculum*:

| Stage | New visual concepts introduced                                           |
| ----- | ------------------------------------------------------------------------ |
| 1     | Black & white primitives (points, lines, triangles, rectangles, circles) |
| 2     | Spatial layouts, colour fills/outlines                                   |
| 3     | Basic 2‑D transforms (rotation, scale, shear)                            |
| 4     | Layer operations (clipping, masking), simple textures                    |
| 5     | Fundamental 3‑D primitives (boxes, spheres)                              |
| 6     | Lighting & material effects                                              |
| 7     | Fully composed 2‑D / 3‑D scenes                                          |

This stepwise progression lets the model master elementary concepts before combining them into complex scenes.

---

4. I realize that the anisotropy created by row major indexing with transformers is going to make my original plan an uphill battle. I search for a new positional encoding mechanism/options
5. I find axial/mixed RoPE, but it has issues (can't see diagonals for instance) and the patching idea I think makes spatial reasoning more difficult
6. I originally think that we could do a 3D embedding, with a z dimension for layers, but then how does one encode the similarities between before and after grids without a time dimension. Turns out you need 4D for layers + time + x + y for the grid.
7. I develop MonSTERs which is a 4D encoding that is a principled extension of RoPE using Lorentz rotors to allow for a fully absolute + relative fusion encoding that works as a dropin replacement for RoPE compatible with linear attention methods. 
8. The hierarchical reasoning model paper drops and with it's clever data augmentation techniques gets a SOTA score on ARC-AGI-2 challenge set with a recurrent transformer on a grid.
9. I use the HRM code base to train my own models, with the assumption that my new encoding method will enhance performance dramatically. However, I only see improvements of like 3%-7%. Small enough to be dismissed as good look, hyper parameters etc. This confuses me.
10. I realize that on SINGLE TASKs the transformer architecture is powerful enough to overcome even arbitrary inductive priors, such as a row major indexing. This explains why, even though all LLMs nowadays have basically converged on 1D RoPE, the vision transformer community still uses a wide variety of positional encodings. From learned to 1D RoPE, to 2D or 3D or even 4D rope with axial etc. If you only train on one task, then there's no interference from other types of data and the transformer can learn to use the texture given by the encoder as an address.
11. This means to prove the advantages of my MonSTERs I will need to train a model that can do 1D (language, symbolic reasoning, math, etc over time) 2D+t (so 3D really) (ARC-AGI, Sudoku, Crosswords) and 3D+t (so really 4D) like maze navigation, rubik's cube, hole punching folding puzzles etc. 
12. This gives us other problems. As you increase the dimensions, you exponentially increase the number of positional encodings you need to have precomputed, you exponentially increase the number of tokens you need to have in context at any given moment, you make everything more and more difficult.
13. Add on top of that the fact that vocab over just RGB values is well over 16 million when even the largest LLMs don't exceed 300-400 thousand. (Constrained by softmax over all tokens in the vocab, plus if you don't get your training data distribution then some tokens would never update because they might not be in training which would throw everything off.
14. Then I read this paper about how Quaternion neural networks are exceptionally good at representing RGB color space where real valued neural networks fail, and I'm like oh shit, quaternions have O(1) invertability. So what if each RGB is represented by a single quaternion, which we then multiply in an up projection of weighted quaternions as a TYPED token. Right? So the model learns the up projection weights for COLOR type, then every RGB value is up projected by those weights to the full token embedding dimension. BOOM. And the O(1) inverse with quaternion multiplication means if we do it right we can do a distribution over each quaternion output from the weights as a vote for a single quaternion and replace the softmax, basically we synthesize tokens on the fly instead of holding them precomputed as weights for each token in the vocab. The same way an electronic keyboard synthesizes all the different instruments. 
15. But that doesn't solve the fact that as dimensions increase, tokens explode, and honestly, they can pollute the context window. Despite doing amazing at sequence to sequence for ARC-AGI, the HRM totally misses the point of meta learning. It doesn't take advantage of the fact that you can have examples in context, and in fact they find that by doing more than one grid, the performance degrades. Oh shit. SO then I realize, what we need is a DYNAMIC MEMORY, or a WORKING MEMORY. Basically, a reinforcement learning policy sees the full set of tokens in their minimal single quaternion form and we can make that bank of tokens in small form the "environment" and it then selects a set of queries, and for each query token it selects it also selects keys to effect that token, so that it can focus and zoom out and focus and zoom out, etc. This means that it would have dynamic attention threads instead of a static context window. Then it could, for example, look at grid one example input, and see, "background-black, width-15, height-10, colors-red,blue,green, objects 2" etc then save those observations to a "notepad" attached to that image, then look at the output, do the same thing, in a loop over all the images, then loop over all the pairs, see what's the same and what's different, then it can write the p5.js code that generated all those images, or a python grid transformation, then it can run the test on the examples, if it doesn't work it tries again, if it works then it submits the guess/answer for the test case bing bang boom we've done it.
16. However, that still leaves one thing, if the vocab is fixed ahead of time, and all we do is pretraining, we're still falling short of the dream. We need something like dreamcoder, paired with our type-value tokens, so that it can compress what it needs to, ignore what it wants, etc. and so the RL policy can continuously grow by writing programs/options it can reuse later. So what do we do? in addition to the 4D positional encoding MonSTERs (Minkowski Space Time Encoding Rotors), in addition to the dynamic attention, in addition to the quaternion type value tokens, we also add in a super generalizable learning objective. which is
17. It learns to compress multiple tokens into a single token such that, in a book, with chapters, paragraphs, sentences, and words, it can combine all the words in a sentence and use a VAE to encode them as a "sentence" token. It then takes several sentence tokens, and some important single word tokens or metadata, and encodes them as a single paragraph token. It then takes the paragraph tokens, and encodes them as a chapter token. Then chapters to book etc. With a reconstruction loss to train on. And it can include any other tokens it wants (tags, metadata, hints etc) in addition to the actual contents as long as it helps with reconstruction. Then we've essentially built like a "concept" token synthesizer where it can develop it's own vocab. And we can do something like dreamcoder to prune it back and refine the best ones so that it can continually learn. And then the reward for the RL model is not "next token prediction" but multiple rewards of next sentence prediction, and next paragraph prediction, along with the reconstruction loss, like CLIP, and then we can use temporal difference learning since the single step objective of the next token, can be seen as a multistep or next sentence. Additionally, on something like sudoku, we can do next state prediction. Where for example, let's say there's multiple logically valid and equivalent moves that could be made at any moment in Sudoku. Like you understand that with a basic CSP, with a single input state, several squares could all be valid next fills. So we collect that state of all of the logical deductions that could be made from the given state, and we reward the model as long as it makes one of those moves, and we can use the CSP solver to build reasoning traces, where it uses rules, logical deductions etc, all on the scratchpad, and it can make as many observations it wants in the recurrent loop as long as it's next output is a valid one. Right?

SOOOOO ANYWAYS. That's what I've got swirling about. And I think I can turn it into 4 papers. 

1. Spatial Reasoning MonSTERs (the flaws of current spatial encoding methods, why researchers use so many different ones, the importance of a domain agnostic encoding that can handle all data types with a unified format, and our training stack + data set of bongard problems, sudoku, etc)
2. Dynamic Attention Threads for Transformer Working Memory (the RL policy of selecting queries, and then keys, as well as the recursive loop module over tokens allowing it to reason spatiotemporally in large contexts)
3. Relay Learning (basically, we cover 1. the quaternion synthesized tokens, 2. the dream coder continual learning/refinement model setup 3. how Rich Sutton when he argues against using human knowledge, somehow doesn't realize that RL and LM diverged as disciplines because RGB and environments already have the human knowledge of physics baked into their semantic representations, whereas words used nominal identifications in computers/electronics, and therefore had to be learned. So the ideal, isn't "no human knowledge." It's a relay where the computers learn things, and we learn things, and we go back and forth. And the real winners will be those who reduce the translation tax and the handoff inefficiencies to nil. Those who realize it's a relay race, and not an either or humans vs machines will win.)

Finally, I take these three dynamic modules (4D MonSTERs, Dynamic Attention Threads, and Synthesized/Structured Type-Value tokens) together for the final paper which is simply:

MonSTER Models are Domain-Agnostic Multitask Learners. 

An homage to the GPT-2 paper title. Where we show our largest model can do all these different tasks because it has dynamic options for the static fixed assumptions of the standard transformer baseline architecture.

Does that all make sense?
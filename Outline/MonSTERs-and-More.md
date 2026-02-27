# MonSTER Models are Domain-Agnostic Multitask Learners







These three assumptions, highly interdependent and entangled by nature, generally preclude isolated modification, and as such, have enjoyed a relatively undisturbed existence. 

Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to sway the Triumvirate; such attempts are left to naive cowboy scientists who, unaware of the task's true nature, dive in headlong. 

This paper outlines the series of painful, though eventually fruitful, realizations made by one such cowboy scientist, the author. First, a new positional encoding mechanism 



 The challenge remains Leaving only cowboy scientists, naive and ignorant to the tasks full difficulty. 


leaving the challenge to so-called cowboy scientists, naive and ignorant 




In exchange for a battle-tested baseline, researchers incur a translation cost 


who adopt these three 


However, this baseline incurs a "translation tax," 



These three assumptions, highly interdependent and entangled by nature, have generally precluded isolated modification, enjoying a relatively undisturbed existence. 

Indeed, no sane scientist* with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

opting instead to labor in the more tractable subspace of architectural patches



At a high level, the baseline transformer interface assumes a static, fixed token vocabulary, a static, fixed attention-window, and uniform attentional treatment of the predetermined vocabulary, within the predetermined window, by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 

At a high level, the baseline transformer interface assumes a static, fixed token vocabulary, a static, fixed attention-window, and a predetermined attentional treatment over tokens within that window, modified only by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 



and a predetermined attentional treatment over tokens within that window, modified only by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 



At a high level, the baseline transformer interface assumes a predetermined, static, and fixed token vocabulary, a predetermined, static, and fixed attention-window, and a uniform attentional treatment of tokens within the window

the predetermined vocabulary, within the predetermined window, by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 

These three assumptions, highly interdependent and entangled by nature, generally preclude isolated modification, and as such, have enjoyed a relatively undisturbed existence. 


 Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

1. positional encoding for higher dimensions remains unsolved (hence the proliferation of so many adaptations, and the absence of a clear/dominant method. RoPE is there for 1D, but for 2D researchers continue to use row-major flattening, learned encodings for a fixed  context area, and the most prominent method, axial RoPE, simply factorizes the 1D case by splitting the embedding into 2 independent arrays, one for x, and one for y, which means it can't see diagonals)

2. vocabulary sets balloon and take up large amounts of memory and are fixed (this is especially problematic when one considers that there are ~16 million RGB values, ~65,000 int values, and most SOTA LLMs have vocabs maxing out at around 200,000 tokens, and you have to remember the softmax at the end has to calculate the probability for each token, which takes O(n) time)

3. context window size is too limited for images (it's why we have to do patches for images with clip and dalle and diffusion)

4. continual learning is not yet possible.


These three assumptions, highly interdependent and entangled by nature, generally preclude isolated modification, and as such, have enjoyed a relatively undisturbed existence, despite their individual faults. Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.





Indeed, any sane scientist with a desire for peace, happiness, or good health, is better served 

Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

leverage multiplier. 

is best served buttressing the Triumvirate, and applies more leverage working around despite their individual faults



Further fortified by modifications that work with the current architecture 


Protected by their combined efficiency and power, Despite their individual shortcomings, their combination 



Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

Alas, despite their shortcomings, it is left to the cowboy scientist, ignorant 


Indeed, no sane scientist with a desire for peace, happiness, or good health, would seek to to unseat them from their rightful throne. 


Despite their shortcomings, interconnected, thus undisturbed



would meddle in even one of the trio's affairs, certain of the 



ever endeavor to meddle in any one of the trio's affairs, 


sway even one of them, cognizant of ....


ever endeavor to seatun them from their rightful throne.






Alas, it is left to the cowboy scientist, 




While researchers may modify x, y, and z, they rarely modify these 3 core elements as their interdependencies preclude isolated modification. 







ARC-AGI highlights several shortcomings of the modern transformer stack:

1. positional encoding flaws (anisotropic (diagonal), implicit address, fixed/ordered/static, dimensionality, no nesting or hierarchy of objects)
2. vocab flaws (exploding vocab size, softmax, fixed vocab)
3. 


---

The modern stack starts with:

1. A fixed, predetermined vocabulary
2. A fixed, predetermined context-length, and
3. Uniform treatment of the predetermined vocabulary, within the predetermined context-window, by implicitly assigned, fixed, and predetermined positional encodings.


 






# MonSTERs for N-Dimension Attention

# MonSTERs are n-Dimensional Multi-Task Learners

# MonSTER Models Are N-Dimensional Multitask Reasoners

# MonSTERs & More
## Structural Encodings for Native 4-Dimension-Attention

### Abstract

- No dominant method for tasks beyond 1D / proliferation of variants
- In single-task settings, the encoding method is inconsequential

- Improvements to RoPE come in two flavors:
   - Extensions
   - Compromises
   
- Majority of researchers take the trade:
   - Abandon structural constraint
   - Implement task-specific improvement







While RoPE has become the dominant positional encoding method for 1D tasks, no equivalent 

The absence of a singular favored method for extending 


“The lack of a single, widely adopted scheme for extending positional encodings beyond 1D suggests that, in fixed domains, transformers can often internalize spatial structure without a canonical inductive bias—but that this brittleness shows up when we demand multitask, cross-domain generalization.”

“That the community has not converged on a RoPE-like standard for 2D/3D/4D positional encoding is a sign that many extensions work ‘well enough’ in narrow settings, yet fail to provide the shared, transferable structure needed for multitask learning.”

“The proliferation of incompatible 2D–4D RoPE variants and learned embeddings—without a clear winner—indicates that positional structure is frequently recoverable from task-specific data, but becomes a limiting factor once we move to general-purpose, multitask models.”

“The absence of consensus on higher-dimensional positional encodings points to a mismatch between what suffices for single-task training and what is required for robust, compositional transfer across tasks and modalities.”

“That no RoPE-equivalent has emerged for higher-dimensional inputs suggests that transformers can compensate for imperfect positional bias in constrained regimes, while revealing the need for a principled, shared encoding when scaling to multitask learners.”


Can you help me polish this into a one sentence intro?

The absence of a singular favored method for extending the positional encodings of transformer models to higher dimensions is an indicator that ....

Basically, I'm trying to say, when we extend to 2d, or to 3d, or to 4d, there's been many papers proposing modifications to RoPE, or using learned positional embeddings, but there hasn't been this consolidation around a single method like there has been for RoPE in 1D. I think it's because the transformer is so powerful, that if you only train on a single task, or if you fix the domain of the task, it can learn those relationships anyway. However, when you want to move onto multitask learners, there's where trouble arises. Do you see where I'm trying to go? Can you give me 5 options for that opening sentence? 


To the cowboy scientist, the sheer variety of adaptations made to squeeze current positional encoding mechanisms into additional dimensions, and the clear absence of a de facto standard is nothing short of perplexing. It appears


, modifications, and optimizations made to positional encodings to squeeze them into ~ill fitting domains~ is nothing short of perplexing. 

Radford et al. showed that aligning positional representations with a domain’s true underlying dimensionality can unlock substantial generalization gains, enabling few- or even zero-shot performance on previously unseen tasks.

Radford et al demonstrated that by aligning positional encoding to the domain's actual (reality) dimension/space unleashes previously unimaginable generalization benefits, permitting few or zero shot performance on previously unseen tasks. Taken seriously, this premise suggests that the obvious dimension for an ideal positional encoding mechansim would be spatio-temporal, and that ....



---



---












---






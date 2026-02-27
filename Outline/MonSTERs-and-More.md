# MonSTER Models are Domain-Agnostic Multitask Learners

## Core Model Interface Claims

The modern stack starts with:

1. A fixed, predetermined vocabulary
2. A fixed, predetermined context-length, and
3. Uniform treatment of the predetermined vocabulary, within the predetermined context-window, by implicitly assigned, fixed, and predetermined positional encodings.

At a high level, the baseline transformer interface assumes a static, fixed token vocabulary, a static, fixed attention-window, and uniform attentional treatment of the predetermined vocabulary, within the predetermined window, by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 

At a high level, the baseline transformer interface assumes a static, fixed token vocabulary, a static, fixed attention-window, and a predetermined attentional treatment over tokens within that window, modified only by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 

and a predetermined attentional treatment over tokens within that window, modified only by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 

At a high level, the baseline transformer interface assumes a predetermined, static, and fixed token vocabulary, a predetermined, static, and fixed attention-window, and a uniform attentional treatment of tokens within the window

the predetermined vocabulary, within the predetermined window, by a static, fixed set of positional encodings, implicitly mapped to their tokens by index. 

While researchers may modify x, y, and z, they rarely modify these 3 core elements as their interdependencies preclude isolated modification. 

## Translation Tax / Triumvirate Framing

These three assumptions, highly interdependent and entangled by nature, generally preclude isolated modification, and as such, have enjoyed a relatively undisturbed existence. 

These three assumptions, highly interdependent and entangled by nature, have generally precluded isolated modification, enjoying a relatively undisturbed existence. 

These three assumptions, highly interdependent and entangled by nature, generally preclude isolated modification, and as such, have enjoyed a relatively undisturbed existence. 

These three assumptions, highly interdependent and entangled by nature, generally preclude isolated modification, and as such, have enjoyed a relatively undisturbed existence, despite their individual faults. Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

In exchange for a battle-tested baseline, researchers incur a translation cost 

who adopt these three 

However, this baseline incurs a "translation tax," 

Further fortified by modifications that work with the current architecture 

Protected by their combined efficiency and power, Despite their individual shortcomings, their combination 

Despite their shortcomings, interconnected, thus undisturbed

would meddle in even one of the trio's affairs, certain of the 

ever endeavor to meddle in any one of the trio's affairs, 

sway even one of them, cognizant of ....

ever endeavor to seatun them from their rightful throne.

## Enumerated Weaknesses

1. positional encoding for higher dimensions remains unsolved (hence the proliferation of so many adaptations, and the absence of a clear/dominant method. RoPE is there for 1D, but for 2D researchers continue to use row-major flattening, learned encodings for a fixed  context area, and the most prominent method, axial RoPE, simply factorizes the 1D case by splitting the embedding into 2 independent arrays, one for x, and one for y, which means it can't see diagonals)

2. vocabulary sets balloon and take up large amounts of memory and are fixed (this is especially problematic when one considers that there are ~16 million RGB values, ~65,000 int values, and most SOTA LLMs have vocabs maxing out at around 200,000 tokens, and you have to remember the softmax at the end has to calculate the probability for each token, which takes O(n) time)

3. context window size is too limited for images (it's why we have to do patches for images with clip and dalle and diffusion)

4. continual learning is not yet possible.

## Cowboy Scientist Narrative Variants

This paper outlines the series of painful, though eventually fruitful, realizations made by one such cowboy scientist, the author. First, a new positional encoding mechanism 

The challenge remains Leaving only cowboy scientists, naive and ignorant to the tasks full difficulty. 

leaving the challenge to so-called cowboy scientists, naive and ignorant 

Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to sway the Triumvirate; such attempts are left to naive cowboy scientists who, unaware of the task's true nature, dive in headlong. 

Indeed, no sane scientist* with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

Indeed, no sane scientist with a desire for peace, happiness, or good health, endeavors to unseat the Triumvirate, opting instead to labor in actually tractable fields.

Indeed, no sane scientist with a desire for peace, happiness, or good health, would seek to to unseat them from their rightful throne. 

Alas, despite their shortcomings, it is left to the cowboy scientist, ignorant 

Alas, it is left to the cowboy scientist, 

Indeed, any sane scientist with a desire for peace, happiness, or good health, is better served 

leverage multiplier. 

is best served buttressing the Triumvirate, and applies more leverage working around despite their individual faults

opting instead to labor in the more tractable subspace of architectural patches


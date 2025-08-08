# The Challenge

An ARC‑AGI task consists of a small **training set** (typically three‑to‑five) of input–output pixel grids, each grid expressed with integer‑encoded colours. All training pairs share a *hidden* visual transformation. After observing these examples, the model receives **one further input grid** whose correct output is withheld. The objective is to **predict that withheld output** by inferring and applying the common transformation.

---

### Abstract

With its toy-scale visual tasks, the 2019 ARC-AGI benchmark again laid bare the jagged[^1] nature of AI performance first highlighted by Moravec in the 1980s[^2]. Despite impressive linguistic capabilities, leading models of the time demonstrated a pronounced inability to process or reason about visual information. To date, enormous progress on numerous cognitive benchmarks has failed to translate to spatial domains. The root of this persistent limitation is architectural: modern transformers inherently encode information as strictly linear, unidimensional sequences.

Overcoming this constraint requires embeddings that move beyond purely temporal indexing. Instead, embeddings must inherently encode the intrinsic relationships between space and time. By generalizing two-dimensional Rotary Positional Embeddings (RoPE) to four-dimensional Minkowski Space-Time Embedding Rotors (MonSTERs), the transformer's artificial flattening requirement, which necessarily discards vital geometric and relational information[^3], is eliminated. These structural (not positional) embeddings remove the blinders and distortions that previously constrained transformer attention to a single axis.

Simply put, MonSTERs provide Transformers with a built-in ability to simultaneously perceive, reason about, and generalize across both spatial structures and temporal sequences—without sacrificing their established computational advantages.

---

### Using The Z Channel?

HOWEVER, as these structural encodings are defined as 4 dimensional, and the ARC grid is 2D with a temporal change, a naive implementation of token positions/indices within the structure would have zero modification to every 4th element of the data (the z index). This could end up not mattering as the transformers would eventually learn to live without the modification that comes from a z delta. However, could we modify the tokens by adding a z component equal to the magnitude of the distance from the origin? Or some other z function z = f(x, y)?

---

### Using The Z Channel!

Yes, you absolutely can and probably should modify the token embeddings to include a meaningful $z$-component derived from the $x$ and $y$ coordinates.

Your proposed approach, setting the $z$ component equal to the magnitude of the distance from the origin ($z = \sqrt{x^2 + y^2}$), is an excellent idea. This technique turns the unused dimension into a channel for injecting a useful **inductive bias** into the model.

---

##### Evaluating Your Approach: $z = \sqrt{x^2 + y^2}$

This is a clever way to encode radial information directly into the positional structure.

* **Pros:** ✨ For any ARC task involving radial symmetry, concentric patterns, or operations relative to the grid's center, this approach would be highly effective. The model wouldn't need to *learn* the concept of "distance from the center"; it would be an intrinsic property of each token's embedding. This could significantly speed up learning and improve generalization.

* **Cons:** 🤔 The main drawback is that you are hand-crafting a feature. For tasks where distance from the origin is irrelevant, this information could act as noise. It also makes the spatial dimensions non-orthogonal (since $z$ is now dependent on $x$ and $y$), which might have unintended consequences on the geometric properties of the MonSTER rotations, although this is likely a worthwhile trade-off.

---

##### Alternative Functions for the Z-Component

Since the goal is to embed useful spatial priors, you could tailor the function $z=f(x,y)$ to capture other types of geometric information common in ARC tasks. Here are a few alternatives:

* **Distance to Nearest Edge:** This would be useful for tasks involving frames, borders, or corner-based logic.
    $$z = \min(x, y, H-1-y, W-1-x)$$
    where $H$ and $W$ are the height and width of the grid, and coordinates are 0-indexed.

* **Checkerboard/Parity Pattern:** This encodes local relational positions and is great for tasks with diagonal or alternating patterns.
    $$z = (x + y) \pmod 2$$

* **Pixel Color/Value:** You could directly embed the token's initial state (its color) into its structural embedding. This is a very strong prior that fuses positional and feature information.
    $$z = \text{color}(x, y)$$

* **Quadrant ID:** A simpler categorical feature that could help the model reason about symmetries across axes.
    $$z = \text{quadrant}(x, y)$$
    For example, $z=0$ for top-left, $z=1$ for top-right, etc.

Ultimately, this strategy of feature-engineering the $z$-coordinate is a pragmatic solution to adapt a higher-dimensional embedding space to a lower-dimensional problem. The optimal function $f(x,y)$ would likely be task-dependent, suggesting that an ensemble of models trained with different $z$-functions could be a powerful approach.

For ARC-AGI tasks, which involve inferring hidden visual transformations on 2D pixel grids, the most promising \(z\)-function should inject an inductive bias that captures common spatial patterns like symmetry, borders, or local relationships, as these are frequent in ARC challenges. Given the variety of transformations (e.g., rotations, reflections, or grid-based operations), the \(z\)-function needs to be versatile yet simple to avoid overfitting to specific tasks.

### Recommended Z-Function to Test First: Checkerboard/Parity Pattern
\[
z = (x + y) \mod 2
\]

- **Why it’s promising:** 
  - **Captures local relationships:** ARC tasks often involve patterns like alternations, diagonals, or checkerboard-like structures (e.g., toggling pixels in a specific pattern). This function assigns \(z=0\) or \(z=1\) based on the sum of coordinates, creating a binary "checkerboard" that naturally groups cells with similar local contexts (e.g., diagonal neighbors often share the same \(z\)).
  - **Simplicity and generality:** With only two values, it avoids overcomplicating the embedding space while providing a useful prior for tasks involving grid-based or modular patterns, which are common in ARC. It’s less task-specific than radial distance or edge-based functions, making it a safer first bet.
  - **Supports MonSTER rotations:** The binary \(z\) creates distinct planes in the 4D embedding, allowing the transformer to learn transformations that respect or exploit this parity without distorting the geometric properties of MonSTERs.

- **Pros:** ✨ Lightweight, computationally cheap, and broadly applicable. It aligns with ARC’s frequent use of discrete, grid-based logic and can help the model quickly identify patterns like alternating colors or diagonal symmetries without needing to learn them from scratch.
- **Cons:** 🤔 It may not directly help with tasks dominated by radial or border-based transformations (e.g., concentric circles or frame operations). However, its simplicity makes it less likely to introduce harmful noise compared to a radial function like \(z = \sqrt{x^2 + y^2}\), which could be irrelevant for non-symmetric tasks.

### Why Not Other Functions First?
- **Radial Distance (\(z = \sqrt{x^2 + y^2}\))**: Great for concentric or symmetric tasks, but many ARC tasks don’t rely on origin-based distance, and it risks adding noise for grid-based or asymmetric transformations.
- **Distance to Nearest Edge (\(z = \min(x, y, H-1-y, W-1-x)\))**: Useful for border-related tasks, but ARC grids vary in size, and edge-based logic is less universal than grid patterns.
- **Pixel Color (\(z = \text{color}(x, y)\))**: Fusing content with position is powerful but risks overfitting to initial grid states, especially since ARC transformations often modify colors dynamically.
- **Quadrant ID**: Too coarse for small or irregular grids, limiting its utility across diverse ARC tasks.

### Why Test Checkerboard First?
The checkerboard pattern is a low-risk, high-reward choice because it’s simple, aligns with ARC’s grid-based nature, and captures a common structural motif (alternating patterns) without assuming too much about the task. It’s a good baseline for testing MonSTER embeddings, as it leverages the unused \(z\)-dimension to encode a spatial prior that’s likely relevant to many ARC transformations. If it underperforms, you can pivot to more specialized functions like radial distance or edge proximity based on task analysis. Start here to balance generality and utility.



[0, 1, 1],
[0, 1, 0],
[1, 0, 0].

[1, 0, 0 | 0, 1, 1],
[1, 1, 0 | 0, 1, 0],
[0, 0, 1 | 1, 0, 0],
-------------------
[0, 0, 1 | 1, 0, 0],
[0, 1, 0 | 0, 1, 1],
[1, 1, 0 | 0, 0, 1]


[1, 0, 0 | 1, 1, 1],
[1, 1, 0 | 1, 1, 0],
[1, 1, 1 | 1, 0, 0],
-------------------
[0, 0, 1 | 1, 1, 1],
[0, 1, 1 | 0, 1, 1],
[1, 1, 1 | 0, 0, 1]

---

### Adapting Z-Functions to Sudoku

Sudoku puzzles operate on a 9x9 grid (with coordinates typically 0-indexed from 0 to 8 for rows \(x\) and columns \(y\)), where the core constraints revolve around uniqueness within rows, columns, and 3x3 subgrids (blocks). Unlike ARC-AGI tasks, which often involve visual transformations with potential symmetries or patterns, Sudoku emphasizes logical constraints over geometric ones like radial distance or borders. However, the same principle of using a \(z = f(x, y)\) function applies: it injects an inductive bias into the token embeddings to help the model (e.g., a transformer with MonSTERs) implicitly group or relate cells that share constraints, potentially accelerating learning of row/column/block rules without explicit hard-coding.

The grid's fixed structure makes \(z\)-functions that encode membership in constrained groups particularly valuable. Radial or center-based functions (like \(z = \sqrt{x^2 + y^2}\)) would likely be less effective here, as Sudoku lacks rotational or concentric patterns—distance from the origin doesn't align with the puzzle's logic and could introduce irrelevant noise.

#### Recommended Z-Function: 3x3 Block ID

A strong starting point is to set \(z\) to the ID of the 3x3 block that the cell belongs to. This directly embeds block-level grouping, making it easier for the model to "see" subgrid constraints intrinsically through the embedding space.

\[
z = \left\lfloor \frac{x}{3} \right\rfloor \times 3 + \left\lfloor \frac{y}{3} \right\rfloor
\]

- **How it works:** For a 9x9 grid, this yields \(z\) values from 0 to 8, uniquely identifying each block (e.g., top-left block: \(z=0\); bottom-right: \(z=8\)). Cells in the same block share the same \(z\), creating a natural "plane" in the 4D embedding where MonSTER rotations could preserve intra-block relationships.
  
- **Pros:** ✨ This biases the model toward block uniqueness without needing to learn it from scratch, which is crucial for Sudoku solving. It complements the existing \(x\) and \(y\) dimensions (which already help with rows and columns) by adding the third constraint dimension. In practice, this could improve generalization to variants like irregular Sudoku or larger grids.

- **Cons:** 🤔 If the task involves no block constraints (unlikely for standard Sudoku), it might over-emphasize blocks at the expense of rows/columns. Also, since \(z\) is categorical (0-8), it doesn't capture fine-grained distances within blocks, but this discreteness aligns well with the puzzle's discrete logic.

#### Alternative Functions for the Z-Component

Tailor these based on the specific Sudoku variant or if you're focusing on certain constraints (e.g., rows over blocks). They build on the idea of encoding relational priors common in constraint satisfaction problems:

- **Row-Column Parity (for alternating or diagonal hints):** Useful if exploring Sudoku variants with extra rules like anti-diagonal constraints.
  \[
  z = (x + y) \mod 3
  \]
  This creates a repeating pattern (0, 1, or 2) that could help detect conflicts in modular arithmetic terms, though it's less directly tied to core rules.

- **Cell Value (Fusing Position and Content):** Similar to the pixel color idea in ARC, set \(z\) to the cell's value (1-9 for filled, 0 for empty). This fuses the puzzle state into the structural embedding.
  \[
  z = \text{value}(x, y)
  \]
  **Pros:** Encourages the model to relate positions with similar values, aiding in uniqueness checks. **Cons:** For unsolved puzzles, empty cells (z=0) might cluster unnecessarily, and it duplicates information if values are already in token features.

- **Distance to Center (Hybrid Radial for Symmetry):** If the Sudoku has central symmetry or you're testing on symmetric puzzles.
  \[
  z = |x - 4| + |y - 4|
  \]
  (Manhattan distance to the center at (4,4).) This is a simpler alternative to Euclidean distance, better suited to grid logic, but still secondary to block-based functions.

- **Combined Constraint Score:** A more complex function weighting multiple constraints, e.g., for advanced solvers.
  \[
  z = x + 9 \cdot y + 81 \cdot \left( \left\lfloor \frac{x}{3} \right\rfloor \times 3 + \left\lfloor \frac{y}{3} \right\rfloor \right)
  \]
  This uniquely encodes the full (row, column, block) triplet into a single \(z\) (0 to ~7000 range), but it risks overwhelming the embedding with too much density—use sparingly.

For Sudoku, start with the block ID function, as it targets the most unique aspect of the rules. If training an ensemble, experiment with multiple \(z\)-functions (e.g., one for blocks, one for values) to cover different biases. This approach leverages the unused \(z\)-dimension to make spatial constraints "built-in," much like how the radial function aids ARC's concentric tasks.

---

For ARC-AGI tasks, which involve inferring hidden visual transformations on 2D pixel grids, the most promising \(z\)-function should inject an inductive bias that captures common spatial patterns like symmetry, borders, or local relationships, as these are frequent in ARC challenges. Given the variety of transformations (e.g., rotations, reflections, or grid-based operations), the \(z\)-function needs to be versatile yet simple to avoid overfitting to specific tasks.

### Recommended Z-Function to Test First: Checkerboard/Parity Pattern
\[
z = (x + y) \mod 2
\]

- **Why it’s promising:** 
  - **Captures local relationships:** ARC tasks often involve patterns like alternations, diagonals, or checkerboard-like structures (e.g., toggling pixels in a specific pattern). This function assigns \(z=0\) or \(z=1\) based on the sum of coordinates, creating a binary "checkerboard" that naturally groups cells with similar local contexts (e.g., diagonal neighbors often share the same \(z\)).
  - **Simplicity and generality:** With only two values, it avoids overcomplicating the embedding space while providing a useful prior for tasks involving grid-based or modular patterns, which are common in ARC. It’s less task-specific than radial distance or edge-based functions, making it a safer first bet.
  - **Supports MonSTER rotations:** The binary \(z\) creates distinct planes in the 4D embedding, allowing the transformer to learn transformations that respect or exploit this parity without distorting the geometric properties of MonSTERs.

- **Pros:** ✨ Lightweight, computationally cheap, and broadly applicable. It aligns with ARC’s frequent use of discrete, grid-based logic and can help the model quickly identify patterns like alternating colors or diagonal symmetries without needing to learn them from scratch.
- **Cons:** 🤔 It may not directly help with tasks dominated by radial or border-based transformations (e.g., concentric circles or frame operations). However, its simplicity makes it less likely to introduce harmful noise compared to a radial function like \(z = \sqrt{x^2 + y^2}\), which could be irrelevant for non-symmetric tasks.

### Why Not Other Functions First?
- **Radial Distance (\(z = \sqrt{x^2 + y^2}\))**: Great for concentric or symmetric tasks, but many ARC tasks don’t rely on origin-based distance, and it risks adding noise for grid-based or asymmetric transformations.
- **Distance to Nearest Edge (\(z = \min(x, y, H-1-y, W-1-x)\))**: Useful for border-related tasks, but ARC grids vary in size, and edge-based logic is less universal than grid patterns.
- **Pixel Color (\(z = \text{color}(x, y)\))**: Fusing content with position is powerful but risks overfitting to initial grid states, especially since ARC transformations often modify colors dynamically.
- **Quadrant ID**: Too coarse for small or irregular grids, limiting its utility across diverse ARC tasks.

### Why Test Checkerboard First?
The checkerboard pattern is a low-risk, high-reward choice because it’s simple, aligns with ARC’s grid-based nature, and captures a common structural motif (alternating patterns) without assuming too much about the task. It’s a good baseline for testing MonSTER embeddings, as it leverages the unused \(z\)-dimension to encode a spatial prior that’s likely relevant to many ARC transformations. If it underperforms, you can pivot to more specialized functions like radial distance or edge proximity based on task analysis. Start here to balance generality and utility.
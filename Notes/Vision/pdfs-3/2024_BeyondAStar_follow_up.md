Distinct tasks evaluated (unique): 2.
Evidence basis and task list (duplicates removed across variants):

- Maze navigation (shortest path in an n-by-n maze). (Section 3 Problem Setup: "We consider two domains: maze navigation ... In maze navigation, the goal is to find the shortest path through an n-by-n maze.")
- Sokoban puzzles (push boxes onto docks). (Section 3 Problem Setup: "We consider two domains: ... solving Sokoban puzzles ... In Sokoban, a worker can move up, down, left, or right and has to push each box onto a dock to solve the puzzle.")

Number of trained model instances required to cover all tasks: 2.
Rationale:
- Maze navigation is trained as its own task family: "In the first experiment set, we train a set of encoder-decoder Transformer models to predict optimal plans for maze navigation tasks." (Section 4.1)
- Sokoban is trained separately by repeating experiments on Sokoban data: "we repeat our experiments for Sokoban puzzles using our non-deterministic A* implementation." (Section 4.2)
- Models are trained from scratch for their task datasets, indicating task-specific training rather than a single joint multi-task instance: "Because every model is trained from scratch, the resulting models are specifically trained to only predict sequences that outline optimal plans for a set of different planning tasks." (Section 3.1)

Task-Model Ratio:
$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$

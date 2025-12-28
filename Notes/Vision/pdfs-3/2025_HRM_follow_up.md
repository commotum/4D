Number of distinct tasks evaluated: 4.
Evidence:
- ARC-AGI-1 and ARC-AGI-2 are described as separate benchmarks (2025_HRM.pdf, Section 3.1: "The initial version, ARC-AGI-1, presents challenges..." and "Addressing the limitations identified in ARC-AGI-1, ARC-AGI-2 significantly expands the benchmark...").
- Sudoku-Extreme is a Sudoku benchmark; Sudoku-Extreme-Full is the same task at full training scale (2025_HRM.pdf, Section 3.1: "Sudoku-Extreme is a down-sampled subset..." and "we use the complete training data, Sudoku-Extreme-Full...").
- Maze-Hard is its own benchmark task (2025_HRM.pdf, Section 3.1: "Maze-Hard This task involves finding the optimal path in a 30x30 maze...").

Number of trained model instances required to cover all tasks: 4 (one per benchmark).
Evidence: "For all benchmarks, HRM models were initialized with random weights and trained in the sequence-to-sequence setup using the input-output pairs." (2025_HRM.pdf, Section 3.2 Evaluation Details).

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$

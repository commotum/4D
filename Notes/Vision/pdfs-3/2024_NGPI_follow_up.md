1. Number of distinct tasks evaluated: 13.
   - ARC-AGI evaluation set (full). Evidence: Table 1 reports results on the full ARC-AGI evaluation set. (2024_NGPI.txt:433-434)
   - ARC-AGI evaluation set solvable subset (29 tasks). Evidence: the subset of 29 tasks (7.25% of the full evaluation set) is reported separately. (2024_NGPI.txt:351, 433-434)
   - ARC-AGI hidden test set. Evidence: Table 2 reports ARC-AGI hidden test set performance. (2024_NGPI.txt:441-442)
   - Out-of-distribution task set (10 distinct tasks) explicitly listed in the paper:
     - Task #48131b3c from ARC-AGI evaluation set: tile the original grid 2x2 and invert colors. (2024_NGPI.txt:391-407)
     - Hand-crafted task 1: gravitate left, then up, then change foreground color. (2024_NGPI.txt:408-413)
     - Hand-crafted task 2: rotate 90, upscale horizontally by two, then upscale vertically by two. (2024_NGPI.txt:415-419)
     - Hand-crafted task 3: task 2 plus horizontal mirroring. (2024_NGPI.txt:421-422)
     - Hand-crafted task 4: task 3 plus color inversion. (2024_NGPI.txt:423-424)
     - Hand-crafted task 5: tile 2x4 alternating rot180 and identity. (2024_NGPI.txt:457-465)
     - Hand-crafted task 6: tile the smallest object twice horizontally. (2024_NGPI.txt:467-471)
     - Hand-crafted task 7: crop object with most sub-objects, rotate 90, duplicate top and bottom rows. (2024_NGPI.txt:472-476)
     - Hand-crafted task 8: filter out the largest object, then rotate 270. (2024_NGPI.txt:477-480)
     - Hand-crafted task 9: crop largest object, split horizontally, merge halves with OR. (2024_NGPI.txt:481-485)

2. Number of trained model instances required to cover all tasks: 1.
   - Rationale: While the paper trains three separate VLM instances for DSL versions 1-3, the DSL v3 model includes all primitives/task generators from v2 and v1 (so it subsumes earlier DSLs). No task-specific heads or per-task fine-tunes are described, so a single DSL v3 model can be used across all reported tasks. (2024_NGPI.txt:359-363)

3. Task-Model Ratio:

$$
\boxed{
\frac{13\ \text{tasks}}{1\ \text{model}} = 13
}
$$

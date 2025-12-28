1. Number of distinct tasks evaluated: 1 (MATH benchmark math problem solving; "we focus on the MATH [13] benchmark" and use the 12k/500 split). (p. 6)

2. Number of trained model instances required to cover all tasks: 4
   - PaLM 2-S* base model used for the task. (p. 6)
   - A separate revision model fine-tuned to iteratively revise answers. (p. 4)
   - A separately fine-tuned PRM verifier (binary classifier) used for search. (p. 22)
   - A separate ORM verifier trained for the revision model's outputs. (p. 26)

$$
\boxed{
\frac{1\ \text{task}}{4\ \text{models}} = 0.25
}
$$

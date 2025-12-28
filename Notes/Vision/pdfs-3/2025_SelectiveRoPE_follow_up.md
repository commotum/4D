1. Number of distinct tasks evaluated: 16.
   - MQAR. (2025_SelectiveRoPE.pdf p.7)
   - MAD suite tasks: Compress Recall, Fuzzy Recall, In-Context Recall, Memorize, Noisy, Selective Copy. (2025_SelectiveRoPE.pdf p.7)
   - String copying. (2025_SelectiveRoPE.pdf p.7; 2025_SelectiveRoPE.pdf p.24)
   - State tracking on S2 and A3. (2025_SelectiveRoPE.pdf p.7)
   - Language-modeling eval tasks: LAMBADA (LMB), PIQA, HellaSwag, Wino, ARC-e, ARC-c. (2025_SelectiveRoPE.pdf p.8)

2. Number of trained model instances required to cover all tasks: 11.
   - Synthetic tasks are trained per task setup: MQAR has its own training recipe (2025_SelectiveRoPE.pdf p.22), MAD runs separate task settings with many trained models per setting (2025_SelectiveRoPE.pdf p.22), state tracking has its own training setup (2025_SelectiveRoPE.pdf p.22), and copying is trained on the copy task (2025_SelectiveRoPE.pdf p.24). This implies separate trained models for MQAR (1) + MAD tasks (6) + copying (1) + state tracking S2/A3 (2) = 10.
   - The language-modeling tasks are evaluated zero-shot using a single trained LM per architecture ("default zero-shot evaluation setup in lm-eval-harness"), so one trained model instance can cover LAMBADA/PIQA/Hella/Wino/ARC-e/ARC-c. (2025_SelectiveRoPE.pdf p.8)
   - Total models = 10 + 1 = 11.

3. Task-Model Ratio:

$$
\boxed{
\frac{16\ \text{tasks}}{11\ \text{models}} \approx 1.45
}
$$

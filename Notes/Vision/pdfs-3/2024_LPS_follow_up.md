Distinct tasks evaluated (excluding ablations/variants):
- Pattern task (synthetic 10x10 grid pattern-paste task). (2024_LPS.txt:401-403)
- ARC-AGI 2024 challenge (main evaluation domain). (2024_LPS.txt:397-398)
- String manipulation is explicitly labeled an ablation, so excluded. (2024_LPS.txt:446-448)

Number of trained model instances required to cover all tasks:
- Pattern task uses a separately trained small LPN model (1M parameters; trained for 20k steps). (2024_LPS.txt:416-419)
- ARC-AGI uses a separately trained 178M-parameter LPN on re-arc. (2024_LPS.txt:554-557)
=> 2 trained model instances.

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$

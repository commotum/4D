Number of distinct tasks evaluated: 5.
- General video understanding (Video-MME).
- Video temporal understanding (MVBench, TempCompass).
- Long video understanding (MLVU, LongVideoBench, EgoSchema).
- Long video retrieval (Video-NIAH).
- Event-based temporal reasoning tasks (EventBench).
Citations: The evaluation benchmarks section enumerates general video understanding, video temporal understanding, long video understanding, and long video retrieval as the benchmark categories used to evaluate VRoPE. (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2025_VRoPE.txt) The Appendix B.1 section adds event-based tasks via EventBench as an additional evaluation focusing on event-based temporal dependencies. (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2025_VRoPE.txt)

Number of trained model instances required to cover all tasks: 1.
Rationale: The paper describes training a single Video-LLM per backbone (pre-training + instruction-tuning) and then evaluating that model across the listed benchmarks; it does not describe task-specific heads or per-task fine-tuning for different benchmarks, so one trained model instance can be used to perform all evaluated tasks. (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2025_VRoPE.txt)

$$
\boxed{
\frac{5\ \text{tasks}}{1\ \text{model}} = 5
}
$$

1. Number of distinct tasks evaluated: 427 tasks total = 400 ARC tasks + 27 BBH tasks. Evidence: ARC defines each puzzle as a task and the ARC training/validation sets have 400 tasks each ("Each puzzle (henceforth referred to as a task) ..." and "The original training and validation sets consist of 400 tasks each." in /home/jake/Developer/4D/Notes/Vision/pdfs-3/2024_TTT.txt). BBH is "a benchmark comprising 27 challenging tasks" and "For the 27 tasks in BBH..." (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2024_TTT.txt).
2. Number of trained model instances required to cover all tasks: 427 models (one task-specific LoRA adapter per task; K adapters where K equals the number of test tasks). Evidence: "By default, we learn task-specific LoRA adapters for each ARC or BBH task at test-time. That is, we obtain K different LoRA adapters, where K is the number of test tasks." (/home/jake/Developer/4D/Notes/Vision/pdfs-3/2024_TTT.txt).
3. Task-Model Ratio:
$$
\boxed{
\frac{427\ \text{tasks}}{427\ \text{models}} = 1
}
$$

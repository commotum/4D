Number of distinct tasks evaluated: 3
- Oxford-IIIT evaluation (image classification on the Oxford-IIIT dataset). (`Notes/Vision/pdfs-3/2025_LOOPE.txt:33`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:387`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:388`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:389`)
- CIFAR-100 evaluation (image classification on the CIFAR-100 dataset). (`Notes/Vision/pdfs-3/2025_LOOPE.txt:33`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:387`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:388`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:389`)
- Three-Cell Experiment (synthetic 224x224 RGB dataset; 6-class image classification). (`Notes/Vision/pdfs-3/2025_LOOPE.txt:210`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:212`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:218`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:220`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:272`)

Number of trained model instances required to cover all tasks: 3
- The paper trains/evaluates on Oxford-IIIT, CIFAR-100, and a separate Three cell dataset (with dataset-specific training settings), so covering all tasks requires separate trained instances per task. (`Notes/Vision/pdfs-3/2025_LOOPE.txt:362`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:368`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:387`, `Notes/Vision/pdfs-3/2025_LOOPE.txt:389`)

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

1. Number of distinct tasks evaluated: 3.
   - Speech emotion recognition on IEMOCAP. Evidence: the paper evaluates a decoder on the IEMOCAP dataset and lists emotion recognition in speech as an evaluated task ("train and evaluate a decoder on the IEMOCAP ... datasets"; "emotion recognition in speech") (Section 1.3 Datasets; Abstract).
   - Text classification on AG News. Evidence: the same evaluation list includes AG News, and the abstract states text classification as an evaluated task ("train and evaluate a decoder on the IEMOCAP, AG News and CIFAR100 datasets"; "text classification") (Section 1.3 Datasets; Abstract).
   - Image classification on CIFAR100. Evidence: the evaluation list includes CIFAR100, and the abstract states image classification as an evaluated task ("train and evaluate a decoder on the IEMOCAP, AG News and CIFAR100 datasets"; "image classification") (Section 1.3 Datasets; Abstract).

2. Number of trained model instances required to cover all tasks: 3.
   - Rationale: The paper uses three distinct modality-specific pre-trained encoders (WavLM-Large for speech, Llama2-13B for text, BEiT-Large for vision) and trains a decoder per downstream dataset, so each task requires its own trained decoder (and its corresponding encoder backbone), yielding separate trained model instances per task. Evidence: "we utilize the pre-trained model weights from three distinct PTMs: (i) WavLM-Large, (ii) Llama2-13B, and (iii) BEiT-Large" and "The frozen encoder of each of the three PTM implementations ... is used to train and evaluate a decoder on the IEMOCAP, AG News and CIFAR100 datasets"; also "the encoder component remains static (frozen), allowing the focus to be on training ... the newly proposed decoder on the designated downstream task" (Section 1.4; Section 1.3).

3. Task-Model Ratio:

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

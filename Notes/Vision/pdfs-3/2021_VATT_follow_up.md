1. Number of distinct tasks evaluated: 4 (video action recognition, audio event classification, image classification, zero-shot text-to-video retrieval). (source: `2021_VATT.txt:30`, `2021_VATT.txt:31`)
2. Number of trained model instances required to cover all tasks: 4. Each task is evaluated with its own task-specific fine-tuning or head: video action recognition fine-tunes the vision Transformer, audio event classification fine-tunes the audio Transformer, image classification fine-tunes the vision Transformer on ImageNet, and zero-shot text-to-video retrieval uses the pretrained VATT-MBS video-text representation model; small datasets also train a linear classifier on frozen backbones. (source: `2021_VATT.txt:363`, `2021_VATT.txt:412`, `2021_VATT.txt:426`, `2021_VATT.txt:436`, `2021_VATT.txt:356`)
3. Task-Model Ratio:

$$
\boxed{
\frac{4\ \text{tasks}}{4\ \text{models}} = 1
}
$$

Number of distinct tasks evaluated: 3.
- The paper explicitly states it evaluates three tasks: image understanding, class-conditional image generation, and text-to-image generation. (2025_HARoPE.pdf, Section 4 Experiments: "This section evaluates HARoPE across image understanding, class-conditional image generation, and text-to-image generation.")
- The experimental setups confirm each task with task-specific models/datasets: ViT-B for image understanding, DiT-B/2 for class-conditional ImageNet generation, and text-to-image generation with FLUX and MMDiT. (2025_HARoPE.pdf, Section 4.1 Implementation/Dataset)

Number of trained model instances required to cover all tasks: 3.
- Image understanding uses a dedicated ViT-B model trained from scratch. (2025_HARoPE.pdf, Section 4.1 Implementation: "For image understanding, we train ViT-B from scratch...")
- Class-conditional ImageNet generation uses a dedicated DiT-B/2 model. (2025_HARoPE.pdf, Section 4.1 Implementation: "For class-conditional image generation, we use DiT-B/2...")
- Text-to-image generation uses a dedicated text-to-image model (FLUX.1-dev fine-tuned, and also MMDiT for a separate backbone). These are separate from the above tasks and indicate a distinct model instance for text-to-image capability. (2025_HARoPE.pdf, Section 4.1 Implementation/Dataset: "For text-to-image generation, we fine-tune the pretrained FLUX.1-dev model..." and "For MMDiT-based text-to-image generation...")
- Per the instructions, different backbones within the same task (FLUX vs MMDiT) are architectural variants, not distinct tasks; thus they do not increase the task count, and one text-to-image model instance suffices for the task capability count.

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

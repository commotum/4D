## 1. Basic Metadata

- Title: Visual Instruction Tuning
- Authors: Haotian Liu; Chunyuan Li; Qingyang Wu; Yong Jae Lee
- Year: 2023 ("arXiv:2304.08485v2 [cs.CV] 11 Dec 2023." - top of page 1)
- Venue: "37th Conference on Neural Information Processing Systems (NeurIPS 2023)." (page 1)

## 2. One-Sentence Contribution Summary

The paper introduces visual instruction tuning to build a general-purpose visual assistant by generating multimodal instruction-following data with GPT-4/ChatGPT and fine-tuning a CLIP+Vicuna-based multimodal model ("we present visual instruction-tuning, the first attempt to extend instruction-tuning to the language-image multimodal space, to pave the way towards building a general-purpose visual assistant" - Section 1 Introduction; "We develop a large multimodal model (LMM), by connecting the open-set visual encoder of CLIP [40] with the language decoder Vicuna [9], and fine-tuning end-to-end on our generated instructional vision-language data." - Section 1 Introduction).

## 3. Tasks Evaluated

Task 1: Multimodal chatbot / instruction-following (conversation, detailed description, complex reasoning)
- Task type: Generation; Reasoning / relational (for complex reasoning)
- Dataset(s): LLaVA-Bench (COCO), LLaVA-Bench (In-the-Wild)
- Domain: COCO images; diverse in-the-wild images (indoor/outdoor scenes, memes, paintings, sketches)
- Evidence:
  - "We assess the performance of LLaVA in instruction-following and visual reasoning capabilities with two primary experimental settings: multimodal chatbot and the ScienceQA dataset, respectively." (Section 5 Experiments)
  - "We randomly select 30 images from COCO-Val-2014, and for each image, we generate three types of questions (conversation, detailed description, complex reasoning)" (Section 5.1 Multimodal Chatbot)
  - "LLaVA-Bench (In-the-Wild). To evaluate the model’s capability in more challenging tasks and generalizability to novel domains, we collect a diverse set of 24 images with 60 questions in total, including indoor and outdoor scenes, memes, paintings, sketches, etc." (Section 5.1 Multimodal Chatbot)

Task 2: ScienceQA multimodal multiple-choice reasoning
- Task type: Reasoning / relational; Other: multiple-choice QA
- Dataset(s): ScienceQA
- Domain: multimodal science questions across subjects/topics/skills
- Evidence:
  - "ScienceQA [34] contains 21k multimodal multiple choice questions with rich domain diversity across 3 subjects, 26 topics, 127 categories, and 379 skills." (Section 5.2 ScienceQA)
  - "Each question is provided a context in the form of natural language or an image. The assistant provides the reasoning process in natural language and selects the answer among multiple choices." (Section 4.2 Training)

## 4. Domain and Modality Scope

- Multiple domains within the same modality: Yes. Evidence: "including indoor and outdoor scenes, memes, paintings, sketches, etc." (Section 5.1 Multimodal Chatbot)
- Multiple modalities: Yes. Evidence: "Each question is provided a context in the form of natural language or an image." (Section 4.2 Training)
- Domain generalization or cross-domain transfer claimed: Yes. Evidence: "generalizability to novel domains" (Section 5.1 Multimodal Chatbot); "Note that while these images are out-of-domain for LLaVA, LLaVA is still able to understand the scenes and follow the question instruction to provide a reasonable response." (Section 5.1 Multimodal Chatbot)

## 5. Model Sharing Across Tasks

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| LLaVA-Bench conversation questions | Yes (shared within chatbot training) | Yes (fine-tuned on LLaVA-Instruct-158K) | Not specified in the paper | "We develop a Chatbot by fine-tuning on the 158K language-image instruction-following data in Section 3. Among the three types of responses, conversation is multi-turn while the other two are single-turn. They are uniformly sampled in training." (Section 4.2 Training) |
| LLaVA-Bench detailed description questions | Yes (shared within chatbot training) | Yes (fine-tuned on LLaVA-Instruct-158K) | Not specified in the paper | Same evidence as above (Section 4.2 Training) |
| LLaVA-Bench complex reasoning questions | Yes (shared within chatbot training) | Yes (fine-tuned on LLaVA-Instruct-158K) | Not specified in the paper | Same evidence as above (Section 4.2 Training) |
| ScienceQA multiple-choice reasoning | Not specified across tasks; described as a separate fine-tuning scenario | Yes (fine-tuned on ScienceQA) | Not specified in the paper | "We consider two specific use case scenarios:" (Section 4.2 Training); "Science QA. We study our method on the ScienceQA benchmark [34], the first large-scale multimodal science question dataset that annotates the answers with detailed lectures and explanations." (Section 4.2 Training); "For training in (2), we organize the data as a single turn conversation, the question & context as Xinstruct , and reasoning & answer as Xa ." (Section 4.2 Training) |

## 6. Input and Representation Constraints

- Vision encoder / representation: "For an input image Xv , we consider the pre-trained CLIP visual encoder ViT-L/14 [40], which provides the visual feature Zv = g(Xv )." (Section 4.1 Architecture)
- Projection into LLM token space: "We consider a simple linear layer to connect image features into the word embedding space. Specifically, we apply a trainable projection matrix W to convert Zv into language embedding tokens Hv , which have the same dimensionality as the word embedding space in the language model:" (Section 4.1 Architecture)
- Grid features: "The grid features before and after the last Transformer layer are considered in our experiments." (Section 4.1 Architecture)
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D): The paper only states that Hv has "the same dimensionality as the word embedding space in the language model" (Section 4.1 Architecture); no other dimensionality constraints are specified.
- Padding or resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure

- Maximum sequence length: Not specified in the paper.
- Fixed vs variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage attention cost (windowing/pooling/token pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)

- Positional encoding mechanism: Not specified in the paper.
- Where it is applied (input only/every layer/attention bias): Not specified in the paper.
- Fixed/modified/ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable

- Treated as core research variable vs fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- PE choice claimed as not critical/secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking (Scale vs Structure)

- Model sizes: "We keep all configurations the same as our best 13B model, and train a 7B model. This yields 89.84% accuracy, which is 1.08% lower than 90.92%, demonstrating the importance of model scale." (Section 5.2 ScienceQA)
- Dataset sizes:
  - "We collect 158K unique language-image instruction-following samples in total, including 58K in conversations, 23K in detailed description, and 77k in complex reasoning, respectively." (Section 3 GPT-assisted Visual Instruction Data Generation)
  - "we filter CC3M to 595K image-text pairs." (Section 4.2 Training)
  - "ScienceQA [34] contains 21k multimodal multiple choice questions..." (Section 5.2 ScienceQA)
- Performance gains attributed to scale/data/training:
  - "The 5.11% absolute degradation indicates the importance of our pre-training stage, in aligning multimodal features while preserving the vast pre-trained knowledge." (Section 5.2 ScienceQA)
  - "This yields 89.84% accuracy, which is 1.08% lower than 90.92%, demonstrating the importance of model scale." (Section 5.2 ScienceQA)

## 11. Architectural Workarounds

- Lightweight vision-language connector: "We consider a simple linear layer to connect image features into the word embedding space." (Section 4.1 Architecture)
- Two-stage tuning (feature alignment then end-to-end fine-tuning): "For LLaVA model training, we consider a two-stage instruction-tuning procedure." (Section 4.2 Training)
- Frozen vision encoder during fine-tuning: "We always keep the visual encoder weights frozen" (Section 4.2 Training)
- Training-time memory saving: "During finetuning, FSDP (Full Shard Data Parallel) and gradient checkpointing is used to save GPU memory" (Appendix C Training Details)

## 12. Explicit Limitations and Non-Claims

- Limitations / weaknesses:
  - "to correctly answer the name of the restaurant, it requires the model to have a large knowledge coverage and multilingual understanding capability; to correctly describe the side dishes, the model may need to retrieve relevant multimodal information from Internet." (Section 5.1 Multimodal Chatbot, Limitations)
  - "We also observed an interesting failure of LLaVA, as it responds with yes when asked if strawberry-flavored yogurt is present, even though the fridge contains only yogurt and strawberries." (Section 5.1 Multimodal Chatbot, Limitations)
- Future work:
  - "We leave exploring possibly more effective and sophisticated architecture designs for LLaVA as future work." (Section 4.1 Architecture)
- Scope statements / non-claims:
  - "This paper is an initial step in visual instruction tuning, and mainly focuses on real-life tasks." (Section 6 Conclusion)
  - Explicit statements about not attempting open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

## 13. Constraint Profile (Synthesis)

- Domain scope: Multiple image domains (COCO and in-the-wild including "indoor and outdoor scenes, memes, paintings, sketches, etc.") plus ScienceQA with text or image context (Sections 5.1, 4.2).
- Task structure: Instruction-following chatbot tasks (conversation, detailed description, complex reasoning) and ScienceQA multiple-choice reasoning (Sections 5.1, 5.2).
- Representation rigidity: Uses CLIP ViT-L/14 features projected into LLM token space; other input constraints (resolution, patch size, token count) are not specified (Section 4.1).
- Model sharing vs specialization: Chatbot tasks share one fine-tuned model on LLaVA-Instruct-158K; ScienceQA is described as a separate fine-tuning scenario (Section 4.2).
- Positional encoding: Not specified in the paper.

## 14. Final Classification

Classification: Multi-task, multi-domain (constrained)

Justification: The paper evaluates multiple tasks (multimodal chatbot with conversation/detailed description/complex reasoning and ScienceQA multiple-choice reasoning) and spans multiple domains (COCO plus in-the-wild images including "indoor and outdoor scenes, memes, paintings, sketches, etc.", and ScienceQA across subjects). At the same time, the evaluation remains constrained to specific benchmarks and fine-tuning scenarios rather than open-ended, unrestrained multi-task learning.

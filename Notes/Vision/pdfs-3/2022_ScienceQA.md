## 1. Basic Metadata
- Title: "Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering" (Title block, p.1)
- Authors: "Pan Lu1,3 , Swaroop Mishra2,3 , Tony Xia1 , Liang Qiu1 , Kai-Wei Chang1 , Song-Chun Zhu1 , Oyvind Tafjord3 , Peter Clark3 , Ashwin Kalyan3" (Author block, p.1)
- Year: 2022 ("arXiv:2209.09513v2 [cs.CL] 17 Oct 2022"; p.1)
- Venue: "36th Conference on Neural Information Processing Systems (NeurIPS 2022)." (footer, p.1); also arXiv

## 2. One-Sentence Contribution Summary
The paper introduces the ScienceQA (S CIENCE QA) multimodal multiple-choice science QA benchmark with lectures and explanations, and shows that generating chain-of-thought lectures and explanations improves science question answering performance.

## 3. Tasks Evaluated

Task 1: Science Question Answering (S CIENCE QA)
- Task type: Classification (multiple-choice question answering)
- Dataset(s): ScienceQA (S CIENCE QA)
- Domain: Science curriculum with multimodal contexts (images and text) across natural, social, and language science subjects
- Evidence: "We collect S CIENCE QA, which is a multimodal multiple-choice science question dataset containing 21,208 examples." (Section 3 Dataset, p.3)
- Evidence: "Given the science question and multimodal contexts, the task is to select the correct answer from multiple options." (Section 3 Dataset, p.3)
- Evidence: "S CIENCE QA covers diverse topics across three subjects: natural science, social science, and language science." (Section 3 Dataset, p.3)
- Evidence: "The image context is in the format of diagrams or natural images, which visualize the critical scenario necessary for question answering or simply illustrate the question for better understanding. Similarly, the textual context can provide either semantically rich information or a simple hint to the question." (Section 3.1 Data Analysis, p.5)

Task 2: Lecture and Explanation Generation (Chain-of-Thought)
- Task type: Generation (text generation of lecture and explanation alongside the answer)
- Dataset(s): ScienceQA (S CIENCE QA)
- Domain: Same ScienceQA multimodal science curriculum contexts
- Evidence: "We further design language models to learn to generate lectures and explanations as the chain of thought (CoT) to mimic the multi-hop reasoning process when answering S CIENCE QA questions." (Abstract, p.1)
- Evidence: "UnifiedQA is fine-tuned to generate a long sequence of text which consists of the answer followed by the lecture and explanation." (Section 4.2, p.6)
- Evidence: "UnifiedQA and GPT-3 treat S CIENCE QA as a text generation problem." (Section 5.1 Experimental Setup, p.7)
- Evidence: "The generated lectures and explanations are evaluated by automatic metrics [44, 28, 49] and human scores by annotators." (Section 5.1 Experimental Setup, p.7)

## 4. Domain and Modality Scope
- Multiple domains within the same modality: Yes. "S CIENCE QA covers diverse topics across three subjects: natural science, social science, and language science." (Section 3 Dataset, p.3)
- Multiple modalities: Yes. "We collect S CIENCE QA, which is a multimodal multiple-choice science question dataset" and "The image context is in the format of diagrams or natural images... Similarly, the textual context can provide either semantically rich information or a simple hint to the question." (Section 3 Dataset, p.3; Section 3.1 Data Analysis, p.5)
- Domain generalization or cross-domain transfer claims: Not claimed in the paper.

## 5. Model Sharing Across Tasks
The paper uses separate model variants/formats for answer-only QA versus answer+lecture+explanation generation; it does not describe joint multi-task training or explicit shared-weight constraints across these tasks.

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Answer selection (multiple-choice QA) | Not specified across tasks; baselines are trained separately from CoT variants | Yes for VQA baselines and UnifiedQA; GPT-3 is prompted only | VQA baselines use a linear classifier head; others not specified | "These VQA baselines take the question, the context, and choices as the textual input, take the image as the visual input, and predict the score distribution over choice candidates via a linear classifier." (Section 4.1 Baselines, p.6) |
| Answer + lecture + explanation generation (CoT) | Not specified across tasks; UnifiedQA (CoT) is a separate fine-tuned format | Yes for UnifiedQA (CoT); GPT-3 uses CoT prompting | Not specified | "UnifiedQA is fine-tuned to generate a long sequence of text which consists of the answer followed by the lecture and explanation." (Section 4.2, p.6) |

## 6. Input and Representation Constraints
- Maximum input length for VQA baselines: "Input sizes: For VQA baselines, we set the maximum number of input words or tokens as 100." (Appendix B.1, p.18)
- Image captioning constraints for visual context: "Captioning model. We use the tool2 to generate captions for the images in the dataset. The maximum length of generated captions is 16, the number of beams is 4, and the maximum number of output tokens is 512." (Appendix B.1, p.18)
- Newline formatting constraint for language models: "Newline character. For language models, the newline separators (\n) in the text are replaced with \\n when encoding the inputs because \n is normally used as a stop symbol, following the original works [5, 19]." (Appendix B.1, p.18)
- No-context handling: "Questions without any context. For questions without any context, the context text is replaced with an empty string." (Appendix B.1, p.18)
- Fixed/variable input resolution, fixed patch size, fixed number of tokens, fixed dimensionality, padding/resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure
- Maximum sequence length: "Input sizes: For VQA baselines, we set the maximum number of input words or tokens as 100." (Appendix B.1, p.18)
- Fixed vs variable length: Not specified in the paper beyond the maximum length for VQA baselines.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage attention cost (windowing, pooling, token pruning, etc.): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
Not specified in the paper.

## 9. Positional Encoding as a Variable
Not specified in the paper.

## 10. Evidence of Constraint Masking
- Model size(s): Not specified in the paper.
- Dataset size(s): "We collect S CIENCE QA, which is a multimodal multiple-choice science question dataset containing 21,208 examples." (Section 3 Dataset, p.3)
- Performance gains attributed to chain-of-thought/explanations (not model or data scaling): "S CIENCE QA demonstrates the utility of CoT in language models, as CoT improves the question answering performance by 1.20% in few-shot GPT-3 and 3.99% in fine-tuned UnifiedQA." (Abstract, p.1)
- Upper-bound effect of explanations: "we observe that it improves the few-shot performance of GPT-3 by 18.96%." (Abstract, p.1)
- Data efficiency claim: "Our analysis further shows that language models, similar to humans, benefit from explanations to learn from fewer data and achieve the same performance with just 40% of the data." (Abstract, p.1)

## 11. Architectural Workarounds
- Image-to-caption conversion for language models: "We extract the caption from the captioning model based on ViT [8] and GPT-2 [47] for the image as the visual context." (Section 4.1, p.6)
- CoT output format for UnifiedQA: "UnifiedQA is fine-tuned to generate a long sequence of text which consists of the answer followed by the lecture and explanation." (Section 4.2, p.6)
- Chain-of-thought prompting for GPT-3: "we build GPT-3 via chain-of-thought (CoT) prompting" (Section 4.2, p.6)

## 12. Explicit Limitations and Non-Claims
- Limitation: "GPT-3 via chain-of-chain prompting obtains promising results but still fails to answer a wide range of challenging questions in S CIENCE QA." (Section 5.4 Error analysis, p.10)
- Limitation details: "The failure cases can be classified into two types: (a) the model fails to understand the multimodal inputs and lacks domain-specific knowledge to arrive at the correct answer; (b) the model generates the wrong chain of thought with irrelevant, incorrect, or incomplete information." (Section 5.4 Error analysis, p.10)
- Limitation: "GPT-3 (CoT) is able to predict the correct answers but fails to generate gold explanations." (Appendix B.4, p.19)
- Limitation: "these captions lack fine-grained semantics and usually do not work well for diagrams" (Appendix B.4, p.20)
- Explicit statements about not attempting open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

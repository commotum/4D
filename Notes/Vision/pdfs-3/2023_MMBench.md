### 1. Basic Metadata
- Title: "MMBench: Is Your Multi-modal Model an All-around\nPlayer?" (Title page)
- Authors: "Yuan Liu1,∗ , Haodong Duan1,∗,‡ , Yuanhan Zhang2,∗ , Bo Li2,∗ , Songyang Zhang1,∗ ,\nWangbo Zhao4 , Yike Yuan5 , Jiaqi Wang1 , Conghui He1 , Ziwei Liu2,† , Kai Chen1,†\nDahua Lin1,3,†" (Title page)
- Year: "arXiv:2307.06281v5 [cs.CV] 20 Aug 2024" (Title page)
- Venue: "arXiv:2307.06281v5 [cs.CV]" and "Technical report" (Title page)

### 2. One-Sentence Contribution Summary
The paper proposes "MMBench, a bilingual benchmark for assessing the multi-modal capabilities of VLMs" and introduces a "rigorous\nCircularEval strategy" within a comprehensive evaluation pipeline (Abstract).

### 3. Tasks Evaluated
MMBench tasks (20 leaf abilities):

| Task | Task Type | Dataset(s) Used | Domain | Evidence (verbatim task quote + section) |
| --- | --- | --- | --- | --- |
| Image Style | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Image Style: Determine which type of image it belongs to, such as photos, paintings, CT scans,\netc." (Appendix A.1, Definition about Each Leaf Ability) |
| Image Scene | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Image Scene: Determine which environment is shown in the image, such as indoors, outdoors,\nforest, city, mountains, waterfront, sunny day, rainy day, etc." (Appendix A.1) |
| Image Emotion | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Image Emotion: Determine which subjective emotion is conveyed by the overall image, such as\ncold, cheerful, sad, or oppressive." (Appendix A.1) |
| Image Quality | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Image Quality: Determine the objective quality of the image, such as whether it is blurry, bright\nor dark, contrast, etc." (Appendix A.1) |
| Image Topic | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Image Topic: Determine what the subject of the image is, such as scenery, portrait, close-up of an\nobject, text, etc." (Appendix A.1) |
| Object Localization | Detection; Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Object Localization: For a single object, determine its position in the image (such as top, bottom,\netc.), its absolute coordinates in the image, count the number of objects, and the orientation of the\nobject." (Appendix A.1) |
| Attribute Recognition | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Attribute Recognition: Recognition of texture, shape, appearance characteristics, emotions,\ncategory." (Appendix A.1) |
| Celebrity Recognition | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Celebrity Recognition: Recognition of celebrities, landmarks, and well-known objects." (Appendix A.1) |
| OCR | Other (OCR); Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "OCR: Recognition of text, formula, and sheet in the image." (Appendix A.1) |
| Spatial Relationship | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Spatial Relationship: Determine the relative position between objects in image." (Appendix A.1) |
| Attribute Comparison | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Attribute Comparison: Compare attributes of different objects in image, such as shape, color,\netc." (Appendix A.1) |
| Action Recognition | Classification | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Action Recognition: Recognizing human actions, including pose motion, human-object interaction,\nand human-human interaction." (Appendix A.1) |
| Physical Property Reasoning | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Physical Property Reasoning: Predict the physical property of an object. Examples: he physical\nproperty of concentrated sulfuric acid is that it is volatile, the physical property of water is its\nfluidity, etc." (Appendix A.1) |
| Function Reasoning | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Function Reasoning: Predict the function of an object. Examples: the function of a broom is to\nsweep the floor, the function of a spatula is to cook, the function of a pen is to write, etc." (Appendix A.1) |
| Identity Reasoning | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Identity Reasoning: Predict the identity of a person. Example: by observing a person’s clothing\nand appearance, one may infer his / her occupation." (Appendix A.1) |
| Social Relation | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Social Relation: Relations in human society or relations defined from the human perspective.\nExamples: Inter-person relations, such as father and son, husband and wife, friend, hostile, etc." (Appendix A.1) |
| Physical Relation | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Physical Relation: All relationships that exist in the physical world, 3D spatial relationships and\nthe connections between objects are." (Appendix A.1) |
| Nature Relation | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Nature Relation: Other abstract relationships that exist in nature. Examples: predation, symbiosis,\ncoexistence, etc." (Appendix A.1) |
| Structuralized Image-Text Understanding | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Structuralized Image-Text Understanding: Structured understanding of images and text, including\nparsing the content of charts (such as the trends of multiple bars in a bar chart), understanding\nthe code in an image, etc." (Appendix A.1) |
| Future Prediction | Reasoning / relational | MMBench (dev/test), MMBench-CN | Images from multiple sources | "Future Prediction: Predict what will happen in the future. Examples: if it is thundering in the\nsky now, it can be predicted that it will rain soon (physical phenomenon); if someone raises their\nfist, it means they are going to hit someone (event occurrence); if someone’s face becomes serious,\nit means they are going to get angry (emotional change)." (Appendix A.1) |

Additional evaluation tasks used for validating choice extraction:

| Task | Task Type | Dataset(s) Used | Domain | Evidence (verbatim task quote + section) |
| --- | --- | --- | --- | --- |
| Visual Question Answering (GQA) | Reasoning / relational; Classification | GQA | Images | "Visual question answering datasets, such as GQA [23], OK-VQA [35], VQAv2 [19], and Vizwiz [20],\ncontain question-answer pairs related to the given image, used to measure the model’s ability on visual\nperception and reasoning." (Section 2.1)
"we also validate the LLM-involved evaluation paradigm on existing multi-modality tasks, including\nGQA [23], OK-VQA [35], and Text-VQA [42]." (Appendix C) |
| Visual Question Answering (OK-VQA) | Reasoning / relational; Classification | OK-VQA | Images | "Visual question answering datasets, such as GQA [23], OK-VQA [35], VQAv2 [19], and Vizwiz [20],\ncontain question-answer pairs related to the given image, used to measure the model’s ability on visual\nperception and reasoning." (Section 2.1)
"we also validate the LLM-involved evaluation paradigm on existing multi-modality tasks, including\nGQA [23], OK-VQA [35], and Text-VQA [42]." (Appendix C) |
| Visual Question Answering (TextVQA) | Reasoning / relational; Other (OCR) | TextVQA | Images with text | "TextVQA [42] proposes questions about text shown in the image, thus involving the OCR task in\nquestion-answering." (Section 2.1)
"we also validate the LLM-involved evaluation paradigm on existing multi-modality tasks, including\nGQA [23], OK-VQA [35], and Text-VQA [42]." (Appendix C) |

Dataset/domain evidence for MMBench tasks (applies to all 20 leaf-ability tasks above):
- "Currently, MMBench contains over 3000 multiple-choice questions covering 20 different ability\ndimensions" (Introduction)
- "MMBench-CN enables an apple-to-apple comparison of VLM performance under English and\nChinese contexts." (Section 3.2)
- "The data — including images, choices, and questions — are manually collected from multiple sources" (Section 3.2)
- "Ii corresponds to the image associated with the question" (Section 3.2)

### 4. Domain and Modality Scope
- Single domain vs multiple domains: Multiple domains within the same visual modality, since "MMBench adopts images / problems from various sources" and "more than 80%\nof questions in MMBench are collected from the Internet." (Section 3; Section 3.2)
- Multiple modalities: Yes. "MMBench is a bilingual multi-modal benchmark" and uses images plus text questions/choices (Section 3; Section 3.2).
- Domain generalization / cross-domain transfer claims: Not claimed in the paper.

### 5. Model Sharing Across Tasks
Evidence for shared evaluation setup: "For a fair comparison, we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1)

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| Image Style | Yes (same model evaluated across tasks) | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Image Scene | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Image Emotion | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Image Quality | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Image Topic | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Object Localization | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Attribute Recognition | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Celebrity Recognition | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| OCR | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Spatial Relationship | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Attribute Comparison | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Action Recognition | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Physical Property Reasoning | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Function Reasoning | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Identity Reasoning | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Social Relation | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Physical Relation | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Nature Relation | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Structuralized Image-Text Understanding | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Future Prediction | Yes | Not specified in the paper (zero-shot evaluation) | Not specified in the paper | "we adopt the zero-shot setting to infer MMBench questions with all VLMs, based on the same prompt." (Section 5.1) |
| Visual Question Answering (GQA) | Not specified in the paper | Not specified in the paper | Not specified in the paper | "we also validate the LLM-involved evaluation paradigm on existing multi-modality tasks, including\nGQA [23], OK-VQA [35], and Text-VQA [42]." (Appendix C) |
| Visual Question Answering (OK-VQA) | Not specified in the paper | Not specified in the paper | Not specified in the paper | "we also validate the LLM-involved evaluation paradigm on existing multi-modality tasks, including\nGQA [23], OK-VQA [35], and Text-VQA [42]." (Appendix C) |
| Visual Question Answering (TextVQA) | Not specified in the paper | Not specified in the paper | Not specified in the paper | "we also validate the LLM-involved evaluation paradigm on existing multi-modality tasks, including\nGQA [23], OK-VQA [35], and Text-VQA [42]." (Appendix C) |

### 6. Input and Representation Constraints
- Multiple-choice structure (explicit constraint): "A problem Pi corresponds to a quadruple (Qi , Ci , Ii , Ai ). Qi\ndenotes the question, Ci represents a set with n (2 ≤ n ≤ 4) choices c1 , c2 , ..., cn" (Section 3.2).
- Fixed or variable input resolution: Not specified in the paper.
- Fixed patch size: Not specified in the paper.
- Fixed number of tokens: Not specified in the paper.
- Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
- Padding or resizing requirements: Not specified in the paper.

### 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed vs variable sequence length: Not specified in the paper.
- Attention type (global/windowed/hierarchical/sparse): Not specified in the paper.
- Mechanisms to manage computational cost (windowing, pooling, token pruning): Not specified in the paper.

### 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Not specified in the paper.
- Where applied: Not specified in the paper.
- Fixed/modified/ablated: Not specified in the paper.

### 9. Positional Encoding as a Variable
- Core research variable vs fixed assumption: Not specified in the paper.
- Multiple positional encodings compared: Not specified in the paper.
- Claims that PE choice is not critical/secondary: Not specified in the paper.

### 10. Evidence of Constraint Masking
- Model sizes (examples, Table 9):
  > OpenFlamingov2[4]
  >
  > MPT 7B
  >
  > CLIP ViT-L/14
  >
  > 9B
  >
  > MiniGPT-4-7B[56]
  >
  > Vicuna 7B
  >
  > EVA-G
  >
  > 8B
  >
  > LLaVA-InternLM2-20B[10]
  >
  > InternLM2-20B
  >
  > CLIP ViT-L/14
  >
  > 23B
- Dataset size: "we have gathered a total of 3,217 data samples spanning across 20 distinct L-3 abilities" (Section 3.3) and
  "MMBench contains over 3000 multiple-choice questions covering 20 different ability dimensions" (Introduction).
- Performance gains attributed to scaling model size:
  - "switching the LLM from Vicuna-v1.5 [53] to the more powerful InternLM2-20B [45] leads to steady improvement across all\nL-2 capabilities" (Section 5.2).
  - "The scaling also holds for variants with different sizes from the same LLM family. By adopting the 13B variant of Vicuna\nrather than the 7B variant, VLMs in the MiniGPT, InstructBLIP, and LLaVA v1.5 series outperform their 7B counterparts by\n8.3%, 1.5%, and 3.5% overall Top-1 accuracies on the MMBench-test split, respectively." (Section 5.2)
- Performance gains attributed to scaling data: Not specified in the paper.
- Performance gains attributed to architectural hierarchy or training tricks: Not specified in the paper.

### 11. Architectural Workarounds
- No model architecture is introduced; the paper focuses on evaluation. The main workarounds are evaluation pipeline design:
  - "We introduce a novel circular evaluation strategy (CircularEval) to improve the robustness of our evaluation process." (Introduction)
  - "GPT-4 is employed to match the model’s prediction with given choices" to handle instruction-following variability (Introduction).

### 12. Explicit Limitations and Non-Claims
- Stated limitations of current VLMs (as found in MMBench): "we find that all existing VLMs have the following limitations: 1. Poor\nat recognizing the low-level features on visual inputs, i.e., they cannot accurately recognize and compare the brightness,\nsharpness, contrast ratio, or artifacts of images. 2. Difficulty in understanding structuralized visual inputs like tables,\ndiagrams, or layouts, even for relatively simple cases like Figure 10(b); 3. Perform badly on recognizing or reasoning about\nthe inter-object spatial relationships, either in 2D or 3D space." (Section 5.3)
- Explicit non-claim about training use: "images are gathered from the validation set of public datasets (if they exist) while\nthe questions are self-constructed, which is not supposed to be used for training." (Section 3.2)
- Statements about not attempting open-world or unrestrained multi-task learning: Not specified in the paper.

### 13. Constraint Profile (Synthesis)
- Domain scope: Multi-source image domain (Internet + public datasets) with bilingual text; not a cross-domain transfer study.
- Task structure: 20 fixed, predefined multiple-choice tasks spanning perception and reasoning; evaluation is tightly curated.
- Representation rigidity: Inputs are fixed as (question, choices, image, answer) with 2-4 choices; no model-side representation constraints are specified.
- Model sharing vs specialization: Same pretrained models are evaluated zero-shot across all tasks with a shared prompt; no per-task training described.
- Role of positional encoding: Not discussed; treated as out of scope for this benchmark.

### 14. Final Classification
**Multi-task, multi-domain (constrained).** The benchmark evaluates "over 3000 multiple-choice questions covering 20 different ability\ndimensions" and pulls "images / problems from various sources" (Introduction; Section 3), which indicates multiple tasks and varied image sources. The evaluation is still constrained to a fixed multiple-choice, image+text format, and the paper does not claim open-world or unrestrained multi-task learning.

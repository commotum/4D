# Gemini: Survey Responses (2023_Gemini.pdf)

## 1. Basic Metadata

Title: Gemini: A Family of Highly Capable Multimodal Models.
Evidence (Front matter): "Gemini: A Family of Highly Capable Multimodal Models"

Authors: Gemini Team, Google.
Evidence (Front matter): "Gemini Team, Google"

Year: 2025.
Evidence (Front matter): "arXiv:2312.11805v5 [cs.CL] 9 May 2025"

Venue: arXiv.
Evidence (Front matter): "arXiv:2312.11805v5 [cs.CL] 9 May 2025"

## 2. One-Sentence Contribution Summary

The paper introduces Gemini, a family of multimodal models trained jointly on image, audio, video, and text to achieve strong generalist understanding and reasoning across modalities.
Evidence (Abstract): "This report introduces a new family of multimodal models, Gemini, that exhibit remarkable capabilities across image, audio, video, and text understanding."
Evidence (Section 1 Introduction): "We trained Gemini models jointly across image, audio, video, and text data for the purpose of building a model with both strong generalist capabilities across modalities alongside cutting-edge understanding and reasoning performance in each respective domain."

## 3. Tasks Evaluated

Evidence (Section 1 Introduction): "We evaluate the performance of pre- and post-trained Gemini models on a comprehensive suite of internal and external benchmarks covering a wide range of language, coding, reasoning, and multimodal tasks."
Evidence (Section 10.3): "We provide a detailed list of benchmarking tasks for six different capabilities in text understanding and generation: factuality, long context, math/science, reasoning, summarization, and multilinguality."

### 3.1 Factuality Benchmarks (Text)
Evidence (Section 10.3): "Factuality: We use 5 benchmarks: BoolQ (Clark et al., 2019), NaturalQuestions-Closed (Kwiatkowski et al., 2019a), NaturalQuestions-Retrieved (Kwiatkowski et al., 2019a), Real- timeQA (Kasai et al., 2022b), TydiQA-noContext and TydiQA-goldP (Clark et al., 2020)."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| BoolQ | Other (factuality benchmark) | BoolQ | Not specified in the paper. | Evidence quote above (Section 10.3). |
| NaturalQuestions-Closed | Other (factuality benchmark) | NaturalQuestions-Closed | Not specified in the paper. | Evidence quote above (Section 10.3). |
| NaturalQuestions-Retrieved | Other (factuality benchmark) | NaturalQuestions-Retrieved | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Real-timeQA | Other (factuality benchmark) | Real-timeQA | Not specified in the paper. | Evidence quote above (Section 10.3). |
| TydiQA-noContext | Other (factuality benchmark) | TydiQA-noContext | Not specified in the paper. | Evidence quote above (Section 10.3). |
| TydiQA-goldP | Other (factuality benchmark) | TydiQA-goldP | Not specified in the paper. | Evidence quote above (Section 10.3). |

### 3.2 Long-Context Benchmarks (Text)
Evidence (Section 10.3): "Long Context: We use 6 benchmarks: NarrativeQA (Kočiský et al., 2018), Scrolls-Qasper, Scrolls-Quality (Shaham et al., 2022), XLsum (En), XLSum (non-English languages) (Hasan et al., 2021), and one other internal benchmark."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| NarrativeQA | Other (long-context benchmark) | NarrativeQA | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Scrolls-Qasper | Other (long-context benchmark) | Scrolls-Qasper | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Scrolls-Quality | Other (long-context benchmark) | Scrolls-Quality | Not specified in the paper. | Evidence quote above (Section 10.3). |
| XLsum (En) | Other (long-context benchmark) | XLsum (En) | Not specified in the paper. | Evidence quote above (Section 10.3). |
| XLSum (non-English languages) | Other (long-context benchmark) | XLSum (non-English languages) | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Internal long-context benchmark (unnamed) | Other (long-context benchmark) | Not specified in the paper. | Not specified in the paper. | Evidence quote above (Section 10.3). |

### 3.3 Math/Science Benchmarks (Text)
Evidence (Section 10.3): "Math/Science: We use 8 benchmarks: GSM8k (with CoT) (Cobbe et al., 2021), Hendryck’s MATH pass@1 (Hendrycks et al., 2021b), MMLU (Hendrycks et al., 2021a), Math-StackExchange, Math-AMC 2022-2023 problems, and three other internal benchmarks."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| GSM8K | Reasoning / relational | GSM8K | Not specified in the paper. | "GSM8K... Grade-school math (Cobbe et al., 2021)" (Section 5.1.1, Table 2). |
| MATH | Reasoning / relational | MATH | Not specified in the paper. | "MATH... Math problems across 5 difficulty levels & 7 subdisciplines (Hendrycks et al., 2021b)" (Section 5.1.1, Table 2). |
| MMLU | Classification (multiple-choice) | MMLU | Text exams (professional & academic). | "Multiple-choice questions in 57 subjects (professional & academic) (Hendrycks et al., 2021a)" (Section 5.1.1, Table 2). "MMLU is a holistic exam benchmark, which measures knowledge across a set of 57 subjects." (Section 5.1.1). |
| Math-StackExchange | Not specified in the paper. | Math-StackExchange | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Math-AMC 2022-2023 problems | Not specified in the paper. | Math-AMC 2022-2023 problems | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Internal math/science benchmarks (3) | Not specified in the paper. | Not specified in the paper. | Not specified in the paper. | Evidence quote above (Section 10.3). |

### 3.4 Reasoning Benchmarks (Text)
Evidence (Section 10.3): "Reasoning: We use 7 benchmarks: BigBench Hard (with CoT) (Srivastava et al., 2022; Suzgun et al., 2022), CLRS (Veličković et al., 2022), Proof Writer (Tafjord et al., 2020), Reasoning-Fermi problems (Kalyan et al., 2021), Lambada (Paperno et al., 2016), HellaSwag (Zellers et al., 2019), DROP (Dua et al., 2019)."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| BigBench Hard | Reasoning / relational | BigBench Hard | Not specified in the paper. | "BIG-Bench-Hard... Subset of hard BIG-bench tasks written as CoT problems (Srivastava et al., 2022)" (Section 5.1.1, Table 2). |
| CLRS | Reasoning / relational | CLRS | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Proof Writer | Reasoning / relational | Proof Writer | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Reasoning-Fermi problems | Reasoning / relational | Reasoning-Fermi problems | Not specified in the paper. | Evidence quote above (Section 10.3). |
| Lambada | Not specified in the paper. | Lambada | Not specified in the paper. | Evidence quote above (Section 10.3). |
| HellaSwag | Classification (multiple-choice) | HellaSwag | Not specified in the paper. | "Common-sense multiple choice questions (Zellers et al., 2019)" (Section 5.1.1, Table 2). |
| DROP | Other (reading comprehension & arithmetic) | DROP | Not specified in the paper. | "DROP... Reading comprehension & arithmetic. (Dua et al., 2019)" (Section 5.1.1, Table 2). |

### 3.5 Summarization Benchmarks (Text)
Evidence (Section 10.3): "Summarization: We use 5 benchmarks: XL Sum (English), XL Sum (non-English languages) (Hasan et al., 2021), WikiLingua (non-English languages), WikiLingua (English) (Ladhak et al., 2020), XSum (Narayan et al., 2018)."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| XL Sum (English) | Generation (summarization) | XL Sum (English) | Text. | Evidence quote above (Section 10.3). |
| XL Sum (non-English languages) | Generation (summarization) | XL Sum (non-English languages) | Text. | Evidence quote above (Section 10.3). |
| WikiLingua (non-English languages) | Generation (summarization) | WikiLingua (non-English languages) | Text. | Evidence quote above (Section 10.3). |
| WikiLingua (English) | Generation (summarization) | WikiLingua (English) | Text. | Evidence quote above (Section 10.3). |
| XSum | Generation (summarization) | XSum | Text. | Evidence quote above (Section 10.3). |

### 3.6 Multilinguality Benchmarks (Text)
Evidence (Section 10.3): "Multilinguality: We use 10 benchmarks: XLSum (Non-English languages) (Hasan et al., 2021), WMT22 (Kocmi et al., 2022), WMT23 (Tom et al., 2023), FRMT (Riley et al., 2023), WikiLingua (Non-English languages) (Ladhak et al., 2020), TydiQA (no context), TydiQA (GoldP) (Clark et al., 2020), MGSM (Shi et al., 2023), translated MMLU (Hendrycks et al., 2021a), NTREX (Federmann et al., 2022), FLORES-200 (Team et al., 2022)."
Evidence (Section 5.1.4): "These tasks include machine translation benchmarks (WMT 23 for high-medium-low resource translation; Flores, NTREX for low and very low resource languages), summarization benchmarks (XLSum, Wikilingua), and translated versions of common benchmarks (MGSM: professionally translated into 11 languages)."
Evidence (Section 5.1.4.2): "We specifically investigated the math benchmark MGSM (Shi et al., 2023), which is a translated variant of the math benchmark GSM8K (Cobbe et al., 2021)."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| XLSum (Non-English languages) | Generation (summarization) | XLSum (Non-English languages) | Text. | Evidence quote above (Section 5.1.4). |
| WMT22 | Not specified in the paper. | WMT22 | Text. | Evidence quote above (Section 10.3). |
| WMT23 | Generation (machine translation) | WMT23 | Text. | "WMT23... Machine translation (metric: BLEURT)" (Section 5.1.1, Table 2). |
| FRMT | Not specified in the paper. | FRMT | Text. | Evidence quote above (Section 10.3). |
| WikiLingua (Non-English languages) | Generation (summarization) | WikiLingua (Non-English languages) | Text. | Evidence quote above (Section 5.1.4). |
| TydiQA (no context) | Not specified in the paper. | TydiQA (no context) | Text. | Evidence quote above (Section 10.3). |
| TydiQA (GoldP) | Not specified in the paper. | TydiQA (GoldP) | Text. | Evidence quote above (Section 10.3). |
| MGSM | Reasoning / relational (math) | MGSM | Text. | Evidence quote above (Section 5.1.4.2). |
| translated MMLU | Not specified in the paper. | translated MMLU | Text. | Evidence quote above (Section 10.3). |
| NTREX | Generation (machine translation) | NTREX | Text. | Evidence quote above (Section 5.1.4). |
| FLORES-200 | Generation (machine translation) | FLORES-200 | Text. | Evidence quote above (Section 5.1.4). |

### 3.7 Additional Text/Code Benchmarks Not Explicitly Listed in Section 10.3

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| HumanEval | Generation (code) | HumanEval | Code (Python). | "HumanEval... Python coding tasks (Chen et al., 2021)" (Section 5.1.1, Table 2). "HumanEval, a standard code-completion benchmark... mapping function descriptions to Python implementations" (Section 5.1.1). |
| Natural2Code | Generation (code) | Natural2Code | Code (Python). | "Natural2Code... Python code generation. (New held-out set with no leakage on web)" (Section 5.1.1, Table 2). |
| MBPP | Not specified in the paper. | MBPP | Not specified in the paper. | "MBPP" (Section 5.1.3, Table 3). |

### 3.8 Image Understanding Benchmarks
Evidence (Section 10.3): "Image and Video: We use 9 benchmarks for image understanding: MMMU (Yue et al., 2023), TextVQA (Singh et al., 2019), DocVQA (Mathew et al., 2021), ChartQA (Masry et al., 2022), InfographicVQA (Mathew et al., 2022), MathVista (Lu et al., 2023), AI2D (Kembhavi et al., 2016), VQAv2 (Goyal et al., 2017), XM3600 (Thapliyal et al., 2022) for multi-lingual image understanding..."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| MMMU | Other (image QA / multimodal reasoning) | MMMU | Images across disciplines. | "MMMU... consists of questions about images across 6 disciplines with multiple subjects within each discipline that require college-level knowledge to solve these questions." (Section 5.2.1). |
| TextVQA | Other (text reading / VQA) | TextVQA | Natural images. | "TextVQA... Text reading on natural images (Singh et al., 2019)" (Section 5.2.1, Table 7). |
| DocVQA | Other (document understanding) | DocVQA | Documents. | "DocVQA... Document understanding (Mathew et al., 2021)" (Section 5.2.1, Table 7). |
| ChartQA | Other (chart understanding) | ChartQA | Charts. | "ChartQA... Chart understanding (Masry et al., 2022)" (Section 5.2.1, Table 7). |
| InfographicVQA | Other (infographic understanding) | InfographicVQA | Infographics. | "InfographicVQA... Infographic understanding (Mathew et al., 2022)" (Section 5.2.1, Table 7). |
| MathVista | Reasoning / relational (math) | MathVista | Visual math reasoning. | "MathVista... Mathematical reasoning (Lu et al., 2023)" (Section 5.2.1, Table 7). |
| AI2D | Other (science diagrams) | AI2D | Science diagrams. | "AI2D... Science diagrams (Kembhavi et al., 2016)" (Section 5.2.1, Table 7). |
| VQAv2 | Other (image understanding / VQA) | VQAv2 | Natural images. | "VQAv2... Natural image understanding (Goyal et al., 2017)" (Section 5.2.1, Table 7). |
| XM-3600 | Generation (image captioning) | XM-3600 | Images (multilingual captioning). | "We evaluate the performance of generating image descriptions on a selected subset of languages in the Crossmodal-3600 (XM-3600) benchmark in a 4-shot setting" (Section 5.2.1). |

### 3.9 Video Understanding Benchmarks
Evidence (Section 10.3): "...and 6 benchmarks for video understanding: VATEX (Wang et al., 2019) for captioning in two different languages, YouCook2 (Zhou et al., 2018), NextQA (Xiao et al., 2021), ActivityNet-QA (Yu et al., 2019), and Perception Test MCQA (Pătrăucean et al., 2023)."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| VATEX (English) | Generation (video captioning) | VATEX | Video. | "VATEX (test)... English video captioning (Wang et al., 2019)" (Section 5.2.2, Table 10). |
| VATEX ZH (Chinese) | Generation (video captioning) | VATEX ZH | Video. | "VATEX ZH (test)... Chinese video captioning (Wang et al., 2019)" (Section 5.2.2, Table 10). |
| YouCook2 | Generation (video captioning) | YouCook2 | Video (cooking). | "YouCook2 (val)... English cooking video captioning (Zhou et al., 2018)" (Section 5.2.2, Table 10). |
| NextQA | Other (video question answering) | NextQA | Video. | "NextQA (test)... Video question answering (Xiao et al., 2021)" (Section 5.2.2, Table 10). |
| ActivityNet-QA | Other (video question answering) | ActivityNet-QA | Video. | "ActivityNet-QA (test)... Video question answering (Yu et al., 2019)" (Section 5.2.2, Table 10). |
| Perception Test MCQA | Other (video question answering) | Perception Test MCQA | Video. | "Perception Test MCQA (test)... Video question answering (Pătrăucean et al., 2023)" (Section 5.2.2, Table 10). |

### 3.10 Audio Understanding Benchmarks
Evidence (Section 10.3): "Audio: We use 5 benchmarks including automatic speech recognition (ASR) tasks such as FLEURS (Conneau et al., 2023), VoxPopuli (Wang et al., 2021), Multi-lingual Librispeech (Pratap et al., 2020), and automatic speech translation task such as CoVoST 2 (Wang et al., 2020)."

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| YouTube (ASR) | Other (speech recognition) | YouTube test set | Audio (speech). | "We also report on an internal benchmark YouTube test set." (Section 5.2.4). "Automatic Speech Recognition... YouTube (en-us)" (Section 5.2.4, Table 11). |
| Multilingual Librispeech (ASR) | Other (speech recognition) | Multi-lingual Librispeech | Audio (speech). | "Multi-lingual Librispeech (Pratap et al., 2020)" listed under "Automatic Speech Recognition" (Section 5.2.4, Table 11). |
| FLEURS (ASR) | Other (speech recognition) | FLEURS | Audio (speech). | "FLEURS (Conneau et al., 2023)" listed under "Automatic Speech Recognition" (Section 5.2.4, Table 11). |
| VoxPopuli (ASR) | Other (speech recognition) | VoxPopuli | Audio (speech). | "VoxPopuli (Wang et al., 2021)" listed under "Automatic Speech Recognition" (Section 5.2.4, Table 11). |
| CoVoST 2 (AST) | Generation (speech translation) | CoVoST 2 | Audio (speech). | "Automatic Speech Translation... CoVoST 2 (Wang et al., 2020)" (Section 5.2.4, Table 11). |

### 3.11 Other Explicit Evaluation Tasks (Qualitative or Internal)

| Task | Task type | Dataset(s) | Domain | Evidence |
| --- | --- | --- | --- | --- |
| Synthetic retrieval test (long context) | Other (retrieval) | Not specified in the paper. | Text. | "We first verify this by running a synthetic retrieval test: we place key-value pairs at the beginning of the context, then add long filler text, and ask for value associated with a particular key." (Section 5.1.5). |
| Closed-Book Factuality | Other (factuality evaluation) | Not specified in the paper. | Text. | "Closed-Book Factuality: If provided with a fact-seeking prompt without any given source, Gemini API models should not hallucinate incorrect information" (Section 5.1.6). |
| Attribution | Other (factuality evaluation) | Not specified in the paper. | Text. | "Attribution: If instructed to generate a response grounded to a given context, we aim to ensure that Gemini API models produce a response with the highest degree of faithfulness to the context" (Section 5.1.6). |
| Hedging | Other (factuality evaluation) | Not specified in the paper. | Text. | "Hedging: If prompted with an input that is “unanswerable”, Gemini API models must acknowledge that it cannot provide a response by hedging to avoid hallucination." (Section 5.1.6). |
| Complex prompts instruction-following evaluation | Other (instruction following) | Internal dataset of complex prompts. | Text. | "Complex prompts evaluation: We investigate performance on complex prompts containing multiple instructions... Table 14 reports results on an internal dataset of prompts with instructions of varying complexity" (Section 6.5.1). |
| Tool-use internal benchmark (travel planning, video discovery) | Other (tool use) | Internal benchmark. | Text (tool-augmented). | "We created an internal benchmark to assess Gemini performance on tasks that may benefit from access to these extensions. This benchmark measures human preference in domains such as travel planning and video discovery." (Section 6.5.2). |
| Competitive programming (AlphaCode 2) | Other (competitive programming) | Codeforces contests (12 contests, 77 problems). | Code. | "AlphaCode 2 is evaluated on Codeforces... on 12 contests from division 1 and 2, for a total of 77 problems." (Section 5.1.7). |
| Image generation (few-shot) | Generation (images) | Not specified in the paper. | Images. | "Gemini models are able to output images natively... generate images with prompts using interleaved sequences of image and text in a few-shot setting." (Section 5.2.3). |

## 4. Domain and Modality Scope

Evaluation modality scope: multiple modalities (text, image, audio, video).
Evidence (Section 5): "The Gemini models are natively multimodal, as they are trained jointly across text, image, audio, and video."
Evidence (Section 1 Introduction): "We trained Gemini models jointly across image, audio, video, and text data..."

Multiple domains within modality: yes for vision inputs and image tasks.
Evidence (Section 2): "Gemini models are trained to accommodate textual input interleaved with a wide variety of audio and visual inputs, such as natural images, charts, screenshots, PDFs, and videos..."
Evidence (Section 5.2.1, Table 7): "Text reading on natural images"; "Document understanding"; "Chart understanding"; "Infographic understanding"; "Science diagrams".

Domain generalization / cross-domain transfer: Not claimed in the paper.
Evidence (Section 5): "One open question is whether this joint training can result in a model which has strong capabilities in each domain – even when compared to models and approaches that are narrowly tailored to single domains. We find this to be the case" (performance across domains is reported, but no explicit claim of domain generalization or cross-domain transfer).

## 5. Model Sharing Across Tasks

Evidence Key:
A (Section 1 Introduction): "We trained Gemini models jointly across image, audio, video, and text data..."
B (Section 5.2.1): "For zero-shot QA evaluation, the model is instructed to provide short answers aligned with the specific benchmark. All numbers are obtained using greedy sampling and without any use of external OCR tools."
C (Section 5.1.3): "These models excel in summarization and reading comprehension tasks with per-task fine-tuning."
D (Section 5.1.7): "AlphaCode 2 uses a specialized version of Gemini Pro – tuned on competitive programming data... Gemini Pro is fine-tuned both to be a coding model... and to be a reward model..."
E (Section 6.5.2 / Table 15 caption): "Gemini API Pro with tools is the same model fine-tuned with tool-use data."
F (Section 1 Introduction): "We evaluate the performance of pre- and post-trained Gemini models on a comprehensive suite of internal and external benchmarks..."
G (Section 5.2.2): "Gemini Ultra achieves state-of-the-art performance on various few-shot video captioning tasks as well as zero-shot video question answering tasks as shown in Table 10."

| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| BoolQ | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| NaturalQuestions-Closed | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| NaturalQuestions-Retrieved | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Real-timeQA | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| TydiQA-noContext | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| TydiQA-goldP | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| NarrativeQA | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| Scrolls-Qasper | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| Scrolls-Quality | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| XLsum (En) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| XLSum (non-English languages) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| Internal long-context benchmark | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| GSM8K | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| MATH | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| MMLU | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Math-StackExchange | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Math-AMC 2022-2023 problems | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Internal math/science benchmarks (3) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| BigBench Hard | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| CLRS | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Proof Writer | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Reasoning-Fermi problems | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Lambada | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| HellaSwag | Not specified in the paper (joint training across modalities stated). | Not specified in the paper (paper mentions additional fine-tuning steps for HellaSwag analysis, but task-level training is not specified). | Not specified in the paper. | A; F |
| DROP | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| XL Sum (English) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| XL Sum (non-English languages) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| WikiLingua (non-English languages) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| WikiLingua (English) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| XSum | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| XLSum (Non-English languages) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| WMT22 | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| WMT23 | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| FRMT | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| WikiLingua (Non-English languages) | Not specified in the paper (joint training across modalities stated). | Not specified; Nano models use per-task fine-tuning for summarization/reading comprehension tasks. | Not specified in the paper. | A; C; F |
| TydiQA (no context) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| TydiQA (GoldP) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| MGSM | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| translated MMLU | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| NTREX | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| FLORES-200 | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| HumanEval | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Natural2Code | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| MBPP | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| MMMU | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| TextVQA | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| DocVQA | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| ChartQA | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| InfographicVQA | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| MathVista | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| AI2D | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| VQAv2 | Not specified in the paper (joint training across modalities stated). | Not specified; evaluated zero-shot. | Not specified in the paper. | A; B; F |
| XM-3600 | Not specified in the paper (joint training across modalities stated). | Not specified (4-shot evaluation reported). | Not specified in the paper. | A; F |
| VATEX (English) | Not specified in the paper (joint training across modalities stated). | Not specified (few-shot evaluation). | Not specified in the paper. | A; G; F |
| VATEX ZH (Chinese) | Not specified in the paper (joint training across modalities stated). | Not specified (few-shot evaluation). | Not specified in the paper. | A; G; F |
| YouCook2 | Not specified in the paper (joint training across modalities stated). | Not specified (few-shot evaluation). | Not specified in the paper. | A; G; F |
| NextQA | Not specified in the paper (joint training across modalities stated). | Not specified (zero-shot evaluation). | Not specified in the paper. | A; G; F |
| ActivityNet-QA | Not specified in the paper (joint training across modalities stated). | Not specified (zero-shot evaluation). | Not specified in the paper. | A; G; F |
| Perception Test MCQA | Not specified in the paper (joint training across modalities stated). | Not specified (zero-shot evaluation). | Not specified in the paper. | A; G; F |
| YouTube (ASR) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Multilingual Librispeech (ASR) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| FLEURS (ASR) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| VoxPopuli (ASR) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| CoVoST 2 (AST) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Synthetic retrieval test (long context) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Closed-Book Factuality | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Attribution | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Hedging | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |
| Complex prompts instruction-following evaluation | Not specified in the paper (joint training across modalities stated). | Not specified in the paper (post-training uses SFT/RLHF broadly; task-specific heads not mentioned). | Not specified in the paper. | A; F |
| Tool-use internal benchmark | Not specified in the paper (joint training across modalities stated). | Yes, fine-tuned with tool-use data. | Not specified in the paper. | A; E |
| Competitive programming (AlphaCode 2) | Specialized weights (task-specific tuned model). | Yes, tuned on competitive programming data. | Not specified in the paper. | D |
| Image generation (few-shot) | Not specified in the paper (joint training across modalities stated). | Not specified in the paper. | Not specified in the paper. | A; F |

## 6. Input and Representation Constraints

Input modality constraints (explicit):
Evidence (Section 2): "Gemini models are trained to accommodate textual input interleaved with a wide variety of audio and visual inputs, such as natural images, charts, screenshots, PDFs, and videos, and they can produce text and image outputs."
Evidence (Section 2): "Video understanding is accomplished by encoding the video as a sequence of frames in the large context window. Video frames or images can be interleaved naturally with text or audio as part of the model input."
Evidence (Section 2): "The models can handle variable input resolution in order to spend more compute on tasks that require fine-grained understanding."
Evidence (Section 2): "Gemini models can directly ingest audio signals at 16kHz from Universal Speech Model (USM) (Zhang et al., 2023) features."
Evidence (Section 5.2.2): "For each video task, we sample 16 equally-spaced frames from each video clip and feed them to the Gemini models."

Fixed or variable input resolution: Variable (explicit).
Fixed patch size: Not specified in the paper.
Fixed number of tokens: Not specified in the paper (see Section 7 for context length). 
Fixed dimensionality (e.g., strictly 2D): Not specified in the paper.
Padding or resizing requirements: Not specified in the paper.

## 7. Context Window and Attention Structure

Maximum sequence length: 32,768 tokens.
Evidence (Section 5.1.5): "Gemini models are trained with a sequence length of 32,768 tokens and we find that they make use of their context length effectively."
Evidence (Section 2): "They are trained to support 32k context length, employing efficient attention mechanisms (for e.g. multi-query attention (Shazeer, 2019a))."

Fixed or variable sequence length: Trained with a sequence length of 32,768 tokens; variable sequence length is not specified in the paper.
Attention type: Efficient attention; multi-query attention mentioned. Windowed/hierarchical/sparse attention is not specified in the paper.
Mechanisms to manage computational cost: "efficient attention mechanisms (for e.g. multi-query attention (Shazeer, 2019a))" (Section 2).

## 8. Positional Encoding (Critical Section)

Positional encoding mechanism: Not specified in the paper.
Where applied: Not specified in the paper.
Fixed vs modified vs ablated: Not specified in the paper.

## 9. Positional Encoding as a Variable

Positional encoding treated as a core research variable or fixed assumption: Not specified in the paper.
Multiple positional encodings compared: Not specified in the paper.
Claims that PE choice is “not critical” or secondary: Not specified in the paper.

## 10. Evidence of Constraint Masking

Model sizes:
Evidence (Table 1): "We trained two versions of Nano, with 1.8B (Nano-1) and 3.25B (Nano-2) parameters... It is trained by distilling from larger Gemini models. It is 4-bit quantized for deployment"

Dataset sizes:
Not specified in the paper.
Evidence (Model card, Section 10.1): "Compute Requirements       Not reported." (no dataset size reported; token counts not provided).

Performance gains attributed to scaling model size:
Evidence (Section 5.1.2): "We observe consistent quality gains with increased model size in Figure 3, especially in reasoning, math/science, summarization and long-context."

Performance gains attributed to training/evaluation tricks:
Evidence (Section 5.1.1): "We find Gemini Ultra achieves highest accuracy when used in combination with a chain-of-thought prompting approach (Wei et al., 2022b) that accounts for model uncertainty."
Evidence (Section 5.1.1): "Gemini Ultra reaches 94.4% accuracy with chain-of-thought prompting and self-consistency (Wang et al., 2022)" (GSM8K).

Architectural hierarchy vs scaling data: Not specified in the paper.

## 11. Architectural Workarounds

Efficient attention:
Evidence (Section 2): "They are trained to support 32k context length, employing efficient attention mechanisms (for e.g. multi-query attention (Shazeer, 2019a))."

Variable input resolution:
Evidence (Section 2): "The models can handle variable input resolution in order to spend more compute on tasks that require fine-grained understanding."

Distillation and quantization for Nano:
Evidence (Table 1): "It is trained by distilling from larger Gemini models. It is 4-bit quantized for deployment"

Multimodal tokenization / discrete image tokens:
Evidence (Section 2): "the models are multimodal from the beginning and can natively output images using discrete image tokens"

Video frames in context window:
Evidence (Section 2): "Video understanding is accomplished by encoding the video as a sequence of frames in the large context window."

## 12. Explicit Limitations and Non-Claims

Stated limitations:
Evidence (Section 8 Discussion): "There is a continued need for ongoing research and development on “hallucinations” generated by LLMs to ensure that model outputs are more reliable and verifiable."
Evidence (Section 8 Discussion): "LLMs also struggle with tasks requiring high-level reasoning abilities like causal understanding, logical deduction, and counterfactual reasoning even though they achieve impressive performance on exam benchmarks."

Non-claims / cautions on deployment:
Evidence (Section 10.1 Model Card): "Gemini should not be made available as part of a general-purpose service or product, or used within a specific downstream application without a prior assessment and mitigation of the safety and fairness concerns specific to the downstream use."

Explicit statements about not attempting open-world learning, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

## 13. Constraint Profile (Synthesis)

Constraint Profile:
- Domain scope: multi-modal (text, image, audio, video) and multi-domain inputs such as "natural images, charts, screenshots, PDFs, and videos" (Section 2).
- Task structure: broad benchmark suite across many tasks and modalities (Section 10.3; Section 5), not a single constrained task.
- Representation rigidity: variable input resolution and 32,768-token context length; audio at 16kHz (Sections 2 and 5.1.5).
- Model sharing vs specialization: jointly trained base model, with some specialized fine-tuning for tool use and competitive programming (Sections 1, 5.1.7, 6.5.2).
- Positional encoding: not specified; treated as an implicit architectural detail in this report.

## 14. Final Classification

Final classification: Unrestrained multi-task / multi-domain.
Justification: The model is "trained jointly across image, audio, video, and text" and evaluated across a wide range of text, image, audio, and video benchmarks (Sections 1 and 5). The evaluation spans many domains and tasks rather than a single constrained domain or task family, indicating an unrestrained multi-task, multi-domain setup.

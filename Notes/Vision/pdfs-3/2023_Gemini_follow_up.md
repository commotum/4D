Distinct tasks evaluated (unique): 68.
Evidence basis and task list (duplicates removed across categories):

Text benchmarks (Appendix 10.3):
- Factuality: BoolQ; NaturalQuestions-Closed; NaturalQuestions-Retrieved; Real-timeQA; TydiQA-noContext; TydiQA-goldP. (Appendix 10.3)
- Long Context: NarrativeQA; Scrolls-Qasper; Scrolls-Quality; XL Sum (English); XLSum (non-English); internal long-context benchmark. (Appendix 10.3)
- Math/Science: GSM8K; MATH; MMLU; Math-StackExchange; Math-AMC 2022-2023 problems; 3 internal math/science benchmarks. (Appendix 10.3)
- Reasoning: BigBench Hard; CLRS; Proof Writer; Reasoning-Fermi problems; Lambada; HellaSwag; DROP. (Appendix 10.3)
- Summarization: XL Sum (English); XL Sum (non-English); WikiLingua (non-English); WikiLingua (English); XSum. (Appendix 10.3)
- Multilinguality (new unique tasks beyond overlaps above): WMT22; WMT23; FRMT; MGSM; translated MMLU; NTREX; FLORES-200. (Appendix 10.3)

Additional text/code benchmarks not in Appendix 10.3:
- HumanEval; Natural2Code. (Section 5.1.1)
- MBPP. (Section 5.1.3)

Image understanding benchmarks:
- MMMU; TextVQA; DocVQA; ChartQA; InfographicVQA; MathVista; AI2D; VQAv2; XM3600. (Appendix 10.3)

Video understanding benchmarks:
- VATEX (English); VATEX (ZH); YouCook2; NextQA; ActivityNet-QA; Perception Test MCQA. (Appendix 10.3)

Audio understanding benchmarks:
- FLEURS; VoxPopuli; Multilingual Librispeech; CoVoST 2; internal YouTube ASR test set. (Appendix 10.3; Section 5.2.4)

Other explicit evaluation tasks:
- Synthetic retrieval test. (Section 5.1.5)
- Closed-Book Factuality; Attribution; Hedging. (Section 5.1.6)
- Complex prompts instruction-following internal benchmark. (Section 6.5.1)
- Tool-use internal benchmark (travel planning, video discovery). (Section 6.5.2)
- Competitive programming (AlphaCode 2 on Codeforces). (Section 5.1.7)
- Image generation (few-shot). (Section 5.2.3)

Number of trained model instances required to cover all tasks: 4.
Rationale:
- 1 jointly trained multimodal Gemini model covers the broad benchmark suite across text, image, audio, and video tasks (trained jointly across modalities). (Section 1; Section 2; Appendix 10.3)
- 1 tool-use fine-tuned Gemini API model is required for the tool-use benchmark (fine-tuned with tool-use data). (Section 6.5.2; Table 15)
- AlphaCode 2 competitive programming uses a specialized Gemini Pro coding model and a separate reward model (both fine-tuned), so 2 additional trained instances. (Section 5.1.7)

Task-Model Ratio:
$$
\boxed{
\frac{68\ \text{tasks}}{4\ \text{models}} = 17
}
$$

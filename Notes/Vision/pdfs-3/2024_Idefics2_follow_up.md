1. Number of distinct tasks evaluated: 8.
   - VQAv2 (visual question answering). Evidence: "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1).
   - TextVQA (text reading / OCR). Evidence: "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1) and "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning, TextVQA (Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2).
   - OKVQA (external knowledge VQA). Evidence: "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1).
   - COCO (captioning). Evidence: "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1).
   - MMMU (multidiscipline college-level problems). Evidence: "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning, TextVQA (Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2).
   - MathVista (mathematical reasoning). Evidence: "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning, TextVQA (Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2).
   - MMBench (perception and reasoning). Evidence: "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning, TextVQA (Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2).
   - DocVQA (document VQA / text in images). Evidence: "For the open-ended questions in TextVQA, DocVQA, and VQAv2, we evaluate with the prompt:" (Appendix A.3.1) and the DocVQA column in Table 15.

2. Number of trained model instances required to cover all tasks: 2.
   - Base model instance (Idefics2-base) is evaluated on VQAv2, TextVQA, OKVQA, COCO: "To evaluate the base model, we consider VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), OKVQA (Marino et al., 2019), and COCO (Lin et al., 2014)." (Section 4.1; Table 8 lists Idefics2-base).
   - A separately instruction-tuned model instance is created and evaluated on MMMU, MathVista, TextVQA, MMBench: "We instruction-tune the base model using DoRA (Liu et al., 2024) (a variant of LoRA)." and "We evaluate Idefics2 on commonly adopted benchmarks: MMMU (Yue et al., 2024) for multidiscipline college-level problems, MathVista (Lu et al., 2024) for mathematical reasoning, TextVQA (Singh et al., 2019) for text reading on natural images, and MMBench Liu et al. (2023) for various perception and reasoning tasks." (Section 4.2). DocVQA and VQAv2 are part of the evaluation setup for this model (Appendix A.3.1; Table 15).

$$
\boxed{
\frac{8\ \text{tasks}}{2\ \text{models}} = 4
}
$$

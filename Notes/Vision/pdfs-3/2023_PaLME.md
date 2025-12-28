# PaLM-E Survey Answers

## 1. Basic Metadata
- Title: PaLM-E: An Embodied Multimodal Language Model.
  Evidence: "PaLM-E: An Embodied Multimodal Language Model" (p.1)
- Authors: Danny Driess; Fei Xia; Mehdi S. M. Sajjadi; Corey Lynch; Aakanksha Chowdhery; Brian Ichter; Ayzaan Wahid; Jonathan Tompson; Quan Vuong; Tianhe Yu; Wenlong Huang; Yevgen Chebotar; Pierre Sermanet; Daniel Duckworth; Sergey Levine; Vincent Vanhoucke; Karol Hausman; Marc Toussaint; Klaus Greff; Andy Zeng; Igor Mordatch; Pete Florence.
  Evidence: "Danny Driess 1 2 Fei Xia 1 Mehdi S. M. Sajjadi 3 Corey Lynch 1 Aakanksha Chowdhery 3" (p.1)
  Evidence: "Brian Ichter 1 Ayzaan Wahid 1 Jonathan Tompson 1 Quan Vuong 1 Tianhe Yu 1 Wenlong Huang 1" (p.1)
  Evidence: "Yevgen Chebotar 1 PierrePROMPT:" (p.1)
  Evidence: "Sermanet 1 Daniel Duckworth" (p.1)
  Evidence: "Sergey Levine 1 Vincent Vanhoucke 1" (p.1)
  Evidence: "Karol Hausman Marc Toussaint" (p.1)
  Evidence: "Klaus Greff Andy" (p.1)
  Evidence: "Zeng" (p.1)
  Evidence: "Igor Mordatch 3 Pete Florence 1" (p.1)
- Year: 2023.
  Evidence: "arXiv:2303.03378v1 [cs.LG] 6 Mar 2023" (p.1)
- Venue: arXiv (preprint).
  Evidence: "arXiv:2303.03378v1 [cs.LG] 6 Mar 2023" (p.1)

## 2. One-Sentence Contribution Summary
One-sentence summary: The paper proposes embodied language models (PaLM-E) that directly incorporate continuous sensor modalities into a pre-trained LLM so the model can perform grounded embodied tasks such as robotic manipulation planning and visual question answering.
Evidence: "We propose embodied language models to directly incorporate real-world continuous sensor modalities into language models and thereby establish the link between words and percepts." (Abstract)
Evidence: "We train these encodings end-to-end, in conjunction with a pretrained large language model, for multiple embodied tasks including sequential robotic manipulation planning, visual question answering, and captioning." (Abstract)

## 3. Tasks Evaluated
### A) Embodied robotics: Task and Motion Planning (TAMP)
- Task: q1 (object color VQA).
  Task type: Other (specify: visual question answering).
  Dataset(s): TAMP environment dataset.
  Domain: robot manipulation with grasping/stacking objects.
  Evidence: "the VQA task q1 is about the color of an object." (Appendix B.1)
  Evidence: "dataset containing 96,000 training scenes of solely the TAMP environment" (Section 6.2)
  Evidence: "Task and Motion Planning (TAMP) domain where a robot has to manipulate (grasp and stack) objects" (Section 6.1)
- Task: q2 (object-table relation VQA).
  Task type: Other (specify: visual question answering / relational).
  Dataset(s): TAMP environment dataset.
  Domain: robot manipulation with tabletop objects.
  Evidence: "q2 : object-table relation. Example prompt: Given <img>. Q: Is the red object left, right, or center of the table?. Target: A: The red object is in the center of the table." (Appendix B.1)
  Evidence: "dataset containing 96,000 training scenes of solely the TAMP environment" (Section 6.2)
- Task: q3 (object-object relations VQA).
  Task type: Other (specify: visual question answering / relational).
  Dataset(s): TAMP environment dataset.
  Domain: robot manipulation with tabletop objects.
  Evidence: "q3 : object-object relations. Example prompt: Given <img>. Q: Is the yellow object below the blue object?. Target: A: No, the yellow object is not below the blue object." (Appendix B.1)
  Evidence: "dataset containing 96,000 training scenes of solely the TAMP environment" (Section 6.2)
- Task: q4 (plan feasibility VQA).
  Task type: Other (specify: visual question answering / plan feasibility).
  Dataset(s): TAMP environment dataset.
  Domain: robot manipulation with tabletop objects.
  Evidence: "q4 : plan feasibility. Example prompt: Given <img>. Q: Is it possible to first grasp the blue object, then place it on the yellow object, and then grasp the yellow object?. Target: A: No, this is not possible." (Appendix B.1)
  Evidence: "dataset containing 96,000 training scenes of solely the TAMP environment" (Section 6.2)
- Task: p1 (grasping plan generation).
  Task type: Other (specify: planning / action sequencing).
  Dataset(s): TAMP environment dataset; planning dataset generated with Driess et al. (2020).
  Domain: robot manipulation with grasping.
  Evidence: "p1 : grasping. Example prompt: Given <img>. Q: How to grasp the green object?. Target: A: First grasp the orange object and place it on the table, then grasp the green object." (Appendix B.1)
  Evidence: "We utilize the planner from Driess et al. (2020) to generate the dataset for the planning tasks." (Appendix B.1)
- Task: p2 (stacking plan generation).
  Task type: Other (specify: planning / action sequencing).
  Dataset(s): TAMP environment dataset; planning dataset generated with Driess et al. (2020).
  Domain: robot manipulation with stacking.
  Evidence: "p2 : stacking. Example prompt: Given <img>. Q: How to stack the white object on top of the red object?. Target: A: First grasp the green object and place it on the table, then grasp the white object and place it on the red object." (Appendix B.1)
  Evidence: "We utilize the planner from Driess et al. (2020) to generate the dataset for the planning tasks." (Appendix B.1)

### B) Embodied robotics: Language-Table (tabletop pushing)
- Task: Task 1 (push closest block to same color).
  Task type: Other (specify: planning / action sequencing).
  Dataset(s): Language-Table dataset (Lynch et al., 2022).
  Domain: tabletop pushing.
  Evidence: "The multi-object tabletop pushing environment is taken from the publicly available Language-Table dataset (Lynch et al., 2022)" (Section 6.1)
  Evidence: "Task 1. Q: There is a block that is closest to {i.e., top right corner}. Push that block to the other block of the same color." (Table 3)
- Task: Task 2 (sort blocks by color into corners).
  Task type: Other (specify: planning / action sequencing).
  Dataset(s): Language-Table dataset (Lynch et al., 2022).
  Domain: tabletop pushing.
  Evidence: "The multi-object tabletop pushing environment is taken from the publicly available Language-Table dataset (Lynch et al., 2022)" (Section 6.1)
  Evidence: "Task 2. Q: How to sort the blocks by colors into corners?" (Table 3)
- Task: Task 3 (push blocks on one side together without moving the other side).
  Task type: Other (specify: planning / action sequencing).
  Dataset(s): Language-Table dataset (Lynch et al., 2022).
  Domain: tabletop pushing.
  Evidence: "The multi-object tabletop pushing environment is taken from the publicly available Language-Table dataset (Lynch et al., 2022)" (Section 6.1)
  Evidence: "Task 3. Q: How to push all the blocks that are on the {left/right} side together, without bringing over any of the blocks that are on the {right/left} side?" (Table 3)

### C) Embodied robotics: Mobile manipulation (kitchen)
- Task: Affordance prediction.
  Task type: Other (specify: affordance prediction / VQA).
  Dataset(s): runs from Ahn et al. (2022) (2912 sequences).
  Domain: mobile manipulation in a kitchen environment.
  Evidence: "Q: Is it possible to <skill> here?." (Section 6.4)
  Evidence: "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (Section 6.4)
  Evidence: "mobile manipulation domain similar to SayCan (Ahn et al., 2022), where a robot has to solve a variety of tasks in a kitchen environment, including finding objects in drawers, picking them, and bringing them to a human." (Section 6.1)
- Task: Failure detection.
  Task type: Other (specify: failure detection / VQA).
  Dataset(s): runs from Ahn et al. (2022) (2912 sequences).
  Domain: mobile manipulation in a kitchen environment.
  Evidence: "The multi-modal prompt is Given <img>. Q: Was <skill> successful?." (Section 6.4)
  Evidence: "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (Section 6.4)
  Evidence: "mobile manipulation domain similar to SayCan (Ahn et al., 2022), where a robot has to solve a variety of tasks in a kitchen environment, including finding objects in drawers, picking them, and bringing them to a human." (Section 6.1)
- Task: Long-horizon planning (mobile manipulation).
  Task type: Other (specify: planning / action sequencing).
  Dataset(s): runs from Ahn et al. (2022) (2912 sequences).
  Domain: mobile manipulation in a kitchen environment.
  Evidence: "Real robot results: Long-horizon planning. Finally, we use PaLM-E to perform embodied planning end-to-end for mobile manipulation tasks. The prompt structure for this task is Human: <instruction> Robot: <step history>. I see <img>. PaLM-E is trained to generate the next step of the plan, conditioned on the history of taken steps and the current image observation of the scene." (Section 6.4)
  Evidence: "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (Section 6.4)

### D) General vision-language tasks
- Task: OK-VQA.
  Task type: Other (specify: visual question answering).
  Dataset(s): OKVQA (Marino et al., 2019).
  Domain: general vision-language tasks (not further specified in the paper).
  Evidence: "results on general vision-language tasks, including OKVQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5)
- Task: VQA v2.
  Task type: Other (specify: visual question answering).
  Dataset(s): VQA v2 (Goyal et al., 2017).
  Domain: general vision-language tasks (not further specified in the paper).
  Evidence: "results on general vision-language tasks, including OKVQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5)
- Task: COCO captioning.
  Task type: Generation.
  Dataset(s): COCO (Chen et al., 2015).
  Domain: general vision-language tasks (not further specified in the paper).
  Evidence: "results on general vision-language tasks, including OKVQA (Marino et al., 2019), VQA v2 (Goyal et al., 2017) and COCO captioning (Chen et al., 2015)." (Section 6.5)

### E) General language tasks (NLU/NLG)
- Task: TriviaQA (wiki) (EM).
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): TriviaQA (wiki) (EM).
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "TriviaQA (wiki) (EM)" (Appendix C, Table 8)
- Task: Natural Questions (EM).
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): Natural Questions (EM).
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "Natural Questions (EM)" (Appendix C, Table 8)
- Task: WebQuestions (EM).
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): WebQuestions (EM).
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "WebQuestions (EM)" (Appendix C, Table 8)
- Task: Lambada.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): Lambada.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "Lambada" (Appendix C, Table 8)
- Task: HellaSwag.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): HellaSwag.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "HellaSwag" (Appendix C, Table 8)
- Task: StoryCloze.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): StoryCloze.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "StoryCloze" (Appendix C, Table 8)
- Task: Winograd.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): Winograd.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "Winograd" (Appendix C, Table 8)
- Task: Winogrande.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): Winogrande.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "Winogrande" (Appendix C, Table 8)
- Task: RACE-M.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): RACE-M.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "RACE-M" (Appendix C, Table 8)
- Task: RACE-H.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): RACE-H.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "RACE-H" (Appendix C, Table 8)
- Task: PIQA.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): PIQA.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "PIQA" (Appendix C, Table 8)
- Task: ARC-e.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): ARC-e.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "ARC-e" (Appendix C, Table 8)
- Task: ARC-c.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): ARC-c.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "ARC-c" (Appendix C, Table 8)
- Task: OpenBookQA.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): OpenBookQA.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "OpenBookQA" (Appendix C, Table 8)
- Task: BoolQ.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): BoolQ.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "BoolQ" (Appendix C, Table 8)
- Task: Copa.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): Copa.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "Copa" (Appendix C, Table 8)
- Task: RTE.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): RTE.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "RTE" (Appendix C, Table 8)
- Task: Wic.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): Wic.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "Wic" (Appendix C, Table 8)
- Task: WSC.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): WSC.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "WSC" (Appendix C, Table 8)
- Task: ReCoRD.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): ReCoRD.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "ReCoRD" (Appendix C, Table 8)
- Task: CB.
  Task type: Other (specify: general language benchmark, NLU/NLG).
  Dataset(s): CB.
  Domain: general language benchmarks for NLU/NLG.
  Evidence: "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks for Natural Language Understanding (NLU) and Natural Language Generation (NLG) tasks." (Section 6.6)
  Evidence: "CB" (Appendix C, Table 8)

## 4. Domain and Modality Scope
- Single domain? No; evaluation spans multiple domains (robotics, vision-language, and language-only tasks).
  Evidence: "Our experiments consider diverse robotic (mobile) manipulation tasks across three different robot embodiments, in simulation and with two different real robots." (Section 6)
  Evidence: "we evaluate PaLM-E also on general vision-language tasks such as visual-question-answering (VQA), image captioning, and established language modeling tasks." (Section 6)
- Multiple domains within the same modality? Yes (multiple robotics domains and multiple language/vision-language tasks).
  Evidence: "Our three robot environments (Fig. 1) include a Task and Motion Planning (TAMP) domain where a robot has to manipulate (grasp and stack) objects, a table-top pushing environment, and a mobile manipulation domain." (Section 6.1)
- Multiple modalities? Yes.
  Evidence: "Input to our embodied language model are multi-modal sentences that interleave visual, continuous state estimation, and textual input encodings." (Abstract)
  Evidence: "PaLM-E operates on multimodal sentences, i.e. sequences of tokens where inputs from arbitrary modalities (e.g. images, neural 3D representations, or states, in green and blue) are inserted alongside text tokens (in orange) as input to an LLM, trained end-to-end." (Figure 1 caption)
- Domain generalization or cross-domain transfer claimed? Yes (positive transfer across domains/tasks).
  Evidence: "exhibits positive transfer: the model benefits from diverse joint training across internet-scale language, vision, and visual-language domains." (Abstract)
  Evidence: "co-training on these datasets enables transfer (Fig. 3): despite different tasks and embodiments, the performance on the individual tasks increases by training on the mixture of tasks." (Section 6)

## 5. Model Sharing Across Tasks
### A) Robotics tasks (TAMP, Language-Table, Mobile Manipulation)
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| TAMP q1 | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| TAMP q2 | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| TAMP q3 | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| TAMP q4 | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| TAMP p1 | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| TAMP p2 | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| Language-Table Task 1 | Yes (generalist mixture model) | Yes (finetuned versions reported) | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6); "To train the finetuned versions of these models, we train a pretrained PaLM-E model for 9,000 additional steps" (Appendix B.2) |
| Language-Table Task 2 | Yes (generalist mixture model) | Yes (finetuned versions reported) | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6); "To train the finetuned versions of these models, we train a pretrained PaLM-E model for 9,000 additional steps" (Appendix B.2) |
| Language-Table Task 3 | Yes (generalist mixture model) | Yes (finetuned versions reported) | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6); "To train the finetuned versions of these models, we train a pretrained PaLM-E model for 9,000 additional steps" (Appendix B.2) |
| Mobile affordance prediction | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| Mobile failure detection | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |
| Mobile long-horizon planning | Yes (generalist mixture model) | Not specified | Not specified | "a single model, trained on a mixture of many datasets, across diverse tasks, and across robot embodiments, can simultaneously achieve high performance on all of those tasks." (Section 6) |

### B) General vision-language tasks
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| OK-VQA | Yes (generalist checkpoint across evaluations); task-specific finetuned models also reported | Not specified for PaLM-E generalist | Not specified | "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use differentfinetuned models for the different tasks." (Table 5) |
| VQA v2 | Yes (generalist checkpoint across evaluations); task-specific finetuned models also reported | Not specified for PaLM-E generalist | Not specified | "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use differentfinetuned models for the different tasks." (Table 5) |
| COCO captioning | Yes (generalist checkpoint across evaluations); task-specific finetuned models also reported | Not specified for PaLM-E generalist | Not specified | "For the generalist models, they are the same checkpoint across the different evaluations, while task-specific finetuned models use differentfinetuned models for the different tasks." (Table 5) |

### C) General language tasks (NLU/NLG)
| Task | Shared Weights? | Fine-Tuned? | Separate Head? | Evidence |
| --- | --- | --- | --- | --- |
| TriviaQA (wiki) (EM) | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| Natural Questions (EM) | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| WebQuestions (EM) | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| Lambada | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| HellaSwag | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| StoryCloze | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| Winograd | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| Winogrande | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| RACE-M | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| RACE-H | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| PIQA | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| ARC-e | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| ARC-c | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| OpenBookQA | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| BoolQ | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| Copa | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| RTE | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| Wic | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| WSC | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| ReCoRD | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |
| CB | Not specified in the paper | Not specified | Not specified | "Tab. 8 reports the averaged performance of PaLM-E on 21 general language benchmarks" (Section 6.6) |

## 6. Input and Representation Constraints
- Input modalities and interleaving: "The inputs to PaLM-E consist of text and (multiple) continuous observations. The multimodal tokens corresponding to these observations are interleaved with the text to form multi-modal sentences." (Section 3)
- Fixed or variable input resolution? Not specified in the paper.
- Fixed patch size? Not specified in the paper.
- Fixed number of tokens? Not specified in the paper; it only notes multiple embeddings per observation/object: "Note that a single observation Oj is usually encoded into multiple embedding vectors." (Section 3) and "individual objects are always tokenized into multiple embeddings each" (Section 4).
- Fixed dimensionality: "vectors with the same dimension as the embedding space of the language tokens." (Section 3)
- Fixed dimensionality (2D/3D) constraints for representations: "We investigate state estimation vectors, Vision Transformers (ViTs) (Dosovitskiy et al., 2020; Chen et al., 2022; Ryoo et al., 2021) for 2D image features, and the 3D-aware Object Scene Representation Transformer (OSRT) (Sajjadi et al., 2022a)." (Section 4).
- Padding/resizing requirements? Not specified in the paper.
- Dynamic placement of observation tokens: "observation embeddings are not inserted at fixed positions, but instead placed dynamically within the surrounding text." (Section 3)

## 7. Context Window and Attention Structure
- Maximum sequence length: Not specified in the paper.
- Fixed or variable sequence length: Not specified in the paper.
- Attention type: Not specified beyond "processed by the self-attention layers of a Transformer-based LLM" (Introduction).
- Mechanisms to manage computational cost (windowing, pooling, pruning): Not specified in the paper.

## 8. Positional Encoding (Critical Section)
- Positional encoding mechanism: Not specified (the paper only states it reuses the LLM's existing positional encodings).
  Evidence: "Injecting the continuous information this way into the LLM reuses its existing positional encodings." (Section 3)
- Where applied: Not specified beyond reuse of existing LLM positional encodings.
- Fixed vs modified across experiments: Not specified in the paper.

## 9. Positional Encoding as a Variable
- Treated as a core research variable? Not specified in the paper.
- Multiple positional encodings compared? Not specified in the paper.
- Any claim that PE choice is not critical? Not specified in the paper.
- Only explicit mention: "Injecting the continuous information this way into the LLM reuses its existing positional encodings." (Section 3)

## 10. Evidence of Constraint Masking (Scale/Structure)
- Model sizes: "Our largest model, PaLM-E-562B with 562B parameters" (Abstract). Also, "We base PaLM-E on the pretrained 8B, 62B, and 540B parameter variants of PaLM as the decoder-only LLM into which we inject the continuous observations through the input encoders. Those encoders are either pre-trained or trained from scratch, see Sec. 4. We refer to an 8B LLM combined with a 4B ViT as PaLM-E12B, similarly a 62B LLM + 22B ViT as PaLM-E-84B, and 540B LLM + 22B ViT as PaLM-E-562B." (Section 5)
- Scaling model size improves performance: "Scaling the 12B model" (Section 6.4) and "to the 84B model leads to improvements on 2 of 3 tasks." (Section 6.4)
- Scaling data / mixture: "Transfer from full mixture is particularly effective. Note that full mixture contains only 1% of the training data (320 examples each) for the tasks evaluated here." (Figure 4 caption)
- Dataset sizes: "dataset containing 96,000 training scenes of solely the TAMP environment" (Section 6.2); "regime with only 10 demos per task" (Section 6.4); "We train the model by using the runs from (Ahn et al., 2022), which contains 2912 sequences." (Section 6.4); "between 10 and 80 for Language Table or 320 for TAMP." (Section 7)
- Scaling and forgetting: "with increasing model scale, there is considerably less catastrophic forgetting of language capabilities." (Section 6.6)

## 11. Architectural Workarounds
- Injecting continuous observations into LLM embedding space: "Multi-modal sentences: injection of continuous observations. Multi-modal information such as image observations" (Section 3). Purpose: enable continuous sensor inputs to be processed by the LLM.
- Object-centric representations to manage object structure: "structured encoders that aim to separate visual inputs into distinct objects before injecting them into the LLM." (Section 4)
- OSRT object slots (3D-aware): "OSRT learns 3D-centric neural scene representations" and maps object slots into embeddings (Section 4). Purpose: 3D-centric object representations without ground-truth segmentation.
- Entity referrals: "Object 1 is <obj 1>. . . . Object j is <obj j>. This enables PaLM-E to reference objects via special tokens of the form obj j in its generated output sentences." (Section 4). Purpose: explicit object reference in plans.
- Freezing the LLM and training encoders only: "freeze the LLM and to just train the" (Section 5). Purpose: ground a frozen LLM via learned encoders.

## 12. Explicit Limitations and Non-Claims
- Future work: "opportunity for future work is to combine this with a method benefitting from large-scale visual data." (Section 7)
- Limitation of a training strategy: "freezing the LLM and only training the input encoders is a viable path for building embodied language models, although this approach occasionally struggled for robotics tasks (Tab. 2)." (Section 7)
- Non-focus statement: "Although it is not the focus of our work, we report in Tab. 5 results on general vision-language tasks" (Section 6.5)
- Explicit non-claims about open-world, unrestrained multi-task learning, or meta-learning: Not specified in the paper.

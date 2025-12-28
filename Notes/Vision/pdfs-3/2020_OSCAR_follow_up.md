Number of distinct tasks evaluated: 7.
Evidence: "We adapt the pre-trained models to seven downstream V+L tasks, including five understanding tasks and two generation tasks." (2020_OSCAR.txt)
- Task 1: Image retrieval (Image-Text Retrieval sub-task). Evidence: "Image-Text Retrieval ... There are two sub-tasks: image retrieval and text retrieval" (2020_OSCAR.txt)
- Task 2: Text retrieval (Image-Text Retrieval sub-task). Evidence: "Image-Text Retrieval ... There are two sub-tasks: image retrieval and text retrieval" (2020_OSCAR.txt)
- Task 3: Image Captioning. Evidence: "Image Captioning requires the model to generate a natural language description... we fine-tune Oscar using the seq2seq objective." (2020_OSCAR.txt)
- Task 4: Novel Object Captioning (NoCaps). Evidence: "Novel Object Captioning (NoCaps) ... train Oscar on COCO without the initialization of pre-training." (2020_OSCAR.txt)
- Task 5: VQA. Evidence: "VQA [9] requires the model to answer natural language questions based on an image." (2020_OSCAR.txt)
- Task 6: GQA. Evidence: "GQA [13] is similar to VQA... We conduct experiments on the public GQA dataset." (2020_OSCAR.txt)
- Task 7: NLVR2. Evidence: "Natural Language Visual Reasoning for Real (NLVR2) [36] takes a pair of images and a natural language..." (2020_OSCAR.txt)

Number of trained model instances required to cover all tasks: 6 separate fine-tuned models (one retrieval model covers both image/text retrieval; plus distinct models for image captioning, NoCaps, VQA, GQA, NLVR2).
Evidence: Retrieval is trained with a binary classifier on the [CLS] representation and used to rank image-text pairs (one model covers both retrieval directions): "The final representation of [CLS] is used as the input to the classifier..." (2020_OSCAR.txt). Image Captioning uses a different seq2seq fine-tuning objective: "we fine-tune Oscar using the seq2seq objective." (2020_OSCAR.txt). VQA uses a task-specific linear classifier head: "the [CLS] output from Oscar is fed to a task-specific linear classifier" (2020_OSCAR.txt). NLVR2 uses a different classifier head: "two [CLS] outputs... concatenated as the joint input for a binary classifier, implemented by an MLP" (2020_OSCAR.txt). NoCaps requires separate training from BERT without pre-training: "train Oscar on COCO without the initialization of pre-training." (2020_OSCAR.txt). GQA uses its own fine-tuned model similar to VQA: "We develop two fine-tuned models using OscarB. One is similar to that of VQA." (2020_OSCAR.txt). The paper notes the approach is "single task fine-tuning," implying distinct fine-tuned models per task. (2020_OSCAR.txt)

$$
\boxed{
\frac{7\ \text{tasks}}{6\ \text{models}} = 1.17
}
$$

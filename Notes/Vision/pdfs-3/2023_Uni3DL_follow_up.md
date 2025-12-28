Number of distinct tasks evaluated: 8.

Evidence and task list:
1) 3D semantic segmentation - "including semantic segmentation..." (Figure 1 caption) and "including semantic segmentation..." (Abstract).
2) 3D object detection - "including ... object detection..." (Figure 1 caption; Abstract) and evaluation grouped with semantic/instance segmentation ("We compare 3D semantic segmentation, object detection, and instance segmentation..." Section 4.3).
3) 3D instance segmentation - "including ... instance segmentation..." (Figure 1 caption; Abstract).
4) 3D visual grounding / grounded segmentation - "visual grounding" (Abstract) and "grounded segmentation" (Figure 1 caption).
5) Grounded localization - "Grounded Localization" and "To produce grounded object location, we directly use grounded object masks to calculate their bounding boxes." (Appendix B.2).
6) 3D captioning - "captioning" (Figure 1 caption; Abstract).
7) Text-3D cross-modal retrieval - "text-3D cross-modality retrieval" (Figure 1 caption) and "We show text-to-3D and 3D-to-text retrieval results..." (Appendix C.4).
8) (Zero-shot) 3D object classification - "(zero-shot) 3D object classification." (Figure 1 caption) and "evaluate zero-shot 3D classification performance..." (Appendix B.1).

Number of trained model instances required to cover all tasks: 4.

Evidence and mapping:
- Model A (semantic/instance segmentation + object detection): finetuning is described for "3D semantic/instance Segmentation" (Appendix A.2). Object detection is reported in the same evaluation section as semantic/instance segmentation ("We compare 3D semantic segmentation, object detection, and instance segmentation..." Section 4.3), and no detection-specific finetuning is described elsewhere.
- Model B (grounded segmentation + grounded localization): finetuning is described for "Grounded Segmentation" (Appendix A.2). Grounded localization is derived from grounded masks ("use grounded object masks to calculate their bounding boxes." Appendix B.2), so no extra model instance is needed.
- Model C (captioning + zero-shot classification): finetuning is described for "3D Captioning" (Appendix A.2). Zero-shot classification uses the model "fine-tuned on the Cap3D Objaverse dataset" (Appendix B.1), so it can share the captioning-finetuned instance.
- Model D (text-3D cross-modal retrieval): finetuning is described for "Text-3D Cross-Modal Retrieval" (Appendix A.2).

$$
\boxed{
\frac{8\ \text{tasks}}{4\ \text{models}} = 2
}
$$

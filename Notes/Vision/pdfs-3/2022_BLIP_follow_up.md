Distinct tasks evaluated (7):
1) Image-text retrieval (image-to-text and text-to-image) on COCO and Flickr30K. Evidence: Section 5.1 describes evaluating both TR and IR on COCO/Flickr30K. (page 6, Sec. 5.1)
2) Image captioning on COCO and NoCaps. Evidence: Section 5.2 lists captioning datasets and evaluation. (page 6, Sec. 5.2)
3) Visual Question Answering (VQA). Evidence: Section 5.3 defines VQA and the VQA evaluation. (page 7, Sec. 5.3)
4) Natural Language Visual Reasoning (NLVR2). Evidence: Section 5.4 defines NLVR2 and its evaluation. (page 8, Sec. 5.4)
5) Visual Dialog (VisDial). Evidence: Section 5.5 defines VisDial and its evaluation. (page 8, Sec. 5.5)
6) Text-to-video retrieval (zero-shot). Evidence: Section 5.6 reports zero-shot transfer to text-to-video retrieval. (page 8, Sec. 5.6)
7) Video question answering (zero-shot). Evidence: Section 5.6 reports zero-shot transfer to video question answering. (page 8, Sec. 5.6)

Trained model instances required to cover all tasks (5):
- Retrieval model: requires finetuning with ITC/ITM losses for retrieval. (page 6, Sec. 5.1)
- Captioning model: requires finetuning with LM loss for captioning. (page 6, Sec. 5.2)
- VQA model: requires finetuning with an answer decoder using LM loss. (page 7, Sec. 5.3)
- NLVR2 model: requires a classifier head on [Encode] with task-specific two-image processing. (page 8, Sec. 5.4)
- VisDial model: requires a dialog encoder trained with ITM loss on VisDial. (page 8, Sec. 5.5)
- No additional model for video tasks: video retrieval uses the COCO-retrieval model and video QA uses the VQA model in zero-shot transfer. (page 8, Sec. 5.6)

$$
\boxed{
\frac{7\ \text{tasks}}{5\ \text{models}} = 1.4
}
$$

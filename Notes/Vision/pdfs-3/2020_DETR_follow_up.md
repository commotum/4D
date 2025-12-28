1) Number of distinct tasks evaluated: 2 - object detection on COCO 2017 detection and panoptic segmentation on COCO 2017 panoptic. (2020_DETR.pdf, Experiments; 2020_DETR.pdf, Section 4.4)

2) Number of trained model instances required to cover all tasks: 2 - detection uses base DETR, while panoptic segmentation requires adding a mask head and training it (jointly or via a two-step process), which is a task-specific head and thus a separate trained model instance. (2020_DETR.pdf, Section 4.4)

3) Task-Model Ratio:

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$

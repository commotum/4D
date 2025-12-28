Distinct tasks evaluated (3): caption selection given an image ("text score"), image selection given a caption ("image score"), and joint matching where all four caption-image pairings must be correctly scored ("group score"). (2022_Winoground.pdf, Sec. 3.2 Metrics: "The first metric is the text score, which measures whether a model can select the correct caption, given an image."; "The second metric is the image score, which measures whether a model can select the correct image, given a caption."; "we also evaluate using the group score, where every combination for a given example {(C0 , I0 ), (C0 , I1 ), (C1 , I0 ), (C1 , I1 )} must be correctly scored by the model")

Number of trained model instances required to cover all tasks: 1. All three tasks are computed from the same image-caption scoring function s(.) (the model's score for an image/caption pair), and the group score explicitly combines the text and image scores rather than requiring a separate task-specific head. (2022_Winoground.pdf, Sec. 3.2 Metrics: "where s(.) is the model's score for the image/caption pair"; group score defined as combination of f and g in Eq. 3)

$$
\boxed{
\frac{3\ \text{tasks}}{1\ \text{model}} = 3
}
$$

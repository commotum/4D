Distinct tasks evaluated: 3 (VQA v2.0, GQA, NLVR2). Evidence: "We use three datasets for evaluating our LXMERT framework: VQA v2.0 dataset (Goyal et al., 2017), GQA (Hudson and Manning, 2019), and NLVR2 ." (2019_LXMERT.pdf.txt)

Trained model instances required: 3 separate fine-tuned models (one per task). Evidence that VQA and GQA are fine-tuned task-specifically: "On VQA and GQA, we fine-tune our model from the pre-trained snapshot..." and NLVR2 uses a task-specific classifier: "Since each datum in NLVR2 has two natural images img0, img1 and one language statement s, we use LXMERT to encode the two image-statement pairs (img0, s) and (img1, s), then train a classifier based on the concatenation of the two cross-modality outputs." (2019_LXMERT.pdf.txt)

$$
\boxed{
\frac{3\ \text{tasks}}{3\ \text{models}} = 1
}
$$

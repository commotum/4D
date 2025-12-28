Number of distinct tasks evaluated: 9.
- ImageNet-1K classification (representations evaluated via supervised fine-tuning or linear probing on IN1K). (2021_MAE.pdf, Section 4: "We do self-supervised pre-training on the ImageNet-1K (IN1K) training set. Then we do supervised training to evaluate the representations with (i) end-to-end fine-tuning or (ii) linear probing.")
- COCO object detection. (2021_MAE.pdf, Section 5: "Object detection and segmentation. We fine-tune Mask R-CNN [24] end-to-end on COCO [37]. We report box AP for object detection...")
- COCO instance segmentation. (2021_MAE.pdf, Section 5: "...and mask AP for instance segmentation.")
- ADE20K semantic segmentation. (2021_MAE.pdf, Section 5: "Semantic segmentation. We experiment on ADE20K [72] using UperNet [63] (see A.4).")
- iNat 2017 classification. (2021_MAE.pdf, Table 6 lists "iNat 2017" under transfer learning classification datasets.)
- iNat 2018 classification. (2021_MAE.pdf, Table 6 lists "iNat 2018" under transfer learning classification datasets.)
- iNat 2019 classification. (2021_MAE.pdf, Table 6 lists "iNat 2019" under transfer learning classification datasets.)
- Places205 classification. (2021_MAE.pdf, Table 6 lists "Places205" under transfer learning classification datasets.)
- Places365 classification. (2021_MAE.pdf, Table 6 lists "Places365" under transfer learning classification datasets.)

Number of trained model instances required to cover all tasks: 9.
- ImageNet classification requires a supervised classifier head via fine-tuning or linear probing. (2021_MAE.pdf, Section 4: "Then we do supervised training to evaluate the representations with (i) end-to-end fine-tuning or (ii) linear probing.")
- COCO object detection and COCO instance segmentation each require a Mask R-CNN model trained for the respective outputs (box AP vs mask AP). (2021_MAE.pdf, Section 5: "We fine-tune Mask R-CNN [24] end-to-end on COCO [37]. We report box AP for object detection and mask AP for instance segmentation.")
- ADE20K semantic segmentation uses a UperNet head with end-to-end fine-tuning. (2021_MAE.pdf, Section 5: "We experiment on ADE20K [72] using UperNet [63]"; 2021_MAE.pdf, Appendix A.4: "We fine-tune end-to-end for 100 epochs.")
- iNat 2017/2018/2019 and Places205/Places365 each require their own fine-tuned classification head; the paper specifies per-dataset fine-tuning for iNaturalist and Places. (2021_MAE.pdf, Appendix A.5: "We follow the setting in Table 9 for iNaturalist and Places fine-tuning (Table 6). We adjust the lr and fine-tuning epochs for each individual dataset.")

$$
\boxed{
\frac{9\ \text{tasks}}{9\ \text{models}} = 1
}
$$

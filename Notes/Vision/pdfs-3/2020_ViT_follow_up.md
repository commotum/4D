1. Number of distinct tasks evaluated: 23.
   - Main transfer tasks: ImageNet (original labels), ImageNet ReaL, CIFAR-10, CIFAR-100, Oxford-IIIT Pets, Oxford Flowers-102. (Section 4.1, p.4)
   - VTAB-1k tasks (19 total). After deduping CIFAR-100, Flowers102, and Pets already counted above, the additional tasks are: Caltech101, DTD, Sun397, SVHN, Camelyon, EuroSAT, Resisc45, Retinopathy, Clevr-Count, Clevr-Dist, DMLab, dSpr-Loc, dSpr-Ori, KITTI-Dist, sNORB-Azim, sNORB-Elev. (Section 4.1, p.4; Appendix D.10, Table 9, p.22)
   - ObjectNet classification. (Appendix D.9, p.20)
   - Total = 6 + 16 + 1 = 23.

2. Number of trained model instances required to cover all tasks: 21.
   - For each downstream dataset, they remove the pre-trained head and replace it with a new linear layer sized to the target classes, implying a separate fine-tuned model per dataset/task. (Appendix B.1.1, p.13)
   - ImageNet and ImageNet ReaL are evaluated as two labelings of the ImageNet task; no separate head is described. (Section 4.1, p.4)
   - ObjectNet is evaluated with the flagship ViT-H/14 model; no ObjectNet-specific head or fine-tuning is described, so it can use the same ImageNet head. (Appendix D.9, p.20)
   - Therefore, 1 model covers ImageNet + ImageNet ReaL + ObjectNet, and 20 additional tasks each require their own model: 1 + 20 = 21.

3. Task-Model Ratio:

$$
\boxed{
\frac{23\ \text{tasks}}{21\ \text{models}} \approx 1.10
}
$$

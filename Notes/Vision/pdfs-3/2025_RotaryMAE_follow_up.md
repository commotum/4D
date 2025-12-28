Distinct tasks evaluated: 13.
- Absolute position reconstruction (synthetic positions; linear head predicts positions). Evidence: "we give the model a sequence of 10 identical values as input... We then use the same linear head to predict the position for all tokens." (Section 5.1, 2025_RotaryMAE.pdf)
- Image classification on Tiny ImageNet. Evidence: "we train three versions of RoMAE on Tiny ImageNet" and "fine-tune" (Section 5.2, 2025_RotaryMAE.pdf).
- Audio classification on ESC-50. Evidence: "For the finetuning audio classification benchmark, we used the ESC-50 dataset" (Section 5.3, 2025_RotaryMAE.pdf).
- Irregular time-series classification on ELAsTiCC plus UEA datasets BM, CT, EP, HB, LSST (6 tasks total). Evidence: "DESC ELAsTiCC Challenge is a multi-variate irregular time-series dataset" and "We evaluate RoMAE on a variety of datasets from the UEA Multivariate Time-series Archive" with Table 6 listing BM, CT, EP, HB, LSST (Section 5.4, Table 6, 2025_RotaryMAE.pdf).
- Irregular time-series regression on the Pendulum dataset. Evidence: "Irregular Time-series Regression: Pendulum Dataset" and "RoMAE is trained directly on regression" (Section 5.4, 2025_RotaryMAE.pdf).
- Irregular time-series interpolation on Spiral, Synthetic, PhysioNet (3 tasks total). Evidence: "We evaluate RoMAE on three interpolation tasks... (i) Spiral... (ii) Synthetic... (iii) PhysioNet" (Section 5.5, 2025_RotaryMAE.pdf).

Trained model instances required: 13.
- Each dataset/task is trained or fine-tuned separately (Tiny ImageNet pretrain + finetune; ESC-50 finetuning; ELAsTiCC pretrain + finetune; UEA per-dataset pretrain + finetune; Pendulum trained directly on regression; interpolation tasks run as separate experiments with dataset-specific setups). Evidence: "After pre-training each model... fine-tune" (Section 5.2), "For the finetuning audio classification benchmark, we used the ESC-50 dataset" (Section 5.3), "We train RoMAE-tiny by conducting full pre-training... then fine-tuning" and "For each dataset we conduct pre-training... change hyper-parameters between different datasets" (Section 5.4), "RoMAE is trained directly on regression" (Section 5.4), and "We evaluate RoMAE on three interpolation tasks... (i) Spiral... (ii) Synthetic... (iii) PhysioNet" (Section 5.5), all in 2025_RotaryMAE.pdf.

$$
\boxed{
\frac{13\ \text{tasks}}{13\ \text{models}} = 1
}
$$

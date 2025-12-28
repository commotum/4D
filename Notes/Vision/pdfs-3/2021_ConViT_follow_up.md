Number of distinct tasks evaluated: 2.
- ImageNet-1k image classification (including the subsampled ImageNet-1k sample-efficiency experiments, which are still ImageNet classification). (2021_ConViT.pdf, Table 1; 2021_ConViT.pdf, Table 2; 2021_ConViT.pdf, Fig. 11)
- CIFAR100 image classification. (2021_ConViT.pdf, Appendix C "Further performance results"; 2021_ConViT.pdf, Fig. 11)

Number of trained model instances required to cover all tasks: 2.
- The paper trains models from scratch on ImageNet-1k and separately trains models on CIFAR100 (rescaled images, longer training), with no single unified multi-task model described. (2021_ConViT.pdf, Table 1; 2021_ConViT.pdf, Appendix C "Further performance results")

$$
\boxed{
\frac{2\ \text{tasks}}{2\ \text{models}} = 1
}
$$

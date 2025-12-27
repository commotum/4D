# Learning Transferable Visual Models From Natural Language Supervision

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.

### Preprint

* **arXiv URL:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
* **Initial Submission Date:** February 26, 2021

### Peer-Reviewed Publication

* **Conference:** International Conference on Machine Learning (ICML 2021)
* **Proceedings:** Proceedings of Machine Learning Research (PMLR), Volume 139
* **Publication Date:** July 2021
* **Pages:** 8748–8763
* **Published URL (PDF):** [https://proceedings.mlr.press/v139/radford21a/radford21a.pdf](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)

## Abstract

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.
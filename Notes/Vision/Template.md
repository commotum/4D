# Learning Transferable Visual Models From Natural Language Supervision (2021_CLIP)

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.

### Preprint

- **arXiv ID:** arXiv:2103.00020
- **arXiv URL:** [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- **Initial Submission Date:** February 26, 2021
- **Latest arXiv Version:** v1 (Feb 26, 2021)

### Peer-Reviewed Publication

- **Conference:** International Conference on Machine Learning (ICML 2021)
- **Proceedings:** Proceedings of Machine Learning Research (PMLR), Volume 139
- **Publication Date:** July 2021
- **Pages:** 8748–8763
- **Published URL (PDF):** [https://proceedings.mlr.press/v139/radford21a/radford21a.pdf](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)
- **Proceedings Page:** [https://proceedings.mlr.press/v139/radford21a.html](https://proceedings.mlr.press/v139/radford21a.html)

### Citations

- **Google Scholar Page:** [https://scholar.google.com/scholar_lookup?arxiv_id=2103.00020](https://scholar.google.com/scholar_lookup?arxiv_id=2103.00020)
- **Google Scholar Citation Count:** 50,271
- **Semantic Scholar Page:** [https://www.semanticscholar.org/paper/Learning-Transferable-Visual-Models-From-Natural-Radford-Kim/6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4](https://www.semanticscholar.org/paper/Learning-Transferable-Visual-Models-From-Natural-Radford-Kim/6f870f7f02a8c59c3e23f407f3ef00dd1dcf8fc4)
- **Semantic Scholar Citation Count:** 40,352

### Bibtex

@InProceedings{pmlr-v139-radford21a,
  title = 	 {Learning Transferable Visual Models From Natural Language Supervision},
  author =       {Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and Krueger, Gretchen and Sutskever, Ilya},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {8748--8763},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/radford21a/radford21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/radford21a.html},
}

## Abstract

State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on.
# Medical Image Segmentation With Disttance Map


## Motivation

Incorporating the distance maps of image segmentation labels  into CNNs-based segmentation tasks has received significant attention in 2019. These methods can be classified into two main classes in terms of the main usage of distance map.

- Designing new loss functions
- Adding an auxiliary task, e.g. distance map regression

However, with these methods on the one hand and the diversity of the specific implementations and dataset-related challenges on the other, it's hard to figure out which design can generalize well beyond the experiments in the original papers. Up to now, there is still no comprehensive comparison among these methods.

In this repository,  we want to re-implement these methods and evaluate them on the same segmentation tasks (heart and liver tumor segmentation), so as to figure out the useful designs.



## Related Work in 2019

### New loss functions

| Date | First author  | Title                       | Official Code  | Publication                    |
| ---- | ------------- | --------------------------- | -------------- | ------------------------------ |
| 2019 | Yuan Xue  | Shape-Aware Organ Segmentation by Predicting Signed Distance Maps [(arxiv)](https://arxiv.org/abs/1912.03849) | None         | AAAI 2020  |
| 2019 | [Hoel Kervadec](https://scholar.google.com.hk/citations?user=yeFGhfgAAAAJ&hl=zh-CN&oi=sra) | Boundary loss for highly unbalanced segmentation | [pytorch](https://github.com/LIVIAETS/surface-loss) | [MIDL 2019](http://proceedings.mlr.press/v102/kervadec19a.html) |     
|2019|Davood Karimi|Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks [(arxiv)](https://arxiv.org/abs/1904.10030) |None|[TMI 2019](https://ieeexplore.ieee.org/document/8767031)|
|2019|[Qikui Zhu](https://scholar.google.com.hk/citations?user=EhYbflYAAAAJ&hl=zh-CN&oi=sra)|Boundary-weighted Domain Adaptive Neural Network for Prostate MR Image Segmentation [(arxiv)](https://arxiv.org/abs/1902.08128)|[keras (incomplete)](https://github.com/ahukui/BOWDANet)|[TMI 2019](https://ieeexplore.ieee.org/document/8795525)|








### New loss functions

| Date | First author  | Title                       | Official Code  | Publication                    |
| ---- | ------------- | --------------------------- | -------------- | ------------------------------ |
| 2019 | Yan Wang     | Deep Distance Transform for Tubular Structure Segmentation in CT Scans | None | [arxiv](https://arxiv.org/abs/1912.03383) |
| 2019 | [Shusil Dangi](https://scholar.google.com.hk/citations?user=h12ifugAAAAJ&hl=zh-CN&oi=sra) | A Distance Map Regularized CNN for Cardiac Cine MR Image Segmentation [(arxiv)](https://arxiv.org/abs/1901.01238) | None | [Medical Physics](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.13853) |
|2019|[Fernando Navarro](https://scholar.google.com.hk/citations?user=rRKrhrwAAAAJ&hl=zh-CN&oi=sra)|Shape-Aware Complementary-Task Learning for Multi-organ Segmentation [(arxiv)](https://arxiv.org/abs/1908.05099)|None| [MICCAI MLMI 2019](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_71)|




The code of this repo is adapted from the following great repos.

- [nnUNet-Fabian](https://github.com/MIC-DKFZ/nnUNet)
> The most powerful U-Net implementation.

- [UAMT-Lequan Yu](https://github.com/yulequan/UA-MT)
> The code is very friendly for pytorch beginners.



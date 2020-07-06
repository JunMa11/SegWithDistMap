# 3D Medical Image Segmentation With Distance Transform Maps

## Motivation: How Distance Transform Maps Boost Segmentation CNNs [(MIDL 2020)](https://2020.midl.io/papers/ma20a.html)

Incorporating the distance Transform maps of image segmentation labels into CNNs-based segmentation tasks has received significant attention in 2019. These methods can be classified into two main classes in terms of the main usage of distance transform maps.

- Designing new loss functions
- Adding an auxiliary task, e.g. distance map regression

![Overview](https://github.com/JunMa11/SegWithDistMap/blob/master/overview.PNG)

However, with these new methods on the one hand and the diversity of the specific implementations and dataset-related challenges on the other, it's hard to figure out which design can generalize well beyond the experiments in the original papers. 
In this repository,  we want to re-implement these methods (published in 2019) and evaluate them on the same 3D segmentation tasks (heart and liver tumor segmentation).

## Experiments

| Task                                   | LA Contributor  | GPU        | LiTS Contributor  | GPU        |
| -------------------------------------- | ------------- | ---------- | ------------ | ---------- |
| Boundary loss                          | [Yiwen Zhang](https://github.com/whisney) | 2080ti     | [Mengzhang Li](https://github.com/MengzhangLI) | TITIAN RTX |
| Hausdorff loss                         | [Yiwen Zhang](https://github.com/whisney)  | 2080ti     | [Mengzhang Li](https://github.com/MengzhangLI) | TITIAN RTX |
| Signed  distance map loss (AAAI 2020)  | [Zhan Wei](https://github.com/zhanwei33)      | 1080ti     | cancel       | -          |
| Multi-Head: FG  DTM regression-L1      | [Yiwen Zhang](https://github.com/whisney)   | 2080ti     | cancel       | -          |
| Multi-Head: FG  DTM regression-L2      | [Jianan Liu]()    | 2080ti     | cancel       | -          |
| Multi-Head: FG  DTM regression-L1 + L2 | [Gaoxiang Chen](https://github.com/AMSTLHX) | 2080ti     | cancel       | -          |
| Multi-Head:  SDF regression-L1         | [Feng Cheng](836155475@qq.com)   | TITAN X    | [Chao Peng](https://github.com/AMSTLHX)    | TITAN RTX  |
| Multi-Head:  SDF regression-L2         | [Rongfei Lv](https://github.com/lrfdl)    | TITAN RTX  | [Rongfei Lv](https://github.com/lrfdl)   | TITAN RTX  |
| Multi-Head:  SDF regression-L1+L2      | [Yixin Wang](https://github.com/Wangyixinxin)    | P100       | cancel       | -          |
| Add-Branch: FG  DTM regression-L1      | [Yaliang Zhao](441926980)  | TITAN RTX  | cancel       | -          |
| Add-Branch: FG  DTM regression-L2      | [Mengzhang Li](https://github.com/MengzhangLI)  | TITIAN RTX | cancel       | -          |
| Add-Branch: FG  DTM regression-L1+L2   | [Yixin Wang](https://github.com/Wangyixinxin)    | P100       | cancel       | -          |
| Add-Branch:  SDF regression-L1         | [Feng Cheng](836155475@qq.com)    | TITAN X    | [Yixin Wang](https://github.com/Wangyixinxin)   | TITAN RTX  |
| Add-Branch:  SDF regression-L2         | [Feng Cheng](836155475@qq.com)    | TITAN X    | [Yixin Wang](https://github.com/Wangyixinxin)   | P100       |
| Add-Branch:  SDF regression-L1+L2      | [Yixin Wang](https://github.com/Wangyixinxin)    | P100       | [Yunpeng Wang]() | TITAN  XP  |

> [Here](https://github.com/JunMa11/SegWithDistMap/tree/master/code) is the code, and trained modles can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1E9SlHw4DXuvsqFQRD_HHag) (pw:mgn0).



## Related Work in 2019

### New loss functions

| Date | First author  | Title                       | Official Code  | Publication                    |
| ---- | ------------- | --------------------------- | -------------- | ------------------------------ |
| 2019 | Yuan Xue  | Shape-Aware Organ Segmentation by Predicting Signed Distance Maps | None         | [AAAI 2020](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-XueY.1482.pdf)  |
| 2019 | [Hoel Kervadec](https://scholar.google.com.hk/citations?user=yeFGhfgAAAAJ&hl=zh-CN&oi=sra) | Boundary loss for highly unbalanced segmentation | [pytorch](https://github.com/LIVIAETS/surface-loss) | [MIDL 2019](http://proceedings.mlr.press/v102/kervadec19a.html) |     
|2019|Davood Karimi|Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks [(arxiv)](https://arxiv.org/abs/1904.10030) |None|[TMI 2019](https://ieeexplore.ieee.org/document/8767031)|



### Auxiliary tasks

| Date | First author  | Title                       | Official Code  | Publication                    |
| ---- | ------------- | --------------------------- | -------------- | ------------------------------ |
| 2019 | Yan Wang     | Deep Distance Transform for Tubular Structure Segmentation in CT Scans | None | [CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Deep_Distance_Transform_for_Tubular_Structure_Segmentation_in_CT_Scans_CVPR_2020_paper.html) |
| 2019 | [Shusil Dangi](https://scholar.google.com.hk/citations?user=h12ifugAAAAJ&hl=zh-CN&oi=sra) |A Distance Map Regularized CNN for Cardiac Cine MR Image Segmentation [(arxiv)](https://arxiv.org/abs/1901.01238) | None | [Medical Physics](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.13853) |
|2019|[Fernando Navarro](https://scholar.google.com.hk/citations?user=rRKrhrwAAAAJ&hl=zh-CN&oi=sra)|Shape-Aware Complementary-Task Learning for Multi-organ Segmentation [(arxiv)](https://arxiv.org/abs/1908.05099)|None| [MICCAI MLMI 2019](https://link.springer.com/chapter/10.1007/978-3-030-32692-0_71)|


## Acknowledgments

The authors would like to thank the organization team of MICCAI 2017 liver tumor segmentation challenge MICCAI 2018 and left atrial segmentation challenge for the publicly available dataset. 
We also thank the reviewers for their valuable comments and suggestions. 
We appreciate Cheng Chen,  Feng Cheng, Mengzhang Li, Chengwei Su, Chengfeng Zhou and Yaliang Zhao to help us finish some experiments.
Last but not least, we thank Lequan Yu for his great PyTorch implementation of [V-Net](https://github.com/yulequan/UA-MT) and Fabian Isensee for his great PyTorch implementation of [nnU-Net](https://github.com/MIC-DKFZ/nnUNett).


## Including the following citation in your work would be highly appreciated.

```
@inproceedings{ma-MIDL2020-SegWithDist,
  title={How Distance Transform Maps Boost Segmentation CNNs: An Empirical Study},
  author={Ma, Jun and Wei, Zhan and Zhang, Yiwen and Wang, Yixin and Lv, Rongfei and Zhu, Cheng and Chen, Gaoxiang and Liu, Jianan and Peng, Chao and Wang, Lei and Wang, Yunpeng and Chen, Jianan},
  booktitle={Medical Imaging with Deep Learning},
  year={2020}
}
```

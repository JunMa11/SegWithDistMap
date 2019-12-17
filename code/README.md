### Requirements
- pytorch>=1.0
- tensorboardX
- scikit-image
- scipy
- tqdm

**Note: the code has been tested on ubuntu. I'm not sure whether it works on windows.**


## Training
### V-Net
- LA Heart MRI dataset: run `python train_LA.py`
- Liver tumor CT dataset: run `python train_LITS.py`

### V-Net with boundary loss
- LA Heart MRI dataset: run `python train_LA_BD.py`
- Liver tumor CT dataset: run `python train_LITS_BD.py`

> You need to set `--exp` properly. Both [compute_sdf](https://github.com/JunMa11/SegWithDistMap/blob/ed55b65889a4ba4cf9f7532e63124fe9ba10fcf0/code/train_LA_BD.py#L94) and [compute_sdf1_1](https://github.com/JunMa11/SegWithDistMap/blob/ed55b65889a4ba4cf9f7532e63124fe9ba10fcf0/code/train_LA_BD.py#L63) are worth to try.

### V-Net with hausdorff distance loss
- LA Heart MRI dataset: run `python train_LA_HD.py`
- Liver tumor CT dataset: run `python train_LITS_HD.py`

> You need to set `--exp` properly. Both [compute_dtm](https://github.com/JunMa11/SegWithDistMap/blob/ed55b65889a4ba4cf9f7532e63124fe9ba10fcf0/code/train_LA_HD.py#L86) and [compute_dtm01](https://github.com/JunMa11/SegWithDistMap/blob/ed55b65889a4ba4cf9f7532e63124fe9ba10fcf0/code/train_LA_HD.py#LL63) are worth to try.


## Testing
### Testing V-Net
- LA heart MRI dataset: run `python test_LA.py`
- Liver tumor CT dataset: run `python test_LITS.py`


## Following methods are ongoing! They do not work now!

### V-Net with [Signed distance maps](https://arxiv.org/abs/1912.03849)
> Xue et al. Shape-Aware Organ Segmentation by Predicting Signed Distance Maps [arxiv](https://arxiv.org/abs/1912.03849)

run `python train_LA_AAAISDF.py`

### V-Net with foreground distance map regression
> Wang et al. Deep Distance Transform for Tubular Structure Segmentation in CT Scans [arxiv](https://arxiv.org/abs/1912.03383)

> Navarro et al. Shape-Aware Complementary-Task Learning for Multi-organ Segmentation [arxiv](https://arxiv.org/abs/1908.05099)

run `python train_LA_Fore_Dist.py`

### V-Net with sighed distance function regression
run `python train_LA_SDF.py`


### Testing V-Net with various distance maps
run
`python test_LA_dis.py --models your_model_name`



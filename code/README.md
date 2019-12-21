### Requirements
- pytorch>=1.0
- tensorboardX
- scikit-image
- scipy
- tqdm

**Note: the code has been tested on ubuntu. I'm not sure whether it works on windows.**


## V-Net with different loss functions
### V-Net Training
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

### Testing
- LA heart MRI dataset: run `python test_LA.py`
- Liver tumor CT dataset: run `python test_LITS.py`


## [Signed distance map loss](https://arxiv.org/abs/1912.03849)

> Xue et al. Shape-Aware Organ Segmentation by Predicting Signed Distance Maps [arxiv](https://arxiv.org/abs/1912.03849)

### Training

- run `python train_LA_AAAISDF.py`
- run `python train_LA_AAAISDF_L1.py`

### Testing
- run `test_LA_AAAISDF.py`


## V-Net with additional heads
> Wang et al. Deep Distance Transform for Tubular Structure Segmentation in CT Scans [arxiv](https://arxiv.org/abs/1912.03383)

> Navarro et al. Shape-Aware Complementary-Task Learning for Multi-organ Segmentation [arxiv](https://arxiv.org/abs/1908.05099)

### Training

- run `python train_LA_MultiHead_FGDTM_L1.py` to regress foreground distance transform map

> L1 can be replaced with L2 or L1PlusL2

- run `python train_LA_MultiHead_SDF_L1.py` to regress signed distance function

> L1 can be replaced with L2 or L1PlusL2

### Testing

- run `test_LA_MultiHead_FGDTM.py `
- run `test_LA_MultiHead_SDF.py`

## V-Net with additional reconstruction branch
### Training

- run `python train_LA_Rec_FGDTM_L1.py` to regress foreground distance transform map

> L1 can be replaced with L2 or L1PlusL2

- run `python train_LA_Rec_SDF_L1.py` to regress signed distance function

> L1 can be replaced with L2 or L1PlusL2


### Testing

- run `test_LA_Rec_FGDTM.py `
- run `test_LA_Rec_SDF.py`

## Tips
- `--model` can be used to specificy the model name
- `--epoch_num` can be used to specificy the checkpoint





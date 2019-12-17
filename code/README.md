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



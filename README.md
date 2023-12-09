# AAAI2024 - MDMS
"Multi-Domain Multi-Scale Diffusion Model for Low-Light Image Enhancement"

## Demo
#### 1. Download pretrained model

Download the Pretrained MDMS model from [Baidu NetDisk](https://pan.baidu.com/s/1O8hOVflnLGLSLP07nXp_sg?pwd=zftu) or [Google Drive](https://drive.google.com/file/d/1RcYn6543x0TtrKwGGJMnk6Jf48qoh1TC/view?usp=sharing).

Put the downloaded ckpt in `datasets/scratch/LLIE/ckpts`.


#### 2. Inference
```
# in {path_to_this_repo}/,
$ python eval_diffusion.py
```
Put the test input in `datasets/scratch/LLIE/data/lowlight/test/input`.

Output results will be saved in `results/images/lowlight/lowlight`.

## Evaluation

Put the test GT in `datasets/scratch/LLIE/data/lowlight/test/gt` for paired evaluation.

```
# in {path_to_this_repo}/,
$ python evaluation.py
```
* Note that our [evaluation metrics](https://github.com/Oli-iver/MDMS/blob/main/evaluation.py) are slightly different from [PyDiff](https://github.com/limuloo/PyDIff/tree/862f8cc428450ef02822fd218b15705e2214ec2d/BasicSR-light/basicsr/metrics) (inherited from [BasicSR](https://github.com/XPixelGroup/BasicSR)).

## Results
All results listed in our paper including the compared methods are available in [Baidu Netdisk](https://pan.baidu.com/s/1O8hOVflnLGLSLP07nXp_sg?pwd=zftu) or [Google Drive](https://drive.google.com/file/d/1k9-vD-I5JaHj7Y9bGq1gen2TKEzEhCzs/view?usp=sharing).

* Note that the provided model is trained on the [LOLv1](https://daooshee.github.io/BMVC2018website/) training set, but generalizes well on other datasets.
* For SSIM, we directly calculate the performance on the [RGB channel](https://github.com/Oli-iver/MDMS/blob/main/evaluation.py#L49-L51) rather than [grayscale images](https://github.com/limuloo/PyDIff/blob/862f8cc428450ef02822fd218b15705e2214ec2d/BasicSR-light/basicsr/metrics/ssim_lol.py#L7C1-L12C132) in PyDiff.
* For LPIPS, we use a different normalization method ([NormA](https://github.com/Oli-iver/MDMS/blob/main/evaluation.py#L74)) compared to PyDiff ([NormB](https://github.com/limuloo/PyDIff/blob/862f8cc428450ef02822fd218b15705e2214ec2d/BasicSR-light/basicsr/metrics/lpips_lol.py#L19)).

Our method remains superior under the same setting as PyDiff.
![All text](https://github.com/Oli-iver/MDMS/blob/main/figs/com.png)

### 1. Test results on LOLv1 test set.
![All text](https://github.com/Oli-iver/MDMS/blob/main/figs/v1.png)

### 2. Generalization results on LOLv2 syn and real test sets.
![All text](https://github.com/Oli-iver/MDMS/blob/main/figs/vis.png)

### 3. Generalization results on other unpaired datasets.
![All text](https://github.com/Oli-iver/MDMS/blob/main/figs/unpaired.png)

We will perform more training and tests on other datasets in the future.

## Training
Put the training dataset in `datasets/scratch/LLIE/data/lowlight/train`.

```
# in {path_to_this_repo}/,
$ python train_diffusion.py
```

Detailed training instructions will be updated soon.

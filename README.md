# MDMS
"Multi-Domain Multi-Scale Diffusion Model for Low-Light Image Enhancement"

## Demo
### 1. Download pretrained model

* [Required] Download Pretrained MDMS: [Baidu NetDisk](https://pan.baidu.com/s/1J8rrUW8K0Cw2L94sgMI-vQ). (key: 1234)

This model is trained on the LOLv1 training set. 

### 2. Inference
Example:
```
# in {path_to_this_repo}/,
$ python evaluation.py
```
Put the test input in `datasets/scratch/LLIE/data/lowlight/test/input`.

Output results will be saved in `results/images/lowlight/lowlight`

## Results
All results and models listed in our paper are available in [Baidu Netdisk](https://pan.baidu.com/s/1O8hOVflnLGLSLP07nXp_sg?pwd=zftu) or [Google Drive](https://pan.baidu.com/s/1O8hOVflnLGLSLP07nXp_sg?pwd=zftu).

### 1. Reuslts on ParisStreet View dataset along with epochs.
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/GIF%202020-8-20%2010-56-41.gif)
### 2. Results on CelebA dataset along with phases.
* Phase 1 for 90,000
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/phase1_step9000.png)
* Phase 2 for 10,000
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/phase2_step1000.png)
* Phase 3 for 400,000
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/phase3_step40000.png)


## Training
* Training instructions will be updated soon.

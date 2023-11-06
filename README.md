# MDMS
"Multi-Domain Multi-Scale Diffusion Model for Low-Light Image Enhancement"

## Demo(Inference)
### 1. Download pretrained generator

* [Required] Pretrained generator model (Completion Network): [Baidu NetDisk](https://pan.baidu.com/s/1J8rrUW8K0Cw2L94sgMI-vQ). (key: 1234)
* [Optional] Pretrained discriminator model (Context Discriminator): [Baidu NetDisk](https://pan.baidu.com/s/1r2T4AKA0S96q0HqV62SC3g). (key: 5678)

Note that you don't need the dicriminator model for inference because only generator is necessary to perform image completion.

Both the generator and discriminator were trained on the CelebA. 

### 2. Inference
Example:
```
# in {path_to_this_repo}/,
$ python predict.py model_cn config.json images/test.jpg output.jpg
```


## Results

### 1. Reuslts on ParisStreet View dataset along with epochs.
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/GIF%202020-8-20%2010-56-41.gif)
### 2. Results on CelebA dataset along with phases.
* Phase 1 for 90,000
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/phase1_step9000.png)
* Phase 2 for 10,000
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/phase2_step1000.png)
* Phase 3 for 400,000
![All text](https://github.com/Oliiveralien/DLclass-Oliver_Inpainting/blob/master/images/phase3_step40000.png)

For all results and models, click [here](https://pan.baidu.com/s/1O8hOVflnLGLSLP07nXp_sg?pwd=zftu ). (key: zftu)

## Training
Training instructions will be updated soon.

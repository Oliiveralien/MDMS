# DLclass-Oliver_Inpainting
"A Context-Based Multi-Scale Discriminant Model for Natural Image Inpainting".

## Note
It is a model for inpainting task, which is based on [GL](https://dl.acm.org/doi/abs/10.1145/3072959.3073659) and [NIN](https://arxiv.org/abs/1312.4400).

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

## Demo(Training)

One can also train the network with their own datasets, by modifying the data path in `config.json`.

### 1. Prepare the dataset
Here we offer some official links used in paper.
* [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
* [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
* [Imagenet](http://www.image-net.org/)
* [LSUN](https://www.yf.io/p/lsun)
* [ParisStreet View](https://github.com/pathak22/context-encoder#6-paris-street-view-dataset) (Request permission through e-mail)

Process the data by running `./datasets/make_dataset.py` .

### 2. Training
Example:
```
# in {path_to_this_repo}/,
$ python python train.py datasets/img_align_celeba results/wc/
```
Training results (trained model snapshots and inference results) are to be saved in `./results/wc`.

The training procedure consists of the following three phases.

1. In phase 1, only Completion Network (i.e., generator) is trained.
2. In phase 2, only Context Discriminator (i.e., discriminator) is trained, while Completion Network is frozen.
3. In phase 3, Both of the networksd are jointly trained.

By default, the training steps during phase 1, 2, and 3 are set to 90,000, 10,000, and 400,000, respectively. 
One can also customize their training settings in `./config.json`.

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

For more results and models during training, click [here](https://pan.baidu.com/s/1dFI-yhNvX0br5cMRNj07cA). (key: 6666)

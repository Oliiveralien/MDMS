
import os
from os import listdir
from os.path import isfile
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL
import re
import random
import torchvision.transforms as transforms

class lowlight:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_loaders(self, parse_patches=True, validation='lowlight'):
        print("=> evaluating lowlight test set...")
        train_dataset = lowlightDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'lowlight', 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=None,
                                        parse_patches=parse_patches,
                                        train=True)
        val_dataset = lowlightDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'lowlight', 'test'),
                                      n=self.config.training.patch_n,
                                      patch_size=self.config.data.image_size,
                                      transforms=self.transforms,
                                      filelist='lowlighttesta.txt',
                                      parse_patches=parse_patches,
                                      train=False)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class lowlightDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, train,filelist=None, parse_patches=True):
        super().__init__()

        if filelist is None:
            lowlight_dir = dir
            input_names, gt_names = [], []



            filepath = os.path.dirname(__file__)
            print(f"filepath:{filepath}")
            lowlight_dir=os.path.join(filepath,lowlight_dir)
            #lowlight_inputs = os.path.join(filepath, lowlight_inputs0)

            # lowlight train filelist
            lowlight_inputs = os.path.join(lowlight_dir, 'input')
            listdir(lowlight_inputs)
            images = [f for f in listdir(lowlight_inputs) if isfile(os.path.join(lowlight_inputs, f))]
            assert len(images) == 485
            input_names += [os.path.join(lowlight_inputs, i) for i in images]
            # gt_names += [os.path.join(os.path.join(lowlight_dir, 'gt'), i.replace('rain', 'clean')) for i in images]
            gt_names += [os.path.join(os.path.join(lowlight_dir, 'gt'), i.replace('', '')) for i in images]
            print(len(input_names))

            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            self.dir = None
        else:
            self.dir = dir
            filepath = os.path.dirname(__file__)
            dir = os.path.join(filepath, dir)
            train_list = os.path.join(dir, filelist)
            with open(train_list) as f:
                contents = f.readlines()
                input_names = [i.strip() for i in contents]
                gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.batchnum = 0
        self.batchsize = 1
        self.train=train

    @staticmethod
    def get_params(img, output_size, n,random_size):
        w, h = img.size
        if random_size == 0:
            output_size = (64,64)
        elif random_size == 1:
            output_size = (128,128)
        else:
            output_size = (256,256)
        # output_size = (192,192)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        osize = [output_size[0] for _ in range(n)]
        return i_list, j_list, th, tw,osize,h,w

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        
            if h!=64:
                resize_transform = transforms.Resize(64)
                new_crop=resize_transform(new_crop)
            crops.append(new_crop)
        return tuple(crops)

    def get_max(self,input):
        T,_=torch.max(input,dim=0)
        T=T+0.1
        input[0,:,:] = input[0,:,:]/ T
        input[1,:,:] = input[1,:,:]/ T
        input[2,:,:]= input[2,:,:] / T
        return input

    def get_images(self, index):
        if self.train==True:
            if self.batchnum==0:
                self.random_size = random.randint(0, 2)
            self.batchnum=self.batchnum+1
            if self.batchnum==self.batchsize:
                self.batchnum=0
        else:
            self.random_size=0

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
                PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:

            i, j, h, w,osize,h_org,w_org = self.get_params(input_img, (self.patch_size, self.patch_size), self.n,self.random_size)
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)


            random_input=0
            if random_input==0:
                outputs = [torch.cat([self.transforms(input_img[i]),self.get_max(self.transforms(input_img[i])),self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            else:
                outputs = [torch.cat([self.get_max(self.transforms(input_img[i])), self.transforms(input_img[i]),
                                      self.transforms(gt_img[i])], dim=0)
                           for i in range(self.n)]
            ii=torch.tensor(i)
            jj = torch.tensor(j)
            ii = (ii / h_org) * 2 - 1
            jj = (jj / w_org) * 2 - 1
            osize=torch.tensor(osize)
            return torch.stack(outputs, dim=0), img_id,ii,jj,osize
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            wd=wd_new
            ht=ht_new
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024


           
            wd_new = int(8 * np.ceil(wd_new / 8.0))
            ht_new = int(8 * np.ceil(ht_new / 8.0))
            input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

            return torch.cat([self.transforms(input_img),self.get_max(self.transforms(input_img)), self.transforms(gt_img)], dim=0), img_id,wd,ht

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)

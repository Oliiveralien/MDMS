import torch
import torch.nn as nn
import utils
import torchvision
import os
import PIL
import re
from torchvision.transforms import Resize

def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='lowlight', r=None,use_align=False):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        with torch.no_grad():
            for i, (x, y,wd,ht) in enumerate(val_loader):
                print(f"starting processing from image {y}")
                y = re.findall(r'\d+', y[0])
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                x_cond = x[:, :6, :, :].to(self.diffusion.device)
                x_output1 = self.diffusive_restoration(x_cond, r=r,fullresolusion=False)
                #x_output = inverse_data_transform(x_output)
                x_output1 = inverse_data_transform(x_output1)
                x_output=x_output1
                #x_output=(x_output+x_output1)/2
                b,c,h,w=x_output.shape
                ht=ht.item()
                wd=wd.item()
                torch_resize = Resize([ht, wd])
                x_output=torch_resize(x_output)
                
                if use_align==True:
                    target = x[:, 6:, :, :]
                    gt_mean = torch.mean(target)
                    sr_mean = torch.mean(x_output)
                    x_output = x_output * gt_mean / sr_mean
                


                utils.logging.save_image(x_output, os.path.join(image_folder, f"{y}.png"))

    def diffusive_restoration(self, x_cond, r=None,fullresolusion=False):
        if fullresolusion==False:##
            # p_size = self.config.data.image_size
            p_size=64
            h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=8)
            corners = [(i, j) for i in h_list for j in w_list]
            h_list1, w_list1 = self.overlapping_grid_indices(x_cond, output_size=96, r=8)
            corners1 = [(i, j) for i in h_list1 for j in w_list1]

            h_list2, w_list2 = self.overlapping_grid_indices(x_cond, output_size=128, r=8)
            corners2 = [(i, j) for i in h_list2 for j in w_list2]


            x = torch.randn(x_cond.size()[0],3,x_cond.size()[2],x_cond.size()[3], device=self.diffusion.device)

            ii = torch.tensor([item[0] for item in corners])
            jj = torch.tensor([item[1] for item in corners])
            ii=ii/x_cond.size()[2]*2-1
            jj=jj/x_cond.size()[3]*2-1
            osize=torch.full((len(corners),), p_size)
            x_output = self.diffusion.sample_image(x_cond, x, ii,jj,osize,patch_locs=corners, patch_size=p_size,patch_locs1=corners1,patch_locs2=corners2)
        else:
            x = torch.randn(x_cond.size()[0], 3, x_cond.size()[2], x_cond.size()[3], device=self.diffusion.device)
            ii=torch.tensor(-1).unsqueeze(0)
            jj=torch.tensor(-1).unsqueeze(0)
            osize=torch.tensor(x_cond.size()[2]).unsqueeze(0)
            x_output = self.diffusion.sample_image(x_cond, x, ii, jj, osize, patch_locs=None, patch_size=None)


        return x_output
        #return x_output,x_output1


    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list

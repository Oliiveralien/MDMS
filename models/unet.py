

import math
import torch
import torch.nn as nn

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    # assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2  
   
    emb = math.log(10000) / (half_dim - 1) 
    
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)  
    
    emb = emb.to(device=timesteps.device) 
  
    emb = timesteps.float()[:, None] * emb[None, :]  
    
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  

    if embedding_dim % 2 != 0:  
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))  
    
    return emb  
   


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class HighMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 **kwargs, ):
        super().__init__()

        self.cnn_in = cnn_in = dim
        # self.pool_in = pool_in = dim-cnn_in

        self.cnn_dim = cnn_dim = cnn_in
        # self.pool_dim = pool_dim = pool_in

        self.conv1 = nn.Conv2d(cnn_in, cnn_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.proj1 = nn.Conv2d(cnn_dim, cnn_dim, kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
                               groups=cnn_dim)
        self.mid_gelu1 = nn.GELU()



    def forward(self, x):
        # B, C H, W

        cx = x[:, :self.cnn_in, :, :].contiguous()
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        return cx


class LowMixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.pool = nn.AvgPool2d(pool_size, stride=pool_size, padding=0,
                                 count_include_pad=False) if pool_size > 1 else nn.Identity()
        self.uppool = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def att_fun(self, q, k, v, B, N, C):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(2, 3).reshape(B, C, N)
        return x

    def forward(self, x):
        # B, C, H, W
        B, _, _, _ = x.shape
        x = self.pool(x)
        xa = x.permute(0, 2, 3, 1).view(B, -1, self.dim)
        B, N, C = xa.shape
        qkv = self.qkv(xa).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        xa = self.att_fun(q, k, v, B, N, C)
        xa = xa.view(B, C, int(N ** 0.5), int(N ** 0.5))  # .permute(0, 3, 1, 2)

        xa = self.uppool(xa)
        return xa


class Mixer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2,
                 **kwargs, ):
        super().__init__()
        self.num_heads = num_heads
        self.low_dim = low_dim = dim //2
        self.high_dim = high_dim = dim - low_dim
        self.high_mixer = HighMixer(high_dim)
        self.low_mixer = LowMixer(low_dim, num_heads=attention_head, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                  pool_size=pool_size, )

        self.conv_fuse = nn.Conv2d(low_dim + high_dim , low_dim + high_dim , kernel_size=3, stride=1, padding=1,
                                   bias=False, groups=low_dim + high_dim)
        self.proj = nn.Conv2d(low_dim + high_dim, dim, kernel_size=1, stride=1, padding=0)
        self.proj_drop = nn.Dropout(proj_drop)

        self.freblock=FreBlock(dim,dim)
        self.finalproj = nn.Conv2d(2 * dim, dim, 1, 1, 0)
    def forward(self, x):

        B, C, H, W = x.shape  #16,128,64,64
        #x = x.permute(0, 3, 1, 2)
        x_ori=x

        hx = x[:, :self.high_dim, :, :].contiguous()#16,64,64,64
        hx = self.high_mixer(hx)#16,64,64,64

        lx = x[:, self.high_dim:, :, :].contiguous()#16,64,64,64
        lx = self.low_mixer(lx)#16,64,64,64
        x = torch.cat((hx, lx), dim=1)#16,128,64,64
        
        x = x + self.conv_fuse(x)
        x_sptial = self.proj(x)
       
        x_freq=self.freblock(x_ori)

        x_out=torch.cat((x_sptial,x_freq),1)
        x_out=self.finalproj(x_out)
        x_out=self.proj_drop(x_out)
        return x_out+x_ori


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512,incep=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        # self.conv1=FreBlock(in_channels,out_channels)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        if incep==True:
            self.conv2 =Mixer(dim=out_channels)
        else:
            self.conv2 = torch.nn.Conv2d(out_channels,
                                         out_channels,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)     #(16,128,64,64)
        h = self.conv1(h)       #(16,128,64,64)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)       #(16,128,64,64)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class FreBlock(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,1,1,0),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(in_channels,out_channels,1,1,0))
        self.processpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self,x):
        xori = x
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out1 = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out1
class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 3 if config.data.conditional else config.model.in_channels##为了max图，所以*3
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch                #128
        self.temb_ch = self.ch*4    #512
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.inceplayers=2

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                if i_level+self.inceplayers<self.num_resolutions:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                else:
                    block.append(ResnetBlock(in_channels=block_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1 :
                if ch_mult[i_level]*2==ch_mult[i_level+1]:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,incep=True)
        # self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout,incep=True)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                if i_level+self.inceplayers>=self.num_resolutions:
                    block.append(ResnetBlock(in_channels=block_in+skip_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                else:
                    block.append(ResnetBlock(in_channels=block_in+skip_in,
                                             out_channels=block_out,
                                             temb_channels=self.temb_ch,
                                             dropout=dropout,incep=True))
                block_in = block_out
                # if curr_res in attn_resolutions:
                #     attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0 :
                if ch_mult[i_level]== ch_mult[i_level-1]*2:
                    up.upsample = Upsample(block_in, resamp_with_conv)
                    curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t,i,j,osize):
        assert x.shape[2] == x.shape[3]
        # timestep embedding
        # t=torch.stack([t.unsqueeze(1),cond],dim=1)
        temb1 = get_timestep_embedding(t, self.ch//4)#(16,32)
        temb2 = get_timestep_embedding(i, self.ch//4)  # (16,32)
        temb3 = get_timestep_embedding(j, self.ch//4)  # (16,32)
        temb4 = get_timestep_embedding(osize, self.ch//4)  # (16,32)

        temb=torch.cat([temb1,temb2,temb3,temb4],dim=1)
        temb = self.temb.dense[0](temb)         #(16,512)
        temb = nonlinearity(temb)               #(16,512)
        temb = self.temb.dense[1](temb)         #(16,512)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
              
            if i_level != self.num_resolutions-1 :
                hs.append(self.down[i_level].downsample(hs[-1]))
               

        # middle
        h = hs[-1]       
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

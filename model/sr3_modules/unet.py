import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import torch.fft as fft
from torchsummary import summary
from thop import profile, clever_format
import numpy as np


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
def create_model(
              in_channel,
              out_channel,
              inner_channel,
              channel_multiplier,
              attn_res,
              res_blocks,
              dropout
):
    model = UNet(in_channel=in_channel,out_channel=out_channel,inner_channel=inner_channel,channel_mults=channel_multiplier,attn_res=attn_res,res_blocks=res_blocks,dropout=dropout)
    return model


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            # print(self.noise_func(noise_embed).view(batch, -1, 1, 1).shape)
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=16):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):


        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def Reverse(lst):
    return [ele for ele in reversed(lst)]

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        inner_channel=16,
        norm_groups=16,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res= [16],
        res_blocks=2,
        dropout=0.2,
        with_noise_level_emb=True,
        image_size=256,
        b1 = 1.2,
        b2 = 1.4,
        b3 = 1.6,
        b4 = 1.8,
        b5 = 1.8,
        s1 = 0.9,
        s2 = 0.7,
        s3 = 0.5,
        s4 = 0.3,
        s5 = 0.2,

    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        self.inner_channel = inner_channel
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5 = s5
        self.init_conv = nn.Conv2d(in_channels=in_channel, out_channels=inner_channel, kernel_size=3, padding=1)
        downs = []
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x,time, feat_need=False):

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None
        # time = 1000
        # time = torch.FloatTensor(np.random.uniform(
        #         time-1,
        #         time,
        #         size=2
        #     )).to('cuda')
        #
        # t = self.noise_level_mlp(timestep_embedding(time,8))



        # First downsampling layer

        x  = self.init_conv(x)


        # Diffusion encoder
        feats = [x]
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)
        
        if feat_need:
            fe = feats.copy()

        # Passing through middle layer
        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        # Saving decoder features for CD Head
        if feat_need:
            fd = []

        # Diffiusion decoder
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
                if feat_need:
                    fd.append(x)
            else:
                x = layer(x)
        # for layer in self.ups:
        #     if isinstance(layer, ResnetBlocWithAttn):
        #         x_s = feats.pop()
        #         print(x.shape[1])
        #         if x.shape[1] == h:
        #             print(x.shape)
        #             hidden_mean = x.mean(1).unsqueeze(1)
        #             B = hidden_mean.shape[0]
        #             hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             print(hidden_max.shape)
        #             hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        #                         hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        #             print(hidden_mean.shape)
        #
        #             x[:, :h/2] = x[:, :h/2] * ((self.b1 - 1) * hidden_mean + 1)
        #             x_s = Fourier_filter(x_s, threshold=1, scale=self.s1)
        #             print(x_s.shape)
        #         if x.shape[1] == h/2:
        #             hidden_mean = x.mean(1).unsqueeze(1)
        #             B = hidden_mean.shape[0]
        #             hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        #                         hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        #
        #             x[:, :h/4] = x[:, :h/4] * ((self.b2 - 1) * hidden_mean + 1)
        #             x_s = Fourier_filter(x_s, threshold=1, scale=self.s2)
        #         if x.shape[1] == h/4:
        #             hidden_mean = x.mean(1).unsqueeze(1)
        #             B = hidden_mean.shape[0]
        #             hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        #                         hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        #
        #             x[:, :h/8] = x[:, :h/8] * ((self.b3 - 1) * hidden_mean + 1)
        #             x_s = Fourier_filter(x_s, threshold=1, scale=self.s3)
        #         if x.shape[1] == h/8:
        #             hidden_mean = x.mean(1).unsqueeze(1)
        #             B = hidden_mean.shape[0]
        #             hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        #                         hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        #
        #             x[:, :h/16] = x[:, :h/16] * ((self.b4 - 1) * hidden_mean + 1)
        #             x_s = Fourier_filter(x_s, threshold=1, scale=self.s4)
        #         if x.shape[1] == h/16:
        #             hidden_mean = x.mean(1).unsqueeze(1)
        #             B = hidden_mean.shape[0]
        #             hidden_max, _ = torch.max(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_min, _ = torch.min(hidden_mean.view(B, -1), dim=-1, keepdim=True)
        #             hidden_mean = (hidden_mean - hidden_min.unsqueeze(2).unsqueeze(3)) / (
        #                         hidden_max - hidden_min).unsqueeze(2).unsqueeze(3)
        #
        #             x[:, :h/32] = x[:, :h/32] * ((self.b5 - 1) * hidden_mean + 1)
        #             x_s = Fourier_filter(x_s, threshold=1, scale=self.s5)
        #
        #         x = torch.cat([x, x_s], dim=1)
        #
        #
        #
        #
        #         x = layer(torch.cat((x, feats.pop()), dim=1), t)
        #         if feat_need:
        #             fd.append(x)
        #     else:
        #         x = layer(x)

        # Final Diffusion layer
        x = self.final_conv(x)

        # Output encoder and decoder features if feat_need
        if feat_need:
            return fe, Reverse(fd)
        else:
            return x


def Fourier_filter(x, threshold, scale):
    dtype = x.dtype
    x = x.type(torch.float32)
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))

    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda()

    crow, ccol = H // 2, W // 2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real

    x_filtered = x_filtered.type(dtype)
    return x_filtered


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     net = UNet().to(device)
#     input_shape = (3, 256, 256)
#     summary(net, input_shape)
#
#     input_tensor = torch.randn(2, *input_shape).to(device)
#     flops, params = profile(net, inputs=(input_tensor,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print("FLOPs: %s" % (flops))
#     print("params: %s" % (params))
# if __name__ == "__main__":
#     main()

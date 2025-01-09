import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import model as Model
import logger as Logger
import argparse
from cross import CrissCrossAttention

EPSION = 1e-5

#############加载pre-diffusion model###############################################
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config/ddpm.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                    help='Run either train(training + validation) or testing', default='train')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_eval', action='store_true')
args = parser.parse_args()
opt = Logger.parse(args)
opt = Logger.dict_to_nonedict(opt)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
diffusion = Model.create_model(opt)
# Set noise schedule for the diffusion model
diffusion.set_new_noise_schedule(opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])





class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):


    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        # output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):


    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        output_tensor = self.cSE(input_tensor) + self.sSE(input_tensor)
        return output_tensor


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.leaky_relu(out, negative_slope=0.2)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out



class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, is_last=True)
        self.ChannelSpatialSELayer = ChannelSpatialSELayer(num_channels=out_channels, reduction_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.ChannelSpatialSELayer(x)
        return x


class Decoder1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, is_last=True)

    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)

        return x


class Decoder2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.ChannelSpatialSELayer = ChannelSpatialSELayer(num_channels=out_channels, reduction_ratio=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ChannelSpatialSELayer(x)
        return x


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        kernel_size = 1
        stride = 1

        self.down1 = nn.AvgPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=4)
        self.up3 = nn.Upsample(scale_factor=8)
        self.up4 = nn.Upsample(scale_factor=16)

        self.conv_in1 = ConvLayer(64, 64, kernel_size =3, stride =1)
        self.conv_in2 = ConvLayer(32, 16, kernel_size, stride)
        self.conv_in3 = ConvLayer(64, 32, kernel_size, stride)
        self.conv_in4 = ConvLayer(128, 64, kernel_size, stride)
        self.conv_in5 = ConvLayer(128, 128, kernel_size, stride)

        self.conv_t4 = ConvLayer(128, 64, kernel_size=3, stride=1)

        self.conv_t3 = ConvLayer(128, 64, kernel_size=3, stride=1)
        self.conv_t2 = ConvLayer(64, 32, kernel_size=3, stride=1)
        self.conv_t1 = Decoder1(64, 1, kernel_size=1, stride=1)

        self.en0 = Encoder(96, 32, kernel_size=3, stride=1)
        self.en1 = Encoder(192, 32, kernel_size=3, stride=1)
        self.en2 = Encoder(384, 32, kernel_size=3, stride=1)
        self.en3 = Encoder(768, 32, kernel_size=3, stride=1)
        self.crossatten = CrissCrossAttention(32)
        self.sigmod = nn.Sigmoid()

    def encoder(self, ir, vi):
        ir1_te0, ir1_td0 = diffusion.get_feats(ir, t=5)
        ir1_te1, ir1_td1 = diffusion.get_feats(ir, t=50)
        ir1_te2, ir1_td2 = diffusion.get_feats(ir, t=100)
        vi1_te0, vi1_td0 = diffusion.get_feats(vi, t=5)
        vi1_te1, vi1_td1 = diffusion.get_feats(vi, t=50)
        vi1_te2, vi1_td2 = diffusion.get_feats(vi, t=100)

        del ir1_te0, vi1_te0, ir1_te1, ir1_te2, vi1_te1, vi1_te2


        ir1 = torch.cat([ir1_td0[2], ir1_td1[2]], 1)
        ir1 = torch.cat([ir1, ir1_td2[2]], 1)
        ir1 = self.en0(ir1)
        ir2 = torch.cat([ir1_td0[5], ir1_td1[5]], 1)
        ir2 = torch.cat([ir2, ir1_td2[5]], 1)
        ir2 = self.en1(ir2)
        ir3 = torch.cat([ir1_td0[8], ir1_td1[8]], 1)
        ir3 = torch.cat([ir3, ir1_td2[8]], 1)
        ir3 = self.en2(ir3)
        ir4 = torch.cat([ir1_td0[11], ir1_td1[11]], 1)
        ir4 = torch.cat([ir4, ir1_td2[11]], 1)
        ir4 = self.en3(ir4)
        ir5 = torch.cat([ir1_td0[14], ir1_td1[14]], 1)
        ir5 = torch.cat([ir5, ir1_td2[14]], 1)
        ir5 = self.en3(ir5)

        vi1 = torch.cat([vi1_td0[2], vi1_td1[2]], 1)
        vi1 = torch.cat([vi1, vi1_td2[2]], 1)
        vi1 = self.en0(vi1)
        vi2 = torch.cat([vi1_td0[5], vi1_td1[5]], 1)
        vi2 = torch.cat([vi2, vi1_td2[5]], 1)
        vi2 = self.en1(vi2)
        vi3 = torch.cat([vi1_td0[8], vi1_td1[8]], 1)
        vi3 = torch.cat([vi3, vi1_td2[8]], 1)
        vi3 = self.en2(vi3)
        vi4 = torch.cat([vi1_td0[11], vi1_td1[11]], 1)
        vi4 = torch.cat([vi4, vi1_td2[11]], 1)
        vi4 = self.en3(vi4)
        vi5 = torch.cat([vi1_td0[14], vi1_td1[14]], 1)
        vi5 = torch.cat([vi5, vi1_td2[14]], 1)
        vi5 = self.en3(vi5)

        return ir1, ir2, ir3, ir4, ir5, vi1, vi2, vi3, vi4, vi5

    def fusion(self, ir_feature, vi_feature):

        ir_c,vi_c = self.crossatten(ir_feature,vi_feature)
        out = torch.cat([ir_c, vi_c], 1)
        # out = ir_c + vi_c
        return out

    def decoder(self, f1, f2, f3, f4, f5):
        f_4 = self.conv_t4(torch.cat([self.conv_in1(self.up1(f5)), f4], 1))
        f_3 = self.conv_t4(torch.cat([self.conv_in1(self.up1(f_4)), f3], 1))
        f_2 = self.conv_t4(torch.cat([self.conv_in1(self.up1(f_3)), f2], 1))
        f_1 = self.conv_t4(torch.cat([self.conv_in1(self.up1(f_2)), f1], 1))
        f = self.conv_t1(f_1)
        return f

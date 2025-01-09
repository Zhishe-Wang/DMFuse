'''
This code is borrowed from Serge-weihao/CCNet-Pure-Pytorch
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        m_batchsize, _, height, width = x.size()
        proj_query_x = self.query_conv(x)
        proj_query_x_H = proj_query_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_x_W = proj_query_x.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_query_y = self.query_conv(y)
        proj_query_y_H = proj_query_y.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_y_W = proj_query_y.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)

        proj_key_x = self.key_conv(x)
        proj_key_x_H = proj_key_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_x_W = proj_key_x .permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_key_y = self.key_conv(y)
        proj_key_y_H = proj_key_y.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_y_W = proj_key_y.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value_x = self.value_conv(x)
        proj_value_x_H = proj_value_x.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_x_W = proj_value_x.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)


        proj_value_y = self.value_conv(y)
        proj_value_y_H = proj_value_y.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_y_W = proj_value_y.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # energy_x_H = (torch.bmm(proj_query_x_H, proj_key_x_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        # energy_x_W = torch.bmm(proj_query_x_W, proj_key_x_W).view(m_batchsize, height, width, width)
        # concate_x = self.softmax(torch.cat([energy_x_H, energy_x_W], 3))
        energy_xy_H = (torch.bmm(proj_query_x_H, proj_key_y_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_xy_W = torch.bmm(proj_query_x_W, proj_key_y_W).view(m_batchsize, height, width, width)
        concate_xy = self.softmax(torch.cat([energy_xy_H, energy_xy_W], 3))

        # energy_y_H = (torch.bmm(proj_query_y_H, proj_key_y_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height, height).permute(0, 2, 1, 3)
        # energy_y_W = torch.bmm(proj_query_y_W, proj_key_y_W).view(m_batchsize, height, width, width)
        # concate_y = self.softmax(torch.cat([energy_y_H, energy_y_W], 3))

        energy_yx_H = (torch.bmm(proj_query_y_H, proj_key_x_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width, height,height).permute( 0, 2, 1, 3)
        energy_yx_W = torch.bmm(proj_query_y_W, proj_key_x_W).view(m_batchsize, height, width, width)
        concate_yx = self.softmax(torch.cat([energy_yx_H, energy_yx_W], 3))


        att_xy_H = concate_xy[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_xy_W = concate_xy[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)


        att_yx_H = concate_yx[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        att_yx_W = concate_yx[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)

        out_H1 = torch.bmm(proj_value_x_H, att_yx_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W1 = torch.bmm(proj_value_x_W, att_yx_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        out_H2 = torch.bmm(proj_value_y_H, att_xy_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W2 = torch.bmm(proj_value_y_W, att_xy_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H1.size(),out_W1.size())
        return self.gamma * (out_H1 + out_W1) + x, self.gamma * (out_H2 + out_W2) + y




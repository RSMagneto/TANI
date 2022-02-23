import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Net(nn.Module):
    def __init__(self, opt, spatial_height, spatial_width):
        super(Net, self).__init__()
        self.ms1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pan1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.exchange = sa_layer(64)
        self.ms2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pan2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.gap = _CrossNeuronBlock(spatial_height, spatial_width)
        self.mp1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.mp2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=3, stride=1, padding=1)

    def forward(self, ms,pan):
        ms1 = self.ms1(ms)
        pan1 = self.pan1(pan)
        ms_ec, pan_ec = (self.exchange(torch.cat([ms1, pan1], 1))).chunk(2, dim=1)
        ms2 = self.ms2(ms_ec)
        pan2 = self.pan2(pan_ec)
        mp = torch.cat((ms2,pan2),1)
        mp = self.gap(mp)
        mp1 = self.mp1(mp)
        mp2 = self.mp2(mp1)
        return mp2


class _CrossNeuronBlock(nn.Module):
    def __init__(self, spatial_height, spatial_width):
        super(_CrossNeuronBlock, self).__init__()
        self.spatial_area = spatial_height * spatial_width
        self.spatial_height = spatial_height
        self.spatial_width = spatial_width
        self.bn = nn.BatchNorm1d(self.spatial_area)

    def forward(self, x):
        bt, c, h, w = x.shape
        residual = x
        x_stretch = x.view(bt, c, h * w)
        x_stacked = x_stretch
        x_stacked = x_stacked.view(bt , c , -1)
        x_v = x_stacked.permute(0, 2, 1).contiguous()
        x_m = x_v.mean(1).view(-1, 1, c ).detach()
        score = -(x_m - x_m.permute(0, 2, 1).contiguous()) ** 2
        attn = F.softmax(score, dim=1)
        out = torch.bmm(x_v, attn)
        out = out.permute(0, 2, 1).contiguous().view(bt, c, h, w)
        return F.relu(residual + 0.1*out)


class sa_layer(nn.Module):
    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))
        self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=1,padding=0)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, padding=0)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        xn = self.avg_pool(x_0)
        xn = self.conv1(xn)
        xn = x_0 * self.sigmoid(xn)
        xs = self.gn(x_1)
        xs = self.conv2(xs)
        xs = x_1 * self.sigmoid(xs)
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)
        out = self.channel_shuffle(out, 2)
        return out
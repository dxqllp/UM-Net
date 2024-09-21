import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class HPPF(nn.Module):
    def __init__(self, in_channels):
        super(HPPF, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 16, 1, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 64, 1, 1), nn.ReLU(inplace=True))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max1 = nn.AdaptiveMaxPool2d(4)
        self.max2 = nn.AdaptiveMaxPool2d(8)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.Sigmoid())
        self.feat_conv = nn.Sequential(nn.Conv2d(in_channels, in_channels // 3, 3, 1, 1),
                                       nn.BatchNorm2d(in_channels // 3),
                                       nn.ReLU(inplace=True))

    def forward(self, x1, x2, x3):
        x2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode='bilinear', align_corners=True)
        feat = torch.cat((x1, x2, x3), 1)

        b, c, h, w = feat.size()
        y1 = self.avg(feat)
        y2 = self.conv1(self.max1(feat))
        y3 = self.conv2(self.max2(feat))
        y2 = y2.reshape(b, c, 1, 1)
        y3 = y3.reshape(b, c, 1, 1)
        z = (y1 + y2 + y3) // 3
        attention = self.mlp(z)
        output1 = attention * feat
        output2 = self.feat_conv(output1)

        return output2


class ALGM(nn.Module):
    def __init__(self, mid_ch, pool_size=(), out_list=(), cascade=False):
        super(ALGM, self).__init__()
        in_channels = mid_ch // 4
        self.cascade = cascade
        self.out_list = out_list
        size = [1, 2, 3]
        LGlist = []
        LGoutlist = []

        LGlist.append(NonLocalBlock(in_channels))
        for i in size:
            LGlist.append(nn.Sequential(
                nn.Conv2d(in_channels*i, in_channels, 3, stride=1, padding=pool_size[i-1], dilation=pool_size[i-1]),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)))
        self.LGmodule = nn.ModuleList(LGlist)

        for j in range(len(self.out_list)):
            LGoutlist.append(nn.Sequential(SELayer(in_channels*4),
                                           nn.Conv2d(in_channels * 4, self.out_list[j], 3, 1, 1),
                                           nn.BatchNorm2d(self.out_list[j]),
                                           nn.ReLU(inplace=True)))
        self.LGoutmodel = nn.ModuleList(LGoutlist)
        self.conv1 = nn.Sequential(nn.Conv2d(mid_ch, in_channels, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def forward(self, x, y=None):
        xsize = x.size()[2:]
        x = self.conv1(x)
        lg_context = []
        for i in range(2):
            lg_context.append(self.LGmodule[i](x))
        x1 = torch.cat((x, lg_context[1]), 1)
        lg_context.append(self.LGmodule[2](x1))
        x2 = torch.cat((x, lg_context[1], lg_context[2]), 1)
        lg_context.append(self.LGmodule[3](x2))
        lg_context = torch.cat(lg_context, dim=1)

        output = []
        for i in range(len(self.LGoutmodel)):
            out = self.LGoutmodel[i](lg_context)
            if self.cascade is True and y is not None:
                m = self.conv2(abs(F.interpolate(y[i], xsize, mode='bilinear', align_corners=True) - out))
                out = out + m
            output.append(out)

        return output


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False))
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c_avg = self.mlp(self.avg_pool(x))
        c_max = self.mlp(self.max_pool(x))
        c_out = self.sigmoid(c_avg + c_max)
        y1 = c_out * x

        s_avg = torch.mean(y1, dim=1, keepdim=True)
        s_max, _ = torch.max(y1, dim=1, keepdim=True)
        s_out = torch.cat((s_max, s_avg), 1)
        s_out = self.sigmoid(self.conv(s_out))
        output = s_out * y1

        return output


class RCG(nn.Module):
    def __init__(self):
        super(RCG, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1),
                                 nn.Sigmoid())

    def forward(self, pre, edge, f):
        f_att = torch.sigmoid(pre)
        r_att = -1 * f_att + 1
        r = r_att * f

        edge1 = F.interpolate(edge, size=f.size()[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat((edge1, r), 1)
        x2 = self.conv1(x1)
        x3 = self.mlp(x2)
        x4 = x3 * x2
        output = x4 + f

        return output


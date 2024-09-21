import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from models.modules import *


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4,  3, 1, 1),
                                   nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels // 4, out_channels, 3, 1, 1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=True)
        return x3


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SideoutBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels // 4, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels // 4),
                                   nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class UM-Net(nn.Module):
    def __init__(self, num_classes):
        super(UM-Net, self).__init__()
        resnet = models.resnet34(pretrained=True)

        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu  # 64
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1  # 64
        self.encoder3 = resnet.layer2  # 128
        self.encoder4 = resnet.layer3  # 256
        self.encoder5 = resnet.layer4  # 512

        self.down3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.down5 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.hpp = HPPF(192)
        self.cbam = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  CBAM(64),
                                  nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.line_predict = nn.Conv2d(64, 1, 3, 1, 1)

        self.lg5 = ALGM(64, pool_size=[1, 3, 5], out_list=[64, 64, 64, 64])
        self.lg4 = ALGM(64, pool_size=[2, 6, 10], out_list=[64, 64, 64], cascade=True)
        self.lg3 = ALGM(64, pool_size=[3, 9, 15], out_list=[64, 64], cascade=True)
        self.lg2 = ALGM(64, pool_size=[4, 12, 20], out_list=[64], cascade=True)

        self.side2 = SideoutBlock(64, 1)
        self.side3 = SideoutBlock(64, 1)
        self.side4 = SideoutBlock(64, 1)
        self.side5 = SideoutBlock(64, 1)

        self.rcg2 = RCG()
        self.rcg3 = RCG()
        self.rcg4 = RCG()

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=64, out_channels=64)
        self.decoder4 = DecoderBlock(in_channels=192, out_channels=64)
        self.decoder3 = DecoderBlock(in_channels=192, out_channels=64)
        self.decoder2 = DecoderBlock(in_channels=192, out_channels=64)

        self.final = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                                   nn.Dropout2d(0.1),
                                   nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, x):
        e1 = self.encoder1_conv(x)
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        e3 = self.down3(e3)  # 64
        e4 = self.down4(e4)  # 64
        e5 = self.down5(e5)  # 64

        lg5 = self.lg5(e5)
        lg4 = self.lg4(e4, lg5[1:])
        lg3 = self.lg3(e3, lg4[1:])
        lg2 = self.lg2(e2, lg3[1:])

        # decoder5
        d5 = self.decoder5(lg5[0])
        out5 = self.side5(d5)

        # e1_Contour
        c1 = self.cbam(e1)
        p_c = self.line_predict(c1)

        # decoder4
        r4 = self.rcg4(out5, c1, e4)
        d41 = torch.cat((d5, lg4[0], r4), dim=1)
        d4 = self.decoder4(d41)
        out4 = self.side4(d4)

        # decoder3
        r3 = self.rcg3(out4, c1, e3)
        d31 = torch.cat((d4, lg3[0], r3), dim=1)
        d3 = self.decoder3(d31)
        out3 = self.side3(d3)

        # decoder2
        r2 = self.rcg2(out3, c1, e2)
        d21 = torch.cat((d3, lg2[0], r2), dim=1)
        d2 = self.decoder2(d21)
        out2 = self.side2(d2)

        # final_output
        p = self.hpp(d2, d3, d4)
        out1 = self.final(p)
        out1 = F.interpolate(out1, size=x.size()[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), torch.sigmoid(out4), torch.sigmoid(out5),\
          torch.sigmoid(p_c)

if __name__ == '__main__':
    from thop import profile, clever_format
    model = UM-Net(num_classes=1)
    input = torch.randn([1, 3, 352, 352])
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.2f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
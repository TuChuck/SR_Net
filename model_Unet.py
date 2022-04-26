import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks.newcrf_layers import upsample

class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, kernel_init = "kaiming_normal"):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(output_dim)
        
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(output_dim)
        
        self.act = nn.ReLU()

        if kernel_init:
            if kernel_init == "kaiming_normal":
                nn.init.kaiming_normal_(self.conv1.weight)
                nn.init.kaiming_normal_(self.conv2.weight)
            else:
                raise Exception("add init method")

    def forward(self,x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.act(x0)

        x1 = self.conv2(x0)
        x1 = self.bn2(x1)
        x1 = self.act(x1)

        return x1

class Encoder_block(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, AvgPool_padding = 0):
        super(Encoder_block,self).__init__()

        self.AvgPool = nn.AvgPool2d(kernel_size=2, padding=AvgPool_padding)
        self.Do = nn.Dropout(dropout_rate)
        self.conv_block = ConvBlock(input_dim, output_dim, kernel_size=3, padding=1)

    def forward(self,x):
        x0 = self.AvgPool(x)
        x0 = self.Do(x0)
        x0 = self.conv_block(x0)

        return x0

class Decoder_block(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, Upsampling_padding = (1,1)):
        super(Decoder_block,self).__init__()

        self.conv_block = ConvBlock(input_dim, output_dim, kernel_size=3, padding=1)
        self.Do = nn.Dropout(dropout_rate)
        self.up_block = Upsampling(output_dim, output_dim // 2, stride=(2,2),output_padding=Upsampling_padding)

    def forward(self, x):
        x0 = self.conv_block(x)
        x0 = self.Do(x0)
        x0 = self.up_block(x0)

        return x0

class Upsampling(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, output_padding = (1,0)
                 ,kernel_init="kaiming_normal"):
        super(Upsampling, self).__init__()
        
        self.conv2d_T = nn.ConvTranspose2d(input_dim, output_dim, 
                                           kernel_size=kernel_size, 
                                           stride=stride, 
                                           padding=padding,
                                           output_padding=output_padding)
        self.bn = nn.BatchNorm2d(output_dim)
        self.act = nn.ReLU()

        if kernel_init:
            nn.init.kaiming_normal_(self.conv2d_T.weight)

    def forward(self, x):
        x0 = self.conv2d_T(x)
        x0 = self.bn(x0)
        x0 = self.act(x0)

        return x0

class Unet(nn.Module):
    def __init__(self, imput_dim, filters, dropout_rate, upsampling_factor):
        super(Unet, self).__init__()
        self.upsampling_factor = upsampling_factor
        
        self.Up_block = []
        for _ in range(upsampling_factor):
            if _ == 0:
                self.Up_block.append(Upsampling(imput_dim, filters , kernel_size=3, stride=(2,1)))
            else:
                self.Up_block.append(Upsampling(filters, filters , kernel_size=3, stride=(2,1)))
        self.Up_block = nn.ModuleList(self.Up_block)
            
        self.conv_block1 = ConvBlock(filters, filters, kernel_size=3, padding=1)

        self.encode_block1 = Encoder_block(filters, filters * 2, dropout_rate=dropout_rate)
        self.encode_block2 = Encoder_block(filters * 2, filters * 4, dropout_rate=dropout_rate)
        self.encode_block3 = Encoder_block(filters * 4, filters * 8, dropout_rate=dropout_rate)
                                           #,AvgPool_padding=(1,0))

        self.decoder_block4 = Decoder_block(filters * 8, filters * 16,dropout_rate=dropout_rate
                                           ,Upsampling_padding=(0,1))
        self.decoder_block3 = Decoder_block(filters * 16, filters * 8,dropout_rate=dropout_rate)
        self.decoder_block2 = Decoder_block(filters * 8, filters * 4,dropout_rate=dropout_rate)
        self.decoder_block1 = Decoder_block(filters * 4, filters * 2,dropout_rate=dropout_rate)

        self.conv_block2 = ConvBlock(filters * 2, filters, kernel_size=3, padding=1)
        self.conv1d = nn.Conv2d(filters,imput_dim, kernel_size=1)

        self.AvgPool = nn.AvgPool2d(kernel_size=2, padding=(1,0))
        self.Do = nn.Dropout(dropout_rate)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        for _ in range(self.upsampling_factor):
            x0 = self.Up_block[_](x0)
        
        x1 = self.conv_block1(x0)
        
        x2 = self.encode_block1(x1)
        x3 = self.encode_block2(x2)
        x4 = self.encode_block3(x3)

        y4 = self.Do(self.AvgPool(x4))
        y4 = self.decoder_block4(y4)

        y3 = self.decoder_block3(torch.cat([x4,y4], dim=1))
        y2 = self.decoder_block2(torch.cat([x3,y3], dim=1))
        y1 = self.decoder_block1(torch.cat([x2,y2], dim=1))

        y0 = self.conv_block2(torch.cat([x1,y1],dim=1))

        output = self.conv1d(y0)
        output = self.act(output)

        return output

def upsample(x, scale_factor=4, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    _, _ ,H ,W = x.size()
    size = [H*scale_factor,W]
    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

if __name__ == '__main__':
    x = torch.rand((16,1,10,256))
    upsampling_factor = 2

    model = Unet(filters=64, 
                 dropout_rate=0.25,
                 upsampling_factor = 2)

    y = model(x)

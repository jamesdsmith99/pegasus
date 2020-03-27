import torch
from torch import nn

class T(nn.Module):
    def __init__(self, f=64):
        super(T, self).__init__()
        self.discriminate = nn.Sequential(
            self.half_conv_block(  1,   f),
            self.half_conv_block(  f, 2*f),
            self.half_conv_block(2*f, 4*f),
            self.half_conv_block(4*f, 8*f),
            self.half_conv_block(8*f,   f),
            self.conv_block(f, 1),
            # self.half_conv_block1d(4*f, 2*f),
            # self.half_conv_block1d(2*f, 1),
            nn.AvgPool2d(kernel_size=(5,1)),
            nn.Sigmoid()
        )
        
    def half_conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(out_channels), # TURN OFF FOR SPEC NORM
            nn.LeakyReLU(0.01, inplace=True)
        )
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)),
            # nn.BatchNorm2d(out_channels), # TURN OFF FOR SPEC NORM
            nn.LeakyReLU(0.01, inplace=True)
        )

    def half_conv_block1d(self, in_channels, out_channels):
          return nn.Sequential(
              nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 2, 2, 0, bias=False)),
              # nn.BatchNorm2d(out_channels), # TURN OFF FOR SPEC NORM
              nn.LeakyReLU(0.01, inplace=True)
          )

    def forward(self, x):
        return self.discriminate(x)
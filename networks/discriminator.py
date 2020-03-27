from torch import nn
from .layers import HalfConvSpec, Swish


class Discriminator(nn.Module):

    def __init__(self, f=64):
        super(Discriminator, self).__init__()

        self.f = f

        self.conv1 = HalfConvSpec(  3,   f, Swish())  # 32x32 -> 16x16
        self.conv2 = HalfConvSpec(  f, 2*f, Swish())  # 16x16 -> 8x8
        self.conv3 = HalfConvSpec(2*f, 4*f, Swish())  # 8x8 -> 4x4
        self.conv4 = HalfConvSpec(4*f, 8*f, Swish())  # 4x4 -> 2x2
        self.conv5 = HalfConvSpec(8*f,   1, nn.Sigmoid()) # 2x2 -> 1x1

    def feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x.view(-1, 2*self.f * 16 * 16)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

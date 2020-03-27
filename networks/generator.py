from torch import nn
from .layers import DoubleConvTranspose, Swish


class Generator(nn.Module):

    def __init__(self, f=64):
        super(Generator, self).__init__()

        self.deconv1 = DoubleConvTranspose(100, 8*f, Swish())  # 1x1 -> 2x2
        self.deconv2 = DoubleConvTranspose(8*f, 4*f, Swish())  # 2x2 -> 4x4
        self.deconv3 = DoubleConvTranspose(4*f, 2*f, Swish())  # 4x4 -> 8x8
        self.deconv4 = DoubleConvTranspose(2*f,   f, Swish())  # 8x8 -> 16x16
        self.deconv5 = DoubleConvTranspose(  f,   3, nn.Sigmoid()) # 16x16 -> 32x32

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return self.deconv5(x)

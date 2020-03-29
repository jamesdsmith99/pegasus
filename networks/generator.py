from torch import nn
from .layers import DoubleConvTranspose, ProjectConv, Swish


class Generator(nn.Module):

    def __init__(self, f=64):
        super(Generator, self).__init__()

        self.deconv1 = ProjectConv(100, 8*f, activation=nn.ReLU())                            # 1x1 -> 4x4
        self.deconv2 = DoubleConvTranspose(8*f, 4*f, batch_norm=True,  activation=nn.ReLU())  # 4x4   -> 8x8
        self.deconv3 = DoubleConvTranspose(4*f, 2*f, batch_norm=True,  activation=nn.ReLU())  # 8x8   -> 16x16
        self.deconv4 = DoubleConvTranspose(2*f,   f, batch_norm=True,  activation=nn.ReLU())  # 16x16 -> 32x32
        self.deconv5 = DoubleConvTranspose(  f,   3, batch_norm=False, activation=nn.Tanh())  # 32x32 -> 64x64

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return self.deconv5(x)

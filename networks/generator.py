from torch import nn
from .layers import DoubleConvTranspose, DoubleConvTransposeNoPad, Swish


class Generator(nn.Module):

    def __init__(self, f=64):
        super(Generator, self).__init__()

        self.deconv1 = DoubleConvTransposeNoPad(100, 8*f, nn.ReLU())  # 1x1 -> 2x2
        self.deconv2 = DoubleConvTranspose(8*f, 4*f, nn.ReLU())  # 2x2 -> 4x4
        self.deconv3 = DoubleConvTranspose(4*f, 2*f, nn.ReLU())  # 4x4 -> 8x8
        self.deconv4 = DoubleConvTranspose(2*f,   f, nn.ReLU())  # 8x8 -> 16x16
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(f, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        ) # 16x16 -> 32x32

    def forward(self, x):
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        return self.deconv5(x)

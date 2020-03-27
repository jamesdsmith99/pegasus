from torch import nn
from torch.nn.utils import spectral_norm
from .functions import Swish as SwishFn

class Conv(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation       

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)


class HalfConv(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super(HalfConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation       

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)

class HalfConvSpec(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super(HalfConvSpec, self).__init__()

        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        )
        self.activation = activation       

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x)


class DoubleConvTranspose(nn.Module):

    def __init__(self, in_channels, out_channels, activation):
        super(DoubleConvTranspose, self).__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x) 

class LinearBatchNorm(nn.Module):

    def __init__(self, in_features, out_features, activation):
        super(LinearBatchNorm, self).__init__()

        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        return self.activation(x) 
        


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return SwishFn.apply(x)

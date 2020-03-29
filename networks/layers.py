from torch import nn
from torch.nn.utils import spectral_norm as spectral_norm_fn
from .functions import Swish as SwishFn

class ConvBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            spec_norm=False,
            batch_norm=False,
            activation=None
        ):

        super(ConvBlock, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        if spec_norm:
            conv = spectral_norm_fn(conv)

        layers = [conv]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation is not None:
            layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class ConvTransBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            batch_norm=False,
            activation=None
        ):

        super(ConvTransBlock, self).__init__()

        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        layers = [conv]

        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation is not None:
            layers.append(activation)

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class Conv(ConvBlock):

    def __init__(self, in_channels, out_channels, activation):
        super(Conv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            spec_norm=False,
            batch_norm=True,
            activation=activation)



class HalfConv(ConvBlock):

    def __init__(self, in_channels, out_channels, spec_norm, batch_norm, activation):
        super(HalfConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            spec_norm=spec_norm,
            batch_norm=batch_norm,
            activation=activation)

class FlattenConv(ConvBlock):

    def __init__(self, in_channels, out_channels, spec_norm, activation):
        super(FlattenConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            spec_norm=spec_norm,
            batch_norm=False,
            activation=activation)


class DoubleConvTranspose(ConvTransBlock):

    def __init__(self, in_channels, out_channels, batch_norm, activation):
        super(DoubleConvTranspose, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            batch_norm=batch_norm,
            activation=activation)

class ProjectConv(ConvTransBlock):

    def __init__(self, in_channels, out_channels, activation):
        super(ProjectConv, self).__init__(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            batch_norm=True,
            activation=activation)

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

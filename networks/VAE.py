import torch
from torch import nn
from .layers import Conv, DoubleConvTranspose, HalfConv, LinearBatchNorm, Swish

class Encoder(nn.Module):

    def __init__(self, image_channels, hidden_size, intermediate_size=128, f=16):
        super(Encoder, self).__init__()

        self.conv1 = Conv(image_channels, f, Swish())                                            # 32x32 -> 32x32
        self.conv2 = HalfConv(  f, 2*f, spec_norm=False, batch_norm=True, activation=Swish())    # 32x32 -> 16x16
        self.conv3 = HalfConv(2*f, 4*f, spec_norm=False, batch_norm=True, activation=Swish())    # 16x16 -> 8x8
        self.conv4 = HalfConv(4*f, 8*f, spec_norm=False, batch_norm=True, activation=Swish())    #   8x8 -> 4x4
        self.conv5 = HalfConv(8*f,16*f, spec_norm=False, batch_norm=True, activation=Swish())    #   4x4 -> 2x2
        
        self.fc = LinearBatchNorm(2*2*16*f, intermediate_size, Swish())

        self.μ        = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.log_var  = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        μ = self.μ(x)
        log_var = self.log_var(x)

        return μ, log_var

class Decoder(nn.Module):

    def __init__(self, image_channels, hidden_size, intermediate_size=128, f=16):
        super(Decoder, self).__init__()

        self.f = f

        self.fc1 = LinearBatchNorm(hidden_size, intermediate_size, Swish())
        self.fc2 = LinearBatchNorm(intermediate_size, 2*2*16*f, Swish())

        self.deconv1 = DoubleConvTranspose(16*f, 8*f, batch_norm=True, activation=Swish())  # 2x2   -> 4x4
        self.deconv2 = DoubleConvTranspose( 8*f, 4*f, batch_norm=True, activation=Swish())  # 4x4   -> 8x8
        self.deconv3 = DoubleConvTranspose( 4*f, 2*f, batch_norm=True, activation=Swish())  # 8x8   -> 16x16
        self.deconv4 = DoubleConvTranspose( 2*f,   f, batch_norm=True, activation=Swish())  # 16x16 -> 32x32

        self.conv = Conv(f, image_channels, nn.Sigmoid())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 16*self.f, 2, 2)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        return self.conv(x)

class VAE(nn.Module):

    def __init__(self, image_channels, hidden_size, intermediate_size=128, f=16):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(image_channels, hidden_size, intermediate_size, f)
        self.decoder = Decoder(image_channels, hidden_size, intermediate_size, f)
    
    def forward(self, x):
        μ, log_var = self.encoder(x)
        z = self.reparameterize(μ, log_var)

        reconstruction = self.decoder(z)

        return reconstruction, μ, log_var
    
    def encode(self, x):
        return self.encoder(x)
    
    def reparameterize(self, μ, log_var):
        if self.training:
            σ = log_var.mul(0.5).exp_()
            eps = torch.randn_like(σ)
            return eps.mul(σ).add_(μ)
        else:
            return μ

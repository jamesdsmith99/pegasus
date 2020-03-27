import torch
from torch.nn import functional as F

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def l1_loss(x, y):
    return (x-y).abs().mean()

def l2_loss(x, y):
    return ((x-y)**2).mean()

'''
    given the discriminators output for the real and fake data calculate the discriminator loss

    log(D(x)) + log(1 - D(G(z)))

    real data has label 1
    fake data has label 0
'''
def discriminator_loss(real, fake):
    loss_real = F.binary_cross_entropy(real, torch.ones_like(real).to(DEVICE))
    loss_fake = F.binary_cross_entropy(fake, torch.zeros_like(fake).to(DEVICE))
    return (loss_real.mean() + loss_fake.mean()) / 2.0

'''
    given the discriminators output for the generated data calculate the generator loss

    log(D(G(z)))

    real data has label 1 and we want to fool the discriminator into thinking our generated data is real
'''

def generator_loss(gen):
    return F.binary_cross_entropy(gen, torch.ones_like(gen).to(DEVICE)).mean()

'''
    calculate the KL divergence between the normal distribution paramaterised by μ and log_var
    and the normal distribution centered at 0 with unit varience

    Used as part of the VAE loss
'''
def KLD_normal(μ, log_var):
    varience = log_var.exp()
    return -0.5 * torch.sum(1 + log_var - μ.pow(2) - varience)
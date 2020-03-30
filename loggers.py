import numpy as np
import torchvision
import wandb

def mean_GAN_loss_wandb(G_loss_arr, D_loss_arr, epoch):
    G_loss = np.mean(G_loss_arr)
    D_loss = np.mean(D_loss_arr)

    wandb.log({
        'Generator Loss': G_loss,
        'Discriminator Loss': D_loss
    }, step=epoch)

def mean_joint_GAN_loss_wandb(G_loss_arr, Da_loss_arr, Db_loss_arr, epoch):
    G_loss = np.mean(G_loss_arr)
    Da_loss = np.mean(Da_loss_arr)
    Db_loss = np.mean(Db_loss_arr)

    wandb.log({
        'Generator Loss': G_loss,
        'Discriminator A Loss': Da_loss,
        'Discriminator B Loss': Db_loss,
    }, step=epoch)


def VAE_loss_wandb(l1_loss_arr, kl_loss_arr, loss_arr, epoch):
    l1_loss = np.mean(l1_loss_arr)
    kl_loss = np.mean(kl_loss_arr)
    loss = np.mean(loss_arr)

    wandb.log({
        'L1 Loss': l1_loss,
        'KL Loss': kl_loss,
        'Loss': loss
    }, step=epoch)

def batch_images_wandb(batch, epoch):
    wandb.log({
        'Samples': wandb.Image(
            to_image_grid(batch)
        )
    }, step=epoch)

def batch_images_wandb_dcgan_norm(batch, epoch):
    wandb.log({
        'Samples': wandb.Image(
            to_image_grid((batch+1)/2)
        )
    }, step=epoch)

def before_after_images_wandb(x, xHat, epoch):
    wandb.log({
        'Input': wandb.Image(
            to_image_grid(x)
        ),
        'Output': wandb.Image(
            to_image_grid(xHat)
        )
    }, step=epoch)

def to_image_grid(x):
    grid = torchvision.utils.make_grid(x).cpu().data.permute(0,2,1).contiguous().permute(2,1,0)
    return (255 * grid.numpy()).astype(np.uint8)
import torch
import numpy as np
from .loss import l1_loss, KLD_normal
from .TorchIO import load_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class VAETrainer:

    def __init__(self, model, optimiser, data_loader, batch_size, num_iters, schedule, metric_logger=None, sample_logger=None):
        self.model = model
        self.optimiser = optimiser
        self.data_loader = data_loader
        self.num_iters = num_iters
        self.batch_size = batch_size

        self.schedule = schedule

        self.log_metrics = metric_logger
        self.log_samples = sample_logger

        self.l1_loss_arr = np.zeros(0)
        self.kl_loss_arr = np.zeros(0)
        self.loss_arr = np.zeros(0)


    def load_state(self, state_path):
        return load_model(state_path, self.model, self.optimiser)
    
    def train(self, epochs, base=0):
        epoch = base

        while epoch < base + epochs:
 
            for i in range(self.num_iters):
                x = next(self.data_loader).to(DEVICE)

                self.optimiser.zero_grad()

                xHat, μ, log_var = self.model(x)

                β = self.schedule(epoch) 

                loss_l1 = l1_loss(x, xHat)
                loss_kl = KLD_normal(μ, log_var)
                loss = loss_l1 + β*loss_kl

                loss.backward()
                self.optimiser.step()

                self.l1_loss_arr = np.append(self.l1_loss_arr, loss_l1.item())
                self.kl_loss_arr = np.append(self.kl_loss_arr, loss_kl.item())
                self.loss_arr = np.append(self.loss_arr, loss.item())

            # record metrics
            if self.log_metrics:
                self.log_metrics(self.l1_loss_arr, self.kl_loss_arr, self.loss_arr, epoch)
            self.l1_loss_arr = np.zeros(0)
            self.kl_loss_arr = np.zeros(0)
            self.loss_arr = np.zeros(0)

            # record samples
            if epoch % 10 == 0 and self.log_samples:
                self.log_samples(x, xHat, epoch)

            epoch += 1

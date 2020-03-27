import torch
import numpy as np
from .loss import discriminator_loss, generator_loss
from .TorchIO import load_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class PingPongTrainer:


    '''
        schedule defines what data sets to train on for how long ordered list of ScheduleObjects
        which specify how long to train on a dataset. the trainer will train the discriminator on these datsets
        cycling around the schedule array for the time specified
    '''
    def __init__(self, G, D, optimiser_G, optimiser_D, schedule, batch_size, num_iters, overtrain_D=1, metric_logger=None, sample_logger=None):
        self.G = G.to(DEVICE)
        self.D = D.to(DEVICE)
        self.optimiser_G = optimiser_G
        self.optimiser_D = optimiser_D
        self.num_iters = num_iters
        self.batch_size = batch_size

        self.schedule = schedule

        self.overtrain_D = overtrain_D

        self.log_metrics = metric_logger
        self.log_samples = sample_logger

        self.G_loss_arr = np.zeros(0)
        self.D_loss_arr = np.zeros(0)


    def load_state(self, state_path_G, state_path_D):
         load_model(state_path_G, self.G, self.optimiser_G)
         load_model(state_path_D, self.D, self.optimiser_D)
        
    def _train_D(self, data_loader):
        real = next(data_loader).to(DEVICE)

        z = torch.randn(self.batch_size, 100, 1, 1).to(DEVICE)
        fake = self.G(z)

        self.optimiser_D.zero_grad()
        loss = discriminator_loss(self.D(real).mean(), self.D(fake).mean())
                
        loss.backward()
        self.optimiser_D.step()

        self.G_loss_arr = np.append(self.G_loss_arr, loss.item())

    def _train_G(self):
        z = torch.randn(self.batch_size, 100, 1, 1).to(DEVICE)

        gen = self.G(z)
        self.gen = gen # save to a field to be accessed for logging

        self.optimiser_G.zero_grad()
        loss = generator_loss(self.D(gen).mean())

        loss.backward()
        self.optimiser_G.step()

        self.D_loss_arr = np.append(self.D_loss_arr, loss.item())
    
    def train(self, epochs, base=0):
        epoch = base

        schedule_index = 0
        current_schedule = self.schedule[schedule_index]
        epochs_spent_on_schedule = 0

        self.G.train()
        self.D.train()

        while epoch < base + epochs:
            
            for i in range(self.num_iters):
                # overtrain discriminator
                for k in range(self.overtrain_D):
                    self._train_D(current_schedule.data_loader)

                # train generator
                self._train_G()

            # record metrics
            if self.log_metrics:
                self.log_metrics(self.G_loss_arr, self.D_loss_arr, epoch)
            self.G_loss_arr = np.zeros(0)
            self.D_loss_arr = np.zeros(0)

            # record samples
            if epoch % 10 == 0 and self.log_samples:
                self.log_samples(self.gen, epoch)

            epoch += 1
            epochs_spent_on_schedule += 1

            if epochs_spent_on_schedule == current_schedule.epochs:
                epochs_spent_on_schedule = 0
                schedule_index = (schedule_index + 1) % len(self.schedule)
                current_schedule = self.schedule[schedule_index]

import torch
import numpy as np
from .loss import discriminator_loss, generator_loss
from .TorchIO import load_model

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class JointTrainer:


    '''
        schedule defines what data sets to train on for how long ordered list of ScheduleObjects
        which specify how long to train on a dataset. the trainer will train the discriminator on these datsets
        cycling around the schedule array for the time specified
    '''
    def __init__(self, G, Da, Db, optimiser_G, optimiser_Da, optimiser_Db, data_loader_a, data_loader_b, a, b, batch_size, num_iters, overtrain_D=1, metric_logger=None, sample_logger=None):
        self.G = G.to(DEVICE)
        self.Da = Da.to(DEVICE)
        self.Db = Db.to(DEVICE)
        self.optimiser_G = optimiser_G
        self.optimiser_Da = optimiser_Da
        self.optimiser_Db = optimiser_Db
        self.num_iters = num_iters
        self.batch_size = batch_size

        self.data_loader_a = data_loader_a
        self.data_loader_b = data_loader_b

        self.a = a
        self.b = b

        self.overtrain_D = overtrain_D

        self.log_metrics = metric_logger
        self.log_samples = sample_logger

        self.G_loss_arr = np.zeros(0)
        self.Da_loss_arr = np.zeros(0)
        self.Db_loss_arr = np.zeros(0)


    def load_state(self, state_path_G, state_path_Da, state_path_Db):
         load_model(state_path_G, self.G, self.optimiser_G)
         load_model(state_path_Da, self.Da, self.optimiser_Da)
         load_model(state_path_Db, self.Db, self.optimiser_Db)
        
    def _train_D(self, real_a, real_b):
        real_a = real_a.to(DEVICE)
        real_b = real_b.to(DEVICE)

        z = torch.randn(self.batch_size, 100, 1, 1).to(DEVICE)
        with torch.no_grad():
            fake = self.G(z)

        self.optimiser_Da.zero_grad()
        self.optimiser_Db.zero_grad()

        # Discriminator A
        Da_real = self.Da(real_a).view(-1)
        Da_fake = self.Da(fake).view(-1)
        loss_a = sum(discriminator_loss(Da_real, Da_fake))

        loss_a.backward()
        self.optimiser_Da.step()

        # Discriminator B
        Db_real = self.Db(real_b).view(-1)
        Db_fake = self.Db(fake).view(-1)
        loss_b = sum(discriminator_loss(Db_real, Db_fake))

        loss_b.backward()
        self.optimiser_Db.step()

        # append metrics to array
        self.Da_loss_arr = np.append(self.Da_loss_arr, loss_a.item())
        self.Db_loss_arr = np.append(self.Db_loss_arr, loss_b.item())

    def _train_G(self):
        z = torch.randn(self.batch_size, 100, 1, 1).to(DEVICE)

        gen = self.G(z)
        self.gen = gen # save to a field to be accessed for logging

        self.optimiser_G.zero_grad()

        discriminator_feedback_a = self.Da(gen).view(-1)
        discriminator_feedback_b = self.Db(gen).view(-1)
        loss = self.a * generator_loss(discriminator_feedback_a) + self.b * generator_loss(discriminator_feedback_b)

        loss.backward()
        self.optimiser_G.step()

        self.G_loss_arr = np.append(self.G_loss_arr, loss.item())
    
    def train(self, epochs, base=0):
        epoch = base

        self.G.train()
        self.Da.train()
        self.Db.train()

        while epoch < base + epochs:
            
            for step in range(self.num_iters):
                # overtrain discriminator
                for k in range(self.overtrain_D):
                    data_a = next(self.data_loader_a)
                    data_b = next(self.data_loader_b)
                    self._train_D(data_a, data_b)

                # train generator
                self._train_G()

            # record metrics
            if self.log_metrics:
                self.log_metrics(self.G_loss_arr, self.Da_loss_arr, self.Db_loss_arr, epoch)
            self.G_loss_arr = np.zeros(0)
            self.D_loss_arr = np.zeros(0)

            # record samples
            if epoch % 10 == 0 and self.log_samples:
                self.log_samples(self.gen, epoch)

            epoch += 1

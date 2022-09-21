# Pegasus Image Generation

## Problem Description

A pegasus is a mythical creature that is a horse with wings. Given images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), produce a model that can generate images of Pegasuses.

It can be interpreted that a pegasus is a creature that is a mixture of horse and bird. Leading to a pegasus having a high joint probability of being a horse and a bird ![probability H and B](https://latex.codecogs.com/gif.latex?P(H\cap%20B)).

## JointGAN

### GAN
The GAN allows a generator network to be trained to sample realistic samples from the distribution that training data is sampled from. GANs can learn to do this sampling without having to estimate the density function of the distribution. This is done by training two networks playing an adversarial mini-max game:

![GAN loss function](https://latex.codecogs.com/gif.latex?\min_G%20\max_D%20V(D%20,G)%3D\mathbb{E}_{x\sim%20p_{data}(x)}[\log%20D(x)]+\mathbb{E}_{z\sim%20p_z(z)}[\log(1-D(G(z)))])

Where the discriminator network ![D](https://latex.codecogs.com/gif.latex?D). learns to estimate ![p data](https://latex.codecogs.com/gif.latex?p_{data}), while the generator network learns produce samples from the data distribution given some random noise vector ![z drawn from some distribution](https://latex.codecogs.com/gif.latex?z\sim%20p_{z}(z)).


### Generating samples from a joint distribution

JointGAN allows for a generator network to be adversarially trained to sample images from some joint distribution ![Joint probability of all classes in C](https://latex.codecogs.com/gif.latex?P\left(\bigcap_{c_i\in%20C}c_i\right)) using samples from the constituent marginal distributions ![Marginal probability of class c](https://latex.codecogs.com/gif.latex?P\left(c_i\right)).

This is achieved by using ![Cardinality of the set of classes](https://latex.codecogs.com/gif.latex?|C|). discriminator networks, where the discriminator ![Discriminator i](https://latex.codecogs.com/gif.latex?D_i) predicts ![probability of an image belonging to class c_i](https://latex.codecogs.com/gif.latex?P\left(c_i\right)).

We can define the class loss as:

![class loss function](https://latex.codecogs.com/gif.latex?L_c%3D\mathbb{E}_{x\sim%20p_{c}(x)}[\log%20D(x)]+\mathbb{E}_{z\sim%20p_z(z)}[\log(1-D_c(G(z)))])

Then JointGAN can be trained on the objective:

![GAN loss function](https://latex.codecogs.com/gif.latex?\min_G%20\max_{D_{c_1},...,D_{c_{|C|}}}%20\sum_{c_i\in%20C}L_{c_i})

This loss allows for the generator to sample from the joint distribution assuming that the marginal probabilities are independent.

## Using JointGAN to generate Pegasus images

The dataset contains images of both birds and horses. We can use JointGAN to train the generator to sample from ![horse and bird](https://latex.codecogs.com/gif.latex?P(H\cap%20B)).

The horse discriminator will be trained on generator outputs and horse images from the dataset. The bird discriminator wil be trained on the generator outputs and bird images from the dataset.

![Architecture diagram](./img/architecture.png)

### Training stability

Care must be taken to ensure that the training does not collapse. It is common for the discriminators to become too powerful too quickly as the generator has to generate data that can fool multiple discriminators at once.

To ensure the discriminators do not learn too quickly and collapse the training process spectral normalization is applied. This ensures that the discriminators loss is K-Lipschitz - the discriminators gradients are bound by a constant K.

## Results
![Sample images](./img/sample_images.png)


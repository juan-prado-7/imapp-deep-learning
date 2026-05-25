from csv import writer

import torchvision
# For image transforms
from torchvision import transforms
# For DATA SET
import torchvision.datasets as datasets
# For Pytorch methods
import torch
import torch.nn as nn
# For Optimizer
import torch.optim as optim
# FOR DATA LOADER
from torch.utils.data import DataLoader
# FOR TENSOR BOARD VISUALIZATION
from torch.utils.tensorboard import SummaryWriter # to print to tensorboard

import typing as tp
import time
import sys

# Load dataset
def load_mnist_data(PATH: str, batch_size: int, download: bool = False) -> DataLoader:
    myTransforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST(root=PATH, transform=myTransforms, download=download)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader

# Train a GAN model
def train_gan(
            data_loader: DataLoader,
            discriminator: nn.Module,
            generator: nn.Module,
            opt_discriminator: torch.optim.Optimizer,
            opt_generator: torch.optim.Optimizer,
            criterion: nn.modules.loss,
            num_epochs: int,
            log_step: int,
            device: str = 'cpu'
        )-> tp.Tuple[list, list]:
    

    generator.to(device)
    discriminator.to(device)

    latent_dim = generator.latent_dim
    image_dim = generator.image_dim

    generator.train()
    discriminator.train()

    disc_losses = []
    gen_losses = []

    step = 0
    for epoch in range(num_epochs):

        start_time = time.time()  # Start time for the epoch
        mean_disc_loss = 0
        mean_gen_loss = 0

        # loop over batches
        for batch_idx, (real, _) in enumerate(data_loader):
            # First we train the discriminator on real images vs. generated images

            # Get the real images and flatten them
            # for simplicity, we flatten the image to a vector and to use simple MLP networks
            # 28 * 28 * 1 flattens to 784
            real = real.view(-1, image_dim).to(device)
            batch_size = real.shape[0]

            # Step 1) generate fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake = generator(noise)

            # Step 2) Train Discriminator:
            # - predict the discriminator output for real images
            # - real images are labeled as 1
            # - calculate the loss for real images
            pred_real = discriminator(real)
            loss_real = criterion(pred_real, torch.ones_like(pred_real))

            # - predict the discriminator output for fake images
            # -fake images are labeled as 0
            # -calculate the loss for fake images
            pred_fake = discriminator(fake.detach())
            loss_fake = criterion(pred_fake, torch.zeros_like(pred_fake))

            # -average the loss for real and fake images
            loss_discriminator = (loss_real + loss_fake) / 2

            # - now upadate the weights of the discriminator by backpropagating the loss through the discriminator
            # the generator is not updated in this step
            # HINT: call the `backward` method of the discriminator with the argument `retain_graph=True` to keep the computational graph
            # this is necessary because we will use the same discriminator to train the generator
            opt_discriminator.zero_grad()
            loss_discriminator.backward(retain_graph=True)
            opt_discriminator.step()

            # Train Generator:
            # Now train the generator by generating fake images and passing them through the discriminator
            # You can do a little trick and modify the original objective function of
            # "minimizing the probability of the discriminator predicting the fake images as fake"
            # to "maximizing the probability of the discriminator predicting the fake images as real"
            # this leads to a faster training of the generator when it does not represent the real data well
            # this is a common trick in GANs
            # for moer information see section 17.1.2 of the book Deep Learning by Bishop and Bishop
            # Todo:
            # - pass the fake images through the discriminator
            # - calculate the loss (by passing the output of the discriminator through the criterion with labels set to 1 (real images
            # - update the weights of the generator
            pred_fake = discriminator(fake)
            loss_generator = criterion(pred_fake, torch.ones_like(pred_fake))

            opt_generator.zero_grad()
            loss_generator.backward()
            opt_generator.step()

            mean_disc_loss += loss_discriminator.item()
            mean_gen_loss += loss_generator.item()

            # print the progress
            if (batch_idx + 1) % 50 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(data_loader)}] -> Discriminator Loss: {loss_discriminator:.4f}, Generator Loss: {loss_generator:.4f}")
                sys.stdout.flush()

            # Log the losses and example images to tensorboard
            if batch_idx % log_step == 0:
                with torch.no_grad():
                    if step == 0:
                        fixed_noise = torch.randn(batch_size, latent_dim).to(device)

                    # Generate noise via Generator, we always use the same noise to see the progression
                    fake = generator(fixed_noise).reshape(-1, 1, 28, 28)
                    # Get real data
                    data = real.reshape(-1, 1, 28, 28)
                    # make grid of pictures and add to tensorboard
                    imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                    imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                    # Add the images and losses to tensorboard
                    # HINT: use the SummaryWriter to add the images and scalars to tensorboard
                    # HINT: use the `add_image` method to add the images to tensorboard
                    # HINT: use the `add_scalar` method to add the losses to tensorboard
                    writer = SummaryWriter(f"logs/mnist/step_{step}")
                    writer.add_image("MNIST Fake Images", imgGridFake, global_step=step)
                    writer.add_image("MNIST Real Images", imgGridReal, global_step=step)
                    writer.add_scalar("Loss Discriminator", loss_discriminator, global_step=step)
                    writer.add_scalar("Loss Generator", loss_generator, global_step=step)
                    writer.close()

                    # increment step
                    step += 1

        mean_disc_loss /= len(data_loader)
        mean_gen_loss /= len(data_loader)

        disc_losses.append(mean_disc_loss)
        gen_losses.append(mean_gen_loss)

        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}] -> Discriminator Loss: {mean_disc_loss:.4f}, Generator Loss: {mean_gen_loss:.4f}, Time: {epoch_time:.2f} seconds\n")
        sys.stdout.flush()

    return disc_losses, gen_losses
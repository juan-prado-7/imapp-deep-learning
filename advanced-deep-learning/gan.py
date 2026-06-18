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
import matplotlib.pyplot as plt

# Load dataset
def load_mnist_data(PATH: str, batch_size: int, download: bool = False) -> DataLoader:
    myTransforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
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
            PATH: str = None,
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

    if PATH is not None:
        writer = SummaryWriter(PATH)

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
            real = real.view(-1, image_dim[0]*image_dim[1]).to(device)
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
            opt_discriminator.zero_grad()
            loss_discriminator.backward()
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
            if PATH is not None and (batch_idx % log_step == 0):
                with torch.no_grad():
                    if step == 0:
                        fixed_noise = torch.randn(batch_size, latent_dim).to(device)

                    # Generate noise via Generator, we always use the same noise to see the progression
                    fake = generator(fixed_noise).reshape(-1, 1, image_dim[0], image_dim[1])
                    # Get real data
                    data = real.reshape(-1, 1, image_dim[0], image_dim[1])
                    # make grid of pictures and add to tensorboard
                    imgGridFake = torchvision.utils.make_grid(fake, normalize=True)
                    imgGridReal = torchvision.utils.make_grid(data, normalize=True)

                    # Add the images and losses to tensorboard
                    # HINT: use the SummaryWriter to add the images and scalars to tensorboard
                    # HINT: use the `add_image` method to add the images to tensorboard
                    # HINT: use the `add_scalar` method to add the losses to tensorboard
                    writer.add_image("MNIST Fake Images", imgGridFake, global_step=step)
                    writer.add_image("MNIST Real Images", imgGridReal, global_step=step)
                    writer.add_scalar("Loss Discriminator", loss_discriminator, global_step=step)
                    writer.add_scalar("Loss Generator", loss_generator, global_step=step)

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

    if PATH is not None:
        writer.close()

    return disc_losses, gen_losses

# Plot the losses of the discriminator and generator
def plot_losses(disc_losses: list, gen_losses: list, PATH: str = None) -> None:
    
    plt.figure(figsize=(6, 4))

    plt.plot(disc_losses[1:], label="Discriminator Loss")
    plt.plot(gen_losses[1:], label="Generator Loss")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    if PATH is not None:
        plt.savefig(f"{PATH}/loss-curves.png", dpi=300)

    plt.show()

    return

# Save and load model functions
def save_gan(discriminator: nn.Module, generator: nn.Module, hyperparams: dict, PATH: str) -> None:
    info = {
        "hyperparams": hyperparams,
        "disc_state_dict": discriminator.state_dict(),
        "gen_state_dict": generator.state_dict(),}
    torch.save(info, PATH)

    return

def load_gan(discriminator_class: tp.Type[nn.Module], generator_class: tp.Type[nn.Module], PATH: str
             ) -> tp.Tuple[nn.Module, nn.Module, dict]:
    info = torch.load(PATH)

    discriminator = discriminator_class(**info["hyperparams"]["disc_hyperparams"])
    discriminator.load_state_dict(info["disc_state_dict"])

    generator = generator_class(**info["hyperparams"]["gen_hyperparams"])
    generator.load_state_dict(info["gen_state_dict"])

    return discriminator, generator, info["hyperparams"]



# Visualize the generated images by the generator
def visualize_generated_images(generator: nn.Module, device: str = 'cpu', shape: tuple = (4, 4), PATH: str = None) -> None:
    generator.to(device)
    latent_dim = generator.latent_dim
    image_dim = generator.image_dim
    generator.eval()

    with torch.no_grad():
        noise = torch.randn(shape[0]*shape[1], latent_dim).to(device)
        fake_images = generator(noise).reshape(-1, 1, image_dim[0], image_dim[1])

        img_grid = torchvision.utils.make_grid(fake_images, nrow=shape[0], normalize=True)

        plt.figure(figsize=(2*shape[0], 2*shape[1]))
        plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        if PATH is not None:
            plt.savefig(f"{PATH}/generated-images.png", dpi=300)

        plt.show()

    return 

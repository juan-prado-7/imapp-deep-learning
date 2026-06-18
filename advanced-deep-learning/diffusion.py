import torchvision
from torch.utils.tensorboard import SummaryWriter
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

import sys
import time
import typing as tp
import matplotlib.pyplot as plt

# Load dataset
def load_mnist_data(PATH: str, batch_size: int, download: bool = False) -> DataLoader:
    myTransforms = transforms.Compose([transforms.ToTensor()]) # Keep the pixel amplitudes in the range [0,1]

    train_dataset = datasets.MNIST(root=PATH, transform=myTransforms, download=download)
    val_dataset = datasets.MNIST(root=PATH, train=False, transform=myTransforms, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader


# Train the diffusion model (adapted from train_nn in helper.py)
def train_diffusion_model(
            train_loader: DataLoader, 
            val_loader: DataLoader,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            num_epochs: int,
            patience: int = None,
            PATH: str = None,
            device: str = 'cpu'
        )-> tp.Tuple[list, list]:

    print(f'Training diffusion model on {device}.')

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0


    if PATH is not None:
        writer = SummaryWriter(PATH)

    model.to(device)  # Move the model to the chosen device

    for epoch in range(num_epochs):  # loop through every epoch
        start_time = time.time()  # Start the timer for this epoch
        # Training
        model.train()  # The model should be in training mode to use batch normalization and dropout
        train_loss = 0

        # loop through every batch
        for step, (batch, _) in enumerate(train_loader):
            # move the batch to the same device as the model
            batch = batch.to(device)

            # Gradient step
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Print progress every 10th step, updating the same line
            if (step + 1) % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}] -> Loss: {loss.item():.4f}")
                sys.stdout.flush()

        # calculate loss per epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        # The model should be in eval mode to not use batch normalization and dropout
        model.eval()
        val_loss = 0

        # make sure the gradients are not changed in this step
        with torch.no_grad():
            for (batch, _) in val_loader:
                # move the batch to the same device as the model
                batch = batch.to(device)

                # calculate the loss
                loss = model(batch)
                val_loss += loss.item()

        # calulate loss per epoch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}] -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} seconds")
        sys.stdout.flush()

        if(patience is not None):
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                sys.stdout.write(f", Patience: [{patience_counter}/{patience}]")
                sys.stdout.flush()
                if patience_counter >= patience:
                    print("\nEarly stopping triggered.")
                    break
        
        sys.stdout.write("\n")

        if PATH is not None:
            with torch.no_grad():
                fake_image = model.sample(batch_size=6)
                imgGrid = torchvision.utils.make_grid(fake_image, normalize=True)

                # Add the images and losses to tensorboard
                writer.add_image("MNIST Fake Images", imgGrid, global_step=epoch)

    print("Training complete.")

    return train_losses, val_losses


# Visualize the generated images by the generator
def visualize_generated_images(model: nn.Module, device: str = 'cpu', shape: tuple = (4, 4), PATH: str = None) -> None:
    model.to(device)
    model.eval()

    with torch.no_grad():
        fake_images = model.sample(batch_size=shape[0]*shape[1])

        img_grid = torchvision.utils.make_grid(fake_images, nrow=shape[0], normalize=True)

        plt.figure(figsize=(2*shape[0], 2*shape[1]))
        plt.imshow(img_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')

        if PATH is not None:
            plt.savefig(f"{PATH}/generated-images.png", dpi=300)

        plt.show()

    return 
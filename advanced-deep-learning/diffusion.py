import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import typing as tp
import sys

# Generate a dataset of 1D data from a mixture of two Gaussians
def get_simple_data(train_samples: int, val_samples: int) -> tp.Tuple[torch.Tensor, torch.Tensor, tp.Callable]:
    data_distribution = torch.distributions.mixture_same_family.MixtureSameFamily(
        torch.distributions.Categorical(torch.tensor([1, 2])),
        torch.distributions.Normal(torch.tensor([-4., 4.]), torch.tensor([1., 1.]))
    )

    train_dataset = data_distribution.sample(torch.Size([train_samples]))  # create training data set
    val_dataset = data_distribution.sample(torch.Size([val_samples])) # create validation data set

    def pdf(x: np.ndarray) -> np.ndarray:
        return np.exp(data_distribution.log_prob(torch.tensor(x)).cpu().numpy())

    return train_dataset, val_dataset, pdf


# Train the diffusion model (adapted from train_nn in helper.py)
def train_diffusion_model(
            train_loader: DataLoader, 
            val_loader: DataLoader,
            model: nn.Module,
            time_steps: int,
            beta: torch.Tensor,
            loss_fn: tp.Callable[[torch.Tensor, nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
            optimizer: torch.optim.Optimizer,
            num_epochs: int,
            patience: int = None,
            device: str = 'cpu'
        )-> tp.Tuple[list, list]:

    print(f'Training diffusion model on {device}.')

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    model.to(device)  # Move the model to the chosen device
    beta = torch.tensor(beta).to(device)  # Move beta to the same device as the model

    for epoch in range(num_epochs):  # loop through every epoch
        start_time = time.time()  # Start the timer for this epoch
        # Training
        model.train()  # The model should be in training mode to use batch normalization and dropout
        train_loss = 0

        # loop through every batch
        for step, batch_x in enumerate(train_loader):
            # move the batch to the same device as the model
            batch_x = batch_x.to(device)

            # Sample random time steps for each sample in the batch
            t = torch.randint(0, time_steps, size=(batch_x.shape[0],), device=device).float()
            alpha_t = (1 - beta) ** (t + 1)  # this is alpha_t bar in the paper
            t_norm = t / time_steps  # Normalize t to be in [0, 1] (input for the model)

            # Add noise to the inputs according to the diffusion process
            noise = torch.randn_like(batch_x)

            # Gradient step
            optimizer.zero_grad()
            loss = loss_fn(batch_x, model, alpha_t, t_norm, noise)
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
            for batch_x in val_loader:
                # move the batch to the same device as the model
                batch_x = batch_x.to(device)

                # Sample random time steps for each sample in the batch
                t = torch.randint(0, time_steps, size=(batch_x.shape[0],), device=device).float()
                alpha_t = (1 - beta) ** (t + 1)  # this is alpha_t bar in the paper
                t_norm = t / time_steps  # Normalize t to be in [0, 1] (input for the model)

                # Add noise to the inputs according to the diffusion process
                noise = torch.randn_like(batch_x)

                # calculate the loss
                loss = loss_fn(batch_x, model, alpha_t, t_norm, noise)
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

    print("Training complete.")

    return train_losses, val_losses


# Sample from the trained diffusion model
def sample_diffusion_model(model: nn.Module, time_steps: int, beta: torch.Tensor, n_samples: int, device: str = 'cpu', monitor_denoising: int = 0) -> np.ndarray:
    model.to(device)
    beta = torch.tensor(beta).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(n_samples, device=device)  # Start from pure noise
        denoising_trajectory = x[:monitor_denoising].cpu().numpy() if monitor_denoising > 0 else None

        for t in reversed(range(1, time_steps+1)):
            if t != 1:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            
            alpha_t = (1 - beta) ** (t + 1) # this is alpha_t bar in the paper
            t_norm = float(t) / time_steps 

            model_input = torch.stack((x, torch.full((n_samples,), t_norm, device=device)), dim=1)
            x = 1/torch.sqrt(1 - beta) * (x - beta / torch.sqrt(1 - alpha_t) * model(model_input).squeeze()) + torch.sqrt(beta) * z

            if denoising_trajectory is not None:
                denoising_trajectory = np.vstack((denoising_trajectory, x[:monitor_denoising].cpu().numpy()))

    return x.cpu().numpy(), denoising_trajectory.T

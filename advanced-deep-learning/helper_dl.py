import time
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import typing as tp

# Choose the best device
def get_device() -> str:
    device = (
        "cuda"
        if torch.cuda.is_available()  # CUDA GPU
        else "mps"
        if torch.backends.mps.is_available()  # Apple Metal Performance Shaders
        else "xpu"
        if torch.xpu.is_available()  # Intel XPU
        else "cpu"  # Fallback to CPU if neither CUDA nor MPS are found
    )
    return device

# Train the neural network
def train_nn(
            train_loader: DataLoader, 
            val_loader: DataLoader,
            model: nn.Module,
            loss_fn: function,
            optimizer: torch.optim.Optimizer,
            num_epochs: int,
            patience: int = None,
            device: str = 'cpu'
            )-> tp.tuple[list, list]:

    print(f'Training {type(model).__name__} on {device}.')

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0

    model.to(device)  # Move the model to the chosen device

    for epoch in range(num_epochs):  # loop through every epoch
        start_time = time.time()  # Start the timer for this epoch
        # Training
        model.train()  # The model should be in training mode to use batch normalization and dropout
        train_loss = 0

        # loop through every batch
        for step, (batch_x, batch_y) in enumerate(train_loader):
            # move the batch to the same device as the model
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # set the gradients to zero
            optimizer.zero_grad()

            # calculate the loss 
            loss = loss_fn(batch_x, batch_y, model)

            # calculated the gradients for the given loss
            loss.backward()

            # updates the weights and biases for the given gradients
            optimizer.step()

            # calculate loss per batch
            train_loss += loss.item()

            # Print progress every 10th step, updating the same line
            if (step + 1) % 10 == 0:
                sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")
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
            for batch_x, batch_y in val_loader:
                # move the batch to the same device as the model
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # calculate the loss
                loss = loss_fn(batch_x, batch_y, model)

                # calculate loss per batch
                val_loss += loss.item()

        # calulate loss per epoch
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print epoch summary
        epoch_time = time.time() - start_time  # Calculate epoch time
        sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} seconds\n")
        sys.stdout.flush()

        if(patience is not None):
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    print("Training complete.")

    return train_losses, val_losses


# Plot training/validation loss
def plot_losses(train_losses: list, val_losses: list, PATH: str = None) -> None:
    fig, ax = plt.subplots(layout="constrained")

    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Validation Loss")

    ax.legend()
    ax.set(
        xlabel="Epochs",
        ylabel="Loss",
    )

    if PATH is not None:
        fig.savefig(PATH, dpi=300)

    return

# Test the model
def test_nn(
            test_loader: DataLoader,
            model: nn.Module,
            loss_fn: function, 
            device: str = 'cpu'
            ) -> tp.Tuple[np.array, float]:
    model.to(device)

    model.eval()
    test_loss = 0
    y_test_pred = np.array([])

    # make sure the gradients are not changed in this step
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # move the batch to the same device as the model
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            # make a prediction with the current model
            predictions = model(batch_x)
            y_test_pred = np.append(y_test_pred, predictions.cpu().numpy())

            # calculate the loss based on the prediction
            loss = loss_fn(batch_x, batch_y, model)

            # calulate loss per batch
            test_loss += loss.item()

    test_loss /= len(test_loader)  # calculate total loss

    return y_test_pred, test_loss
import time
import sys
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import typing as tp

# ---- DATA PREPROCESSING FUNCTIONS ----

# Standard label normalization (subtract mean and divide by std)
def normalize_labels(labels: np.ndarray, n_labels: int) -> tp.Tuple[np.ndarray, list[list, list]]:
    labels_norm = np.zeros_like(labels)
    means, stds = [], []
    for i in range(n_labels):
        mean = np.mean(labels[:, i])
        std = np.std(labels[:, i])
        means.append(mean)
        stds.append(std)
        labels_norm[:, i] = (labels[:, i] - mean) / std
    return labels_norm, [means, stds]

# Denormalize the labels
def denormalize_labels(labels_norm: np.ndarray, n_labels: int, means: list, stds: list) -> np.ndarray:
    labels = np.zeros_like(labels_norm)
    for i in range(n_labels):
        labels[:, i] = (labels_norm[:, i] * stds[i]) + means[i]
    return labels

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

# ---- TRAINING FUNCTIONS ----

# Train the neural network
def train_nn(
            train_loader: DataLoader, 
            val_loader: DataLoader,
            model: nn.Module,
            loss_fn: tp.Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor],
            optimizer: torch.optim.Optimizer,
            num_epochs: int,
            patience: int = None,
            device: str = 'cpu'
        )-> tp.Tuple[list, list]:

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
        sys.stdout.write(f"\rEpoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_time:.2f} seconds")
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


# Plot training/validation loss
def plot_losses(train_losses: list, val_losses: list, PATH: str = None) -> None:
    fig, ax = plt.subplots(layout="constrained")

    ax.plot(train_losses[1:], label="Train Loss")
    ax.plot(val_losses[1:], label="Validation Loss")

    ax.legend()
    ax.set(
        xlabel="Epochs",
        ylabel="Loss",
    )

    if PATH is not None:
        fig.savefig(f"{PATH}/loss-curves.png", dpi=300)

    return

# Save and load model functions
def save_model(model: nn.Module, hyperparams: dict, PATH: str) -> None:
    info = {
        "hyperparams": hyperparams,
        "state_dict": model.state_dict(),}
    torch.save(info, PATH)

    return

def load_model(model_class: tp.Type[nn.Module], PATH: str) -> nn.Module:
    info = torch.load(PATH)

    model = model_class(**info["hyperparams"]["model_hyperparams"])
    model.load_state_dict(info["state_dict"])

    return model, info["hyperparams"]

# ---- EVALUATION FUNCTIONS ----

# Test the model
def test_nn(
            test_loader: DataLoader,
            model: nn.Module,
            loss_fn: tp.Callable[[torch.Tensor, torch.Tensor, nn.Module], torch.Tensor], 
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

# Scatter plot of predicted values vs true values
def plot_predicted_vs_true(
                        labels: np.ndarray, 
                        labels_pred: np.ndarray, 
                        n_labels: int, 
                        label_names: list = None, 
                        units: list = None, 
                        errors: np.ndarray=None, 
                        PATH: str = None
                    ) -> None:
    
    plt.figure(figsize=(4*n_labels, 4))
    for i in range(n_labels):
        plt.subplot(1, n_labels, i+1)
        if errors is not None:
            plt.errorbar(labels[:, i], labels_pred[:, i], yerr=errors[:, i],
                    fmt='o', markersize=3, capsize=2, alpha=0.3)
        else:
            plt.scatter(labels[:, i], labels_pred[:, i], alpha=0.5, s=3)
        plt.plot([labels[:, i].min(), labels[:, i].max()], 
                [labels[:, i].min(), labels[:, i].max()], 'r', lw=1) # Add a y=x line for reference
        plt.xlabel("True Values")
        plt.ylabel("Predictions")
        if units is not None:
            plt.xlabel(f"True Values ({units[i]})")
            plt.ylabel(f"Predictions ({units[i]})")
        if label_names is not None:
            plt.title(label_names[i])
    plt.tight_layout()

    if PATH is not None:
        plt.savefig(f"{PATH}/pred-vs-true.png", dpi=300)

    plt.show()

    return

# Calculate bias and std of the predictions
def get_bias_std(labels: np.ndarray, labels_pred: np.ndarray)-> tp.Tuple[np.ndarray, np.ndarray]:
    residuals = labels_pred - labels
    bias = np.mean(residuals, axis=0)
    std = np.std(residuals, axis=0)

    return bias, std

# Plot residual distribution
def plot_residuals(
            labels: np.ndarray, 
            labels_pred: np.ndarray, 
            n_labels: int, 
            label_names: list = None, 
            units: list = None,
            PATH: str = None
        ) -> None:
    
    residuals = labels_pred - labels

    # Residual distribution
    plt.figure(figsize=(3*n_labels, 3))
    for i in range(n_labels):
        plt.subplot(1, n_labels, i+1)
        plt.hist(residuals[:, i], bins=30, alpha=0.7)
        plt.vlines(0, 0, plt.ylim()[1], color='r', lw=1) # Add a vertical line at x=0 for reference
        plt.xlabel("Residuals")
        if units is not None:
            plt.xlabel(f"Residuals ({units[i]})")
        if label_names is not None:
            plt.title(label_names[i])
    plt.tight_layout()

    if PATH is not None:
        plt.savefig(f"{PATH}/residuals-hist.png", dpi=300)

    plt.show()

    # Residual dependency on true values
    plt.figure(figsize=(3*n_labels, 3))
    for i in range(n_labels):
        plt.subplot(1, n_labels, i+1)
        plt.scatter(labels[:, i], residuals[:, i], label=labels[i], alpha=0.5, s=3)
        plt.axhline(0, color='r', lw=1) # Add a horizontal line at y=0 for reference
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        if units[i]:
            plt.xlabel(f"True Values ({units[i]})")
            plt.ylabel(f"Residuals ({units[i]})")
        if label_names is not None:
            plt.title(label_names[i])
    plt.tight_layout()

    if PATH is not None:
        plt.savefig(f"{PATH}/residuals-vs-true.png", dpi=300)

    plt.show()

    return 
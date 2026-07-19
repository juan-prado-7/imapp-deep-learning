import torch
import torch.nn as nn
import awkward as ak
import typing as tp
from torch.utils.data import DataLoader
import sys
import time
import numpy as np



def collate_fn_transformer(batch: list[dict]) -> tuple[list[list | torch.Tensor], torch.Tensor]:
    """
    Custom function that defines how batches are formed.

    To process the batch items that each have a different number of hits, it is efficient
    to first concatenate all the data into a single tensor and save the lengths of each
    individual event to be able to split the data again later.

    # F: input_dim, number of features (time, x, y)
    # N: number of hits (different for each event)
    # B: batch size

    The resulting 2D tensor has the shape (B x N, F) where B is the batch size, N is the total number of hits of all events
    in the batch, and F is the number of features (time, x, y).


    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    lengths: list[int] = []
    
    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the features are (time, x, y)
        tensor_data = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensor_data = tensor_data.to(dtype=torch.float32)

        lengths.append(tensor_data.shape[0])

        data_list.append(tensor_data)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    ## return a list [data_list, lengths]
    return [
        torch.cat(data_list),  # (B, N, F)  -> (BxN, F) where B is the batch size, N is the number of hits, and F is the number of features (time, x, y)
        lengths,
    ], torch.cat(labels, dim=0)

# Create the dataloaders using the previous collate function
def create_dataloaders(
            train_dataset: ak.Array, 
            val_dataset: ak.Array, 
            test_dataset: ak.Array, 
            batch_size: int
        ) -> tp.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_transformer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer)

    return train_loader, val_loader, test_loader

# Train the neural network (adjustement in the way of sending the batches to device)
def train_transformer_model(
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
            batch_x[0], batch_y = batch_x[0].to(device), batch_y.to(device)

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
            for batch_x, batch_y in val_loader:
                # move the batch to the same device as the model
                batch_x[0], batch_y = batch_x[0].to(device), batch_y.to(device)

                # calculate the loss
                loss = loss_fn(batch_x, batch_y, model)

                # calculate loss per batch
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


# Test the model (adjustement in the way of sending the batches to device)
def test_transformer_model(
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
            batch_x[0], batch_y = batch_x[0].to(device), batch_y.to(device)
            
            # make a prediction with the current model
            predictions = model(batch_x)
            y_test_pred = np.append(y_test_pred, predictions.cpu().numpy())

            # calculate the loss based on the prediction
            loss = loss_fn(batch_x, batch_y, model)

            # calulate loss per batch
            test_loss += loss.item()

    test_loss /= len(test_loader)  # calculate total loss

    return y_test_pred, test_loss

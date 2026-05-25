import os
import typing as tp
import numpy as np
from matplotlib import pyplot as plt
import awkward as ak
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch

import helper as dl

# Load the dataset
def load_icecube_data(PATH: str) -> tp.Tuple[ak.Array, ak.Array, ak.Array]:
    train_dataset = ak.from_parquet(os.path.join(PATH, "train.pq"))
    val_dataset = ak.from_parquet(os.path.join(PATH, "val.pq"))
    test_dataset = ak.from_parquet(os.path.join(PATH, "test.pq"))

    return train_dataset, val_dataset, test_dataset


# Normalize data and labels
# working with Awkward arrays is a bit tricky because the ['data'] field can't be assigned in-place,
# so we need to extract the time, x, and y coordinates, normalize them separately,
# and then concatenate them back together.
def normalize_data(dataset: ak.Array) -> tp.Tuple[ak.Array, list[list, list]]:
    dataset_norm = ak.Array({
        "data": dataset["data"],
        "xpos": dataset["xpos"],
        "ypos": dataset["ypos"]
    })

    times = dataset["data"][:, 0:1, :]  # important to index the time dimension with 0:1 to keep this dimension (n_events, 1, n_hits)
    norm_times = times - np.min(times, axis=-1, keepdims=True)
    norm_times = norm_times/np.max(norm_times)

    x = dataset["data"][:, 1:2, :]
    norm_x = (x - np.mean(x)) / np.std(x)
    y = dataset["data"][:, 2:3, :]
    norm_y = (y - np.mean(y)) / np.std(y)

    # Concatenate the normalized data back together
    dataset_norm["data"] = ak.concatenate([norm_times, norm_x, norm_y], axis=1)

    # Normalize labels (this can be done in-place), e.g. by
    labels = np.column_stack([dataset["xpos"], dataset["ypos"]]).to_numpy()
    labels_norm, [means, stds] = dl.normalize_labels(labels, n_labels=2)
    dataset_norm["xpos"] = labels_norm[:, 0]
    dataset_norm["ypos"] = labels_norm[:, 1]

    return dataset_norm, [means, stds]



# Create the DataLoader for training, validation, and test datasets
# Important: We use the custom collate function to preprocess the data for GNN (see the description of the collate function for details)
def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

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
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the feautures are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensordata = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensordata = tensordata.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensordata)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    labels = torch.cat(labels, dim=0) # convert the list of tensors to a single tensor
    packed_data = Batch.from_data_list(data_list) # convert the list of Data objects to a single Batch object
    return packed_data, labels

# Create the dataloaders using the previous collate function
def create_dataloaders(
            train_dataset: ak.Array, 
            val_dataset: ak.Array, 
            test_dataset: ak.Array, 
            batch_size: int
        ) -> tp.Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_gnn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_gnn)

    return train_loader, val_loader, test_loader

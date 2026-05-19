import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load spectra and labels
def load_galah_data(PATH):
    # Load spectra and labels
    spectra = np.load(f"{PATH}/spectra.npy")
    labels = np.load(f"{PATH}/labels.npy")

    # labels: mass, age, l_bol, dist, t_eff, log_g, fe_h, SNR
    labelNames = ["mass", "age", "l_bol", "dist", "t_eff", "log_g", "fe_h", "SNR"]
    units = ["M_sun", "Gyr", "L_sun", "pc", "K", "", "", ""]
    labels = np.load(f"../datasets/labels.npy")

    # We only use the three labels: t_eff, log_g, fe_h, SNR
    labelNames = labelNames[-4:-1]
    labels = labels[:, -4:-1]
    units = units[-4:-1]

    # Get the number of labels and samples
    n_labels = labels.shape[1]
    n_samples = spectra.shape[0]
    spectra_length = spectra.shape[1]

    return spectra, labels, labelNames, units, spectra_length, n_labels, n_samples

# Split data into training, validation and test
def split_data(spectra, labels, val_fraction=0.15, test_fraction=0.15, random_state=42):
    r1 = val_fraction + test_fraction
    r2 = val_fraction / r1

    X_train, X_temp, labels_train, labels_temp = train_test_split(spectra, labels, test_size=r1, random_state=random_state)
    X_val, X_test, labels_val, labels_test = train_test_split(X_temp, labels_temp, test_size=r2, random_state=random_state)

    return X_train, X_val, X_test, labels_train, labels_val, labels_test

# Scale labels using only the training set statistics
def scale_labels(labels_train, labels_val, labels_test):
    scaler = StandardScaler()
    y_train = scaler.fit_transform(labels_train)
    y_val = scaler.transform(labels_val)
    y_test = scaler.transform(labels_test)

    return y_train, y_val, y_test, scaler

# Define a custom Dataset class
class CustomDataset(TensorDataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataloaders
def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, spectra_length, n_labels, batch_size=32):
    # Create datasets with torch tensors
    train_dataset = CustomDataset(torch.as_tensor(X_train, dtype=torch.float32).view(-1, 1, spectra_length),
                                torch.as_tensor(y_train, dtype=torch.float32).view(-1, n_labels))
    val_dataset = CustomDataset(torch.as_tensor(X_val, dtype=torch.float32).view(-1, 1, spectra_length),
                                torch.as_tensor(y_val, dtype=torch.float32).view(-1, n_labels))
    test_dataset = CustomDataset(torch.as_tensor(X_test, dtype=torch.float32).view(-1, 1, spectra_length),
                             torch.as_tensor(y_test, dtype=torch.float32).view(-1, n_labels))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Scatter plot of predicted values vs true values
def plot_predicted_vs_true(labels, labels_pred, n_labels, label_names, units=None, errors=None, PATH=None):
    plt.figure(figsize=(12, 4))
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
        plt.title(label_names[i])
    plt.tight_layout()

    if PATH is not None:
        plt.savefig(PATH, dpi=300)

    plt.show()

    return

from scipy.stats import norm

# Plot pull distribution
def plot_pull_distribution(labels, labels_pred, n_labels, label_names, errors, PATH=None):
    residuals = (labels - labels_pred) / errors

    plt.figure(figsize=(12, 4))
    for i in range(n_labels):
        plt.subplot(1, n_labels, i+1)
        mu_fit = np.mean(residuals[:, i])
        std_fit = np.std(residuals[:, i])
        plt.hist(residuals[:, i], bins=30, alpha=0.7, density=True)
        t = np.linspace(-4*std_fit, 4*std_fit, 100)
        plt.plot(t, norm.pdf(t, mu_fit, std_fit), color='r', linewidth=1,
                label=rf'$\mu={mu_fit:.2f}${'\n'}$\sigma={std_fit:.2f}$')
        plt.legend()
        plt.xlabel("Residuals/Uncertainties")
        plt.title(label_names[i])
    plt.tight_layout()

    if PATH is not None:
        plt.savefig(PATH, dpi=300)

    plt.show()
    
    return

def get_quantiles(test_loader, model, device='cpu'):
    quantiles = np.array([])

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            quantiles = np.append(quantiles, model.get_quantiles(batch_x, batch_y))

    return quantiles

def plot_coverage_test(quantiles, PATH=None):
    plt.figure()
    plt.hist(quantiles, bins=30, density=True)
    plt.xlabel('Quantile')
    plt.ylabel('Probability')
    plt.plot([0, 1], [1, 1], color='red', ls='--')

    if PATH is not None:
        plt.savefig(PATH, dpi=300)

    plt.show()
    return
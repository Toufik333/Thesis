# client.py

import argparse
import warnings
from collections import OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Check for CUDA availability and set the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. Model Definition ---
# Define a simple Multi-Layer Perceptron (MLP) for binary classification.
class MLP(nn.Module):
    """A simple Multi-Layer Perceptron model."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Data Simulation (Non-IID) ---
def load_non_iid_data(client_id: int, num_clients: int) -> Tuple[DataLoader, DataLoader]:
    """
    Simulates loading a non-IID dataset for breast cancer classification.

    This function generates a synthetic dataset and distributes it among clients
    in a non-IID fashion based on class labels (benign vs. malignant).
    Each client receives a skewed distribution of the two classes.

    Args:
        client_id (int): The ID of the current client (e.g., 0, 1, 2, ...).
        num_clients (int): The total number of clients in the federation.

    Returns:
        Tuple[DataLoader, DataLoader]: A tuple containing the training and
                                       validation data loaders for the client.
    """
    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.01,
        random_state=42,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Split data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define non-IID label proportions for 4 clients
    # Client 0: 80% class 0, 20% class 1
    # Client 1: 20% class 0, 80% class 1
    # Client 2: 60% class 0, 40% class 1
    # Client 3: 40% class 0, 60% class 1
    proportions = [(0.8, 0.2), (0.2, 0.8), (0.6, 0.4), (0.4, 0.6)]
    client_prop = proportions[client_id % len(proportions)]

    # Separate train data by class
    X_train_0, y_train_0 = X_train[y_train == 0], y_train[y_train == 0]
    X_train_1, y_train_1 = X_train[y_train == 1], y_train[y_train == 1]

    # Sample data for the current client according to its proportion
    num_samples_0 = int(len(X_train_0) / num_clients * client_prop[0] * num_clients)
    num_samples_1 = int(len(X_train_1) / num_clients * client_prop[1] * num_clients)

    start_idx_0 = int(len(X_train_0) / num_clients * client_id * client_prop[0])
    end_idx_0 = start_idx_0 + num_samples_0
    
    start_idx_1 = int(len(X_train_1) / num_clients * client_id * client_prop[1])
    end_idx_1 = start_idx_1 + num_samples_1

    client_X_train = np.concatenate((X_train_0[start_idx_0:end_idx_0], X_train_1[start_idx_1:end_idx_1]))
    client_y_train = np.concatenate((y_train_0[start_idx_0:end_idx_0], y_train_1[start_idx_1:end_idx_1]))
    
    # Shuffle the client's training data
    shuffle_idx = np.random.permutation(len(client_X_train))
    client_X_train, client_y_train = client_X_train[shuffle_idx], client_y_train[shuffle_idx]

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(client_X_train), torch.from_numpy(client_y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    # The validation set is kept the same across all clients for fair comparison
    test_loader = DataLoader(test_dataset, batch_size=32)

    return train_loader, test_loader

# --- 3. Training and Evaluation Logic ---
def train(net, trainloader, epochs, mu, device):
    """
    Train the neural network on the client's local data.

    Args:
        net: The neural network model.
        trainloader: DataLoader for the training set.
        epochs (int): The number of training epochs.
        mu (float): The proximal term coefficient for FedProx.
        device: The device to train on (CPU or CUDA).
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Store the initial global model weights for the proximal term calculation
    global_weights = [param.data.clone() for param in net.parameters()]

    net.train()
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            
            # Standard Cross-Entropy loss
            loss = criterion(outputs, labels)
            
            # --- FedProx proximal term ---
            # This term is added to the loss to pull local weights back
            # towards the global model, mitigating drift from non-IID data.
            if mu > 0:
                proximal_term = 0.0
                for local_weights, global_w in zip(net.parameters(), global_weights):
                    proximal_term += (local_weights - global_w).norm(2)
                loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()

def test(net, testloader, device) -> Tuple[float, float]:
    """
    Evaluate the neural network on the validation set.

    Args:
        net: The neural network model.
        testloader: DataLoader for the test set.
        device: The device to evaluate on.

    Returns:
        A tuple containing the average loss and accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = loss / len(testloader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy

# --- 4. Flower Client Implementation ---
class BreastCancerClient(fl.client.NumPyClient):
    """Flower client for breast cancer detection."""

    def __init__(self, client_id: int):
        self.client_id = client_id
        
        # Instantiate model and move to the appropriate device
        self.net = MLP(input_size=30, hidden_size=64, output_size=2).to(DEVICE)
        
        # Load the client's local, non-IID data partition
        self.trainloader, self.testloader = load_non_iid_data(client_id, num_clients=4)

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Update the local model with parameters from the server."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """Train the model on the local dataset."""
        self.set_parameters(parameters)
        
        # Extract hyperparameters from the server's config
        epochs = int(config.get("local_epochs", 1))
        mu = float(config.get("mu", 0.01)) # FedProx mu, default to 0.01 if not provided

        print(f"[Client {self.client_id}] Training for {epochs} epochs with mu={mu}")
        train(self.net, self.trainloader, epochs=epochs, mu=mu, device=DEVICE)
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """Evaluate the model on the local validation set."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, device=DEVICE)
        
        print(f"[Client {self.client_id}] Evaluate accuracy: {accuracy:.4f}")
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

# --- 5. Client Execution ---
if __name__ == "__main__":
    # Parse command line argument for client_id
    parser = argparse.ArgumentParser(description="Flower Breast Cancer Client")
    parser.add_argument(
        "--client-id",
        type=int,
        required=True,
        choices=range(0, 4),
        help="Client ID, from 0 to 3.",
    )
    args = parser.parse_args()

    print(f"Starting client {args.client_id} on device: {DEVICE}")

    # Instantiate and start the Flower client
    client = BreastCancerClient(client_id=args.client_id).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
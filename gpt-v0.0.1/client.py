#!/usr/bin/env python3
"""
client.py — Flower Federated Learning client for breast cancer detection on a non-IID local dataset (PyTorch).

- Simulates non-IID data via client-specific label proportions (4 clients).
- Implements a simple MLP (or small CNN) for binary classification.
- Uses Flower's NumPyClient interface and shows where to add a FedProx proximal term.
- Trains/evaluates locally and communicates weights with the server.

References (for developers):
- Flower NumPyClient API and usage expectations. [web:2]
- Flower docs on FedProx and proximal term in client-side loss. [web:6][web:12]
- General Flower PyTorch client tutorials/patterns. [web:1][web:8][web:3]
- Label distribution skew (non-IID) concept and simulation inspiration. [web:7][web:18][web:16]
"""

import argparse
import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import flwr as fl

# ----------------------------
# Reproducibility
# ----------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data Simulation (Non-IID)
# ----------------------------

def make_dummy_breast_cancer_csv(path: str, n_samples: int = 800) -> None:
    """
    Create a tiny dummy CSV to emulate breast cancer data if not present.
    Features are random floats; label is Bernoulli with p=0.5.
    Real workflows should replace this with actual loading code.

    CSV columns:
      f0, f1, ..., f19, label

    Note: The non-IID split happens in load_non_iid_data by sampling with different label priors per client. [web:7][web:18][web:16]
    """
    if os.path.exists(path):
        return
    rng = np.random.default_rng(123)
    X = rng.normal(0, 1, size=(n_samples, 20))
    y = rng.integers(0, 2, size=(n_samples,))
    header = ",".join([f"f{i}" for i in range(20)] + ["label"])
    rows = [",".join([f"{x:.6f}" for x in X[i]] + [str(int(y[i]))]) for i in range(n_samples)]
    with open(path, "w") as f:
        f.write(header + "\n")
        f.write("\n".join(rows))


def load_non_iid_data(client_id: int, num_clients: int = 4, batch_size: int = 32) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Simulate non-IID label distribution across clients by biased sampling of labels.

    Design:
    - 4 clients with different label priors (label 0 == benign, label 1 == malignant for illustration).
      Client 0: P(y=0)=0.85, Client 1: P(y=1)=0.85, Client 2: P(y=0)=0.65, Client 3: P(y=1)=0.65.
      This creates "label distribution skew", a common non-IID pattern in FL. [web:7][web:16][web:18]
    - Sample from a dummy CSV or, if desired, could generate tensors directly.

    Returns:
      train_loader, val_loader
    """
    assert num_clients == 4, "This example expects exactly 4 clients for the defined label priors. [web:18]"
    csv_path = "C:\Users\touf1000\Documents\GIThub\Thesis\Non-IID-dataset\breast_cancer_dataset.csv"
    make_dummy_breast_cancer_csv(csv_path)

    # Load CSV
    feats, labels = [], []
    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            x = np.array([float(v) for v in parts[:-1]], dtype=np.float32)
            y = int(parts[-1])
            feats.append(x)
            labels.append(y)
    X = np.stack(feats, axis=0)
    y = np.array(labels, dtype=np.int64)

    # Separate indices by label
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng = np.random.default_rng(100 + client_id)

    # Define label prior per client (non-IID)
    priors = {
        0: 0.85,  # mostly benign (0)
        1: 0.15,  # mostly malignant (1) -> equivalent to P(y=0)=0.15, so P(y=1)=0.85
        2: 0.65,  # moderately benign-heavy
        3: 0.35,  # moderately malignant-heavy
    }
    p0 = priors.get(client_id, 0.5)
    p1 = 1.0 - p0

    # Create a client-specific dataset by biased sampling without replacement where possible
    n_total = 600  # client local total samples
    n0 = int(n_total * p0)
    n1 = n_total - n0

    chosen0 = rng.choice(idx0, size=min(n0, len(idx0)), replace=False)
    chosen1 = rng.choice(idx1, size=min(n1, len(idx1)), replace=False)
    chosen = np.concatenate([chosen0, chosen1])
    rng.shuffle(chosen)

    X_client = X[chosen]
    y_client = y[chosen]

    # Split into train/val
    n_train = int(0.8 * len(X_client))
    X_train, y_train = X_client[:n_train], y_client[:n_train]
    X_val, y_val = X_client[n_train:], y_client[n_train:]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_ds = data.TensorDataset(X_train_t, y_train_t)
    val_ds = data.TensorDataset(X_val_t, y_val_t)

    train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


# ----------------------------
# Models
# ----------------------------

class MLP(nn.Module):
    """
    Simple MLP for binary classification with 20-dimensional inputs.
    Suitable for tabular-like dummy data; change to CNN if using images. [web:1][web:3][web:8]
    """
    def __init__(self, in_features: int = 20, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ----------------------------
# Training/Evaluation Utilities
# ----------------------------

def train_one_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    global_params_snapshot: List[torch.Tensor] = None,
    proximal_mu: float = 0.0,
) -> Tuple[float, float]:
    """
    Train for one epoch.

    FedProx proximal term:
      Add (mu/2) * ||w - w_global||^2 to loss, where w_global is a frozen copy of global params.
      In Flower, this is implemented on the client training step. [web:6][web:12]

    If proximal_mu == 0, standard local training is used.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    # Prepare reference for FedProx if enabled
    if proximal_mu > 0.0 and global_params_snapshot is not None:
        # Ensure no gradient on the snapshot
        ref_params = [p.detach().clone() for p in global_params_snapshot]
        for rp in ref_params:
            rp.requires_grad = False
    else:
        ref_params = None

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)

        # Add proximal term (FedProx)
        # Reference: Flower strategy notes show adding a proximal term during client training. [web:6][web:12]
        if proximal_mu > 0.0 and ref_params is not None:
            prox = 0.0
            for (lp, gp) in zip(model.parameters(), ref_params):
                prox = prox + torch.norm(lp - gp, p=2) ** 2
            loss = loss + (proximal_mu / 2.0) * prox

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluation without proximal term; report loss and accuracy. [web:2]
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


# ----------------------------
# Flower Client
# ----------------------------

class BreastCancerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for non-IID breast cancer detection with PyTorch. [web:2][web:1][web:3][web:8]
    """
    def __init__(
        self,
        client_id: int,
        num_clients: int = 4,
        lr: float = 1e-3,
        local_epochs: int = 2,
        batch_size: int = 32,
        proximal_mu: float = 0.0,
        device: str = "cpu",
    ):
        super().__init__()
        self.client_id = client_id
        self.num_clients = num_clients
        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.proximal_mu = proximal_mu
        self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

        # Load non-IID local data
        self.train_loader, self.val_loader = load_non_iid_data(client_id=self.client_id, num_clients=self.num_clients, batch_size=self.batch_size)

        # Initialize model/criterion/optimizer
        self.model = MLP(in_features=20, hidden=64).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def get_parameters(self, config: Dict[str, str]):
        """
        Return current local model parameters as a list of NumPy arrays. [web:2]
        """
        params = [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from a list of NumPy arrays in the same order as state_dict. [web:2]
        """
        state_dict = self.model.state_dict()
        new_state_dict = {}
        for (k, v), w in zip(state_dict.items(), parameters):
            new_state_dict[k] = torch.tensor(w, dtype=v.dtype)
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]):
        """
        Local training for a few epochs on the client's non-IID data. Adds FedProx proximal term if proximal_mu > 0.

        Config can override:
          - epochs
          - lr
          - proximal_mu

        FedProx note:
          The proximal term (mu/2)||w - w_global||^2 is added during local updates to reduce client drift under non-IID. [web:6][web:12]
        """
        # 1) Load global params into local model
        self.set_parameters(parameters)

        # 2) Capture snapshot of global parameters for FedProx
        global_snapshot = [p.detach().clone() for p in self.model.parameters()]

        # 3) Read config overrides
        epochs = int(config.get("epochs", self.local_epochs))
        lr = float(config.get("lr", self.lr))
        proximal_mu = float(config.get("proximal_mu", self.proximal_mu))

        # Update optimizer if lr changed
        if lr != self.lr:
            self.lr = lr
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 4) Local training
        for _ in range(epochs):
            train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                global_params_snapshot=global_snapshot,
                proximal_mu=proximal_mu,
            )

        # 5) Return updated params, number of samples, and metrics
        new_params = self.get_parameters(config={})
        num_examples = len(self.train_loader.dataset)
        # Optionally report train metrics (compute once at end to save time)
        train_loss, train_acc = evaluate(self.model, self.train_loader, self.criterion, self.device)
        metrics = {"train_loss": float(train_loss), "train_acc": float(train_acc), "proximal_mu": float(proximal_mu)}
        return new_params, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]):
        """
        Evaluate the provided parameters on the local validation set. [web:2]
        """
        self.set_parameters(parameters)
        val_loss, val_acc = evaluate(self.model, self.val_loader, self.criterion, self.device)
        num_examples = len(self.val_loader.dataset)
        metrics = {"val_acc": float(val_acc)}
        return float(val_loss), num_examples, metrics


# ----------------------------
# Entrypoint
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flower FL Client for non-IID breast cancer detection (PyTorch).")
    parser.add_argument("--client_id", type=int, required=True, choices=[0, 1, 2, 3], help="Client ID (0-3). [web:18]")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Flower server address host:port. [web:2]")
    parser.add_argument("--epochs", type=int, default=2, help="Local epochs per round. [web:2]")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate. [web:1]")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size. [web:1]")
    parser.add_argument("--proximal_mu", type=float, default=0.0, help="FedProx proximal coefficient mu. Set >0 to enable. [web:6][web:12]")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu or cuda. [web:1]")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Build client
    def client_fn() -> BreastCancerClient:
        return BreastCancerClient(
            client_id=args.client_id,
            num_clients=4,
            lr=args.lr,
            local_epochs=args.epochs,
            batch_size=args.batch_size,
            proximal_mu=args.proximal_mu,
            device=args.device,
        )

    # Start Flower client
    # NumPyClient is wrapped automatically; server must run separately with compatible strategy. [web:2][web:1]
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client_fn(),
    )


if __name__ == "__main__":
    main()
